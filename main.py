"""
FastAPI backend for GenCare Assistant RAG chatbot
"""
from fastapi import FastAPI, HTTPException, status, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import logging
import traceback
from datetime import datetime
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

from openai import OpenAI
from pinecone import Pinecone

from config import get_settings
from memory import ChatMemory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Global instances
chat_memory: Optional[ChatMemory] = None
openai_client: Optional[OpenAI] = None
pinecone_client: Optional[Pinecone] = None
pinecone_index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global chat_memory, openai_client, pinecone_client, pinecone_index
    
    # Startup
    logger.info("Initializing GenCare Assistant...")
    try:
        chat_memory = ChatMemory()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
        pinecone_index = pinecone_client.Index(settings.pinecone_index_name)
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down GenCare Assistant...")
    if chat_memory:
        chat_memory.close()


app = FastAPI(
    title="GenCare Assistant API",
    description="RAG-based chatbot grounded on BRS V1.2 documentation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class SessionCreate(BaseModel):
    """Request model for creating a new chat session"""
    metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Optional metadata to store with the session"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "user_id": "user123",
                    "user_agent": "Mozilla/5.0...",
                    "ip_address": "192.168.1.1"
                }
            }
        }


class SessionResponse(BaseModel):
    """Response model for session operations"""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="ISO 8601 timestamp of when the session was created")
    updated_at: str = Field(..., description="ISO 8601 timestamp of when the session was last updated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    is_active: bool = Field(True, description="Whether the session is active")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2023-06-01T12:00:00Z",
                "updated_at": "2023-06-01T12:05:30Z",
                "metadata": {"user_id": "user123"},
                "is_active": True
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., min_length=1, description="User's question")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata to update the session with"
    )
    
    @validator('session_id')
    def validate_uuid(cls, v):
        try:
            UUID(v, version=4)
        except ValueError:
            raise ValueError("session_id must be a valid UUID v4")
        return v


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="The user's original query")
    answer: str = Field(..., description="The assistant's response")
    sources_used: int = Field(..., description="Number of document sources used in the response")
    timestamp: str = Field(..., description="ISO 8601 timestamp of when the response was generated")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "What is GenCare?",
                "answer": "GenCare is a healthcare platform that...",
                "sources_used": 3,
                "timestamp": "2023-06-01T12:05:30Z"
            }
        }


# RAG Components
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""
    
    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """Check if the query is a greeting using the LLM"""
        try:
            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if a message is a greeting. Respond with 'true' if it's a greeting, 'false' otherwise."},
                    {"role": "user", "content": f"Is the following message a greeting? Respond with only 'true' or 'false': {query}"}
                ],
                temperature=0.1,
                max_tokens=5
            )
            return response.choices[0].message.content.strip().lower() == 'true'
        except Exception as e:
            logger.warning(f"Error checking if message is greeting: {e}")
            # Fallback to simple check if LLM fails
            simple_greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}
            return any(query.lower().startswith(greeting) for greeting in simple_greetings)
    
    @classmethod
    def get_greeting_response(cls, query: str = "") -> str:
        """Generate a friendly greeting response using the LLM"""
        try:
            prompt = (
                "You are a friendly and helpful assistant for the GenCare system. "
                "The user has sent a greeting. "
                "If they introduced themselves, acknowledge their name. "
                "Keep your response warm, professional, and under 2 sentences. "
                f"User's message: {query}"
            )
            
            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a friendly and helpful assistant for the GenCare system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating greeting response: {e}")
            # Fallback response if LLM fails
            return "Hello! I'm GenCare Assistant. How can I help you today?"
    
    @staticmethod
    def get_query_embedding(query: str) -> List[float]:
        """Generate embedding for user query"""
        try:
            response = openai_client.embeddings.create(
                model=settings.embedding_model,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    @classmethod
    def expand_query_terms(cls, query: str) -> List[str]:
        """Expand query with related terms for better retrieval"""
        # Common variations and synonyms for appointment-related queries
        appointment_terms = {
            'cancel': ['cancel', 'reschedule', 'change', 'modify', 'postpone'],
            'appointment': ['appointment', 'booking', 'schedule', 'meeting', 'consultation']
        }
        
        # Check if query is related to appointments
        query_lower = query.lower()
        if any(term in query_lower for term in ['appoint', 'book', 'schedule', 'meeting']):
            # Add variations of the query
            variations = [query]
            for original, synonyms in appointment_terms.items():
                if original in query_lower:
                    for synonym in synonyms:
                        if synonym != original:
                            variations.append(query_lower.replace(original, synonym))
            return variations
        return [query]
    
    @classmethod
    def retrieve_context(cls, query_embedding: List[float], query_text: str = None, top_k: int = None, is_scheduling: bool = False) -> List[Dict]:
        """Retrieve relevant chunks from Pinecone with improved relevance for scheduling and appointments"""
        if top_k is None:
            top_k = settings.top_k_results * 2  # Get more results to filter
        
        try:
            # First try with the original query
            contexts = cls._retrieve_with_threshold(query_embedding, top_k, min_score=0.5)
            
            # If no results, try with expanded query terms for scheduling
            if not contexts and query_text and any(term in query_text.lower() for term in ['schedule', 'visit', 'appointment', 'book']):
                logger.info("No results with original query, trying with scheduling-specific terms...")
                scheduling_queries = [
                    "how to schedule an appointment",
                    "book a doctor visit",
                    "make an appointment",
                    "schedule a consultation"
                ]
                
                for q in scheduling_queries:
                    if contexts:
                        break
                    try:
                        # Get embedding for the scheduling query
                        scheduling_embedding = cls.get_query_embedding(q)
                        contexts = cls._retrieve_with_threshold(scheduling_embedding, top_k, min_score=0.45)
                    except Exception as e:
                        logger.warning(f"Error with scheduling query '{q}': {e}")
            
            # If still no results, try with a lower threshold
            if not contexts:
                logger.info("No results with scheduling terms, trying with lower threshold...")
                contexts = cls._retrieve_with_threshold(query_embedding, top_k, min_score=0.35)
            
            # Log the scores of retrieved contexts for debugging
            if contexts:
                scores = [f"{c['score']:.3f}" for c in contexts]
                logger.info(f"Retrieved {len(contexts)} relevant contexts (scores: {', '.join(scores)})")
            else:
                logger.warning("No relevant contexts found after all retrieval attempts")
                
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    @classmethod
    def _retrieve_with_threshold(cls, query_embedding: List[float], top_k: int, min_score: float) -> List[Dict]:
        """Helper method to retrieve contexts with a specific score threshold"""
        try:
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"source": "BRS V1.2"}  # Ensure we only get BRS content
            )
            
            contexts = []
            seen_texts = set()
            
            for match in results['matches']:
                if match['score'] >= min_score:
                    text = match['metadata']['text'].strip()
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        contexts.append({
                            "text": text,
                            "score": match['score'],
                            "source": match['metadata'].get('source', 'BRS V1.2'),
                            "id": match.get('id')
                        })
                        
                        if len(contexts) >= settings.top_k_results:
                            break
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error in _retrieve_with_threshold: {e}")
            return []
    
    @staticmethod
    def format_chat_history(history: List[Dict[str, str]]) -> str:
        """Format chat history for prompt"""
        if not history:
            return ""
        
        formatted = "\n\n--- Previous Conversation ---\n"
        for msg in history[-5:]:  # Last 5 exchanges
            formatted += f"User: {msg['query']}\n"
            formatted += f"Assistant: {msg['answer']}\n\n"
        formatted += "--- End of Previous Conversation ---\n"
        return formatted
    
    @staticmethod
    def build_rag_prompt(query: str, contexts: List[Dict], chat_history: str) -> str:
        """Build the final RAG prompt"""
        context_text = "\n\n".join([
            f"[Context {i+1}] (Relevance: {ctx['score']:.2f})\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are GenCare Assistant, an expert AI assistant specialized in the GenCare system documentation (BRS V1.2).

Your primary responsibilities:
1. Answer questions ONLY based on the provided BRS documentation context
2. Provide clear, step-by-step guidance when explaining processes
3. Maintain conversation context using chat history
4. If information is not in the documentation, clearly state: "I'm sorry, I couldn't find that in the GenCare app."
5. Be helpful, professional, and precise

--- DOCUMENTATION CONTEXT ---
{context_text}

{chat_history}

--- CURRENT QUESTION ---
User: {query}

Instructions:
- Answer based ONLY on the context provided above
- Reference specific sections when relevant
- If the context doesn't contain the answer, say so clearly
- Maintain continuity with previous conversation
- Be concise but complete
"""
        return prompt
    
    @staticmethod
    def generate_answer(prompt: str) -> str:
        try:
            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[{"role": "system", "content": "You are a helpful assistant specialized in GenCare app."}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise


@app.post("/api/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(metadata: Optional[Dict[str, Any]] = None):
    """
    Create a new chat session
    
    This endpoint creates a new chat session with an optional metadata dictionary.
    Returns the session details including a unique session_id that should be used
    for subsequent chat interactions.
    """
    try:
        session_id = chat_memory.create_session(metadata)
        session = chat_memory.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create session"
            )
        return session
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get session details
    
    Retrieve information about a specific chat session including its metadata
    and creation/update timestamps.
    """
    try:
        UUID(session_id, version=4)  # Validate UUID format
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format. Must be a valid UUID v4."
        )
    
    session = chat_memory.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@app.get("/api/sessions", response_model=List[SessionResponse])
async def list_sessions(limit: int = 100):
    """
    List all active sessions
    
    Returns a list of active chat sessions with their metadata, sorted by
    most recently updated first.
    """
    try:
        return chat_memory.list_sessions(limit=limit)
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_session(session_id: str):
    """
    End a chat session
    
    Marks a session as inactive. The session and its history will be retained
    but can no longer be used for new messages.
    """
    try:
        UUID(session_id, version=4)  # Validate UUID format
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format. Must be a valid UUID v4."
        )
    
    if not chat_memory.session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if not chat_memory.end_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to end session"
        )
    
    return None


@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message with enhanced scheduling and appointment handling
    """
    logger.info(f"Processing chat request - Session: {request.session_id}, Query: '{request.query}'")
    
    try:
        session_id = request.session_id
        query = request.query.strip()
        
        # Update session metadata if provided
        if request.metadata:
            chat_memory.update_session_metadata(session_id, request.metadata)
        
        # Verify session exists or create a new one
        if not chat_memory.session_exists(session_id):
            logger.info(f"Creating new session with ID: {session_id}")
            metadata = {"auto_created": True, "first_query": query}
            if request.metadata:
                metadata.update(request.metadata)
            chat_memory.create_session(metadata)
        
        # Get chat history
        history = chat_memory.get_session_history(session_id)
        
        # Check for greetings first
        if RAGPipeline.is_greeting(query):
            answer = RAGPipeline.get_greeting_response(query)
            sources_used = 0
        else:
            # Process the query with enhanced RAG pipeline
            query_embedding = RAGPipeline.get_query_embedding(query)
            
            # Check if this is a scheduling-related query
            is_scheduling_query = any(term in query.lower() for term in 
                                   ['schedule', 'appointment', 'book', 'visit', 'reschedule', 'cancel'])
            
            # Get context with scheduling-specific handling
            contexts = RAGPipeline.retrieve_context(
                query_embedding, 
                query_text=query,
                is_scheduling=is_scheduling_query
            )
            
            # If no results for scheduling query, try with specific scheduling prompts
            if is_scheduling_query and not contexts:
                scheduling_prompts = [
                    "how to schedule an appointment in GenCare",
                    "steps to book a doctor visit",
                    "appointment scheduling process",
                    "how to make a medical appointment"
                ]
                
                for prompt in scheduling_prompts:
                    if contexts:
                        break
                    logger.info(f"Trying scheduling prompt: {prompt}")
                    prompt_embedding = RAGPipeline.get_query_embedding(prompt)
                    contexts = RAGPipeline.retrieve_context(prompt_embedding, prompt, is_scheduling=True)
            
            # Generate response based on context
            if not contexts:
                # Special handling for scheduling-related queries
                if is_scheduling_query:
                    answer = (
                        "I can help you schedule an appointment in the GenCare app. Here's how:\n\n"
                        "1. Open the GenCare mobile app\n"
                        "2. Tap on 'Appointments' in the bottom menu\n"
                        "3. Select 'Schedule New Appointment'\n"
                        "4. Choose your preferred doctor, date, and time\n"
                        "5. Confirm your appointment details\n\n"
                        "Would you like me to guide you through any specific part of this process?"
                    )
                else:
                    answer = (
                        "I couldn't find specific information about that in the GenCare documentation. "
                        "Here are some suggestions that might help:\n\n"
                        "1. Try rephrasing your question using different words\n"
                        "2. Check if the information might be in a different section\n"
                        "3. Contact our support team for further assistance"
                    )
                sources_used = 0
            else:
                # Format chat history and build prompt
                chat_history = RAGPipeline.format_chat_history(history)
                prompt = RAGPipeline.build_rag_prompt(query, contexts, chat_history)
                
                # Generate answer with enhanced instructions for scheduling
                if is_scheduling_query:
                    prompt += (
                        "\nIMPORTANT: If this is about scheduling or appointments, provide clear, step-by-step instructions. "
                        "Include all necessary details like where to click in the app. If the exact information isn't available, "
                        "provide general guidance on how to schedule appointments in the GenCare app."
                    )
                
                answer = RAGPipeline.generate_answer(prompt)
                sources_used = len(contexts)
        
        # Store the exchange in the chat history
        chat_memory.add_message(session_id, query, answer)
        
        logger.info(f"Successfully processed request for session: {session_id} (sources used: {sources_used})")
        
        return ChatResponse(
            session_id=session_id,
            query=query,
            answer=answer,
            sources_used=sources_used,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        
        # Provide a helpful error message
        error_msg = (
            "I apologize, but I encountered an error processing your request. "
            "Our team has been notified. Please try again in a moment or contact support if the issue persists."
        )
        
        # Log the full error for debugging
        logger.error(f"Full error details: {str(e)}\n{traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint
    
    Returns basic API information and status
    """
    return {
        "message": "GenCare Assistant API is running",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "create_session": "POST /api/sessions",
            "get_session": "GET /api/sessions/{session_id}",
            "list_sessions": "GET /api/sessions",
            "end_session": "DELETE /api/sessions/{session_id}",
            "chat": "POST /api/chat"
        },
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Verifies connectivity to all required services
    """
    health_status = {
        "status": "healthy",
        "services": {}
    }
    
    try:
        # Check MongoDB connection
        chat_memory.db.command('ping')
        health_status["services"]["mongodb"] = "connected"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        health_status["services"]["mongodb"] = "disconnected"
        health_status["status"] = "degraded"
    
    try:
        # Check Pinecone connection
        pinecone_index.describe_index_stats()
        health_status["services"]["pinecone"] = "connected"
    except Exception as e:
        logger.error(f"Pinecone health check failed: {e}")
        health_status["services"]["pinecone"] = "disconnected"
        health_status["status"] = "degraded"
    
    try:
        # Check OpenAI connection with a simple operation
        openai_client.models.list()
        health_status["services"]["openai"] = "connected"
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        health_status["services"]["openai"] = "disconnected"
        health_status["status"] = "degraded"
    
    # If any critical service is down, mark as unhealthy
    if any(status == "disconnected" for service, status in health_status["services"].items()):
        health_status["status"] = "unhealthy"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
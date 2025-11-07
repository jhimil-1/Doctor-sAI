"""
FastAPI backend for GenCare Assistant RAG chatbot with LangChain integration
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
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    description="RAG-based chatbot with LangChain integration, grounded on BRS V1.2 documentation",
    version="2.0.0",
    lifespan=lifespan
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
    context_types: List[str] = Field(default_factory=list, description="Types of contexts used (section, table, etc.)")
    timestamp: str = Field(..., description="ISO 8601 timestamp of when the response was generated")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "What is GenCare?",
                "answer": "GenCare is a healthcare platform that...",
                "sources_used": 3,
                "context_types": ["section", "table"],
                "timestamp": "2023-06-01T12:05:30Z"
            }
        }

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(request: SessionCreate = Body(...)):
    """Create a new chat session"""
    try:
        session_id = str(uuid4())
        created_at = datetime.utcnow()
        
        session_data = {
            "session_id": session_id,
            "created_at": created_at,
            "updated_at": created_at,
            "metadata": request.metadata or {},
            "history": [],
            "is_active": True
        }
        
        chat_memory.create_session(session_data)
        
        logger.info(f"New session created with ID: {session_id}")
        
        return SessionResponse(
            session_id=session_id,
            created_at=created_at.isoformat() + "Z",
            updated_at=created_at.isoformat() + "Z",
            metadata=request.metadata or {},
            is_active=True
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create a new session."
        )


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details"""
    try:
        if not chat_memory.session_exists(session_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
            
        session = chat_memory.get_session(session_id)
        
        return SessionResponse(
            session_id=session["session_id"],
            created_at=session["created_at"].isoformat() + "Z",
            updated_at=session["updated_at"].isoformat() + "Z",
            metadata=session["metadata"],
            is_active=session["is_active"]
        )
    except Exception as e:
        logger.error(f"Failed to retrieve session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session information."
        )


# Enhanced RAG Pipeline with LangChain
class EnhancedRAGPipeline:
    """Improved Retrieval-Augmented Generation pipeline with LangChain"""
    
    @classmethod
    def is_greeting(cls, query: str) -> bool:
        """Check if the query is a greeting using the LLM"""
        try:
            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a classifier. Respond with ONLY 'true' or 'false'."
                    },
                    {
                        "role": "user", 
                        "content": f"Is this a greeting or introduction? '{query}'"
                    }
                ],
                temperature=0.1,
                max_tokens=5
            )
            return response.choices[0].message.content.strip().lower() == 'true'
        except Exception as e:
            logger.warning(f"Error checking if message is greeting: {e}")
            # Fallback to simple check
            simple_greetings = {"hi", "hello", "hey", "greetings", "good morning", 
                              "good afternoon", "good evening", "howdy"}
            return any(query.lower().strip().startswith(greeting) for greeting in simple_greetings)
    
    @classmethod
    def get_greeting_response(cls, query: str = "") -> str:
        """Generate a friendly greeting response using the LLM"""
        try:
            prompt = (
                f"User said: '{query}'\n\n"
                "Respond warmly and professionally as GenCare Assistant. "
                "If they introduced themselves, acknowledge their name. "
                "Keep it under 2 sentences and offer help."
            )
            
            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are GenCare Assistant, a helpful healthcare system assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating greeting response: {e}")
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
    def enhance_query_for_retrieval(cls, query: str) -> List[str]:
        """
        Generate query variations for better retrieval using LLM
        """
        try:
            prompt = f"""Given this user query about a healthcare system: "{query}"

Generate 2-3 alternative phrasings that would help retrieve relevant documentation.
Focus on:
- Technical terminology
- Common variations
- Related concepts

Return as a simple list, one per line, no numbering."""

            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate query variations for document retrieval."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=150
            )
            
            variations = [query]  # Always include original
            content = response.choices[0].message.content.strip()
            for line in content.split('\n'):
                line = line.strip().strip('-•*123456789. ')
                if line and line != query:
                    variations.append(line)
            
            logger.info(f"Generated {len(variations)} query variations")
            return variations[:3]  # Limit to 3 total
            
        except Exception as e:
            logger.warning(f"Failed to generate query variations: {e}")
            return [query]
    
    @classmethod
    def retrieve_context(
        cls, 
        query_embedding: List[float], 
        query_text: str = None, 
        top_k: int = None,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks from Pinecone with enhanced filtering
        """
        if top_k is None:
            top_k = settings.top_k_results
        
        try:
            # Build filter
            base_filter = {"source": "BRS V1.2"}
            if filter_dict:
                base_filter.update(filter_dict)
            
            # Primary retrieval
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more to filter by score
                include_metadata=True,
                filter=base_filter
            )
            
            contexts = []
            seen_texts = set()
            
            # Process results with adaptive threshold
            min_score = 0.7  # Start with high threshold
            
            for match in results['matches']:
                if match['score'] >= min_score:
                    text = match['metadata'].get('text', '').strip()
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        contexts.append({
                            "text": text,
                            "score": match['score'],
                            "source": match['metadata'].get('source', 'BRS V1.2'),
                            "section": match['metadata'].get('section', 'Unknown'),
                            "type": match['metadata'].get('type', 'section'),
                            "id": match.get('id')
                        })
                        
                        if len(contexts) >= top_k:
                            break
            
            # If no results, try with lower threshold
            if not contexts and query_text:
                logger.info("No high-confidence matches, trying with lower threshold...")
                min_score = 0.5
                
                for match in results['matches']:
                    if match['score'] >= min_score:
                        text = match['metadata'].get('text', '').strip()
                        if text and text not in seen_texts:
                            seen_texts.add(text)
                            contexts.append({
                                "text": text,
                                "score": match['score'],
                                "source": match['metadata'].get('source', 'BRS V1.2'),
                                "section": match['metadata'].get('section', 'Unknown'),
                                "type": match['metadata'].get('type', 'section'),
                                "id": match.get('id')
                            })
                            
                            if len(contexts) >= top_k:
                                break
            
            # Try with query variations if still no results
            if not contexts and query_text:
                logger.info("Trying query variations...")
                variations = cls.enhance_query_for_retrieval(query_text)
                
                for variation in variations[1:]:  # Skip first (original)
                    if contexts:
                        break
                    try:
                        var_embedding = cls.get_query_embedding(variation)
                        var_results = pinecone_index.query(
                            vector=var_embedding,
                            top_k=top_k,
                            include_metadata=True,
                            filter=base_filter
                        )
                        
                        for match in var_results['matches']:
                            if match['score'] >= 0.6:
                                text = match['metadata'].get('text', '').strip()
                                if text and text not in seen_texts:
                                    seen_texts.add(text)
                                    contexts.append({
                                        "text": text,
                                        "score": match['score'],
                                        "source": match['metadata'].get('source', 'BRS V1.2'),
                                        "section": match['metadata'].get('section', 'Unknown'),
                                        "type": match['metadata'].get('type', 'section'),
                                        "id": match.get('id'),
                                        "retrieved_via": "variation"
                                    })
                                    
                                    if len(contexts) >= top_k:
                                        break
                    except Exception as e:
                        logger.warning(f"Error with variation '{variation}': {e}")
            
            # Log retrieval statistics
            if contexts:
                scores = [f"{c['score']:.3f}" for c in contexts]
                types = list(set(c.get('type', 'unknown') for c in contexts))
                logger.info(f"Retrieved {len(contexts)} contexts (scores: {', '.join(scores)}, types: {types})")
            else:
                logger.warning("No relevant contexts found after all retrieval attempts")
                
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    @staticmethod
    def format_chat_history(history: List[Dict[str, str]], max_exchanges: int = 5) -> str:
        """Format chat history for prompt with context window management"""
        if not history:
            return ""
        
        formatted = "\n\n--- Previous Conversation ---\n"
        recent_history = history[-max_exchanges:]
        
        for msg in recent_history:
            formatted += f"User: {msg['query']}\n"
            formatted += f"Assistant: {msg['answer']}\n\n"
        
        formatted += "--- End of Previous Conversation ---\n"
        return formatted
    
    @staticmethod
    def build_rag_prompt(query: str, contexts: List[Dict], chat_history: str) -> str:
        """Build enhanced RAG prompt with structured context"""
        # Group contexts by type
        sections = [c for c in contexts if c.get('type') == 'section']
        tables = [c for c in contexts if c.get('type') == 'table']
        
        context_text = ""
        
        if sections:
            context_text += "=== DOCUMENTATION SECTIONS ===\n\n"
            for i, ctx in enumerate(sections, 1):
                section_name = ctx.get('section', 'Unknown Section')
                context_text += f"[Section {i}: {section_name}] (Relevance: {ctx['score']:.2f})\n"
                context_text += f"{ctx['text']}\n\n"
        
        if tables:
            context_text += "=== RELATED TABLES ===\n\n"
            for i, ctx in enumerate(tables, 1):
                context_text += f"[Table {i}] (Relevance: {ctx['score']:.2f})\n"
                context_text += f"{ctx['text']}\n\n"
        
        prompt = f"""You are GenCare Assistant, an expert AI assistant specialized in the GenCare healthcare system (BRS V1.2 documentation).

Answer the user's question based *only* on the provided documentation.
Use simple, plain language that a non-expert can easily understand.

**DOCUMENTATION CONTEXT:**
{context_text}

{chat_history}

**USER'S QUESTION:** {query}

**RESPONSE FORMAT:**
1.  **Summary:** Provide a one-sentence summary of the answer.
2.  **Key Points:** List 3-5 key points as bullet points.
3.  **Details:** If applicable, provide a more detailed explanation.
4.  **Citations:** Include citations like [Section 1] next to the information they support.

Your response should be clear, concise, and easy to read."""
        
        return prompt
    
    @staticmethod
    def generate_answer(prompt: str) -> str:
        """Generate answer using OpenAI with enhanced parameters"""
        try:
            response = openai_client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are GenCare Assistant, an expert on the GenCare healthcare system. Answer accurately based only on the provided documentation. Use the specified format (Summary, Key Points, Details, Citations) to create a clear, concise, and easy-to-read response."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return (
                "I apologize, but I encountered an error while generating a response. "
                "Please try again shortly."
            )

@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat message with enhanced RAG pipeline
    """
    logger.info(f"Processing chat request - Session: {request.session_id}, Query: '{request.query}'")
    
    try:
        session_id = request.session_id
        query = request.query.strip()
        
        # Update session metadata if provided
        if request.metadata:
            chat_memory.update_session_metadata(session_id, request.metadata)
        
        # Verify session exists or create one
        if not chat_memory.session_exists(session_id):
            logger.info(f"Creating new session with ID: {session_id}")
            metadata = {"auto_created": True, "first_query": query}
            if request.metadata:
                metadata.update(request.metadata)
            chat_memory.create_session(metadata)
        
        # Get chat history
        history = chat_memory.get_session_history(session_id)
        
        # Check for greetings
        if EnhancedRAGPipeline.is_greeting(query):
            answer = EnhancedRAGPipeline.get_greeting_response(query)
            sources_used = 0
            context_types = []
            rag_context = None
            message_metadata = {"is_greeting": True}
        else:
            # Process with enhanced RAG
            query_embedding = EnhancedRAGPipeline.get_query_embedding(query)
            
            # Retrieve contexts
            contexts = EnhancedRAGPipeline.retrieve_context(
                query_embedding, 
                query_text=query
            )
            
            if not contexts:
                answer = (
                    "I couldn't find specific information about that in the GenCare documentation. "
                    "Try rephrasing with different keywords, or break the question into smaller parts. "
                    "If you can share a bit more detail, I can search more precisely."
                )
                sources_used = 0
                context_types = []
                rag_context = {
                    "sources_used": 0,
                    "context_types": [],
                    "source_documents": [],
                    "retrieval_scores": [],
                    "average_score": 0,
                    "max_score": 0,
                    "min_score": 0,
                    "embedding_model": settings.embedding_model,
                    "retrieval_strategy": "semantic_search",
                    "query_variations_used": False,
                    "retrieval_method": "standard"
                }
                message_metadata = {"is_greeting": False}
            else:
                # Generate answer
                chat_history = EnhancedRAGPipeline.format_chat_history(history)
                prompt = EnhancedRAGPipeline.build_rag_prompt(query, contexts, chat_history)
                answer = EnhancedRAGPipeline.generate_answer(prompt)
                sources_used = len(contexts)
                context_types = list(set(c.get('type', 'unknown') for c in contexts))
                # Build RAG context for storage
                scores = [c.get('score', 0) for c in contexts]
                rag_context = {
                    "sources_used": sources_used,
                    "context_types": context_types,
                    "source_documents": [
                        {
                            "id": c.get("id"),
                            "section": c.get("section"),
                            "type": c.get("type"),
                            "score": c.get("score"),
                            "source": c.get("source")
                        } for c in contexts
                    ],
                    "retrieval_scores": scores,
                    "average_score": sum(scores) / len(scores) if scores else 0,
                    "max_score": max(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "embedding_model": settings.embedding_model,
                    "retrieval_strategy": "semantic_search",
                    "query_variations_used": any(c.get("retrieved_via") == "variation" for c in contexts),
                    "retrieval_method": "standard"
                }
                message_metadata = {"is_greeting": False}
        
        # Store in history
        chat_memory.add_message(session_id, query, answer, rag_context=rag_context, message_metadata=message_metadata)
        
        logger.info(f"✅ Request processed - Session: {session_id}, Sources: {sources_used}, Types: {context_types}")
        
        return ChatResponse(
            session_id=session_id,
            query=query,
            answer=answer,
            sources_used=sources_used,
            context_types=context_types,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        
        error_msg = (
            "I apologize, but I encountered an error processing your request. "
            "Please try again in a moment, or contact support if the issue persists."
        )
        
        logger.error(f"Full error: {str(e)}\n{traceback.format_exc()}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GenCare Assistant API with LangChain integration",
        "version": "2.0.0",
        "status": "healthy",
        "features": [
            "Enhanced document chunking with LangChain",
            "Intelligent query expansion",
            "Structured context retrieval",
            "Multi-turn conversation support"
        ],
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
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "services": {},
        "version": "2.0.0"
    }
    
    try:
        chat_memory.db.command('ping')
        health_status["services"]["mongodb"] = "connected"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        health_status["services"]["mongodb"] = "disconnected"
        health_status["status"] = "degraded"
    
    try:
        pinecone_index.describe_index_stats()
        health_status["services"]["pinecone"] = "connected"
    except Exception as e:
        logger.error(f"Pinecone health check failed: {e}")
        health_status["services"]["pinecone"] = "disconnected"
        health_status["status"] = "degraded"
    
    try:
        openai_client.models.list()
        health_status["services"]["openai"] = "connected"
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        health_status["services"]["openai"] = "disconnected"
        health_status["status"] = "degraded"
    
    if any(status == "disconnected" for service, status in health_status["services"].items()):
        health_status["status"] = "unhealthy"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port,
        log_level="info"
    )
"""
FastAPI backend for GenCare Assistant RAG chatbot
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
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
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., min_length=1, description="User's question")


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    sources_used: int


# RAG Components
class RAGPipeline:
    """Retrieval-Augmented Generation pipeline"""
    
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
    
    @staticmethod
    def retrieve_context(query_embedding: List[float], top_k: int = None) -> List[Dict]:
        """Retrieve relevant chunks from Pinecone"""
        if top_k is None:
            top_k = settings.top_k_results
        
        try:
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            contexts = []
            for match in results['matches']:
                if match['score'] > 0.7:  # Relevance threshold
                    contexts.append({
                        "text": match['metadata']['text'],
                        "score": match['score'],
                        "source": match['metadata'].get('source', 'BRS V1.2')
                    })
            
            logger.info(f"Retrieved {len(contexts)} relevant contexts")
            return contexts
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise
    
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
4. If information is not in the documentation, clearly state: "I'm sorry, I couldn't find that in the GenCare documentation."
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
                messages=[{"role": "system", "content": "You are a helpful assistant specialized in GenCare documentation."}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise


@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received chat request from session: {request.session_id}")
    try:
        session_id = request.session_id
        query = request.query
        history = chat_memory.get_session_history(session_id)
        query_embedding = RAGPipeline.get_query_embedding(query)
        contexts = RAGPipeline.retrieve_context(query_embedding)
        if not contexts:
            answer = "I'm sorry, I couldn't find that in the GenCare documentation."
            sources_used = 0
        else:
            chat_history = RAGPipeline.format_chat_history(history)
            prompt = RAGPipeline.build_rag_prompt(query, contexts, chat_history)
            answer = RAGPipeline.generate_answer(prompt)
            sources_used = len(contexts)
        chat_memory.add_message(session_id, query, answer)
        logger.info(f"Successfully processed request for session: {session_id}")
        return ChatResponse(session_id=session_id, query=query, answer=answer, sources_used=sources_used)
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {str(e)}")


@app.get("/")
async def root():
    return {"message": "GenCare Assistant API is running", "version": "1.0.0", "status": "healthy"}


@app.get("/health")
async def health_check():
    try:
        pinecone_index.describe_index_stats()
        return {"status": "healthy", "database": "connected", "vector_store": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)"""
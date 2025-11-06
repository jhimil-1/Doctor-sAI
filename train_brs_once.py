"""
One-time script to process BRS V1.2 document and upload to Pinecone
"""
import os
import logging
from typing import List, Dict
from docx import Document
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import time

from config import get_settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


class BRSDocumentProcessor:
    """Process BRS document and upload to Pinecone"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def read_docx(self, file_path: str) -> str:
        """
        Read content from DOCX file
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            doc = Document(file_path)
            full_text = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            full_text.append(cell.text)
            
            content = "\n".join(full_text)
            logger.info(f"Successfully read {len(full_text)} paragraphs from {file_path}")
            return content
        except Exception as e:
            logger.error(f"Failed to read DOCX file: {e}")
            raise
    
    def _get_semantic_boundaries(self, text: str) -> List[tuple]:
        """
        Use LLM to identify semantic boundaries in the text
        """
        try:
            # Use a prompt to identify section boundaries
            prompt = """Analyze the following text and identify natural section breaks where the topic shifts. 
            Return the character positions where these breaks occur as a comma-separated list of integers. 
            Only include major topic shifts, not minor transitions.
            
            Text:""" + text[:4000]  # Limit context window
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies topic boundaries in text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            # Parse the response to get boundary positions
            boundaries = [0]  # Start with beginning of text
            try:
                # Try to parse the response as comma-separated integers
                boundaries.extend([int(pos.strip()) for pos in response.choices[0].message.content.split(',') if pos.strip().isdigit()])
            except:
                # Fallback to a simpler approach if parsing fails
                pass
                
            boundaries.append(len(text))  # End with end of text
            return sorted(list(set(boundaries)))  # Remove duplicates and sort
            
        except Exception as e:
            logger.warning(f"Failed to get semantic boundaries: {e}. Falling back to paragraph-based chunking.")
            return [0, len(text)]

    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into semantic chunks using LLM-based boundary detection
        
        Args:
            text: Full document text
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # First, try to split by semantic boundaries
        boundaries = self._get_semantic_boundaries(text)
        
        # Create initial chunks based on semantic boundaries
        semantic_chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            chunk_text = text[start:end].strip()
            if chunk_text:  # Only add non-empty chunks
                semantic_chunks.append(chunk_text)
        
        # If no semantic chunks were found, fall back to paragraph-based chunking
        if not semantic_chunks or len(semantic_chunks) == 1:
            semantic_chunks = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Now process chunks to ensure they're within size limits
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for chunk in semantic_chunks:
            chunk_tokens = len(self.encoding.encode(chunk))
            
            # If chunk is too large, split it further by paragraphs
            if chunk_tokens > settings.chunk_size * 1.5:  # Allow some flexibility
                paragraphs = [p.strip() for p in chunk.split('\n') if p.strip()]
                for para in paragraphs:
                    para_tokens = len(self.encoding.encode(para))
                    
                    # If adding this paragraph exceeds chunk size, save current chunk
                    if current_length + para_tokens > settings.chunk_size and current_chunk:
                        chunk_text = "\n".join(current_chunk)
                        chunks.append({
                            "id": f"brs_chunk_{chunk_id}",
                            "text": chunk_text,
                            "metadata": {
                                "source": "BRS V1.2",
                                "chunk_id": chunk_id,
                                "token_count": current_length,
                                "chunk_type": "semantic"
                            }
                        })
                        
                        # Keep overlap for context
                        overlap_size = 0
                        overlap_chunks = []
                        for i in range(len(current_chunk) - 1, -1, -1):
                            overlap_tokens = len(self.encoding.encode(current_chunk[i]))
                            if overlap_size + overlap_tokens <= settings.chunk_overlap:
                                overlap_chunks.insert(0, current_chunk[i])
                                overlap_size += overlap_tokens
                            else:
                                break
                        
                        current_chunk = overlap_chunks
                        current_length = overlap_size
                        chunk_id += 1
                    
                    current_chunk.append(para)
                    current_length += para_tokens
            else:
                # If chunk is within size limits, add it as is
                if current_chunk:  # If there's an existing chunk, save it first
                    chunk_text = "\n".join(current_chunk)
                    chunks.append({
                        "id": f"brs_chunk_{chunk_id}",
                        "text": chunk_text,
                        "metadata": {
                            "source": "BRS V1.2",
                            "chunk_id": chunk_id,
                            "token_count": current_length,
                            "chunk_type": "semantic"
                        }
                    })
                    chunk_id += 1
                
                # Add the new chunk
                chunks.append({
                    "id": f"brs_chunk_{chunk_id}",
                    "text": chunk,
                    "metadata": {
                        "source": "BRS V1.2",
                        "chunk_id": chunk_id,
                        "token_count": chunk_tokens,
                        "chunk_type": "semantic"
                    }
                })
                chunk_id += 1
                current_chunk = []
                current_length = 0
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "id": f"brs_chunk_{chunk_id}",
                "text": chunk_text,
                "metadata": {
                    "source": "BRS V1.2",
                    "chunk_id": chunk_id,
                    "token_count": current_length,
                    "chunk_type": "semantic"
                }
            })
        
        logger.info(f"Created {len(chunks)} semantic chunks from document")
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.openai_client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def create_pinecone_index(self):
        """Create Pinecone index with correct dimensions"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if settings.pinecone_index_name in existing_indexes:
                # Delete existing index to ensure correct dimensions
                logger.info(f"Deleting existing index: {settings.pinecone_index_name}")
                self.pc.delete_index(settings.pinecone_index_name)
                # Wait for the index to be fully deleted
                time.sleep(10)
            
            # Create new index with correct dimensions
            logger.info(f"Creating index: {settings.pinecone_index_name}")
            self.pc.create_index(
                name=settings.pinecone_index_name,
                dimension=3072,  # text-embedding-3-large dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.pinecone_environment
                )
            )
            logger.info("Index created successfully with dimension 3072")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def upload_to_pinecone(self, chunks: List[Dict[str, str]]):
        """
        Upload chunks with embeddings to Pinecone
        
        Args:
            chunks: List of chunk dictionaries
        """
        try:
            index = self.pc.Index(settings.pinecone_index_name)
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk["text"] for chunk in batch]
                
                # Get embeddings
                embeddings = self.get_embeddings(texts)
                
                # Prepare vectors for upsert
                vectors = []
                for j, chunk in enumerate(batch):
                    vectors.append({
                        "id": chunk["id"],
                        "values": embeddings[j],
                        "metadata": {
                            **chunk["metadata"],
                            "text": chunk["text"]
                        }
                    })
                
                # Upsert to Pinecone
                index.upsert(vectors=vectors)
                logger.info(f"Uploaded batch {i // batch_size + 1} ({len(vectors)} vectors)")
            
            # Get index stats
            stats = index.describe_index_stats()
            logger.info(f"Total vectors in index: {stats['total_vector_count']}")
        except Exception as e:
            logger.error(f"Failed to upload to Pinecone: {e}")
            raise
    
    def process_and_upload(self, docx_path: str):
        """
        Main pipeline: read, chunk, embed, and upload
        
        Args:
            docx_path: Path to BRS DOCX file
        """
        logger.info("=" * 80)
        logger.info("Starting BRS Document Processing Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Read document
        logger.info("Step 1: Reading BRS document...")
        text = self.read_docx(docx_path)
        
        # Step 2: Chunk text
        logger.info("Step 2: Chunking document...")
        chunks = self.chunk_text(text)
        
        # Step 3: Create Pinecone index
        logger.info("Step 3: Setting up Pinecone index...")
        self.create_pinecone_index()
        
        # Step 4: Upload to Pinecone
        logger.info("Step 4: Uploading to Pinecone...")
        self.upload_to_pinecone(chunks)
        
        logger.info("=" * 80)
        logger.info("BRS Processing Complete!")
        logger.info(f"Total chunks created and uploaded: {len(chunks)}")
        logger.info("=" * 80)


def main():
    """Main execution function"""
    # Path to your BRS document
    BRS_DOCX_PATH = "C:/Users/nhz/Desktop/GenCare Assistant/BRS V1.2 (1).docx"
    
    if not os.path.exists(BRS_DOCX_PATH):
        logger.error(f"BRS document not found at: {BRS_DOCX_PATH}")
        logger.error("Please ensure the file exists in the current directory")
        return
    
    try:
        processor = BRSDocumentProcessor()
        processor.process_and_upload(BRS_DOCX_PATH)
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
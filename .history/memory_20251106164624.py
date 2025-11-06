"""
Memory management using MongoDB for session-based chat history
"""
from typing import List, Dict, Optional
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
import logging

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatMemory:
    """Manages chat history per session in MongoDB"""
    
    def __init__(self):
        try:
            self.client = MongoClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_db_name]
            self.collection = self.db[settings.mongodb_collection_name]
            
            # Create index on session_id for faster queries
            self.collection.create_index([("session_id", ASCENDING)])
            logger.info("MongoDB connection established successfully")
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def add_message(self, session_id: str, query: str, answer: str) -> bool:
        """
        Add a new message exchange to the session history
        
        Args:
            session_id: Unique session identifier
            query: User's question
            answer: Assistant's response
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            document = {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "timestamp": datetime.utcnow()
            }
            self.collection.insert_one(document)
            logger.info(f"Message added for session: {session_id}")
            return True
        except PyMongoError as e:
            logger.error(f"Failed to add message for session {session_id}: {e}")
            return False
    
    def get_session_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        Retrieve chat history for a session
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of recent messages to retrieve
            
        Returns:
            List of message dictionaries with 'query' and 'answer'
        """
        try:
            cursor = self.collection.find(
                {"session_id": session_id}
            ).sort("timestamp", ASCENDING).limit(limit)
            
            history = []
            for doc in cursor:
                history.append({
                    "query": doc["query"],
                    "answer": doc["answer"]
                })
            
            logger.info(f"Retrieved {len(history)} messages for session: {session_id}")
            return history
        except PyMongoError as e:
            logger.error(f"Failed to retrieve history for session {session_id}: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages for a specific session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.collection.delete_many({"session_id": session_id})
            logger.info(f"Cleared {result.deleted_count} messages for session: {session_id}")
            return True
        except PyMongoError as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session has any history
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if session exists, False otherwise
        """
        try:
            count = self.collection.count_documents({"session_id": session_id}, limit=1)
            return count > 0
        except PyMongoError as e:
            logger.error(f"Failed to check session existence {session_id}: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
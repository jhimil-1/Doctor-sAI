"""
Memory management using MongoDB for session-based chat history
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from uuid import uuid4
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError
import logging

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatMemory:
    """Manages chat history and sessions in MongoDB"""
    
    def __init__(self):
        try:
            self.client = MongoClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_db_name]
            self.messages_collection = self.db[f"{settings.mongodb_collection_name}_messages"]
            self.sessions_collection = self.db[f"{settings.mongodb_collection_name}_sessions"]
            
            # Create indexes for better query performance
            self.messages_collection.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
            self.sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
            self.sessions_collection.create_index([("created_at", DESCENDING)])
            logger.info("MongoDB connection established successfully")
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new chat session with optional metadata
        
        Args:
            metadata: Optional metadata to store with the session
            
        Returns:
            str: The newly created session ID
        """
        session_id = str(uuid4())
        try:
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "metadata": metadata or {},
                "is_active": True
            }
            self.sessions_collection.insert_one(session_data)
            logger.info(f"Created new session: {session_id}")
            return session_id
        except PyMongoError as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information by ID
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            Optional[Dict]: Session data if found, None otherwise
        """
        try:
            session = self.sessions_collection.find_one({"session_id": session_id})
            if session:
                # Convert ObjectId to string for JSON serialization
                session['_id'] = str(session['_id'])
                # Convert datetime to ISO format
                for time_field in ['created_at', 'updated_at']:
                    if time_field in session and session[time_field]:
                        session[time_field] = session[time_field].isoformat()
            return session
        except PyMongoError as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update session metadata
        
        Args:
            session_id: The session ID to update
            metadata: Dictionary of metadata to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "metadata": metadata,
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Failed to update metadata for session {session_id}: {e}")
            return False
    
    def add_message(self, session_id: str, query: str, answer: str) -> bool:
        """
        Add a new message exchange to the session history and update session timestamp
        
        Args:
            session_id: Unique session identifier
            query: User's question
            answer: Assistant's response
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, verify session exists
            if not self.sessions_collection.find_one({"session_id": session_id}):
                logger.warning(f"Attempted to add message to non-existent session: {session_id}")
                return False
                
            # Add the message
            document = {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "timestamp": datetime.now(timezone.utc)
            }
            self.messages_collection.insert_one(document)
            
            # Update session's last updated timestamp
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"Message added for session: {session_id}")
            return True
        except PyMongoError as e:
            logger.error(f"Failed to add message for session {session_id}: {e}")
            return False
    
    def get_session_history(
        self, 
        session_id: str, 
        limit: int = 10,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of recent messages to retrieve
            include_metadata: Whether to include MongoDB _id and timestamp
            
        Returns:
            List of message dictionaries with 'query' and 'answer'
        """
        try:
            cursor = self.messages_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", DESCENDING).limit(limit)
            
            history = []
            for doc in cursor:
                message = {
                    "query": doc["query"],
                    "answer": doc["answer"],
                    "timestamp": doc["timestamp"].isoformat()
                }
                if include_metadata:
                    message["id"] = str(doc["_id"])
                history.append(message)
            
            # Return in chronological order
            history.reverse()
            
            logger.info(f"Retrieved {len(history)} messages for session: {session_id}")
            return history
        except PyMongoError as e:
            logger.error(f"Failed to retrieve history for session {session_id}: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages for a specific session and update session metadata
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete all messages for this session
            result = self.messages_collection.delete_many({"session_id": session_id})
            
            # Update session metadata
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            
            logger.info(f"Cleared {result.deleted_count} messages for session: {session_id}")
            return True
        except PyMongoError as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists in the sessions collection
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if session exists, False otherwise
        """
        try:
            count = self.sessions_collection.count_documents(
                {"session_id": session_id, "is_active": True}, 
                limit=1
            )
            return count > 0
        except PyMongoError as e:
            logger.error(f"Failed to check session existence {session_id}: {e}")
            return False
            
    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all active sessions with their metadata
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries with metadata
        """
        try:
            sessions = []
            cursor = self.sessions_collection.find(
                {"is_active": True}
            ).sort("updated_at", DESCENDING).limit(limit)
            
            for doc in cursor:
                session = {
                    "session_id": doc["session_id"],
                    "created_at": doc["created_at"].isoformat(),
                    "updated_at": doc["updated_at"].isoformat(),
                    "message_count": self.messages_collection.count_documents({"session_id": doc["session_id"]}),
                    "metadata": doc.get("metadata", {})
                }
                sessions.append(session)
                
            return sessions
        except PyMongoError as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
            
    def end_session(self, session_id: str) -> bool:
        """
        Mark a session as inactive
        
        Args:
            session_id: The session ID to end
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "is_active": False,
                        "ended_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                }
            )
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
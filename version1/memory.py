"""
Enhanced memory management using MongoDB for session-based chat history with LangChain integration
"""
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError
import logging
import json
from bson import json_util
import hashlib

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ChatMemory:
    """Enhanced chat memory management with LangChain integration support"""
    
    def __init__(self):
        try:
            self.client = MongoClient(
                settings.mongodb_uri,
                maxPoolSize=50,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                retryWrites=True,
                tz_aware=True
            )
            self.db = self.client[settings.mongodb_db_name]
            self.messages_collection = self.db[f"{settings.mongodb_collection_name}_messages"]
            self.sessions_collection = self.db[f"{settings.mongodb_collection_name}_sessions"]
            self.rag_context_collection = self.db[f"{settings.mongodb_collection_name}_rag_context"]
            self.query_cache_collection = self.db[f"{settings.mongodb_collection_name}_query_cache"]
            
            # Create indexes for better query performance
            self._create_indexes()
            
            # Verify connection
            self.client.admin.command('ping')
            logger.info("✅ MongoDB connection established successfully")
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create necessary database indexes for optimal performance"""
        try:
            # Sessions collection indexes
            self.sessions_collection.create_index([("session_id", ASCENDING)], unique=True)
            self.sessions_collection.create_index([("created_at", DESCENDING)])
            self.sessions_collection.create_index([("updated_at", DESCENDING)])
            self.sessions_collection.create_index([("last_activity", DESCENDING)])
            self.sessions_collection.create_index([("is_active", ASCENDING)])
            self.sessions_collection.create_index([("metadata.user_id", ASCENDING)])
            
            # Messages collection indexes
            self.messages_collection.create_index([("session_id", ASCENDING), ("timestamp", ASCENDING)])
            self.messages_collection.create_index([("session_id", ASCENDING), ("timestamp", DESCENDING)])
            self.messages_collection.create_index([("timestamp", DESCENDING)])
            self.messages_collection.create_index([("session_id", ASCENDING), ("message_type", ASCENDING)])
            
            # RAG context collection indexes
            self.rag_context_collection.create_index([("session_id", ASCENDING), ("timestamp", DESCENDING)])
            self.rag_context_collection.create_index([("message_id", ASCENDING)])
            self.rag_context_collection.create_index([("query_hash", ASCENDING)])
            
            # Query cache collection indexes
            self.query_cache_collection.create_index([("query_hash", ASCENDING)], unique=True)
            self.query_cache_collection.create_index([("timestamp", ASCENDING)], expireAfterSeconds=3600)  # 1 hour TTL
            
            logger.info("Database indexes created successfully")
            
        except PyMongoError as e:
            logger.warning(f"Index creation warning: {e}")
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new chat session with enhanced metadata tracking
        
        Args:
            metadata: Optional metadata to store with the session
            
        Returns:
            str: The newly created session ID
        """
        session_id = str(uuid4())
        try:
            current_time = datetime.now(timezone.utc)
            
            session_data = {
                "session_id": session_id,
                "created_at": current_time,
                "updated_at": current_time,
                "last_activity": current_time,
                "metadata": metadata or {},
                "is_active": True,
                "message_count": 0,
                "total_queries": 0,
                "total_tokens_estimated": 0,
                "rag_queries_count": 0,
                "greeting_count": 0,
                "user_id": metadata.get("user_id") if metadata else None,
                "user_agent": metadata.get("user_agent") if metadata else None,
                "ip_address": metadata.get("ip_address") if metadata else None,
                "session_duration_seconds": 0,
                "conversation_topics": []
            }
            
            self.sessions_collection.insert_one(session_data)
            logger.info(f"✅ Created new session: {session_id}")
            return session_id
            
        except DuplicateKeyError:
            logger.warning(f"Session {session_id} already exists")
            return session_id
        except PyMongoError as e:
            logger.error(f"❌ Failed to create session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session information by ID with enhanced statistics
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            Optional[Dict]: Session data if found, None otherwise
        """
        try:
            session = self.sessions_collection.find_one({"session_id": session_id})
            if not session:
                return None
            
            # Convert to serializable format
            session = self._serialize_document(session)
            
            # Add real-time statistics
            message_count = self.messages_collection.count_documents({"session_id": session_id})
            session["message_count"] = message_count
            
            # Calculate session duration
            if session.get("created_at"):
                created_at = datetime.fromisoformat(session["created_at"].replace('Z', '+00:00'))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                duration = (now_utc - created_at).total_seconds()
                session["session_duration_seconds"] = round(duration, 2)
                session["session_duration_minutes"] = round(duration / 60, 2)
            
            return session
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to get session {session_id}: {e}")
            return None
    
    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update session metadata with merge functionality
        
        Args:
            session_id: The session ID to update
            metadata: Dictionary of metadata to update (will be merged with existing)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Prepare metadata updates
            metadata_updates = {f"metadata.{k}": v for k, v in metadata.items()}
            
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "updated_at": current_time,
                        "last_activity": current_time,
                        **metadata_updates
                    }
                },
                upsert=False
            )
            
            return result.modified_count > 0
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to update metadata for session {session_id}: {e}")
            return False
    
    def add_message(
        self, 
        session_id: str, 
        query: str, 
        answer: str, 
        rag_context: Optional[Dict[str, Any]] = None,
        message_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new message exchange to the session history with enhanced RAG context support
        
        Args:
            session_id: Unique session identifier
            query: User's question
            answer: Assistant's response
            rag_context: Optional RAG context information (sources, scores, etc.)
            message_metadata: Optional additional metadata for this specific message
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Verify session exists or create it
            if not self.session_exists(session_id):
                logger.warning(f"⚠️ Session {session_id} not found, creating new one")
                self.create_session({"auto_created": True, "first_query": query})
            
            current_time = datetime.now(timezone.utc)
            
            # Determine message type
            is_greeting = message_metadata.get("is_greeting", False) if message_metadata else False
            has_rag = rag_context is not None and rag_context.get("sources_used", 0) > 0
            
            # Create message document
            message_doc = {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "timestamp": current_time,
                "message_type": "greeting" if is_greeting else "rag_query" if has_rag else "standard",
                "query_length": len(query),
                "answer_length": len(answer),
                "query_word_count": len(query.split()),
                "answer_word_count": len(answer.split()),
                "has_rag_context": has_rag,
                "sources_used": rag_context.get("sources_used", 0) if rag_context else 0,
                "metadata": message_metadata or {}
            }
            
            message_result = self.messages_collection.insert_one(message_doc)
            message_id = str(message_result.inserted_id)
            
            # Store RAG context if provided
            if has_rag:
                self._store_rag_context(session_id, message_id, query, rag_context)
            
            # Update session statistics
            self._update_session_stats(
                session_id, 
                current_time, 
                is_greeting=is_greeting, 
                has_rag=has_rag
            )
            
            logger.info(f"✅ Message added for session: {session_id} (Type: {message_doc['message_type']}, RAG: {has_rag})")
            return True
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to add message for session {session_id}: {e}")
            return False
    
    def _store_rag_context(
        self, 
        session_id: str, 
        message_id: str, 
        query: str, 
        rag_context: Dict[str, Any]
    ):
        """Store detailed RAG context information for analysis and debugging"""
        try:
            # Generate query hash for caching/deduplication
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            
            context_doc = {
                "session_id": session_id,
                "message_id": message_id,
                "query": query,
                "query_hash": query_hash,
                "timestamp": datetime.now(timezone.utc),
                "sources_used": rag_context.get("sources_used", 0),
                "context_types": rag_context.get("context_types", []),
                "source_documents": rag_context.get("source_documents", []),
                "retrieval_scores": rag_context.get("retrieval_scores", []),
                "average_score": rag_context.get("average_score", 0),
                "max_score": rag_context.get("max_score", 0),
                "min_score": rag_context.get("min_score", 0),
                "embedding_model": rag_context.get("embedding_model", settings.embedding_model),
                "retrieval_strategy": rag_context.get("retrieval_strategy", "semantic_search"),
                "query_variations_used": rag_context.get("query_variations_used", False),
                "retrieval_method": rag_context.get("retrieval_method", "standard")
            }
            
            self.rag_context_collection.insert_one(context_doc)
            logger.debug(f"RAG context stored for message {message_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to store RAG context: {e}")
    
    def _update_session_stats(
        self, 
        session_id: str, 
        current_time: datetime,
        is_greeting: bool = False,
        has_rag: bool = False
    ):
        """Update session statistics with detailed tracking"""
        try:
            # Get current statistics
            message_count = self.messages_collection.count_documents({"session_id": session_id})
            rag_count = self.messages_collection.count_documents({
                "session_id": session_id,
                "has_rag_context": True
            })
            greeting_count = self.messages_collection.count_documents({
                "session_id": session_id,
                "message_type": "greeting"
            })
            
            # Calculate estimated tokens (rough estimate)
            pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": None,
                    "total_chars": {"$sum": {"$add": ["$query_length", "$answer_length"]}}
                }}
            ]
            token_stats = list(self.messages_collection.aggregate(pipeline))
            estimated_tokens = token_stats[0]["total_chars"] // 4 if token_stats else 0  # Rough estimate
            
            # Update session
            update_data = {
                "updated_at": current_time,
                "last_activity": current_time,
                "message_count": message_count,
                "total_queries": message_count,
                "rag_queries_count": rag_count,
                "greeting_count": greeting_count,
                "total_tokens_estimated": estimated_tokens
            }
            
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to update session stats: {e}")
    
    def get_session_history(
        self, 
        session_id: str, 
        limit: int = 20,
        offset: int = 0,
        include_rag_context: bool = False,
        message_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session with pagination and filtering
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of recent messages to retrieve
            offset: Number of messages to skip (for pagination)
            include_rag_context: Whether to include RAG context information
            message_type: Filter by message type (greeting, rag_query, standard)
            
        Returns:
            List of message dictionaries
        """
        try:
            # Build query filter
            query_filter = {"session_id": session_id}
            if message_type:
                query_filter["message_type"] = message_type
            
            cursor = self.messages_collection.find(
                query_filter
            ).sort("timestamp", ASCENDING).skip(offset).limit(limit)
            
            history = []
            for doc in cursor:
                message = {
                    "id": str(doc["_id"]),
                    "query": doc["query"],
                    "answer": doc["answer"],
                    "timestamp": doc["timestamp"].isoformat(),
                    "message_type": doc.get("message_type", "standard"),
                    "has_rag_context": doc.get("has_rag_context", False),
                    "sources_used": doc.get("sources_used", 0)
                }
                
                # Include RAG context if requested
                if include_rag_context and doc.get("has_rag_context"):
                    rag_context = self.rag_context_collection.find_one({
                        "message_id": str(doc["_id"])
                    })
                    if rag_context:
                        message["rag_context"] = {
                            "sources_used": rag_context.get("sources_used", 0),
                            "context_types": rag_context.get("context_types", []),
                            "retrieval_scores": rag_context.get("retrieval_scores", []),
                            "average_score": rag_context.get("average_score", 0),
                            "retrieval_strategy": rag_context.get("retrieval_strategy", "unknown")
                        }
                
                history.append(message)
            
            logger.debug(f"Retrieved {len(history)} messages for session: {session_id}")
            return history
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to retrieve history for session {session_id}: {e}")
            return []
    
    def get_recent_conversation(
        self, 
        session_id: str, 
        message_count: int = 5,
        exclude_greetings: bool = True
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation for context (LangChain-friendly format)
        
        Args:
            session_id: Unique session identifier
            message_count: Number of recent messages to include
            exclude_greetings: Whether to exclude greeting messages
            
        Returns:
            List of message dictionaries in simplified format
        """
        try:
            # Build filter
            query_filter = {"session_id": session_id}
            if exclude_greetings:
                query_filter["message_type"] = {"$ne": "greeting"}
            
            cursor = self.messages_collection.find(
                query_filter
            ).sort("timestamp", DESCENDING).limit(message_count)
            
            # Convert to list and reverse to get chronological order
            messages = list(cursor)
            messages.reverse()
            
            # Convert to simplified format
            conversation = []
            for msg in messages:
                conversation.append({
                    "query": msg["query"],
                    "answer": msg["answer"]
                })
            
            return conversation
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent conversation for session {session_id}: {e}")
            return []
    
    def get_conversation_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive conversation summary and statistics
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Dictionary with detailed conversation statistics
        """
        try:
            # Get session info
            session = self.get_session(session_id)
            if not session:
                return None
            
            # Get message statistics
            pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": "$session_id",
                    "total_messages": {"$sum": 1},
                    "total_query_length": {"$sum": "$query_length"},
                    "total_answer_length": {"$sum": "$answer_length"},
                    "total_query_words": {"$sum": "$query_word_count"},
                    "total_answer_words": {"$sum": "$answer_word_count"},
                    "avg_query_length": {"$avg": "$query_length"},
                    "avg_answer_length": {"$avg": "$answer_length"},
                    "first_message": {"$min": "$timestamp"},
                    "last_message": {"$max": "$timestamp"}
                }}
            ]
            
            stats = list(self.messages_collection.aggregate(pipeline))
            stat_info = stats[0] if stats else {}
            
            # Get message type distribution
            type_pipeline = [
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": "$message_type",
                    "count": {"$sum": 1}
                }}
            ]
            type_stats = list(self.messages_collection.aggregate(type_pipeline))
            message_types = {item["_id"]: item["count"] for item in type_stats}
            
            # Get RAG usage statistics
            rag_stats = list(self.rag_context_collection.aggregate([
                {"$match": {"session_id": session_id}},
                {"$group": {
                    "_id": "$session_id",
                    "rag_usage_count": {"$sum": 1},
                    "avg_sources": {"$avg": "$sources_used"},
                    "avg_score": {"$avg": "$average_score"},
                    "total_sources": {"$sum": "$sources_used"}
                }}
            ]))
            
            rag_info = rag_stats[0] if rag_stats else {}
            
            # Build comprehensive summary
            summary = {
                "session_id": session_id,
                "message_count": stat_info.get("total_messages", 0),
                "message_types": message_types,
                "conversation_stats": {
                    "total_query_length": stat_info.get("total_query_length", 0),
                    "total_answer_length": stat_info.get("total_answer_length", 0),
                    "total_words": stat_info.get("total_query_words", 0) + stat_info.get("total_answer_words", 0),
                    "average_query_length": round(stat_info.get("avg_query_length", 0), 2),
                    "average_answer_length": round(stat_info.get("avg_answer_length", 0), 2)
                },
                "rag_usage": {
                    "queries_with_rag": rag_info.get("rag_usage_count", 0),
                    "average_sources_per_query": round(rag_info.get("avg_sources", 0), 2),
                    "average_retrieval_score": round(rag_info.get("avg_score", 0), 3),
                    "total_sources_retrieved": rag_info.get("total_sources", 0)
                },
                "conversation_duration_minutes": 0,
                "active": session.get("is_active", False),
                "created_at": session.get("created_at"),
                "last_activity": session.get("last_activity")
            }
            
            # Calculate duration
            if stat_info.get("first_message") and stat_info.get("last_message"):
                first_msg = stat_info["first_message"]
                last_msg = stat_info["last_message"]
                duration = (last_msg - first_msg).total_seconds() / 60
                summary["conversation_duration_minutes"] = round(duration, 2)
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get conversation summary for session {session_id}: {e}")
            return None
    
    def cache_query_result(
        self, 
        query: str, 
        result: Dict[str, Any], 
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Cache query results for performance optimization
        
        Args:
            query: The query string
            result: The result to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            
            cache_doc = {
                "query_hash": query_hash,
                "query": query,
                "result": result,
                "timestamp": datetime.now(timezone.utc),
                "hit_count": 1
            }
            
            self.query_cache_collection.update_one(
                {"query_hash": query_hash},
                {"$set": cache_doc, "$inc": {"hit_count": 1}},
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to cache query result: {e}")
            return False
    
    def get_cached_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached query result
        
        Args:
            query: The query string
            
        Returns:
            Cached result if found and valid, None otherwise
        """
        try:
            query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()
            
            cached = self.query_cache_collection.find_one({"query_hash": query_hash})
            
            if cached:
                # Update hit count
                self.query_cache_collection.update_one(
                    {"query_hash": query_hash},
                    {"$inc": {"hit_count": 1}}
                )
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached.get("result")
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get cached query: {e}")
            return None
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages and RAG context for a specific session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete all messages for this session
            messages_result = self.messages_collection.delete_many({"session_id": session_id})
            
            # Delete RAG context for this session
            rag_result = self.rag_context_collection.delete_many({"session_id": session_id})
            
            # Reset session statistics
            current_time = datetime.now(timezone.utc)
            self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "updated_at": current_time,
                        "last_activity": current_time,
                        "message_count": 0,
                        "rag_queries_count": 0,
                        "greeting_count": 0,
                        "total_queries": 0,
                        "total_tokens_estimated": 0
                    }
                }
            )
            
            logger.info(f"✅ Cleared {messages_result.deleted_count} messages and "
                       f"{rag_result.deleted_count} RAG contexts for session: {session_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to clear session {session_id}: {e}")
            return False
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists and is active
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            bool: True if session exists and is active, False otherwise
        """
        try:
            count = self.sessions_collection.count_documents(
                {"session_id": session_id, "is_active": True}, 
                limit=1
            )
            return count > 0
        except PyMongoError as e:
            logger.error(f"❌ Failed to check session existence {session_id}: {e}")
            return False
    
    def list_sessions(
        self, 
        limit: int = 100, 
        include_stats: bool = True,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List sessions with enhanced filtering and statistics
        
        Args:
            limit: Maximum number of sessions to return
            include_stats: Whether to include detailed statistics
            active_only: Whether to only return active sessions
            
        Returns:
            List of session dictionaries with metadata and statistics
        """
        try:
            sessions = []
            
            # Build filter
            query_filter = {}
            if active_only:
                query_filter["is_active"] = True
            
            cursor = self.sessions_collection.find(
                query_filter
            ).sort("last_activity", DESCENDING).limit(limit)
            
            for doc in cursor:
                session = self._serialize_document(doc)
                
                if include_stats:
                    # Add real-time message count
                    session["message_count"] = self.messages_collection.count_documents(
                        {"session_id": session["session_id"]}
                    )
                    
                    # Get conversation summary
                    summary = self.get_conversation_summary(session["session_id"])
                    if summary:
                        session["conversation_stats"] = summary
                
                sessions.append(session)
                
            logger.debug(f"Listed {len(sessions)} sessions")
            return sessions
            
        except PyMongoError as e:
            logger.error(f"❌ Failed to list sessions: {e}")
            return []
    
    def end_session(self, session_id: str) -> bool:
        """
        Mark a session as inactive with end timestamp
        
        Args:
            session_id: The session ID to end
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc)
            
            # Get session start time to calculate total duration
            session = self.sessions_collection.find_one({"session_id": session_id})
            if session:
                created_at = session.get("created_at")
                if created_at:
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    duration = (current_time - created_at).total_seconds()
                else:
                    duration = 0
            else:
                duration = 0
            
            result = self.sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "is_active": False,
                        "ended_at": current_time,
                        "updated_at": current_time,
                        "session_duration_seconds": duration
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Ended session: {session_id} (Duration: {duration/60:.1f} minutes)")
                return True
            else:
                logger.warning(f"⚠️ Session not found or already ended: {session_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"❌ Failed to end session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 30, dry_run: bool = False) -> Tuple[int, int]:
        """
        Clean up old inactive sessions and their messages
        
        Args:
            days_old: Number of days after which to clean up sessions
            dry_run: If True, only report what would be deleted
            
        Returns:
            Tuple of (sessions_deleted, messages_deleted) counts
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
            
            # Find old inactive sessions
            old_sessions = list(self.sessions_collection.find({
                "is_active": False,
                "ended_at": {"$lt": cutoff_date}
            }))
            
            sessions_count = len(old_sessions)
            messages_count = 0
            
            if dry_run:
                # Just count messages
                for session in old_sessions:
                    session_id = session["session_id"]
                    count = self.messages_collection.count_documents({"session_id": session_id})
                    messages_count += count
                
                logger.info(f"🔍 Dry run: Would delete {sessions_count} sessions and {messages_count} messages")
                return sessions_count, messages_count
            
            # Actually delete
            sessions_deleted = 0
            messages_deleted = 0
            
            for session in old_sessions:
                session_id = session["session_id"]
                
                try:
                    # Delete messages
                    msg_result = self.messages_collection.delete_many({"session_id": session_id})
                    messages_deleted += msg_result.deleted_count
                    
                    # Delete RAG context
                    self.rag_context_collection.delete_many({"session_id": session_id})
                    
                    # Delete session
                    self.sessions_collection.delete_one({"session_id": session_id})
                    sessions_deleted += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete session {session_id}: {e}")
            
            logger.info(f"✅ Cleaned up {sessions_deleted} sessions and {messages_deleted} messages")
            return sessions_deleted, messages_deleted
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old sessions: {e}")
            return 0, 0
    
    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get analytics for the specified time period
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Session analytics
            session_stats = {
                "total_sessions": self.sessions_collection.count_documents({}),
                "active_sessions": self.sessions_collection.count_documents({"is_active": True}),
                "sessions_in_period": self.sessions_collection.count_documents({
                    "created_at": {"$gte": cutoff_date}
                })
            }
            
            # Message analytics
            message_pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": None,
                    "total_messages": {"$sum": 1},
                    "avg_query_length": {"$avg": "$query_length"},
                    "avg_answer_length": {"$avg": "$answer_length"},
                    "total_with_rag": {"$sum": {"$cond": ["$has_rag_context", 1, 0]}}
                }}
            ]
            message_stats = list(self.messages_collection.aggregate(message_pipeline))
            message_info = message_stats[0] if message_stats else {}
            
            # Message type distribution
            type_pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": "$message_type",
                    "count": {"$sum": 1}
                }}
            ]
            type_distribution = {
                item["_id"]: item["count"] 
                for item in self.messages_collection.aggregate(type_pipeline)
            }
            
            # RAG analytics
            rag_pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {"$group": {
                    "_id": None,
                    "total_rag_queries": {"$sum": 1},
                    "avg_sources": {"$avg": "$sources_used"},
                    "avg_score": {"$avg": "$average_score"},
                    "total_sources": {"$sum": "$sources_used"}
                }}
            ]
            rag_stats = list(self.rag_context_collection.aggregate(rag_pipeline))
            rag_info = rag_stats[0] if rag_stats else {}
            
            # Query cache analytics
            cache_stats = {
                "total_cached_queries": self.query_cache_collection.count_documents({}),
                "cache_hit_rate": 0  # Would need additional tracking
            }
            
            analytics = {
                "period_days": days,
                "period_start": cutoff_date.isoformat(),
                "period_end": datetime.now(timezone.utc).isoformat(),
                "sessions": session_stats,
                "messages": {
                    "total": message_info.get("total_messages", 0),
                    "avg_query_length": round(message_info.get("avg_query_length", 0), 2),
                    "avg_answer_length": round(message_info.get("avg_answer_length", 0), 2),
                    "with_rag_context": message_info.get("total_with_rag", 0),
                    "type_distribution": type_distribution
                },
                "rag_usage": {
                    "total_rag_queries": rag_info.get("total_rag_queries", 0),
                    "avg_sources_per_query": round(rag_info.get("avg_sources", 0), 2),
                    "avg_retrieval_score": round(rag_info.get("avg_score", 0), 3),
                    "total_sources_retrieved": rag_info.get("total_sources", 0)
                },
                "cache": cache_stats
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Failed to get analytics: {e}")
            return {}
    
    def get_popular_queries(self, limit: int = 10, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get most popular queries in the specified time period
        
        Args:
            limit: Number of top queries to return
            days: Time period in days
            
        Returns:
            List of popular queries with counts
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            pipeline = [
                {"$match": {
                    "timestamp": {"$gte": cutoff_date},
                    "message_type": {"$ne": "greeting"}
                }},
                {"$group": {
                    "_id": {"$toLower": "$query"},
                    "count": {"$sum": 1},
                    "avg_sources": {"$avg": "$sources_used"},
                    "latest": {"$max": "$timestamp"}
                }},
                {"$sort": {"count": DESCENDING}},
                {"$limit": limit}
            ]
            
            results = []
            for item in self.messages_collection.aggregate(pipeline):
                results.append({
                    "query": item["_id"],
                    "count": item["count"],
                    "avg_sources_used": round(item.get("avg_sources", 0), 2),
                    "latest_occurrence": item["latest"].isoformat()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get popular queries: {e}")
            return []
    
    def export_session_data(self, session_id: str, include_rag: bool = True) -> Optional[Dict[str, Any]]:
        """
        Export complete session data for backup or analysis
        
        Args:
            session_id: Session to export
            include_rag: Whether to include RAG context
            
        Returns:
            Complete session data dictionary
        """
        try:
            # Get session
            session = self.get_session(session_id)
            if not session:
                return None
            
            # Get all messages
            messages = self.get_session_history(
                session_id, 
                limit=10000,  # High limit to get all
                include_rag_context=include_rag
            )
            
            # Get summary
            summary = self.get_conversation_summary(session_id)
            
            export_data = {
                "session": session,
                "messages": messages,
                "summary": summary,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "export_version": "1.0"
            }
            
            logger.info(f"✅ Exported session {session_id} with {len(messages)} messages")
            return export_data
            
        except Exception as e:
            logger.error(f"❌ Failed to export session {session_id}: {e}")
            return None
    
    def _serialize_document(self, doc: Dict) -> Dict:
        """Convert MongoDB document to JSON-serializable format"""
        if not doc:
            return {}
        
        serialized = {}
        for key, value in doc.items():
            if key == "_id":
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_document(value)
            elif isinstance(value, list):
                serialized[key] = [
                    self._serialize_document(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                serialized[key] = value
        
        return serialized
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check MongoDB connection health and collection stats
        
        Returns:
            Health status dictionary
        """
        try:
            # Ping database
            self.client.admin.command('ping')
            
            # Get collection stats
            stats = {
                "status": "healthy",
                "collections": {
                    "sessions": {
                        "count": self.sessions_collection.count_documents({}),
                        "active": self.sessions_collection.count_documents({"is_active": True})
                    },
                    "messages": {
                        "count": self.messages_collection.count_documents({})
                    },
                    "rag_contexts": {
                        "count": self.rag_context_collection.count_documents({})
                    },
                    "query_cache": {
                        "count": self.query_cache_collection.count_documents({})
                    }
                },
                "database": settings.mongodb_db_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return stats
            
        except PyMongoError as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    def close(self):
        """Close MongoDB connection gracefully"""
        if hasattr(self, 'client') and self.client:
            try:
                self.client.close()
                logger.info("✅ MongoDB connection closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing MongoDB connection: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False  # Don't suppress exceptions
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close()
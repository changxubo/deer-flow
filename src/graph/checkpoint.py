# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from langgraph.store.memory import InMemoryStore


class ChatStreamManager:
    """
    Manages chat stream messages with persistent storage and in-memory caching.
    
    This class handles the storage and retrieval of chat messages using both
    an in-memory store for temporary data and MongoDB for persistent storage.
    It tracks message chunks and consolidates them when a conversation finishes.
    
    Attributes:
        store (InMemoryStore): In-memory storage for temporary message chunks
        mongo_client (MongoClient): MongoDB client connection
        mongo_db (Database): MongoDB database instance
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, db_uri: Optional[str] = None) -> None:
        """
        Initialize the ChatStreamManager with database connections.
        
        Args:
            db_uri: MongoDB connection URI. If None, uses MONGODB_URI env var
                   or defaults to localhost
        """
        self.logger = logging.getLogger(__name__)
        self.store = InMemoryStore()
        
        # Use provided URI or fall back to environment variable or default
        self._db_uri = db_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        
        try:
            self.mongo_client = MongoClient(self._db_uri)
            self.mongo_db: Database = self.mongo_client.checkpointing_db
            # Test connection
            self.mongo_client.admin.command('ping')
            self.logger.info("Successfully connected to MongoDB")
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def process_stream_message(self, thread_id: str, message: str, finish_reason: str) -> bool:
        """
        Process and store a chat stream message chunk.
        
        This method handles individual message chunks during streaming and consolidates
        them into a complete message when the stream finishes. Messages are stored
        temporarily in memory and permanently in MongoDB when complete.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            message: The message content or chunk to store
            finish_reason: Reason for message completion ("stop", "interrupt", or partial)
        
        Returns:
            bool: True if message was processed successfully, False otherwise
        """
        if not thread_id or not isinstance(thread_id, str):
            self.logger.warning("Invalid thread_id provided")
            return False
            
        if not message:
            self.logger.warning("Empty message provided")
            return False
        
        try:
            # Create namespace for this thread's messages
            store_namespace: Tuple[str, str] = ("messages", thread_id)
            
            # Get or initialize message cursor for tracking chunks
            cursor = self.store.get(store_namespace, "cursor")
            current_index = 0
            
            if cursor is None:
                # Initialize cursor for new conversation
                self.store.put(store_namespace, "cursor", {"index": 0})
            else:
                # Increment index for next chunk
                current_index = int(cursor.value.get("index", 0)) + 1
                self.store.put(store_namespace, "cursor", {"index": current_index})
            
            # Store the current message chunk
            self.store.put(store_namespace, f"chunk_{current_index}", message)
            
            # Check if conversation is complete and should be persisted
            if finish_reason in ("stop", "interrupt"):
                return self._persist_complete_conversation(thread_id, store_namespace, current_index)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing stream message for thread {thread_id}: {e}")
            return False
    
    def _persist_complete_conversation(self, thread_id: str, store_namespace: Tuple[str, str], 
                                     final_index: int) -> bool:
        """
        Persist completed conversation to MongoDB.
        
        Retrieves all message chunks from memory store and saves the complete
        conversation to MongoDB for permanent storage.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            store_namespace: Namespace tuple for accessing stored messages
            final_index: The final chunk index for this conversation
        
        Returns:
            bool: True if persistence was successful, False otherwise
        """
        try:
            # Retrieve all message chunks from memory store
            memories = self.store.search(store_namespace, limit=final_index + 1)
            
            # Extract message content, filtering out cursor metadata
            messages: List[str] = []
            for item in memories:
                value = item.dict().get("value", "")
                # Skip cursor metadata, only include actual message chunks
                if value and not isinstance(value, dict):
                    messages.append(str(value))
            
            if not messages:
                self.logger.warning(f"No messages found for thread {thread_id}")
                return False
            
            # Get MongoDB collection for chat streams
            collection: Collection = self.mongo_db.chat_streams
            
            # Check if conversation already exists in database
            existing_document = collection.find_one({"thread_id": thread_id})
            
            current_timestamp = datetime.now()
            
            if existing_document:
                # Update existing conversation with new messages
                update_result = collection.update_one(
                    {"thread_id": thread_id},
                    {"$set": {"messages": messages, "ts": current_timestamp}},
                )
                self.logger.info(
                    f"Updated conversation for thread {thread_id}: "
                    f"{update_result.modified_count} documents modified"
                )
                return update_result.modified_count > 0
            else:
                # Create new conversation document
                new_document = {
                    "thread_id": thread_id,
                    "messages": messages,
                    "ts": current_timestamp,
                    "id": uuid.uuid4().hex,
                }
                insert_result = collection.insert_one(new_document)
                self.logger.info(f"Created new conversation: {insert_result.inserted_id}")
                return insert_result.inserted_id is not None
                
        except Exception as e:
            self.logger.error(f"Error persisting conversation for thread {thread_id}: {e}")
            return False


# Global instance for backward compatibility
# TODO: Consider using dependency injection instead of global instance
_default_manager = ChatStreamManager()

def chat_stream_message(thread_id: str, message: str, finish_reason: str) -> bool:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        message: The message content to store
        finish_reason: Reason for message completion
    
    Returns:
        bool: True if message was processed successfully
    """
    return _default_manager.process_stream_message(thread_id, message, finish_reason)

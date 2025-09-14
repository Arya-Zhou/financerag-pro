"""
Lightweight Entity Storage using SQLite
Replace PostgreSQL with SQLite for Cloud Studio environment
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from loguru import logger
import os

class LightweightMetadataManager:
    """Lightweight metadata manager using SQLite instead of PostgreSQL"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Use SQLite database file
        self.db_path = config.get("database", {}).get("sqlite", {}).get("path", "./data/financerag.db")
        self.connection = None
        
    async def connect(self):
        """Connect to SQLite database"""
        try:
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            # Enable JSON support
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Create tables
            await self._create_tables()
            
            logger.info(f"Connected to SQLite database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables"""
        
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                file_path TEXT,
                file_name TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT REFERENCES documents(document_id),
                content TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                entities TEXT,
                relations TEXT,
                metadata TEXT,
                embedding TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT,
                name TEXT,
                aliases TEXT,
                properties TEXT
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            """
        ]
        
        cursor = self.connection.cursor()
        for statement in create_statements:
            cursor.execute(statement)
        self.connection.commit()
    
    async def store_chunk_metadata(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        entities: List[Any],
        relations: List[Any],
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """Store chunk with metadata"""
        
        # Convert to JSON strings
        entities_json = json.dumps([asdict(e) if hasattr(e, '__dict__') else e for e in entities])
        relations_json = json.dumps([asdict(r) if hasattr(r, '__dict__') else r for r in relations])
        metadata_json = json.dumps(metadata)
        embedding_json = json.dumps(embedding) if embedding else None
        
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO chunks 
            (chunk_id, document_id, content, page_number, chunk_index, entities, relations, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                document_id,
                content,
                metadata.get("page_number", 0),
                metadata.get("chunk_index", 0),
                entities_json,
                relations_json,
                metadata_json,
                embedding_json
            )
        )
        self.connection.commit()
    
    async def query_by_entities(
        self,
        entity_ids: List[str],
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query chunks by entities"""
        
        cursor = self.connection.cursor()
        
        # Simple text search in entities JSON
        results = []
        
        query = "SELECT * FROM chunks WHERE 1=1"
        params = []
        
        # Filter by entity IDs using LIKE
        if entity_ids:
            entity_conditions = []
            for entity_id in entity_ids:
                entity_conditions.append("entities LIKE ?")
                params.append(f'%"{entity_id}"%')
            
            if entity_conditions:
                query += f" AND ({' OR '.join(entity_conditions)})"
        
        query += " LIMIT 100"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        for row in rows:
            result = dict(row)
            # Parse JSON fields
            if result.get('entities'):
                result['entities'] = json.loads(result['entities'])
            if result.get('relations'):
                result['relations'] = json.loads(result['relations'])
            if result.get('metadata'):
                result['metadata'] = json.loads(result['metadata'])
            results.append(result)
        
        return results
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Alternative: Using in-memory storage for very lightweight deployment
class InMemoryMetadataManager:
    """In-memory metadata storage for testing and lightweight deployment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.documents = {}
        self.chunks = {}
        self.entities = {}
        
    async def connect(self):
        """Initialize in-memory storage"""
        logger.info("Using in-memory metadata storage")
        
    async def store_chunk_metadata(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        entities: List[Any],
        relations: List[Any],
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """Store chunk in memory"""
        
        self.chunks[chunk_id] = {
            "document_id": document_id,
            "content": content,
            "entities": entities,
            "relations": relations,
            "metadata": metadata,
            "embedding": embedding
        }
        
    async def query_by_entities(
        self,
        entity_ids: List[str],
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query chunks from memory"""
        
        results = []
        
        for chunk_id, chunk_data in self.chunks.items():
            # Check if any entity matches
            chunk_entities = chunk_data.get("entities", [])
            
            for entity in chunk_entities:
                entity_dict = asdict(entity) if hasattr(entity, '__dict__') else entity
                if entity_dict.get("entity_id") in entity_ids:
                    results.append({
                        "chunk_id": chunk_id,
                        "content": chunk_data["content"],
                        "entities": chunk_entities,
                        "metadata": chunk_data.get("metadata", {})
                    })
                    break
        
        return results[:100]  # Limit results
    
    async def close(self):
        """Clear memory"""
        self.chunks.clear()
        self.documents.clear()
        self.entities.clear()
        logger.info("In-memory storage cleared")
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from loguru import logger

@dataclass
class Entity:
    """Entity structure"""
    entity_id: str
    entity_type: str
    name: str
    aliases: List[str]
    properties: Dict[str, Any]
    
@dataclass
class EntityMention:
    """Entity mention in document"""
    mention_text: str
    entity_id: str
    entity_type: str
    confidence: float
    position: Dict[str, int]  # start, end positions

@dataclass
class Relation:
    """Relation between entities"""
    subject_id: str
    predicate: str
    object_id: str
    properties: Dict[str, Any]
    confidence: float

class FinancialEntityKnowledgeBase:
    """Financial entity knowledge base"""
    
    def __init__(self):
        self.entities = self._initialize_entities()
        self.entity_index = self._build_entity_index()
    
    def _initialize_entities(self) -> Dict[str, Entity]:
        """Initialize predefined financial entities"""
        
        entities = {}
        
        # Company entities
        companies = [
            Entity(
                entity_id="AAPL",
                entity_type="company",
                name="Apple Inc.",
                aliases=["苹果公司", "Apple", "AAPL", "苹果"],
                properties={
                    "sector": "Technology",
                    "market": "NASDAQ",
                    "country": "USA",
                    "founded": 1976
                }
            ),
            Entity(
                entity_id="MSFT",
                entity_type="company",
                name="Microsoft Corporation",
                aliases=["微软", "Microsoft", "MSFT", "微软公司"],
                properties={
                    "sector": "Technology",
                    "market": "NASDAQ",
                    "country": "USA",
                    "founded": 1975
                }
            ),
            Entity(
                entity_id="0700.HK",
                entity_type="company",
                name="Tencent Holdings",
                aliases=["腾讯", "腾讯控股", "Tencent", "0700"],
                properties={
                    "sector": "Technology",
                    "market": "HKSE",
                    "country": "China",
                    "founded": 1998
                }
            ),
            Entity(
                entity_id="BABA",
                entity_type="company",
                name="Alibaba Group",
                aliases=["阿里巴巴", "Alibaba", "BABA", "阿里"],
                properties={
                    "sector": "E-commerce",
                    "market": "NYSE",
                    "country": "China",
                    "founded": 1999
                }
            )
        ]
        
        for company in companies:
            entities[company.entity_id] = company
        
        # Financial metric entities
        metrics = [
            Entity(
                entity_id="revenue",
                entity_type="metric",
                name="Revenue",
                aliases=["营业收入", "收入", "销售额", "营收", "sales", "turnover"],
                properties={
                    "category": "income",
                    "unit": "currency",
                    "aggregation": "sum"
                }
            ),
            Entity(
                entity_id="net_profit",
                entity_type="metric",
                name="Net Profit",
                aliases=["净利润", "净利", "纯利润", "net income", "profit"],
                properties={
                    "category": "income",
                    "unit": "currency",
                    "aggregation": "sum"
                }
            ),
            Entity(
                entity_id="rd_expense",
                entity_type="metric",
                name="R&D Expense",
                aliases=["研发费用", "研发投入", "研发支出", "R&D", "research expense"],
                properties={
                    "category": "expense",
                    "unit": "currency",
                    "aggregation": "sum"
                }
            ),
            Entity(
                entity_id="total_assets",
                entity_type="metric",
                name="Total Assets",
                aliases=["总资产", "资产总额", "assets", "total assets"],
                properties={
                    "category": "balance",
                    "unit": "currency",
                    "aggregation": "point"
                }
            ),
            Entity(
                entity_id="roe",
                entity_type="metric",
                name="Return on Equity",
                aliases=["净资产收益率", "ROE", "股东回报率"],
                properties={
                    "category": "ratio",
                    "unit": "percentage",
                    "formula": "net_profit / equity"
                }
            )
        ]
        
        for metric in metrics:
            entities[metric.entity_id] = metric
        
        # Time period entities
        periods = [
            Entity(
                entity_id="2023",
                entity_type="period",
                name="Year 2023",
                aliases=["2023年", "2023", "二零二三年"],
                properties={
                    "year": 2023,
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                }
            ),
            Entity(
                entity_id="2023Q4",
                entity_type="period",
                name="Q4 2023",
                aliases=["2023年第四季度", "2023Q4", "23Q4"],
                properties={
                    "year": 2023,
                    "quarter": 4,
                    "start_date": "2023-10-01",
                    "end_date": "2023-12-31"
                }
            ),
            Entity(
                entity_id="2022",
                entity_type="period",
                name="Year 2022",
                aliases=["2022年", "2022", "二零二二年"],
                properties={
                    "year": 2022,
                    "start_date": "2022-01-01",
                    "end_date": "2022-12-31"
                }
            )
        ]
        
        for period in periods:
            entities[period.entity_id] = period
        
        return entities
    
    def _build_entity_index(self) -> Dict[str, List[str]]:
        """Build index for entity lookup"""
        
        index = {}
        
        for entity_id, entity in self.entities.items():
            # Index by name
            name_lower = entity.name.lower()
            if name_lower not in index:
                index[name_lower] = []
            index[name_lower].append(entity_id)
            
            # Index by aliases
            for alias in entity.aliases:
                alias_lower = alias.lower()
                if alias_lower not in index:
                    index[alias_lower] = []
                index[alias_lower].append(entity_id)
        
        return index
    
    def search_entity(self, text: str) -> Optional[Entity]:
        """Search for entity by text"""
        
        text_lower = text.lower().strip()
        
        # Direct lookup
        if text_lower in self.entity_index:
            entity_ids = self.entity_index[text_lower]
            if entity_ids:
                return self.entities[entity_ids[0]]
        
        # Fuzzy matching
        for key in self.entity_index:
            if key in text_lower or text_lower in key:
                entity_ids = self.entity_index[key]
                if entity_ids:
                    return self.entities[entity_ids[0]]
        
        return None
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)

class EntityLinker:
    """Entity linking engine"""
    
    def __init__(self, knowledge_base: FinancialEntityKnowledgeBase):
        self.kb = knowledge_base
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for entity recognition"""
        
        patterns = {}
        
        # Company patterns
        patterns["company"] = re.compile(
            r'(公司|集团|控股|Inc\.|Corp\.|Ltd\.|Limited|Group|Holdings)',
            re.IGNORECASE
        )
        
        # Metric patterns
        patterns["metric"] = re.compile(
            r'(收入|利润|费用|资产|负债|成本|投入|支出|income|profit|revenue|expense|cost|asset)',
            re.IGNORECASE
        )
        
        # Time patterns
        patterns["year"] = re.compile(r'20\d{2}年?')
        patterns["quarter"] = re.compile(r'(Q[1-4]|[1-4]季度|第[一二三四]季度)')
        
        # Number patterns
        patterns["number"] = re.compile(r'[\d,，]+\.?\d*[亿万千百]?')
        patterns["percentage"] = re.compile(r'[\d\.]+%')
        
        return patterns
    
    async def link_entities(self, text: str) -> List[EntityMention]:
        """Link entities in text"""
        
        mentions = []
        
        # Split text into tokens/phrases
        tokens = self._tokenize(text)
        
        # Check each token/phrase
        for i, token in enumerate(tokens):
            # Try to link to knowledge base
            entity = self.kb.search_entity(token)
            
            if entity:
                # Calculate position
                position = self._find_position(text, token)
                
                mention = EntityMention(
                    mention_text=token,
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    confidence=0.9,
                    position=position
                )
                mentions.append(mention)
            
            # Check patterns for additional entities
            else:
                pattern_mentions = self._check_patterns(token, text)
                mentions.extend(pattern_mentions)
        
        # Merge overlapping mentions
        mentions = self._merge_mentions(mentions)
        
        return mentions
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into potential entity mentions"""
        
        tokens = []
        
        # Split by common delimiters
        parts = re.split(r'[，。；：、\s,;:\n]+', text)
        
        for part in parts:
            if part:
                tokens.append(part)
                
                # Also check substrings for compound entities
                if len(part) > 2:
                    # Check 2-4 character substrings (common for Chinese entities)
                    for length in range(2, min(len(part), 10)):
                        for start in range(len(part) - length + 1):
                            substring = part[start:start + length]
                            tokens.append(substring)
        
        return list(set(tokens))  # Remove duplicates
    
    def _find_position(self, text: str, mention: str) -> Dict[str, int]:
        """Find position of mention in text"""
        
        start = text.find(mention)
        if start == -1:
            return {"start": 0, "end": 0}
        
        return {
            "start": start,
            "end": start + len(mention)
        }
    
    def _check_patterns(self, token: str, text: str) -> List[EntityMention]:
        """Check patterns for entity mentions"""
        
        mentions = []
        
        # Check year pattern
        if self.patterns["year"].search(token):
            year_match = re.search(r'(20\d{2})', token)
            if year_match:
                year = year_match.group(1)
                entity = self.kb.search_entity(year)
                
                if entity:
                    position = self._find_position(text, token)
                    mention = EntityMention(
                        mention_text=token,
                        entity_id=entity.entity_id,
                        entity_type="period",
                        confidence=0.8,
                        position=position
                    )
                    mentions.append(mention)
        
        return mentions
    
    def _merge_mentions(self, mentions: List[EntityMention]) -> List[EntityMention]:
        """Merge overlapping entity mentions"""
        
        if not mentions:
            return mentions
        
        # Sort by position
        mentions.sort(key=lambda x: x.position["start"])
        
        merged = []
        current = mentions[0]
        
        for mention in mentions[1:]:
            # Check for overlap
            if mention.position["start"] < current.position["end"]:
                # Keep the one with higher confidence
                if mention.confidence > current.confidence:
                    current = mention
            else:
                merged.append(current)
                current = mention
        
        merged.append(current)
        
        return merged

class RelationExtractor:
    """Extract relations between entities"""
    
    def __init__(self, knowledge_base: FinancialEntityKnowledgeBase):
        self.kb = knowledge_base
        self.relation_patterns = self._compile_relation_patterns()
    
    def _compile_relation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for relation extraction"""
        
        patterns = {}
        
        # Company-metric relations
        patterns["has_metric"] = re.compile(
            r'(.+?)(的|之|公司|集团).{0,10}(收入|利润|资产|费用|成本).{0,10}(为|是|达到|约|超过)(.+)',
            re.IGNORECASE
        )
        
        # Comparison relations
        patterns["compare"] = re.compile(
            r'(.+?)(高于|低于|超过|不及|大于|小于|多于|少于)(.+)',
            re.IGNORECASE
        )
        
        # Temporal relations
        patterns["temporal"] = re.compile(
            r'(.+?)(在|于|during|in)\s*(20\d{2}年?|Q[1-4])',
            re.IGNORECASE
        )
        
        return patterns
    
    async def extract_relations(
        self,
        text: str,
        entities: List[EntityMention]
    ) -> List[Relation]:
        """Extract relations from text"""
        
        relations = []
        
        # Create entity map for quick lookup
        entity_map = {e.mention_text: e for e in entities}
        
        # Check for metric value relations
        for entity in entities:
            if entity.entity_type == "metric":
                # Look for values near the metric
                value_relation = self._extract_metric_value(text, entity, entities)
                if value_relation:
                    relations.append(value_relation)
        
        # Check for comparison relations
        comparison_relations = self._extract_comparisons(text, entities)
        relations.extend(comparison_relations)
        
        # Check for temporal relations
        temporal_relations = self._extract_temporal_relations(text, entities)
        relations.extend(temporal_relations)
        
        return relations
    
    def _extract_metric_value(
        self,
        text: str,
        metric_entity: EntityMention,
        all_entities: List[EntityMention]
    ) -> Optional[Relation]:
        """Extract metric value relation"""
        
        # Look for numbers near the metric mention
        metric_pos = metric_entity.position
        context = text[max(0, metric_pos["start"] - 50):min(len(text), metric_pos["end"] + 50)]
        
        # Find numbers in context
        number_pattern = re.compile(r'([\d,，]+\.?\d*)\s*([亿万千百]?)(元|美元|人民币)?')
        matches = number_pattern.finditer(context)
        
        for match in matches:
            value_str = match.group(1).replace(",", "").replace("，", "")
            
            try:
                value = float(value_str)
                
                # Apply unit multiplier
                unit = match.group(2)
                if unit == "亿":
                    value *= 100000000
                elif unit == "万":
                    value *= 10000
                
                # Find associated company
                company_entity = None
                for entity in all_entities:
                    if entity.entity_type == "company":
                        company_entity = entity
                        break
                
                if company_entity:
                    return Relation(
                        subject_id=company_entity.entity_id,
                        predicate=f"has_{metric_entity.entity_id}",
                        object_id=str(value),
                        properties={
                            "metric": metric_entity.entity_id,
                            "value": value,
                            "unit": match.group(3) or "元"
                        },
                        confidence=0.8
                    )
            
            except ValueError:
                continue
        
        return None
    
    def _extract_comparisons(
        self,
        text: str,
        entities: List[EntityMention]
    ) -> List[Relation]:
        """Extract comparison relations"""
        
        relations = []
        
        for pattern_name, pattern in self.relation_patterns.items():
            if pattern_name == "compare":
                matches = pattern.finditer(text)
                
                for match in matches:
                    subject = match.group(1)
                    predicate = match.group(2)
                    object_text = match.group(3)
                    
                    # Find entities in subject and object
                    subject_entity = None
                    object_entity = None
                    
                    for entity in entities:
                        if entity.mention_text in subject:
                            subject_entity = entity
                        if entity.mention_text in object_text:
                            object_entity = entity
                    
                    if subject_entity and object_entity:
                        relations.append(Relation(
                            subject_id=subject_entity.entity_id,
                            predicate=f"compare_{predicate}",
                            object_id=object_entity.entity_id,
                            properties={
                                "comparison_type": predicate
                            },
                            confidence=0.7
                        ))
        
        return relations
    
    def _extract_temporal_relations(
        self,
        text: str,
        entities: List[EntityMention]
    ) -> List[Relation]:
        """Extract temporal relations"""
        
        relations = []
        
        # Find period entities
        period_entities = [e for e in entities if e.entity_type == "period"]
        
        # Find other entities with temporal context
        for entity in entities:
            if entity.entity_type != "period":
                for period in period_entities:
                    # Check if entities are close in text
                    distance = abs(entity.position["start"] - period.position["start"])
                    
                    if distance < 50:  # Within 50 characters
                        relations.append(Relation(
                            subject_id=entity.entity_id,
                            predicate="occurs_in",
                            object_id=period.entity_id,
                            properties={
                                "distance": distance
                            },
                            confidence=0.6
                        ))
        
        return relations

class MetadataManager:
    """Metadata management and storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get("database", {}).get("postgresql", {})
        self.connection = None
    
    async def connect(self):
        """Connect to PostgreSQL database"""
        
        try:
            self.connection = psycopg2.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("database", "financerag"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "password")
            )
            
            # Create tables if not exist
            await self._create_tables()
            
            logger.info("Connected to PostgreSQL database")
            
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables"""
        
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id VARCHAR(255) PRIMARY KEY,
                file_path TEXT,
                file_name VARCHAR(255),
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR(255) PRIMARY KEY,
                document_id VARCHAR(255) REFERENCES documents(document_id),
                content TEXT,
                page_number INTEGER,
                chunk_index INTEGER,
                entities JSONB,
                relations JSONB,
                metadata JSONB,
                embedding FLOAT[]
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id VARCHAR(255) PRIMARY KEY,
                entity_type VARCHAR(50),
                name VARCHAR(255),
                aliases TEXT[],
                properties JSONB
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_entities ON chunks USING GIN (entities);
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            """
        ]
        
        with self.connection.cursor() as cursor:
            for statement in create_statements:
                cursor.execute(statement)
            self.connection.commit()
    
    async def store_document_metadata(
        self,
        document_id: str,
        file_path: str,
        metadata: Dict[str, Any]
    ):
        """Store document metadata"""
        
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (document_id, file_path, file_name, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                SET metadata = EXCLUDED.metadata
                """,
                (
                    document_id,
                    file_path,
                    metadata.get("file_name", ""),
                    Json(metadata)
                )
            )
            self.connection.commit()
    
    async def store_chunk_metadata(
        self,
        chunk_id: str,
        document_id: str,
        content: str,
        entities: List[EntityMention],
        relations: List[Relation],
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """Store chunk with metadata"""
        
        # Convert entities and relations to dict
        entities_data = [asdict(e) for e in entities]
        relations_data = [asdict(r) for r in relations]
        
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, content, page_number,
                    chunk_index, entities, relations, metadata, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE
                SET entities = EXCLUDED.entities,
                    relations = EXCLUDED.relations,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                (
                    chunk_id,
                    document_id,
                    content,
                    metadata.get("page_number", 0),
                    metadata.get("chunk_index", 0),
                    Json(entities_data),
                    Json(relations_data),
                    Json(metadata),
                    embedding
                )
            )
            self.connection.commit()
    
    async def query_by_entities(
        self,
        entity_ids: List[str],
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query chunks by entities"""
        
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
                SELECT chunk_id, document_id, content, entities, relations, metadata
                FROM chunks
                WHERE 1=1
            """
            
            params = []
            
            # Filter by entity IDs
            if entity_ids:
                entity_conditions = []
                for entity_id in entity_ids:
                    entity_conditions.append(
                        "entities @> %s"
                    )
                    params.append(Json([{"entity_id": entity_id}]))
                
                if entity_conditions:
                    query += f" AND ({' OR '.join(entity_conditions)})"
            
            # Filter by entity types
            if entity_types:
                type_conditions = []
                for entity_type in entity_types:
                    type_conditions.append(
                        "entities @> %s"
                    )
                    params.append(Json([{"entity_type": entity_type}]))
                
                if type_conditions:
                    query += f" AND ({' OR '.join(type_conditions)})"
            
            query += " ORDER BY chunk_id LIMIT 100"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [dict(row) for row in results]
    
    async def close(self):
        """Close database connection"""
        
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

class EntityLinkingEngine:
    """Main entity linking and metadata management engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kb = FinancialEntityKnowledgeBase()
        self.linker = EntityLinker(self.kb)
        self.extractor = RelationExtractor(self.kb)
        self.metadata_manager = MetadataManager(config)
    
    async def initialize(self):
        """Initialize the engine"""
        await self.metadata_manager.connect()
    
    async def process_document_chunks(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process document chunks with entity linking"""
        
        processed_chunks = []
        
        for chunk in chunks:
            # Extract entities
            entities = await self.linker.link_entities(chunk["content"])
            
            # Extract relations
            relations = await self.extractor.extract_relations(
                chunk["content"],
                entities
            )
            
            # Store metadata
            await self.metadata_manager.store_chunk_metadata(
                chunk_id=chunk["chunk_id"],
                document_id=document_id,
                content=chunk["content"],
                entities=entities,
                relations=relations,
                metadata=chunk.get("metadata", {}),
                embedding=chunk.get("embedding")
            )
            
            processed_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "entities": entities,
                "relations": relations
            })
        
        logger.info(f"Processed {len(processed_chunks)} chunks with entity linking")
        
        return processed_chunks
    
    async def search_by_entities(
        self,
        query_entities: List[str]
    ) -> List[Dict[str, Any]]:
        """Search chunks by entities"""
        
        # Map query entities to entity IDs
        entity_ids = []
        for query_entity in query_entities:
            entity = self.kb.search_entity(query_entity)
            if entity:
                entity_ids.append(entity.entity_id)
        
        if not entity_ids:
            return []
        
        # Query database
        results = await self.metadata_manager.query_by_entities(entity_ids)
        
        return results
    
    async def close(self):
        """Close the engine"""
        await self.metadata_manager.close()
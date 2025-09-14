import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from loguru import logger

class QueryType(Enum):
    """Query type enumeration"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"

class RetrievalStrategy(Enum):
    """Retrieval strategy enumeration"""
    ENTITY_SEARCH = "entity_search"
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_MATCH = "keyword_match"
    TABLE_SEARCH = "table_search"
    CHART_SEARCH = "chart_search"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID_SEARCH = "hybrid_search"

@dataclass
class QueryAnalysis:
    """Query analysis result structure"""
    query_type: QueryType
    key_entities: List[str]
    sub_queries: List[str]
    required_modalities: List[str]
    temporal_range: Optional[Dict[str, str]]
    expected_output: str
    confidence: float

@dataclass
class RoutingDecision:
    """Routing decision structure"""
    primary_strategy: RetrievalStrategy
    secondary_strategies: List[RetrievalStrategy]
    parallel_execution: bool
    priority_order: List[int]
    metadata_filters: Dict[str, Any]

class QueryUnderstandingEngine:
    """Query understanding and intent analysis engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("models", {}).get("query_understanding", {}).get("model", "gpt-3.5-turbo")
        self.temperature = config.get("models", {}).get("query_understanding", {}).get("temperature", 0.1)
        self.max_tokens = config.get("models", {}).get("query_understanding", {}).get("max_tokens", 1000)
        
        # Initialize OpenAI client
        api_key = config.get("api_keys", {}).get("openai", "")
        if api_key:
            openai.api_key = api_key
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze user query and extract structured information"""
        
        prompt = self._build_analysis_prompt(query)
        
        try:
            # Call LLM for query analysis
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            analysis_data = json.loads(response)
            
            # Convert to QueryAnalysis object
            query_analysis = QueryAnalysis(
                query_type=QueryType(analysis_data.get("query_type", "factual")),
                key_entities=analysis_data.get("key_entities", []),
                sub_queries=analysis_data.get("sub_queries", []),
                required_modalities=analysis_data.get("required_modalities", ["text"]),
                temporal_range=analysis_data.get("temporal_range"),
                expected_output=analysis_data.get("expected_output", "single_value"),
                confidence=analysis_data.get("confidence", 0.8)
            )
            
            logger.info(f"Query analysis completed: {query_analysis.query_type.value}")
            return query_analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            # Return default analysis on failure
            return self._get_default_analysis(query)
    
    def _build_analysis_prompt(self, query: str) -> str:
        """Build prompt for query analysis"""
        return f"""
        Analyze the following financial query and return a JSON response with structured information.
        
        Query: "{query}"
        
        Return JSON with the following structure:
        {{
            "query_type": "factual|analytical|comparative|temporal|aggregation",
            "key_entities": ["list of identified entities like company names, metrics, etc."],
            "sub_queries": ["list of sub-queries if the main query needs decomposition"],
            "required_modalities": ["text", "table", "chart"],
            "temporal_range": {{"start": "YYYY", "end": "YYYY"}} or null,
            "expected_output": "single_value|trend|comparison|list|summary",
            "confidence": 0.0-1.0
        }}
        
        Focus on:
        1. Identifying financial entities (companies, metrics, time periods)
        2. Understanding the analytical intent
        3. Determining if multiple data sources are needed
        4. Assessing query complexity
        """
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API for analysis"""
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial query analysis expert. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _get_default_analysis(self, query: str) -> QueryAnalysis:
        """Get default analysis when LLM fails"""
        return QueryAnalysis(
            query_type=QueryType.FACTUAL,
            key_entities=[],
            sub_queries=[query],
            required_modalities=["text"],
            temporal_range=None,
            expected_output="single_value",
            confidence=0.5
        )

class QueryRouter:
    """Query routing and strategy selection engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy_rules = self._initialize_strategy_rules()
    
    def _initialize_strategy_rules(self) -> Dict[QueryType, Dict[str, Any]]:
        """Initialize routing rules for different query types"""
        return {
            QueryType.FACTUAL: {
                "primary": RetrievalStrategy.ENTITY_SEARCH,
                "secondary": [RetrievalStrategy.KEYWORD_MATCH, RetrievalStrategy.TABLE_SEARCH],
                "parallel": False,
                "priority": [1, 2, 3]
            },
            QueryType.ANALYTICAL: {
                "primary": RetrievalStrategy.SEMANTIC_SEARCH,
                "secondary": [RetrievalStrategy.GRAPH_TRAVERSAL, RetrievalStrategy.CHART_SEARCH],
                "parallel": True,
                "priority": [1, 1, 2]
            },
            QueryType.COMPARATIVE: {
                "primary": RetrievalStrategy.HYBRID_SEARCH,
                "secondary": [RetrievalStrategy.TABLE_SEARCH, RetrievalStrategy.SEMANTIC_SEARCH],
                "parallel": True,
                "priority": [1, 1, 1]
            },
            QueryType.TEMPORAL: {
                "primary": RetrievalStrategy.TABLE_SEARCH,
                "secondary": [RetrievalStrategy.CHART_SEARCH, RetrievalStrategy.ENTITY_SEARCH],
                "parallel": False,
                "priority": [1, 2, 3]
            },
            QueryType.AGGREGATION: {
                "primary": RetrievalStrategy.TABLE_SEARCH,
                "secondary": [RetrievalStrategy.SEMANTIC_SEARCH],
                "parallel": True,
                "priority": [1, 2]
            }
        }
    
    async def determine_routing(self, analysis: QueryAnalysis) -> RoutingDecision:
        """Determine routing strategy based on query analysis"""
        
        # Get base strategy from rules
        base_strategy = self.strategy_rules.get(
            analysis.query_type,
            self.strategy_rules[QueryType.FACTUAL]
        )
        
        # Build metadata filters
        metadata_filters = self._build_metadata_filters(analysis)
        
        # Adjust strategies based on modality requirements
        strategies = self._adjust_strategies_for_modalities(
            base_strategy,
            analysis.required_modalities
        )
        
        # Create routing decision
        routing_decision = RoutingDecision(
            primary_strategy=strategies["primary"],
            secondary_strategies=strategies["secondary"],
            parallel_execution=strategies["parallel"],
            priority_order=strategies["priority"],
            metadata_filters=metadata_filters
        )
        
        logger.info(f"Routing decision: {routing_decision.primary_strategy.value}")
        return routing_decision
    
    def _build_metadata_filters(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Build metadata filters from query analysis"""
        filters = {}
        
        # Add entity filters
        if analysis.key_entities:
            filters["entities"] = analysis.key_entities
        
        # Add temporal filters
        if analysis.temporal_range:
            filters["temporal_range"] = analysis.temporal_range
        
        # Add modality filters
        if analysis.required_modalities:
            filters["modalities"] = analysis.required_modalities
        
        return filters
    
    def _adjust_strategies_for_modalities(
        self,
        base_strategy: Dict[str, Any],
        required_modalities: List[str]
    ) -> Dict[str, Any]:
        """Adjust retrieval strategies based on required modalities"""
        
        adjusted = base_strategy.copy()
        
        # If tables are required, prioritize table search
        if "table" in required_modalities:
            if RetrievalStrategy.TABLE_SEARCH not in adjusted["secondary"]:
                adjusted["secondary"].insert(0, RetrievalStrategy.TABLE_SEARCH)
        
        # If charts are required, add chart search
        if "chart" in required_modalities:
            if RetrievalStrategy.CHART_SEARCH not in adjusted["secondary"]:
                adjusted["secondary"].append(RetrievalStrategy.CHART_SEARCH)
        
        return adjusted

class QueryDecomposer:
    """Query decomposition for complex queries"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_sub_queries = 5
    
    async def decompose_query(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> List[Tuple[str, QueryType]]:
        """Decompose complex query into sub-queries"""
        
        if not analysis.sub_queries:
            return [(query, analysis.query_type)]
        
        decomposed = []
        for sub_query in analysis.sub_queries[:self.max_sub_queries]:
            # Determine sub-query type
            sub_type = self._determine_sub_query_type(sub_query, analysis)
            decomposed.append((sub_query, sub_type))
        
        logger.info(f"Query decomposed into {len(decomposed)} sub-queries")
        return decomposed
    
    def _determine_sub_query_type(
        self,
        sub_query: str,
        parent_analysis: QueryAnalysis
    ) -> QueryType:
        """Determine the type of a sub-query"""
        
        # Simple heuristic-based classification
        sub_query_lower = sub_query.lower()
        
        if any(word in sub_query_lower for word in ["compare", "versus", "vs", "difference"]):
            return QueryType.COMPARATIVE
        elif any(word in sub_query_lower for word in ["trend", "growth", "change", "evolution"]):
            return QueryType.TEMPORAL
        elif any(word in sub_query_lower for word in ["analyze", "explain", "why", "reason"]):
            return QueryType.ANALYTICAL
        elif any(word in sub_query_lower for word in ["total", "sum", "average", "aggregate"]):
            return QueryType.AGGREGATION
        else:
            return QueryType.FACTUAL

class QueryRoutingEngine:
    """Main query routing engine combining all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.understanding_engine = QueryUnderstandingEngine(config)
        self.router = QueryRouter(config)
        self.decomposer = QueryDecomposer(config)
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query and return routing decisions"""
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Analyze query
        analysis = await self.understanding_engine.analyze_query(query)
        
        # Step 2: Decompose if necessary
        sub_queries = await self.decomposer.decompose_query(query, analysis)
        
        # Step 3: Determine routing for each sub-query
        routing_decisions = []
        for sub_query, sub_type in sub_queries:
            # Create sub-analysis
            sub_analysis = QueryAnalysis(
                query_type=sub_type,
                key_entities=analysis.key_entities,
                sub_queries=[],
                required_modalities=analysis.required_modalities,
                temporal_range=analysis.temporal_range,
                expected_output=analysis.expected_output,
                confidence=analysis.confidence
            )
            
            # Get routing decision
            routing = await self.router.determine_routing(sub_analysis)
            routing_decisions.append({
                "query": sub_query,
                "routing": asdict(routing)
            })
        
        return {
            "original_query": query,
            "analysis": asdict(analysis),
            "routing_decisions": routing_decisions
        }
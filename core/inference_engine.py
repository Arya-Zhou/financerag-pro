import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from loguru import logger

class AnswerType(Enum):
    """Types of answers"""
    SINGLE_VALUE = "single_value"
    MULTIPLE_VALUES = "multiple_values"
    COMPARISON = "comparison"
    TREND = "trend"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    TABLE = "table"
    LIST = "list"

@dataclass
class Answer:
    """Answer structure"""
    content: str
    answer_type: AnswerType
    confidence: float
    supporting_evidence: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    sources: List[str]

@dataclass
class InferenceContext:
    """Context for inference generation"""
    query: str
    query_analysis: Dict[str, Any]
    retrieval_results: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    user_preferences: Dict[str, Any]

class PromptBuilder:
    """Build prompts for LLM inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates"""
        
        return {
            "factual": """
Based on the following verified information, answer the user's question directly and concisely.

Question: {query}

Verified Information:
{evidence}

Instructions:
1. Provide a direct answer to the question
2. Include specific numbers and dates when available
3. Cite the source of information
4. If there's uncertainty, indicate the confidence level

Answer:
""",
            
            "analytical": """
Analyze the following information to answer the user's question with insights.

Question: {query}

Available Data:
{evidence}

Instructions:
1. Provide analytical insights based on the data
2. Identify patterns, trends, or relationships
3. Support conclusions with specific evidence
4. Explain the reasoning behind your analysis

Analysis:
""",
            
            "comparative": """
Compare and contrast the following information to answer the user's question.

Question: {query}

Data Points:
{evidence}

Instructions:
1. Compare the relevant entities or metrics
2. Highlight key differences and similarities
3. Provide quantitative comparisons when possible
4. Draw meaningful conclusions from the comparison

Comparison:
""",
            
            "summary": """
Summarize the following information to answer the user's question comprehensively.

Question: {query}

Information:
{evidence}

Instructions:
1. Provide a comprehensive summary
2. Include all relevant points
3. Organize information logically
4. Maintain factual accuracy

Summary:
"""
        }
    
    def build_prompt(
        self,
        context: InferenceContext,
        template_type: str = "factual"
    ) -> str:
        """Build prompt for inference"""
        
        # Select template
        template = self.templates.get(template_type, self.templates["factual"])
        
        # Format evidence
        evidence = self._format_evidence(context.retrieval_results)
        
        # Build prompt
        prompt = template.format(
            query=context.query,
            evidence=evidence
        )
        
        return prompt
    
    def _format_evidence(self, retrieval_results: List[Dict[str, Any]]) -> str:
        """Format retrieval results as evidence"""
        
        evidence_parts = []
        
        for i, result in enumerate(retrieval_results[:5]):  # Limit to top 5
            evidence = f"""
Evidence {i+1}:
Source: {result.get('source', 'Unknown')}
Content: {result.get('content', '')}
Confidence: {result.get('score', 0):.2f}
"""
            evidence_parts.append(evidence)
        
        return "\n".join(evidence_parts)

class LocalInferenceEngine:
    """Local model inference using quantized models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("models", {}).get("generation", {}).get("model", "Qwen/Qwen2.5-7B-Instruct-GPTQ")
        self.max_length = config.get("models", {}).get("generation", {}).get("max_length", 2048)
        self.temperature = config.get("models", {}).get("generation", {}).get("temperature", 0.7)
        
        self.model = None
        self.tokenizer = None
    
    async def initialize(self):
        """Initialize local model"""
        
        try:
            logger.info(f"Loading local model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with 4-bit quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True
            )
            
            logger.info("Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    async def generate(self, prompt: str) -> str:
        """Generate response using local model"""
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model not initialized")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Local generation failed: {str(e)}")
            raise

class APIInferenceEngine:
    """API-based inference using OpenAI or similar"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("models", {}).get("query_understanding", {}).get("model", "gpt-3.5-turbo")
        self.temperature = config.get("models", {}).get("query_understanding", {}).get("temperature", 0.1)
        self.max_tokens = config.get("models", {}).get("query_understanding", {}).get("max_tokens", 1000)
        
        # Set API key
        api_key = config.get("api_keys", {}).get("openai", "")
        if api_key:
            openai.api_key = api_key
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response using API"""
        
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"API generation failed: {str(e)}")
            raise

class AnswerFormatter:
    """Format answers for different output types"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def format_answer(
        self,
        raw_answer: str,
        answer_type: AnswerType,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Format answer based on type"""
        
        if answer_type == AnswerType.SINGLE_VALUE:
            return self._format_single_value(raw_answer, metadata)
        elif answer_type == AnswerType.TABLE:
            return self._format_table(raw_answer, metadata)
        elif answer_type == AnswerType.LIST:
            return self._format_list(raw_answer, metadata)
        elif answer_type == AnswerType.COMPARISON:
            return self._format_comparison(raw_answer, metadata)
        elif answer_type == AnswerType.TREND:
            return self._format_trend(raw_answer, metadata)
        else:
            return self._format_default(raw_answer, metadata)
    
    def _format_single_value(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Format single value answer"""
        
        # Extract the main value
        import re
        
        # Try to find numbers in the answer
        numbers = re.findall(r'[\d,，]+\.?\d*', answer)
        
        if numbers:
            main_value = numbers[0]
            
            # Add unit if available
            if metadata and "unit" in metadata:
                formatted = f"{main_value} {metadata['unit']}"
            else:
                formatted = main_value
            
            # Add context if available
            if metadata and "context" in metadata:
                formatted += f" ({metadata['context']})"
            
            return formatted
        
        return answer
    
    def _format_table(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Format table answer"""
        
        # Try to parse as JSON and format as markdown table
        try:
            data = json.loads(answer)
            
            if isinstance(data, list) and data:
                # Get headers
                headers = list(data[0].keys())
                
                # Build markdown table
                table_lines = []
                
                # Header row
                table_lines.append("| " + " | ".join(headers) + " |")
                table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                
                # Data rows
                for row in data:
                    values = [str(row.get(h, "")) for h in headers]
                    table_lines.append("| " + " | ".join(values) + " |")
                
                return "\n".join(table_lines)
        
        except:
            pass
        
        return answer
    
    def _format_list(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Format list answer"""
        
        # Split into items and format as numbered list
        lines = answer.strip().split('\n')
        
        formatted_lines = []
        for i, line in enumerate(lines, 1):
            if line.strip():
                formatted_lines.append(f"{i}. {line.strip()}")
        
        return "\n".join(formatted_lines)
    
    def _format_comparison(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Format comparison answer"""
        
        # Structure comparison with clear sections
        sections = []
        
        # Try to identify comparison elements
        if "vs" in answer.lower() or "compared to" in answer.lower():
            sections.append("**Comparison Results:**")
            sections.append(answer)
        else:
            sections.append(answer)
        
        return "\n\n".join(sections)
    
    def _format_trend(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Format trend answer"""
        
        # Add trend indicators
        formatted = "**Trend Analysis:**\n\n"
        
        # Look for trend keywords
        if "increase" in answer.lower() or "growth" in answer.lower():
            formatted += "📈 "
        elif "decrease" in answer.lower() or "decline" in answer.lower():
            formatted += "📉 "
        else:
            formatted += "➡️ "
        
        formatted += answer
        
        return formatted
    
    def _format_default(self, answer: str, metadata: Dict[str, Any]) -> str:
        """Default formatting"""
        return answer

class CitationGenerator:
    """Generate citations for answers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def generate_citations(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """Generate citations for answer"""
        
        # Extract unique sources
        unique_sources = []
        source_map = {}
        
        for source in sources:
            source_id = source.get("source", "Unknown")
            
            if source_id not in source_map:
                source_map[source_id] = len(unique_sources) + 1
                unique_sources.append(source)
        
        # Add inline citations to answer
        cited_answer = answer
        
        # Simple approach: add citations at the end of sentences mentioning data
        import re
        
        sentences = re.split(r'([.!?])', cited_answer)
        cited_sentences = []
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            # Check if sentence contains data that needs citation
            if any(char.isdigit() for char in sentence):
                # Add citation
                citation_nums = list(source_map.values())[:2]  # Use first 2 sources
                citations = " [" + ",".join(str(n) for n in citation_nums) + "]"
                cited_sentences.append(sentence + citations + punctuation)
            else:
                cited_sentences.append(sentence + punctuation)
        
        cited_answer = "".join(cited_sentences)
        
        # Format source list
        source_list = []
        for source in unique_sources:
            source_text = f"[{source_map[source.get('source', 'Unknown')]}] "
            source_text += f"{source.get('source', 'Unknown')} - "
            source_text += f"{source.get('metadata', {}).get('file_name', 'Document')}"
            
            if "page_number" in source.get("metadata", {}):
                source_text += f", Page {source['metadata']['page_number']}"
            
            source_list.append(source_text)
        
        return cited_answer, source_list

class InferenceEngine:
    """Main inference and answer generation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prompt_builder = PromptBuilder(config)
        self.api_engine = APIInferenceEngine(config)
        self.local_engine = LocalInferenceEngine(config)
        self.formatter = AnswerFormatter(config)
        self.citation_generator = CitationGenerator(config)
        
        self.use_local = config.get("models", {}).get("use_local", False)
    
    async def initialize(self):
        """Initialize inference engine"""
        
        if self.use_local:
            await self.local_engine.initialize()
    
    async def generate_answer(
        self,
        context: InferenceContext
    ) -> Answer:
        """Generate answer from context"""
        
        logger.info(f"Generating answer for query: {context.query}")
        
        # Determine answer type
        answer_type = self._determine_answer_type(context.query_analysis)
        
        # Select prompt template
        template_type = self._select_template(context.query_analysis)
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(context, template_type)
        
        # Generate raw answer
        try:
            if self.use_local:
                raw_answer = await self.local_engine.generate(prompt)
            else:
                system_prompt = "You are a financial analysis expert. Provide accurate, fact-based answers."
                raw_answer = await self.api_engine.generate(prompt, system_prompt)
        
        except Exception as e:
            logger.error(f"Generation failed, using fallback: {str(e)}")
            raw_answer = self._generate_fallback_answer(context)
        
        # Format answer
        formatted_answer = self.formatter.format_answer(
            raw_answer,
            answer_type,
            context.query_analysis
        )
        
        # Generate citations
        cited_answer, source_list = self.citation_generator.generate_citations(
            formatted_answer,
            context.retrieval_results
        )
        
        # Calculate confidence
        confidence = self._calculate_answer_confidence(
            context.validation_results,
            context.retrieval_results
        )
        
        # Create answer object
        answer = Answer(
            content=cited_answer,
            answer_type=answer_type,
            confidence=confidence,
            supporting_evidence=[
                {
                    "content": r.get("content", ""),
                    "source": r.get("source", ""),
                    "score": r.get("score", 0)
                }
                for r in context.retrieval_results[:3]
            ],
            metadata={
                "query": context.query,
                "template_type": template_type,
                "generation_time": datetime.now().isoformat()
            },
            sources=source_list
        )
        
        logger.info(f"Answer generated with confidence: {confidence:.2f}")
        
        return answer
    
    def _determine_answer_type(self, query_analysis: Dict[str, Any]) -> AnswerType:
        """Determine answer type from query analysis"""
        
        expected_output = query_analysis.get("expected_output", "single_value")
        
        mapping = {
            "single_value": AnswerType.SINGLE_VALUE,
            "multiple_values": AnswerType.MULTIPLE_VALUES,
            "trend": AnswerType.TREND,
            "comparison": AnswerType.COMPARISON,
            "list": AnswerType.LIST,
            "summary": AnswerType.SUMMARY
        }
        
        return mapping.get(expected_output, AnswerType.SUMMARY)
    
    def _select_template(self, query_analysis: Dict[str, Any]) -> str:
        """Select prompt template based on query type"""
        
        query_type = query_analysis.get("query_type", "factual")
        
        mapping = {
            "factual": "factual",
            "analytical": "analytical",
            "comparative": "comparative",
            "temporal": "analytical",
            "aggregation": "summary"
        }
        
        return mapping.get(query_type, "factual")
    
    def _generate_fallback_answer(self, context: InferenceContext) -> str:
        """Generate fallback answer when inference fails"""
        
        if context.retrieval_results:
            # Use top retrieval result
            top_result = context.retrieval_results[0]
            return f"Based on available information: {top_result.get('content', 'No specific answer found.')}"
        
        return "I couldn't find specific information to answer your question. Please try rephrasing or providing more context."
    
    def _calculate_answer_confidence(
        self,
        validation_results: Dict[str, Any],
        retrieval_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate answer confidence"""
        
        confidence = 0.5  # Base confidence
        
        # Factor in validation confidence
        if validation_results:
            validation_confidence = validation_results.get("overall_confidence", 0.5)
            confidence = confidence * 0.3 + validation_confidence * 0.7
        
        # Factor in retrieval scores
        if retrieval_results:
            avg_score = sum(r.get("score", 0) for r in retrieval_results[:3]) / min(3, len(retrieval_results))
            confidence = confidence * 0.7 + avg_score * 0.3
        
        return min(max(confidence, 0.0), 1.0)

class AnswerPostProcessor:
    """Post-process generated answers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def post_process(self, answer: Answer) -> Answer:
        """Post-process answer"""
        
        # Clean up formatting
        answer.content = self._clean_formatting(answer.content)
        
        # Add confidence indicators
        answer.content = self._add_confidence_indicators(answer.content, answer.confidence)
        
        # Add source summary
        if answer.sources:
            source_summary = "\n\n**Sources:**\n" + "\n".join(answer.sources)
            answer.content += source_summary
        
        return answer
    
    def _clean_formatting(self, content: str) -> str:
        """Clean up answer formatting"""
        
        # Remove extra whitespace
        lines = content.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return '\n\n'.join(cleaned_lines)
    
    def _add_confidence_indicators(self, content: str, confidence: float) -> str:
        """Add confidence indicators to answer"""
        
        if confidence < 0.5:
            prefix = "⚠️ **Low Confidence Answer:**\n\n"
        elif confidence < 0.8:
            prefix = "ℹ️ **Moderate Confidence Answer:**\n\n"
        else:
            prefix = "✅ **High Confidence Answer:**\n\n"
        
        return prefix + content
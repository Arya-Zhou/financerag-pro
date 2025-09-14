import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import openai
from loguru import logger
import json

class ConflictType(Enum):
    """Types of conflicts"""
    NUMERICAL_MISMATCH = "numerical_mismatch"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    SOURCE_DISAGREEMENT = "source_disagreement"
    UNIT_MISMATCH = "unit_mismatch"

class SourceAuthority(Enum):
    """Source authority levels"""
    OFFICIAL_REPORT = 1.0
    REGULATORY_FILING = 0.95
    ANNUAL_REPORT = 0.9
    QUARTERLY_REPORT = 0.85
    OFFICIAL_NEWS = 0.8
    ANALYST_REPORT = 0.6
    NEWS_ARTICLE = 0.5
    SOCIAL_MEDIA = 0.3

@dataclass
class DataPoint:
    """Data point structure"""
    value: Any
    source: str
    source_type: str
    timestamp: Optional[datetime]
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class Conflict:
    """Conflict structure"""
    conflict_type: ConflictType
    data_points: List[DataPoint]
    description: str
    severity: float  # 0-1, higher is more severe
    resolution: Optional[DataPoint] = None

@dataclass
class ValidationResult:
    """Validation result structure"""
    is_valid: bool
    confidence: float
    conflicts: List[Conflict]
    resolved_value: Optional[Any]
    explanation: str

class ConflictDetector:
    """Detect conflicts in retrieved data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tolerance = 0.01  # Numerical tolerance
    
    async def detect_conflicts(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """Detect conflicts in retrieval results"""
        
        conflicts = []
        
        # Group results by entity and metric
        grouped_data = self._group_by_entity_metric(retrieval_results)
        
        for key, data_points in grouped_data.items():
            if len(data_points) > 1:
                # Check for numerical conflicts
                numerical_conflicts = self._detect_numerical_conflicts(data_points)
                conflicts.extend(numerical_conflicts)
                
                # Check for temporal conflicts
                temporal_conflicts = self._detect_temporal_conflicts(data_points)
                conflicts.extend(temporal_conflicts)
                
                # Check for logical conflicts
                logical_conflicts = self._detect_logical_conflicts(data_points)
                conflicts.extend(logical_conflicts)
        
        logger.info(f"Detected {len(conflicts)} conflicts")
        
        return conflicts
    
    def _group_by_entity_metric(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[DataPoint]]:
        """Group results by entity and metric"""
        
        grouped = {}
        
        for result in results:
            # Extract entities and metrics from metadata
            entities = result.get("metadata", {}).get("entities", [])
            metrics = result.get("metadata", {}).get("metrics", [])
            
            for entity in entities:
                for metric in metrics:
                    key = f"{entity}_{metric}"
                    
                    if key not in grouped:
                        grouped[key] = []
                    
                    # Extract value from content
                    value = self._extract_value(result["content"], metric)
                    
                    if value is not None:
                        data_point = DataPoint(
                            value=value,
                            source=result.get("source", "unknown"),
                            source_type=result.get("source_type", "unknown"),
                            timestamp=self._parse_timestamp(result.get("metadata", {})),
                            confidence=result.get("score", 0.5),
                            metadata=result.get("metadata", {})
                        )
                        grouped[key].append(data_point)
        
        return grouped
    
    def _extract_value(self, content: str, metric: str) -> Optional[Any]:
        """Extract value for metric from content"""
        
        # Simple pattern matching for numbers
        pattern = rf'{metric}[^0-9]*([\d,，]+\.?\d*)\s*([亿万千百]?)(元|美元|%)?'
        
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            value_str = match.group(1).replace(",", "").replace("，", "")
            
            try:
                value = float(value_str)
                
                # Apply unit multiplier
                unit = match.group(2)
                if unit == "亿":
                    value *= 100000000
                elif unit == "万":
                    value *= 10000
                elif unit == "千":
                    value *= 1000
                elif unit == "百":
                    value *= 100
                
                return value
            
            except ValueError:
                pass
        
        return None
    
    def _parse_timestamp(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """Parse timestamp from metadata"""
        
        # Try to parse date from metadata
        if "date" in metadata:
            try:
                return datetime.fromisoformat(metadata["date"])
            except:
                pass
        
        if "year" in metadata:
            try:
                year = int(metadata["year"])
                return datetime(year, 1, 1)
            except:
                pass
        
        return None
    
    def _detect_numerical_conflicts(
        self,
        data_points: List[DataPoint]
    ) -> List[Conflict]:
        """Detect numerical conflicts"""
        
        conflicts = []
        
        if len(data_points) < 2:
            return conflicts
        
        # Get numerical values
        numerical_points = [
            dp for dp in data_points
            if isinstance(dp.value, (int, float))
        ]
        
        if len(numerical_points) < 2:
            return conflicts
        
        # Check for significant differences
        values = [dp.value for dp in numerical_points]
        mean_value = sum(values) / len(values)
        
        for i, dp1 in enumerate(numerical_points):
            for dp2 in numerical_points[i+1:]:
                relative_diff = abs(dp1.value - dp2.value) / max(abs(mean_value), 1)
                
                if relative_diff > self.tolerance:
                    conflict = Conflict(
                        conflict_type=ConflictType.NUMERICAL_MISMATCH,
                        data_points=[dp1, dp2],
                        description=f"Numerical mismatch: {dp1.value} vs {dp2.value}",
                        severity=min(relative_diff, 1.0)
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_temporal_conflicts(
        self,
        data_points: List[DataPoint]
    ) -> List[Conflict]:
        """Detect temporal inconsistencies"""
        
        conflicts = []
        
        # Sort by timestamp
        timestamped_points = [
            dp for dp in data_points
            if dp.timestamp is not None
        ]
        
        if len(timestamped_points) < 2:
            return conflicts
        
        timestamped_points.sort(key=lambda x: x.timestamp)
        
        # Check for inconsistent temporal progression
        for i in range(len(timestamped_points) - 1):
            dp1 = timestamped_points[i]
            dp2 = timestamped_points[i + 1]
            
            # Check if values are inconsistent with time
            if isinstance(dp1.value, (int, float)) and isinstance(dp2.value, (int, float)):
                # For metrics that should generally increase (like revenue)
                if dp2.timestamp > dp1.timestamp and dp2.value < dp1.value * 0.5:
                    conflict = Conflict(
                        conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                        data_points=[dp1, dp2],
                        description=f"Temporal inconsistency: value decreased significantly over time",
                        severity=0.7
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_logical_conflicts(
        self,
        data_points: List[DataPoint]
    ) -> List[Conflict]:
        """Detect logical contradictions"""
        
        conflicts = []
        
        # Check for contradicting statements
        for i, dp1 in enumerate(data_points):
            for dp2 in data_points[i+1:]:
                # Simple contradiction detection
                if self._are_contradictory(dp1, dp2):
                    conflict = Conflict(
                        conflict_type=ConflictType.LOGICAL_CONTRADICTION,
                        data_points=[dp1, dp2],
                        description="Logical contradiction detected",
                        severity=0.8
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _are_contradictory(self, dp1: DataPoint, dp2: DataPoint) -> bool:
        """Check if two data points are contradictory"""
        
        # Simple implementation - can be enhanced
        if isinstance(dp1.value, bool) and isinstance(dp2.value, bool):
            return dp1.value != dp2.value
        
        return False

class ConflictResolver:
    """Resolve detected conflicts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authority_scores = self._init_authority_scores()
    
    def _init_authority_scores(self) -> Dict[str, float]:
        """Initialize source authority scores"""
        
        return {
            "annual_report": 1.0,
            "quarterly_report": 0.9,
            "regulatory_filing": 0.95,
            "official_news": 0.8,
            "analyst_report": 0.6,
            "news_article": 0.5,
            "unknown": 0.3
        }
    
    async def resolve_conflicts(
        self,
        conflicts: List[Conflict]
    ) -> List[Conflict]:
        """Resolve detected conflicts"""
        
        resolved_conflicts = []
        
        for conflict in conflicts:
            # Try different resolution strategies
            resolution = None
            
            # Strategy 1: Source authority
            resolution = self._resolve_by_authority(conflict)
            
            # Strategy 2: Timestamp (most recent)
            if resolution is None:
                resolution = self._resolve_by_timestamp(conflict)
            
            # Strategy 3: Consensus
            if resolution is None:
                resolution = self._resolve_by_consensus(conflict)
            
            # Strategy 4: LLM arbitration
            if resolution is None:
                resolution = await self._resolve_by_llm(conflict)
            
            conflict.resolution = resolution
            resolved_conflicts.append(conflict)
        
        logger.info(f"Resolved {len(resolved_conflicts)} conflicts")
        
        return resolved_conflicts
    
    def _resolve_by_authority(self, conflict: Conflict) -> Optional[DataPoint]:
        """Resolve by source authority"""
        
        if not conflict.data_points:
            return None
        
        # Sort by authority score
        sorted_points = sorted(
            conflict.data_points,
            key=lambda x: self.authority_scores.get(x.source_type, 0.3),
            reverse=True
        )
        
        # Check if highest authority is significantly better
        if len(sorted_points) >= 2:
            auth1 = self.authority_scores.get(sorted_points[0].source_type, 0.3)
            auth2 = self.authority_scores.get(sorted_points[1].source_type, 0.3)
            
            if auth1 > auth2 * 1.2:  # 20% higher authority
                return sorted_points[0]
        
        return None
    
    def _resolve_by_timestamp(self, conflict: Conflict) -> Optional[DataPoint]:
        """Resolve by most recent timestamp"""
        
        timestamped_points = [
            dp for dp in conflict.data_points
            if dp.timestamp is not None
        ]
        
        if not timestamped_points:
            return None
        
        # Sort by timestamp (most recent first)
        sorted_points = sorted(
            timestamped_points,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        # Check if most recent is significantly newer
        if len(sorted_points) >= 2:
            time_diff = sorted_points[0].timestamp - sorted_points[1].timestamp
            
            if time_diff > timedelta(days=30):  # More than 30 days newer
                return sorted_points[0]
        
        return sorted_points[0] if sorted_points else None
    
    def _resolve_by_consensus(self, conflict: Conflict) -> Optional[DataPoint]:
        """Resolve by consensus (for numerical values)"""
        
        numerical_points = [
            dp for dp in conflict.data_points
            if isinstance(dp.value, (int, float))
        ]
        
        if len(numerical_points) < 3:
            return None
        
        # Calculate median value
        values = [dp.value for dp in numerical_points]
        values.sort()
        median_value = values[len(values) // 2]
        
        # Find data point closest to median
        closest_point = min(
            numerical_points,
            key=lambda x: abs(x.value - median_value)
        )
        
        # Create consensus data point
        consensus_point = DataPoint(
            value=median_value,
            source="consensus",
            source_type="calculated",
            timestamp=datetime.now(),
            confidence=0.7,
            metadata={"method": "median_consensus"}
        )
        
        return consensus_point
    
    async def _resolve_by_llm(self, conflict: Conflict) -> Optional[DataPoint]:
        """Resolve using LLM arbitration"""
        
        prompt = self._build_arbitration_prompt(conflict)
        
        try:
            response = await self._call_llm(prompt)
            
            # Parse LLM response
            resolution_data = json.loads(response)
            
            # Find the chosen data point
            chosen_index = resolution_data.get("chosen_index", 0)
            
            if 0 <= chosen_index < len(conflict.data_points):
                chosen_point = conflict.data_points[chosen_index]
                chosen_point.metadata["llm_reason"] = resolution_data.get("reason", "")
                return chosen_point
        
        except Exception as e:
            logger.error(f"LLM arbitration failed: {str(e)}")
        
        return None
    
    def _build_arbitration_prompt(self, conflict: Conflict) -> str:
        """Build prompt for LLM arbitration"""
        
        data_descriptions = []
        for i, dp in enumerate(conflict.data_points):
            desc = f"""
            Data Point {i+1}:
            - Value: {dp.value}
            - Source: {dp.source}
            - Source Type: {dp.source_type}
            - Timestamp: {dp.timestamp}
            - Confidence: {dp.confidence}
            """
            data_descriptions.append(desc)
        
        prompt = f"""
        You are a financial data arbitration expert. The following conflicting data points were found:
        
        Conflict Type: {conflict.conflict_type.value}
        Description: {conflict.description}
        
        {chr(10).join(data_descriptions)}
        
        Please analyze these data points and determine which one is most likely correct.
        
        Return a JSON response:
        {{
            "chosen_index": 0-based index of the chosen data point,
            "reason": "explanation for the choice",
            "confidence": 0.0-1.0
        }}
        
        Consider:
        1. Source authority and credibility
        2. Temporal relevance
        3. Consistency with financial principles
        4. Data quality indicators
        """
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for arbitration"""
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial data validation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

class DataValidator:
    """Validate data consistency and quality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = self._init_validation_rules()
    
    def _init_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules"""
        
        return {
            "revenue": {
                "min": 0,
                "max": 1e15,  # 1 trillion
                "unit": "currency"
            },
            "profit_margin": {
                "min": -1,
                "max": 1,
                "unit": "percentage"
            },
            "growth_rate": {
                "min": -1,
                "max": 10,
                "unit": "percentage"
            },
            "employee_count": {
                "min": 1,
                "max": 1e7,
                "unit": "count"
            }
        }
    
    async def validate_data(
        self,
        data_points: List[DataPoint],
        metric_type: str
    ) -> ValidationResult:
        """Validate data points"""
        
        # Check basic validity
        is_valid = True
        issues = []
        
        # Apply validation rules
        if metric_type in self.validation_rules:
            rules = self.validation_rules[metric_type]
            
            for dp in data_points:
                if isinstance(dp.value, (int, float)):
                    # Check range
                    if "min" in rules and dp.value < rules["min"]:
                        is_valid = False
                        issues.append(f"Value {dp.value} below minimum {rules['min']}")
                    
                    if "max" in rules and dp.value > rules["max"]:
                        is_valid = False
                        issues.append(f"Value {dp.value} above maximum {rules['max']}")
        
        # Check for consistency
        if len(data_points) > 1:
            consistency_check = self._check_consistency(data_points)
            if not consistency_check["is_consistent"]:
                is_valid = False
                issues.append(consistency_check["issue"])
        
        # Calculate confidence
        confidence = self._calculate_confidence(data_points, is_valid)
        
        # Determine resolved value
        resolved_value = None
        if data_points:
            # Use highest confidence data point
            best_point = max(data_points, key=lambda x: x.confidence)
            resolved_value = best_point.value
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            conflicts=[],  # Conflicts handled separately
            resolved_value=resolved_value,
            explanation="; ".join(issues) if issues else "Data validation passed"
        )
    
    def _check_consistency(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Check data consistency"""
        
        numerical_points = [
            dp.value for dp in data_points
            if isinstance(dp.value, (int, float))
        ]
        
        if len(numerical_points) < 2:
            return {"is_consistent": True}
        
        # Calculate coefficient of variation
        mean_val = sum(numerical_points) / len(numerical_points)
        
        if mean_val == 0:
            return {"is_consistent": True}
        
        variance = sum((x - mean_val) ** 2 for x in numerical_points) / len(numerical_points)
        std_dev = variance ** 0.5
        cv = std_dev / abs(mean_val)
        
        # High coefficient of variation indicates inconsistency
        if cv > 0.5:  # More than 50% variation
            return {
                "is_consistent": False,
                "issue": f"High data variance (CV={cv:.2f})"
            }
        
        return {"is_consistent": True}
    
    def _calculate_confidence(
        self,
        data_points: List[DataPoint],
        is_valid: bool
    ) -> float:
        """Calculate overall confidence"""
        
        if not data_points:
            return 0.0
        
        # Base confidence on data point confidences
        avg_confidence = sum(dp.confidence for dp in data_points) / len(data_points)
        
        # Adjust for validity
        if not is_valid:
            avg_confidence *= 0.5
        
        # Adjust for consistency
        if len(data_points) > 1:
            consistency = self._check_consistency(data_points)
            if not consistency.get("is_consistent", True):
                avg_confidence *= 0.7
        
        return min(avg_confidence, 1.0)

class CrossValidator:
    """Cross-validate data across different sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def cross_validate(
        self,
        primary_data: Dict[str, Any],
        supporting_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cross-validate primary data with supporting sources"""
        
        validation_results = {
            "primary_data": primary_data,
            "validation_score": 0.0,
            "supporting_evidence": [],
            "contradicting_evidence": []
        }
        
        for support in supporting_data:
            correlation = self._calculate_correlation(primary_data, support)
            
            if correlation > 0.7:
                validation_results["supporting_evidence"].append({
                    "source": support.get("source", "unknown"),
                    "correlation": correlation,
                    "data": support
                })
            elif correlation < 0.3:
                validation_results["contradicting_evidence"].append({
                    "source": support.get("source", "unknown"),
                    "correlation": correlation,
                    "data": support
                })
        
        # Calculate validation score
        total_evidence = len(validation_results["supporting_evidence"]) + \
                        len(validation_results["contradicting_evidence"])
        
        if total_evidence > 0:
            validation_results["validation_score"] = \
                len(validation_results["supporting_evidence"]) / total_evidence
        
        return validation_results
    
    def _calculate_correlation(
        self,
        data1: Dict[str, Any],
        data2: Dict[str, Any]
    ) -> float:
        """Calculate correlation between two data sources"""
        
        # Simple implementation - compare common fields
        common_fields = set(data1.keys()) & set(data2.keys())
        
        if not common_fields:
            return 0.0
        
        matches = 0
        for field in common_fields:
            if self._values_match(data1[field], data2[field]):
                matches += 1
        
        return matches / len(common_fields)
    
    def _values_match(self, val1: Any, val2: Any) -> bool:
        """Check if two values match"""
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Numerical comparison with tolerance
            return abs(val1 - val2) / max(abs(val1), abs(val2), 1) < 0.1
        
        return val1 == val2

class ValidationEngine:
    """Main validation and conflict resolution engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conflict_detector = ConflictDetector(config)
        self.conflict_resolver = ConflictResolver(config)
        self.data_validator = DataValidator(config)
        self.cross_validator = CrossValidator(config)
    
    async def validate_retrieval_results(
        self,
        retrieval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate retrieval results and resolve conflicts"""
        
        logger.info(f"Validating {len(retrieval_results)} retrieval results")
        
        # Step 1: Detect conflicts
        conflicts = await self.conflict_detector.detect_conflicts(retrieval_results)
        
        # Step 2: Resolve conflicts
        resolved_conflicts = await self.conflict_resolver.resolve_conflicts(conflicts)
        
        # Step 3: Validate data
        validation_results = []
        
        for result in retrieval_results:
            # Extract data points
            data_points = self._extract_data_points(result)
            
            # Validate each metric
            for metric, points in data_points.items():
                validation = await self.data_validator.validate_data(points, metric)
                validation_results.append({
                    "metric": metric,
                    "validation": asdict(validation)
                })
        
        # Step 4: Cross-validate
        cross_validation_results = []
        
        if len(retrieval_results) > 1:
            for i, primary in enumerate(retrieval_results):
                supporting = retrieval_results[:i] + retrieval_results[i+1:]
                cross_validation = await self.cross_validator.cross_validate(
                    primary,
                    supporting
                )
                cross_validation_results.append(cross_validation)
        
        return {
            "conflicts": [asdict(c) for c in resolved_conflicts],
            "validation_results": validation_results,
            "cross_validation": cross_validation_results,
            "overall_confidence": self._calculate_overall_confidence(
                resolved_conflicts,
                validation_results,
                cross_validation_results
            )
        }
    
    def _extract_data_points(self, result: Dict[str, Any]) -> Dict[str, List[DataPoint]]:
        """Extract data points from result"""
        
        data_points = {}
        
        # Extract from metadata
        metadata = result.get("metadata", {})
        metrics = metadata.get("metrics", {})
        
        for metric, value in metrics.items():
            if metric not in data_points:
                data_points[metric] = []
            
            data_point = DataPoint(
                value=value,
                source=result.get("source", "unknown"),
                source_type=result.get("source_type", "unknown"),
                timestamp=None,
                confidence=result.get("score", 0.5),
                metadata=metadata
            )
            
            data_points[metric].append(data_point)
        
        return data_points
    
    def _calculate_overall_confidence(
        self,
        conflicts: List[Conflict],
        validation_results: List[Dict[str, Any]],
        cross_validation_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score"""
        
        confidence = 1.0
        
        # Reduce confidence for conflicts
        if conflicts:
            avg_severity = sum(c.severity for c in conflicts) / len(conflicts)
            confidence *= (1.0 - avg_severity * 0.5)
        
        # Consider validation results
        if validation_results:
            valid_count = sum(1 for v in validation_results if v["validation"]["is_valid"])
            validation_rate = valid_count / len(validation_results)
            confidence *= validation_rate
        
        # Consider cross-validation
        if cross_validation_results:
            avg_cross_score = sum(
                cv["validation_score"] for cv in cross_validation_results
            ) / len(cross_validation_results)
            confidence *= (0.5 + avg_cross_score * 0.5)
        
        return max(0.0, min(1.0, confidence))
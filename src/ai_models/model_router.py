"""
Smart Model Router
Intelligently routes requests to the most suitable AI model based on task requirements,
performance metrics, cost optimization, and availability
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .model_interface import ModelInterface, ModelProvider, ModelRequest, ModelResponse, TaskType, ModelCapability

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Available routing strategies"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    FALLBACK_CASCADE = "fallback_cascade"


class RoutingRule(BaseModel):
    """Configuration for routing rules"""
    task_types: List[TaskType] = Field(default_factory=list)
    required_capabilities: List[ModelCapability] = Field(default_factory=list)
    preferred_providers: List[ModelProvider] = Field(default_factory=list)
    blacklisted_providers: List[ModelProvider] = Field(default_factory=list)
    max_cost_per_request: Optional[float] = Field(None)
    max_response_time_ms: Optional[float] = Field(None)
    min_confidence_score: Optional[float] = Field(None)
    priority: int = Field(default=5, description="Rule priority (1-10)")


class ModelScore(BaseModel):
    """Scoring for model selection"""
    provider: ModelProvider
    performance_score: float = Field(description="Performance-based score (0-1)")
    cost_score: float = Field(description="Cost efficiency score (0-1)")
    capability_score: float = Field(description="Capability match score (0-1)")
    availability_score: float = Field(description="Availability score (0-1)")
    combined_score: float = Field(description="Weighted combined score (0-1)")
    reasoning: str = Field(description="Explanation for the score")


class RoutingDecision(BaseModel):
    """Result of routing decision"""
    selected_provider: ModelProvider
    fallback_providers: List[ModelProvider]
    routing_strategy: RoutingStrategy
    score_breakdown: ModelScore
    decision_time_ms: float
    reasoning: str


class SmartModelRouter:
    """
    Intelligent router that selects the best AI model for each request
    """

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

        # Routing configuration
        self.default_strategy = RoutingStrategy.BALANCED
        self.routing_rules: List[RoutingRule] = []
        self.provider_weights = {
            "performance": 0.3,
            "cost": 0.2,
            "capability": 0.3,
            "availability": 0.2
        }

        # Performance tracking
        self.routing_history: List[Dict[str, Any]] = []
        self.provider_performance: Dict[ModelProvider, Dict[str, Any]] = {}

        # Fallback configuration
        self.fallback_enabled = True
        self.max_fallback_attempts = 3
        self.fallback_delay_ms = 500

    async def route_request(
        self,
        request: ModelRequest,
        strategy: Optional[RoutingStrategy] = None
    ) -> RoutingDecision:
        """
        Route a request to the most suitable model provider
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        try:
            # Get available providers
            available_providers = self.model_manager.get_available_providers()
            if not available_providers:
                raise Exception("No model providers available")

            # Apply routing rules to filter providers
            eligible_providers = await self._apply_routing_rules(request, available_providers)

            if not eligible_providers:
                self.logger.warning("No providers match routing rules, using all available")
                eligible_providers = available_providers

            # Score providers based on strategy
            provider_scores = await self._score_providers(request, eligible_providers, strategy)

            if not provider_scores:
                raise Exception("No suitable providers found")

            # Select primary and fallback providers
            selected_provider = provider_scores[0].provider
            fallback_providers = [score.provider for score in provider_scores[1:]]

            decision_time = (time.time() - start_time) * 1000

            decision = RoutingDecision(
                selected_provider=selected_provider,
                fallback_providers=fallback_providers,
                routing_strategy=strategy,
                score_breakdown=provider_scores[0],
                decision_time_ms=decision_time,
                reasoning=self._generate_routing_reasoning(provider_scores[0], strategy)
            )

            # Record routing decision
            self._record_routing_decision(request, decision)

            return decision

        except Exception as e:
            self.logger.error(f"Error in routing decision: {e}")
            # Emergency fallback to first available provider
            if available_providers:
                fallback_provider = available_providers[0]
                return RoutingDecision(
                    selected_provider=fallback_provider,
                    fallback_providers=available_providers[1:],
                    routing_strategy=RoutingStrategy.FALLBACK_CASCADE,
                    score_breakdown=ModelScore(
                        provider=fallback_provider,
                        performance_score=0.5,
                        cost_score=0.5,
                        capability_score=0.5,
                        availability_score=1.0,
                        combined_score=0.5,
                        reasoning="Emergency fallback due to routing error"
                    ),
                    decision_time_ms=(time.time() - start_time) * 1000,
                    reasoning=f"Emergency fallback: {str(e)}"
                )
            else:
                raise Exception("No providers available for emergency fallback")

    async def execute_request_with_routing(self, request: ModelRequest) -> ModelResponse:
        """
        Execute a request with intelligent routing and fallback handling
        """
        try:
            # Get routing decision
            routing_decision = await self.route_request(request)

            # Try primary provider
            provider = self.model_manager.get_provider(routing_decision.selected_provider)
            if provider:
                try:
                    response = await provider.generate_response(request)
                    if response.error is None:
                        self._record_success(routing_decision.selected_provider, response)
                        return response
                except Exception as e:
                    self.logger.warning(f"Primary provider {routing_decision.selected_provider} failed: {e}")
                    self._record_failure(routing_decision.selected_provider, str(e))

            # Try fallback providers if enabled
            if self.fallback_enabled:
                for fallback_provider in routing_decision.fallback_providers[:self.max_fallback_attempts]:
                    try:
                        await asyncio.sleep(self.fallback_delay_ms / 1000)  # Brief delay between attempts

                        provider = self.model_manager.get_provider(fallback_provider)
                        if provider:
                            response = await provider.generate_response(request)
                            if response.error is None:
                                self.logger.info(f"Successful fallback to {fallback_provider}")
                                self._record_success(fallback_provider, response)
                                return response
                    except Exception as e:
                        self.logger.warning(f"Fallback provider {fallback_provider} failed: {e}")
                        self._record_failure(fallback_provider, str(e))

            # All providers failed
            raise Exception("All providers failed to generate response")

        except Exception as e:
            self.logger.error(f"Request execution failed: {e}")
            # Return error response
            return ModelResponse(
                content="",
                provider=ModelProvider.LOCAL,  # Default fallback
                model_name="error",
                response_time_ms=0,
                error=f"Routing and execution failed: {str(e)}"
            )

    async def _apply_routing_rules(
        self,
        request: ModelRequest,
        available_providers: List[ModelProvider]
    ) -> List[ModelProvider]:
        """Apply routing rules to filter eligible providers"""
        eligible_providers = available_providers.copy()

        # Sort rules by priority
        sorted_rules = sorted(self.routing_rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            # Check if rule applies to this request
            if rule.task_types and request.task_type not in rule.task_types:
                continue

            # Apply blacklist
            if rule.blacklisted_providers:
                eligible_providers = [p for p in eligible_providers if p not in rule.blacklisted_providers]

            # Apply whitelist if specified
            if rule.preferred_providers:
                preferred_in_eligible = [p for p in rule.preferred_providers if p in eligible_providers]
                if preferred_in_eligible:
                    eligible_providers = preferred_in_eligible

            # Filter by capabilities
            if rule.required_capabilities:
                filtered_providers = []
                for provider_type in eligible_providers:
                    provider = self.model_manager.get_provider(provider_type)
                    if provider and all(provider.supports_capability(cap) for cap in rule.required_capabilities):
                        filtered_providers.append(provider_type)
                eligible_providers = filtered_providers

        return eligible_providers

    async def _score_providers(
        self,
        request: ModelRequest,
        providers: List[ModelProvider],
        strategy: RoutingStrategy
    ) -> List[ModelScore]:
        """Score providers based on routing strategy"""
        scores = []

        for provider_type in providers:
            provider = self.model_manager.get_provider(provider_type)
            if not provider:
                continue

            score = await self._calculate_provider_score(provider, request, strategy)
            scores.append(score)

        # Sort by combined score
        scores.sort(key=lambda s: s.combined_score, reverse=True)
        return scores

    async def _calculate_provider_score(
        self,
        provider: ModelInterface,
        request: ModelRequest,
        strategy: RoutingStrategy
    ) -> ModelScore:
        """Calculate comprehensive score for a provider"""

        # Performance score
        performance_score = await self._calculate_performance_score(provider)

        # Cost score
        cost_score = await self._calculate_cost_score(provider, request)

        # Capability score
        capability_score = await self._calculate_capability_score(provider, request)

        # Availability score
        availability_score = await self._calculate_availability_score(provider)

        # Apply strategy weights
        weights = self._get_strategy_weights(strategy)
        combined_score = (
            performance_score * weights["performance"] +
            cost_score * weights["cost"] +
            capability_score * weights["capability"] +
            availability_score * weights["availability"]
        )

        reasoning = self._generate_score_reasoning(
            provider.provider, performance_score, cost_score,
            capability_score, availability_score, strategy
        )

        return ModelScore(
            provider=provider.provider,
            performance_score=performance_score,
            cost_score=cost_score,
            capability_score=capability_score,
            availability_score=availability_score,
            combined_score=combined_score,
            reasoning=reasoning
        )

    async def _calculate_performance_score(self, provider: ModelInterface) -> float:
        """Calculate performance score based on historical metrics"""
        metrics = provider.get_performance_metrics()

        if metrics["total_requests"] == 0:
            return 0.7  # Default score for new providers

        # Factors: response time, error rate, consistency
        response_time_score = max(0, 1 - (metrics["avg_response_time"] / 10000))  # Normalize to 10s max
        error_rate_score = max(0, 1 - metrics["error_rate"])

        # Weight the factors
        performance_score = (response_time_score * 0.6 + error_rate_score * 0.4)

        return min(1.0, max(0.0, performance_score))

    async def _calculate_cost_score(self, provider: ModelInterface, request: ModelRequest) -> float:
        """Calculate cost efficiency score"""
        try:
            estimated_cost = provider.estimate_cost(request)
            if estimated_cost == 0:
                return 1.0  # Free is best

            # Normalize cost (assuming $0.10 as high cost per request)
            cost_score = max(0, 1 - (estimated_cost / 0.10))
            return min(1.0, cost_score)

        except Exception:
            return 0.5  # Default if cost estimation fails

    async def _calculate_capability_score(self, provider: ModelInterface, request: ModelRequest) -> float:
        """Calculate how well provider capabilities match request requirements"""
        if not request.required_capabilities:
            return 1.0  # No specific requirements

        supported_capabilities = provider.get_capabilities()
        required_capabilities = request.required_capabilities

        if not required_capabilities:
            return 1.0

        # Calculate percentage of required capabilities that are supported
        supported_count = sum(1 for cap in required_capabilities if cap in supported_capabilities)
        capability_score = supported_count / len(required_capabilities)

        # Bonus for task type support
        if request.task_type and provider.supports_task_type(request.task_type):
            capability_score = min(1.0, capability_score + 0.1)

        return capability_score

    async def _calculate_availability_score(self, provider: ModelInterface) -> float:
        """Calculate provider availability score"""
        try:
            # Simple health check for availability
            health_result = await provider.health_check()
            if health_result.get("status") == "healthy":
                return 1.0
            else:
                return 0.1  # Provider is available but not healthy
        except Exception:
            return 0.0  # Provider not available

    def _get_strategy_weights(self, strategy: RoutingStrategy) -> Dict[str, float]:
        """Get weights for different routing strategies"""
        strategy_weights = {
            RoutingStrategy.PERFORMANCE_OPTIMIZED: {
                "performance": 0.6, "cost": 0.1, "capability": 0.2, "availability": 0.1
            },
            RoutingStrategy.COST_OPTIMIZED: {
                "performance": 0.1, "cost": 0.6, "capability": 0.2, "availability": 0.1
            },
            RoutingStrategy.BALANCED: {
                "performance": 0.3, "cost": 0.2, "capability": 0.3, "availability": 0.2
            },
            RoutingStrategy.CAPABILITY_BASED: {
                "performance": 0.2, "cost": 0.1, "capability": 0.6, "availability": 0.1
            }
        }

        return strategy_weights.get(strategy, self.provider_weights)

    def _generate_score_reasoning(
        self,
        provider: ModelProvider,
        performance: float,
        cost: float,
        capability: float,
        availability: float,
        strategy: RoutingStrategy
    ) -> str:
        """Generate human-readable reasoning for the score"""
        reasons = []

        if strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            reasons.append(f"Performance: {performance:.2f}")
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            reasons.append(f"Cost efficiency: {cost:.2f}")
        elif strategy == RoutingStrategy.CAPABILITY_BASED:
            reasons.append(f"Capability match: {capability:.2f}")

        reasons.extend([
            f"Availability: {availability:.2f}",
            f"Overall fit for {strategy.value} strategy"
        ])

        return f"{provider.value} selected - " + ", ".join(reasons)

    def _generate_routing_reasoning(self, score: ModelScore, strategy: RoutingStrategy) -> str:
        """Generate reasoning for the routing decision"""
        return (
            f"Selected {score.provider.value} with {strategy.value} strategy. "
            f"Combined score: {score.combined_score:.3f}. {score.reasoning}"
        )

    def _record_routing_decision(self, request: ModelRequest, decision: RoutingDecision):
        """Record routing decision for analysis"""
        self.routing_history.append({
            "timestamp": datetime.utcnow(),
            "task_type": request.task_type,
            "selected_provider": decision.selected_provider,
            "strategy": decision.routing_strategy,
            "decision_time_ms": decision.decision_time_ms,
            "session_id": request.session_id
        })

        # Keep only recent history (last 1000 decisions)
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]

    def _record_success(self, provider: ModelProvider, response: ModelResponse):
        """Record successful response from provider"""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                "success_count": 0,
                "failure_count": 0,
                "last_success": None
            }

        self.provider_performance[provider]["success_count"] += 1
        self.provider_performance[provider]["last_success"] = datetime.utcnow()

    def _record_failure(self, provider: ModelProvider, error: str):
        """Record failed response from provider"""
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                "success_count": 0,
                "failure_count": 0,
                "last_failure": None,
                "recent_errors": []
            }

        self.provider_performance[provider]["failure_count"] += 1
        self.provider_performance[provider]["last_failure"] = datetime.utcnow()

        # Keep recent errors for analysis
        recent_errors = self.provider_performance[provider].get("recent_errors", [])
        recent_errors.append({"timestamp": datetime.utcnow(), "error": error})
        self.provider_performance[provider]["recent_errors"] = recent_errors[-10:]  # Keep last 10

    def add_routing_rule(self, rule: RoutingRule):
        """Add a new routing rule"""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions"""
        if not self.routing_history:
            return {"message": "No routing history available"}

        recent_decisions = [d for d in self.routing_history
                          if d["timestamp"] > datetime.utcnow() - timedelta(hours=24)]

        provider_usage = {}
        strategy_usage = {}

        for decision in recent_decisions:
            provider = decision["selected_provider"]
            strategy = decision["strategy"]

            provider_usage[provider.value] = provider_usage.get(provider.value, 0) + 1
            strategy_usage[strategy.value] = strategy_usage.get(strategy.value, 0) + 1

        avg_decision_time = sum(d["decision_time_ms"] for d in recent_decisions) / len(recent_decisions) if recent_decisions else 0

        return {
            "total_decisions": len(self.routing_history),
            "recent_decisions_24h": len(recent_decisions),
            "avg_decision_time_ms": avg_decision_time,
            "provider_usage": provider_usage,
            "strategy_usage": strategy_usage,
            "provider_performance": {k.value: v for k, v in self.provider_performance.items()}
        }
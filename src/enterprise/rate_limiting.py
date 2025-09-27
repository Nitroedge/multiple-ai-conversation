"""
API Rate Limiting System
Sophisticated throttling and quota management
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """Rate limit types"""
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    BURST = "burst"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


class RateLimitScope(str, Enum):
    """Rate limit scope"""
    GLOBAL = "global"
    TENANT = "tenant"
    USER = "user"
    API_KEY = "api_key"
    IP_ADDRESS = "ip_address"
    ENDPOINT = "endpoint"


class RateLimitAction(str, Enum):
    """Action to take when rate limit is exceeded"""
    BLOCK = "block"
    THROTTLE = "throttle"
    QUEUE = "queue"
    DOWNGRADE = "downgrade"
    WARN = "warn"


class RateLimitRule(BaseModel):
    """Rate limiting rule configuration"""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str

    # Scope and targeting
    scope: RateLimitScope
    scope_value: Optional[str] = None  # Specific tenant_id, user_id, etc.
    endpoint_pattern: Optional[str] = None  # Regex pattern for endpoints

    # Rate limit configuration
    limit_type: RateLimitType
    requests_per_window: int
    window_size_seconds: int
    burst_allowance: int = 0  # Additional requests allowed in burst

    # Actions
    action: RateLimitAction = RateLimitAction.BLOCK
    queue_timeout_seconds: int = 30
    throttle_delay_seconds: float = 1.0

    # Metadata
    priority: int = 100  # Lower numbers = higher priority
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"

    @property
    def key_pattern(self) -> str:
        """Generate key pattern for this rule"""
        return f"rate_limit:{self.scope.value}:{self.limit_type.value}"


class RateLimitViolation(BaseModel):
    """Rate limit violation record"""
    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    rule_id: str
    scope: RateLimitScope
    scope_value: str
    endpoint: str

    # Request details
    requests_made: int
    limit_allowed: int
    window_start: datetime
    window_end: datetime

    # Client information
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None

    # Action taken
    action_taken: RateLimitAction
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"


class RateLimitStatus(BaseModel):
    """Current rate limit status for a request"""
    allowed: bool
    requests_made: int
    limit: int
    remaining: int
    reset_time: datetime
    retry_after_seconds: Optional[float] = None
    action: Optional[RateLimitAction] = None

    class Config:
        extra = "forbid"


class TokenBucket:
    """Token bucket algorithm implementation"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        now = time.time()

        # Add tokens based on time elapsed
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_tokens(self) -> float:
        """Get current token count"""
        now = time.time()
        elapsed = now - self.last_update
        return min(self.capacity, self.tokens + (elapsed * self.refill_rate))


class SlidingWindowCounter:
    """Sliding window rate limiting implementation"""

    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # seconds
        self.max_requests = max_requests
        self.requests: deque = deque()

    def is_allowed(self) -> Tuple[bool, int]:
        """Check if request is allowed and return current count"""
        now = time.time()

        # Remove old requests outside the window
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()

        current_count = len(self.requests)

        if current_count < self.max_requests:
            self.requests.append(now)
            return True, current_count + 1

        return False, current_count


class RateLimitStorage:
    """Storage backend for rate limiting data"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_storage: Dict[str, Any] = {}  # Fallback to local storage
        self.token_buckets: Dict[str, TokenBucket] = {}
        self.sliding_windows: Dict[str, SlidingWindowCounter] = {}

    async def get_count(self, key: str) -> int:
        """Get current count for key"""
        if self.redis_client:
            try:
                count = await self.redis_client.get(key)
                return int(count) if count else 0
            except Exception as e:
                logger.error(f"Redis error getting count: {e}")

        return self.local_storage.get(key, 0)

    async def increment(self, key: str, window_seconds: int) -> int:
        """Increment count with expiration"""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                results = await pipe.execute()
                return results[0]
            except Exception as e:
                logger.error(f"Redis error incrementing: {e}")

        # Fallback to local storage
        self.local_storage[key] = self.local_storage.get(key, 0) + 1
        return self.local_storage[key]

    async def set_count(self, key: str, count: int, window_seconds: int) -> None:
        """Set count with expiration"""
        if self.redis_client:
            try:
                await self.redis_client.setex(key, window_seconds, count)
                return
            except Exception as e:
                logger.error(f"Redis error setting count: {e}")

        self.local_storage[key] = count

    def get_token_bucket(self, key: str, capacity: int, refill_rate: float) -> TokenBucket:
        """Get or create token bucket"""
        if key not in self.token_buckets:
            self.token_buckets[key] = TokenBucket(capacity, refill_rate)
        return self.token_buckets[key]

    def get_sliding_window(self, key: str, window_size: int, max_requests: int) -> SlidingWindowCounter:
        """Get or create sliding window counter"""
        if key not in self.sliding_windows:
            self.sliding_windows[key] = SlidingWindowCounter(window_size, max_requests)
        return self.sliding_windows[key]


class RateLimiter:
    """Main rate limiting system"""

    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.rules: Dict[str, RateLimitRule] = {}
        self.violations: List[RateLimitViolation] = []
        self.request_queue: Dict[str, List[asyncio.Future]] = defaultdict(list)

        # Default rules
        self._create_default_rules()

    def _create_default_rules(self) -> None:
        """Create default rate limiting rules"""
        # Global rate limits
        self.add_rule(RateLimitRule(
            name="Global API Rate Limit",
            description="Global rate limit for all API requests",
            scope=RateLimitScope.GLOBAL,
            limit_type=RateLimitType.PER_MINUTE,
            requests_per_window=10000,
            window_size_seconds=60,
            priority=1000
        ))

        # Per-tenant limits
        self.add_rule(RateLimitRule(
            name="Tenant API Rate Limit",
            description="Per-tenant API rate limiting",
            scope=RateLimitScope.TENANT,
            limit_type=RateLimitType.PER_MINUTE,
            requests_per_window=1000,
            window_size_seconds=60,
            priority=100
        ))

        # Per-user limits
        self.add_rule(RateLimitRule(
            name="User API Rate Limit",
            description="Per-user API rate limiting",
            scope=RateLimitScope.USER,
            limit_type=RateLimitType.PER_MINUTE,
            requests_per_window=100,
            window_size_seconds=60,
            priority=50
        ))

        # IP-based limits
        self.add_rule(RateLimitRule(
            name="IP Rate Limit",
            description="Per-IP address rate limiting",
            scope=RateLimitScope.IP_ADDRESS,
            limit_type=RateLimitType.PER_MINUTE,
            requests_per_window=200,
            window_size_seconds=60,
            priority=75
        ))

        # Burst protection
        self.add_rule(RateLimitRule(
            name="Burst Protection",
            description="Protect against burst attacks",
            scope=RateLimitScope.IP_ADDRESS,
            limit_type=RateLimitType.BURST,
            requests_per_window=10,
            window_size_seconds=1,
            priority=10,
            action=RateLimitAction.THROTTLE,
            throttle_delay_seconds=2.0
        ))

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limiting rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rate limit rule: {rule.name}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rate limiting rule"""
        if rule_id in self.rules:
            rule = self.rules.pop(rule_id)
            logger.info(f"Removed rate limit rule: {rule.name}")
            return True
        return False

    def get_applicable_rules(self,
                           tenant_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           api_key_id: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           endpoint: Optional[str] = None) -> List[RateLimitRule]:
        """Get rules that apply to this request"""
        applicable_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check if rule applies to this request
            if rule.scope == RateLimitScope.GLOBAL:
                applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.TENANT and tenant_id:
                if not rule.scope_value or rule.scope_value == tenant_id:
                    applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.USER and user_id:
                if not rule.scope_value or rule.scope_value == user_id:
                    applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.API_KEY and api_key_id:
                if not rule.scope_value or rule.scope_value == api_key_id:
                    applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.IP_ADDRESS and ip_address:
                if not rule.scope_value or rule.scope_value == ip_address:
                    applicable_rules.append(rule)
            elif rule.scope == RateLimitScope.ENDPOINT and endpoint:
                if rule.endpoint_pattern:
                    import re
                    if re.match(rule.endpoint_pattern, endpoint):
                        applicable_rules.append(rule)

        # Sort by priority (lower number = higher priority)
        applicable_rules.sort(key=lambda r: r.priority)
        return applicable_rules

    async def check_rate_limit(self,
                             tenant_id: Optional[str] = None,
                             user_id: Optional[str] = None,
                             api_key_id: Optional[str] = None,
                             ip_address: Optional[str] = None,
                             endpoint: Optional[str] = None,
                             user_agent: Optional[str] = None) -> RateLimitStatus:
        """Check if request is within rate limits"""

        applicable_rules = self.get_applicable_rules(
            tenant_id, user_id, api_key_id, ip_address, endpoint
        )

        # Check each rule (highest priority first)
        for rule in applicable_rules:
            status = await self._check_rule(rule, tenant_id, user_id, api_key_id, ip_address, endpoint)

            if not status.allowed:
                # Record violation
                await self._record_violation(rule, status, tenant_id, user_id, ip_address, endpoint, user_agent)
                return status

        # If we get here, all rules passed
        return RateLimitStatus(
            allowed=True,
            requests_made=0,
            limit=float('inf'),
            remaining=float('inf'),
            reset_time=datetime.utcnow() + timedelta(hours=1)
        )

    async def _check_rule(self,
                        rule: RateLimitRule,
                        tenant_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        api_key_id: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        endpoint: Optional[str] = None) -> RateLimitStatus:
        """Check a specific rule"""

        # Generate key for this rule and request
        key_parts = [rule.key_pattern]

        if rule.scope == RateLimitScope.GLOBAL:
            key_parts.append("global")
        elif rule.scope == RateLimitScope.TENANT:
            key_parts.append(tenant_id or "unknown")
        elif rule.scope == RateLimitScope.USER:
            key_parts.append(user_id or "unknown")
        elif rule.scope == RateLimitScope.API_KEY:
            key_parts.append(api_key_id or "unknown")
        elif rule.scope == RateLimitScope.IP_ADDRESS:
            key_parts.append(ip_address or "unknown")
        elif rule.scope == RateLimitScope.ENDPOINT:
            key_parts.append(endpoint or "unknown")

        key = ":".join(key_parts)

        # Check based on limit type
        if rule.limit_type == RateLimitType.TOKEN_BUCKET:
            return await self._check_token_bucket(rule, key)
        elif rule.limit_type == RateLimitType.SLIDING_WINDOW:
            return await self._check_sliding_window(rule, key)
        else:
            return await self._check_fixed_window(rule, key)

    async def _check_token_bucket(self, rule: RateLimitRule, key: str) -> RateLimitStatus:
        """Check token bucket rate limit"""
        refill_rate = rule.requests_per_window / rule.window_size_seconds
        bucket = self.storage.get_token_bucket(key, rule.requests_per_window, refill_rate)

        allowed = bucket.consume(1)
        current_tokens = bucket.get_tokens()

        return RateLimitStatus(
            allowed=allowed,
            requests_made=rule.requests_per_window - int(current_tokens),
            limit=rule.requests_per_window,
            remaining=int(current_tokens),
            reset_time=datetime.utcnow() + timedelta(seconds=rule.window_size_seconds),
            action=rule.action if not allowed else None,
            retry_after_seconds=1.0 / (refill_rate) if not allowed else None
        )

    async def _check_sliding_window(self, rule: RateLimitRule, key: str) -> RateLimitStatus:
        """Check sliding window rate limit"""
        window = self.storage.get_sliding_window(key, rule.window_size_seconds, rule.requests_per_window)
        allowed, current_count = window.is_allowed()

        return RateLimitStatus(
            allowed=allowed,
            requests_made=current_count - (1 if allowed else 0),
            limit=rule.requests_per_window,
            remaining=max(0, rule.requests_per_window - current_count),
            reset_time=datetime.utcnow() + timedelta(seconds=rule.window_size_seconds),
            action=rule.action if not allowed else None,
            retry_after_seconds=rule.throttle_delay_seconds if not allowed else None
        )

    async def _check_fixed_window(self, rule: RateLimitRule, key: str) -> RateLimitStatus:
        """Check fixed window rate limit"""
        current_count = await self.storage.get_count(key)

        if current_count < rule.requests_per_window:
            # Allow request and increment
            new_count = await self.storage.increment(key, rule.window_size_seconds)

            return RateLimitStatus(
                allowed=True,
                requests_made=new_count,
                limit=rule.requests_per_window,
                remaining=max(0, rule.requests_per_window - new_count),
                reset_time=datetime.utcnow() + timedelta(seconds=rule.window_size_seconds)
            )
        else:
            # Rate limit exceeded
            return RateLimitStatus(
                allowed=False,
                requests_made=current_count,
                limit=rule.requests_per_window,
                remaining=0,
                reset_time=datetime.utcnow() + timedelta(seconds=rule.window_size_seconds),
                action=rule.action,
                retry_after_seconds=rule.throttle_delay_seconds
            )

    async def _record_violation(self,
                              rule: RateLimitRule,
                              status: RateLimitStatus,
                              tenant_id: Optional[str],
                              user_id: Optional[str],
                              ip_address: Optional[str],
                              endpoint: Optional[str],
                              user_agent: Optional[str]) -> None:
        """Record a rate limit violation"""

        violation = RateLimitViolation(
            rule_id=rule.rule_id,
            scope=rule.scope,
            scope_value=tenant_id or user_id or ip_address or "unknown",
            endpoint=endpoint or "unknown",
            requests_made=status.requests_made,
            limit_allowed=status.limit,
            window_start=status.reset_time - timedelta(seconds=rule.window_size_seconds),
            window_end=status.reset_time,
            ip_address=ip_address,
            user_agent=user_agent,
            user_id=user_id,
            tenant_id=tenant_id,
            action_taken=status.action or RateLimitAction.BLOCK
        )

        self.violations.append(violation)

        # Keep only recent violations (last 1000)
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]

        logger.warning(f"Rate limit violation: {rule.name} for {violation.scope_value}")

    async def handle_rate_limited_request(self, status: RateLimitStatus, request_id: str) -> bool:
        """Handle a rate-limited request based on the action"""
        if status.action == RateLimitAction.BLOCK:
            return False

        elif status.action == RateLimitAction.THROTTLE:
            if status.retry_after_seconds:
                await asyncio.sleep(status.retry_after_seconds)
            return True

        elif status.action == RateLimitAction.QUEUE:
            # Add to queue and wait
            future = asyncio.Future()
            queue_key = f"queue:{status.action}"
            self.request_queue[queue_key].append(future)

            try:
                await asyncio.wait_for(future, timeout=30)  # 30 second timeout
                return True
            except asyncio.TimeoutError:
                return False

        elif status.action == RateLimitAction.WARN:
            logger.warning(f"Rate limit warning for request {request_id}")
            return True

        return False

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        now = datetime.utcnow()
        recent_violations = [v for v in self.violations if (now - v.timestamp).total_seconds() < 3600]

        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "violations_last_hour": len(recent_violations),
            "total_violations": len(self.violations),
            "violations_by_scope": {
                scope.value: len([v for v in recent_violations if v.scope == scope])
                for scope in RateLimitScope
            },
            "violations_by_action": {
                action.value: len([v for v in recent_violations if v.action_taken == action])
                for action in RateLimitAction
            }
        }

    def get_violations(self,
                      scope: Optional[RateLimitScope] = None,
                      scope_value: Optional[str] = None,
                      hours: int = 24,
                      limit: int = 100) -> List[RateLimitViolation]:
        """Get rate limit violations with filters"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        violations = [v for v in self.violations if v.timestamp >= cutoff]

        if scope:
            violations = [v for v in violations if v.scope == scope]

        if scope_value:
            violations = [v for v in violations if v.scope_value == scope_value]

        # Sort by timestamp (newest first)
        violations.sort(key=lambda v: v.timestamp, reverse=True)

        return violations[:limit]


class RateLimitManager:
    """Main rate limiting management system"""

    def __init__(self, redis_url: Optional[str] = None):
        # Initialize Redis connection if URL provided
        redis_client = None
        if redis_url:
            try:
                redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")

        self.storage = RateLimitStorage(redis_client)
        self.rate_limiter = RateLimiter(self.storage)

        # Background cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self) -> None:
        """Start the rate limiting system"""
        if self.running:
            return

        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Rate limiting system started")

    async def stop(self) -> None:
        """Stop the rate limiting system"""
        self.running = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Close Redis connection if exists
        if self.storage.redis_client:
            await self.storage.redis_client.close()

        logger.info("Rate limiting system stopped")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self.running:
            try:
                # Clean up old violations
                cutoff = datetime.utcnow() - timedelta(days=7)  # Keep 7 days
                original_count = len(self.rate_limiter.violations)
                self.rate_limiter.violations = [
                    v for v in self.rate_limiter.violations if v.timestamp >= cutoff
                ]

                if len(self.rate_limiter.violations) < original_count:
                    logger.info(f"Cleaned up {original_count - len(self.rate_limiter.violations)} old violations")

                await asyncio.sleep(3600)  # Run every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)

    async def check_rate_limit(self, **kwargs) -> RateLimitStatus:
        """Check rate limit for request"""
        return await self.rate_limiter.check_rate_limit(**kwargs)

    async def handle_rate_limited_request(self, status: RateLimitStatus, request_id: str) -> bool:
        """Handle rate limited request"""
        return await self.rate_limiter.handle_rate_limited_request(status, request_id)

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add rate limiting rule"""
        self.rate_limiter.add_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove rate limiting rule"""
        return self.rate_limiter.remove_rule(rule_id)

    def get_rules(self) -> List[RateLimitRule]:
        """Get all rate limiting rules"""
        return list(self.rate_limiter.rules.values())

    def get_violations(self, **kwargs) -> List[RateLimitViolation]:
        """Get rate limit violations"""
        return self.rate_limiter.get_violations(**kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        return self.rate_limiter.get_rate_limit_stats()
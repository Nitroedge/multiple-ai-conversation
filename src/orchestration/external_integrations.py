"""
External API Integration Framework
Comprehensive system for webhooks, third-party services, and API orchestration
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from pydantic import BaseModel, Field, validator
from uuid import uuid4
import httpx
from urllib.parse import urljoin
import hmac
import hashlib
import base64

logger = logging.getLogger(__name__)


class IntegrationStatus(str, Enum):
    """Status of external integrations"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    PENDING = "pending"
    DISABLED = "disabled"
    RATE_LIMITED = "rate_limited"


class HTTPMethod(str, Enum):
    """HTTP methods for API calls"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthenticationType(str, Enum):
    """Authentication types for external APIs"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    CUSTOM_HEADER = "custom_header"
    WEBHOOK_SIGNATURE = "webhook_signature"


class TriggerEvent(str, Enum):
    """Events that can trigger external integrations"""
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    AGENT_RESPONSE = "agent_response"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    ROLE_ASSIGNED = "role_assigned"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    ERROR_OCCURRED = "error_occurred"
    CUSTOM_EVENT = "custom_event"


class APIEndpoint(BaseModel):
    """Configuration for an external API endpoint"""
    endpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""

    # Connection details
    base_url: str
    path: str = ""
    method: HTTPMethod = HTTPMethod.POST
    timeout_seconds: int = 30

    # Authentication
    auth_type: AuthenticationType = AuthenticationType.NONE
    auth_config: Dict[str, Any] = Field(default_factory=dict)

    # Headers and parameters
    default_headers: Dict[str, str] = Field(default_factory=dict)
    default_params: Dict[str, str] = Field(default_factory=dict)

    # Request/Response configuration
    request_format: str = "json"  # json, form, xml
    response_format: str = "json"
    content_type: str = "application/json"

    # Rate limiting
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    # Health checking
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 300

    # Metadata
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Base URL must start with http:// or https://')
        return v

    def get_full_url(self) -> str:
        """Get the full URL for this endpoint"""
        return urljoin(self.base_url.rstrip('/') + '/', self.path.lstrip('/'))


class WebhookConfig(BaseModel):
    """Configuration for webhook endpoints"""
    webhook_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""

    # Webhook details
    url: str
    secret: Optional[str] = None
    signature_header: str = "X-Signature"
    signature_algorithm: str = "sha256"

    # Trigger configuration
    trigger_events: List[TriggerEvent] = Field(default_factory=list)
    event_filters: Dict[str, Any] = Field(default_factory=dict)

    # Delivery configuration
    delivery_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 2.0

    # Security
    verify_ssl: bool = True
    allowed_ip_ranges: List[str] = Field(default_factory=list)

    # Status
    status: IntegrationStatus = IntegrationStatus.ACTIVE
    last_delivery_at: Optional[datetime] = None
    delivery_success_count: int = 0
    delivery_failure_count: int = 0

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Webhook URL must start with http:// or https://')
        return v


class IntegrationRule(BaseModel):
    """Rule for triggering external integrations"""
    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""

    # Trigger conditions
    trigger_event: TriggerEvent
    conditions: List[Dict[str, Any]] = Field(default_factory=list)

    # Actions
    webhook_ids: List[str] = Field(default_factory=list)
    api_endpoint_ids: List[str] = Field(default_factory=list)
    custom_actions: List[Dict[str, Any]] = Field(default_factory=list)

    # Data transformation
    data_mapping: Dict[str, str] = Field(default_factory=dict)
    data_filters: List[str] = Field(default_factory=list)

    # Execution settings
    is_active: bool = True
    priority: int = Field(default=5, ge=1, le=10)
    async_execution: bool = True

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    execution_count: int = 0
    last_executed_at: Optional[datetime] = None


class APICall(BaseModel):
    """Record of an API call"""
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    endpoint_id: str

    # Request details
    method: str
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    body: Optional[str] = None

    # Response details
    status_code: Optional[int] = None
    response_headers: Dict[str, str] = Field(default_factory=dict)
    response_body: Optional[str] = None
    response_time_ms: Optional[float] = None

    # Execution details
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None

    # Context
    trigger_event: Optional[str] = None
    workflow_id: Optional[str] = None
    session_id: Optional[str] = None


class WebhookDelivery(BaseModel):
    """Record of a webhook delivery"""
    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    webhook_id: str

    # Event details
    event_type: TriggerEvent
    event_data: Dict[str, Any] = Field(default_factory=dict)

    # Delivery details
    attempt_number: int = 1
    delivered_at: datetime = Field(default_factory=datetime.utcnow)
    response_status_code: Optional[int] = None
    response_body: Optional[str] = None
    response_time_ms: Optional[float] = None

    # Status
    success: bool = False
    error_message: Optional[str] = None
    will_retry: bool = False
    next_retry_at: Optional[datetime] = None


class ThirdPartyIntegration(BaseModel):
    """Configuration for third-party service integration"""
    integration_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    service_type: str  # slack, discord, teams, jira, github, etc.
    description: str = ""

    # Service-specific configuration
    service_config: Dict[str, Any] = Field(default_factory=dict)

    # API endpoints for this service
    endpoints: List[str] = Field(default_factory=list)

    # Webhook configurations
    webhooks: List[str] = Field(default_factory=list)

    # Features enabled
    enabled_features: List[str] = Field(default_factory=list)

    # Status
    status: IntegrationStatus = IntegrationStatus.ACTIVE
    last_sync_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExternalAPIManager:
    """Manager for external API integrations"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.integrations: Dict[str, ThirdPartyIntegration] = {}

        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}

        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

        # Call history
        self.call_history: List[APICall] = []
        self.max_history_size = 1000

    async def register_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Register a new API endpoint"""
        try:
            # Validate endpoint
            if await self._validate_endpoint(endpoint):
                self.endpoints[endpoint.endpoint_id] = endpoint
                self.logger.info(f"Registered API endpoint: {endpoint.name}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error registering endpoint: {e}")
            return False

    async def call_endpoint(
        self,
        endpoint_id: str,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make an API call to a registered endpoint"""
        try:
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint:
                raise Exception(f"Endpoint not found: {endpoint_id}")

            # Check rate limits
            if not await self._check_rate_limits(endpoint):
                raise Exception(f"Rate limit exceeded for endpoint: {endpoint.name}")

            # Prepare request
            request_data = await self._prepare_request(endpoint, data, headers, params)

            # Execute call with retries
            call_record = await self._execute_api_call(endpoint, request_data, context)

            # Update rate limits
            await self._update_rate_limits(endpoint)

            # Store call history
            self._store_call_record(call_record)

            return {
                "success": call_record.success,
                "status_code": call_record.status_code,
                "data": json.loads(call_record.response_body) if call_record.response_body else None,
                "response_time_ms": call_record.response_time_ms,
                "error": call_record.error_message
            }

        except Exception as e:
            self.logger.error(f"Error calling endpoint: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None,
                "data": None,
                "response_time_ms": None
            }

    async def _validate_endpoint(self, endpoint: APIEndpoint) -> bool:
        """Validate endpoint configuration"""
        try:
            # Test connection if health check is enabled
            if endpoint.health_check_enabled:
                test_url = endpoint.get_full_url()

                # Simple HEAD request to check connectivity
                async with self.client as client:
                    response = await client.head(
                        test_url,
                        timeout=5.0,
                        follow_redirects=True
                    )
                    # Accept any response that indicates the endpoint exists
                    if response.status_code < 500:
                        return True

            # If health check is disabled, assume valid
            return True

        except Exception as e:
            self.logger.warning(f"Endpoint validation failed: {e}")
            return False

    async def _check_rate_limits(self, endpoint: APIEndpoint) -> bool:
        """Check if API call is within rate limits"""
        endpoint_id = endpoint.endpoint_id
        current_time = time.time()

        if endpoint_id not in self.rate_limits:
            self.rate_limits[endpoint_id] = {
                "minute_calls": [],
                "hour_calls": []
            }

        limits = self.rate_limits[endpoint_id]

        # Check per-minute limit
        if endpoint.rate_limit_per_minute:
            minute_ago = current_time - 60
            limits["minute_calls"] = [t for t in limits["minute_calls"] if t > minute_ago]

            if len(limits["minute_calls"]) >= endpoint.rate_limit_per_minute:
                return False

        # Check per-hour limit
        if endpoint.rate_limit_per_hour:
            hour_ago = current_time - 3600
            limits["hour_calls"] = [t for t in limits["hour_calls"] if t > hour_ago]

            if len(limits["hour_calls"]) >= endpoint.rate_limit_per_hour:
                return False

        return True

    async def _update_rate_limits(self, endpoint: APIEndpoint):
        """Update rate limit tracking after successful call"""
        endpoint_id = endpoint.endpoint_id
        current_time = time.time()

        if endpoint_id in self.rate_limits:
            limits = self.rate_limits[endpoint_id]
            limits["minute_calls"].append(current_time)
            limits["hour_calls"].append(current_time)

    async def _prepare_request(
        self,
        endpoint: APIEndpoint,
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Prepare request data for API call"""
        request_data = {
            "method": endpoint.method,
            "url": endpoint.get_full_url(),
            "headers": {**endpoint.default_headers},
            "params": {**endpoint.default_params},
            "data": data or {}
        }

        # Add custom headers and params
        if headers:
            request_data["headers"].update(headers)
        if params:
            request_data["params"].update(params)

        # Add authentication
        await self._add_authentication(endpoint, request_data)

        # Set content type
        if endpoint.content_type and "Content-Type" not in request_data["headers"]:
            request_data["headers"]["Content-Type"] = endpoint.content_type

        return request_data

    async def _add_authentication(self, endpoint: APIEndpoint, request_data: Dict[str, Any]):
        """Add authentication to request"""
        auth_config = endpoint.auth_config

        if endpoint.auth_type == AuthenticationType.API_KEY:
            api_key = auth_config.get("api_key")
            header_name = auth_config.get("header_name", "X-API-Key")
            if api_key:
                request_data["headers"][header_name] = api_key

        elif endpoint.auth_type == AuthenticationType.BEARER_TOKEN:
            token = auth_config.get("token")
            if token:
                request_data["headers"]["Authorization"] = f"Bearer {token}"

        elif endpoint.auth_type == AuthenticationType.BASIC_AUTH:
            username = auth_config.get("username")
            password = auth_config.get("password")
            if username and password:
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                request_data["headers"]["Authorization"] = f"Basic {credentials}"

        elif endpoint.auth_type == AuthenticationType.CUSTOM_HEADER:
            header_name = auth_config.get("header_name")
            header_value = auth_config.get("header_value")
            if header_name and header_value:
                request_data["headers"][header_name] = header_value

    async def _execute_api_call(
        self,
        endpoint: APIEndpoint,
        request_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> APICall:
        """Execute API call with retry logic"""
        call_record = APICall(
            endpoint_id=endpoint.endpoint_id,
            method=request_data["method"],
            url=request_data["url"],
            headers=request_data["headers"],
            params=request_data["params"],
            trigger_event=context.get("trigger_event") if context else None,
            workflow_id=context.get("workflow_id") if context else None,
            session_id=context.get("session_id") if context else None
        )

        for attempt in range(endpoint.max_retries + 1):
            try:
                start_time = time.time()

                # Prepare request body
                request_kwargs = {
                    "method": request_data["method"],
                    "url": request_data["url"],
                    "headers": request_data["headers"],
                    "params": request_data["params"],
                    "timeout": endpoint.timeout_seconds
                }

                # Add data based on method and format
                if request_data["data"] and request_data["method"] in ["POST", "PUT", "PATCH"]:
                    if endpoint.request_format == "json":
                        request_kwargs["json"] = request_data["data"]
                        call_record.body = json.dumps(request_data["data"])
                    elif endpoint.request_format == "form":
                        request_kwargs["data"] = request_data["data"]
                        call_record.body = str(request_data["data"])

                # Make the request
                async with self.client as client:
                    response = await client.request(**request_kwargs)

                # Record response
                response_time = (time.time() - start_time) * 1000
                call_record.status_code = response.status_code
                call_record.response_headers = dict(response.headers)
                call_record.response_body = response.text
                call_record.response_time_ms = response_time
                call_record.completed_at = datetime.utcnow()

                # Check if successful
                if 200 <= response.status_code < 300:
                    call_record.success = True
                    break
                else:
                    call_record.error_message = f"HTTP {response.status_code}: {response.text}"

            except Exception as e:
                call_record.error_message = str(e)
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")

            # Calculate retry delay
            if attempt < endpoint.max_retries:
                if endpoint.exponential_backoff:
                    delay = endpoint.retry_delay_seconds * (2 ** attempt)
                else:
                    delay = endpoint.retry_delay_seconds

                await asyncio.sleep(delay)

        return call_record

    def _store_call_record(self, call_record: APICall):
        """Store API call record in history"""
        self.call_history.append(call_record)

        # Maintain history size limit
        if len(self.call_history) > self.max_history_size:
            self.call_history = self.call_history[-self.max_history_size:]

    async def register_integration(self, integration: ThirdPartyIntegration) -> bool:
        """Register a third-party service integration"""
        try:
            self.integrations[integration.integration_id] = integration
            self.logger.info(f"Registered integration: {integration.name} ({integration.service_type})")
            return True

        except Exception as e:
            self.logger.error(f"Error registering integration: {e}")
            return False

    async def get_endpoint_health(self, endpoint_id: str) -> Dict[str, Any]:
        """Get health status of an endpoint"""
        try:
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint:
                return {"status": "not_found"}

            # Simple health check
            try:
                start_time = time.time()

                async with self.client as client:
                    response = await client.head(
                        endpoint.get_full_url(),
                        timeout=5.0
                    )

                response_time = (time.time() - start_time) * 1000

                return {
                    "status": "healthy" if response.status_code < 500 else "unhealthy",
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "last_checked": datetime.utcnow().isoformat()
                }

            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_checked": datetime.utcnow().isoformat()
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_integration_metrics(self) -> Dict[str, Any]:
        """Get metrics about API integrations"""
        try:
            total_calls = len(self.call_history)
            successful_calls = len([call for call in self.call_history if call.success])

            if total_calls == 0:
                return {
                    "total_endpoints": len(self.endpoints),
                    "total_integrations": len(self.integrations),
                    "total_calls": 0,
                    "success_rate": 0.0
                }

            # Calculate metrics
            success_rate = successful_calls / total_calls
            avg_response_time = sum(
                call.response_time_ms for call in self.call_history
                if call.response_time_ms
            ) / len([call for call in self.call_history if call.response_time_ms])

            # Endpoint usage
            endpoint_usage = {}
            for call in self.call_history:
                endpoint_id = call.endpoint_id
                endpoint_usage[endpoint_id] = endpoint_usage.get(endpoint_id, 0) + 1

            return {
                "total_endpoints": len(self.endpoints),
                "total_integrations": len(self.integrations),
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": total_calls - successful_calls,
                "success_rate": success_rate,
                "avg_response_time_ms": avg_response_time,
                "endpoint_usage": endpoint_usage
            }

        except Exception as e:
            self.logger.error(f"Error calculating integration metrics: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.client.aclose()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class WebhookManager:
    """Manager for webhook delivery and processing"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.integration_rules: Dict[str, IntegrationRule] = {}

        # Delivery tracking
        self.delivery_history: List[WebhookDelivery] = []
        self.max_history_size = 1000

        # HTTP client for webhook delivery
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
        )

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.processing_task: Optional[asyncio.Task] = None

    async def register_webhook(self, webhook: WebhookConfig) -> bool:
        """Register a new webhook"""
        try:
            self.webhooks[webhook.webhook_id] = webhook
            self.logger.info(f"Registered webhook: {webhook.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error registering webhook: {e}")
            return False

    async def register_integration_rule(self, rule: IntegrationRule) -> bool:
        """Register an integration rule"""
        try:
            self.integration_rules[rule.rule_id] = rule
            self.logger.info(f"Registered integration rule: {rule.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error registering integration rule: {e}")
            return False

    async def trigger_event(
        self,
        event_type: TriggerEvent,
        event_data: Dict[str, Any],
        context: Dict[str, Any] = None
    ):
        """Trigger an event that may activate webhooks"""
        try:
            event = {
                "event_type": event_type,
                "event_data": event_data,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add to queue for async processing
            try:
                self.event_queue.put_nowait(event)
            except asyncio.QueueFull:
                self.logger.warning("Event queue is full, dropping event")

        except Exception as e:
            self.logger.error(f"Error triggering event: {e}")

    async def start_processing(self):
        """Start background event processing"""
        if self.processing_task and not self.processing_task.done():
            return

        self.processing_task = asyncio.create_task(self._process_events())
        self.logger.info("Started webhook event processing")

    async def stop_processing(self):
        """Stop background event processing"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None

        self.logger.info("Stopped webhook event processing")

    async def _process_events(self):
        """Background task to process webhook events"""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()

                # Find matching integration rules
                matching_rules = await self._find_matching_rules(event)

                # Execute matching rules
                for rule in matching_rules:
                    await self._execute_integration_rule(rule, event)

                # Mark task as done
                self.event_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing webhook event: {e}")

    async def _find_matching_rules(self, event: Dict[str, Any]) -> List[IntegrationRule]:
        """Find integration rules that match the event"""
        matching_rules = []

        for rule in self.integration_rules.values():
            if not rule.is_active:
                continue

            # Check event type match
            if rule.trigger_event != event["event_type"]:
                continue

            # Check conditions
            if await self._evaluate_rule_conditions(rule, event):
                matching_rules.append(rule)

        # Sort by priority
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        return matching_rules

    async def _evaluate_rule_conditions(
        self,
        rule: IntegrationRule,
        event: Dict[str, Any]
    ) -> bool:
        """Evaluate if rule conditions are met"""
        if not rule.conditions:
            return True

        for condition in rule.conditions:
            condition_type = condition.get("type")
            field_path = condition.get("field_path")
            expected_value = condition.get("expected_value")

            # Get value from event data
            event_value = self._get_nested_value(event["event_data"], field_path)

            # Evaluate condition
            if condition_type == "equals":
                if event_value != expected_value:
                    return False
            elif condition_type == "contains":
                if expected_value not in str(event_value):
                    return False
            elif condition_type == "greater_than":
                if float(event_value) <= float(expected_value):
                    return False
            elif condition_type == "exists":
                if event_value is None:
                    return False

        return True

    async def _execute_integration_rule(
        self,
        rule: IntegrationRule,
        event: Dict[str, Any]
    ):
        """Execute an integration rule"""
        try:
            # Transform event data
            transformed_data = await self._transform_event_data(rule, event)

            # Execute webhook deliveries
            for webhook_id in rule.webhook_ids:
                webhook = self.webhooks.get(webhook_id)
                if webhook and webhook.status == IntegrationStatus.ACTIVE:
                    await self._deliver_webhook(webhook, event["event_type"], transformed_data)

            # Update rule execution tracking
            rule.execution_count += 1
            rule.last_executed_at = datetime.utcnow()

        except Exception as e:
            self.logger.error(f"Error executing integration rule {rule.name}: {e}")

    async def _transform_event_data(
        self,
        rule: IntegrationRule,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform event data according to rule configuration"""
        event_data = event["event_data"].copy()

        # Apply data mapping
        if rule.data_mapping:
            transformed_data = {}
            for target_field, source_path in rule.data_mapping.items():
                value = self._get_nested_value(event_data, source_path)
                transformed_data[target_field] = value
        else:
            transformed_data = event_data

        # Apply data filters
        if rule.data_filters:
            for filter_path in rule.data_filters:
                self._remove_nested_value(transformed_data, filter_path)

        # Add metadata
        transformed_data["_metadata"] = {
            "event_type": event["event_type"],
            "timestamp": event["timestamp"],
            "rule_id": rule.rule_id,
            "rule_name": rule.name
        }

        return transformed_data

    async def _deliver_webhook(
        self,
        webhook: WebhookConfig,
        event_type: TriggerEvent,
        data: Dict[str, Any]
    ):
        """Deliver webhook with retry logic"""
        delivery = WebhookDelivery(
            webhook_id=webhook.webhook_id,
            event_type=event_type,
            event_data=data
        )

        for attempt in range(webhook.max_retries + 1):
            try:
                delivery.attempt_number = attempt + 1
                start_time = time.time()

                # Prepare request
                headers = {"Content-Type": "application/json"}

                # Add webhook signature if configured
                if webhook.secret:
                    signature = self._generate_webhook_signature(
                        webhook.secret,
                        json.dumps(data),
                        webhook.signature_algorithm
                    )
                    headers[webhook.signature_header] = signature

                # Make request
                async with self.client as client:
                    response = await client.post(
                        webhook.url,
                        json=data,
                        headers=headers,
                        timeout=webhook.delivery_timeout_seconds,
                        verify=webhook.verify_ssl
                    )

                # Record response
                delivery.response_status_code = response.status_code
                delivery.response_body = response.text
                delivery.response_time_ms = (time.time() - start_time) * 1000

                # Check success
                if 200 <= response.status_code < 300:
                    delivery.success = True
                    webhook.delivery_success_count += 1
                    webhook.last_delivery_at = datetime.utcnow()
                    break
                else:
                    delivery.error_message = f"HTTP {response.status_code}: {response.text}"

            except Exception as e:
                delivery.error_message = str(e)

            # Schedule retry if needed
            if attempt < webhook.max_retries:
                delivery.will_retry = True
                retry_delay = webhook.retry_delay_seconds * (2 ** attempt)
                delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)
                await asyncio.sleep(retry_delay)

        # Record delivery failure if all attempts failed
        if not delivery.success:
            webhook.delivery_failure_count += 1

        # Store delivery record
        self._store_delivery_record(delivery)

    def _generate_webhook_signature(
        self,
        secret: str,
        payload: str,
        algorithm: str = "sha256"
    ) -> str:
        """Generate webhook signature for verification"""
        if algorithm == "sha256":
            signature = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            return f"sha256={signature}"
        else:
            # Support other algorithms as needed
            return ""

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation"""
        try:
            value = data
            for key in path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                elif isinstance(value, list) and key.isdigit():
                    value = value[int(key)]
                else:
                    return None
            return value
        except (KeyError, IndexError, ValueError):
            return None

    def _remove_nested_value(self, data: Dict[str, Any], path: str):
        """Remove nested value using dot notation"""
        try:
            keys = path.split('.')
            current = data
            for key in keys[:-1]:
                current = current[key]
            if keys[-1] in current:
                del current[keys[-1]]
        except (KeyError, TypeError):
            pass

    def _store_delivery_record(self, delivery: WebhookDelivery):
        """Store webhook delivery record"""
        self.delivery_history.append(delivery)

        # Maintain history size limit
        if len(self.delivery_history) > self.max_history_size:
            self.delivery_history = self.delivery_history[-self.max_history_size:]

    async def get_webhook_metrics(self) -> Dict[str, Any]:
        """Get webhook delivery metrics"""
        try:
            total_deliveries = len(self.delivery_history)
            successful_deliveries = len([d for d in self.delivery_history if d.success])

            if total_deliveries == 0:
                return {
                    "total_webhooks": len(self.webhooks),
                    "total_rules": len(self.integration_rules),
                    "total_deliveries": 0,
                    "success_rate": 0.0
                }

            success_rate = successful_deliveries / total_deliveries

            # Average response time
            response_times = [
                d.response_time_ms for d in self.delivery_history
                if d.response_time_ms
            ]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            # Event type distribution
            event_distribution = {}
            for delivery in self.delivery_history:
                event_type = delivery.event_type.value
                event_distribution[event_type] = event_distribution.get(event_type, 0) + 1

            return {
                "total_webhooks": len(self.webhooks),
                "total_rules": len(self.integration_rules),
                "total_deliveries": total_deliveries,
                "successful_deliveries": successful_deliveries,
                "failed_deliveries": total_deliveries - successful_deliveries,
                "success_rate": success_rate,
                "avg_response_time_ms": avg_response_time,
                "event_distribution": event_distribution
            }

        except Exception as e:
            self.logger.error(f"Error calculating webhook metrics: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.stop_processing()
            await self.client.aclose()
        except Exception as e:
            self.logger.error(f"Error during webhook manager cleanup: {e}")
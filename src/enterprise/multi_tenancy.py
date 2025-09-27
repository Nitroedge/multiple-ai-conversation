"""
Multi-Tenancy Support System
Enterprise-grade tenant isolation and management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Tenant status types"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    ARCHIVED = "archived"
    TRIAL = "trial"


class TenantPlan(str, Enum):
    """Tenant subscription plans"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class ResourceType(str, Enum):
    """Resource types for quotas"""
    API_CALLS = "api_calls"
    STORAGE_MB = "storage_mb"
    AGENTS = "agents"
    WORKFLOWS = "workflows"
    MEMORY_OPERATIONS = "memory_operations"
    VOICE_MINUTES = "voice_minutes"
    AI_TOKENS = "ai_tokens"
    CONCURRENT_SESSIONS = "concurrent_sessions"


class TenantQuota(BaseModel):
    """Resource quota configuration"""
    resource_type: ResourceType
    limit: int
    current_usage: int = 0
    reset_period: str = "monthly"  # "daily", "weekly", "monthly", "yearly"
    last_reset: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "forbid"

    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage"""
        if self.limit <= 0:
            return 0.0
        return min(100.0, (self.current_usage / self.limit) * 100)

    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded"""
        return self.current_usage >= self.limit

    def should_reset(self) -> bool:
        """Check if quota should be reset based on period"""
        now = datetime.utcnow()

        if self.reset_period == "daily":
            return (now - self.last_reset).days >= 1
        elif self.reset_period == "weekly":
            return (now - self.last_reset).days >= 7
        elif self.reset_period == "monthly":
            return (now - self.last_reset).days >= 30
        elif self.reset_period == "yearly":
            return (now - self.last_reset).days >= 365

        return False

    def reset_if_needed(self) -> bool:
        """Reset quota if needed, return True if reset occurred"""
        if self.should_reset():
            self.current_usage = 0
            self.last_reset = datetime.utcnow()
            return True
        return False


class TenantConfiguration(BaseModel):
    """Tenant-specific configuration"""
    # AI Model preferences
    preferred_ai_models: List[str] = Field(default_factory=list)
    model_fallback_enabled: bool = True
    max_tokens_per_request: int = 4000

    # Feature flags
    voice_processing_enabled: bool = True
    home_assistant_enabled: bool = False
    analytics_enabled: bool = True
    collaboration_enabled: bool = True

    # Security settings
    ip_whitelist: List[str] = Field(default_factory=list)
    require_2fa: bool = False
    session_timeout_minutes: int = 60

    # Performance settings
    max_concurrent_workflows: int = 10
    max_agents_per_workflow: int = 5
    memory_retention_days: int = 30

    # Notification settings
    webhook_url: Optional[str] = None
    alert_email: Optional[str] = None
    quota_warning_threshold: float = 0.8  # 80%

    class Config:
        extra = "forbid"


class Tenant(BaseModel):
    """Tenant entity"""
    tenant_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    subdomain: str  # tenant.platform.com
    status: TenantStatus = TenantStatus.TRIAL
    plan: TenantPlan = TenantPlan.FREE

    # Contact information
    admin_email: str
    admin_name: str
    company_name: Optional[str] = None

    # Subscription details
    created_at: datetime = Field(default_factory=datetime.utcnow)
    trial_expires_at: Optional[datetime] = None
    subscription_expires_at: Optional[datetime] = None

    # Resource quotas
    quotas: Dict[str, TenantQuota] = Field(default_factory=dict)

    # Configuration
    configuration: TenantConfiguration = Field(default_factory=TenantConfiguration)

    # Usage tracking
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    total_api_calls: int = 0
    total_storage_used: int = 0  # MB

    # Metadata
    tags: List[str] = Field(default_factory=list)
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"

    @validator('subdomain')
    def validate_subdomain(cls, v):
        """Validate subdomain format"""
        if not v.isalnum() and '-' not in v:
            raise ValueError('Subdomain must be alphanumeric with optional hyphens')
        if len(v) < 3 or len(v) > 63:
            raise ValueError('Subdomain must be 3-63 characters long')
        return v.lower()

    def is_active(self) -> bool:
        """Check if tenant is active"""
        if self.status != TenantStatus.ACTIVE:
            return False

        # Check subscription expiry
        if self.subscription_expires_at and datetime.utcnow() > self.subscription_expires_at:
            return False

        # Check trial expiry
        if self.status == TenantStatus.TRIAL and self.trial_expires_at:
            if datetime.utcnow() > self.trial_expires_at:
                return False

        return True

    def get_quota(self, resource_type: ResourceType) -> TenantQuota:
        """Get quota for resource type"""
        if resource_type.value not in self.quotas:
            # Create default quota based on plan
            self.quotas[resource_type.value] = self._create_default_quota(resource_type)

        quota = self.quotas[resource_type.value]
        quota.reset_if_needed()
        return quota

    def _create_default_quota(self, resource_type: ResourceType) -> TenantQuota:
        """Create default quota based on plan"""
        limits = {
            TenantPlan.FREE: {
                ResourceType.API_CALLS: 1000,
                ResourceType.STORAGE_MB: 100,
                ResourceType.AGENTS: 2,
                ResourceType.WORKFLOWS: 5,
                ResourceType.MEMORY_OPERATIONS: 5000,
                ResourceType.VOICE_MINUTES: 10,
                ResourceType.AI_TOKENS: 50000,
                ResourceType.CONCURRENT_SESSIONS: 1
            },
            TenantPlan.BASIC: {
                ResourceType.API_CALLS: 10000,
                ResourceType.STORAGE_MB: 1000,
                ResourceType.AGENTS: 5,
                ResourceType.WORKFLOWS: 20,
                ResourceType.MEMORY_OPERATIONS: 50000,
                ResourceType.VOICE_MINUTES: 100,
                ResourceType.AI_TOKENS: 500000,
                ResourceType.CONCURRENT_SESSIONS: 5
            },
            TenantPlan.PROFESSIONAL: {
                ResourceType.API_CALLS: 100000,
                ResourceType.STORAGE_MB: 10000,
                ResourceType.AGENTS: 20,
                ResourceType.WORKFLOWS: 100,
                ResourceType.MEMORY_OPERATIONS: 500000,
                ResourceType.VOICE_MINUTES: 1000,
                ResourceType.AI_TOKENS: 5000000,
                ResourceType.CONCURRENT_SESSIONS: 25
            },
            TenantPlan.ENTERPRISE: {
                ResourceType.API_CALLS: 1000000,
                ResourceType.STORAGE_MB: 100000,
                ResourceType.AGENTS: 100,
                ResourceType.WORKFLOWS: 1000,
                ResourceType.MEMORY_OPERATIONS: 5000000,
                ResourceType.VOICE_MINUTES: 10000,
                ResourceType.AI_TOKENS: 50000000,
                ResourceType.CONCURRENT_SESSIONS: 100
            }
        }

        limit = limits.get(self.plan, limits[TenantPlan.FREE]).get(resource_type, 0)
        return TenantQuota(resource_type=resource_type, limit=limit)

    def check_quota(self, resource_type: ResourceType, amount: int = 1) -> bool:
        """Check if resource usage is within quota"""
        quota = self.get_quota(resource_type)
        return quota.current_usage + amount <= quota.limit

    def consume_quota(self, resource_type: ResourceType, amount: int = 1) -> bool:
        """Consume quota for resource, return True if successful"""
        quota = self.get_quota(resource_type)

        if quota.current_usage + amount <= quota.limit:
            quota.current_usage += amount
            return True
        return False

    def get_quota_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all quotas"""
        status = {}
        for resource_type in ResourceType:
            quota = self.get_quota(resource_type)
            status[resource_type.value] = {
                "limit": quota.limit,
                "current_usage": quota.current_usage,
                "usage_percentage": quota.usage_percentage,
                "is_exceeded": quota.is_exceeded,
                "reset_period": quota.reset_period,
                "last_reset": quota.last_reset
            }
        return status


class TenantContext(BaseModel):
    """Current tenant context for requests"""
    tenant_id: str
    tenant: Optional[Tenant] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    class Config:
        extra = "forbid"


class MultiTenantManager:
    """Multi-tenant management system"""

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.subdomain_map: Dict[str, str] = {}  # subdomain -> tenant_id
        self.current_contexts: Dict[str, TenantContext] = {}  # session_id -> context

    async def create_tenant(self,
                          name: str,
                          subdomain: str,
                          admin_email: str,
                          admin_name: str,
                          plan: TenantPlan = TenantPlan.TRIAL,
                          **kwargs) -> Tenant:
        """Create a new tenant"""

        # Check subdomain availability
        if subdomain in self.subdomain_map:
            raise ValueError(f"Subdomain '{subdomain}' is already taken")

        # Create tenant
        tenant_data = {
            "name": name,
            "subdomain": subdomain,
            "admin_email": admin_email,
            "admin_name": admin_name,
            "plan": plan,
            **kwargs
        }

        # Set trial expiry for trial accounts
        if plan == TenantPlan.TRIAL:
            tenant_data["trial_expires_at"] = datetime.utcnow() + timedelta(days=30)

        tenant = Tenant(**tenant_data)

        # Store tenant
        self.tenants[tenant.tenant_id] = tenant
        self.subdomain_map[subdomain] = tenant.tenant_id

        logger.info(f"Created tenant: {name} ({tenant.tenant_id}) with subdomain: {subdomain}")
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self.tenants.get(tenant_id)

    async def get_tenant_by_subdomain(self, subdomain: str) -> Optional[Tenant]:
        """Get tenant by subdomain"""
        tenant_id = self.subdomain_map.get(subdomain)
        if tenant_id:
            return await self.get_tenant(tenant_id)
        return None

    async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Update tenant configuration"""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False

        # Update allowed fields
        allowed_fields = {
            'name', 'status', 'plan', 'admin_email', 'admin_name',
            'company_name', 'subscription_expires_at', 'configuration',
            'tags', 'custom_data'
        }

        for field, value in updates.items():
            if field in allowed_fields and hasattr(tenant, field):
                setattr(tenant, field, value)

        # Update quotas if plan changed
        if 'plan' in updates:
            await self._update_plan_quotas(tenant)

        logger.info(f"Updated tenant: {tenant_id}")
        return True

    async def _update_plan_quotas(self, tenant: Tenant) -> None:
        """Update quotas when plan changes"""
        # Reset quotas to new plan limits
        tenant.quotas.clear()
        for resource_type in ResourceType:
            tenant.get_quota(resource_type)  # This will create default quota

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant (archive)"""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False

        # Archive instead of delete
        tenant.status = TenantStatus.ARCHIVED

        # Remove from subdomain map
        if tenant.subdomain in self.subdomain_map:
            del self.subdomain_map[tenant.subdomain]

        logger.info(f"Archived tenant: {tenant_id}")
        return True

    async def list_tenants(self,
                         status: Optional[TenantStatus] = None,
                         plan: Optional[TenantPlan] = None,
                         limit: Optional[int] = None,
                         offset: int = 0) -> List[Tenant]:
        """List tenants with filters"""
        tenants = list(self.tenants.values())

        # Apply filters
        if status:
            tenants = [t for t in tenants if t.status == status]

        if plan:
            tenants = [t for t in tenants if t.plan == plan]

        # Sort by creation date
        tenants.sort(key=lambda x: x.created_at, reverse=True)

        # Apply pagination
        if limit:
            tenants = tenants[offset:offset + limit]

        return tenants

    async def create_context(self,
                           tenant_id: str,
                           user_id: Optional[str] = None,
                           session_id: Optional[str] = None) -> TenantContext:
        """Create tenant context for request"""
        tenant = await self.get_tenant(tenant_id)

        context = TenantContext(
            tenant_id=tenant_id,
            tenant=tenant,
            user_id=user_id,
            session_id=session_id or str(uuid4()),
            request_id=str(uuid4())
        )

        if context.session_id:
            self.current_contexts[context.session_id] = context

        return context

    async def get_context(self, session_id: str) -> Optional[TenantContext]:
        """Get tenant context by session ID"""
        return self.current_contexts.get(session_id)

    async def check_quota(self,
                        tenant_id: str,
                        resource_type: ResourceType,
                        amount: int = 1) -> bool:
        """Check quota for tenant and resource"""
        tenant = await self.get_tenant(tenant_id)
        if not tenant or not tenant.is_active():
            return False

        return tenant.check_quota(resource_type, amount)

    async def consume_quota(self,
                          tenant_id: str,
                          resource_type: ResourceType,
                          amount: int = 1) -> bool:
        """Consume quota for tenant and resource"""
        tenant = await self.get_tenant(tenant_id)
        if not tenant or not tenant.is_active():
            return False

        success = tenant.consume_quota(resource_type, amount)

        if success:
            # Update usage tracking
            tenant.last_activity = datetime.utcnow()
            if resource_type == ResourceType.API_CALLS:
                tenant.total_api_calls += amount
            elif resource_type == ResourceType.STORAGE_MB:
                tenant.total_storage_used += amount

            # Check for quota warnings
            await self._check_quota_warnings(tenant, resource_type)

        return success

    async def _check_quota_warnings(self, tenant: Tenant, resource_type: ResourceType) -> None:
        """Check and send quota warnings"""
        quota = tenant.get_quota(resource_type)

        if quota.usage_percentage >= tenant.configuration.quota_warning_threshold * 100:
            logger.warning(f"Tenant {tenant.tenant_id} quota warning: {resource_type.value} at {quota.usage_percentage:.1f}%")

            # Send notification if configured
            if tenant.configuration.webhook_url or tenant.configuration.alert_email:
                await self._send_quota_warning(tenant, resource_type, quota)

    async def _send_quota_warning(self, tenant: Tenant, resource_type: ResourceType, quota: TenantQuota) -> None:
        """Send quota warning notification"""
        message = {
            "type": "quota_warning",
            "tenant_id": tenant.tenant_id,
            "tenant_name": tenant.name,
            "resource_type": resource_type.value,
            "usage_percentage": quota.usage_percentage,
            "current_usage": quota.current_usage,
            "limit": quota.limit,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Implementation would send webhook or email
        logger.info(f"Quota warning notification: {message}")

    async def get_tenant_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """Get analytics for tenant"""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return {}

        return {
            "tenant_id": tenant_id,
            "status": tenant.status,
            "plan": tenant.plan,
            "created_at": tenant.created_at,
            "last_activity": tenant.last_activity,
            "quota_status": tenant.get_quota_status(),
            "total_api_calls": tenant.total_api_calls,
            "total_storage_used": tenant.total_storage_used,
            "days_active": (datetime.utcnow() - tenant.created_at).days,
            "is_active": tenant.is_active()
        }

    async def cleanup_expired_contexts(self) -> None:
        """Clean up expired contexts"""
        expired = []
        cutoff = datetime.utcnow() - timedelta(hours=24)  # 24 hour context expiry

        for session_id, context in self.current_contexts.items():
            # Implementation would check context timestamp
            expired.append(session_id)

        for session_id in expired:
            del self.current_contexts[session_id]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired contexts")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide tenant statistics"""
        tenants = list(self.tenants.values())

        stats = {
            "total_tenants": len(tenants),
            "active_tenants": len([t for t in tenants if t.is_active()]),
            "by_status": {},
            "by_plan": {},
            "total_api_calls": sum(t.total_api_calls for t in tenants),
            "total_storage_used": sum(t.total_storage_used for t in tenants),
            "active_contexts": len(self.current_contexts)
        }

        # Status breakdown
        for status in TenantStatus:
            count = len([t for t in tenants if t.status == status])
            if count > 0:
                stats["by_status"][status.value] = count

        # Plan breakdown
        for plan in TenantPlan:
            count = len([t for t in tenants if t.plan == plan])
            if count > 0:
                stats["by_plan"][plan.value] = count

        return stats
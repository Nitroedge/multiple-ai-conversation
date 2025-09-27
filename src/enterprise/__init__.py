"""
Enterprise Package
Enterprise-grade features for the Multi-Agent Conversation Engine
"""

from .multi_tenancy import (
    MultiTenantManager, Tenant, TenantStatus, TenantPlan, TenantQuota,
    TenantConfiguration, TenantContext, ResourceType
)

from .security import (
    SecurityManager, User, UserRole, Permission, SecurityEvent,
    TwoFactorAuth, APIKey, JWTToken, EncryptionManager, AuthMethod
)

from .scaling import (
    ScalingManager, LoadBalancer, HealthChecker, AutoScaler,
    ServiceNode, NodeStatus, LoadBalancingStrategy, ScalingAction,
    HealthCheckResult, NodeMetrics
)

from .admin_dashboard import (
    AdminDashboard, SystemAlert, SystemMetrics, TenantAnalytics,
    UserAnalytics, AlertLevel, DashboardPermission
)

from .rate_limiting import (
    RateLimitManager, RateLimiter, RateLimitRule, RateLimitViolation,
    RateLimitStatus, RateLimitType, RateLimitScope, RateLimitAction
)

__all__ = [
    # Multi-tenancy
    "MultiTenantManager",
    "Tenant",
    "TenantStatus",
    "TenantPlan",
    "TenantQuota",
    "TenantConfiguration",
    "TenantContext",
    "ResourceType",

    # Security
    "SecurityManager",
    "User",
    "UserRole",
    "Permission",
    "SecurityEvent",
    "TwoFactorAuth",
    "APIKey",
    "JWTToken",
    "EncryptionManager",
    "AuthMethod",

    # Scaling
    "ScalingManager",
    "LoadBalancer",
    "HealthChecker",
    "AutoScaler",
    "ServiceNode",
    "NodeStatus",
    "LoadBalancingStrategy",
    "ScalingAction",
    "HealthCheckResult",
    "NodeMetrics",

    # Admin Dashboard
    "AdminDashboard",
    "SystemAlert",
    "SystemMetrics",
    "TenantAnalytics",
    "UserAnalytics",
    "AlertLevel",
    "DashboardPermission",

    # Rate Limiting
    "RateLimitManager",
    "RateLimiter",
    "RateLimitRule",
    "RateLimitViolation",
    "RateLimitStatus",
    "RateLimitType",
    "RateLimitScope",
    "RateLimitAction"
]
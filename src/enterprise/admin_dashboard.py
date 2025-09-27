"""
Admin Dashboard System
System administration, user management, and analytics interface
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from .multi_tenancy import MultiTenantManager, Tenant, TenantStatus, TenantPlan, ResourceType
from .security import SecurityManager, User, UserRole, Permission, SecurityEvent
from .scaling import ScalingManager

logger = logging.getLogger(__name__)


class DashboardPermission(str, Enum):
    """Admin dashboard permissions"""
    DASHBOARD_VIEW = "dashboard:view"
    DASHBOARD_ADMIN = "dashboard:admin"

    TENANTS_VIEW = "tenants:view"
    TENANTS_MANAGE = "tenants:manage"

    USERS_VIEW = "users:view"
    USERS_MANAGE = "users:manage"

    SYSTEM_VIEW = "system:view"
    SYSTEM_MANAGE = "system:manage"

    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"

    LOGS_VIEW = "logs:view"
    LOGS_EXPORT = "logs:export"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemAlert(BaseModel):
    """System alert"""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    level: AlertLevel
    title: str
    message: str
    component: str  # Which system component
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    class Config:
        extra = "forbid"


class SystemMetrics(BaseModel):
    """System-wide metrics"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Tenant metrics
    total_tenants: int = 0
    active_tenants: int = 0
    trial_tenants: int = 0
    paid_tenants: int = 0

    # User metrics
    total_users: int = 0
    active_users: int = 0
    new_users_today: int = 0

    # Resource usage
    total_api_calls: int = 0
    total_storage_mb: int = 0
    total_ai_tokens: int = 0

    # Performance metrics
    average_response_time: float = 0.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0

    # Infrastructure
    active_nodes: int = 0
    total_nodes: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    class Config:
        extra = "forbid"


class TenantAnalytics(BaseModel):
    """Tenant analytics data"""
    tenant_id: str
    tenant_name: str
    plan: TenantPlan
    status: TenantStatus

    # Usage metrics
    api_calls_last_30d: int = 0
    storage_used_mb: int = 0
    active_users: int = 0
    workflows_created: int = 0

    # Financial metrics
    monthly_revenue: float = 0.0
    cost_per_api_call: float = 0.0

    # Engagement metrics
    last_activity: Optional[datetime] = None
    daily_active_users: int = 0
    feature_usage: Dict[str, int] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class UserAnalytics(BaseModel):
    """User analytics data"""
    user_id: str
    username: str
    tenant_id: Optional[str] = None
    roles: List[str] = Field(default_factory=list)

    # Activity metrics
    last_login: Optional[datetime] = None
    total_logins: int = 0
    api_calls_last_30d: int = 0
    workflows_created: int = 0

    # Security metrics
    failed_login_attempts: int = 0
    two_factor_enabled: bool = False
    last_password_change: Optional[datetime] = None

    class Config:
        extra = "forbid"


class AdminDashboard:
    """Main admin dashboard system"""

    def __init__(self,
                 tenant_manager: MultiTenantManager,
                 security_manager: SecurityManager,
                 scaling_manager: ScalingManager):
        self.tenant_manager = tenant_manager
        self.security_manager = security_manager
        self.scaling_manager = scaling_manager

        self.alerts: List[SystemAlert] = []
        self.metrics_history: List[SystemMetrics] = []

        self.monitoring_enabled = True
        self.monitoring_task: Optional[asyncio.Task] = None

    async def start_monitoring(self) -> None:
        """Start system monitoring"""
        if self.monitoring_task:
            return

        self.monitoring_enabled = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Admin dashboard monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Admin dashboard monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                # Collect system metrics
                await self._collect_system_metrics()

                # Check for alerts
                await self._check_system_alerts()

                # Cleanup old data
                await self._cleanup_old_data()

                await asyncio.sleep(60)  # Run every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)

    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics"""
        try:
            # Get tenant metrics
            tenants = await self.tenant_manager.list_tenants()
            active_tenants = [t for t in tenants if t.is_active()]
            trial_tenants = [t for t in tenants if t.plan == TenantPlan.TRIAL]
            paid_tenants = [t for t in tenants if t.plan != TenantPlan.TRIAL and t.plan != TenantPlan.FREE]

            # Get user metrics
            users = list(self.security_manager.users.values())
            today = datetime.utcnow().date()
            new_users_today = [u for u in users if u.created_at.date() == today]
            active_users = [u for u in users if u.last_login and
                          (datetime.utcnow() - u.last_login).days <= 30]

            # Calculate resource usage
            total_api_calls = sum(t.total_api_calls for t in tenants)
            total_storage_mb = sum(t.total_storage_used for t in tenants)

            # Get infrastructure metrics
            scaling_status = self.scaling_manager.get_system_status()
            load_balancer = scaling_status.get("load_balancer", {})

            metrics = SystemMetrics(
                total_tenants=len(tenants),
                active_tenants=len(active_tenants),
                trial_tenants=len(trial_tenants),
                paid_tenants=len(paid_tenants),
                total_users=len(users),
                active_users=len(active_users),
                new_users_today=len(new_users_today),
                total_api_calls=total_api_calls,
                total_storage_mb=total_storage_mb,
                active_nodes=load_balancer.get("available_nodes", 0),
                total_nodes=load_balancer.get("total_nodes", 0)
            )

            self.metrics_history.append(metrics)

            # Keep only last 24 hours of metrics
            cutoff = datetime.utcnow() - timedelta(hours=24)
            self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff]

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _check_system_alerts(self) -> None:
        """Check for system alerts"""
        try:
            # Check tenant-related alerts
            await self._check_tenant_alerts()

            # Check user-related alerts
            await self._check_user_alerts()

            # Check infrastructure alerts
            await self._check_infrastructure_alerts()

            # Check security alerts
            await self._check_security_alerts()

        except Exception as e:
            logger.error(f"Failed to check system alerts: {e}")

    async def _check_tenant_alerts(self) -> None:
        """Check tenant-related alerts"""
        tenants = await self.tenant_manager.list_tenants()

        for tenant in tenants:
            # Check quota usage
            quota_status = tenant.get_quota_status()
            for resource_type, status in quota_status.items():
                if status["usage_percentage"] > 90:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"High quota usage for tenant {tenant.name}",
                        f"Resource {resource_type} is at {status['usage_percentage']:.1f}% usage",
                        "tenant_quota",
                        {"tenant_id": tenant.tenant_id, "resource_type": resource_type}
                    )
                elif status["is_exceeded"]:
                    await self._create_alert(
                        AlertLevel.ERROR,
                        f"Quota exceeded for tenant {tenant.name}",
                        f"Resource {resource_type} quota has been exceeded",
                        "tenant_quota",
                        {"tenant_id": tenant.tenant_id, "resource_type": resource_type}
                    )

            # Check trial expiry
            if tenant.plan == TenantPlan.TRIAL and tenant.trial_expires_at:
                days_until_expiry = (tenant.trial_expires_at - datetime.utcnow()).days
                if days_until_expiry <= 3:
                    await self._create_alert(
                        AlertLevel.WARNING,
                        f"Trial expiring soon for {tenant.name}",
                        f"Trial expires in {days_until_expiry} days",
                        "trial_expiry",
                        {"tenant_id": tenant.tenant_id}
                    )

    async def _check_user_alerts(self) -> None:
        """Check user-related alerts"""
        users = list(self.security_manager.users.values())

        # Check for locked accounts
        locked_users = [u for u in users if u.account_locked]
        if locked_users:
            await self._create_alert(
                AlertLevel.WARNING,
                f"{len(locked_users)} locked user accounts",
                f"Multiple user accounts are locked due to failed login attempts",
                "user_accounts",
                {"locked_count": len(locked_users)}
            )

    async def _check_infrastructure_alerts(self) -> None:
        """Check infrastructure alerts"""
        scaling_status = self.scaling_manager.get_system_status()
        load_balancer = scaling_status.get("load_balancer", {})

        # Check node availability
        available_nodes = load_balancer.get("available_nodes", 0)
        total_nodes = load_balancer.get("total_nodes", 0)

        if total_nodes > 0:
            availability_ratio = available_nodes / total_nodes
            if availability_ratio < 0.5:
                await self._create_alert(
                    AlertLevel.CRITICAL,
                    "Low node availability",
                    f"Only {available_nodes}/{total_nodes} nodes are available",
                    "infrastructure",
                    {"available_nodes": available_nodes, "total_nodes": total_nodes}
                )
            elif availability_ratio < 0.8:
                await self._create_alert(
                    AlertLevel.WARNING,
                    "Reduced node availability",
                    f"{available_nodes}/{total_nodes} nodes are available",
                    "infrastructure",
                    {"available_nodes": available_nodes, "total_nodes": total_nodes}
                )

    async def _check_security_alerts(self) -> None:
        """Check security alerts"""
        # Get recent security events
        recent_events = await self.security_manager.get_security_events(hours=1)

        # Check for high failed login rate
        failed_logins = [e for e in recent_events if e.event_type == "login_failed"]
        if len(failed_logins) > 10:  # More than 10 failed logins in an hour
            await self._create_alert(
                AlertLevel.WARNING,
                "High failed login rate",
                f"{len(failed_logins)} failed login attempts in the last hour",
                "security",
                {"failed_login_count": len(failed_logins)}
            )

        # Check for critical security events
        critical_events = [e for e in recent_events if e.severity == "critical"]
        if critical_events:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "Critical security events detected",
                f"{len(critical_events)} critical security events in the last hour",
                "security",
                {"critical_event_count": len(critical_events)}
            )

    async def _create_alert(self,
                          level: AlertLevel,
                          title: str,
                          message: str,
                          component: str,
                          details: Optional[Dict[str, Any]] = None) -> None:
        """Create a system alert"""
        # Check if similar alert already exists and is recent
        recent_alerts = [a for a in self.alerts if
                        a.component == component and
                        a.title == title and
                        (datetime.utcnow() - a.timestamp).total_seconds() < 3600]  # 1 hour

        if recent_alerts:
            return  # Don't create duplicate recent alerts

        alert = SystemAlert(
            level=level,
            title=title,
            message=message,
            component=component,
            details=details or {}
        )

        self.alerts.append(alert)
        logger.warning(f"System alert created: {title}")

    async def _cleanup_old_data(self) -> None:
        """Clean up old data"""
        cutoff = datetime.utcnow() - timedelta(days=30)

        # Clean up old alerts
        original_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        if len(self.alerts) < original_count:
            logger.info(f"Cleaned up {original_count - len(self.alerts)} old alerts")

    # Dashboard API Methods

    async def get_dashboard_overview(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard overview data"""
        # Check permissions
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.SYSTEM_ADMIN):
            raise PermissionError("Insufficient permissions for dashboard access")

        latest_metrics = self.metrics_history[-1] if self.metrics_history else SystemMetrics()

        # Get recent alerts
        recent_alerts = [a for a in self.alerts if not a.acknowledged]
        critical_alerts = [a for a in recent_alerts if a.level == AlertLevel.CRITICAL]

        return {
            "metrics": latest_metrics.dict(),
            "alerts": {
                "total": len(recent_alerts),
                "critical": len(critical_alerts),
                "recent": [a.dict() for a in recent_alerts[:10]]
            },
            "system_health": await self._get_system_health_summary(),
            "tenant_summary": await self._get_tenant_summary(),
            "user_summary": await self._get_user_summary()
        }

    async def _get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        scaling_status = self.scaling_manager.get_system_status()
        security_stats = self.security_manager.get_system_security_stats()

        return {
            "overall_status": "healthy",  # Would be calculated based on alerts and metrics
            "infrastructure": {
                "nodes_available": scaling_status.get("load_balancer", {}).get("available_nodes", 0),
                "auto_scaling_enabled": scaling_status.get("auto_scaler", {}).get("enabled", False)
            },
            "security": {
                "locked_accounts": security_stats.get("locked_accounts", 0),
                "failed_logins_24h": security_stats.get("failed_logins_24h", 0)
            }
        }

    async def _get_tenant_summary(self) -> Dict[str, Any]:
        """Get tenant summary"""
        tenants = await self.tenant_manager.list_tenants()

        return {
            "total": len(tenants),
            "active": len([t for t in tenants if t.is_active()]),
            "by_plan": {
                plan.value: len([t for t in tenants if t.plan == plan])
                for plan in TenantPlan
            },
            "recent_signups": len([t for t in tenants if
                                  (datetime.utcnow() - t.created_at).days <= 7])
        }

    async def _get_user_summary(self) -> Dict[str, Any]:
        """Get user summary"""
        users = list(self.security_manager.users.values())

        return {
            "total": len(users),
            "active": len([u for u in users if u.last_login and
                          (datetime.utcnow() - u.last_login).days <= 30]),
            "new_today": len([u for u in users if
                             u.created_at.date() == datetime.utcnow().date()]),
            "locked": len([u for u in users if u.account_locked])
        }

    async def get_tenant_analytics(self, user_id: str, limit: int = 50) -> List[TenantAnalytics]:
        """Get tenant analytics"""
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.TENANT_READ):
            raise PermissionError("Insufficient permissions")

        tenants = await self.tenant_manager.list_tenants(limit=limit)
        analytics = []

        for tenant in tenants:
            tenant_analytics = await self.tenant_manager.get_tenant_analytics(tenant.tenant_id)

            analytics.append(TenantAnalytics(
                tenant_id=tenant.tenant_id,
                tenant_name=tenant.name,
                plan=tenant.plan,
                status=tenant.status,
                api_calls_last_30d=tenant_analytics.get("total_api_calls", 0),
                storage_used_mb=tenant_analytics.get("total_storage_used", 0),
                last_activity=tenant.last_activity
            ))

        return analytics

    async def get_user_analytics(self, user_id: str, limit: int = 50) -> List[UserAnalytics]:
        """Get user analytics"""
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.USER_READ):
            raise PermissionError("Insufficient permissions")

        users = list(self.security_manager.users.values())[:limit]
        analytics = []

        for u in users:
            analytics.append(UserAnalytics(
                user_id=u.user_id,
                username=u.username,
                tenant_id=u.tenant_id,
                roles=[role.value for role in u.roles],
                last_login=u.last_login,
                failed_login_attempts=u.failed_login_attempts,
                two_factor_enabled=u.two_factor is not None and u.two_factor.enabled,
                last_password_change=u.last_password_change
            ))

        return analytics

    async def acknowledge_alert(self, user_id: str, alert_id: str) -> bool:
        """Acknowledge a system alert"""
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.SYSTEM_ADMIN):
            raise PermissionError("Insufficient permissions")

        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.utcnow()
                return True

        return False

    async def resolve_alert(self, user_id: str, alert_id: str) -> bool:
        """Resolve a system alert"""
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.SYSTEM_ADMIN):
            raise PermissionError("Insufficient permissions")

        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return True

        return False

    async def get_system_logs(self,
                            user_id: str,
                            component: Optional[str] = None,
                            level: Optional[str] = None,
                            hours: int = 24,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get system logs"""
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.SYSTEM_LOGS):
            raise PermissionError("Insufficient permissions")

        # Get security events as logs
        security_events = await self.security_manager.get_security_events(
            hours=hours, limit=limit
        )

        logs = []
        for event in security_events:
            if component and event.event_type != component:
                continue
            if level and event.severity != level:
                continue

            logs.append({
                "timestamp": event.timestamp,
                "level": event.severity,
                "component": event.event_type,
                "message": f"{event.event_type}: {event.result}",
                "details": event.details,
                "user_id": event.user_id,
                "tenant_id": event.tenant_id
            })

        return logs

    async def export_analytics(self,
                             user_id: str,
                             export_type: str,
                             format: str = "json") -> Dict[str, Any]:
        """Export analytics data"""
        user = await self.security_manager.get_user(user_id)
        if not user or not user.has_permission(Permission.ANALYTICS_EXPORT):
            raise PermissionError("Insufficient permissions")

        data = {}

        if export_type == "tenants":
            data["tenants"] = [analytics.dict() for analytics in await self.get_tenant_analytics(user_id)]
        elif export_type == "users":
            data["users"] = [analytics.dict() for analytics in await self.get_user_analytics(user_id)]
        elif export_type == "metrics":
            data["metrics"] = [metrics.dict() for metrics in self.metrics_history]
        elif export_type == "alerts":
            data["alerts"] = [alert.dict() for alert in self.alerts]
        else:
            raise ValueError(f"Invalid export type: {export_type}")

        data["exported_at"] = datetime.utcnow()
        data["exported_by"] = user_id

        return data

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        return {
            "alerts_count": len(self.alerts),
            "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "critical_alerts": len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            "metrics_collected": len(self.metrics_history),
            "monitoring_enabled": self.monitoring_enabled,
            "last_metrics_collection": self.metrics_history[-1].timestamp if self.metrics_history else None
        }
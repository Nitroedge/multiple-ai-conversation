"""
Advanced Orchestration Package
Complex workflow engine for sophisticated multi-agent coordination
"""

from .workflow_engine import (
    WorkflowEngine, WorkflowDefinition, WorkflowStep, WorkflowCondition,
    WorkflowExecutor, WorkflowState, StepType, ConditionType
)
from .dynamic_roles import (
    DynamicRoleManager, RoleOptimizer, RoleAssignment,
    PerformanceBasedRoleSelection, SkillBasedRoleSelection
)
from .external_integrations import (
    ExternalAPIManager, WebhookManager, ThirdPartyIntegration,
    APIEndpoint, WebhookConfig, IntegrationStatus
)
from .workflow_templates import (
    WorkflowTemplateManager, TemplateLibrary, WorkflowTemplate,
    TemplateCategory, TemplateCustomization
)
from .analytics_engine import (
    PerformanceAnalyticsEngine, MetricsCollector, AnalyticsReporter,
    OptimizationRecommendations, TrendAnalyzer
)

__all__ = [
    # Workflow Engine
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowCondition",
    "WorkflowExecutor",
    "WorkflowState",
    "StepType",
    "ConditionType",

    # Dynamic Roles
    "DynamicRoleManager",
    "RoleOptimizer",
    "RoleAssignment",
    "PerformanceBasedRoleSelection",
    "SkillBasedRoleSelection",

    # External Integrations
    "ExternalAPIManager",
    "WebhookManager",
    "ThirdPartyIntegration",
    "APIEndpoint",
    "WebhookConfig",
    "IntegrationStatus",

    # Workflow Templates
    "WorkflowTemplateManager",
    "TemplateLibrary",
    "WorkflowTemplate",
    "TemplateCategory",
    "TemplateCustomization",

    # Analytics
    "PerformanceAnalyticsEngine",
    "MetricsCollector",
    "AnalyticsReporter",
    "OptimizationRecommendations",
    "TrendAnalyzer"
]
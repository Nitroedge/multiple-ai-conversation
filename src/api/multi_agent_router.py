"""
Multi-Agent Coordination API Router

This module provides REST API endpoints for advanced multi-agent coordination,
including conflict resolution, collaboration workflows, agent communication,
and role management.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from fastapi import APIRouter, HTTPException, Query, Path, Body, BackgroundTasks
from pydantic import BaseModel
import uuid

from ..multi_agent.conflict_resolver import (
    ConflictResolver, ConflictType, ConflictSeverity, ResolutionStrategy,
    ConflictParticipant, Conflict
)
from ..multi_agent.collaboration_engine import (
    CollaborationEngine, CollaborationPattern, TaskType, AgentRole,
    CollaborationWorkflow, CollaborationTask, CollaborationTemplate
)
from ..multi_agent.agent_coordinator import AgentCoordinator, CoordinationStrategy, AgentStatus
from ..multi_agent.agent_communication import AgentCommunication, MessageType, CommunicationProtocol
from ..multi_agent.conversation_orchestrator import ConversationOrchestrator, FlowState, TurnType
from ..multi_agent.role_manager import RoleManager
from ..multi_agent.context_sharing import ContextSharingManager

logger = logging.getLogger(__name__)
multi_agent_router = APIRouter(prefix="/multi-agent", tags=["multi-agent"])

# Global instances (will be initialized in main.py)
conflict_resolver: Optional[ConflictResolver] = None
collaboration_engine: Optional[CollaborationEngine] = None
agent_coordinator: Optional[AgentCoordinator] = None
agent_communication: Optional[AgentCommunication] = None
conversation_orchestrator: Optional[ConversationOrchestrator] = None
role_manager: Optional[RoleManager] = None
context_sharing_manager: Optional[ContextSharingManager] = None


# Pydantic models for API requests/responses

# Conflict Resolution Models
class ConflictDetectionRequest(BaseModel):
    context: Dict[str, Any]
    agents: Dict[str, Dict[str, Any]]
    resource_claims: Optional[Dict[str, List[str]]] = None
    proposed_actions: Optional[List[Dict[str, Any]]] = None
    agent_priorities: Optional[Dict[str, int]] = None

class ConflictParticipantModel(BaseModel):
    agent_id: str
    position: Dict[str, Any]
    priority: int = 0
    confidence: float = 0.5
    expertise_score: float = 0.5
    performance_history: float = 0.5
    resources_claimed: List[str] = []
    metadata: Dict[str, Any] = {}

class ConflictResolutionRequest(BaseModel):
    conflict_id: str
    preferred_strategy: Optional[str] = None

class ConflictResponse(BaseModel):
    conflict_id: str
    conflict_type: str
    severity: str
    participants: List[ConflictParticipantModel]
    status: str
    resolution_strategy: Optional[str] = None
    resolution_result: Optional[Dict[str, Any]] = None
    detected_at: str
    resolved_at: Optional[str] = None

# Collaboration Models
class WorkflowCreationRequest(BaseModel):
    template_id: str
    title: str
    description: str
    context: Optional[Dict[str, Any]] = None

class AgentAssignmentRequest(BaseModel):
    workflow_id: str
    available_agents: Dict[str, Dict[str, Any]]

class TaskExecutionRequest(BaseModel):
    task_id: str
    agent_results: Dict[str, Any]
    quality_score: Optional[float] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    pattern: str
    title: str
    description: str
    phase: str
    completion_percentage: float
    participating_agents: List[str]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

# Agent Coordination Models
class AgentRegistrationRequest(BaseModel):
    agent_id: str
    capabilities: Dict[str, Any]
    specializations: List[str] = []
    max_concurrent_tasks: int = 3
    priority_level: str = "normal"

class TaskAssignmentRequest(BaseModel):
    task_description: str
    requirements: Dict[str, Any]
    preferred_agents: Optional[List[str]] = None
    coordination_strategy: Optional[str] = None

class AgentStatusRequest(BaseModel):
    agent_id: str
    status: str
    metadata: Optional[Dict[str, Any]] = None

# Communication Models
class MessageSendRequest(BaseModel):
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast
    message_type: str
    content: Dict[str, Any]
    priority: str = "normal"
    metadata: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    message_id: str
    sender_id: str
    recipient_id: Optional[str]
    message_type: str
    content: Dict[str, Any]
    timestamp: str
    status: str


# Conflict Resolution Endpoints

@multi_agent_router.post("/conflicts/detect", response_model=List[ConflictResponse])
async def detect_conflicts(request: ConflictDetectionRequest):
    """Detect conflicts in multi-agent context"""
    if not conflict_resolver:
        raise HTTPException(status_code=503, detail="Conflict resolver not initialized")

    try:
        context = request.context.copy()
        context["agents"] = request.agents

        if request.resource_claims:
            context["resource_claims"] = request.resource_claims
        if request.proposed_actions:
            context["proposed_actions"] = request.proposed_actions
        if request.agent_priorities:
            context["agent_priorities"] = request.agent_priorities

        conflicts = await conflict_resolver.detect_conflicts(context)

        return [
            ConflictResponse(
                conflict_id=c.conflict_id,
                conflict_type=c.conflict_type.value,
                severity=c.severity.value,
                participants=[
                    ConflictParticipantModel(
                        agent_id=p.agent_id,
                        position=p.position,
                        priority=p.priority,
                        confidence=p.confidence,
                        expertise_score=p.expertise_score,
                        performance_history=p.performance_history,
                        resources_claimed=list(p.resources_claimed),
                        metadata=p.metadata
                    ) for p in c.participants
                ],
                status=c.status.value,
                resolution_strategy=c.resolution_strategy.value if c.resolution_strategy else None,
                resolution_result=c.resolution_result,
                detected_at=c.detected_at.isoformat(),
                resolved_at=c.resolved_at.isoformat() if c.resolved_at else None
            ) for c in conflicts
        ]

    except Exception as e:
        logger.error(f"Error detecting conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: str = Path(..., description="Conflict ID"),
    request: Optional[ConflictResolutionRequest] = None
):
    """Resolve a specific conflict"""
    if not conflict_resolver:
        raise HTTPException(status_code=503, detail="Conflict resolver not initialized")

    try:
        if conflict_id not in conflict_resolver.active_conflicts:
            raise HTTPException(status_code=404, detail="Conflict not found")

        conflict = conflict_resolver.active_conflicts[conflict_id]

        # Override resolution strategy if provided
        if request and request.preferred_strategy:
            try:
                conflict.resolution_strategy = ResolutionStrategy(request.preferred_strategy)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid resolution strategy")

        result = await conflict_resolver.resolve_conflict(conflict)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving conflict {conflict_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/conflicts/active")
async def get_active_conflicts():
    """Get all active conflicts"""
    if not conflict_resolver:
        raise HTTPException(status_code=503, detail="Conflict resolver not initialized")

    try:
        active_conflicts = []
        for conflict in conflict_resolver.active_conflicts.values():
            active_conflicts.append(ConflictResponse(
                conflict_id=conflict.conflict_id,
                conflict_type=conflict.conflict_type.value,
                severity=conflict.severity.value,
                participants=[
                    ConflictParticipantModel(
                        agent_id=p.agent_id,
                        position=p.position,
                        priority=p.priority,
                        confidence=p.confidence,
                        expertise_score=p.expertise_score,
                        performance_history=p.performance_history,
                        resources_claimed=list(p.resources_claimed),
                        metadata=p.metadata
                    ) for p in conflict.participants
                ],
                status=conflict.status.value,
                resolution_strategy=conflict.resolution_strategy.value if conflict.resolution_strategy else None,
                resolution_result=conflict.resolution_result,
                detected_at=conflict.detected_at.isoformat(),
                resolved_at=conflict.resolved_at.isoformat() if conflict.resolved_at else None
            ))

        return active_conflicts

    except Exception as e:
        logger.error(f"Error getting active conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/conflicts/metrics")
async def get_conflict_metrics():
    """Get conflict resolution metrics"""
    if not conflict_resolver:
        raise HTTPException(status_code=503, detail="Conflict resolver not initialized")

    try:
        metrics = await conflict_resolver.get_conflict_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting conflict metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Collaboration Workflow Endpoints

@multi_agent_router.post("/workflows/create", response_model=WorkflowResponse)
async def create_workflow(request: WorkflowCreationRequest):
    """Create a new collaboration workflow from template"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        workflow = await collaboration_engine.create_workflow_from_template(
            template_id=request.template_id,
            title=request.title,
            description=request.description,
            context=request.context
        )

        return WorkflowResponse(
            workflow_id=workflow.workflow_id,
            pattern=workflow.pattern.value,
            title=workflow.title,
            description=workflow.description,
            phase=workflow.phase.value,
            completion_percentage=workflow.get_completion_percentage(),
            participating_agents=list(workflow.participating_agents),
            created_at=workflow.created_at.isoformat(),
            started_at=workflow.started_at.isoformat() if workflow.started_at else None,
            completed_at=workflow.completed_at.isoformat() if workflow.completed_at else None
        )

    except Exception as e:
        logger.error(f"Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.post("/workflows/{workflow_id}/assign-agents")
async def assign_agents_to_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    request: AgentAssignmentRequest = Body(...)
):
    """Assign agents to workflow tasks"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        assignments = await collaboration_engine.assign_agents_to_workflow(
            workflow_id=workflow_id,
            available_agents=request.available_agents
        )

        return {
            "workflow_id": workflow_id,
            "assignments": assignments,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error assigning agents to workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.post("/workflows/{workflow_id}/start")
async def start_workflow(workflow_id: str = Path(..., description="Workflow ID")):
    """Start execution of a workflow"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        success = await collaboration_engine.start_workflow(workflow_id)

        if not success:
            raise HTTPException(status_code=400, detail="Could not start workflow")

        workflow_status = await collaboration_engine.get_workflow_status(workflow_id)
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "workflow": workflow_status,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str = Path(..., description="Workflow ID")):
    """Get current workflow status"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        workflow_status = await collaboration_engine.get_workflow_status(workflow_id)

        if not workflow_status:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return workflow_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    reason: str = Body("User cancelled", embed=True)
):
    """Cancel an active workflow"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        success = await collaboration_engine.cancel_workflow(workflow_id, reason)

        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or already completed")

        return {
            "workflow_id": workflow_id,
            "status": "cancelled",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/workflows/templates")
async def get_collaboration_templates():
    """Get available collaboration templates"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        templates = []
        for template in collaboration_engine.collaboration_templates.values():
            templates.append({
                "template_id": template.template_id,
                "pattern": template.pattern.value,
                "name": template.name,
                "description": template.description,
                "complexity_level": template.complexity_level,
                "estimated_duration": template.estimated_duration.total_seconds() if template.estimated_duration else None,
                "role_requirements": {role.value: count for role, count in template.role_requirements.items()},
                "tags": list(template.tags)
            })

        return {"templates": templates}

    except Exception as e:
        logger.error(f"Error getting collaboration templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/workflows/metrics")
async def get_collaboration_metrics():
    """Get collaboration workflow metrics"""
    if not collaboration_engine:
        raise HTTPException(status_code=503, detail="Collaboration engine not initialized")

    try:
        metrics = await collaboration_engine.get_collaboration_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting collaboration metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent Coordination Endpoints

@multi_agent_router.post("/agents/register")
async def register_agent(request: AgentRegistrationRequest):
    """Register a new agent with the coordinator"""
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    try:
        success = await agent_coordinator.register_agent(
            agent_id=request.agent_id,
            capabilities=request.capabilities,
            metadata={
                "specializations": request.specializations,
                "max_concurrent_tasks": request.max_concurrent_tasks,
                "priority_level": request.priority_level
            }
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to register agent")

        return {
            "agent_id": request.agent_id,
            "status": "registered",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering agent {request.agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.post("/agents/{agent_id}/status")
async def update_agent_status(
    agent_id: str = Path(..., description="Agent ID"),
    request: AgentStatusRequest = Body(...)
):
    """Update agent status"""
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    try:
        success = await agent_coordinator.update_agent_status(
            agent_id=agent_id,
            status=AgentStatus(request.status),
            metadata=request.metadata or {}
        )

        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")

        return {
            "agent_id": agent_id,
            "status": request.status,
            "updated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent status {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.post("/tasks/assign")
async def assign_task(request: TaskAssignmentRequest):
    """Assign a task to the most suitable agent"""
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    try:
        strategy = CoordinationStrategy.EXPERTISE_BASED
        if request.coordination_strategy:
            strategy = CoordinationStrategy(request.coordination_strategy)

        assignment = await agent_coordinator.assign_task(
            task_description=request.task_description,
            requirements=request.requirements,
            preferred_agents=request.preferred_agents,
            strategy=strategy
        )

        if not assignment:
            raise HTTPException(status_code=404, detail="No suitable agent found")

        return assignment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/agents/active")
async def get_active_agents():
    """Get all active agents"""
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    try:
        agents = await agent_coordinator.get_active_agents()
        return {"agents": agents}

    except Exception as e:
        logger.error(f"Error getting active agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/coordination/metrics")
async def get_coordination_metrics():
    """Get agent coordination metrics"""
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")

    try:
        metrics = await agent_coordinator.get_coordination_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting coordination metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Communication Endpoints

@multi_agent_router.post("/communication/send", response_model=MessageResponse)
async def send_message(request: MessageSendRequest):
    """Send a message between agents"""
    if not agent_communication:
        raise HTTPException(status_code=503, detail="Agent communication not initialized")

    try:
        message = await agent_communication.send_message(
            sender_id=request.sender_id,
            recipient_id=request.recipient_id,
            message_type=MessageType(request.message_type),
            content=request.content,
            priority=request.priority,
            metadata=request.metadata
        )

        return MessageResponse(
            message_id=message.message_id,
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            message_type=message.message_type.value,
            content=message.content,
            timestamp=message.timestamp.isoformat(),
            status=message.status
        )

    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/communication/{agent_id}/messages")
async def get_agent_messages(
    agent_id: str = Path(..., description="Agent ID"),
    limit: int = Query(50, description="Maximum number of messages"),
    message_type: Optional[str] = Query(None, description="Filter by message type")
):
    """Get messages for a specific agent"""
    if not agent_communication:
        raise HTTPException(status_code=503, detail="Agent communication not initialized")

    try:
        messages = await agent_communication.get_messages_for_agent(
            agent_id=agent_id,
            limit=limit,
            message_type=MessageType(message_type) if message_type else None
        )

        return {
            "agent_id": agent_id,
            "messages": [
                {
                    "message_id": msg.message_id,
                    "sender_id": msg.sender_id,
                    "recipient_id": msg.recipient_id,
                    "message_type": msg.message_type.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "status": msg.status
                } for msg in messages
            ]
        }

    except Exception as e:
        logger.error(f"Error getting messages for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@multi_agent_router.get("/communication/metrics")
async def get_communication_metrics():
    """Get communication system metrics"""
    if not agent_communication:
        raise HTTPException(status_code=503, detail="Agent communication not initialized")

    try:
        metrics = await agent_communication.get_communication_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Error getting communication metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Check and Status Endpoints

@multi_agent_router.get("/health")
async def health_check():
    """Check health of all multi-agent systems"""
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "systems": {
            "conflict_resolver": conflict_resolver is not None,
            "collaboration_engine": collaboration_engine is not None,
            "agent_coordinator": agent_coordinator is not None,
            "agent_communication": agent_communication is not None,
            "conversation_orchestrator": conversation_orchestrator is not None,
            "role_manager": role_manager is not None,
            "context_sharing_manager": context_sharing_manager is not None
        }
    }

    all_healthy = all(health_status["systems"].values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "details": health_status
    }


@multi_agent_router.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status = {
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }

        if conflict_resolver:
            status["systems"]["conflict_resolver"] = await conflict_resolver.get_conflict_metrics()

        if collaboration_engine:
            status["systems"]["collaboration_engine"] = await collaboration_engine.get_collaboration_metrics()

        if agent_coordinator:
            status["systems"]["agent_coordinator"] = await agent_coordinator.get_coordination_metrics()

        if agent_communication:
            status["systems"]["agent_communication"] = await agent_communication.get_communication_metrics()

        return status

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Initialization function for main.py
async def initialize_multi_agent_systems(redis_client):
    """Initialize all multi-agent coordination systems"""
    global conflict_resolver, collaboration_engine, agent_coordinator
    global agent_communication, conversation_orchestrator, role_manager, context_sharing_manager

    try:
        # Initialize systems
        conflict_resolver = ConflictResolver(redis_client)
        collaboration_engine = CollaborationEngine(redis_client)
        agent_coordinator = AgentCoordinator(redis_client)
        agent_communication = AgentCommunication(redis_client)
        conversation_orchestrator = ConversationOrchestrator()
        role_manager = RoleManager()
        context_sharing_manager = ContextSharingManager(redis_client)

        logger.info("✅ Multi-agent coordination systems initialized")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize multi-agent systems: {e}")
        return False


def get_multi_agent_systems():
    """Get all initialized multi-agent systems"""
    return {
        "conflict_resolver": conflict_resolver,
        "collaboration_engine": collaboration_engine,
        "agent_coordinator": agent_coordinator,
        "agent_communication": agent_communication,
        "conversation_orchestrator": conversation_orchestrator,
        "role_manager": role_manager,
        "context_sharing_manager": context_sharing_manager
    }
"""
Horizontal Scaling Infrastructure
Load balancing, distributed deployment, and auto-scaling capabilities
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean, median
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4

import aiohttp
import psutil
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Node status types"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    STARTING = "starting"
    STOPPING = "stopping"


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CONSISTENT_HASH = "consistent_hash"
    GEOGRAPHIC = "geographic"


class ScalingAction(str, Enum):
    """Auto-scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Add nodes
    SCALE_IN = "scale_in"   # Remove nodes
    REBALANCE = "rebalance"
    NONE = "none"


class HealthCheckResult(BaseModel):
    """Health check result"""
    node_id: str
    endpoint: str
    status: NodeStatus
    response_time_ms: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    error_rate: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class NodeMetrics(BaseModel):
    """Node performance metrics"""
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0

    # Application metrics
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0

    # Business metrics
    active_workflows: int = 0
    active_agents: int = 0
    memory_operations_per_second: float = 0.0
    ai_tokens_per_second: float = 0.0

    class Config:
        extra = "forbid"


class ServiceNode(BaseModel):
    """Service node configuration"""
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    host: str
    port: int
    region: str = "default"
    zone: str = "default"

    # Node characteristics
    weight: float = 1.0  # For weighted load balancing
    max_connections: int = 1000
    capacity_factor: float = 1.0  # Relative capacity compared to baseline

    # Health and status
    status: NodeStatus = NodeStatus.STARTING
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    added_at: datetime = Field(default_factory=datetime.utcnow)

    # Current load
    current_connections: int = 0
    current_load: float = 0.0  # 0.0 to 1.0

    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"

    @property
    def endpoint(self) -> str:
        """Get node endpoint URL"""
        return f"http://{self.host}:{self.port}"

    @property
    def is_available(self) -> bool:
        """Check if node is available for requests"""
        return self.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]

    def calculate_load_score(self, strategy: LoadBalancingStrategy) -> float:
        """Calculate load score for balancing decisions"""
        if not self.is_available:
            return float('inf')

        if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self.current_connections / self.max_connections

        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self.current_load / self.weight

        elif strategy == LoadBalancingStrategy.RESOURCE_BASED:
            # Combine multiple factors
            connection_load = self.current_connections / self.max_connections
            cpu_load = self.current_load
            return (connection_load + cpu_load) / 2

        else:  # Default to current load
            return self.current_load


class LoadBalancer:
    """Load balancer for distributing requests across nodes"""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.nodes: Dict[str, ServiceNode] = {}
        self.current_index = 0  # For round-robin
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    def add_node(self, node: ServiceNode) -> None:
        """Add a node to the load balancer"""
        self.nodes[node.node_id] = node
        logger.info(f"Added node to load balancer: {node.name} ({node.endpoint})")

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the load balancer"""
        if node_id in self.nodes:
            node = self.nodes.pop(node_id)
            logger.info(f"Removed node from load balancer: {node.name}")
            return True
        return False

    def get_available_nodes(self) -> List[ServiceNode]:
        """Get all available nodes"""
        return [node for node in self.nodes.values() if node.is_available]

    def select_node(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceNode]:
        """Select the best node for a request"""
        available_nodes = self.get_available_nodes()
        if not available_nodes:
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_nodes)

        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_nodes)

        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_nodes)

        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_nodes)

        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(available_nodes)

        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_selection(available_nodes, request_context)

        elif self.strategy == LoadBalancingStrategy.GEOGRAPHIC:
            return self._geographic_selection(available_nodes, request_context)

        else:
            return available_nodes[0]  # Fallback

    def _round_robin_selection(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Round-robin node selection"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return node

    def _least_connections_selection(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Select node with least connections"""
        return min(nodes, key=lambda n: n.current_connections)

    def _weighted_round_robin_selection(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Weighted round-robin selection"""
        # Simple weighted selection based on node weights
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return nodes[0]

        # Normalize and select based on weights
        weights = [node.weight / total_weight for node in nodes]
        import random
        return random.choices(nodes, weights=weights)[0]

    def _least_response_time_selection(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Select node with lowest average response time"""
        def avg_response_time(node_id: str) -> float:
            times = self.response_times[node_id]
            return mean(times) if times else 0.0

        return min(nodes, key=lambda n: avg_response_time(n.node_id))

    def _resource_based_selection(self, nodes: List[ServiceNode]) -> ServiceNode:
        """Select node based on resource utilization"""
        return min(nodes, key=lambda n: n.calculate_load_score(self.strategy))

    def _consistent_hash_selection(self, nodes: List[ServiceNode], context: Optional[Dict[str, Any]]) -> ServiceNode:
        """Consistent hash-based selection"""
        if not context or 'session_id' not in context:
            return self._least_connections_selection(nodes)

        # Simple hash-based selection
        session_id = context['session_id']
        hash_value = hash(session_id) % len(nodes)
        return nodes[hash_value]

    def _geographic_selection(self, nodes: List[ServiceNode], context: Optional[Dict[str, Any]]) -> ServiceNode:
        """Geographic proximity-based selection"""
        if not context or 'region' not in context:
            return self._least_connections_selection(nodes)

        preferred_region = context['region']

        # Prefer nodes in the same region
        same_region_nodes = [n for n in nodes if n.region == preferred_region]
        if same_region_nodes:
            return self._least_connections_selection(same_region_nodes)

        # Fallback to any available node
        return self._least_connections_selection(nodes)

    def record_request(self, node_id: str, response_time_ms: float) -> None:
        """Record request metrics for a node"""
        self.request_counts[node_id] += 1
        self.response_times[node_id].append(response_time_ms)

        # Update node connections
        if node_id in self.nodes:
            self.nodes[node_id].current_connections += 1

    def release_connection(self, node_id: str) -> None:
        """Release a connection from a node"""
        if node_id in self.nodes and self.nodes[node_id].current_connections > 0:
            self.nodes[node_id].current_connections -= 1

    def get_load_distribution(self) -> Dict[str, Dict[str, Any]]:
        """Get current load distribution across nodes"""
        distribution = {}

        for node_id, node in self.nodes.items():
            avg_response_time = 0.0
            if self.response_times[node_id]:
                avg_response_time = mean(self.response_times[node_id])

            distribution[node_id] = {
                "name": node.name,
                "status": node.status,
                "current_connections": node.current_connections,
                "max_connections": node.max_connections,
                "utilization": node.current_connections / node.max_connections if node.max_connections > 0 else 0,
                "request_count": self.request_counts[node_id],
                "average_response_time": avg_response_time,
                "current_load": node.current_load,
                "weight": node.weight
            }

        return distribution


class HealthChecker:
    """Health checking system for nodes"""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self.health_task: Optional[asyncio.Task] = None
        self.health_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)

    async def start(self, load_balancer: LoadBalancer) -> None:
        """Start health checking"""
        if self.running:
            return

        self.running = True
        self.health_task = asyncio.create_task(self._health_check_loop(load_balancer))
        logger.info("Health checker started")

    async def stop(self) -> None:
        """Stop health checking"""
        self.running = False
        if self.health_task:
            self.health_task.cancel()
            try:
                await self.health_task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _health_check_loop(self, load_balancer: LoadBalancer) -> None:
        """Health check loop"""
        while self.running:
            try:
                await self._check_all_nodes(load_balancer)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)

    async def _check_all_nodes(self, load_balancer: LoadBalancer) -> None:
        """Check health of all nodes"""
        tasks = []
        for node in load_balancer.nodes.values():
            task = asyncio.create_task(self._check_node_health(node))
            tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, HealthCheckResult):
                    self._process_health_result(result, load_balancer)
                elif isinstance(result, Exception):
                    logger.error(f"Health check failed: {result}")

    async def _check_node_health(self, node: ServiceNode) -> HealthCheckResult:
        """Check health of a single node"""
        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Check health endpoint
                health_url = f"{node.endpoint}/health"
                async with session.get(health_url) as response:
                    response_time_ms = (time.time() - start_time) * 1000

                    if response.status == 200:
                        data = await response.json()

                        return HealthCheckResult(
                            node_id=node.node_id,
                            endpoint=node.endpoint,
                            status=NodeStatus.HEALTHY,
                            response_time_ms=response_time_ms,
                            cpu_usage=data.get('cpu_usage', 0.0),
                            memory_usage=data.get('memory_usage', 0.0),
                            disk_usage=data.get('disk_usage', 0.0),
                            active_connections=data.get('active_connections', 0),
                            error_rate=data.get('error_rate', 0.0),
                            details=data
                        )
                    else:
                        return HealthCheckResult(
                            node_id=node.node_id,
                            endpoint=node.endpoint,
                            status=NodeStatus.DEGRADED,
                            response_time_ms=response_time_ms,
                            cpu_usage=0.0,
                            memory_usage=0.0,
                            disk_usage=0.0,
                            active_connections=0,
                            error_rate=1.0,
                            details={"http_status": response.status}
                        )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                node_id=node.node_id,
                endpoint=node.endpoint,
                status=NodeStatus.OFFLINE,
                response_time_ms=response_time_ms,
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                active_connections=0,
                error_rate=1.0,
                details={"error": str(e)}
            )

    def _process_health_result(self, result: HealthCheckResult, load_balancer: LoadBalancer) -> None:
        """Process health check result and update node status"""
        node = load_balancer.nodes.get(result.node_id)
        if not node:
            return

        # Store health history
        self.health_history[result.node_id].append(result)

        # Keep only recent history (last 100 checks)
        if len(self.health_history[result.node_id]) > 100:
            self.health_history[result.node_id] = self.health_history[result.node_id][-100:]

        # Update node status
        previous_status = node.status
        node.status = result.status
        node.last_health_check = result.timestamp
        node.current_load = max(result.cpu_usage, result.memory_usage) / 100.0

        # Track consecutive failures
        if result.status == NodeStatus.OFFLINE:
            node.consecutive_failures += 1
        else:
            node.consecutive_failures = 0

        # Log status changes
        if previous_status != node.status:
            logger.info(f"Node {node.name} status changed: {previous_status} -> {node.status}")

        # Determine if node should be marked as overloaded
        if result.cpu_usage > 90 or result.memory_usage > 90 or result.error_rate > 0.1:
            node.status = NodeStatus.OVERLOADED

    def get_node_health_summary(self, node_id: str) -> Dict[str, Any]:
        """Get health summary for a node"""
        history = self.health_history.get(node_id, [])
        if not history:
            return {}

        recent_checks = history[-10:]  # Last 10 checks

        return {
            "node_id": node_id,
            "total_checks": len(history),
            "recent_checks": len(recent_checks),
            "current_status": recent_checks[-1].status if recent_checks else None,
            "average_response_time": mean([h.response_time_ms for h in recent_checks]),
            "average_cpu_usage": mean([h.cpu_usage for h in recent_checks]),
            "average_memory_usage": mean([h.memory_usage for h in recent_checks]),
            "uptime_percentage": len([h for h in recent_checks if h.status != NodeStatus.OFFLINE]) / len(recent_checks) * 100,
            "last_check": recent_checks[-1].timestamp if recent_checks else None
        }


class AutoScaler:
    """Auto-scaling system for dynamic resource management"""

    def __init__(self):
        self.enabled = True
        self.min_nodes = 2
        self.max_nodes = 20
        self.scale_up_threshold = 0.8  # CPU/memory threshold for scaling up
        self.scale_down_threshold = 0.3  # CPU/memory threshold for scaling down
        self.scale_up_cooldown = 300  # 5 minutes
        self.scale_down_cooldown = 600  # 10 minutes
        self.last_scale_action: Optional[datetime] = None
        self.scaling_history: List[Dict[str, Any]] = []

    async def evaluate_scaling(self, load_balancer: LoadBalancer, health_checker: HealthChecker) -> ScalingAction:
        """Evaluate if scaling action is needed"""
        if not self.enabled:
            return ScalingAction.NONE

        # Check cooldown period
        if self.last_scale_action:
            time_since_last = (datetime.utcnow() - self.last_scale_action).total_seconds()
            if time_since_last < self.scale_down_cooldown:  # Use longer cooldown
                return ScalingAction.NONE

        available_nodes = load_balancer.get_available_nodes()
        if not available_nodes:
            return ScalingAction.SCALE_OUT  # Emergency scaling

        # Calculate average metrics
        avg_cpu = mean([node.current_load for node in available_nodes])
        avg_connections = mean([node.current_connections / node.max_connections for node in available_nodes])
        avg_load = max(avg_cpu, avg_connections)

        # Determine scaling action
        if avg_load > self.scale_up_threshold and len(available_nodes) < self.max_nodes:
            return ScalingAction.SCALE_OUT

        elif avg_load < self.scale_down_threshold and len(available_nodes) > self.min_nodes:
            return ScalingAction.SCALE_IN

        # Check if rebalancing is needed
        load_variance = max([node.current_load for node in available_nodes]) - min([node.current_load for node in available_nodes])
        if load_variance > 0.4:  # 40% variance threshold
            return ScalingAction.REBALANCE

        return ScalingAction.NONE

    async def execute_scaling_action(self, action: ScalingAction, load_balancer: LoadBalancer) -> bool:
        """Execute scaling action"""
        if action == ScalingAction.NONE:
            return True

        success = False

        if action == ScalingAction.SCALE_OUT:
            success = await self._scale_out(load_balancer)

        elif action == ScalingAction.SCALE_IN:
            success = await self._scale_in(load_balancer)

        elif action == ScalingAction.REBALANCE:
            success = await self._rebalance(load_balancer)

        if success:
            self.last_scale_action = datetime.utcnow()
            self._record_scaling_action(action, success)

        return success

    async def _scale_out(self, load_balancer: LoadBalancer) -> bool:
        """Add new nodes (scale out)"""
        try:
            # This would typically interact with container orchestration (Docker Swarm, Kubernetes)
            # or cloud provider APIs to launch new instances

            # For now, we'll simulate by creating a new node configuration
            new_node = ServiceNode(
                name=f"auto-scaled-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                host="auto-scaled-host",  # Would be dynamically assigned
                port=8000,
                region="default",
                zone="default"
            )

            load_balancer.add_node(new_node)
            logger.info(f"Scaled out: added node {new_node.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to scale out: {e}")
            return False

    async def _scale_in(self, load_balancer: LoadBalancer) -> bool:
        """Remove nodes (scale in)"""
        try:
            available_nodes = load_balancer.get_available_nodes()
            if len(available_nodes) <= self.min_nodes:
                return False

            # Find node with lowest load for removal
            node_to_remove = min(available_nodes, key=lambda n: n.current_load)

            # Gracefully drain connections
            await self._drain_node(node_to_remove, load_balancer)

            # Remove from load balancer
            load_balancer.remove_node(node_to_remove.node_id)
            logger.info(f"Scaled in: removed node {node_to_remove.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to scale in: {e}")
            return False

    async def _rebalance(self, load_balancer: LoadBalancer) -> bool:
        """Rebalance load across nodes"""
        try:
            # Adjust node weights based on current performance
            nodes = load_balancer.get_available_nodes()

            for node in nodes:
                # Adjust weight inversely to current load
                if node.current_load > 0:
                    new_weight = 1.0 / (node.current_load + 0.1)  # Avoid division by zero
                    node.weight = max(0.1, min(2.0, new_weight))  # Clamp between 0.1 and 2.0

            logger.info("Rebalanced node weights")
            return True

        except Exception as e:
            logger.error(f"Failed to rebalance: {e}")
            return False

    async def _drain_node(self, node: ServiceNode, load_balancer: LoadBalancer) -> None:
        """Gracefully drain connections from a node"""
        node.status = NodeStatus.STOPPING

        # Wait for connections to naturally decrease
        max_wait = 60  # 1 minute max wait
        wait_time = 0

        while node.current_connections > 0 and wait_time < max_wait:
            await asyncio.sleep(5)
            wait_time += 5

        if node.current_connections > 0:
            logger.warning(f"Node {node.name} still has {node.current_connections} connections after drain timeout")

    def _record_scaling_action(self, action: ScalingAction, success: bool) -> None:
        """Record scaling action in history"""
        record = {
            "timestamp": datetime.utcnow(),
            "action": action.value,
            "success": success
        }

        self.scaling_history.append(record)

        # Keep only recent history (last 100 actions)
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        recent_actions = [a for a in self.scaling_history if
                         (datetime.utcnow() - a["timestamp"]).total_seconds() < 86400]  # Last 24 hours

        return {
            "enabled": self.enabled,
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "last_scale_action": self.last_scale_action,
            "total_actions": len(self.scaling_history),
            "actions_24h": len(recent_actions),
            "scale_out_24h": len([a for a in recent_actions if a["action"] == "scale_out"]),
            "scale_in_24h": len([a for a in recent_actions if a["action"] == "scale_in"]),
            "rebalance_24h": len([a for a in recent_actions if a["action"] == "rebalance"])
        }


class ScalingManager:
    """Main scaling management system"""

    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.load_balancer = LoadBalancer(strategy)
        self.health_checker = HealthChecker()
        self.auto_scaler = AutoScaler()
        self.running = False
        self.management_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scaling management system"""
        if self.running:
            return

        self.running = True
        await self.health_checker.start(self.load_balancer)

        self.management_task = asyncio.create_task(self._management_loop())
        logger.info("Scaling manager started")

    async def stop(self) -> None:
        """Stop the scaling management system"""
        self.running = False

        if self.management_task:
            self.management_task.cancel()
            try:
                await self.management_task
            except asyncio.CancelledError:
                pass

        await self.health_checker.stop()
        logger.info("Scaling manager stopped")

    async def _management_loop(self) -> None:
        """Main management loop for auto-scaling"""
        while self.running:
            try:
                # Evaluate and execute scaling actions
                action = await self.auto_scaler.evaluate_scaling(self.load_balancer, self.health_checker)
                if action != ScalingAction.NONE:
                    await self.auto_scaler.execute_scaling_action(action, self.load_balancer)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scaling management loop: {e}")
                await asyncio.sleep(30)

    def add_node(self, node: ServiceNode) -> None:
        """Add a node to the system"""
        self.load_balancer.add_node(node)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the system"""
        return self.load_balancer.remove_node(node_id)

    def select_node(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceNode]:
        """Select best node for a request"""
        return self.load_balancer.select_node(request_context)

    def record_request(self, node_id: str, response_time_ms: float) -> None:
        """Record request completion"""
        self.load_balancer.record_request(node_id, response_time_ms)

    def release_connection(self, node_id: str) -> None:
        """Release connection from node"""
        self.load_balancer.release_connection(node_id)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "running": self.running,
            "load_balancer": {
                "strategy": self.load_balancer.strategy.value,
                "total_nodes": len(self.load_balancer.nodes),
                "available_nodes": len(self.load_balancer.get_available_nodes()),
                "load_distribution": self.load_balancer.get_load_distribution()
            },
            "auto_scaler": self.auto_scaler.get_scaling_stats(),
            "health_checker": {
                "check_interval": self.health_checker.check_interval,
                "nodes_monitored": len(self.health_checker.health_history)
            }
        }
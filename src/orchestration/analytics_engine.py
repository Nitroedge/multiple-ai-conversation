"""
Performance Analytics Engine
Advanced metrics collection and optimization recommendations for multi-agent workflows
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COLLABORATION = "collaboration"
    RESOURCE = "resource"
    USER_SATISFACTION = "user_satisfaction"
    COST = "cost"
    RELIABILITY = "reliability"


class TimeWindow(str, Enum):
    """Time windows for analytics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TrendDirection(str, Enum):
    """Trend directions"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class Metric(BaseModel):
    """Individual metric data point"""
    metric_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: MetricType
    value: Union[float, int, str, bool]
    unit: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Context information
    workflow_id: Optional[str] = None
    agent_id: Optional[str] = None
    step_id: Optional[str] = None
    session_id: Optional[str] = None

    # Additional metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "forbid"


class MetricSummary(BaseModel):
    """Summary statistics for a metric over a time period"""
    metric_name: str
    type: MetricType
    time_window: TimeWindow
    start_time: datetime
    end_time: datetime

    # Statistical measures
    count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    min_value: Optional[Union[float, int]] = None
    max_value: Optional[Union[float, int]] = None

    # Percentiles
    p25: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    class Config:
        extra = "forbid"


class TrendAnalysis(BaseModel):
    """Trend analysis results"""
    metric_name: str
    time_window: TimeWindow
    direction: TrendDirection
    confidence: float  # 0-1 scale
    slope: Optional[float] = None
    correlation: Optional[float] = None

    # Change metrics
    absolute_change: Optional[float] = None
    percentage_change: Optional[float] = None

    # Forecast
    next_period_forecast: Optional[float] = None
    forecast_confidence: Optional[float] = None

    class Config:
        extra = "forbid"


class PerformanceAlert(BaseModel):
    """Performance alert"""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str
    severity: AlertSeverity
    message: str
    threshold_value: Union[float, int]
    actual_value: Union[float, int]

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    workflow_id: Optional[str] = None
    agent_id: Optional[str] = None

    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None

    class Config:
        extra = "forbid"


class OptimizationRecommendation(BaseModel):
    """Optimization recommendation"""
    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    priority: str  # "high", "medium", "low"
    category: str

    # Impact estimation
    estimated_improvement: Optional[float] = None
    estimated_effort: str  # "low", "medium", "high"
    estimated_timeline: Optional[str] = None

    # Implementation details
    implementation_steps: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)

    # Metadata
    affected_workflows: List[str] = Field(default_factory=list)
    affected_agents: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # "pending", "in_progress", "completed", "dismissed"

    class Config:
        extra = "forbid"


class MetricsCollector:
    """Collects and stores performance metrics"""

    def __init__(self, max_metrics_per_type: int = 10000):
        self.max_metrics_per_type = max_metrics_per_type
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_type))
        self.metric_schemas: Dict[str, Dict[str, Any]] = {}

    async def collect_metric(self, metric: Metric) -> None:
        """Collect a single metric"""
        try:
            self.metrics[metric.name].append(metric)

            # Update schema if needed
            if metric.name not in self.metric_schemas:
                self.metric_schemas[metric.name] = {
                    "type": metric.type,
                    "unit": metric.unit,
                    "value_type": type(metric.value).__name__
                }

            logger.debug(f"Collected metric: {metric.name} = {metric.value}")

        except Exception as e:
            logger.error(f"Failed to collect metric {metric.name}: {e}")

    async def collect_metrics_batch(self, metrics: List[Metric]) -> None:
        """Collect multiple metrics efficiently"""
        try:
            for metric in metrics:
                await self.collect_metric(metric)
            logger.info(f"Collected batch of {len(metrics)} metrics")
        except Exception as e:
            logger.error(f"Failed to collect metrics batch: {e}")

    async def collect_workflow_metrics(self, workflow_id: str, metrics_data: Dict[str, Any]) -> None:
        """Collect workflow-specific metrics"""
        timestamp = datetime.utcnow()

        workflow_metrics = [
            Metric(
                name="workflow_duration",
                type=MetricType.PERFORMANCE,
                value=metrics_data.get("duration_seconds", 0),
                unit="seconds",
                workflow_id=workflow_id,
                timestamp=timestamp
            ),
            Metric(
                name="workflow_success_rate",
                type=MetricType.QUALITY,
                value=1.0 if metrics_data.get("success", False) else 0.0,
                unit="ratio",
                workflow_id=workflow_id,
                timestamp=timestamp
            ),
            Metric(
                name="workflow_steps_completed",
                type=MetricType.EFFICIENCY,
                value=metrics_data.get("steps_completed", 0),
                unit="count",
                workflow_id=workflow_id,
                timestamp=timestamp
            ),
            Metric(
                name="workflow_agents_used",
                type=MetricType.RESOURCE,
                value=metrics_data.get("agents_used", 0),
                unit="count",
                workflow_id=workflow_id,
                timestamp=timestamp
            )
        ]

        await self.collect_metrics_batch(workflow_metrics)

    async def collect_agent_metrics(self, agent_id: str, metrics_data: Dict[str, Any]) -> None:
        """Collect agent-specific metrics"""
        timestamp = datetime.utcnow()

        agent_metrics = [
            Metric(
                name="agent_response_time",
                type=MetricType.PERFORMANCE,
                value=metrics_data.get("response_time_ms", 0),
                unit="milliseconds",
                agent_id=agent_id,
                timestamp=timestamp
            ),
            Metric(
                name="agent_task_success_rate",
                type=MetricType.QUALITY,
                value=1.0 if metrics_data.get("task_success", False) else 0.0,
                unit="ratio",
                agent_id=agent_id,
                timestamp=timestamp
            ),
            Metric(
                name="agent_memory_usage",
                type=MetricType.RESOURCE,
                value=metrics_data.get("memory_usage_mb", 0),
                unit="megabytes",
                agent_id=agent_id,
                timestamp=timestamp
            ),
            Metric(
                name="agent_collaboration_score",
                type=MetricType.COLLABORATION,
                value=metrics_data.get("collaboration_score", 0.5),
                unit="score",
                agent_id=agent_id,
                timestamp=timestamp
            )
        ]

        await self.collect_metrics_batch(agent_metrics)

    def get_metrics(self, metric_name: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[Metric]:
        """Get metrics by name and time range"""
        metrics = list(self.metrics.get(metric_name, []))

        # Filter by time range
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            metrics = filtered_metrics

        # Apply limit
        if limit:
            metrics = metrics[-limit:]

        return metrics

    def get_metric_names(self) -> List[str]:
        """Get all available metric names"""
        return list(self.metrics.keys())


class TrendAnalyzer:
    """Analyzes trends in metrics data"""

    def __init__(self, min_data_points: int = 5):
        self.min_data_points = min_data_points

    async def analyze_trend(self, metric_name: str,
                          metrics: List[Metric],
                          time_window: TimeWindow) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric"""
        if len(metrics) < self.min_data_points:
            logger.warning(f"Insufficient data points for trend analysis: {len(metrics)}")
            return None

        try:
            # Extract numeric values and timestamps
            values = []
            timestamps = []

            for metric in metrics:
                if isinstance(metric.value, (int, float)):
                    values.append(float(metric.value))
                    timestamps.append(metric.timestamp.timestamp())

            if len(values) < self.min_data_points:
                return None

            # Convert to numpy arrays for analysis
            values_array = np.array(values)
            timestamps_array = np.array(timestamps)

            # Linear regression for trend
            slope, intercept = np.polyfit(timestamps_array, values_array, 1)
            correlation = np.corrcoef(timestamps_array, values_array)[0, 1]

            # Determine trend direction
            direction = self._determine_trend_direction(slope, correlation)
            confidence = abs(correlation) if not np.isnan(correlation) else 0.0

            # Calculate changes
            if len(values) >= 2:
                absolute_change = values[-1] - values[0]
                percentage_change = (absolute_change / values[0] * 100) if values[0] != 0 else 0
            else:
                absolute_change = 0
                percentage_change = 0

            # Simple forecast (linear extrapolation)
            next_timestamp = timestamps_array[-1] + self._get_time_window_seconds(time_window)
            next_forecast = slope * next_timestamp + intercept
            forecast_confidence = confidence * 0.8  # Reduce confidence for forecast

            return TrendAnalysis(
                metric_name=metric_name,
                time_window=time_window,
                direction=direction,
                confidence=confidence,
                slope=slope,
                correlation=correlation,
                absolute_change=absolute_change,
                percentage_change=percentage_change,
                next_period_forecast=next_forecast,
                forecast_confidence=forecast_confidence
            )

        except Exception as e:
            logger.error(f"Failed to analyze trend for {metric_name}: {e}")
            return None

    def _determine_trend_direction(self, slope: float, correlation: float) -> TrendDirection:
        """Determine trend direction based on slope and correlation"""
        if abs(correlation) < 0.3:
            return TrendDirection.VOLATILE
        elif abs(slope) < 0.01:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING

    def _get_time_window_seconds(self, time_window: TimeWindow) -> int:
        """Get time window in seconds"""
        window_seconds = {
            TimeWindow.HOUR: 3600,
            TimeWindow.DAY: 86400,
            TimeWindow.WEEK: 604800,
            TimeWindow.MONTH: 2592000,
            TimeWindow.QUARTER: 7776000,
            TimeWindow.YEAR: 31536000
        }
        return window_seconds.get(time_window, 86400)


class AnalyticsReporter:
    """Generates analytics reports and insights"""

    def __init__(self, metrics_collector: MetricsCollector, trend_analyzer: TrendAnalyzer):
        self.metrics_collector = metrics_collector
        self.trend_analyzer = trend_analyzer
        self.alert_thresholds = self._initialize_alert_thresholds()

    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds"""
        return {
            "workflow_duration": {
                "warning": 300,  # 5 minutes
                "critical": 600  # 10 minutes
            },
            "workflow_success_rate": {
                "warning": 0.8,
                "critical": 0.6
            },
            "agent_response_time": {
                "warning": 5000,  # 5 seconds
                "critical": 10000  # 10 seconds
            },
            "agent_memory_usage": {
                "warning": 1000,  # 1GB
                "critical": 2000  # 2GB
            }
        }

    async def generate_metric_summary(self, metric_name: str,
                                    time_window: TimeWindow,
                                    end_time: Optional[datetime] = None) -> Optional[MetricSummary]:
        """Generate summary statistics for a metric"""
        if not end_time:
            end_time = datetime.utcnow()

        start_time = self._get_window_start_time(end_time, time_window)
        metrics = self.metrics_collector.get_metrics(metric_name, start_time, end_time)

        if not metrics:
            return None

        # Extract numeric values
        numeric_values = []
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                numeric_values.append(float(metric.value))

        if not numeric_values:
            return None

        try:
            # Calculate statistics
            values_array = np.array(numeric_values)

            summary = MetricSummary(
                metric_name=metric_name,
                type=metrics[0].type,
                time_window=time_window,
                start_time=start_time,
                end_time=end_time,
                count=len(numeric_values),
                mean=float(np.mean(values_array)),
                median=float(np.median(values_array)),
                std_dev=float(np.std(values_array)),
                min_value=float(np.min(values_array)),
                max_value=float(np.max(values_array)),
                p25=float(np.percentile(values_array, 25)),
                p75=float(np.percentile(values_array, 75)),
                p90=float(np.percentile(values_array, 90)),
                p95=float(np.percentile(values_array, 95)),
                p99=float(np.percentile(values_array, 99))
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary for {metric_name}: {e}")
            return None

    async def check_alerts(self, metric_name: str) -> List[PerformanceAlert]:
        """Check for performance alerts"""
        alerts = []

        if metric_name not in self.alert_thresholds:
            return alerts

        thresholds = self.alert_thresholds[metric_name]
        recent_metrics = self.metrics_collector.get_metrics(metric_name, limit=10)

        if not recent_metrics:
            return alerts

        # Check latest metric value against thresholds
        latest_metric = recent_metrics[-1]
        if not isinstance(latest_metric.value, (int, float)):
            return alerts

        value = float(latest_metric.value)

        # Check critical threshold
        if "critical" in thresholds:
            if (metric_name in ["workflow_success_rate", "agent_task_success_rate"] and value < thresholds["critical"]) or \
               (metric_name not in ["workflow_success_rate", "agent_task_success_rate"] and value > thresholds["critical"]):
                alerts.append(PerformanceAlert(
                    metric_name=metric_name,
                    severity=AlertSeverity.CRITICAL,
                    message=f"{metric_name} is at critical level: {value}",
                    threshold_value=thresholds["critical"],
                    actual_value=value,
                    workflow_id=latest_metric.workflow_id,
                    agent_id=latest_metric.agent_id
                ))

        # Check warning threshold
        elif "warning" in thresholds:
            if (metric_name in ["workflow_success_rate", "agent_task_success_rate"] and value < thresholds["warning"]) or \
               (metric_name not in ["workflow_success_rate", "agent_task_success_rate"] and value > thresholds["warning"]):
                alerts.append(PerformanceAlert(
                    metric_name=metric_name,
                    severity=AlertSeverity.WARNING,
                    message=f"{metric_name} is at warning level: {value}",
                    threshold_value=thresholds["warning"],
                    actual_value=value,
                    workflow_id=latest_metric.workflow_id,
                    agent_id=latest_metric.agent_id
                ))

        return alerts

    async def generate_comprehensive_report(self, time_window: TimeWindow = TimeWindow.DAY) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        report = {
            "timestamp": datetime.utcnow(),
            "time_window": time_window,
            "metrics_summary": {},
            "trends": {},
            "alerts": [],
            "recommendations": []
        }

        # Generate summaries for all metrics
        metric_names = self.metrics_collector.get_metric_names()

        for metric_name in metric_names:
            summary = await self.generate_metric_summary(metric_name, time_window)
            if summary:
                report["metrics_summary"][metric_name] = summary

            # Analyze trends
            metrics = self.metrics_collector.get_metrics(metric_name)
            trend = await self.trend_analyzer.analyze_trend(metric_name, metrics, time_window)
            if trend:
                report["trends"][metric_name] = trend

            # Check alerts
            alerts = await self.check_alerts(metric_name)
            report["alerts"].extend(alerts)

        return report

    def _get_window_start_time(self, end_time: datetime, time_window: TimeWindow) -> datetime:
        """Get start time for time window"""
        if time_window == TimeWindow.HOUR:
            return end_time - timedelta(hours=1)
        elif time_window == TimeWindow.DAY:
            return end_time - timedelta(days=1)
        elif time_window == TimeWindow.WEEK:
            return end_time - timedelta(weeks=1)
        elif time_window == TimeWindow.MONTH:
            return end_time - timedelta(days=30)
        elif time_window == TimeWindow.QUARTER:
            return end_time - timedelta(days=90)
        elif time_window == TimeWindow.YEAR:
            return end_time - timedelta(days=365)
        else:
            return end_time - timedelta(days=1)


class PerformanceAnalyticsEngine:
    """Main analytics engine coordinating all components"""

    def __init__(self, max_metrics_per_type: int = 10000):
        self.metrics_collector = MetricsCollector(max_metrics_per_type)
        self.trend_analyzer = TrendAnalyzer()
        self.analytics_reporter = AnalyticsReporter(self.metrics_collector, self.trend_analyzer)

        self.recommendations_cache: List[OptimizationRecommendation] = []
        self.running = False
        self.analysis_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the analytics engine"""
        if self.running:
            return

        self.running = True
        self.analysis_task = asyncio.create_task(self._continuous_analysis())
        logger.info("Performance Analytics Engine started")

    async def stop(self) -> None:
        """Stop the analytics engine"""
        self.running = False
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance Analytics Engine stopped")

    async def _continuous_analysis(self) -> None:
        """Continuous analysis loop"""
        while self.running:
            try:
                await self._run_analysis_cycle()
                await asyncio.sleep(300)  # Run every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _run_analysis_cycle(self) -> None:
        """Run one analysis cycle"""
        # Generate recommendations based on current metrics
        recommendations = await self._generate_optimization_recommendations()

        # Update recommendations cache
        self.recommendations_cache.extend(recommendations)

        # Clean up old recommendations
        self._cleanup_old_recommendations()

    async def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current data"""
        recommendations = []

        # Analyze workflow performance
        workflow_recommendations = await self._analyze_workflow_performance()
        recommendations.extend(workflow_recommendations)

        # Analyze agent performance
        agent_recommendations = await self._analyze_agent_performance()
        recommendations.extend(agent_recommendations)

        # Analyze resource utilization
        resource_recommendations = await self._analyze_resource_utilization()
        recommendations.extend(resource_recommendations)

        # Analyze collaboration patterns
        collaboration_recommendations = await self._analyze_collaboration_patterns()
        recommendations.extend(collaboration_recommendations)

        return recommendations

    async def _analyze_workflow_performance(self) -> List[OptimizationRecommendation]:
        """Analyze workflow performance and generate recommendations"""
        recommendations = []

        # Check workflow duration trends
        duration_metrics = self.metrics_collector.get_metrics("workflow_duration", limit=100)
        if duration_metrics:
            trend = await self.trend_analyzer.analyze_trend("workflow_duration", duration_metrics, TimeWindow.DAY)

            if trend and trend.direction == TrendDirection.DECLINING and trend.confidence > 0.7:
                recommendations.append(OptimizationRecommendation(
                    title="Workflow Duration Increasing",
                    description="Workflow execution times are trending upward, indicating potential performance degradation.",
                    priority="high",
                    category="performance",
                    estimated_improvement=20.0,
                    estimated_effort="medium",
                    estimated_timeline="1-2 weeks",
                    implementation_steps=[
                        "Profile workflow execution to identify bottlenecks",
                        "Optimize slow-performing steps",
                        "Consider parallel execution where possible",
                        "Review agent allocation and load balancing"
                    ],
                    success_criteria=[
                        "Reduce average workflow duration by 15%",
                        "Stabilize performance trends"
                    ],
                    tags=["performance", "workflow", "optimization"]
                ))

        # Check success rate trends
        success_metrics = self.metrics_collector.get_metrics("workflow_success_rate", limit=100)
        if success_metrics:
            recent_success_rate = mean([m.value for m in success_metrics[-10:] if isinstance(m.value, (int, float))])

            if recent_success_rate < 0.9:
                recommendations.append(OptimizationRecommendation(
                    title="Low Workflow Success Rate",
                    description=f"Recent workflow success rate is {recent_success_rate:.1%}, below optimal threshold.",
                    priority="critical",
                    category="reliability",
                    estimated_improvement=15.0,
                    estimated_effort="high",
                    estimated_timeline="2-4 weeks",
                    implementation_steps=[
                        "Analyze failure patterns and root causes",
                        "Implement better error handling and recovery",
                        "Add validation steps at critical points",
                        "Improve agent coordination mechanisms"
                    ],
                    success_criteria=[
                        "Achieve >95% workflow success rate",
                        "Reduce failure cascades"
                    ],
                    tags=["reliability", "workflow", "error-handling"]
                ))

        return recommendations

    async def _analyze_agent_performance(self) -> List[OptimizationRecommendation]:
        """Analyze agent performance and generate recommendations"""
        recommendations = []

        # Check agent response times
        response_metrics = self.metrics_collector.get_metrics("agent_response_time", limit=100)
        if response_metrics:
            recent_avg_response = mean([m.value for m in response_metrics[-20:] if isinstance(m.value, (int, float))])

            if recent_avg_response > 3000:  # 3 seconds
                recommendations.append(OptimizationRecommendation(
                    title="High Agent Response Times",
                    description=f"Average agent response time is {recent_avg_response:.0f}ms, impacting user experience.",
                    priority="medium",
                    category="performance",
                    estimated_improvement=30.0,
                    estimated_effort="medium",
                    estimated_timeline="1-3 weeks",
                    implementation_steps=[
                        "Profile agent processing to identify bottlenecks",
                        "Optimize model inference and caching",
                        "Implement response streaming",
                        "Consider agent load balancing"
                    ],
                    success_criteria=[
                        "Reduce average response time to <2 seconds",
                        "Maintain consistent response times"
                    ],
                    tags=["performance", "agent", "response-time"]
                ))

        # Check memory usage
        memory_metrics = self.metrics_collector.get_metrics("agent_memory_usage", limit=100)
        if memory_metrics:
            trend = await self.trend_analyzer.analyze_trend("agent_memory_usage", memory_metrics, TimeWindow.DAY)

            if trend and trend.direction == TrendDirection.DECLINING and trend.confidence > 0.6:
                recommendations.append(OptimizationRecommendation(
                    title="Increasing Agent Memory Usage",
                    description="Agent memory usage is trending upward, potentially indicating memory leaks.",
                    priority="medium",
                    category="resource",
                    estimated_improvement=25.0,
                    estimated_effort="medium",
                    estimated_timeline="1-2 weeks",
                    implementation_steps=[
                        "Profile memory usage patterns",
                        "Implement memory cleanup routines",
                        "Optimize data structures and caching",
                        "Add memory monitoring and alerts"
                    ],
                    success_criteria=[
                        "Stabilize memory usage trends",
                        "Reduce peak memory usage by 20%"
                    ],
                    tags=["resource", "memory", "optimization"]
                ))

        return recommendations

    async def _analyze_resource_utilization(self) -> List[OptimizationRecommendation]:
        """Analyze resource utilization and generate recommendations"""
        recommendations = []

        # Check agent utilization patterns
        agent_metrics = self.metrics_collector.get_metrics("workflow_agents_used", limit=50)
        if agent_metrics:
            avg_agents = mean([m.value for m in agent_metrics if isinstance(m.value, (int, float))])

            if avg_agents < 2:
                recommendations.append(OptimizationRecommendation(
                    title="Underutilized Multi-Agent Capabilities",
                    description="Workflows are using fewer agents than optimal, missing collaboration benefits.",
                    priority="low",
                    category="efficiency",
                    estimated_improvement=15.0,
                    estimated_effort="low",
                    estimated_timeline="1 week",
                    implementation_steps=[
                        "Review workflow designs for parallelization opportunities",
                        "Implement parallel task execution",
                        "Add specialized agent roles",
                        "Optimize agent coordination"
                    ],
                    success_criteria=[
                        "Increase average agents per workflow",
                        "Improve task parallelization"
                    ],
                    tags=["efficiency", "collaboration", "parallelization"]
                ))

        return recommendations

    async def _analyze_collaboration_patterns(self) -> List[OptimizationRecommendation]:
        """Analyze collaboration patterns and generate recommendations"""
        recommendations = []

        # Check collaboration scores
        collab_metrics = self.metrics_collector.get_metrics("agent_collaboration_score", limit=100)
        if collab_metrics:
            avg_collab_score = mean([m.value for m in collab_metrics if isinstance(m.value, (int, float))])

            if avg_collab_score < 0.7:
                recommendations.append(OptimizationRecommendation(
                    title="Low Agent Collaboration Effectiveness",
                    description=f"Agent collaboration score is {avg_collab_score:.1f}, below optimal threshold.",
                    priority="medium",
                    category="collaboration",
                    estimated_improvement=20.0,
                    estimated_effort="medium",
                    estimated_timeline="2-3 weeks",
                    implementation_steps=[
                        "Analyze communication patterns between agents",
                        "Improve inter-agent communication protocols",
                        "Implement better conflict resolution mechanisms",
                        "Add collaboration training data"
                    ],
                    success_criteria=[
                        "Achieve >0.8 collaboration score",
                        "Reduce agent conflicts"
                    ],
                    tags=["collaboration", "communication", "coordination"]
                ))

        return recommendations

    def _cleanup_old_recommendations(self) -> None:
        """Clean up old and completed recommendations"""
        cutoff_time = datetime.utcnow() - timedelta(days=30)

        self.recommendations_cache = [
            rec for rec in self.recommendations_cache
            if rec.created_at > cutoff_time and rec.status != "completed"
        ]

    # Public API methods

    async def collect_metric(self, metric: Metric) -> None:
        """Collect a single metric"""
        await self.metrics_collector.collect_metric(metric)

    async def collect_workflow_metrics(self, workflow_id: str, metrics_data: Dict[str, Any]) -> None:
        """Collect workflow metrics"""
        await self.metrics_collector.collect_workflow_metrics(workflow_id, metrics_data)

    async def collect_agent_metrics(self, agent_id: str, metrics_data: Dict[str, Any]) -> None:
        """Collect agent metrics"""
        await self.metrics_collector.collect_agent_metrics(agent_id, metrics_data)

    async def get_metric_summary(self, metric_name: str, time_window: TimeWindow = TimeWindow.DAY) -> Optional[MetricSummary]:
        """Get metric summary"""
        return await self.analytics_reporter.generate_metric_summary(metric_name, time_window)

    async def get_trend_analysis(self, metric_name: str, time_window: TimeWindow = TimeWindow.DAY) -> Optional[TrendAnalysis]:
        """Get trend analysis for a metric"""
        metrics = self.metrics_collector.get_metrics(metric_name)
        return await self.trend_analyzer.analyze_trend(metric_name, metrics, time_window)

    async def get_performance_alerts(self) -> List[PerformanceAlert]:
        """Get current performance alerts"""
        all_alerts = []
        for metric_name in self.metrics_collector.get_metric_names():
            alerts = await self.analytics_reporter.check_alerts(metric_name)
            all_alerts.extend(alerts)
        return all_alerts

    async def get_optimization_recommendations(self,
                                            category: Optional[str] = None,
                                            priority: Optional[str] = None,
                                            limit: Optional[int] = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations"""
        recommendations = self.recommendations_cache.copy()

        # Filter by category
        if category:
            recommendations = [r for r in recommendations if r.category == category]

        # Filter by priority
        if priority:
            recommendations = [r for r in recommendations if r.priority == priority]

        # Sort by priority and creation time
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: (priority_order.get(x.priority, 4), x.created_at), reverse=True)

        # Apply limit
        if limit:
            recommendations = recommendations[:limit]

        return recommendations

    async def generate_comprehensive_report(self, time_window: TimeWindow = TimeWindow.DAY) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        report = await self.analytics_reporter.generate_comprehensive_report(time_window)

        # Add recommendations to report
        report["recommendations"] = await self.get_optimization_recommendations(limit=10)

        return report

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "running": self.running,
            "metrics_collected": sum(len(metrics) for metrics in self.metrics_collector.metrics.values()),
            "metric_types": len(self.metrics_collector.metrics),
            "recommendations_pending": len([r for r in self.recommendations_cache if r.status == "pending"]),
            "last_analysis": datetime.utcnow() if self.running else None
        }
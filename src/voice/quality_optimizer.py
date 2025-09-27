"""
Voice quality optimization system for adaptive quality management
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
import statistics
from collections import deque

from pydantic import BaseModel, Field

from .stt_processor import STTModel, TranscriptionQuality
from .tts_processor import TTSQuality, AudioFormat
from .audio_streaming import StreamingQuality

logger = logging.getLogger(__name__)


class QualityMetric(str, Enum):
    """Quality metrics for optimization"""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    AUDIO_QUALITY = "audio_quality"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_LOAD = "system_load"
    NETWORK_BANDWIDTH = "network_bandwidth"
    ERROR_RATE = "error_rate"


class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCE_QUALITY_SPEED = "balance_quality_speed"
    MINIMIZE_BANDWIDTH = "minimize_bandwidth"
    ADAPTIVE = "adaptive"


class QualityLevel(str, Enum):
    """Quality levels for adaptive optimization"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class QualityMeasurement:
    """Single quality measurement"""
    metric: QualityMetric
    value: float
    timestamp: float
    context: Optional[Dict[str, Any]] = None


@dataclass
class SystemPerformance:
    """System performance snapshot"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    network_latency: float
    available_bandwidth: float
    active_connections: int
    timestamp: float


class QualityThresholds(BaseModel):
    """Quality thresholds for optimization decisions"""
    max_latency: float = Field(default=2.0, description="Maximum acceptable latency in seconds")
    min_accuracy: float = Field(default=0.8, description="Minimum transcription accuracy")
    max_error_rate: float = Field(default=0.05, description="Maximum error rate")
    min_audio_quality: float = Field(default=0.7, description="Minimum audio quality score")
    max_cpu_usage: float = Field(default=80.0, description="Maximum CPU usage percentage")
    max_memory_usage: float = Field(default=80.0, description="Maximum memory usage percentage")
    min_bandwidth: float = Field(default=100.0, description="Minimum bandwidth in KB/s")


class AdaptiveConfiguration(BaseModel):
    """Adaptive configuration settings"""
    stt_model: STTModel = STTModel.WHISPER_BASE
    stt_quality: TranscriptionQuality = TranscriptionQuality.BALANCED
    tts_quality: TTSQuality = TTSQuality.BALANCED
    audio_format: AudioFormat = AudioFormat.MP3
    streaming_quality: StreamingQuality = StreamingQuality.BALANCED
    sample_rate: int = 16000
    enable_noise_reduction: bool = True
    enable_compression: bool = True
    optimization_level: int = 1  # 0-3 scale


class VoiceQualityOptimizer:
    """Adaptive voice quality optimization system"""

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()

        # Quality measurements history
        self.measurements: Dict[QualityMetric, deque] = {
            metric: deque(maxlen=100) for metric in QualityMetric
        }

        # System performance history
        self.performance_history: deque = deque(maxlen=50)

        # Current configuration
        self.current_config = AdaptiveConfiguration()

        # Optimization state
        self.optimization_strategy = OptimizationStrategy.ADAPTIVE
        self.is_optimizing = False
        self.last_optimization_time = 0.0

        # Quality mapping tables
        self.quality_mappings = self._initialize_quality_mappings()

        # Callbacks for configuration changes
        self.config_change_callbacks: List[Callable[[AdaptiveConfiguration], None]] = []

    def _initialize_quality_mappings(self) -> Dict[str, Dict]:
        """Initialize quality level mappings"""
        return {
            "stt_models": {
                QualityLevel.VERY_LOW: STTModel.WHISPER_TINY,
                QualityLevel.LOW: STTModel.WHISPER_BASE,
                QualityLevel.MEDIUM: STTModel.WHISPER_SMALL,
                QualityLevel.HIGH: STTModel.WHISPER_MEDIUM,
                QualityLevel.VERY_HIGH: STTModel.WHISPER_LARGE_V3
            },
            "tts_quality": {
                QualityLevel.VERY_LOW: TTSQuality.FAST,
                QualityLevel.LOW: TTSQuality.FAST,
                QualityLevel.MEDIUM: TTSQuality.BALANCED,
                QualityLevel.HIGH: TTSQuality.BALANCED,
                QualityLevel.VERY_HIGH: TTSQuality.HIGH_QUALITY
            },
            "streaming_quality": {
                QualityLevel.VERY_LOW: StreamingQuality.LOW_LATENCY,
                QualityLevel.LOW: StreamingQuality.LOW_LATENCY,
                QualityLevel.MEDIUM: StreamingQuality.BALANCED,
                QualityLevel.HIGH: StreamingQuality.BALANCED,
                QualityLevel.VERY_HIGH: StreamingQuality.HIGH_QUALITY
            },
            "sample_rates": {
                QualityLevel.VERY_LOW: 8000,
                QualityLevel.LOW: 16000,
                QualityLevel.MEDIUM: 16000,
                QualityLevel.HIGH: 22050,
                QualityLevel.VERY_HIGH: 44100
            }
        }

    def add_measurement(
        self,
        metric: QualityMetric,
        value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add quality measurement"""
        measurement = QualityMeasurement(
            metric=metric,
            value=value,
            timestamp=time.time(),
            context=context
        )

        self.measurements[metric].append(measurement)

        # Trigger optimization if needed
        if self._should_optimize():
            asyncio.create_task(self.optimize_quality())

    def add_performance_snapshot(self, performance: SystemPerformance) -> None:
        """Add system performance snapshot"""
        self.performance_history.append(performance)

        # Trigger optimization if performance degraded significantly
        if self._performance_degraded():
            asyncio.create_task(self.optimize_quality())

    def _should_optimize(self) -> bool:
        """Check if optimization should be triggered"""
        current_time = time.time()

        # Don't optimize too frequently
        if current_time - self.last_optimization_time < 10.0:
            return False

        # Check if any thresholds are violated
        violations = self._check_threshold_violations()
        return len(violations) > 0

    def _performance_degraded(self) -> bool:
        """Check if system performance has degraded"""
        if len(self.performance_history) < 5:
            return False

        recent_performance = list(self.performance_history)[-5:]
        avg_cpu = statistics.mean(p.cpu_usage for p in recent_performance)
        avg_memory = statistics.mean(p.memory_usage for p in recent_performance)

        return (
            avg_cpu > self.thresholds.max_cpu_usage or
            avg_memory > self.thresholds.max_memory_usage
        )

    def _check_threshold_violations(self) -> List[Tuple[QualityMetric, float]]:
        """Check for threshold violations"""
        violations = []

        # Check latency
        if self.measurements[QualityMetric.LATENCY]:
            recent_latency = [m.value for m in list(self.measurements[QualityMetric.LATENCY])[-10:]]
            avg_latency = statistics.mean(recent_latency)

            if avg_latency > self.thresholds.max_latency:
                violations.append((QualityMetric.LATENCY, avg_latency))

        # Check accuracy
        if self.measurements[QualityMetric.ACCURACY]:
            recent_accuracy = [m.value for m in list(self.measurements[QualityMetric.ACCURACY])[-10:]]
            avg_accuracy = statistics.mean(recent_accuracy)

            if avg_accuracy < self.thresholds.min_accuracy:
                violations.append((QualityMetric.ACCURACY, avg_accuracy))

        # Check error rate
        if self.measurements[QualityMetric.ERROR_RATE]:
            recent_errors = [m.value for m in list(self.measurements[QualityMetric.ERROR_RATE])[-10:]]
            avg_error_rate = statistics.mean(recent_errors)

            if avg_error_rate > self.thresholds.max_error_rate:
                violations.append((QualityMetric.ERROR_RATE, avg_error_rate))

        return violations

    async def optimize_quality(self) -> bool:
        """Perform quality optimization"""
        if self.is_optimizing:
            return False

        self.is_optimizing = True
        self.last_optimization_time = time.time()

        try:
            logger.info("Starting voice quality optimization...")

            # Analyze current state
            current_state = self._analyze_current_state()

            # Determine optimization actions
            new_config = await self._generate_optimized_config(current_state)

            # Apply configuration if it's different
            if self._config_changed(new_config):
                await self._apply_configuration(new_config)
                logger.info(f"Applied optimized configuration: {self._config_summary(new_config)}")
                return True
            else:
                logger.debug("No configuration changes needed")
                return False

        except Exception as e:
            logger.error(f"Quality optimization failed: {e}")
            return False
        finally:
            self.is_optimizing = False

    def _analyze_current_state(self) -> Dict[str, Any]:
        """Analyze current system state for optimization"""
        state = {
            "threshold_violations": self._check_threshold_violations(),
            "performance_trend": self._get_performance_trend(),
            "quality_trends": self._get_quality_trends(),
            "system_load": self._get_current_system_load(),
            "optimization_history": self._get_optimization_history()
        }

        return state

    def _get_performance_trend(self) -> str:
        """Get performance trend (improving/degrading/stable)"""
        if len(self.performance_history) < 10:
            return "insufficient_data"

        recent = list(self.performance_history)[-5:]
        older = list(self.performance_history)[-10:-5]

        recent_avg_cpu = statistics.mean(p.cpu_usage for p in recent)
        older_avg_cpu = statistics.mean(p.cpu_usage for p in older)

        if recent_avg_cpu > older_avg_cpu + 5:
            return "degrading"
        elif recent_avg_cpu < older_avg_cpu - 5:
            return "improving"
        else:
            return "stable"

    def _get_quality_trends(self) -> Dict[QualityMetric, str]:
        """Get quality trends for each metric"""
        trends = {}

        for metric, measurements in self.measurements.items():
            if len(measurements) < 10:
                trends[metric] = "insufficient_data"
                continue

            recent_values = [m.value for m in list(measurements)[-5:]]
            older_values = [m.value for m in list(measurements)[-10:-5]]

            recent_avg = statistics.mean(recent_values)
            older_avg = statistics.mean(older_values)

            # For metrics where higher is better (accuracy, audio_quality)
            if metric in [QualityMetric.ACCURACY, QualityMetric.AUDIO_QUALITY, QualityMetric.USER_SATISFACTION]:
                if recent_avg > older_avg * 1.05:
                    trends[metric] = "improving"
                elif recent_avg < older_avg * 0.95:
                    trends[metric] = "degrading"
                else:
                    trends[metric] = "stable"
            # For metrics where lower is better (latency, error_rate, system_load)
            else:
                if recent_avg < older_avg * 0.95:
                    trends[metric] = "improving"
                elif recent_avg > older_avg * 1.05:
                    trends[metric] = "degrading"
                else:
                    trends[metric] = "stable"

        return trends

    def _get_current_system_load(self) -> float:
        """Get current system load estimate"""
        if not self.performance_history:
            return 0.5  # Default estimate

        latest = self.performance_history[-1]
        load_score = (
            latest.cpu_usage / 100.0 * 0.4 +
            latest.memory_usage / 100.0 * 0.3 +
            (latest.gpu_usage or 0) / 100.0 * 0.2 +
            min(latest.active_connections / 10, 1.0) * 0.1
        )

        return min(1.0, load_score)

    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        # This would track recent optimization decisions
        # For now, return empty list
        return []

    async def _generate_optimized_config(self, state: Dict[str, Any]) -> AdaptiveConfiguration:
        """Generate optimized configuration based on current state"""
        new_config = AdaptiveConfiguration(**self.current_config.dict())

        violations = state["threshold_violations"]
        system_load = state["system_load"]
        performance_trend = state["performance_trend"]

        # Apply optimization strategy
        if self.optimization_strategy == OptimizationStrategy.MINIMIZE_LATENCY:
            new_config = self._optimize_for_latency(new_config, state)
        elif self.optimization_strategy == OptimizationStrategy.MAXIMIZE_QUALITY:
            new_config = self._optimize_for_quality(new_config, state)
        elif self.optimization_strategy == OptimizationStrategy.BALANCE_QUALITY_SPEED:
            new_config = self._optimize_for_balance(new_config, state)
        elif self.optimization_strategy == OptimizationStrategy.MINIMIZE_BANDWIDTH:
            new_config = self._optimize_for_bandwidth(new_config, state)
        else:  # ADAPTIVE
            new_config = self._adaptive_optimize(new_config, state)

        return new_config

    def _optimize_for_latency(
        self,
        config: AdaptiveConfiguration,
        state: Dict[str, Any]
    ) -> AdaptiveConfiguration:
        """Optimize configuration for minimum latency"""
        config.stt_model = STTModel.WHISPER_TINY
        config.stt_quality = TranscriptionQuality.FAST
        config.tts_quality = TTSQuality.FAST
        config.streaming_quality = StreamingQuality.LOW_LATENCY
        config.sample_rate = 8000
        config.enable_noise_reduction = False
        config.enable_compression = True
        config.optimization_level = 0

        return config

    def _optimize_for_quality(
        self,
        config: AdaptiveConfiguration,
        state: Dict[str, Any]
    ) -> AdaptiveConfiguration:
        """Optimize configuration for maximum quality"""
        system_load = state["system_load"]

        if system_load < 0.6:  # System can handle high quality
            config.stt_model = STTModel.WHISPER_LARGE_V3
            config.tts_quality = TTSQuality.HIGH_QUALITY
            config.sample_rate = 22050
        else:  # Moderate quality due to system load
            config.stt_model = STTModel.WHISPER_MEDIUM
            config.tts_quality = TTSQuality.BALANCED
            config.sample_rate = 16000

        config.stt_quality = TranscriptionQuality.ACCURATE
        config.streaming_quality = StreamingQuality.HIGH_QUALITY
        config.enable_noise_reduction = True
        config.enable_compression = False
        config.optimization_level = 3

        return config

    def _optimize_for_balance(
        self,
        config: AdaptiveConfiguration,
        state: Dict[str, Any]
    ) -> AdaptiveConfiguration:
        """Optimize configuration for balanced quality and speed"""
        config.stt_model = STTModel.WHISPER_SMALL
        config.stt_quality = TranscriptionQuality.BALANCED
        config.tts_quality = TTSQuality.BALANCED
        config.streaming_quality = StreamingQuality.BALANCED
        config.sample_rate = 16000
        config.enable_noise_reduction = True
        config.enable_compression = True
        config.optimization_level = 2

        return config

    def _optimize_for_bandwidth(
        self,
        config: AdaptiveConfiguration,
        state: Dict[str, Any]
    ) -> AdaptiveConfiguration:
        """Optimize configuration for minimum bandwidth usage"""
        config.audio_format = AudioFormat.MP3
        config.sample_rate = 8000
        config.enable_compression = True
        config.streaming_quality = StreamingQuality.LOW_LATENCY

        # Use lighter models
        config.stt_model = STTModel.WHISPER_BASE
        config.tts_quality = TTSQuality.FAST

        return config

    def _adaptive_optimize(
        self,
        config: AdaptiveConfiguration,
        state: Dict[str, Any]
    ) -> AdaptiveConfiguration:
        """Adaptive optimization based on current conditions"""
        violations = state["threshold_violations"]
        system_load = state["system_load"]
        performance_trend = state["performance_trend"]

        # Start with current config
        new_config = AdaptiveConfiguration(**config.dict())

        # Handle threshold violations
        for metric, value in violations:
            if metric == QualityMetric.LATENCY:
                # Reduce quality to improve latency
                new_config = self._reduce_quality_for_speed(new_config)
            elif metric == QualityMetric.ACCURACY:
                # Increase quality to improve accuracy
                new_config = self._increase_quality_for_accuracy(new_config)
            elif metric == QualityMetric.ERROR_RATE:
                # Increase stability and quality
                new_config = self._increase_stability(new_config)

        # Adjust based on system load
        if system_load > 0.8:
            new_config = self._reduce_system_load(new_config)
        elif system_load < 0.4 and performance_trend in ["stable", "improving"]:
            new_config = self._increase_quality_when_possible(new_config)

        return new_config

    def _reduce_quality_for_speed(self, config: AdaptiveConfiguration) -> AdaptiveConfiguration:
        """Reduce quality settings to improve speed"""
        # Downgrade STT model
        model_hierarchy = [
            STTModel.WHISPER_LARGE_V3,
            STTModel.WHISPER_LARGE,
            STTModel.WHISPER_MEDIUM,
            STTModel.WHISPER_SMALL,
            STTModel.WHISPER_BASE,
            STTModel.WHISPER_TINY
        ]

        current_index = model_hierarchy.index(config.stt_model)
        if current_index < len(model_hierarchy) - 1:
            config.stt_model = model_hierarchy[current_index + 1]

        # Reduce other quality settings
        if config.stt_quality == TranscriptionQuality.ACCURATE:
            config.stt_quality = TranscriptionQuality.BALANCED
        elif config.stt_quality == TranscriptionQuality.BALANCED:
            config.stt_quality = TranscriptionQuality.FAST

        if config.tts_quality == TTSQuality.HIGH_QUALITY:
            config.tts_quality = TTSQuality.BALANCED
        elif config.tts_quality == TTSQuality.BALANCED:
            config.tts_quality = TTSQuality.FAST

        return config

    def _increase_quality_for_accuracy(self, config: AdaptiveConfiguration) -> AdaptiveConfiguration:
        """Increase quality settings to improve accuracy"""
        # Upgrade STT model
        model_hierarchy = [
            STTModel.WHISPER_TINY,
            STTModel.WHISPER_BASE,
            STTModel.WHISPER_SMALL,
            STTModel.WHISPER_MEDIUM,
            STTModel.WHISPER_LARGE,
            STTModel.WHISPER_LARGE_V3
        ]

        current_index = model_hierarchy.index(config.stt_model)
        if current_index < len(model_hierarchy) - 1:
            config.stt_model = model_hierarchy[current_index + 1]

        # Increase quality settings
        if config.stt_quality == TranscriptionQuality.FAST:
            config.stt_quality = TranscriptionQuality.BALANCED
        elif config.stt_quality == TranscriptionQuality.BALANCED:
            config.stt_quality = TranscriptionQuality.ACCURATE

        config.enable_noise_reduction = True

        return config

    def _increase_stability(self, config: AdaptiveConfiguration) -> AdaptiveConfiguration:
        """Increase stability to reduce errors"""
        config.enable_noise_reduction = True
        config.optimization_level = min(3, config.optimization_level + 1)

        # Use more stable models
        if config.stt_model == STTModel.WHISPER_TINY:
            config.stt_model = STTModel.WHISPER_BASE

        return config

    def _reduce_system_load(self, config: AdaptiveConfiguration) -> AdaptiveConfiguration:
        """Reduce system load by using lighter settings"""
        return self._reduce_quality_for_speed(config)

    def _increase_quality_when_possible(self, config: AdaptiveConfiguration) -> AdaptiveConfiguration:
        """Increase quality when system resources allow"""
        return self._increase_quality_for_accuracy(config)

    def _config_changed(self, new_config: AdaptiveConfiguration) -> bool:
        """Check if configuration has changed"""
        return new_config.dict() != self.current_config.dict()

    async def _apply_configuration(self, new_config: AdaptiveConfiguration) -> None:
        """Apply new configuration"""
        old_config = self.current_config
        self.current_config = new_config

        # Notify callbacks
        for callback in self.config_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(new_config)
                else:
                    callback(new_config)
            except Exception as e:
                logger.error(f"Configuration change callback error: {e}")

        logger.info(f"Configuration updated: {self._config_summary(new_config)}")

    def _config_summary(self, config: AdaptiveConfiguration) -> str:
        """Get configuration summary for logging"""
        return f"STT:{config.stt_model.value}, TTS:{config.tts_quality.value}, SR:{config.sample_rate}Hz"

    def register_config_callback(self, callback: Callable[[AdaptiveConfiguration], None]) -> None:
        """Register callback for configuration changes"""
        self.config_change_callbacks.append(callback)

    def set_optimization_strategy(self, strategy: OptimizationStrategy) -> None:
        """Set optimization strategy"""
        self.optimization_strategy = strategy
        logger.info(f"Optimization strategy set to: {strategy.value}")

    def get_current_metrics(self) -> Dict[QualityMetric, Optional[float]]:
        """Get current metric values"""
        metrics = {}

        for metric, measurements in self.measurements.items():
            if measurements:
                recent_values = [m.value for m in list(measurements)[-5:]]
                metrics[metric] = statistics.mean(recent_values)
            else:
                metrics[metric] = None

        return metrics

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            "current_config": self.current_config.dict(),
            "optimization_strategy": self.optimization_strategy.value,
            "current_metrics": self.get_current_metrics(),
            "threshold_violations": self._check_threshold_violations(),
            "performance_trend": self._get_performance_trend(),
            "quality_trends": self._get_quality_trends(),
            "system_load": self._get_current_system_load(),
            "last_optimization": self.last_optimization_time,
            "is_optimizing": self.is_optimizing
        }

    async def manual_optimize(self, target_metric: QualityMetric) -> bool:
        """Manually trigger optimization for specific metric"""
        logger.info(f"Manual optimization triggered for metric: {target_metric}")

        # Temporarily adjust strategy based on target metric
        old_strategy = self.optimization_strategy

        if target_metric == QualityMetric.LATENCY:
            self.optimization_strategy = OptimizationStrategy.MINIMIZE_LATENCY
        elif target_metric in [QualityMetric.ACCURACY, QualityMetric.AUDIO_QUALITY]:
            self.optimization_strategy = OptimizationStrategy.MAXIMIZE_QUALITY
        elif target_metric == QualityMetric.NETWORK_BANDWIDTH:
            self.optimization_strategy = OptimizationStrategy.MINIMIZE_BANDWIDTH

        # Perform optimization
        result = await self.optimize_quality()

        # Restore original strategy
        self.optimization_strategy = old_strategy

        return result


# Factory function
def create_quality_optimizer(thresholds: Optional[QualityThresholds] = None) -> VoiceQualityOptimizer:
    """Create and initialize voice quality optimizer"""
    return VoiceQualityOptimizer(thresholds)
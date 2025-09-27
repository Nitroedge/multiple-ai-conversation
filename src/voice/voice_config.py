"""
Voice configuration management system for centralized voice settings
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from pydantic import BaseModel, Field, validator

from .stt_processor import STTConfiguration, STTModel, TranscriptionQuality
from .tts_processor import TTSConfiguration, TTSProvider, AudioFormat, TTSQuality
from .voice_pipeline import VoiceConfiguration
from .audio_utils import AudioProcessingConfig
from .audio_streaming import AudioStreamConfig, StreamingQuality
from .personality_adaptation import AdaptationIntensity

logger = logging.getLogger(__name__)


class VoiceConfigurationPreset(str, Enum):
    """Predefined voice configuration presets"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    HIGH_QUALITY = "high_quality"
    LOW_LATENCY = "low_latency"
    MOBILE = "mobile"
    STUDIO = "studio"


class VoiceConfigScope(str, Enum):
    """Scope levels for voice configuration"""
    GLOBAL = "global"
    AGENT = "agent"
    SESSION = "session"
    USER = "user"


@dataclass
class VoiceMetrics:
    """Voice processing performance metrics"""
    avg_stt_latency: float = 0.0
    avg_tts_latency: float = 0.0
    transcription_accuracy: float = 0.0
    synthesis_quality: float = 0.0
    total_voice_commands: int = 0
    successful_commands: int = 0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0


class ComprehensiveVoiceConfig(BaseModel):
    """Comprehensive voice configuration combining all voice-related settings"""

    # Identification
    config_id: str = Field(description="Unique configuration identifier")
    name: str = Field(description="Human-readable configuration name")
    description: Optional[str] = Field(default=None, description="Configuration description")
    version: str = Field(default="1.0", description="Configuration version")
    scope: VoiceConfigScope = Field(default=VoiceConfigScope.GLOBAL, description="Configuration scope")

    # Core configurations
    stt_config: STTConfiguration = Field(default_factory=STTConfiguration)
    tts_config: TTSConfiguration = Field(default_factory=TTSConfiguration)
    voice_pipeline_config: VoiceConfiguration = Field(default_factory=VoiceConfiguration)
    audio_processing_config: AudioProcessingConfig = Field(default_factory=AudioProcessingConfig)
    audio_streaming_config: AudioStreamConfig = Field(default_factory=AudioStreamConfig)

    # Voice adaptation settings
    enable_personality_adaptation: bool = Field(default=True, description="Enable personality-based voice adaptation")
    adaptation_intensity: AdaptationIntensity = Field(default=AdaptationIntensity.MODERATE)
    emotion_adaptation_enabled: bool = Field(default=True, description="Enable emotion-based voice adaptation")

    # Quality settings
    prioritize_quality: bool = Field(default=True, description="Prioritize quality over speed")
    streaming_quality: StreamingQuality = Field(default=StreamingQuality.BALANCED)

    # Agent-specific settings
    agent_voice_mappings: Dict[str, str] = Field(default_factory=dict, description="Agent ID to voice ID mappings")
    agent_personality_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Advanced settings
    fallback_voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", description="Fallback voice when primary fails")
    max_concurrent_requests: int = Field(default=5, description="Maximum concurrent voice processing requests")
    request_timeout: float = Field(default=30.0, description="Request timeout in seconds")

    # Performance thresholds
    min_transcription_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    max_response_latency: float = Field(default=3.0, description="Maximum acceptable response latency")

    # Feature flags
    enable_voice_activity_detection: bool = Field(default=True)
    enable_noise_suppression: bool = Field(default=True)
    enable_echo_cancellation: bool = Field(default=True)
    enable_auto_gain_control: bool = Field(default=True)
    enable_streaming_mode: bool = Field(default=False)
    enable_voice_cloning: bool = Field(default=False)

    # Security and privacy
    log_audio_data: bool = Field(default=False, description="Log audio data for debugging")
    encrypt_voice_data: bool = Field(default=True, description="Encrypt voice data in transit")
    retain_voice_history: bool = Field(default=True, description="Retain voice interaction history")
    voice_data_retention_days: int = Field(default=30, description="Days to retain voice data")

    @validator('agent_voice_mappings')
    def validate_agent_mappings(cls, v):
        """Validate agent voice mappings"""
        if not isinstance(v, dict):
            raise ValueError("Agent voice mappings must be a dictionary")
        return v

    @validator('max_concurrent_requests')
    def validate_concurrent_requests(cls, v):
        """Validate concurrent requests limit"""
        if v < 1 or v > 50:
            raise ValueError("Concurrent requests must be between 1 and 50")
        return v


class VoiceConfigurationManager:
    """Centralized voice configuration management"""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/voice")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Active configurations
        self.configurations: Dict[str, ComprehensiveVoiceConfig] = {}
        self.active_config_id: Optional[str] = None

        # Configuration history
        self.config_history: List[Dict[str, Any]] = []

        # Performance metrics
        self.metrics: Dict[str, VoiceMetrics] = {}

        # Load existing configurations
        self._load_configurations()

    def _load_configurations(self) -> None:
        """Load all configurations from disk"""
        try:
            config_files = list(self.config_dir.glob("*.json"))

            for config_file in config_files:
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)

                    config = ComprehensiveVoiceConfig(**config_data)
                    self.configurations[config.config_id] = config

                    logger.debug(f"Loaded voice configuration: {config.name}")

                except Exception as e:
                    logger.error(f"Failed to load configuration {config_file}: {e}")

            logger.info(f"Loaded {len(self.configurations)} voice configurations")

            # Set default active configuration
            if self.configurations and not self.active_config_id:
                self.active_config_id = list(self.configurations.keys())[0]

        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")

    def create_configuration(
        self,
        config_id: str,
        name: str,
        preset: Optional[VoiceConfigurationPreset] = None,
        base_config: Optional[ComprehensiveVoiceConfig] = None,
        **kwargs
    ) -> ComprehensiveVoiceConfig:
        """Create new voice configuration"""
        try:
            if config_id in self.configurations:
                raise ValueError(f"Configuration {config_id} already exists")

            # Start with preset or base configuration
            if preset:
                config_data = self._get_preset_config(preset)
                config_data.update(kwargs)
            elif base_config:
                config_data = base_config.dict()
                config_data.update(kwargs)
            else:
                config_data = kwargs

            # Set basic properties
            config_data.update({
                "config_id": config_id,
                "name": name
            })

            # Create configuration
            config = ComprehensiveVoiceConfig(**config_data)

            # Store configuration
            self.configurations[config_id] = config
            self._save_configuration(config)

            # Initialize metrics
            self.metrics[config_id] = VoiceMetrics()

            logger.info(f"Created voice configuration: {name} ({config_id})")
            return config

        except Exception as e:
            logger.error(f"Failed to create configuration {config_id}: {e}")
            raise

    def _get_preset_config(self, preset: VoiceConfigurationPreset) -> Dict[str, Any]:
        """Get configuration data for preset"""
        presets = {
            VoiceConfigurationPreset.DEVELOPMENT: {
                "stt_config": {
                    "model": STTModel.WHISPER_BASE,
                    "quality": TranscriptionQuality.FAST
                },
                "tts_config": {
                    "quality": TTSQuality.FAST,
                    "optimize_streaming_latency": 2
                },
                "prioritize_quality": False,
                "streaming_quality": StreamingQuality.LOW_LATENCY,
                "log_audio_data": True
            },

            VoiceConfigurationPreset.PRODUCTION: {
                "stt_config": {
                    "model": STTModel.WHISPER_SMALL,
                    "quality": TranscriptionQuality.BALANCED
                },
                "tts_config": {
                    "quality": TTSQuality.BALANCED,
                    "optimize_streaming_latency": 1
                },
                "prioritize_quality": True,
                "streaming_quality": StreamingQuality.BALANCED,
                "log_audio_data": False,
                "max_concurrent_requests": 10
            },

            VoiceConfigurationPreset.HIGH_QUALITY: {
                "stt_config": {
                    "model": STTModel.WHISPER_LARGE_V3,
                    "quality": TranscriptionQuality.ACCURATE
                },
                "tts_config": {
                    "quality": TTSQuality.HIGH_QUALITY,
                    "optimize_streaming_latency": 0
                },
                "prioritize_quality": True,
                "streaming_quality": StreamingQuality.HIGH_QUALITY,
                "adaptation_intensity": AdaptationIntensity.STRONG
            },

            VoiceConfigurationPreset.LOW_LATENCY: {
                "stt_config": {
                    "model": STTModel.WHISPER_TINY,
                    "quality": TranscriptionQuality.FAST
                },
                "tts_config": {
                    "quality": TTSQuality.FAST,
                    "optimize_streaming_latency": 4
                },
                "prioritize_quality": False,
                "streaming_quality": StreamingQuality.LOW_LATENCY,
                "enable_streaming_mode": True
            },

            VoiceConfigurationPreset.MOBILE: {
                "stt_config": {
                    "model": STTModel.WHISPER_BASE,
                    "quality": TranscriptionQuality.FAST
                },
                "tts_config": {
                    "quality": TTSQuality.FAST,
                    "format": AudioFormat.MP3
                },
                "audio_streaming_config": {
                    "sample_rate": 16000,
                    "chunk_size": 512
                },
                "max_concurrent_requests": 3
            },

            VoiceConfigurationPreset.STUDIO: {
                "stt_config": {
                    "model": STTModel.WHISPER_LARGE_V3,
                    "quality": TranscriptionQuality.ACCURATE,
                    "fp16": False
                },
                "tts_config": {
                    "quality": TTSQuality.HIGH_QUALITY,
                    "format": AudioFormat.WAV,
                    "sample_rate": 44100
                },
                "audio_processing_config": {
                    "target_sample_rate": 44100,
                    "normalize": True,
                    "noise_reduction": True
                },
                "adaptation_intensity": AdaptationIntensity.EXTREME
            }
        }

        return presets.get(preset, {})

    def get_configuration(self, config_id: str) -> Optional[ComprehensiveVoiceConfig]:
        """Get configuration by ID"""
        return self.configurations.get(config_id)

    def get_active_configuration(self) -> Optional[ComprehensiveVoiceConfig]:
        """Get currently active configuration"""
        if self.active_config_id:
            return self.configurations.get(self.active_config_id)
        return None

    def set_active_configuration(self, config_id: str) -> bool:
        """Set active configuration"""
        if config_id not in self.configurations:
            logger.error(f"Configuration {config_id} not found")
            return False

        old_config_id = self.active_config_id
        self.active_config_id = config_id

        logger.info(f"Active configuration changed: {old_config_id} -> {config_id}")
        return True

    def update_configuration(
        self,
        config_id: str,
        updates: Dict[str, Any],
        save: bool = True
    ) -> bool:
        """Update existing configuration"""
        if config_id not in self.configurations:
            logger.error(f"Configuration {config_id} not found")
            return False

        try:
            config = self.configurations[config_id]

            # Store old configuration in history
            self.config_history.append({
                "config_id": config_id,
                "timestamp": time.time(),
                "old_config": config.dict(),
                "updates": updates
            })

            # Apply updates
            config_data = config.dict()
            config_data.update(updates)

            # Create updated configuration
            updated_config = ComprehensiveVoiceConfig(**config_data)
            self.configurations[config_id] = updated_config

            if save:
                self._save_configuration(updated_config)

            logger.info(f"Updated configuration: {config_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update configuration {config_id}: {e}")
            return False

    def delete_configuration(self, config_id: str) -> bool:
        """Delete configuration"""
        if config_id not in self.configurations:
            return False

        try:
            # Remove from memory
            del self.configurations[config_id]

            # Remove metrics
            if config_id in self.metrics:
                del self.metrics[config_id]

            # Delete file
            config_file = self.config_dir / f"{config_id}.json"
            if config_file.exists():
                config_file.unlink()

            # Update active configuration if necessary
            if self.active_config_id == config_id:
                self.active_config_id = list(self.configurations.keys())[0] if self.configurations else None

            logger.info(f"Deleted configuration: {config_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete configuration {config_id}: {e}")
            return False

    def _save_configuration(self, config: ComprehensiveVoiceConfig) -> None:
        """Save configuration to disk"""
        try:
            config_file = self.config_dir / f"{config.config_id}.json"

            with open(config_file, 'w') as f:
                json.dump(config.dict(), f, indent=2, default=str)

            logger.debug(f"Saved configuration: {config.config_id}")

        except Exception as e:
            logger.error(f"Failed to save configuration {config.config_id}: {e}")

    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all configurations with basic info"""
        return [
            {
                "config_id": config.config_id,
                "name": config.name,
                "description": config.description,
                "version": config.version,
                "scope": config.scope.value,
                "is_active": config.config_id == self.active_config_id
            }
            for config in self.configurations.values()
        ]

    def clone_configuration(
        self,
        source_config_id: str,
        new_config_id: str,
        new_name: str
    ) -> Optional[ComprehensiveVoiceConfig]:
        """Clone existing configuration"""
        if source_config_id not in self.configurations:
            logger.error(f"Source configuration {source_config_id} not found")
            return None

        if new_config_id in self.configurations:
            logger.error(f"Configuration {new_config_id} already exists")
            return None

        try:
            source_config = self.configurations[source_config_id]

            # Clone configuration data
            cloned_data = source_config.dict()
            cloned_data.update({
                "config_id": new_config_id,
                "name": new_name,
                "version": "1.0"  # Reset version for cloned config
            })

            # Create new configuration
            cloned_config = ComprehensiveVoiceConfig(**cloned_data)

            # Store and save
            self.configurations[new_config_id] = cloned_config
            self._save_configuration(cloned_config)

            # Initialize metrics
            self.metrics[new_config_id] = VoiceMetrics()

            logger.info(f"Cloned configuration: {source_config_id} -> {new_config_id}")
            return cloned_config

        except Exception as e:
            logger.error(f"Failed to clone configuration: {e}")
            return None

    def export_configuration(self, config_id: str, export_path: Path) -> bool:
        """Export configuration to file"""
        if config_id not in self.configurations:
            return False

        try:
            config = self.configurations[config_id]

            with open(export_path, 'w') as f:
                json.dump(config.dict(), f, indent=2, default=str)

            logger.info(f"Exported configuration {config_id} to {export_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration {config_id}: {e}")
            return False

    def import_configuration(self, import_path: Path) -> Optional[str]:
        """Import configuration from file"""
        try:
            with open(import_path, 'r') as f:
                config_data = json.load(f)

            config = ComprehensiveVoiceConfig(**config_data)

            # Ensure unique config ID
            original_id = config.config_id
            counter = 1
            while config.config_id in self.configurations:
                config.config_id = f"{original_id}_{counter}"
                counter += 1

            # Store configuration
            self.configurations[config.config_id] = config
            self._save_configuration(config)

            # Initialize metrics
            self.metrics[config.config_id] = VoiceMetrics()

            logger.info(f"Imported configuration: {config.config_id}")
            return config.config_id

        except Exception as e:
            logger.error(f"Failed to import configuration from {import_path}: {e}")
            return None

    def update_metrics(self, config_id: str, metrics_update: Dict[str, Any]) -> None:
        """Update performance metrics for configuration"""
        if config_id in self.metrics:
            current_metrics = self.metrics[config_id]

            for key, value in metrics_update.items():
                if hasattr(current_metrics, key):
                    setattr(current_metrics, key, value)

            logger.debug(f"Updated metrics for configuration {config_id}")

    def get_metrics(self, config_id: str) -> Optional[VoiceMetrics]:
        """Get performance metrics for configuration"""
        return self.metrics.get(config_id)

    def get_configuration_history(self, config_id: str) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        return [
            entry for entry in self.config_history
            if entry["config_id"] == config_id
        ]

    def cleanup_old_configurations(self, days: int = 90) -> int:
        """Cleanup old unused configurations"""
        import time

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        removed_count = 0

        # Find configurations to remove based on history
        configs_to_remove = []

        for config_id in self.configurations:
            if config_id == self.active_config_id:
                continue  # Don't remove active configuration

            # Check if configuration has been used recently
            recent_usage = any(
                entry["timestamp"] > cutoff_time
                for entry in self.config_history
                if entry["config_id"] == config_id
            )

            if not recent_usage:
                configs_to_remove.append(config_id)

        # Remove old configurations
        for config_id in configs_to_remove:
            if self.delete_configuration(config_id):
                removed_count += 1

        logger.info(f"Cleaned up {removed_count} old configurations")
        return removed_count


# Factory function
def create_voice_config_manager(config_dir: Optional[Path] = None) -> VoiceConfigurationManager:
    """Create and initialize voice configuration manager"""
    return VoiceConfigurationManager(config_dir)
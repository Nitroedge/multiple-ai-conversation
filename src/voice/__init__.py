"""
Voice processing pipeline for speech-to-text, text-to-speech, and audio handling
"""

from .stt_processor import (
    STTProcessor,
    WhisperSTTProcessor,
    STTConfiguration,
    TranscriptionResult,
    STTModel,
    TranscriptionQuality,
    create_stt_processor
)
from .tts_processor import (
    TTSProcessor,
    ElevenLabsTTSProcessor,
    TTSConfiguration,
    SynthesisResult,
    VoiceProfile,
    VoiceGender,
    VoiceAge,
    AudioFormat,
    TTSQuality,
    TTSProvider,
    create_tts_processor
)
from .audio_utils import (
    AudioProcessor,
    AudioFormat,
    AudioQuality,
    VoiceActivityDetector,
    AudioProcessingConfig,
    AudioMetadata,
    AudioSegment,
    bytes_to_audio,
    audio_to_bytes,
    mix_audio
)
from .voice_pipeline import (
    VoicePipeline,
    VoiceCommand,
    VoiceResponse,
    VoiceConfiguration,
    VoiceCommandType,
    VoiceResponseType,
    ProcessingState
)
from .audio_streaming import (
    AudioStreamer,
    AudioStreamHandler,
    VoiceStreamingHandler,
    AudioStreamingManager,
    AudioStreamConfig,
    StreamingMode,
    StreamingQuality,
    StreamingStats,
    get_optimal_chunk_size,
    get_recommended_config
)
from .personality_adaptation import (
    VoicePersonalityAdapter,
    PersonalityVoiceProfile,
    VoiceCharacteristic,
    AdaptationIntensity,
    VoiceAdaptationRule,
    EmotionVoiceMapping,
    create_personality_adapter
)
from .voice_config import (
    VoiceConfigurationManager,
    ComprehensiveVoiceConfig,
    VoiceConfigurationPreset,
    VoiceConfigScope,
    VoiceMetrics,
    create_voice_config_manager
)
from .quality_optimizer import (
    VoiceQualityOptimizer,
    QualityMetric,
    OptimizationStrategy,
    QualityLevel,
    QualityThresholds,
    AdaptiveConfiguration,
    SystemPerformance,
    create_quality_optimizer
)

__all__ = [
    # STT Components
    "STTProcessor",
    "WhisperSTTProcessor",
    "STTConfiguration",
    "TranscriptionResult",
    "STTModel",
    "TranscriptionQuality",
    "create_stt_processor",

    # TTS Components
    "TTSProcessor",
    "ElevenLabsTTSProcessor",
    "TTSConfiguration",
    "SynthesisResult",
    "VoiceProfile",
    "VoiceGender",
    "VoiceAge",
    "AudioFormat",
    "TTSQuality",
    "TTSProvider",
    "create_tts_processor",

    # Audio Processing
    "AudioProcessor",
    "AudioQuality",
    "VoiceActivityDetector",
    "AudioProcessingConfig",
    "AudioMetadata",
    "AudioSegment",
    "bytes_to_audio",
    "audio_to_bytes",
    "mix_audio",

    # Voice Pipeline
    "VoicePipeline",
    "VoiceCommand",
    "VoiceResponse",
    "VoiceConfiguration",
    "VoiceCommandType",
    "VoiceResponseType",
    "ProcessingState",

    # Audio Streaming
    "AudioStreamer",
    "AudioStreamHandler",
    "VoiceStreamingHandler",
    "AudioStreamingManager",
    "AudioStreamConfig",
    "StreamingMode",
    "StreamingQuality",
    "StreamingStats",
    "get_optimal_chunk_size",
    "get_recommended_config",

    # Personality Adaptation
    "VoicePersonalityAdapter",
    "PersonalityVoiceProfile",
    "VoiceCharacteristic",
    "AdaptationIntensity",
    "VoiceAdaptationRule",
    "EmotionVoiceMapping",
    "create_personality_adapter",

    # Configuration Management
    "VoiceConfigurationManager",
    "ComprehensiveVoiceConfig",
    "VoiceConfigurationPreset",
    "VoiceConfigScope",
    "VoiceMetrics",
    "create_voice_config_manager",

    # Quality Optimization
    "VoiceQualityOptimizer",
    "QualityMetric",
    "OptimizationStrategy",
    "QualityLevel",
    "QualityThresholds",
    "AdaptiveConfiguration",
    "SystemPerformance",
    "create_quality_optimizer"
]
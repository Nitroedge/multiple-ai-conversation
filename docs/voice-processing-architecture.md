# Voice Processing Architecture

## Overview

The voice processing system is a comprehensive, modular architecture that enables natural voice interactions with AI agents. It integrates speech-to-text (STT), text-to-speech (TTS), real-time audio streaming, personality-based voice adaptation, and quality optimization.

## Architecture Components

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     Voice Processing Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│  Audio Input → STT → Command Analysis → Agent Processing →     │
│  Response Generation → TTS → Audio Output → Quality Monitor    │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Speech-to-Text (STT) System

**File**: `src/voice/stt_processor.py`

**Features**:
- OpenAI Whisper integration with multiple model sizes
- Configurable quality vs speed tradeoffs
- Multi-language support
- Streaming transcription capabilities
- Confidence scoring and quality metrics

**Models Available**:
- `whisper-tiny`: Ultra-fast, lower accuracy
- `whisper-base`: Balanced speed/accuracy
- `whisper-small`: Good accuracy, moderate speed
- `whisper-medium`: High accuracy, slower
- `whisper-large-v3`: Maximum accuracy, slowest

**Configuration Example**:
```python
stt_config = STTConfiguration(
    model=STTModel.WHISPER_BASE,
    quality=TranscriptionQuality.BALANCED,
    language="en",
    temperature=0.0,
    beam_size=5
)
```

### 2. Text-to-Speech (TTS) System

**File**: `src/voice/tts_processor.py`

**Features**:
- ElevenLabs integration with voice cloning
- Multiple voice profiles and characteristics
- Streaming synthesis for low latency
- Voice quality optimization
- Custom voice model training support

**Voice Profile Structure**:
```python
voice_profile = VoiceProfile(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    name="Rachel",
    gender=VoiceGender.FEMALE,
    age=VoiceAge.MIDDLE_AGED,
    stability=0.5,
    similarity_boost=0.5,
    style=0.0
)
```

### 3. Voice Command Pipeline

**File**: `src/voice/voice_pipeline.py`

**Command Types**:
- `CONVERSATION`: General conversation with agents
- `AGENT_SELECTION`: Switch between different AI agents
- `SYSTEM_CONTROL`: Control system functions (pause, resume, etc.)
- `MEMORY_QUERY`: Search and retrieve memories
- `HOME_AUTOMATION`: Control smart home devices
- `VOICE_CONTROL`: Adjust voice settings

**Processing Flow**:
1. Audio input received
2. Voice activity detection
3. STT transcription
4. Command type analysis
5. Agent routing
6. Response generation
7. TTS synthesis
8. Audio output

### 4. Audio Processing Utilities

**File**: `src/voice/audio_utils.py`

**Capabilities**:
- Audio format conversion (WAV, MP3, FLAC, OGG)
- Noise reduction and audio enhancement
- Voice activity detection (VAD)
- Audio segmentation by silence
- Real-time audio filtering
- Dynamic range compression

**Audio Enhancement Pipeline**:
```python
# Example audio enhancement
enhanced_audio = audio_processor.enhance_audio(
    audio_data, sample_rate
)
# Includes: filtering, noise reduction, compression, normalization
```

### 5. Real-Time Audio Streaming

**File**: `src/voice/audio_streaming.py`

**Features**:
- Low-latency real-time audio processing
- Configurable buffer sizes and quality levels
- Voice activity detection integration
- Multiple streaming modes (input, output, duplex)
- Performance monitoring and statistics

**Streaming Configuration**:
```python
stream_config = AudioStreamConfig(
    sample_rate=16000,
    channels=1,
    chunk_size=1024,
    buffer_size=4,
    enable_agc=True,
    enable_noise_suppression=True
)
```

### 6. Personality-Based Voice Adaptation

**File**: `src/voice/personality_adaptation.py`

**Adaptation Features**:
- Big Five personality model integration
- Emotional state-driven voice changes
- Dynamic voice characteristic adjustment
- Agent-specific voice profiles
- Real-time adaptation based on conversation context

**Adaptation Rules**:
- **Extroversion** → Energy, volume, speaking speed
- **Agreeableness** → Warmth, stability, pitch
- **Conscientiousness** → Stability, consistency
- **Neuroticism** → Stability variation (inverted)
- **Openness** → Style expressiveness, creativity

### 7. Voice Configuration Management

**File**: `src/voice/voice_config.py`

**Configuration Presets**:
- `DEVELOPMENT`: Fast processing, debug logging
- `PRODUCTION`: Balanced quality and performance
- `HIGH_QUALITY`: Maximum quality, slower processing
- `LOW_LATENCY`: Fastest response, reduced quality
- `MOBILE`: Optimized for mobile devices
- `STUDIO`: Professional quality recording

**Configuration Structure**:
```python
config = ComprehensiveVoiceConfig(
    config_id="production_config",
    name="Production Voice Config",
    stt_config=stt_settings,
    tts_config=tts_settings,
    enable_personality_adaptation=True,
    adaptation_intensity=AdaptationIntensity.MODERATE
)
```

### 8. Adaptive Quality Optimization

**File**: `src/voice/quality_optimizer.py`

**Optimization Strategies**:
- `MINIMIZE_LATENCY`: Fastest response times
- `MAXIMIZE_QUALITY`: Best audio quality
- `BALANCE_QUALITY_SPEED`: Optimal trade-off
- `MINIMIZE_BANDWIDTH`: Lowest data usage
- `ADAPTIVE`: Dynamic adjustment based on conditions

**Quality Metrics Monitored**:
- Processing latency
- Transcription accuracy
- Audio quality scores
- System resource usage
- Network bandwidth
- Error rates

## API Integration

### Voice Processing Endpoints

**FastAPI Routes** (in `src/api/router.py`):

```python
# Core voice processing
POST /api/voice/transcribe        # STT transcription
POST /api/voice/synthesize        # TTS synthesis
POST /api/voice/broadcast         # WebSocket audio broadcast

# Streaming support
POST /api/voice/stream/process    # Process audio chunks
POST /api/voice/stream/finalize   # Complete stream processing

# Configuration and monitoring
GET  /api/voice/config            # Get voice configuration
POST /api/voice/config            # Update voice configuration
POST /api/metrics/voice           # Log voice metrics

# Command processing
POST /api/conversation/process    # Process voice commands
POST /api/agents/select           # Agent selection
POST /api/system/control          # System control
POST /api/home/control            # Home automation
```

## n8n Workflow Integration

**Workflow File**: `n8n/workflows/voice-processing-pipeline.json`

**Workflow Components**:
1. **Voice Input Webhook** - Receives audio data
2. **STT Transcription** - Converts speech to text
3. **Confidence Filter** - Validates transcription quality
4. **Command Analysis** - Determines command type and routing
5. **Command Routing** - Routes to appropriate handler
6. **Response Generation** - Generates appropriate response
7. **TTS Synthesis** - Converts response to speech
8. **Voice Broadcast** - Sends audio to clients
9. **Metrics Collection** - Logs performance data

**Parallel Processing Support**:
- **Voice Stream Processing** - Real-time audio chunks
- **Error Handling** - Low confidence fallbacks
- **Quality Monitoring** - Performance tracking

## Integration Points

### Memory System Integration
- Voice commands stored as episodic memories
- Personality adaptation based on memory patterns
- Context-aware response generation

### Agent System Integration
- Voice-driven agent selection
- Personality-specific voice profiles
- Dynamic voice adaptation based on agent characteristics

### WebSocket Integration
- Real-time audio streaming
- Live transcription updates
- Voice response broadcasting

## Performance Considerations

### Latency Optimization
- Streaming STT for real-time processing
- Chunked audio processing
- Optimized buffer management
- GPU acceleration support

### Quality vs Speed Trade-offs
- Configurable model sizes
- Dynamic quality adjustment
- Adaptive processing based on system load
- Quality monitoring and optimization

### Resource Management
- Memory usage optimization
- CPU/GPU resource balancing
- Network bandwidth management
- Concurrent request limiting

## Security and Privacy

### Data Protection
- Optional audio data encryption
- Configurable data retention policies
- Secure API key management
- Privacy-compliant logging

### Access Control
- Session-based authentication
- Rate limiting for voice requests
- User permission validation
- Secure WebSocket connections

## Monitoring and Analytics

### Performance Metrics
- STT/TTS processing times
- Transcription accuracy rates
- Audio quality scores
- System resource usage
- Error rates and recovery

### Quality Monitoring
- Continuous quality assessment
- Automatic optimization triggers
- Performance trend analysis
- Alert system for degradation

## Future Enhancements

### Planned Features
- Advanced voice cloning capabilities
- Multi-language conversation support
- Enhanced emotion detection
- Voice biometric authentication
- Advanced noise cancellation

### Scalability Improvements
- Distributed processing support
- Load balancing for voice services
- Caching optimization
- Performance analytics dashboard

## Getting Started

### Installation
```bash
# Install voice processing dependencies
pip install -r requirements-voice.txt

# Optional: Install GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage
```python
from src.voice import VoicePipeline, VoiceConfiguration

# Initialize voice pipeline
config = VoiceConfiguration()
pipeline = VoicePipeline(config)
await pipeline.initialize()

# Process voice command
command = await pipeline.process_voice_command(audio_data)
response = await pipeline.generate_voice_response(response_text)
```

### Configuration
```python
from src.voice import VoiceConfigurationManager, VoiceConfigurationPreset

# Load configuration manager
config_manager = VoiceConfigurationManager()

# Create production configuration
config = config_manager.create_configuration(
    "prod_config",
    "Production Config",
    preset=VoiceConfigurationPreset.PRODUCTION
)
```

This voice processing architecture provides a robust, scalable foundation for natural voice interactions with AI agents, supporting real-time processing, personality adaptation, and comprehensive quality optimization.
"""
Voice processing pipeline for complete voice interaction workflow
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, AsyncIterable, Union
from pathlib import Path

from pydantic import BaseModel, Field

from .stt_processor import STTProcessor, STTConfiguration, TranscriptionResult, create_stt_processor
from .tts_processor import TTSProcessor, TTSConfiguration, SynthesisResult, VoiceProfile, create_tts_processor

logger = logging.getLogger(__name__)


class VoiceCommandType(str, Enum):
    """Types of voice commands"""
    CONVERSATION = "conversation"
    SYSTEM_CONTROL = "system_control"
    AGENT_SELECTION = "agent_selection"
    MEMORY_QUERY = "memory_query"
    HOME_AUTOMATION = "home_automation"
    VOICE_CONTROL = "voice_control"


class VoiceResponseType(str, Enum):
    """Types of voice responses"""
    DIRECT_ANSWER = "direct_answer"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_STATUS = "system_status"
    ERROR_MESSAGE = "error_message"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"


class ProcessingState(str, Enum):
    """Voice processing states"""
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    PROCESSING = "processing"
    GENERATING = "generating"
    SPEAKING = "speaking"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class VoiceCommand:
    """Voice command data structure"""
    text: str
    command_type: VoiceCommandType
    confidence: float
    timestamp: float
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_target: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    transcription_result: Optional[TranscriptionResult] = None


@dataclass
class VoiceResponse:
    """Voice response data structure"""
    text: str
    response_type: VoiceResponseType
    voice_profile: VoiceProfile
    timestamp: float
    processing_time: float
    session_id: Optional[str] = None
    agent_source: Optional[str] = None
    synthesis_result: Optional[SynthesisResult] = None
    metadata: Optional[Dict[str, Any]] = None


class VoiceConfiguration(BaseModel):
    """Configuration for voice pipeline"""
    # STT Configuration
    stt_config: STTConfiguration = Field(default_factory=STTConfiguration)

    # TTS Configuration
    tts_config: TTSConfiguration = Field(default_factory=TTSConfiguration)

    # Voice Processing Settings
    voice_activation_threshold: float = Field(default=0.7, description="Voice activation confidence threshold")
    command_timeout: float = Field(default=10.0, description="Command processing timeout in seconds")
    silence_timeout: float = Field(default=3.0, description="Silence timeout for end of speech")

    # Voice Activity Detection
    vad_enabled: bool = Field(default=True, description="Enable voice activity detection")
    vad_sensitivity: float = Field(default=0.5, description="VAD sensitivity (0.0-1.0)")

    # Response Settings
    interrupt_enabled: bool = Field(default=True, description="Allow interrupting TTS playback")
    response_delay: float = Field(default=0.5, description="Delay before starting response")

    # Audio Settings
    input_sample_rate: int = Field(default=16000, description="Input audio sample rate")
    output_sample_rate: int = Field(default=22050, description="Output audio sample rate")

    # Personality Adaptation
    adapt_voice_to_personality: bool = Field(default=True, description="Adapt voice characteristics to agent personality")
    voice_emotion_mapping: bool = Field(default=True, description="Map emotional state to voice parameters")


class VoicePipeline:
    """Complete voice processing pipeline"""

    def __init__(self, config: VoiceConfiguration):
        self.config = config
        self.stt_processor: Optional[STTProcessor] = None
        self.tts_processor: Optional[TTSProcessor] = None

        # Voice profiles mapping for different agents/personalities
        self.voice_profiles: Dict[str, VoiceProfile] = {}

        # Processing state
        self.current_state = ProcessingState.LISTENING
        self.is_initialized = False

        # Callbacks
        self.command_callbacks: Dict[VoiceCommandType, List[Callable]] = {}
        self.state_change_callbacks: List[Callable] = []

        # Performance metrics
        self.metrics = {
            "total_commands": 0,
            "successful_commands": 0,
            "average_processing_time": 0.0,
            "transcription_accuracy": 0.0
        }

    async def initialize(self) -> None:
        """Initialize the voice pipeline"""
        try:
            logger.info("Initializing voice pipeline...")

            # Initialize STT processor
            self.stt_processor = create_stt_processor(self.config.stt_config)
            await self.stt_processor.initialize()
            logger.info("STT processor initialized")

            # Initialize TTS processor
            self.tts_processor = create_tts_processor(self.config.tts_config)
            await self.tts_processor.initialize()
            logger.info("TTS processor initialized")

            # Load voice profiles
            await self._load_voice_profiles()

            self.is_initialized = True
            await self._set_state(ProcessingState.LISTENING)

            logger.info("Voice pipeline initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize voice pipeline: {e}")
            await self._set_state(ProcessingState.ERROR)
            raise

    async def _load_voice_profiles(self) -> None:
        """Load available voice profiles"""
        try:
            if self.tts_processor:
                available_voices = await self.tts_processor.get_available_voices()

                # Create default mappings for different personality types
                personality_voice_mapping = {
                    "assistant": "helpful",
                    "creative": "enthusiastic",
                    "analytical": "professional",
                    "empathetic": "warm",
                    "humorous": "playful"
                }

                for personality, voice_type in personality_voice_mapping.items():
                    # Find suitable voice for each personality
                    suitable_voice = self._find_suitable_voice(available_voices, voice_type)
                    if suitable_voice:
                        self.voice_profiles[personality] = suitable_voice

                logger.info(f"Loaded {len(self.voice_profiles)} voice profiles for personalities")

        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")

    def _find_suitable_voice(self, voices: List[VoiceProfile], voice_type: str) -> Optional[VoiceProfile]:
        """Find suitable voice based on type characteristics"""
        # Simple heuristic to match voice type to available voices
        for voice in voices:
            if voice_type.lower() in voice.name.lower() or \
               (voice.description and voice_type.lower() in voice.description.lower()):
                return voice

        # Return first available voice as fallback
        return voices[0] if voices else None

    async def process_voice_command(
        self,
        audio_data: Union[bytes, str, Path],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> VoiceCommand:
        """Process voice input to extract command"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            await self._set_state(ProcessingState.TRANSCRIBING)

            # Transcribe audio to text
            transcription = await self.stt_processor.transcribe_audio(audio_data)

            if transcription.confidence < self.config.voice_activation_threshold:
                logger.warning(f"Low confidence transcription: {transcription.confidence}")

            await self._set_state(ProcessingState.PROCESSING)

            # Analyze command type and extract parameters
            command = await self._analyze_command(
                transcription,
                session_id=session_id,
                user_id=user_id
            )

            # Update metrics
            self.metrics["total_commands"] += 1
            if command.confidence >= self.config.voice_activation_threshold:
                self.metrics["successful_commands"] += 1

            processing_time = time.time() - start_time
            self.metrics["average_processing_time"] = (
                (self.metrics["average_processing_time"] * (self.metrics["total_commands"] - 1) + processing_time) /
                self.metrics["total_commands"]
            )

            # Trigger command callbacks
            await self._trigger_command_callbacks(command)

            logger.info(f"Voice command processed: '{command.text[:50]}...' ({command.command_type})")

            return command

        except Exception as e:
            await self._set_state(ProcessingState.ERROR)
            logger.error(f"Voice command processing failed: {e}")
            raise

    async def _analyze_command(
        self,
        transcription: TranscriptionResult,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> VoiceCommand:
        """Analyze transcribed text to determine command type and parameters"""
        text = transcription.text.lower().strip()

        # Command type detection patterns
        command_patterns = {
            VoiceCommandType.AGENT_SELECTION: [
                "talk to", "switch to", "use agent", "activate", "hey"
            ],
            VoiceCommandType.SYSTEM_CONTROL: [
                "stop", "pause", "resume", "restart", "shutdown", "status"
            ],
            VoiceCommandType.MEMORY_QUERY: [
                "remember", "recall", "what did", "tell me about", "search"
            ],
            VoiceCommandType.HOME_AUTOMATION: [
                "turn on", "turn off", "set", "adjust", "lights", "temperature"
            ],
            VoiceCommandType.VOICE_CONTROL: [
                "volume", "speak", "voice", "louder", "quieter", "faster", "slower"
            ]
        }

        detected_type = VoiceCommandType.CONVERSATION  # Default
        agent_target = None
        parameters = {}

        # Pattern matching for command type
        for cmd_type, patterns in command_patterns.items():
            if any(pattern in text for pattern in patterns):
                detected_type = cmd_type
                break

        # Extract agent target for agent selection commands
        if detected_type == VoiceCommandType.AGENT_SELECTION:
            agent_names = ["alice", "bob", "charlie", "dana", "assistant", "creative", "analytical"]
            for name in agent_names:
                if name in text:
                    agent_target = name
                    break

        # Extract parameters for specific command types
        if detected_type == VoiceCommandType.HOME_AUTOMATION:
            parameters = self._extract_home_automation_params(text)
        elif detected_type == VoiceCommandType.VOICE_CONTROL:
            parameters = self._extract_voice_control_params(text)

        return VoiceCommand(
            text=transcription.text,
            command_type=detected_type,
            confidence=transcription.confidence,
            timestamp=time.time(),
            session_id=session_id,
            user_id=user_id,
            agent_target=agent_target,
            parameters=parameters,
            transcription_result=transcription
        )

    def _extract_home_automation_params(self, text: str) -> Dict[str, Any]:
        """Extract home automation parameters from text"""
        params = {}

        # Extract device type
        devices = ["lights", "fan", "ac", "heater", "music", "tv"]
        for device in devices:
            if device in text:
                params["device"] = device
                break

        # Extract action
        if "turn on" in text or "on" in text:
            params["action"] = "on"
        elif "turn off" in text or "off" in text:
            params["action"] = "off"
        elif "set" in text or "adjust" in text:
            params["action"] = "set"

        # Extract location
        locations = ["living room", "bedroom", "kitchen", "bathroom"]
        for location in locations:
            if location in text:
                params["location"] = location
                break

        return params

    def _extract_voice_control_params(self, text: str) -> Dict[str, Any]:
        """Extract voice control parameters from text"""
        params = {}

        if "volume" in text:
            if "up" in text or "louder" in text:
                params["volume_change"] = "increase"
            elif "down" in text or "quieter" in text:
                params["volume_change"] = "decrease"

        if "speed" in text or "rate" in text:
            if "faster" in text or "speed up" in text:
                params["speed_change"] = "increase"
            elif "slower" in text or "slow down" in text:
                params["speed_change"] = "decrease"

        return params

    async def generate_voice_response(
        self,
        text: str,
        response_type: VoiceResponseType = VoiceResponseType.DIRECT_ANSWER,
        agent_source: Optional[str] = None,
        session_id: Optional[str] = None,
        voice_profile: Optional[VoiceProfile] = None,
        **kwargs
    ) -> VoiceResponse:
        """Generate voice response from text"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            await self._set_state(ProcessingState.GENERATING)

            # Select appropriate voice profile
            if voice_profile is None:
                voice_profile = await self._select_voice_profile(agent_source, response_type)

            # Adapt voice based on personality if enabled
            if self.config.adapt_voice_to_personality and agent_source:
                voice_profile = await self._adapt_voice_to_personality(voice_profile, agent_source)

            await self._set_state(ProcessingState.SPEAKING)

            # Synthesize speech
            synthesis_result = await self.tts_processor.synthesize_text(
                text,
                voice_profile=voice_profile,
                **kwargs
            )

            processing_time = time.time() - start_time

            response = VoiceResponse(
                text=text,
                response_type=response_type,
                voice_profile=voice_profile,
                timestamp=time.time(),
                processing_time=processing_time,
                session_id=session_id,
                agent_source=agent_source,
                synthesis_result=synthesis_result,
                metadata={
                    "audio_size": len(synthesis_result.audio_data),
                    "estimated_duration": synthesis_result.duration
                }
            )

            await self._set_state(ProcessingState.COMPLETE)

            logger.info(f"Voice response generated: {len(text)} chars in {processing_time:.2f}s")

            return response

        except Exception as e:
            await self._set_state(ProcessingState.ERROR)
            logger.error(f"Voice response generation failed: {e}")
            raise

    async def _select_voice_profile(
        self,
        agent_source: Optional[str],
        response_type: VoiceResponseType
    ) -> VoiceProfile:
        """Select appropriate voice profile based on agent and response type"""
        # Use agent-specific voice if available
        if agent_source and agent_source in self.voice_profiles:
            return self.voice_profiles[agent_source]

        # Use response type specific voice mapping
        response_voice_mapping = {
            VoiceResponseType.ERROR_MESSAGE: "professional",
            VoiceResponseType.SYSTEM_STATUS: "professional",
            VoiceResponseType.CONFIRMATION: "warm",
            VoiceResponseType.CLARIFICATION: "helpful"
        }

        voice_type = response_voice_mapping.get(response_type, "assistant")
        if voice_type in self.voice_profiles:
            return self.voice_profiles[voice_type]

        # Fallback to default voice
        if self.voice_profiles:
            return list(self.voice_profiles.values())[0]

        # Create emergency fallback voice
        from .tts_processor import VoiceProfile, VoiceGender, VoiceAge
        return VoiceProfile(
            voice_id=self.config.tts_config.voice_id,
            name="Default",
            gender=VoiceGender.FEMALE,
            age=VoiceAge.MIDDLE_AGED
        )

    async def _adapt_voice_to_personality(
        self,
        voice_profile: VoiceProfile,
        agent_source: str
    ) -> VoiceProfile:
        """Adapt voice characteristics based on agent personality"""
        # This would integrate with the personality system
        # For now, return the original profile
        # TODO: Integrate with personality.big_five_model to adjust voice parameters
        return voice_profile

    async def process_streaming_voice(
        self,
        audio_stream: AsyncIterable[bytes],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AsyncIterable[VoiceCommand]:
        """Process streaming voice input"""
        if not self.is_initialized:
            await self.initialize()

        try:
            await self._set_state(ProcessingState.TRANSCRIBING)

            async for transcription in self.stt_processor.transcribe_streaming(audio_stream):
                if transcription.text.strip():  # Only process non-empty transcriptions
                    command = await self._analyze_command(
                        transcription,
                        session_id=session_id,
                        user_id=user_id
                    )

                    await self._trigger_command_callbacks(command)
                    yield command

        except Exception as e:
            await self._set_state(ProcessingState.ERROR)
            logger.error(f"Streaming voice processing failed: {e}")
            raise

    def register_command_callback(
        self,
        command_type: VoiceCommandType,
        callback: Callable[[VoiceCommand], None]
    ) -> None:
        """Register callback for specific command type"""
        if command_type not in self.command_callbacks:
            self.command_callbacks[command_type] = []
        self.command_callbacks[command_type].append(callback)

    def register_state_change_callback(self, callback: Callable[[ProcessingState], None]) -> None:
        """Register callback for state changes"""
        self.state_change_callbacks.append(callback)

    async def _trigger_command_callbacks(self, command: VoiceCommand) -> None:
        """Trigger callbacks for command type"""
        if command.command_type in self.command_callbacks:
            for callback in self.command_callbacks[command.command_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(command)
                    else:
                        callback(command)
                except Exception as e:
                    logger.error(f"Command callback error: {e}")

    async def _set_state(self, new_state: ProcessingState) -> None:
        """Set processing state and trigger callbacks"""
        old_state = self.current_state
        self.current_state = new_state

        if old_state != new_state:
            logger.debug(f"Voice pipeline state: {old_state} -> {new_state}")

            for callback in self.state_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(new_state)
                    else:
                        callback(new_state)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return {
            **self.metrics,
            "current_state": self.current_state.value,
            "is_initialized": self.is_initialized,
            "available_voices": len(self.voice_profiles)
        }

    async def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        logger.info("Cleaning up voice pipeline...")

        if self.stt_processor:
            await self.stt_processor.cleanup()

        if self.tts_processor:
            await self.tts_processor.cleanup()

        self.voice_profiles.clear()
        self.command_callbacks.clear()
        self.state_change_callbacks.clear()

        self.is_initialized = False
        await self._set_state(ProcessingState.COMPLETE)

        logger.info("Voice pipeline cleanup complete")
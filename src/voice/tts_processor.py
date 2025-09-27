"""
Text-to-Speech processing with ElevenLabs integration
"""

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Union, AsyncIterable, List
from pathlib import Path

import aiohttp
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TTSProvider(str, Enum):
    """Available TTS providers"""
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"


class VoiceGender(str, Enum):
    """Voice gender options"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class VoiceAge(str, Enum):
    """Voice age categories"""
    YOUNG = "young"
    MIDDLE_AGED = "middle_aged"
    OLD = "old"


class AudioFormat(str, Enum):
    """Audio output formats"""
    MP3 = "mp3"
    WAV = "wav"
    PCM = "pcm"
    FLAC = "flac"
    OGG = "ogg"


class TTSQuality(str, Enum):
    """TTS quality levels"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    voice_id: str
    name: str
    gender: VoiceGender
    age: VoiceAge
    accent: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    similarity_boost: float = 0.5
    stability: float = 0.5
    style: float = 0.0
    use_speaker_boost: bool = True


@dataclass
class SynthesisResult:
    """Result of TTS synthesis"""
    audio_data: bytes
    format: AudioFormat
    duration: Optional[float]
    processing_time: float
    voice_profile: VoiceProfile
    text: str
    metadata: Optional[Dict[str, Any]] = None


class TTSConfiguration(BaseModel):
    """Configuration for TTS processing"""
    provider: TTSProvider = Field(default=TTSProvider.ELEVENLABS, description="TTS provider")
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", description="Voice ID (default: ElevenLabs Rachel)")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    model_id: str = Field(default="eleven_monolingual_v1", description="TTS model ID")
    quality: TTSQuality = Field(default=TTSQuality.BALANCED, description="Quality vs speed tradeoff")
    format: AudioFormat = Field(default=AudioFormat.MP3, description="Output audio format")
    sample_rate: int = Field(default=22050, description="Audio sample rate")

    # Voice settings
    stability: float = Field(default=0.5, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: float = Field(default=0.5, ge=0.0, le=1.0, description="Similarity boost")
    style: float = Field(default=0.0, ge=0.0, le=1.0, description="Style exaggeration")
    use_speaker_boost: bool = Field(default=True, description="Enable speaker boost")

    # Processing settings
    optimize_streaming_latency: int = Field(default=0, ge=0, le=4, description="Streaming latency optimization")
    output_format: str = Field(default="mp3_22050_32", description="Output format specification")

    # Advanced settings
    chunk_length_schedule: List[int] = Field(default=[120, 160, 250, 290], description="Chunk length schedule")
    enable_logging: bool = Field(default=True, description="Enable request logging")


class TTSProcessor(ABC):
    """Abstract base class for TTS processors"""

    def __init__(self, config: TTSConfiguration):
        self.config = config
        self._session = None
        self._voice_profiles: Dict[str, VoiceProfile] = {}

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS processor"""
        pass

    @abstractmethod
    async def synthesize_text(
        self,
        text: str,
        voice_profile: Optional[VoiceProfile] = None,
        **kwargs
    ) -> SynthesisResult:
        """Synthesize text to audio"""
        pass

    @abstractmethod
    async def synthesize_streaming(
        self,
        text: str,
        voice_profile: Optional[VoiceProfile] = None,
        **kwargs
    ) -> AsyncIterable[bytes]:
        """Synthesize text to streaming audio"""
        pass

    @abstractmethod
    async def get_available_voices(self) -> List[VoiceProfile]:
        """Get list of available voices"""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self._session and not self._session.closed:
            await self._session.close()


class ElevenLabsTTSProcessor(TTSProcessor):
    """ElevenLabs TTS processor"""

    def __init__(self, config: TTSConfiguration):
        super().__init__(config)
        self.base_url = "https://api.elevenlabs.io/v1"

    async def initialize(self) -> None:
        """Initialize the ElevenLabs TTS processor"""
        if not self.config.api_key:
            raise ValueError("ElevenLabs API key is required")

        self._session = aiohttp.ClientSession(
            headers={
                "xi-api-key": self.config.api_key,
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=60)
        )

        # Load available voices
        try:
            await self._load_voices()
            logger.info(f"ElevenLabs TTS initialized with {len(self._voice_profiles)} voices")
        except Exception as e:
            logger.error(f"Failed to load voices: {e}")
            raise

    async def _load_voices(self) -> None:
        """Load available voices from ElevenLabs"""
        try:
            async with self._session.get(f"{self.base_url}/voices") as response:
                if response.status == 200:
                    data = await response.json()
                    self._voice_profiles = {}

                    for voice_data in data.get("voices", []):
                        voice_profile = VoiceProfile(
                            voice_id=voice_data["voice_id"],
                            name=voice_data["name"],
                            gender=self._infer_gender(voice_data),
                            age=self._infer_age(voice_data),
                            accent=voice_data.get("labels", {}).get("accent"),
                            description=voice_data.get("labels", {}).get("description"),
                            preview_url=voice_data.get("preview_url"),
                            similarity_boost=self.config.similarity_boost,
                            stability=self.config.stability,
                            style=self.config.style,
                            use_speaker_boost=self.config.use_speaker_boost
                        )
                        self._voice_profiles[voice_data["voice_id"]] = voice_profile

                else:
                    logger.error(f"Failed to load voices: {response.status}")

        except Exception as e:
            logger.error(f"Error loading voices: {e}")
            raise

    def _infer_gender(self, voice_data: Dict[str, Any]) -> VoiceGender:
        """Infer gender from voice data"""
        labels = voice_data.get("labels", {})
        gender = labels.get("gender", "").lower()

        if "male" in gender:
            return VoiceGender.MALE
        elif "female" in gender:
            return VoiceGender.FEMALE
        else:
            return VoiceGender.NEUTRAL

    def _infer_age(self, voice_data: Dict[str, Any]) -> VoiceAge:
        """Infer age from voice data"""
        labels = voice_data.get("labels", {})
        age = labels.get("age", "").lower()

        if any(word in age for word in ["young", "teen", "child"]):
            return VoiceAge.YOUNG
        elif any(word in age for word in ["old", "elderly", "senior"]):
            return VoiceAge.OLD
        else:
            return VoiceAge.MIDDLE_AGED

    async def synthesize_text(
        self,
        text: str,
        voice_profile: Optional[VoiceProfile] = None,
        **kwargs
    ) -> SynthesisResult:
        """Synthesize text using ElevenLabs API"""
        if self._session is None:
            await self.initialize()

        start_time = time.time()

        # Use provided voice profile or default
        if voice_profile is None:
            voice_id = self.config.voice_id
            voice_profile = self._voice_profiles.get(voice_id)
            if voice_profile is None:
                # Create default voice profile
                voice_profile = VoiceProfile(
                    voice_id=voice_id,
                    name="Default",
                    gender=VoiceGender.FEMALE,
                    age=VoiceAge.MIDDLE_AGED,
                    stability=self.config.stability,
                    similarity_boost=self.config.similarity_boost,
                    style=self.config.style,
                    use_speaker_boost=self.config.use_speaker_boost
                )
        else:
            voice_id = voice_profile.voice_id

        # Prepare voice settings
        voice_settings = {
            "stability": voice_profile.stability,
            "similarity_boost": voice_profile.similarity_boost,
            "style": voice_profile.style,
            "use_speaker_boost": voice_profile.use_speaker_boost
        }

        # Prepare request payload
        payload = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": voice_settings
        }

        # Add optional parameters
        if self.config.optimize_streaming_latency > 0:
            payload["optimize_streaming_latency"] = self.config.optimize_streaming_latency

        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}"

            async with self._session.post(url, json=payload) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    processing_time = time.time() - start_time

                    # Calculate audio duration (approximation for MP3)
                    duration = self._estimate_audio_duration(audio_data, text)

                    result = SynthesisResult(
                        audio_data=audio_data,
                        format=self.config.format,
                        duration=duration,
                        processing_time=processing_time,
                        voice_profile=voice_profile,
                        text=text,
                        metadata={
                            "model_id": self.config.model_id,
                            "voice_settings": voice_settings,
                            "status_code": response.status,
                            "audio_size": len(audio_data)
                        }
                    )

                    logger.debug(f"TTS synthesis completed in {processing_time:.2f}s for {len(text)} characters")
                    return result

                else:
                    error_text = await response.text()
                    logger.error(f"TTS synthesis failed: {response.status} - {error_text}")
                    raise Exception(f"TTS synthesis failed: {response.status} - {error_text}")

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"TTS synthesis error after {processing_time:.2f}s: {e}")
            raise

    async def synthesize_streaming(
        self,
        text: str,
        voice_profile: Optional[VoiceProfile] = None,
        **kwargs
    ) -> AsyncIterable[bytes]:
        """Synthesize text to streaming audio using ElevenLabs streaming API"""
        if self._session is None:
            await self.initialize()

        # Use provided voice profile or default
        if voice_profile is None:
            voice_id = self.config.voice_id
            voice_profile = self._voice_profiles.get(voice_id, VoiceProfile(
                voice_id=voice_id,
                name="Default",
                gender=VoiceGender.FEMALE,
                age=VoiceAge.MIDDLE_AGED,
                stability=self.config.stability,
                similarity_boost=self.config.similarity_boost,
                style=self.config.style,
                use_speaker_boost=self.config.use_speaker_boost
            ))
        else:
            voice_id = voice_profile.voice_id

        # Prepare voice settings
        voice_settings = {
            "stability": voice_profile.stability,
            "similarity_boost": voice_profile.similarity_boost,
            "style": voice_profile.style,
            "use_speaker_boost": voice_profile.use_speaker_boost
        }

        # Prepare request payload
        payload = {
            "text": text,
            "model_id": self.config.model_id,
            "voice_settings": voice_settings,
            "optimize_streaming_latency": self.config.optimize_streaming_latency,
            "output_format": self.config.output_format
        }

        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}/stream"

            async with self._session.post(url, json=payload) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(8192):
                        if chunk:
                            yield chunk
                else:
                    error_text = await response.text()
                    logger.error(f"Streaming TTS failed: {response.status} - {error_text}")
                    raise Exception(f"Streaming TTS failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Streaming TTS error: {e}")
            raise

    async def get_available_voices(self) -> List[VoiceProfile]:
        """Get list of available voices"""
        if not self._voice_profiles:
            await self._load_voices()
        return list(self._voice_profiles.values())

    async def clone_voice(
        self,
        name: str,
        description: str,
        audio_files: List[Union[bytes, Path]],
        **kwargs
    ) -> VoiceProfile:
        """Clone a voice from audio samples"""
        if self._session is None:
            await self.initialize()

        try:
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('name', name)
            data.add_field('description', description)

            # Add audio files
            for i, audio_file in enumerate(audio_files):
                if isinstance(audio_file, bytes):
                    data.add_field(
                        'files',
                        audio_file,
                        filename=f'sample_{i}.wav',
                        content_type='audio/wav'
                    )
                elif isinstance(audio_file, Path):
                    with open(audio_file, 'rb') as f:
                        data.add_field(
                            'files',
                            f.read(),
                            filename=audio_file.name,
                            content_type='audio/wav'
                        )

            url = f"{self.base_url}/voices/add"

            async with self._session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    voice_id = result.get("voice_id")

                    # Create voice profile
                    voice_profile = VoiceProfile(
                        voice_id=voice_id,
                        name=name,
                        gender=VoiceGender.NEUTRAL,  # Cannot infer from cloned voice
                        age=VoiceAge.MIDDLE_AGED,
                        description=description,
                        stability=self.config.stability,
                        similarity_boost=self.config.similarity_boost,
                        style=self.config.style,
                        use_speaker_boost=self.config.use_speaker_boost
                    )

                    # Add to voice profiles
                    self._voice_profiles[voice_id] = voice_profile

                    logger.info(f"Voice cloned successfully: {name} (ID: {voice_id})")
                    return voice_profile

                else:
                    error_text = await response.text()
                    logger.error(f"Voice cloning failed: {response.status} - {error_text}")
                    raise Exception(f"Voice cloning failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Voice cloning error: {e}")
            raise

    def _estimate_audio_duration(self, audio_data: bytes, text: str) -> Optional[float]:
        """Estimate audio duration based on text and audio data size"""
        try:
            # Rough estimation: average speaking rate is ~150 words per minute
            words = len(text.split())
            estimated_duration = (words / 150) * 60  # seconds

            # Adjust based on audio data size (very rough approximation)
            # MP3 at ~128kbps: ~16KB per second
            size_based_duration = len(audio_data) / 16000

            # Use average of both estimates
            return (estimated_duration + size_based_duration) / 2

        except Exception:
            return None


# Factory function for creating TTS processors
def create_tts_processor(config: TTSConfiguration) -> TTSProcessor:
    """Create a TTS processor based on configuration"""
    if config.provider == TTSProvider.ELEVENLABS:
        return ElevenLabsTTSProcessor(config)
    else:
        raise ValueError(f"Unsupported TTS provider: {config.provider}")
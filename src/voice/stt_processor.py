"""
Speech-to-Text processing with Whisper integration
"""

import asyncio
import io
import logging
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO

import numpy as np
import torch
import whisper
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class STTModel(str, Enum):
    """Available STT models"""
    WHISPER_TINY = "tiny"
    WHISPER_BASE = "base"
    WHISPER_SMALL = "small"
    WHISPER_MEDIUM = "medium"
    WHISPER_LARGE = "large"
    WHISPER_LARGE_V2 = "large-v2"
    WHISPER_LARGE_V3 = "large-v3"


class TranscriptionQuality(str, Enum):
    """Transcription quality levels"""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


@dataclass
class TranscriptionResult:
    """Result of STT transcription"""
    text: str
    confidence: float
    processing_time: float
    language: Optional[str] = None
    segments: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None


class STTConfiguration(BaseModel):
    """Configuration for STT processing"""
    model: STTModel = Field(default=STTModel.WHISPER_BASE, description="STT model to use")
    quality: TranscriptionQuality = Field(default=TranscriptionQuality.BALANCED, description="Quality vs speed tradeoff")
    language: Optional[str] = Field(default=None, description="Expected language (auto-detect if None)")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    beam_size: int = Field(default=5, description="Beam search size")
    best_of: int = Field(default=5, description="Number of candidates for beam search")
    patience: float = Field(default=1.0, description="Patience for beam search")
    length_penalty: float = Field(default=1.0, description="Length penalty for beam search")
    suppress_tokens: str = Field(default="-1", description="Tokens to suppress")
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for context")
    condition_on_previous_text: bool = Field(default=True, description="Use previous text as context")
    fp16: bool = Field(default=True, description="Use FP16 precision")
    compression_ratio_threshold: float = Field(default=2.4, description="Compression ratio threshold")
    logprob_threshold: float = Field(default=-1.0, description="Log probability threshold")
    no_speech_threshold: float = Field(default=0.6, description="No speech threshold")


class STTProcessor(ABC):
    """Abstract base class for STT processors"""

    def __init__(self, config: STTConfiguration):
        self.config = config
        self._model = None
        self._device = None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the STT processor"""
        pass

    @abstractmethod
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, np.ndarray, str, Path, BinaryIO],
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio data to text"""
        pass

    @abstractmethod
    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterable[bytes],
        **kwargs
    ) -> AsyncIterable[TranscriptionResult]:
        """Transcribe streaming audio data"""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class WhisperSTTProcessor(STTProcessor):
    """Whisper-based STT processor"""

    def __init__(self, config: STTConfiguration):
        super().__init__(config)
        self._device = self._get_optimal_device()

    def _get_optimal_device(self) -> str:
        """Get the optimal device for processing"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    async def initialize(self) -> None:
        """Initialize the Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.config.model.value} on {self._device}")
            start_time = time.time()

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: whisper.load_model(
                    self.config.model.value,
                    device=self._device
                )
            )

            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise

    def _prepare_transcribe_options(self, **kwargs) -> Dict[str, Any]:
        """Prepare transcription options"""
        options = {
            "language": self.config.language,
            "temperature": self.config.temperature,
            "beam_size": self.config.beam_size,
            "best_of": self.config.best_of,
            "patience": self.config.patience,
            "length_penalty": self.config.length_penalty,
            "suppress_tokens": self.config.suppress_tokens,
            "initial_prompt": self.config.initial_prompt,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "fp16": self.config.fp16 and self._device != "cpu",
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "logprob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
        }

        # Override with any provided kwargs
        options.update(kwargs)

        # Remove None values
        return {k: v for k, v in options.items() if v is not None}

    async def transcribe_audio(
        self,
        audio_data: Union[bytes, np.ndarray, str, Path, BinaryIO],
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio data using Whisper"""
        if self._model is None:
            await self.initialize()

        start_time = time.time()

        try:
            # Prepare transcription options
            options = self._prepare_transcribe_options(**kwargs)

            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                audio_path = str(audio_data)
                logger.debug(f"Transcribing audio file: {audio_path}")
            elif isinstance(audio_data, bytes):
                # Save bytes to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    audio_path = tmp_file.name
                logger.debug(f"Transcribing audio from bytes (temp file: {audio_path})")
            elif isinstance(audio_data, np.ndarray):
                # Save numpy array to temporary file
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, 16000)  # Assume 16kHz sample rate
                    audio_path = tmp_file.name
                logger.debug(f"Transcribing audio from numpy array (temp file: {audio_path})")
            elif hasattr(audio_data, 'read'):
                # Handle file-like objects
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_data.read())
                    audio_path = tmp_file.name
                logger.debug(f"Transcribing audio from file object (temp file: {audio_path})")
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(audio_path, **options)
            )

            processing_time = time.time() - start_time

            # Calculate confidence score (approximation based on logprobs if available)
            confidence = self._calculate_confidence(result)

            # Clean up temporary file if created
            if isinstance(audio_data, (bytes, np.ndarray)) or hasattr(audio_data, 'read'):
                try:
                    Path(audio_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file {audio_path}: {e}")

            transcription_result = TranscriptionResult(
                text=result["text"].strip(),
                confidence=confidence,
                processing_time=processing_time,
                language=result.get("language"),
                segments=result.get("segments"),
                metadata={
                    "model": self.config.model.value,
                    "device": self._device,
                    "options": options
                }
            )

            logger.debug(f"Transcription completed in {processing_time:.2f}s: '{transcription_result.text[:100]}...'")
            return transcription_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Transcription failed after {processing_time:.2f}s: {e}")
            raise

    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from transcription result"""
        try:
            # If segments are available, use average logprob
            if "segments" in result and result["segments"]:
                logprobs = []
                for segment in result["segments"]:
                    if "avg_logprob" in segment:
                        logprobs.append(segment["avg_logprob"])

                if logprobs:
                    avg_logprob = sum(logprobs) / len(logprobs)
                    # Convert logprob to confidence (rough approximation)
                    confidence = max(0.0, min(1.0, (avg_logprob + 1.0) / 1.0))
                    return confidence

            # Fallback: use compression ratio as confidence indicator
            if "segments" in result and result["segments"]:
                total_length = sum(len(seg.get("text", "")) for seg in result["segments"])
                if total_length > 0:
                    compression_ratio = len(result["text"]) / total_length
                    # Lower compression ratio = higher confidence
                    confidence = max(0.1, min(1.0, 1.0 / max(1.0, compression_ratio)))
                    return confidence

            # Default confidence
            return 0.5

        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterable[bytes],
        chunk_duration: float = 5.0,
        overlap_duration: float = 1.0,
        **kwargs
    ) -> AsyncIterable[TranscriptionResult]:
        """Transcribe streaming audio data with chunking and overlap"""
        if self._model is None:
            await self.initialize()

        buffer = bytearray()
        chunk_size = int(16000 * chunk_duration * 2)  # 16kHz, 16-bit samples
        overlap_size = int(16000 * overlap_duration * 2)

        try:
            async for audio_chunk in audio_stream:
                buffer.extend(audio_chunk)

                # Process complete chunks
                while len(buffer) >= chunk_size:
                    chunk_data = bytes(buffer[:chunk_size])

                    try:
                        result = await self.transcribe_audio(chunk_data, **kwargs)
                        if result.text.strip():  # Only yield non-empty transcriptions
                            yield result
                    except Exception as e:
                        logger.error(f"Failed to transcribe audio chunk: {e}")

                    # Keep overlap for context
                    buffer = buffer[chunk_size - overlap_size:]

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise
        finally:
            # Process remaining buffer
            if len(buffer) > overlap_size:
                try:
                    result = await self.transcribe_audio(bytes(buffer), **kwargs)
                    if result.text.strip():
                        yield result
                except Exception as e:
                    logger.error(f"Failed to transcribe final audio chunk: {e}")


# Factory function for creating STT processors
def create_stt_processor(config: STTConfiguration) -> STTProcessor:
    """Create an STT processor based on configuration"""
    if config.model.value.startswith("whisper") or config.model in STTModel:
        return WhisperSTTProcessor(config)
    else:
        raise ValueError(f"Unsupported STT model: {config.model}")
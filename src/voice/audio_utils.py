"""
Audio processing utilities for voice pipeline
"""

import asyncio
import io
import logging
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union, List, AsyncIterable, BinaryIO
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    PCM = "pcm"


class AudioQuality(str, Enum):
    """Audio quality levels"""
    LOW = "low"          # 8kHz, mono
    MEDIUM = "medium"    # 16kHz, mono
    HIGH = "high"        # 22kHz, stereo
    STUDIO = "studio"    # 44.1kHz, stereo


@dataclass
class AudioMetadata:
    """Audio file metadata"""
    duration: float
    sample_rate: int
    channels: int
    format: AudioFormat
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    file_size: Optional[int] = None


@dataclass
class AudioSegment:
    """Audio segment with timing information"""
    audio_data: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    confidence: Optional[float] = None
    metadata: Optional[dict] = None


class AudioProcessingConfig(BaseModel):
    """Configuration for audio processing"""
    target_sample_rate: int = Field(default=16000, description="Target sample rate")
    target_channels: int = Field(default=1, description="Target number of channels (1=mono, 2=stereo)")
    normalize: bool = Field(default=True, description="Normalize audio levels")
    noise_reduction: bool = Field(default=True, description="Apply noise reduction")
    vad_enabled: bool = Field(default=True, description="Enable voice activity detection")

    # Noise reduction settings
    noise_reduction_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Noise reduction strength")
    spectral_gate_strength: float = Field(default=0.02, description="Spectral gating strength")

    # Audio enhancement
    dynamic_range_compression: bool = Field(default=True, description="Apply dynamic range compression")
    low_pass_filter: bool = Field(default=True, description="Apply low-pass filter")
    high_pass_filter: bool = Field(default=True, description="Apply high-pass filter")

    # Voice activity detection
    vad_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="VAD threshold")
    vad_window_length: float = Field(default=0.025, description="VAD window length in seconds")
    vad_hop_length: float = Field(default=0.010, description="VAD hop length in seconds")


class AudioProcessor:
    """Audio processing utilities"""

    def __init__(self, config: AudioProcessingConfig = None):
        self.config = config or AudioProcessingConfig()

    def load_audio(
        self,
        file_path: Union[str, Path, BinaryIO],
        target_sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate"""
        try:
            if isinstance(file_path, (str, Path)):
                audio_data, sr = librosa.load(
                    str(file_path),
                    sr=target_sr or self.config.target_sample_rate,
                    mono=mono
                )
            else:
                # Handle file-like objects
                audio_data, sr = sf.read(file_path)
                if target_sr and sr != target_sr:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                if mono and audio_data.ndim > 1:
                    audio_data = librosa.to_mono(audio_data.T)

            logger.debug(f"Loaded audio: {len(audio_data)} samples at {sr}Hz")
            return audio_data, sr

        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise

    def save_audio(
        self,
        audio_data: np.ndarray,
        file_path: Union[str, Path],
        sample_rate: int,
        format: AudioFormat = AudioFormat.WAV
    ) -> None:
        """Save audio data to file"""
        try:
            sf.write(str(file_path), audio_data, sample_rate, format=format.value)
            logger.debug(f"Saved audio to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise

    def convert_format(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        target_format: AudioFormat
    ) -> bytes:
        """Convert audio data to specified format"""
        try:
            # Use BytesIO to create in-memory file
            buffer = io.BytesIO()

            # Write audio data to buffer
            sf.write(buffer, audio_data, sample_rate, format=target_format.value)
            buffer.seek(0)

            return buffer.read()

        except Exception as e:
            logger.error(f"Failed to convert audio format: {e}")
            raise

    def resample_audio(
        self,
        audio_data: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_sr == target_sr:
            return audio_data

        try:
            resampled = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
            logger.debug(f"Resampled audio: {orig_sr}Hz -> {target_sr}Hz")
            return resampled

        except Exception as e:
            logger.error(f"Failed to resample audio: {e}")
            raise

    def normalize_audio(
        self,
        audio_data: np.ndarray,
        target_level: float = 0.9
    ) -> np.ndarray:
        """Normalize audio levels"""
        try:
            # Calculate current peak level
            peak_level = np.max(np.abs(audio_data))

            if peak_level > 0:
                # Scale audio to target level
                scale_factor = target_level / peak_level
                normalized = audio_data * scale_factor
            else:
                normalized = audio_data

            logger.debug(f"Normalized audio: peak {peak_level:.3f} -> {target_level:.3f}")
            return normalized

        except Exception as e:
            logger.error(f"Failed to normalize audio: {e}")
            raise

    def reduce_noise(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        strength: float = 0.5
    ) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            # Simple spectral subtraction-based noise reduction
            # Convert to frequency domain
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            phase = np.angle(fft)

            # Estimate noise spectrum from first 0.5 seconds
            noise_frames = int(0.5 * sample_rate)
            noise_spectrum = np.mean(magnitude[:noise_frames])

            # Apply spectral subtraction
            alpha = strength * 2.0  # Oversubtraction factor
            magnitude_clean = magnitude - alpha * noise_spectrum

            # Ensure magnitude doesn't go negative
            magnitude_clean = np.maximum(magnitude_clean, 0.1 * magnitude)

            # Reconstruct signal
            fft_clean = magnitude_clean * np.exp(1j * phase)
            audio_clean = np.fft.irfft(fft_clean, len(audio_data))

            logger.debug(f"Applied noise reduction (strength: {strength})")
            return audio_clean

        except Exception as e:
            logger.error(f"Failed to reduce noise: {e}")
            return audio_data

    def apply_filters(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        high_pass_freq: float = 80.0,
        low_pass_freq: float = 8000.0
    ) -> np.ndarray:
        """Apply high-pass and low-pass filters"""
        try:
            filtered = audio_data.copy()

            # High-pass filter (remove low-frequency noise)
            if self.config.high_pass_filter:
                sos_high = signal.butter(
                    4, high_pass_freq, btype='highpass',
                    fs=sample_rate, output='sos'
                )
                filtered = signal.sosfilt(sos_high, filtered)

            # Low-pass filter (anti-aliasing)
            if self.config.low_pass_filter:
                sos_low = signal.butter(
                    4, low_pass_freq, btype='lowpass',
                    fs=sample_rate, output='sos'
                )
                filtered = signal.sosfilt(sos_low, filtered)

            logger.debug(f"Applied filters: HP={high_pass_freq}Hz, LP={low_pass_freq}Hz")
            return filtered

        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
            return audio_data

    def compress_dynamic_range(
        self,
        audio_data: np.ndarray,
        threshold: float = 0.5,
        ratio: float = 4.0
    ) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple dynamic range compressor
            compressed = audio_data.copy()

            # Find samples above threshold
            above_threshold = np.abs(compressed) > threshold

            # Apply compression to samples above threshold
            compressed[above_threshold] = (
                np.sign(compressed[above_threshold]) *
                (threshold + (np.abs(compressed[above_threshold]) - threshold) / ratio)
            )

            logger.debug(f"Applied compression: threshold={threshold}, ratio={ratio}")
            return compressed

        except Exception as e:
            logger.error(f"Failed to compress dynamic range: {e}")
            return audio_data

    def enhance_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """Apply full audio enhancement pipeline"""
        try:
            enhanced = audio_data.copy()

            # Apply filters
            enhanced = self.apply_filters(enhanced, sample_rate)

            # Noise reduction
            if self.config.noise_reduction:
                enhanced = self.reduce_noise(
                    enhanced, sample_rate,
                    strength=self.config.noise_reduction_strength
                )

            # Dynamic range compression
            if self.config.dynamic_range_compression:
                enhanced = self.compress_dynamic_range(enhanced)

            # Normalization
            if self.config.normalize:
                enhanced = self.normalize_audio(enhanced)

            logger.debug("Applied audio enhancement pipeline")
            return enhanced

        except Exception as e:
            logger.error(f"Failed to enhance audio: {e}")
            return audio_data

    def split_audio_by_silence(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 0.5,
        min_segment_duration: float = 1.0
    ) -> List[AudioSegment]:
        """Split audio into segments based on silence detection"""
        try:
            segments = []

            # Calculate frame parameters
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop

            # Calculate RMS energy for each frame
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            # Convert frame indices to time
            times = librosa.frames_to_time(
                np.arange(len(rms)),
                sr=sample_rate,
                hop_length=hop_length
            )

            # Detect silence regions
            silent_frames = rms < silence_threshold
            silence_regions = []

            # Find continuous silence regions
            in_silence = False
            silence_start = 0

            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_silence:
                    # Start of silence
                    silence_start = times[i]
                    in_silence = True
                elif not is_silent and in_silence:
                    # End of silence
                    silence_duration = times[i] - silence_start
                    if silence_duration >= min_silence_duration:
                        silence_regions.append((silence_start, times[i]))
                    in_silence = False

            # Split audio based on silence regions
            segment_start = 0.0

            for silence_start, silence_end in silence_regions:
                # Create segment before silence
                if silence_start - segment_start >= min_segment_duration:
                    start_sample = int(segment_start * sample_rate)
                    end_sample = int(silence_start * sample_rate)

                    segment = AudioSegment(
                        audio_data=audio_data[start_sample:end_sample],
                        start_time=segment_start,
                        end_time=silence_start,
                        sample_rate=sample_rate
                    )
                    segments.append(segment)

                # Next segment starts after silence
                segment_start = silence_end

            # Add final segment
            if len(audio_data) / sample_rate - segment_start >= min_segment_duration:
                start_sample = int(segment_start * sample_rate)

                segment = AudioSegment(
                    audio_data=audio_data[start_sample:],
                    start_time=segment_start,
                    end_time=len(audio_data) / sample_rate,
                    sample_rate=sample_rate
                )
                segments.append(segment)

            logger.debug(f"Split audio into {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Failed to split audio by silence: {e}")
            return []

    def get_audio_metadata(
        self,
        file_path: Union[str, Path]
    ) -> AudioMetadata:
        """Extract metadata from audio file"""
        try:
            info = sf.info(str(file_path))

            metadata = AudioMetadata(
                duration=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels,
                format=AudioFormat(info.format.lower()),
                bit_depth=info.subtype_info.name if hasattr(info, 'subtype_info') else None,
                file_size=Path(file_path).stat().st_size
            )

            logger.debug(f"Audio metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to get audio metadata: {e}")
            raise


class VoiceActivityDetector:
    """Voice Activity Detection (VAD) implementation"""

    def __init__(self, config: AudioProcessingConfig = None):
        self.config = config or AudioProcessingConfig()

    def detect_voice_activity(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        return_timestamps: bool = False
    ) -> Union[bool, List[Tuple[float, float]]]:
        """Detect voice activity in audio"""
        try:
            # Calculate frame parameters
            frame_length = int(self.config.vad_window_length * sample_rate)
            hop_length = int(self.config.vad_hop_length * sample_rate)

            # Calculate features for VAD
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            zcr = librosa.feature.zero_crossing_rate(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            # Combine features for voice activity detection
            energy_threshold = np.mean(rms) + self.config.vad_threshold * np.std(rms)
            zcr_threshold = np.mean(zcr) + 0.5 * np.std(zcr)

            voice_activity = (rms > energy_threshold) & (zcr < zcr_threshold)

            if not return_timestamps:
                # Return True if any voice activity detected
                return np.any(voice_activity)

            # Return timestamp ranges of voice activity
            times = librosa.frames_to_time(
                np.arange(len(voice_activity)),
                sr=sample_rate,
                hop_length=hop_length
            )

            voice_segments = []
            in_voice = False
            segment_start = 0.0

            for i, is_voice in enumerate(voice_activity):
                if is_voice and not in_voice:
                    # Start of voice activity
                    segment_start = times[i]
                    in_voice = True
                elif not is_voice and in_voice:
                    # End of voice activity
                    voice_segments.append((segment_start, times[i]))
                    in_voice = False

            # Add final segment if needed
            if in_voice:
                voice_segments.append((segment_start, times[-1]))

            logger.debug(f"Detected {len(voice_segments)} voice activity segments")
            return voice_segments

        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            return False if not return_timestamps else []

    async def detect_voice_activity_streaming(
        self,
        audio_stream: AsyncIterable[bytes],
        sample_rate: int,
        chunk_duration: float = 1.0
    ) -> AsyncIterable[bool]:
        """Detect voice activity in streaming audio"""
        buffer = bytearray()
        chunk_size = int(sample_rate * chunk_duration * 2)  # 16-bit samples

        try:
            async for audio_chunk in audio_stream:
                buffer.extend(audio_chunk)

                # Process complete chunks
                while len(buffer) >= chunk_size:
                    chunk_bytes = bytes(buffer[:chunk_size])

                    # Convert bytes to numpy array
                    audio_data = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32)
                    audio_data = audio_data / 32768.0  # Normalize to [-1, 1]

                    # Detect voice activity
                    has_voice = self.detect_voice_activity(audio_data, sample_rate)
                    yield has_voice

                    # Remove processed chunk
                    buffer = buffer[chunk_size:]

        except Exception as e:
            logger.error(f"Streaming VAD failed: {e}")
            raise


# Utility functions
def bytes_to_audio(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert audio bytes to numpy array"""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def audio_to_bytes(audio_data: np.ndarray) -> bytes:
    """Convert numpy audio array to bytes"""
    # Ensure audio is in valid range [-1, 1]
    audio_data = np.clip(audio_data, -1.0, 1.0)

    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    return audio_int16.tobytes()


def mix_audio(
    audio1: np.ndarray,
    audio2: np.ndarray,
    weight1: float = 0.5,
    weight2: float = 0.5
) -> np.ndarray:
    """Mix two audio arrays with specified weights"""
    # Ensure same length
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    # Mix with weights
    mixed = weight1 * audio1 + weight2 * audio2

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val

    return mixed
"""
Audio streaming capabilities for real-time voice processing
"""

import asyncio
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, AsyncIterable, Callable, Union
from queue import Queue
import threading

import numpy as np
import pyaudio
from pydantic import BaseModel, Field

from .audio_utils import AudioProcessor, VoiceActivityDetector, bytes_to_audio, audio_to_bytes

logger = logging.getLogger(__name__)


class StreamingMode(str, Enum):
    """Audio streaming modes"""
    INPUT_ONLY = "input_only"
    OUTPUT_ONLY = "output_only"
    DUPLEX = "duplex"
    MONITOR = "monitor"


class StreamingQuality(str, Enum):
    """Streaming quality levels"""
    LOW_LATENCY = "low_latency"      # 8kHz, 16ms buffers
    BALANCED = "balanced"            # 16kHz, 32ms buffers
    HIGH_QUALITY = "high_quality"    # 24kHz, 64ms buffers


@dataclass
class AudioStreamConfig:
    """Configuration for audio streaming"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    input_device_index: Optional[int] = None
    output_device_index: Optional[int] = None
    buffer_size: int = 4
    enable_agc: bool = True          # Automatic Gain Control
    enable_noise_suppression: bool = True
    enable_echo_cancellation: bool = True


@dataclass
class StreamingStats:
    """Streaming performance statistics"""
    total_chunks_processed: int = 0
    total_bytes_processed: int = 0
    average_latency: float = 0.0
    buffer_underruns: int = 0
    buffer_overruns: int = 0
    dropped_chunks: int = 0
    processing_time: float = 0.0
    start_time: float = 0.0


class AudioStreamHandler(ABC):
    """Abstract base class for audio stream handlers"""

    @abstractmethod
    async def handle_audio_chunk(self, chunk: bytes, timestamp: float) -> Optional[bytes]:
        """Handle incoming audio chunk and optionally return processed audio"""
        pass

    @abstractmethod
    async def on_stream_start(self) -> None:
        """Called when stream starts"""
        pass

    @abstractmethod
    async def on_stream_stop(self) -> None:
        """Called when stream stops"""
        pass


class AudioStreamer:
    """Real-time audio streaming manager"""

    def __init__(self, config: AudioStreamConfig):
        self.config = config
        self.pyaudio = pyaudio.PyAudio()

        # Streaming state
        self.is_streaming = False
        self.input_stream = None
        self.output_stream = None

        # Audio processing
        self.audio_processor = AudioProcessor()
        self.vad = VoiceActivityDetector()

        # Handlers and callbacks
        self.handlers: List[AudioStreamHandler] = []
        self.voice_activity_callback: Optional[Callable[[bool], None]] = None

        # Buffers and queues
        self.input_buffer = Queue(maxsize=self.config.buffer_size)
        self.output_buffer = Queue(maxsize=self.config.buffer_size)

        # Statistics
        self.stats = StreamingStats()

        # Processing thread
        self.processing_thread = None
        self.stop_processing = threading.Event()

    async def initialize(self) -> None:
        """Initialize audio streaming"""
        try:
            logger.info("Initializing audio streaming...")

            # List available devices
            await self._list_audio_devices()

            # Validate device indices
            if self.config.input_device_index is not None:
                device_info = self.pyaudio.get_device_info_by_index(self.config.input_device_index)
                logger.info(f"Input device: {device_info['name']}")

            if self.config.output_device_index is not None:
                device_info = self.pyaudio.get_device_info_by_index(self.config.output_device_index)
                logger.info(f"Output device: {device_info['name']}")

            logger.info("Audio streaming initialized")

        except Exception as e:
            logger.error(f"Failed to initialize audio streaming: {e}")
            raise

    async def _list_audio_devices(self) -> None:
        """List available audio devices"""
        try:
            device_count = self.pyaudio.get_device_count()
            logger.info(f"Available audio devices ({device_count}):")

            for i in range(device_count):
                info = self.pyaudio.get_device_info_by_index(i)
                logger.info(f"  {i}: {info['name']} (in: {info['maxInputChannels']}, out: {info['maxOutputChannels']})")

        except Exception as e:
            logger.warning(f"Failed to list audio devices: {e}")

    def add_handler(self, handler: AudioStreamHandler) -> None:
        """Add audio stream handler"""
        self.handlers.append(handler)
        logger.debug(f"Added audio stream handler: {handler.__class__.__name__}")

    def remove_handler(self, handler: AudioStreamHandler) -> None:
        """Remove audio stream handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.debug(f"Removed audio stream handler: {handler.__class__.__name__}")

    def set_voice_activity_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback for voice activity detection"""
        self.voice_activity_callback = callback

    async def start_streaming(self, mode: StreamingMode = StreamingMode.DUPLEX) -> None:
        """Start audio streaming"""
        if self.is_streaming:
            logger.warning("Audio streaming already active")
            return

        try:
            logger.info(f"Starting audio streaming in {mode} mode...")

            self.stats = StreamingStats()
            self.stats.start_time = time.time()
            self.stop_processing.clear()

            # Start input stream
            if mode in [StreamingMode.INPUT_ONLY, StreamingMode.DUPLEX]:
                self.input_stream = self.pyaudio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    input_device_index=self.config.input_device_index,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=self._input_callback
                )

            # Start output stream
            if mode in [StreamingMode.OUTPUT_ONLY, StreamingMode.DUPLEX]:
                self.output_stream = self.pyaudio.open(
                    format=self.config.format,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    output=True,
                    output_device_index=self.config.output_device_index,
                    frames_per_buffer=self.config.chunk_size,
                    stream_callback=self._output_callback
                )

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()

            # Notify handlers
            for handler in self.handlers:
                await handler.on_stream_start()

            self.is_streaming = True
            logger.info("Audio streaming started successfully")

        except Exception as e:
            logger.error(f"Failed to start audio streaming: {e}")
            await self.stop_streaming()
            raise

    async def stop_streaming(self) -> None:
        """Stop audio streaming"""
        if not self.is_streaming:
            return

        try:
            logger.info("Stopping audio streaming...")

            self.is_streaming = False
            self.stop_processing.set()

            # Stop streams
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None

            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None

            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)

            # Notify handlers
            for handler in self.handlers:
                await handler.on_stream_stop()

            # Clear buffers
            while not self.input_buffer.empty():
                self.input_buffer.get_nowait()
            while not self.output_buffer.empty():
                self.output_buffer.get_nowait()

            logger.info("Audio streaming stopped")

        except Exception as e:
            logger.error(f"Error stopping audio streaming: {e}")

    def _input_callback(self, in_data, frame_count, time_info, status):
        """PyAudio input stream callback"""
        try:
            if status:
                logger.warning(f"Input stream status: {status}")
                self.stats.buffer_underruns += 1

            # Add to input buffer with timestamp
            timestamp = time.time()

            try:
                self.input_buffer.put_nowait((in_data, timestamp))
            except:
                # Buffer full, drop chunk
                self.stats.dropped_chunks += 1
                logger.debug("Input buffer full, dropping chunk")

            return (None, pyaudio.paContinue)

        except Exception as e:
            logger.error(f"Input callback error: {e}")
            return (None, pyaudio.paAbort)

    def _output_callback(self, in_data, frame_count, time_info, status):
        """PyAudio output stream callback"""
        try:
            if status:
                logger.warning(f"Output stream status: {status}")
                self.stats.buffer_underruns += 1

            # Get data from output buffer
            try:
                audio_data = self.output_buffer.get_nowait()
                return (audio_data, pyaudio.paContinue)
            except:
                # No data available, return silence
                silence = b'\x00' * (frame_count * self.config.channels * 2)  # 16-bit samples
                return (silence, pyaudio.paContinue)

        except Exception as e:
            logger.error(f"Output callback error: {e}")
            return (None, pyaudio.paAbort)

    def _processing_loop(self) -> None:
        """Main processing loop running in separate thread"""
        logger.debug("Audio processing loop started")

        try:
            while not self.stop_processing.is_set():
                try:
                    # Get audio chunk from input buffer
                    chunk_data, timestamp = self.input_buffer.get(timeout=0.1)

                    # Process chunk
                    processed_chunk = asyncio.run(self._process_audio_chunk(chunk_data, timestamp))

                    # Add to output buffer if processed
                    if processed_chunk:
                        try:
                            self.output_buffer.put_nowait(processed_chunk)
                        except:
                            # Output buffer full, drop chunk
                            self.stats.dropped_chunks += 1

                    # Update statistics
                    self.stats.total_chunks_processed += 1
                    self.stats.total_bytes_processed += len(chunk_data)

                except:
                    # No data available, continue
                    continue

        except Exception as e:
            logger.error(f"Processing loop error: {e}")

        logger.debug("Audio processing loop stopped")

    async def _process_audio_chunk(self, chunk_data: bytes, timestamp: float) -> Optional[bytes]:
        """Process individual audio chunk"""
        try:
            processing_start = time.time()

            # Convert to numpy array for processing
            audio_array = bytes_to_audio(chunk_data, self.config.sample_rate)

            # Voice activity detection
            if hasattr(self, 'vad'):
                has_voice = self.vad.detect_voice_activity(audio_array, self.config.sample_rate)

                if self.voice_activity_callback:
                    self.voice_activity_callback(has_voice)

            # Apply audio enhancements
            if self.config.enable_noise_suppression:
                audio_array = self.audio_processor.reduce_noise(
                    audio_array,
                    self.config.sample_rate,
                    strength=0.3
                )

            if self.config.enable_agc:
                audio_array = self.audio_processor.normalize_audio(audio_array, target_level=0.7)

            # Process through handlers
            processed_chunk = chunk_data
            for handler in self.handlers:
                result = await handler.handle_audio_chunk(processed_chunk, timestamp)
                if result:
                    processed_chunk = result

            # Update processing time stats
            processing_time = time.time() - processing_start
            self.stats.processing_time += processing_time

            if self.stats.total_chunks_processed > 0:
                self.stats.average_latency = (
                    self.stats.processing_time / self.stats.total_chunks_processed
                )

            return processed_chunk if processed_chunk != chunk_data else None

        except Exception as e:
            logger.error(f"Audio chunk processing error: {e}")
            return None

    async def play_audio(self, audio_data: Union[bytes, np.ndarray]) -> None:
        """Play audio through output stream"""
        try:
            if isinstance(audio_data, np.ndarray):
                audio_bytes = audio_to_bytes(audio_data)
            else:
                audio_bytes = audio_data

            # Split into chunks and add to output buffer
            chunk_size = self.config.chunk_size * self.config.channels * 2  # 16-bit samples

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]

                # Pad last chunk if necessary
                if len(chunk) < chunk_size:
                    chunk += b'\x00' * (chunk_size - len(chunk))

                # Add to output buffer
                try:
                    self.output_buffer.put_nowait(chunk)
                except:
                    # Buffer full, wait a bit
                    await asyncio.sleep(0.01)
                    try:
                        self.output_buffer.put_nowait(chunk)
                    except:
                        logger.warning("Failed to queue audio chunk for playback")
                        break

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    async def get_streaming_stats(self) -> StreamingStats:
        """Get current streaming statistics"""
        current_stats = StreamingStats(
            total_chunks_processed=self.stats.total_chunks_processed,
            total_bytes_processed=self.stats.total_bytes_processed,
            average_latency=self.stats.average_latency,
            buffer_underruns=self.stats.buffer_underruns,
            buffer_overruns=self.stats.buffer_overruns,
            dropped_chunks=self.stats.dropped_chunks,
            processing_time=self.stats.processing_time,
            start_time=self.stats.start_time
        )

        return current_stats

    async def cleanup(self) -> None:
        """Cleanup streaming resources"""
        try:
            await self.stop_streaming()

            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None

            logger.info("Audio streaming cleanup complete")

        except Exception as e:
            logger.error(f"Audio streaming cleanup error: {e}")


class VoiceStreamingHandler(AudioStreamHandler):
    """Voice-specific streaming handler"""

    def __init__(self, voice_pipeline):
        self.voice_pipeline = voice_pipeline
        self.is_processing = False
        self.current_session_id = None

    async def handle_audio_chunk(self, chunk: bytes, timestamp: float) -> Optional[bytes]:
        """Handle voice audio chunk"""
        try:
            # Avoid concurrent processing
            if self.is_processing:
                return None

            self.is_processing = True

            # Process voice command (non-blocking)
            asyncio.create_task(self._process_voice_chunk(chunk, timestamp))

            return None  # Don't return processed audio for voice commands

        except Exception as e:
            logger.error(f"Voice streaming handler error: {e}")
            return None
        finally:
            self.is_processing = False

    async def _process_voice_chunk(self, chunk: bytes, timestamp: float) -> None:
        """Process voice chunk asynchronously"""
        try:
            if self.voice_pipeline:
                command = await self.voice_pipeline.process_voice_command(
                    chunk,
                    session_id=self.current_session_id
                )
                logger.debug(f"Voice command processed: {command.text[:50]}...")

        except Exception as e:
            logger.debug(f"Voice chunk processing failed: {e}")

    async def on_stream_start(self) -> None:
        """Called when stream starts"""
        logger.info("Voice streaming handler started")

    async def on_stream_stop(self) -> None:
        """Called when stream stops"""
        logger.info("Voice streaming handler stopped")


class AudioStreamingManager:
    """High-level audio streaming management"""

    def __init__(self):
        self.streamers: Dict[str, AudioStreamer] = {}
        self.configs: Dict[str, AudioStreamConfig] = {}

    async def create_streamer(
        self,
        name: str,
        config: AudioStreamConfig
    ) -> AudioStreamer:
        """Create new audio streamer"""
        try:
            streamer = AudioStreamer(config)
            await streamer.initialize()

            self.streamers[name] = streamer
            self.configs[name] = config

            logger.info(f"Created audio streamer: {name}")
            return streamer

        except Exception as e:
            logger.error(f"Failed to create audio streamer {name}: {e}")
            raise

    async def get_streamer(self, name: str) -> Optional[AudioStreamer]:
        """Get existing streamer by name"""
        return self.streamers.get(name)

    async def remove_streamer(self, name: str) -> None:
        """Remove and cleanup streamer"""
        if name in self.streamers:
            await self.streamers[name].cleanup()
            del self.streamers[name]

            if name in self.configs:
                del self.configs[name]

            logger.info(f"Removed audio streamer: {name}")

    async def list_streamers(self) -> List[str]:
        """List all streamer names"""
        return list(self.streamers.keys())

    async def cleanup_all(self) -> None:
        """Cleanup all streamers"""
        logger.info("Cleaning up all audio streamers...")

        for name in list(self.streamers.keys()):
            await self.remove_streamer(name)

        logger.info("All audio streamers cleaned up")


# Utility functions for streaming
def get_optimal_chunk_size(sample_rate: int, latency_ms: int = 20) -> int:
    """Calculate optimal chunk size for given latency"""
    return int(sample_rate * latency_ms / 1000)


def get_recommended_config(quality: StreamingQuality) -> AudioStreamConfig:
    """Get recommended config for streaming quality level"""
    configs = {
        StreamingQuality.LOW_LATENCY: AudioStreamConfig(
            sample_rate=8000,
            chunk_size=128,
            buffer_size=2
        ),
        StreamingQuality.BALANCED: AudioStreamConfig(
            sample_rate=16000,
            chunk_size=512,
            buffer_size=4
        ),
        StreamingQuality.HIGH_QUALITY: AudioStreamConfig(
            sample_rate=24000,
            chunk_size=1024,
            buffer_size=8
        )
    }

    return configs.get(quality, configs[StreamingQuality.BALANCED])
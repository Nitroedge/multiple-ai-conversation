"""
Configuration settings for the Multi-Agent Conversation Engine
"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings"""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API Configuration
    api_title: str = "Multi-Agent Conversation Engine"
    api_version: str = "2.0.0"
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5678"],
        env="CORS_ORIGINS"
    )

    # Database URLs
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    mongodb_url: str = Field(
        default="mongodb://admin:multi_agent_pass_2024@localhost:27017/multi_agent_conversations?authSource=admin",
        env="MONGODB_URL"
    )
    postgres_url: str = Field(
        default="postgresql://agent_user:agent_analytics_2024@localhost:5432/multi_agent_analytics",
        env="POSTGRES_URL"
    )

    # External API Keys
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    elevenlabs_api_key: str = Field(env="ELEVENLABS_API_KEY")

    # n8n Configuration
    n8n_webhook_url: str = Field(default="http://localhost:5678/webhook", env="N8N_WEBHOOK_URL")
    n8n_basic_auth_user: str = Field(default="admin", env="N8N_BASIC_AUTH_USER")
    n8n_basic_auth_password: str = Field(default="multi_agent_n8n_2024", env="N8N_BASIC_AUTH_PASSWORD")

    # Memory System Configuration
    memory_consolidation_threshold: float = Field(default=0.7, env="MEMORY_CONSOLIDATION_THRESHOLD")
    working_memory_ttl: int = Field(default=3600, env="WORKING_MEMORY_TTL")  # 1 hour
    context_window_size: int = Field(default=4000, env="CONTEXT_WINDOW_SIZE")

    # Voice Processing Configuration
    tts_cache_duration: int = Field(default=86400, env="TTS_CACHE_DURATION")  # 24 hours
    stt_model: str = Field(default="whisper-1", env="STT_MODEL")
    default_voice_provider: str = Field(default="elevenlabs", env="DEFAULT_VOICE_PROVIDER")

    # Character Configuration
    character_adaptation_rate: float = Field(default=0.1, env="CHARACTER_ADAPTATION_RATE")
    personality_stability_threshold: float = Field(default=0.05, env="PERSONALITY_STABILITY_THRESHOLD")

    # Safety and Moderation
    content_moderation_enabled: bool = Field(default=True, env="CONTENT_MODERATION_ENABLED")
    safety_confidence_threshold: float = Field(default=0.7, env="SAFETY_CONFIDENCE_THRESHOLD")
    human_review_queue_size: int = Field(default=100, env="HUMAN_REVIEW_QUEUE_SIZE")

    # Performance Settings
    max_concurrent_conversations: int = Field(default=50, env="MAX_CONCURRENT_CONVERSATIONS")
    redis_connection_pool_size: int = Field(default=20, env="REDIS_CONNECTION_POOL_SIZE")
    mongodb_connection_pool_size: int = Field(default=10, env="MONGODB_CONNECTION_POOL_SIZE")

    # Monitoring and Analytics
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    metrics_export_interval: int = Field(default=300, env="METRICS_EXPORT_INTERVAL")  # 5 minutes
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Development Configuration
    enable_debug_endpoints: bool = Field(default=False, env="ENABLE_DEBUG_ENDPOINTS")
    mock_ai_responses: bool = Field(default=False, env="MOCK_AI_RESPONSES")
    log_conversation_content: bool = Field(default=False, env="LOG_CONVERSATION_CONTENT")

    # Security
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=30)

    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


# Singleton instance
_settings = None


def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Voice Model Configurations
VOICE_MODELS = {
    "openai": {
        "tts_models": ["tts-1", "tts-1-hd"],
        "stt_models": ["whisper-1"],
        "voices": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    },
    "elevenlabs": {
        "models": ["eleven_monolingual_v1", "eleven_turbo_v2"],
        "voices": {
            "Dougsworth": "voice_id_placeholder",
            "Tony Emperor of New York": "voice_id_placeholder",
            "Victoria": "voice_id_placeholder"
        }
    }
}

# Character Default Configurations
CHARACTER_DEFAULTS = {
    "OSWALD": {
        "personality": {
            "extraversion": 0.95,
            "agreeableness": 0.85,
            "conscientiousness": 0.75,
            "neuroticism": 0.15,
            "openness": 0.90
        },
        "voice": {
            "provider": "elevenlabs",
            "voice_id": "Dougsworth",
            "settings": {"stability": 0.75, "similarity_boost": 0.85}
        },
        "response_style": {
            "energy": "high",
            "enthusiasm": "extreme",
            "curiosity": "insatiable",
            "family_friendly": True
        }
    },
    "TONY_KING": {
        "personality": {
            "extraversion": 0.80,
            "agreeableness": 0.30,
            "conscientiousness": 0.40,
            "neuroticism": 0.60,
            "openness": 0.70
        },
        "voice": {
            "provider": "elevenlabs",
            "voice_id": "Tony Emperor of New York",
            "settings": {"stability": 0.6, "similarity_boost": 0.9}
        },
        "response_style": {
            "energy": "intense",
            "sarcasm": "high",
            "controversy": "moderate",
            "accent": "new_york"
        }
    },
    "VICTORIA": {
        "personality": {
            "extraversion": 0.65,
            "agreeableness": 0.45,
            "conscientiousness": 0.90,
            "neuroticism": 0.55,
            "openness": 0.95
        },
        "voice": {
            "provider": "elevenlabs",
            "voice_id": "Victoria",
            "settings": {"stability": 0.8, "similarity_boost": 0.8}
        },
        "response_style": {
            "energy": "analytical",
            "intensity": "high",
            "criticism": "constructive",
            "accent": "british"
        }
    }
}

# n8n Workflow Templates
N8N_WORKFLOW_TEMPLATES = {
    "voice_processing": {
        "name": "Voice Command Processing",
        "description": "Process voice input through STT, agent routing, and TTS response",
        "webhook_path": "/webhook/voice-command"
    },
    "agent_coordination": {
        "name": "Multi-Agent Coordination",
        "description": "Coordinate responses between multiple AI agents",
        "webhook_path": "/webhook/agent-coordination"
    },
    "home_automation": {
        "name": "Home Assistant Integration",
        "description": "Execute home automation commands based on conversation",
        "webhook_path": "/webhook/home-automation"
    },
    "memory_consolidation": {
        "name": "Memory Consolidation",
        "description": "Periodic consolidation of conversation memories",
        "schedule": "0 */6 * * *"  # Every 6 hours
    }
}

# API Rate Limits
RATE_LIMITS = {
    "conversation_messages": "60/minute",
    "memory_searches": "30/minute",
    "agent_responses": "20/minute",
    "webhook_calls": "100/minute"
}

# Error Messages
ERROR_MESSAGES = {
    "memory_store_failed": "Failed to store memory item",
    "conversation_not_found": "Conversation session not found",
    "agent_not_available": "Requested agent is not available",
    "invalid_voice_input": "Invalid voice input format",
    "rate_limit_exceeded": "Rate limit exceeded for this operation",
    "safety_violation": "Content violates safety guidelines",
    "service_unavailable": "Service temporarily unavailable"
}
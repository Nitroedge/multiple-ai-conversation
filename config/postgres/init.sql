-- PostgreSQL Initialization Script for Multi-Agent Conversation Engine

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create database for n8n workflows if not exists
CREATE DATABASE n8n_workflows;

-- Create schema for analytics data
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS moderation;
CREATE SCHEMA IF NOT EXISTS performance;

-- User Management and Authentication
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),

    -- User preferences and settings
    preferences JSONB DEFAULT '{}',
    privacy_settings JSONB DEFAULT '{
        "data_retention": "12_months",
        "analytics_opt_in": true,
        "conversation_logging": true
    }',

    -- Subscription and permissions
    subscription_level VARCHAR(50) DEFAULT 'free' CHECK (subscription_level IN ('free', 'premium', 'enterprise')),
    permissions JSONB DEFAULT '[]',

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Character Definitions and Versions
CREATE TABLE agent_characters (
    character_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    archetype VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Personality Configuration
    base_personality JSONB NOT NULL DEFAULT '{
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "conscientiousness": 0.5,
        "neuroticism": 0.5,
        "openness": 0.5
    }',
    voice_configuration JSONB NOT NULL DEFAULT '{}',
    prompt_templates JSONB NOT NULL DEFAULT '{}',
    behavioral_constraints JSONB DEFAULT '{}',

    -- Character Development
    adaptation_settings JSONB DEFAULT '{
        "adaptation_rate": 0.1,
        "stability_threshold": 0.05
    }',
    learning_parameters JSONB DEFAULT '{}',

    -- Metadata
    creator_id UUID REFERENCES users(user_id),
    is_active BOOLEAN DEFAULT true,
    tags TEXT[],
    description TEXT,

    UNIQUE(name, version)
);

-- Conversation Sessions Index
CREATE TABLE conversation_sessions (
    session_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ended_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'archived')),

    -- MongoDB Reference
    mongodb_document_id VARCHAR(255) UNIQUE NOT NULL,

    -- Quick Access Metrics
    total_messages INTEGER DEFAULT 0,
    duration_minutes INTEGER,
    engagement_score DECIMAL(3,2) CHECK (engagement_score >= 0 AND engagement_score <= 1),
    quality_score DECIMAL(3,2) CHECK (quality_score >= 0 AND quality_score <= 1),

    -- Categorization
    topic_category VARCHAR(100),
    conversation_type VARCHAR(50) DEFAULT 'casual',
    privacy_level VARCHAR(20) DEFAULT 'standard' CHECK (privacy_level IN ('public', 'standard', 'private', 'confidential')),

    -- Performance metrics
    avg_response_time_ms INTEGER,
    total_tokens_used INTEGER,
    total_api_cost DECIMAL(10,4),

    -- Indexing support
    created_date DATE GENERATED ALWAYS AS (started_at::date) STORED,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Agent Performance Tracking
CREATE TABLE performance.agent_performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    character_id UUID REFERENCES agent_characters(character_id),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Performance Metrics
    response_time_ms INTEGER NOT NULL CHECK (response_time_ms >= 0),
    character_consistency_score DECIMAL(3,2) CHECK (character_consistency_score >= 0 AND character_consistency_score <= 1),
    engagement_contribution DECIMAL(3,2) CHECK (engagement_contribution >= 0 AND engagement_contribution <= 1),
    safety_score DECIMAL(3,2) CHECK (safety_score >= 0 AND safety_score <= 1),

    -- Context
    turn_number INTEGER NOT NULL CHECK (turn_number > 0),
    message_length INTEGER CHECK (message_length >= 0),
    complexity_score DECIMAL(3,2) CHECK (complexity_score >= 0 AND complexity_score <= 1),

    -- Token usage
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,

    -- Aggregation support
    recorded_date DATE GENERATED ALWAYS AS (recorded_at::date) STORED,
    recorded_hour INTEGER GENERATED ALWAYS AS (EXTRACT(HOUR FROM recorded_at)) STORED
);

-- Moderation and Safety Logs
CREATE TABLE moderation.moderation_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Content Reference (hashed for privacy)
    content_hash VARCHAR(64) NOT NULL,
    message_id VARCHAR(255),
    content_length INTEGER,

    -- Moderation Results
    risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('safe', 'low_risk', 'medium_risk', 'high_risk', 'blocked')),
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    action_taken VARCHAR(50) NOT NULL,
    reasons TEXT[] NOT NULL,

    -- Detection details
    detection_method VARCHAR(50), -- 'rule_based', 'ml_classifier', 'human_review'
    model_version VARCHAR(20),
    processing_time_ms INTEGER,

    -- Review Information
    reviewed_by UUID REFERENCES users(user_id),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    review_decision VARCHAR(20) CHECK (review_decision IN ('approve', 'reject', 'modify', 'escalate')),
    review_notes TEXT,

    -- Compliance
    compliance_flags TEXT[],
    retention_date DATE,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Analytics Views and Materialized Views
CREATE VIEW analytics.conversation_analytics AS
SELECT
    cs.session_id,
    cs.user_id,
    cs.started_at,
    cs.ended_at,
    cs.duration_minutes,
    cs.total_messages,
    cs.engagement_score,
    cs.quality_score,
    cs.topic_category,
    cs.conversation_type,

    -- Agent Performance Aggregations
    AVG(apm.character_consistency_score) as avg_character_consistency,
    AVG(apm.response_time_ms) as avg_response_time_ms,
    SUM(apm.total_tokens) as total_tokens_used,

    -- Safety Metrics
    COUNT(me.event_id) as moderation_events_count,
    COUNT(CASE WHEN me.risk_level IN ('high_risk', 'blocked') THEN 1 END) as high_risk_events,
    AVG(me.confidence_score) as avg_safety_confidence,

    -- Engagement Patterns
    EXTRACT(HOUR FROM cs.started_at) as start_hour,
    EXTRACT(DOW FROM cs.started_at) as day_of_week,
    EXTRACT(MONTH FROM cs.started_at) as month,

    -- User behavior
    u.subscription_level,
    u.status as user_status

FROM conversation_sessions cs
LEFT JOIN performance.agent_performance_metrics apm ON cs.session_id = apm.session_id
LEFT JOIN moderation.moderation_events me ON cs.session_id = me.session_id
LEFT JOIN users u ON cs.user_id = u.user_id
GROUP BY cs.session_id, cs.user_id, cs.started_at, cs.ended_at, cs.duration_minutes,
         cs.total_messages, cs.engagement_score, cs.quality_score, cs.topic_category,
         cs.conversation_type, u.subscription_level, u.status;

-- Materialized view for performance dashboard
CREATE MATERIALIZED VIEW analytics.daily_performance_summary AS
SELECT
    DATE(recorded_at) as date,
    COUNT(*) as total_interactions,
    AVG(response_time_ms) as avg_response_time,
    AVG(character_consistency_score) as avg_consistency,
    AVG(engagement_contribution) as avg_engagement,
    AVG(safety_score) as avg_safety,
    SUM(total_tokens) as daily_token_usage,
    COUNT(DISTINCT session_id) as unique_sessions
FROM performance.agent_performance_metrics
GROUP BY DATE(recorded_at)
ORDER BY date DESC;

-- Data Retention and Archival
CREATE TABLE analytics.archived_conversations (
    archive_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_session_id UUID NOT NULL,
    archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    archive_location TEXT NOT NULL,
    compression_ratio DECIMAL(4,2),
    metadata JSONB DEFAULT '{}',

    -- Retention policy
    retention_until DATE,
    deletion_eligible BOOLEAN DEFAULT FALSE
);

-- Performance Optimization Indexes
CREATE INDEX idx_conversations_user_date ON conversation_sessions(user_id, created_date DESC);
CREATE INDEX idx_conversations_status ON conversation_sessions(status) WHERE status = 'active';
CREATE INDEX idx_conversations_engagement ON conversation_sessions(engagement_score DESC) WHERE engagement_score IS NOT NULL;
CREATE INDEX idx_conversations_topic ON conversation_sessions(topic_category);
CREATE INDEX idx_conversations_type ON conversation_sessions(conversation_type);

-- Agent performance indexes
CREATE INDEX idx_agent_performance_session ON performance.agent_performance_metrics(session_id, recorded_at DESC);
CREATE INDEX idx_agent_performance_character ON performance.agent_performance_metrics(character_id, recorded_date);
CREATE INDEX idx_agent_performance_date_hour ON performance.agent_performance_metrics(recorded_date, recorded_hour);

-- Moderation indexes
CREATE INDEX idx_moderation_session ON moderation.moderation_events(session_id, occurred_at DESC);
CREATE INDEX idx_moderation_risk_level ON moderation.moderation_events(risk_level, occurred_at DESC);
CREATE INDEX idx_moderation_review ON moderation.moderation_events(reviewed_by, review_decision) WHERE reviewed_by IS NOT NULL;

-- Character development indexes
CREATE INDEX idx_characters_active ON agent_characters(is_active, name) WHERE is_active = true;
CREATE INDEX idx_characters_version ON agent_characters(name, version DESC);

-- User management indexes
CREATE INDEX idx_users_status ON users(status, last_active_at DESC) WHERE status = 'active';
CREATE INDEX idx_users_subscription ON users(subscription_level);

-- Full-text search indexes
CREATE INDEX idx_conversations_search ON conversation_sessions USING gin(to_tsvector('english', COALESCE(metadata->>'title', '')));

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW analytics.daily_performance_summary;
    -- Add other materialized views here as needed
END;
$$ LANGUAGE plpgsql;

-- Create user for application access
CREATE USER conversation_analytics WITH PASSWORD 'analytics_pass_2024';
GRANT CONNECT ON DATABASE multi_agent_analytics TO conversation_analytics;
GRANT USAGE ON SCHEMA public, analytics, moderation, performance TO conversation_analytics;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public, analytics, moderation, performance TO conversation_analytics;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public, analytics, moderation, performance TO conversation_analytics;

-- Grant permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO conversation_analytics;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT, INSERT, UPDATE ON TABLES TO conversation_analytics;
ALTER DEFAULT PRIVILEGES IN SCHEMA moderation GRANT SELECT, INSERT, UPDATE ON TABLES TO conversation_analytics;
ALTER DEFAULT PRIVILEGES IN SCHEMA performance GRANT SELECT, INSERT, UPDATE ON TABLES TO conversation_analytics;

-- Initial data
INSERT INTO agent_characters (name, archetype, version, base_personality, voice_configuration, prompt_templates) VALUES
('OSWALD', 'Enthusiastic Adventurer', '2.0.0',
 '{"extraversion": 0.95, "agreeableness": 0.85, "conscientiousness": 0.75, "neuroticism": 0.15, "openness": 0.90}',
 '{"provider": "elevenlabs", "voice_id": "Dougsworth", "settings": {"stability": 0.75, "similarity_boost": 0.85}}',
 '{"system_template": "enthusiastic_adventurer", "response_modifiers": ["high_energy", "curious", "family_friendly"]}'),

('TONY KING OF NEW YORK', 'Cynical Jester', '2.0.0',
 '{"extraversion": 0.80, "agreeableness": 0.30, "conscientiousness": 0.40, "neuroticism": 0.60, "openness": 0.70}',
 '{"provider": "elevenlabs", "voice_id": "Tony Emperor of New York", "settings": {"stability": 0.6, "similarity_boost": 0.9}}',
 '{"system_template": "cynical_jester", "response_modifiers": ["sarcastic", "controversial", "new_york_accent"]}'),

('VICTORIA', 'Intense Philosopher', '2.0.0',
 '{"extraversion": 0.65, "agreeableness": 0.45, "conscientiousness": 0.90, "neuroticism": 0.55, "openness": 0.95}',
 '{"provider": "elevenlabs", "voice_id": "Victoria", "settings": {"stability": 0.8, "similarity_boost": 0.8}}',
 '{"system_template": "intense_philosopher", "response_modifiers": ["analytical", "critical", "british_accent"]}');

-- Create trigger to update updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();
CREATE TRIGGER update_characters_updated_at BEFORE UPDATE ON agent_characters FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column();

COMMIT;
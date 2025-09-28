# Multi-Agent Conversation Engine - User Guide

**Version**: 1.0
**Last Updated**: September 27, 2025
**Completion Status**: 95% - Enterprise Production Ready

---

## 📚 Table of Contents

1. [System Overview](#-system-overview)
2. [Prerequisites](#-prerequisites)
3. [Installation Guide](#-installation-guide)
4. [First-Time Setup](#-first-time-setup)
5. [Configuration](#-configuration)
6. [Home Assistant Integration](#-home-assistant-integration)
7. [Voice Processing Setup](#-voice-processing-setup)
8. [Using the System](#-using-the-system)
9. [Admin Dashboard](#-admin-dashboard)
10. [Multi-Agent Coordination](#-multi-agent-coordination)
11. [Troubleshooting](#-troubleshooting)
12. [Advanced Configuration](#-advanced-configuration)
13. [API Reference](#-api-reference)
14. [Security Considerations](#-security-considerations)
15. [Performance Optimization](#-performance-optimization)

---

## 🎯 System Overview

The Multi-Agent Conversation Engine is an enterprise-grade platform that enables sophisticated AI-powered conversations with multiple AI agents, voice processing, home automation integration, and real-time collaboration.

### Key Features:
- **Multi-AI Model Support**: Claude, GPT-4, Gemini, and Local models
- **Real-time Multi-Agent Coordination**: Up to 100+ concurrent AI agents
- **Voice Processing**: Complete speech-to-text and text-to-speech pipeline
- **Home Assistant Integration**: Full home automation control
- **Enterprise Security**: Multi-tenancy, RBAC, API keys, encryption
- **Scalable Architecture**: Load balancing and auto-scaling
- **Professional Interface**: Modern React dashboard with real-time updates

---

## 📋 Prerequisites

### System Requirements:
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space minimum, 50GB recommended
- **CPU**: 4+ cores recommended
- **Network**: Stable internet connection for AI model APIs

### Required Software:
- **Docker Desktop** (latest version)
- **Git** (for cloning the repository)
- **Text Editor** (VS Code recommended)

### Optional but Recommended:
- **Home Assistant** (if using home automation features)
- **ESP32 devices** (for advanced home automation)

---

## 🚀 Installation Guide

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Nitroedge/multiple-ai-conversation.git

# Navigate to the project directory
cd multiple-ai-conversation
```

### Step 2: Install Docker Desktop

1. **Download Docker Desktop**:
   - Windows/macOS: https://www.docker.com/products/docker-desktop
   - Linux: Follow Docker's official installation guide

2. **Start Docker Desktop** and ensure it's running

3. **Verify Docker installation**:
   ```bash
   docker --version
   docker-compose --version
   ```

### Step 3: Environment Configuration

1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your preferred text editor:
   ```bash
   # For VS Code users
   code .env

   # Or use any text editor
   notepad .env  # Windows
   nano .env     # Linux/macOS
   ```

### Step 4: Configure API Keys

**Edit your `.env` file** and add your API keys:

```env
# AI Model API Keys (at least one required)
ANTHROPIC_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here

# Voice Processing (optional)
ELEVENLABS_API_KEY=your_elevenlabs_key_here

# Home Assistant (optional)
HOME_ASSISTANT_URL=http://your-ha-instance:8123
HOME_ASSISTANT_TOKEN=your_ha_long_lived_token

# Database Configuration (defaults are fine for development)
REDIS_URL=redis://redis:6379
MONGODB_URL=mongodb://mongodb:27017/multi_agent_db
POSTGRES_URL=postgresql://user:password@postgres:5432/n8n_db

# Security (generate strong passwords for production)
JWT_SECRET_KEY=your_super_secret_jwt_key_here
ENCRYPTION_KEY=your_32_byte_encryption_key_here
```

### Step 5: Start the System

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# View logs (optional)
docker-compose -f docker-compose.dev.yml logs -f
```

### Step 6: Verify Installation

**Check that all services are running**:
```bash
docker-compose -f docker-compose.dev.yml ps
```

You should see 8 services running:
- `redis` - Memory caching
- `mongodb` - Long-term storage
- `postgres` - n8n database
- `n8n` - Workflow orchestration
- `api` - Main FastAPI backend
- `frontend` - React user interface
- `redis-commander` - Redis management (optional)
- `mongo-express` - MongoDB management (optional)

---

## 🔧 First-Time Setup

### Step 1: Access the System

Open your web browser and navigate to:
- **Main Dashboard**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs
- **n8n Workflow Manager**: http://localhost:5678
- **Redis Commander**: http://localhost:8081 (optional)
- **MongoDB Express**: http://localhost:8082 (optional)

### Step 2: Create Your First Admin User

1. **Access the API**: http://localhost:8000/docs
2. **Navigate to the `/auth/register` endpoint**
3. **Create an admin account**:
   ```json
   {
     "username": "admin",
     "email": "admin@yourcompany.com",
     "password": "your_secure_password",
     "full_name": "System Administrator",
     "roles": ["super_admin"]
   }
   ```

### Step 3: Get Your Authentication Token

1. **Use the `/auth/login` endpoint**
2. **Copy the JWT token** from the response
3. **Click "Authorize"** at the top of the API docs
4. **Enter**: `Bearer your_jwt_token_here`

### Step 4: Verify System Health

1. **Navigate to**: http://localhost:8000/health
2. **Check that all services** return "healthy" status
3. **Access the dashboard**: http://localhost:8080

---

## ⚙️ Configuration

### Basic Configuration

The system uses a hierarchical configuration system:

1. **Environment Variables** (`.env` file) - Primary configuration
2. **Config Files** (`config/` directory) - Service-specific settings
3. **Runtime Settings** - Dashboard and API configuration

### Key Configuration Areas:

#### AI Model Configuration
```env
# Primary AI model (used as default)
PRIMARY_AI_MODEL=claude

# Model routing strategy
MODEL_ROUTING_STRATEGY=performance  # options: performance, cost, balanced

# Model-specific settings
CLAUDE_MODEL=claude-3-sonnet-20240229
GPT4_MODEL=gpt-4-1106-preview
GEMINI_MODEL=gemini-pro
```

#### Memory System Configuration
```env
# Memory retention periods
WORKING_MEMORY_TTL=3600  # 1 hour in seconds
LONG_TERM_MEMORY_DAYS=90  # 90 days
VECTOR_MEMORY_DIMENSIONS=1536

# Memory consolidation
MEMORY_CONSOLIDATION_INTERVAL=3600  # 1 hour
```

#### Multi-Agent Configuration
```env
# Agent limits
MAX_CONCURRENT_AGENTS=10
MAX_AGENTS_PER_WORKFLOW=5
AGENT_TIMEOUT_SECONDS=300

# Coordination settings
CONFLICT_RESOLUTION_STRATEGY=consensus
COLLABORATION_MODE=dynamic
```

### Advanced Configuration Files

#### Redis Configuration (`config/redis.conf`)
```conf
# Memory management
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
```

#### MongoDB Configuration
The system automatically configures MongoDB with optimal settings for conversation storage.

#### n8n Workflows
Pre-configured workflows are included in `n8n/workflows/`. These handle:
- Voice command processing
- Conversation state management
- Session persistence
- Error handling and fallback
- System monitoring and logging

---

## 🏠 Home Assistant Integration

### Prerequisites

1. **Home Assistant instance** running and accessible
2. **Long-lived access token** from Home Assistant
3. **Network connectivity** between the containers and Home Assistant

### Setup Instructions

#### Step 1: Generate Home Assistant Token

1. **Open Home Assistant** in your browser
2. **Go to Profile** (click your username in the bottom left)
3. **Scroll down to "Long-Lived Access Tokens"**
4. **Click "Create Token"**
5. **Copy the token** (you won't see it again)

#### Step 2: Configure Home Assistant Connection

**Update your `.env` file**:
```env
# Home Assistant Configuration
HOME_ASSISTANT_URL=http://your-homeassistant-ip:8123
HOME_ASSISTANT_TOKEN=your_long_lived_token_here

# Optional: WebSocket configuration
HA_WEBSOCKET_ENABLED=true
HA_WEBSOCKET_RECONNECT_INTERVAL=30
```

#### Step 3: Test Home Assistant Connection

1. **Access the API docs**: http://localhost:8000/docs
2. **Navigate to `/home-assistant/status`**
3. **Execute the endpoint**
4. **Verify connection status** is "connected"

#### Step 4: Configure Home Assistant Entities

The system automatically discovers your Home Assistant entities. You can configure specific entities for voice control:

```json
{
  "voice_controlled_entities": [
    "light.living_room",
    "switch.coffee_maker",
    "climate.main_thermostat",
    "media_player.living_room_tv"
  ],
  "entity_aliases": {
    "living room lights": "light.living_room",
    "coffee machine": "switch.coffee_maker",
    "temperature": "climate.main_thermostat"
  }
}
```

#### Step 5: Voice Command Examples

Once configured, you can use voice commands like:
- "Turn on the living room lights"
- "Start the coffee machine"
- "Set temperature to 72 degrees"
- "Turn off all lights"

### ESP32 Integration (Advanced)

If you have ESP32 devices for additional sensors/controls:

#### ESP32 Configuration
```cpp
// Example ESP32 code for integration
#include <WiFi.h>
#include <HTTPClient.h>

const char* apiEndpoint = "http://your-api-ip:8000/esp32/sensor-data";
const char* apiKey = "your_esp32_api_key";

void sendSensorData() {
  HTTPClient http;
  http.begin(apiEndpoint);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", apiKey);

  String payload = "{\"sensor\":\"temperature\",\"value\":25.6,\"location\":\"living_room\"}";
  int httpCode = http.POST(payload);

  http.end();
}
```

---

## 🎤 Voice Processing Setup

### Prerequisites

1. **Microphone** for speech input
2. **Speakers** or headphones for speech output
3. **ElevenLabs API key** (recommended for high-quality TTS)

### Configuration

#### Step 1: Audio Device Setup

**Update your `.env` file**:
```env
# Voice Processing Configuration
VOICE_PROCESSING_ENABLED=true

# Speech-to-Text (Whisper)
STT_MODEL=whisper-1
STT_LANGUAGE=en  # or your preferred language

# Text-to-Speech (ElevenLabs recommended)
TTS_PROVIDER=elevenlabs  # options: elevenlabs, openai, local
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_preferred_voice_id

# Audio settings
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_SIZE=1024
```

#### Step 2: Voice Personality Configuration

Configure how different AI agents sound:
```json
{
  "voice_personalities": {
    "helpful_assistant": {
      "voice_id": "EXAVITQu4vr4xnSDxMaL",  # ElevenLabs voice ID
      "speaking_rate": 1.0,
      "pitch": 0.0,
      "emotional_range": "friendly"
    },
    "technical_expert": {
      "voice_id": "ErXwobaYiN019PkySvjV",
      "speaking_rate": 0.9,
      "pitch": -0.1,
      "emotional_range": "professional"
    }
  }
}
```

#### Step 3: Voice Command Setup

The system includes pre-configured voice commands:
- **"Hey Assistant"** - Wake word activation
- **"List agents"** - Show available AI agents
- **"Switch to [agent name]"** - Change active agent
- **"Start workflow [workflow name]"** - Execute a workflow
- **"What's the status?"** - System status check

#### Step 4: Test Voice Processing

1. **Access the dashboard**: http://localhost:8080
2. **Navigate to "Voice Processing"**
3. **Click "Test Microphone"**
4. **Speak a test phrase**
5. **Verify speech-to-text transcription**
6. **Test text-to-speech output**

---

## 💻 Using the System

### Dashboard Overview

The main dashboard (http://localhost:8080) provides access to all system features:

#### Main Sections:
1. **Dashboard**: System overview and health metrics
2. **Agent Coordination**: Manage and monitor AI agents
3. **Conflict Resolution**: Handle agent conflicts and coordination
4. **Collaboration Workflows**: Create and manage multi-agent workflows
5. **Communication**: Inter-agent messaging and notifications
6. **Analytics**: Performance metrics and usage analytics
7. **Settings**: System configuration and preferences

### Basic Usage Workflow

#### Step 1: Start a Conversation

1. **Access the dashboard**
2. **Click "New Conversation"**
3. **Select your preferred AI model(s)**
4. **Choose conversation mode**:
   - **Single Agent**: One AI assistant
   - **Multi-Agent**: Multiple AI agents collaborating
   - **Voice Enabled**: Voice input/output
   - **Home Assistant**: Home automation control

#### Step 2: Configure Agents

For multi-agent conversations:
1. **Click "Add Agent"**
2. **Select agent role**:
   - **Coordinator**: Manages the conversation flow
   - **Specialist**: Domain-specific expertise
   - **Analyst**: Data analysis and insights
   - **Creative**: Creative thinking and ideas
   - **Critic**: Critical evaluation and feedback
3. **Set agent personality** using Big Five traits
4. **Configure agent memory** preferences

#### Step 3: Conversation Management

- **Message History**: View full conversation context
- **Agent Status**: Monitor agent activity and health
- **Memory Usage**: Track working and long-term memory
- **Performance Metrics**: Response times and success rates

### Workflow Management

#### Creating Custom Workflows

1. **Navigate to "Collaboration Workflows"**
2. **Click "Create New Workflow"**
3. **Choose from templates**:
   - **Brainstorming Session**: Creative idea generation
   - **Problem Solving**: Structured problem resolution
   - **Research Project**: Multi-source research and analysis
   - **Decision Making**: Consensus-based decision processes
4. **Customize workflow steps**
5. **Assign agent roles**
6. **Set success criteria**

#### Workflow Templates Available:

- **Collaborative Brainstorming**: Multi-agent idea generation
- **Peer Review Process**: Structured feedback and validation
- **Consensus Building**: Multi-viewpoint consensus development
- **Pros and Cons Analysis**: Systematic decision analysis
- **Risk Assessment**: Comprehensive risk evaluation
- **Literature Review**: Academic research compilation
- **Competitive Analysis**: Market and competitor research
- **Root Cause Analysis**: Problem investigation
- **Solution Design**: Systematic solution development
- **Implementation Planning**: Project planning and coordination

---

## 👨‍💼 Admin Dashboard

### Accessing Admin Features

1. **Log in with admin credentials**
2. **Navigate to the Admin Dashboard**
3. **Verify admin permissions** (super_admin role required)

### Key Admin Functions

#### System Monitoring

- **Service Health**: Real-time status of all 8 services
- **Performance Metrics**: Response times, error rates, throughput
- **Resource Usage**: CPU, memory, storage utilization
- **Active Sessions**: Current user and agent activity

#### User Management

- **Create Users**: Add new system users
- **Manage Roles**: Assign roles and permissions
- **Security Monitoring**: Failed logins, security events
- **API Key Management**: Generate and manage API keys

#### Tenant Management (Multi-tenancy)

- **Tenant Creation**: Set up new organizational tenants
- **Quota Management**: Monitor and adjust resource quotas
- **Usage Analytics**: Per-tenant usage metrics
- **Billing Integration**: Usage-based billing data

#### System Configuration

- **AI Model Settings**: Configure model priorities and routing
- **Security Policies**: Set password policies, session timeouts
- **Rate Limiting**: Configure API rate limits and quotas
- **Scaling Settings**: Auto-scaling thresholds and policies

### Alerts and Notifications

The system generates automated alerts for:
- **High resource usage** (>90% CPU/memory)
- **Service failures** or degraded performance
- **Security events** (multiple failed logins, unusual activity)
- **Quota violations** (tenants exceeding limits)
- **API rate limiting** (excessive request rates)

---

## 🤝 Multi-Agent Coordination

### Understanding Agent Roles

#### Core Agent Types:
- **Coordinator**: Manages conversation flow and task distribution
- **Specialist**: Provides domain-specific expertise
- **Analyst**: Performs data analysis and provides insights
- **Creative**: Generates creative ideas and solutions
- **Critic**: Provides critical evaluation and feedback
- **Facilitator**: Guides group discussions and consensus
- **Researcher**: Gathers and synthesizes information
- **Executor**: Implements solutions and takes action
- **Validator**: Verifies results and ensures quality
- **Moderator**: Manages conflicts and maintains order

### Coordination Strategies

#### Available Strategies:
1. **Round Robin**: Agents take turns in sequence
2. **Expertise-Based**: Route to agent with relevant expertise
3. **Workload-Balanced**: Distribute based on current load
4. **Hierarchical**: Follow organizational hierarchy
5. **Democratic**: Agents vote on decisions
6. **Consensus**: Require agreement from all agents
7. **Performance-Based**: Route to highest-performing agents

### Conflict Resolution

When agents disagree, the system uses several resolution methods:

#### Conflict Types:
- **Resource Contention**: Multiple agents wanting same resources
- **Contradictory Actions**: Agents suggesting opposing actions
- **Priority Conflicts**: Disagreement on task priorities
- **Information Conflicts**: Contradictory information sources

#### Resolution Strategies:
- **Hierarchical**: Higher-ranking agent decides
- **Consensus**: Negotiate until agreement
- **Majority Vote**: Democratic decision making
- **Expertise-Based**: Defer to most knowledgeable agent
- **Performance-Based**: Trust highest-performing agent
- **Compromise**: Find middle-ground solution
- **Escalation**: Escalate to human operator
- **Round-Robin**: Rotate decision authority

### Performance Monitoring

Track multi-agent performance through:
- **Collaboration Scores**: How well agents work together
- **Task Completion Rates**: Success rate for assigned tasks
- **Response Times**: Speed of agent responses
- **Resource Utilization**: Efficiency of resource usage
- **Conflict Frequency**: How often conflicts arise
- **Resolution Success**: Effectiveness of conflict resolution

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Docker/Container Issues

**Problem**: Services won't start
```bash
# Solution: Check Docker status and restart
docker-compose -f docker-compose.dev.yml down
docker system prune -f
docker-compose -f docker-compose.dev.yml up -d
```

**Problem**: Port conflicts (8080, 8000, etc. already in use)
```bash
# Solution: Find and stop conflicting processes
# Windows
netstat -ano | findstr :8080
taskkill /PID <process_id> /F

# Linux/macOS
lsof -i :8080
kill -9 <process_id>
```

**Problem**: Out of memory errors
```bash
# Solution: Increase Docker memory allocation
# Docker Desktop -> Settings -> Resources -> Memory
# Increase to at least 8GB, recommended 16GB
```

#### API Issues

**Problem**: API returns 401 Unauthorized
- **Check JWT token** is valid and not expired
- **Verify API key** is correctly configured
- **Ensure user has** required permissions

**Problem**: AI model API errors
- **Verify API keys** are valid and have credits
- **Check network connectivity** to AI service providers
- **Review rate limits** and usage quotas

**Problem**: Database connection errors
- **Verify containers** are running: `docker ps`
- **Check network connectivity** between containers
- **Review database logs** for specific errors

#### Voice Processing Issues

**Problem**: Microphone not detected
- **Check browser permissions** for microphone access
- **Verify audio device** is connected and working
- **Test with system audio** settings

**Problem**: Poor voice recognition accuracy
- **Reduce background noise**
- **Speak clearly** and at moderate pace
- **Check microphone quality** and positioning
- **Verify language settings** match your speech

**Problem**: Text-to-speech not working
- **Check ElevenLabs API key** and credits
- **Verify speaker/audio output** device
- **Review TTS provider** configuration

#### Home Assistant Integration Issues

**Problem**: Can't connect to Home Assistant
- **Verify Home Assistant URL** is accessible
- **Check long-lived token** is valid
- **Ensure network connectivity** between containers and HA
- **Review firewall settings**

**Problem**: Voice commands not controlling devices
- **Verify entity names** match Home Assistant entities
- **Check entity aliases** configuration
- **Ensure devices are available** in Home Assistant
- **Review voice command logs**

### Log Analysis

#### Viewing System Logs
```bash
# View all service logs
docker-compose -f docker-compose.dev.yml logs

# View specific service logs
docker-compose -f docker-compose.dev.yml logs api
docker-compose -f docker-compose.dev.yml logs frontend

# Follow live logs
docker-compose -f docker-compose.dev.yml logs -f api
```

#### Important Log Locations
- **API Logs**: Container stdout/stderr
- **n8n Logs**: http://localhost:5678 -> Executions
- **Frontend Logs**: Browser developer console
- **Database Logs**: Container logs for mongodb, redis, postgres

#### Log Levels and Filtering
```bash
# Filter by log level
docker-compose logs api | grep ERROR
docker-compose logs api | grep WARNING

# Search for specific errors
docker-compose logs | grep "connection refused"
docker-compose logs | grep "authentication failed"
```

### Performance Issues

#### System Running Slowly
1. **Check resource usage**:
   ```bash
   docker stats
   ```
2. **Verify adequate RAM** allocation (minimum 8GB)
3. **Check disk space** availability
4. **Review database performance** metrics
5. **Consider scaling up** resources

#### High Memory Usage
1. **Monitor Redis memory** usage
2. **Check for memory leaks** in application logs
3. **Optimize conversation history** retention
4. **Consider implementing** memory cleanup routines

#### Network Connectivity Issues
1. **Test container networking**:
   ```bash
   docker network ls
   docker network inspect multiple-ai-conversation_default
   ```
2. **Check DNS resolution** within containers
3. **Verify port mappings** are correct
4. **Review firewall/security** group settings

---

## 🔧 Advanced Configuration

### Environment Variables Reference

#### Core System Settings
```env
# Application
APP_NAME=Multi-Agent Conversation Engine
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Security
JWT_SECRET_KEY=your_super_secret_key_here
JWT_EXPIRATION_HOURS=24
ENCRYPTION_KEY=your_32_byte_encryption_key
BCRYPT_ROUNDS=12

# Database URLs
REDIS_URL=redis://redis:6379/0
MONGODB_URL=mongodb://mongodb:27017/multi_agent_db
POSTGRES_URL=postgresql://user:password@postgres:5432/n8n_db

# AI Model Configuration
PRIMARY_AI_MODEL=claude
MODEL_ROUTING_STRATEGY=performance
ANTHROPIC_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_gemini_key

# Voice Processing
VOICE_PROCESSING_ENABLED=true
ELEVENLABS_API_KEY=your_elevenlabs_key
STT_MODEL=whisper-1
TTS_PROVIDER=elevenlabs

# Home Assistant
HOME_ASSISTANT_URL=http://homeassistant:8123
HOME_ASSISTANT_TOKEN=your_ha_token
HA_WEBSOCKET_ENABLED=true

# Performance Tuning
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT_SECONDS=300
MEMORY_CONSOLIDATION_INTERVAL=3600
API_RATE_LIMIT_PER_MINUTE=1000
```

#### Multi-Tenancy Configuration
```env
# Tenant Settings
MULTI_TENANCY_ENABLED=true
DEFAULT_TENANT_PLAN=free
TENANT_ISOLATION_LEVEL=strict

# Quota Defaults (per tenant)
DEFAULT_API_CALLS_PER_MONTH=10000
DEFAULT_STORAGE_MB=1000
DEFAULT_MAX_AGENTS=5
DEFAULT_MAX_WORKFLOWS=20
```

#### Scaling Configuration
```env
# Load Balancing
LOAD_BALANCING_STRATEGY=least_connections
HEALTH_CHECK_INTERVAL=30
FAILOVER_ENABLED=true

# Auto-scaling
AUTO_SCALING_ENABLED=true
SCALE_UP_THRESHOLD=0.8
SCALE_DOWN_THRESHOLD=0.3
MIN_INSTANCES=2
MAX_INSTANCES=10
```

### Custom Docker Configuration

#### Production Docker Compose
Create `docker-compose.prod.yml` for production deployment:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - DEBUG=false
      - LOG_LEVEL=WARNING
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: ../Dockerfile.frontend
    deploy:
      replicas: 2
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 2G
    restart: unless-stopped

volumes:
  redis_data:
  mongodb_data:
  postgres_data:
```

#### Custom Nginx Configuration
For production deployments with custom domains:

```nginx
# config/nginx/nginx.conf
upstream api_backend {
    server api:8000;
}

upstream frontend_backend {
    server frontend:3000;
}

server {
    listen 80;
    server_name your-domain.com;

    # API routes
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket routes
    location /ws/ {
        proxy_pass http://api_backend/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # Frontend
    location / {
        proxy_pass http://frontend_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Database Optimization

#### Redis Configuration
```conf
# config/redis.conf

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Performance
tcp-keepalive 300
timeout 300

# Security
requirepass your_redis_password
```

#### MongoDB Optimization
```javascript
// config/mongodb/init-mongo.js

// Create indexes for better performance
db = db.getSiblingDB('multi_agent_db');

// Conversation indexes
db.conversations.createIndex({ "tenant_id": 1, "created_at": -1 });
db.conversations.createIndex({ "user_id": 1, "updated_at": -1 });
db.conversations.createIndex({ "session_id": 1 });

// Memory indexes
db.memories.createIndex({ "conversation_id": 1, "timestamp": -1 });
db.memories.createIndex({ "importance_score": -1 });
db.memories.createIndex({ "tags": 1 });

// Vector indexes (requires MongoDB 6.0+)
db.vector_memories.createIndex({
    "embedding": "2dsphere"
});
```

---

## 📖 API Reference

### Authentication

#### Register New User
```http
POST /api/auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "password": "secure_password",
  "full_name": "John Doe",
  "roles": ["user"]
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "newuser",
  "password": "secure_password"
}
```

#### Response
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Conversation Management

#### Start New Conversation
```http
POST /api/conversations
Authorization: Bearer <token>
Content-Type: application/json

{
  "title": "My Conversation",
  "agents": ["claude", "gpt4"],
  "mode": "collaborative",
  "voice_enabled": true,
  "home_assistant_enabled": false
}
```

#### Send Message
```http
POST /api/conversations/{conversation_id}/messages
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "Hello, can you help me with a coding problem?",
  "sender": "user",
  "message_type": "text"
}
```

#### Get Conversation History
```http
GET /api/conversations/{conversation_id}/messages?limit=50&offset=0
Authorization: Bearer <token>
```

### Agent Management

#### List Available Agents
```http
GET /api/agents
Authorization: Bearer <token>
```

#### Create Agent
```http
POST /api/agents
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Research Assistant",
  "model": "claude",
  "role": "researcher",
  "personality": {
    "openness": 0.8,
    "conscientiousness": 0.9,
    "extraversion": 0.6,
    "agreeableness": 0.7,
    "neuroticism": 0.3
  },
  "capabilities": ["research", "analysis", "writing"]
}
```

#### Get Agent Status
```http
GET /api/agents/{agent_id}/status
Authorization: Bearer <token>
```

### Workflow Management

#### List Workflow Templates
```http
GET /api/workflows/templates
Authorization: Bearer <token>
```

#### Create Workflow from Template
```http
POST /api/workflows/from-template
Authorization: Bearer <token>
Content-Type: application/json

{
  "template_id": "brainstorming_template",
  "parameters": {
    "topic": "Product feature ideas"
  },
  "agents": ["creative", "analyst", "critic"]
}
```

#### Execute Workflow
```http
POST /api/workflows/{workflow_id}/execute
Authorization: Bearer <token>
```

#### Get Workflow Status
```http
GET /api/workflows/{workflow_id}/status
Authorization: Bearer <token>
```

### Voice Processing

#### Start Voice Session
```http
POST /api/voice/sessions
Authorization: Bearer <token>
Content-Type: application/json

{
  "conversation_id": "conv_123",
  "language": "en",
  "voice_id": "elevenlabs_voice_id"
}
```

#### Upload Audio
```http
POST /api/voice/transcribe
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <audio_file>
```

#### Generate Speech
```http
POST /api/voice/synthesize
Authorization: Bearer <token>
Content-Type: application/json

{
  "text": "Hello, this is a test message",
  "voice_id": "elevenlabs_voice_id",
  "voice_settings": {
    "speed": 1.0,
    "pitch": 0.0
  }
}
```

### Home Assistant Integration

#### Get Home Assistant Status
```http
GET /api/home-assistant/status
Authorization: Bearer <token>
```

#### List Entities
```http
GET /api/home-assistant/entities
Authorization: Bearer <token>
```

#### Control Entity
```http
POST /api/home-assistant/entities/{entity_id}/control
Authorization: Bearer <token>
Content-Type: application/json

{
  "action": "turn_on",
  "parameters": {
    "brightness": 255,
    "color": "blue"
  }
}
```

### Analytics and Monitoring

#### Get System Health
```http
GET /api/health
```

#### Get Metrics
```http
GET /api/metrics
Authorization: Bearer <token>
```

#### Get Usage Analytics
```http
GET /api/analytics/usage?period=30d
Authorization: Bearer <token>
```

### WebSocket Events

Connect to WebSocket for real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'agent_message':
            // Handle new agent message
            break;
        case 'agent_status_update':
            // Handle agent status change
            break;
        case 'workflow_update':
            // Handle workflow progress
            break;
        case 'system_alert':
            // Handle system alert
            break;
    }
};
```

---

## 🔒 Security Considerations

### Production Security Checklist

#### Environment Security
- [ ] **Change all default passwords** in `.env` file
- [ ] **Use strong, unique JWT secret** (32+ characters)
- [ ] **Generate secure encryption key** (32 bytes)
- [ ] **Enable HTTPS** with valid SSL certificates
- [ ] **Configure firewall** to restrict access to necessary ports only

#### Authentication & Authorization
- [ ] **Implement strong password policies** (minimum 12 characters)
- [ ] **Enable two-factor authentication** for admin accounts
- [ ] **Regularly rotate API keys** and access tokens
- [ ] **Review user permissions** and remove unnecessary access
- [ ] **Monitor authentication logs** for suspicious activity

#### Database Security
- [ ] **Use encrypted connections** to databases
- [ ] **Enable database authentication** with strong passwords
- [ ] **Regular database backups** with encryption
- [ ] **Limit database network access** to application servers only
- [ ] **Monitor database queries** for suspicious activity

#### API Security
- [ ] **Implement rate limiting** for all API endpoints
- [ ] **Use API key authentication** for service-to-service communication
- [ ] **Validate all input** and sanitize user data
- [ ] **Enable CORS** with specific allowed origins
- [ ] **Log all API requests** for audit purposes

#### Infrastructure Security
- [ ] **Keep Docker images updated** with latest security patches
- [ ] **Use non-root users** in Docker containers
- [ ] **Implement network segmentation** between services
- [ ] **Regular security scanning** of containers and dependencies
- [ ] **Monitor system logs** for security events

### Security Configuration Examples

#### Secure JWT Configuration
```env
# Use a strong, randomly generated secret
JWT_SECRET_KEY=your_super_long_random_secret_key_here_at_least_32_characters
JWT_EXPIRATION_HOURS=8  # Shorter for production
JWT_REFRESH_ENABLED=true
```

#### Rate Limiting Configuration
```env
# API Rate limits
GLOBAL_RATE_LIMIT_PER_MINUTE=10000
USER_RATE_LIMIT_PER_MINUTE=1000
API_KEY_RATE_LIMIT_PER_MINUTE=5000

# Progressive rate limiting
RATE_LIMIT_BURST_SIZE=100
RATE_LIMIT_WINDOW_SIZE=3600
```

#### HTTPS Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_dhparam /path/to/dhparam.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
}
```

---

## ⚡ Performance Optimization

### System Performance Tuning

#### Docker Resource Allocation
```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  redis:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

#### Database Optimization

##### Redis Performance
```conf
# config/redis.conf

# Memory optimization
maxmemory 2gb
maxmemory-policy allkeys-lru

# Network optimization
tcp-nodelay yes
tcp-keepalive 300

# Persistence optimization
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error no
```

##### MongoDB Performance
```javascript
// Optimize MongoDB for conversation storage
db.conversations.createIndex({ "tenant_id": 1, "created_at": -1 });
db.conversations.createIndex({ "user_id": 1, "session_id": 1 });
db.memories.createIndex({ "conversation_id": 1, "importance_score": -1 });

// Configure for better performance
db.adminCommand({
    "setParameter": 1,
    "internalQueryMaxBlockingSortMemoryUsageBytes": 335544320
});
```

#### Application Performance

##### AI Model Optimization
```env
# Model routing for performance
MODEL_ROUTING_STRATEGY=performance
PRIMARY_AI_MODEL=claude  # Fastest for most tasks

# Caching configuration
MODEL_RESPONSE_CACHE_TTL=3600
EMBEDDING_CACHE_TTL=86400

# Parallel processing
MAX_CONCURRENT_AI_REQUESTS=10
AI_REQUEST_TIMEOUT=30
```

##### Memory Management
```env
# Working memory optimization
WORKING_MEMORY_TTL=1800  # 30 minutes
WORKING_MEMORY_MAX_SIZE=10000

# Long-term memory optimization
MEMORY_CONSOLIDATION_INTERVAL=1800  # 30 minutes
MEMORY_CLEANUP_INTERVAL=3600  # 1 hour
```

### Scaling Configuration

#### Horizontal Scaling
```env
# Load balancing
LOAD_BALANCING_STRATEGY=least_response_time
HEALTH_CHECK_INTERVAL=15
CIRCUIT_BREAKER_ENABLED=true

# Auto-scaling
AUTO_SCALING_ENABLED=true
SCALE_UP_CPU_THRESHOLD=70
SCALE_DOWN_CPU_THRESHOLD=30
SCALE_UP_MEMORY_THRESHOLD=80
SCALE_DOWN_MEMORY_THRESHOLD=40

# Instance limits
MIN_API_INSTANCES=2
MAX_API_INSTANCES=10
MIN_AGENT_INSTANCES=1
MAX_AGENT_INSTANCES=20
```

#### Caching Strategy
```env
# Redis caching
REDIS_CACHE_ENABLED=true
CACHE_DEFAULT_TTL=3600

# Application caching
CONVERSATION_CACHE_TTL=1800
AGENT_STATUS_CACHE_TTL=300
MODEL_RESPONSE_CACHE_TTL=3600
```

### Monitoring and Alerts

#### Performance Monitoring
```env
# Metrics collection
METRICS_ENABLED=true
METRICS_INTERVAL=60
DETAILED_METRICS=true

# Performance thresholds
RESPONSE_TIME_THRESHOLD_MS=2000
ERROR_RATE_THRESHOLD=0.05
CPU_ALERT_THRESHOLD=80
MEMORY_ALERT_THRESHOLD=85
```

#### Health Check Configuration
```env
# Health check settings
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10

# Service dependencies
REDIS_HEALTH_CHECK=true
MONGODB_HEALTH_CHECK=true
AI_MODEL_HEALTH_CHECK=true
```

---

## 🆘 Getting Help

### Documentation Resources
- **API Documentation**: http://localhost:8000/docs (when running)
- **GitHub Repository**: https://github.com/Nitroedge/multiple-ai-conversation
- **Project Wiki**: Available in the GitHub repository

### Support Channels
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support and questions
- **Documentation**: This user guide and inline code documentation

### Common Support Questions

#### Q: How do I update the system?
A: Pull the latest changes from GitHub and restart the containers:
```bash
git pull origin main
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.dev.yml up -d --build
```

#### Q: Can I use this commercially?
A: Yes, the system is designed for commercial use. Review the license terms in the repository.

#### Q: How do I backup my data?
A: Backup the Docker volumes and your `.env` configuration:
```bash
# Backup volumes
docker run --rm -v multiple-ai-conversation_mongodb_data:/data -v $(pwd):/backup ubuntu tar czf /backup/mongodb_backup.tar.gz /data
docker run --rm -v multiple-ai-conversation_redis_data:/data -v $(pwd):/backup ubuntu tar czf /backup/redis_backup.tar.gz /data
```

#### Q: How do I migrate to a different server?
A: Copy your `.env` file, docker-compose configuration, and data volumes to the new server, then start the containers.

#### Q: What are the minimum system requirements?
A: 8GB RAM, 4 CPU cores, 10GB disk space, Docker Desktop, and stable internet connection.

---

## 📝 Changelog and Updates

### Version 1.0.0 (September 27, 2025)
- ✅ **Initial Release**: Complete enterprise-ready platform
- ✅ **Multi-Agent System**: Full coordination and collaboration
- ✅ **AI Integration**: Claude, GPT-4, Gemini, and local models
- ✅ **Voice Processing**: Complete STT/TTS pipeline
- ✅ **Home Assistant**: Full integration with ESP32 support
- ✅ **Enterprise Features**: Multi-tenancy, security, scaling
- ✅ **Admin Dashboard**: Real-time monitoring and management

### Upcoming Features (Phase 5)
- **Fine-tuning Support**: Custom model training
- **Multimodal Integration**: Vision and document processing
- **Mobile Applications**: iOS and Android apps
- **Plugin Marketplace**: Third-party extensions
- **Advanced Analytics**: Predictive insights and optimization

---

**🎉 Congratulations!** You now have a complete understanding of the Multi-Agent Conversation Engine. The system is production-ready and capable of handling enterprise-scale deployments with sophisticated AI-powered conversations, voice processing, home automation, and advanced coordination features.

For additional support or questions, please refer to the GitHub repository or create an issue for community support.

**Happy conversing with your AI agents!** 🤖✨
# Multi-Agent Conversation Engine v2.0

A next-generation multi-AI conversation system built with modern architecture, featuring dynamic character personalities, hierarchical memory management, and seamless integration with n8n workflows, Home Assistant, and voice services.

## 🚀 Features

### Core Architecture
- **Event-driven orchestration** using n8n workflows
- **Hierarchical memory system** with Redis working memory and MongoDB long-term storage
- **Vector-based memory retrieval** using embeddings for contextual responses
- **Dynamic character personalities** with Big Five psychological model adaptation
- **Multi-layer safety controls** with ML-based content moderation

### Integrations
- **n8n Workflows**: Automated conversation orchestration and home automation
- **Home Assistant**: Voice commands and IoT device control
- **ESP32 Hardware**: Edge audio processing and real-time streaming
- **Voice Services**: OpenAI Whisper (STT) + ElevenLabs (TTS) with voice cloning

### Advanced Features
- **Real-time WebSocket communication** with animated frontend
- **Character development tracking** with personality evolution
- **Conversation analytics** with engagement metrics
- **Privacy protection** with PII detection and anonymization
- **Comprehensive audit logging** for regulatory compliance

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Server    │    │   n8n Workflows │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Orchestrate) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼─────┐
        │    Redis     │ │   MongoDB   │ │PostgreSQL │
        │ (Working Mem)│ │(Long Memory)│ │(Analytics)│
        └──────────────┘ └─────────────┘ └───────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.11+), WebSockets, Celery
- **Frontend**: React 18, TypeScript, WebSocket client
- **Databases**: Redis 7 (state), MongoDB 6 (memory), PostgreSQL 16 (analytics)
- **Orchestration**: n8n workflows, NGINX reverse proxy
- **AI/ML**: OpenAI GPT-4, Sentence Transformers, scikit-learn
- **Voice**: OpenAI Whisper, ElevenLabs TTS, WebRTC streaming
- **Infrastructure**: Docker, Docker Compose, CI/CD pipelines

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- ElevenLabs API key
- 16GB+ RAM (recommended)
- NVIDIA GPU (optional, for local Whisper)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multiple-ai-conversation
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start development environment**
   ```bash
   chmod +x start-dev.sh
   ./start-dev.sh
   ```

4. **Access the application**
   - Main App: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - n8n Workflows: http://localhost:5678
   - Development Tools: See start-dev.sh output

### Manual Docker Setup

```bash
# Start core services
docker-compose -f docker-compose.dev.yml up -d

# Start with development tools
docker-compose -f docker-compose.dev.yml --profile dev-tools up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f api

# Stop services
docker-compose -f docker-compose.dev.yml down
```

## 📋 Development Roadmap

### Phase 1: Foundation & Core Architecture ✅ **COMPLETED**
- ✅ **Sprint 1.1**: Infrastructure setup with Docker orchestration
- ✅ **Sprint 1.2**: Core memory system with hierarchical architecture
- ✅ **Sprint 1.3**: Basic n8n orchestration workflows
- ✅ **Sprint 1.4**: Character framework foundation

### Phase 2: Advanced Features ✅ **COMPLETED**
- ✅ **Sprint 2.1**: Voice processing pipeline integration
- ✅ **Sprint 2.2**: Home Assistant integration
- ✅ **Sprint 2.3**: Advanced agent coordination

### Phase 3: Frontend Interface Development ✅ **COMPLETED**
- ✅ **Sprint 3.1**: React application architecture
- ✅ **Sprint 3.2**: Real-time WebSocket integration
- ✅ **Sprint 3.3**: Agent coordination dashboard
- ✅ **Sprint 3.4**: Professional UI components

### Phase 4: Production Deployment & Enterprise Features ✅ **COMPLETED**
- ✅ **Sprint 4.1**: API stabilization & production optimization
- ✅ **Sprint 4.2**: Advanced AI integration (Multi-model support)
- ✅ **Sprint 4.3**: Advanced orchestration features
- ✅ **Sprint 4.4**: Enterprise features & scaling

### Phase 5: Future Enhancements 📋 **PLANNED**
- 📋 **Sprint 5.1**: Advanced AI capabilities (Fine-tuning, multimodal)
- 📋 **Sprint 5.2**: Enterprise integration (SSO, API gateway)
- 📋 **Sprint 5.3**: Advanced analytics & AI optimization
- 📋 **Sprint 5.4**: Ecosystem expansion (plugins, mobile apps)

## 🎯 **Current Status: 95% Complete - Production Ready!**

**🎉 Major Achievement**: All core development phases (1-4) are now **COMPLETE**! The system is enterprise-ready with advanced AI integration, multi-tenancy, security, scaling, and comprehensive monitoring.

## 🏛️ System Architecture

### Memory Hierarchy
```
Working Memory (Redis)     Long-term Memory (MongoDB)
├── Active conversations   ├── Consolidated memories
├── Agent states          ├── Character development
├── Context cache         ├── Conversation analytics
└── Session data          └── Vector embeddings
```

### Character System
- **Dynamic Personalities**: Big Five model with real-time adaptation
- **Emotional States**: Multi-dimensional emotion tracking with decay
- **Memory Integration**: Character-specific memory and relationships
- **Development Tracking**: Personality evolution over time

### Safety Architecture
- **Layer 1**: Rule-based input validation
- **Layer 2**: ML content classification
- **Layer 3**: Contextual safety evaluation
- **Layer 4**: Agent response monitoring
- **Layer 5**: Human oversight and escalation

## 🔧 API Documentation

### Core Endpoints
- `POST /api/conversations/start` - Start new conversation
- `GET /api/conversations/{id}/state` - Get conversation state
- `POST /api/memory/store` - Store memory item
- `GET /api/memory/search` - Search memories
- `POST /api/agents/response` - Generate agent response

### WebSocket Events
- `conversation_message` - New message in conversation
- `agent_typing` - Agent is generating response
- `state_update` - Conversation state changed
- `error` - Error occurred

### n8n Webhooks
- `/webhook/voice-command` - Process voice input
- `/webhook/home-automation` - Execute home automation
- `/webhook/agent-coordination` - Coordinate agent responses

## 🧪 Testing

```bash
# Run all tests
docker-compose -f docker-compose.dev.yml exec api pytest

# Run specific test category
docker-compose -f docker-compose.dev.yml exec api pytest tests/memory/
docker-compose -f docker-compose.dev.yml exec api pytest tests/agents/

# Run with coverage
docker-compose -f docker-compose.dev.yml exec api pytest --cov=src
```

## 📊 Monitoring & Analytics

### Built-in Dashboards
- **Conversation Analytics**: Engagement metrics, topic analysis
- **Agent Performance**: Response times, consistency scores
- **Memory Usage**: Storage optimization, retrieval patterns
- **Safety Metrics**: Moderation events, risk assessments

### Health Checks
- `GET /health` - Overall system health
- `GET /health/redis` - Redis connection status
- `GET /health/mongodb` - MongoDB connection status
- `GET /health/memory` - Memory system status

## 🔒 Security & Privacy

### Privacy Protection
- **PII Detection**: Automatic identification and anonymization
- **Data Minimization**: Configurable retention policies
- **Consent Management**: User control over data processing
- **Audit Trails**: Comprehensive logging for compliance

### Security Measures
- **Input Validation**: Multi-layer content filtering
- **Rate Limiting**: API and conversation throttling
- **Authentication**: JWT-based user authentication
- **Encryption**: Data at rest and in transit

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive tests
- Update documentation for new features
- Use conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Multi-Agent GPT by DougDoug
- OpenAI for GPT-4 and Whisper models
- ElevenLabs for voice synthesis technology
- n8n community for workflow automation
- Home Assistant for IoT integration

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Wiki**: [GitHub Wiki](../../wiki)

---

Built with ❤️ by the Multi-Agent Conversation Engine team
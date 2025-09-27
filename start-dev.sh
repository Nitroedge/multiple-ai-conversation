#!/bin/bash

# Multi-Agent Conversation Engine - Development Startup Script
# This script sets up and starts the complete development environment

set -e

echo "🚀 Starting Multi-Agent Conversation Engine Development Environment"
echo "================================================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if required environment file exists
if [ ! -f .env ]; then
    echo "📝 Creating environment file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before proceeding:"
    echo "   - OPENAI_API_KEY"
    echo "   - ELEVENLABS_API_KEY"
    echo ""
    read -p "Press Enter after updating .env file..."
fi

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs audio_cache frontend/src frontend/public n8n/custom-nodes

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.dev.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."

# Check Redis
echo "🔍 Checking Redis connection..."
until docker-compose -f docker-compose.dev.yml exec redis-primary redis-cli ping >/dev/null 2>&1; do
    echo "   Redis not ready, waiting..."
    sleep 2
done
echo "✅ Redis is ready"

# Check MongoDB
echo "🔍 Checking MongoDB connection..."
until docker-compose -f docker-compose.dev.yml exec mongodb mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; do
    echo "   MongoDB not ready, waiting..."
    sleep 2
done
echo "✅ MongoDB is ready"

# Check PostgreSQL
echo "🔍 Checking PostgreSQL connection..."
until docker-compose -f docker-compose.dev.yml exec postgres pg_isready -U agent_user >/dev/null 2>&1; do
    echo "   PostgreSQL not ready, waiting..."
    sleep 2
done
echo "✅ PostgreSQL is ready"

# Check n8n
echo "🔍 Checking n8n connection..."
until curl -f http://localhost:5678/healthz >/dev/null 2>&1; do
    echo "   n8n not ready, waiting..."
    sleep 5
done
echo "✅ n8n is ready"

echo ""
echo "🎉 Development environment is ready!"
echo "==============================================="
echo ""
echo "📊 Service URLs:"
echo "   🌐 Main Application:     http://localhost:3000"
echo "   🔧 API Documentation:    http://localhost:8000/docs"
echo "   🔀 n8n Workflows:        http://localhost:5678"
echo "   🗄️  Redis Commander:      http://localhost:8081"
echo "   🍃 Mongo Express:        http://localhost:8082"
echo "   🐘 pgAdmin:              http://localhost:8083"
echo ""
echo "🔐 Default Credentials:"
echo "   n8n:           admin / multi_agent_n8n_2024"
echo "   Redis Comm.:   (no auth required)"
echo "   Mongo Express: admin / multi_agent_mongo_2024"
echo "   pgAdmin:       admin@multiagent.dev / multi_agent_pg_2024"
echo ""
echo "📋 Useful Commands:"
echo "   View logs:     docker-compose -f docker-compose.dev.yml logs -f [service]"
echo "   Stop all:      docker-compose -f docker-compose.dev.yml down"
echo "   Restart:       docker-compose -f docker-compose.dev.yml restart [service]"
echo "   Shell access:  docker-compose -f docker-compose.dev.yml exec [service] /bin/bash"
echo ""
echo "🔧 Development Tools:"
echo "   To start with dev tools: docker-compose -f docker-compose.dev.yml --profile dev-tools up -d"
echo ""
echo "Happy coding! 🎯"
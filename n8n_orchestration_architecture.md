# n8n-Based Multi-Agent Conversation Engine Architecture

## Executive Summary

This document presents a comprehensive n8n-based orchestration system architecture to replace the current threading-based multi-agent conversation engine. The new architecture leverages n8n's event-driven workflow capabilities to create a more scalable, maintainable, and robust system for managing multi-agent AI conversations, voice processing, and home automation integration.

## Current System Analysis & Limitations

### Threading-Based Architecture Issues
1. **Tight Coupling**: All components run in a single Python process with shared locks
2. **Resource Contention**: Speaking and conversation locks create bottlenecks
3. **Single Point of Failure**: Entire system crashes if any component fails
4. **Scaling Limitations**: Cannot distribute load across multiple machines
5. **State Management**: Global variables and shared memory create race conditions
6. **Debugging Complexity**: Multi-threaded execution makes troubleshooting difficult

### Current Components Analysis
- **Main Thread**: Flask web app with SocketIO
- **Agent Threads**: 3 AI agents with conversation/speaking locks
- **Human Input Thread**: Keyboard input processing
- **Service Dependencies**: OpenAI, ElevenLabs, Whisper, OBS WebSockets

## 1. n8n Workflow Architecture

### Core Design Principles
- **Event-Driven**: Asynchronous message passing between workflows
- **Microservice Pattern**: Each component as independent workflow
- **State Isolation**: Persistent state in database, not memory
- **Horizontal Scaling**: Multiple n8n instances with load balancing
- **Fault Tolerance**: Individual workflow failures don't crash system

### Primary Workflow Categories

#### 1.1 Conversation Orchestrator Workflow
**Purpose**: Central coordinator for all conversation events
```
Trigger: Webhook (conversation events)
├─ Event Router Node
│  ├─ Agent Activation Request
│  ├─ Human Input Request
│  ├─ Conversation State Change
│  └─ System Control Commands
├─ State Validation Node
├─ Queue Management Node
└─ Event Distribution Node
```

#### 1.2 Agent Processing Workflows (3 instances)
**Purpose**: Individual AI agent conversation processing
```
Trigger: Webhook (agent activation)
├─ Agent State Check Node
├─ Conversation Lock Acquisition Node
├─ OpenAI API Call Node
├─ Response Processing Node
├─ State Update Node
├─ TTS Generation Trigger Node
└─ Next Agent Selection Node
```

#### 1.3 Voice Processing Pipeline
**Purpose**: Handle TTS generation and audio playback coordination
```
Trigger: Webhook (TTS request)
├─ ElevenLabs TTS Generation Node
├─ Audio File Storage Node
├─ Whisper Subtitle Generation Node
├─ Audio Queue Management Node
├─ Playback Coordination Node
└─ Cleanup Node
```

#### 1.4 Human Input Processing Workflow
**Purpose**: Process human voice input and commands
```
Trigger: Webhook (human input)
├─ Input Type Detection Node
│  ├─ Voice Recording Branch
│  │  ├─ Audio Processing Node
│  │  ├─ Whisper Transcription Node
│  │  └─ Conversation Integration Node
│  └─ Command Branch
│     ├─ Command Parsing Node
│     └─ System Control Node
└─ Response Routing Node
```

## 2. Integration Patterns

### 2.1 Webhook Endpoints for Real-time Communication

#### Central Event Hub
```
POST /webhooks/conversation/event
Body: {
  "type": "agent_activation|human_input|system_control",
  "payload": {
    "agent_id": "agent_1|agent_2|agent_3|human",
    "content": "message content",
    "metadata": {}
  },
  "timestamp": "ISO-8601",
  "correlation_id": "uuid"
}
```

#### Agent-Specific Endpoints
```
POST /webhooks/agent/{agent_id}/activate
POST /webhooks/agent/{agent_id}/response
POST /webhooks/conversation/add_message
POST /webhooks/audio/generate_tts
POST /webhooks/audio/play
POST /webhooks/system/pause
POST /webhooks/system/resume
```

#### State Management Endpoints
```
GET /webhooks/conversation/state
POST /webhooks/conversation/state/update
GET /webhooks/agent/{agent_id}/status
POST /webhooks/queue/speaking/acquire
POST /webhooks/queue/speaking/release
```

### 2.2 Home Assistant Service Integration

#### Home Assistant Workflow
```
Trigger: Webhook (HA integration)
├─ Service Call Router Node
│  ├─ Automation Trigger Branch
│  │  ├─ Entity State Change Detection
│  │  ├─ Event Filtering Node
│  │  └─ Conversation Trigger Node
│  ├─ TTS Announcement Branch
│  │  ├─ Message Formatting Node
│  │  ├─ Priority Queue Node
│  │  └─ Audio Output Node
│  └─ Device Control Branch
│     ├─ Command Parsing Node
│     ├─ Service Call Node
│     └─ Confirmation Response Node
└─ Response Handler Node
```

#### Integration Points
- **Voice Commands**: "Turn on living room lights" → HA service call
- **Automation Triggers**: Motion sensor → conversation activation
- **Status Announcements**: "Low battery on smoke detector" → agent notification
- **Scene Control**: "Movie time" → lights + conversation pause

### 2.3 TTS/STT Service Integration Architecture

#### Multi-Provider TTS Workflow
```
Trigger: Webhook (TTS request)
├─ Provider Selection Node
│  ├─ ElevenLabs Branch (primary)
│  ├─ OpenAI TTS Branch (fallback)
│  └─ Azure Speech Branch (backup)
├─ Rate Limiting Node
├─ Quality Assessment Node
├─ Caching Check Node
├─ Audio Generation Node
├─ Post-Processing Node
└─ Storage & Delivery Node
```

#### STT Processing Pipeline
```
Trigger: Webhook (audio upload)
├─ Audio Format Detection Node
├─ Preprocessing Node
├─ Provider Selection Node
│  ├─ Whisper Local Branch (primary)
│  ├─ OpenAI Whisper API Branch (fallback)
│  └─ Azure Speech-to-Text Branch (backup)
├─ Transcription Node
├─ Confidence Scoring Node
├─ Text Processing Node
└─ Integration Response Node
```

### 2.4 Database Persistence Architecture

#### Conversation History Schema
```sql
CREATE TABLE conversations (
  id UUID PRIMARY KEY,
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  participants JSON,
  metadata JSON
);

CREATE TABLE messages (
  id UUID PRIMARY KEY,
  conversation_id UUID REFERENCES conversations(id),
  agent_id VARCHAR(50),
  role VARCHAR(20), -- 'user', 'assistant', 'system'
  content TEXT,
  audio_file_path VARCHAR(500),
  created_at TIMESTAMP,
  metadata JSON
);

CREATE TABLE agent_states (
  agent_id VARCHAR(50) PRIMARY KEY,
  is_active BOOLEAN,
  current_conversation_id UUID,
  voice_settings JSON,
  system_prompt TEXT,
  updated_at TIMESTAMP
);

CREATE TABLE speaking_queue (
  id UUID PRIMARY KEY,
  agent_id VARCHAR(50),
  priority INTEGER,
  audio_file_path VARCHAR(500),
  estimated_duration FLOAT,
  status VARCHAR(20), -- 'queued', 'playing', 'completed'
  created_at TIMESTAMP
);
```

#### Database Integration Workflow
```
Trigger: Database Operation Request
├─ Operation Type Router Node
│  ├─ Message Storage Branch
│  │  ├─ Conversation History Update
│  │  ├─ Vector Embedding Generation
│  │  └─ Index Update
│  ├─ State Management Branch
│  │  ├─ Agent Status Update
│  │  ├─ Queue State Update
│  │  └─ System State Update
│  └─ Retrieval Branch
│     ├─ Conversation History Query
│     ├─ Semantic Search Node
│     └─ Response Formatting
└─ Response Node
```

## 3. Specific n8n Workflow Examples

### 3.1 Voice Command Processing Pipeline

```json
{
  "name": "Voice Command Processing Pipeline",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "voice-command",
        "responseMode": "responseNode"
      },
      "id": "webhook-trigger",
      "name": "Voice Command Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": true,
            "leftValue": "={{ $json.command_type }}",
            "operation": "equal",
            "rightValue": "voice_recording"
          }
        }
      },
      "id": "command-type-switch",
      "name": "Command Type Switch",
      "type": "n8n-nodes-base.if",
      "position": [460, 300]
    },
    {
      "parameters": {
        "url": "http://whisper-service:8000/transcribe",
        "options": {
          "bodyContentType": "multipart-form-data",
          "headers": {
            "Authorization": "Bearer {{ $env.WHISPER_API_KEY }}"
          }
        },
        "sendBinaryData": true,
        "binaryPropertyName": "audio_file"
      },
      "id": "whisper-transcription",
      "name": "Whisper Transcription",
      "type": "n8n-nodes-base.httpRequest",
      "position": [680, 200]
    },
    {
      "parameters": {
        "jsCode": "const transcription = $input.first().json.transcription;\nconst confidence = $input.first().json.confidence;\n\nif (confidence < 0.7) {\n  return [{ json: { error: 'Low confidence transcription', confidence } }];\n}\n\n// Parse commands\nconst commands = {\n  'activate agent': /activate\\s+(agent\\s+)?(\\d+|one|two|three)/i,\n  'pause conversation': /pause|stop|hold/i,\n  'resume conversation': /resume|continue|start/i,\n  'home automation': /(turn|switch)\\s+(on|off)\\s+(.+)/i\n};\n\nlet command_type = 'conversation';\nlet parsed_command = null;\n\nfor (const [type, regex] of Object.entries(commands)) {\n  const match = transcription.match(regex);\n  if (match) {\n    command_type = type;\n    parsed_command = match;\n    break;\n  }\n}\n\nreturn [{\n  json: {\n    transcription,\n    confidence,\n    command_type,\n    parsed_command,\n    timestamp: new Date().toISOString()\n  }\n}];"
      },
      "id": "command-parser",
      "name": "Command Parser",
      "type": "n8n-nodes-base.code",
      "position": [900, 200]
    },
    {
      "parameters": {
        "conditions": {
          "options": {
            "caseSensitive": false,
            "leftValue": "={{ $json.command_type }}",
            "operation": "equal"
          }
        },
        "combineOperation": "any",
        "conditions": {
          "string": [
            {
              "leftValue": "={{ $json.command_type }}",
              "operation": "equal",
              "rightValue": "activate agent"
            }
          ]
        }
      },
      "id": "command-router",
      "name": "Command Router",
      "type": "n8n-nodes-base.switch",
      "position": [1120, 200],
      "parameters": {
        "options": {
          "allMatchingOutputs": false
        },
        "rules": {
          "rules": [
            {
              "operation": "equal",
              "value": "activate agent",
              "output": 0
            },
            {
              "operation": "equal",
              "value": "pause conversation",
              "output": 1
            },
            {
              "operation": "equal",
              "value": "home automation",
              "output": 2
            },
            {
              "operation": "equal",
              "value": "conversation",
              "output": 3
            }
          ]
        }
      }
    },
    {
      "parameters": {
        "url": "http://n8n:5678/webhook/agent/activate",
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "agent_id",
              "value": "={{ $json.parsed_command[2] }}"
            },
            {
              "name": "trigger_source",
              "value": "voice_command"
            }
          ]
        }
      },
      "id": "agent-activation",
      "name": "Agent Activation",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1340, 100]
    },
    {
      "parameters": {
        "url": "http://n8n:5678/webhook/conversation/add-message",
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "agent_id",
              "value": "human"
            },
            {
              "name": "content",
              "value": "={{ $json.transcription }}"
            },
            {
              "name": "role",
              "value": "user"
            }
          ]
        }
      },
      "id": "conversation-integration",
      "name": "Add to Conversation",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1340, 300]
    }
  ],
  "connections": {
    "Voice Command Webhook": {
      "main": [
        [
          {
            "node": "Command Type Switch",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Command Type Switch": {
      "main": [
        [
          {
            "node": "Whisper Transcription",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Whisper Transcription": {
      "main": [
        [
          {
            "node": "Command Parser",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Command Parser": {
      "main": [
        [
          {
            "node": "Command Router",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Command Router": {
      "main": [
        [
          {
            "node": "Agent Activation",
            "type": "main",
            "index": 0
          }
        ],
        [],
        [],
        [
          {
            "node": "Add to Conversation",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### 3.2 Multi-Agent Conversation Flow

```json
{
  "name": "Multi-Agent Conversation Controller",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "agent/{{ $parameter.agent_id }}/activate",
        "responseMode": "responseNode"
      },
      "id": "agent-activation-webhook",
      "name": "Agent Activation Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "operation": "get",
        "key": "conversation_lock",
        "options": {
          "ttl": 30
        }
      },
      "id": "acquire-conversation-lock",
      "name": "Acquire Conversation Lock",
      "type": "n8n-nodes-base.redis",
      "position": [460, 300]
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "leftValue": "={{ $json.lock_acquired }}",
              "operation": "equal",
              "rightValue": true
            }
          ]
        }
      },
      "id": "lock-check",
      "name": "Lock Check",
      "type": "n8n-nodes-base.if",
      "position": [680, 300]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT content, role, created_at FROM messages WHERE conversation_id = $1 ORDER BY created_at DESC LIMIT 20",
        "additionalFields": {
          "queryParameters": "{{ $json.conversation_id }}"
        }
      },
      "id": "get-conversation-history",
      "name": "Get Conversation History",
      "type": "n8n-nodes-base.postgres",
      "position": [900, 200]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT agent_id, system_prompt, voice_settings FROM agent_states WHERE agent_id = $1",
        "additionalFields": {
          "queryParameters": "{{ $json.agent_id }}"
        }
      },
      "id": "get-agent-config",
      "name": "Get Agent Config",
      "type": "n8n-nodes-base.postgres",
      "position": [900, 300]
    },
    {
      "parameters": {
        "jsCode": "const history = $('Get Conversation History').all();\nconst config = $('Get Agent Config').first().json;\n\n// Build OpenAI messages array\nconst messages = [{\n  role: 'system',\n  content: config.system_prompt\n}];\n\n// Add conversation history\nhistory.forEach(msg => {\n  messages.push({\n    role: msg.json.role,\n    content: msg.json.content\n  });\n});\n\n// Add conversation prompt\nmessages.push({\n  role: 'user',\n  content: 'Okay what is your response? Try to be as chaotic and bizarre and adult-humor oriented as possible. Again, 3 sentences maximum.'\n});\n\nreturn [{\n  json: {\n    messages,\n    agent_id: config.agent_id,\n    voice_settings: config.voice_settings\n  }\n}];"
      },
      "id": "build-openai-request",
      "name": "Build OpenAI Request",
      "type": "n8n-nodes-base.code",
      "position": [1120, 250]
    },
    {
      "parameters": {
        "resource": "chat",
        "operation": "create",
        "model": "gpt-4o",
        "messages": "={{ $json.messages }}",
        "options": {
          "maxTokens": 150,
          "temperature": 0.9
        }
      },
      "id": "openai-chat",
      "name": "OpenAI Chat",
      "type": "n8n-nodes-base.openAi",
      "position": [1340, 250]
    },
    {
      "parameters": {
        "jsCode": "const response = $input.first().json;\nconst content = response.choices[0].message.content.replace(/\\*/g, '');\nconst agent_id = $('Build OpenAI Request').first().json.agent_id;\n\nreturn [{\n  json: {\n    agent_id,\n    content,\n    role: 'assistant',\n    timestamp: new Date().toISOString()\n  }\n}];"
      },
      "id": "process-response",
      "name": "Process Response",
      "type": "n8n-nodes-base.code",
      "position": [1560, 250]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "INSERT INTO messages (conversation_id, agent_id, role, content, created_at) VALUES ($1, $2, $3, $4, $5) RETURNING id",
        "additionalFields": {
          "queryParameters": "{{ $json.conversation_id }},{{ $json.agent_id }},{{ $json.role }},{{ $json.content }},{{ $json.timestamp }}"
        }
      },
      "id": "save-message",
      "name": "Save Message",
      "type": "n8n-nodes-base.postgres",
      "position": [1780, 250]
    },
    {
      "parameters": {
        "url": "http://n8n:5678/webhook/audio/generate-tts",
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "text",
              "value": "={{ $json.content }}"
            },
            {
              "name": "agent_id",
              "value": "={{ $json.agent_id }}"
            },
            {
              "name": "voice_settings",
              "value": "={{ $('Build OpenAI Request').first().json.voice_settings }}"
            }
          ]
        }
      },
      "id": "trigger-tts",
      "name": "Trigger TTS Generation",
      "type": "n8n-nodes-base.httpRequest",
      "position": [2000, 250]
    },
    {
      "parameters": {
        "operation": "delete",
        "key": "conversation_lock"
      },
      "id": "release-conversation-lock",
      "name": "Release Conversation Lock",
      "type": "n8n-nodes-base.redis",
      "position": [2220, 250]
    },
    {
      "parameters": {
        "jsCode": "// Select next agent randomly\nconst currentAgent = $json.agent_id;\nconst allAgents = ['agent_1', 'agent_2', 'agent_3'];\nconst otherAgents = allAgents.filter(id => id !== currentAgent);\nconst nextAgent = otherAgents[Math.floor(Math.random() * otherAgents.length)];\n\n// Check if agents are paused\nconst agentsPaused = $('Check System State').first().json.agents_paused;\n\nreturn [{\n  json: {\n    next_agent: nextAgent,\n    should_activate: !agentsPaused,\n    current_agent: currentAgent\n  }\n}];"
      },
      "id": "select-next-agent",
      "name": "Select Next Agent",
      "type": "n8n-nodes-base.code",
      "position": [2440, 250]
    }
  ],
  "connections": {
    "Agent Activation Webhook": {
      "main": [
        [
          {
            "node": "Acquire Conversation Lock",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Acquire Conversation Lock": {
      "main": [
        [
          {
            "node": "Lock Check",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Lock Check": {
      "main": [
        [
          {
            "node": "Get Conversation History",
            "type": "main",
            "index": 0
          },
          {
            "node": "Get Agent Config",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Get Conversation History": {
      "main": [
        [
          {
            "node": "Build OpenAI Request",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Get Agent Config": {
      "main": [
        [
          {
            "node": "Build OpenAI Request",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Build OpenAI Request": {
      "main": [
        [
          {
            "node": "OpenAI Chat",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "OpenAI Chat": {
      "main": [
        [
          {
            "node": "Process Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Process Response": {
      "main": [
        [
          {
            "node": "Save Message",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Save Message": {
      "main": [
        [
          {
            "node": "Trigger TTS Generation",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Trigger TTS Generation": {
      "main": [
        [
          {
            "node": "Release Conversation Lock",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Release Conversation Lock": {
      "main": [
        [
          {
            "node": "Select Next Agent",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### 3.3 Home Automation Execution Workflow

```json
{
  "name": "Home Automation Execution",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "home-automation/execute",
        "responseMode": "responseNode"
      },
      "id": "ha-webhook",
      "name": "Home Automation Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "jsCode": "const command = $json.command.toLowerCase();\nconst patterns = {\n  lights: {\n    on: /(turn|switch)\\s+on\\s+(.+?)\\s+light/,\n    off: /(turn|switch)\\s+off\\s+(.+?)\\s+light/,\n    dim: /dim\\s+(.+?)\\s+light\\s+to\\s+(\\d+)/\n  },\n  climate: {\n    temp: /set\\s+temperature\\s+to\\s+(\\d+)/,\n    ac: /(turn|switch)\\s+(on|off)\\s+air\\s+conditioning/\n  },\n  security: {\n    arm: /arm\\s+security\\s+system/,\n    disarm: /disarm\\s+security\\s+system/\n  },\n  scene: {\n    activate: /activate\\s+(.+?)\\s+scene/\n  }\n};\n\nlet deviceType = null;\nlet action = null;\nlet target = null;\nlet value = null;\n\nfor (const [type, actions] of Object.entries(patterns)) {\n  for (const [act, regex] of Object.entries(actions)) {\n    const match = command.match(regex);\n    if (match) {\n      deviceType = type;\n      action = act;\n      target = match[2] || match[1];\n      value = match[3] || match[2];\n      break;\n    }\n  }\n  if (deviceType) break;\n}\n\nreturn [{\n  json: {\n    deviceType,\n    action,\n    target,\n    value,\n    originalCommand: command,\n    success: deviceType !== null\n  }\n}];"
      },
      "id": "command-parser",
      "name": "Command Parser",
      "type": "n8n-nodes-base.code",
      "position": [460, 300]
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "leftValue": "={{ $json.success }}",
              "operation": "equal",
              "rightValue": true
            }
          ]
        }
      },
      "id": "command-validation",
      "name": "Command Validation",
      "type": "n8n-nodes-base.if",
      "position": [680, 300]
    },
    {
      "parameters": {
        "options": {
          "allMatchingOutputs": false
        },
        "rules": {
          "rules": [
            {
              "operation": "equal",
              "value": "lights",
              "output": 0
            },
            {
              "operation": "equal",
              "value": "climate",
              "output": 1
            },
            {
              "operation": "equal",
              "value": "security",
              "output": 2
            },
            {
              "operation": "equal",
              "value": "scene",
              "output": 3
            }
          ]
        },
        "conditions": {
          "string": [
            {
              "leftValue": "={{ $json.deviceType }}",
              "operation": "equal"
            }
          ]
        }
      },
      "id": "device-type-router",
      "name": "Device Type Router",
      "type": "n8n-nodes-base.switch",
      "position": [900, 300]
    },
    {
      "parameters": {
        "url": "http://homeassistant:8123/api/services/light/{{ $json.action }}",
        "authentication": "headerAuth",
        "headerAuth": {
          "name": "Authorization",
          "value": "Bearer {{ $env.HA_TOKEN }}"
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "entity_id",
              "value": "light.{{ $json.target.replace(' ', '_') }}"
            }
          ]
        },
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        }
      },
      "id": "lights-control",
      "name": "Lights Control",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1120, 200]
    },
    {
      "parameters": {
        "url": "http://homeassistant:8123/api/services/climate/set_temperature",
        "authentication": "headerAuth",
        "headerAuth": {
          "name": "Authorization",
          "value": "Bearer {{ $env.HA_TOKEN }}"
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "entity_id",
              "value": "climate.main_thermostat"
            },
            {
              "name": "temperature",
              "value": "={{ $json.value }}"
            }
          ]
        },
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        }
      },
      "id": "climate-control",
      "name": "Climate Control",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1120, 300]
    },
    {
      "parameters": {
        "url": "http://homeassistant:8123/api/services/scene/turn_on",
        "authentication": "headerAuth",
        "headerAuth": {
          "name": "Authorization",
          "value": "Bearer {{ $env.HA_TOKEN }}"
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "entity_id",
              "value": "scene.{{ $json.target.replace(' ', '_') }}"
            }
          ]
        },
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        }
      },
      "id": "scene-activation",
      "name": "Scene Activation",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1120, 500]
    },
    {
      "parameters": {
        "url": "http://n8n:5678/webhook/conversation/add-message",
        "options": {
          "headers": {
            "Content-Type": "application/json"
          }
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "agent_id",
              "value": "system"
            },
            {
              "name": "content",
              "value": "{{ $json.action }} completed for {{ $json.target }}"
            },
            {
              "name": "role",
              "value": "system"
            }
          ]
        }
      },
      "id": "confirmation-message",
      "name": "Confirmation Message",
      "type": "n8n-nodes-base.httpRequest",
      "position": [1340, 350]
    }
  ],
  "connections": {
    "Home Automation Webhook": {
      "main": [
        [
          {
            "node": "Command Parser",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Command Parser": {
      "main": [
        [
          {
            "node": "Command Validation",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Command Validation": {
      "main": [
        [
          {
            "node": "Device Type Router",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Device Type Router": {
      "main": [
        [
          {
            "node": "Lights Control",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Climate Control",
            "type": "main",
            "index": 0
          }
        ],
        [],
        [
          {
            "node": "Scene Activation",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Lights Control": {
      "main": [
        [
          {
            "node": "Confirmation Message",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Climate Control": {
      "main": [
        [
          {
            "node": "Confirmation Message",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Scene Activation": {
      "main": [
        [
          {
            "node": "Confirmation Message",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### 3.4 Memory Persistence and Retrieval Workflow

```json
{
  "name": "Memory Persistence and Retrieval",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "memory/{{ $parameter.operation }}",
        "responseMode": "responseNode"
      },
      "id": "memory-webhook",
      "name": "Memory Operation Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "options": {
          "allMatchingOutputs": false
        },
        "rules": {
          "rules": [
            {
              "operation": "equal",
              "value": "store",
              "output": 0
            },
            {
              "operation": "equal",
              "value": "retrieve",
              "output": 1
            },
            {
              "operation": "equal",
              "value": "search",
              "output": 2
            }
          ]
        },
        "conditions": {
          "string": [
            {
              "leftValue": "={{ $json.operation }}",
              "operation": "equal"
            }
          ]
        }
      },
      "id": "operation-router",
      "name": "Operation Router",
      "type": "n8n-nodes-base.switch",
      "position": [460, 300]
    },
    {
      "parameters": {
        "url": "http://embedding-service:8000/embed",
        "options": {
          "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {{ $env.OPENAI_API_KEY }}"
          }
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "text",
              "value": "={{ $json.content }}"
            },
            {
              "name": "model",
              "value": "text-embedding-3-small"
            }
          ]
        }
      },
      "id": "generate-embedding",
      "name": "Generate Embedding",
      "type": "n8n-nodes-base.httpRequest",
      "position": [680, 200]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "INSERT INTO conversation_memory (conversation_id, agent_id, content, embedding, metadata, created_at) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        "additionalFields": {
          "queryParameters": "{{ $json.conversation_id }},{{ $json.agent_id }},{{ $json.content }},{{ $('Generate Embedding').first().json.embedding }},{{ $json.metadata }},{{ $json.timestamp }}"
        }
      },
      "id": "store-memory",
      "name": "Store Memory",
      "type": "n8n-nodes-base.postgres",
      "position": [900, 200]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT content, metadata, created_at FROM conversation_memory WHERE conversation_id = $1 AND agent_id = $2 ORDER BY created_at DESC LIMIT $3",
        "additionalFields": {
          "queryParameters": "{{ $json.conversation_id }},{{ $json.agent_id }},{{ $json.limit || 10 }}"
        }
      },
      "id": "retrieve-memory",
      "name": "Retrieve Memory",
      "type": "n8n-nodes-base.postgres",
      "position": [680, 300]
    },
    {
      "parameters": {
        "url": "http://embedding-service:8000/embed",
        "options": {
          "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer {{ $env.OPENAI_API_KEY }}"
          }
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "text",
              "value": "={{ $json.query }}"
            },
            {
              "name": "model",
              "value": "text-embedding-3-small"
            }
          ]
        }
      },
      "id": "generate-search-embedding",
      "name": "Generate Search Embedding",
      "type": "n8n-nodes-base.httpRequest",
      "position": [680, 400]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT content, metadata, created_at, (embedding <=> $1) as similarity FROM conversation_memory WHERE conversation_id = $2 ORDER BY similarity ASC LIMIT $3",
        "additionalFields": {
          "queryParameters": "{{ $('Generate Search Embedding').first().json.embedding }},{{ $json.conversation_id }},{{ $json.limit || 5 }}"
        }
      },
      "id": "semantic-search",
      "name": "Semantic Search",
      "type": "n8n-nodes-base.postgres",
      "position": [900, 400]
    },
    {
      "parameters": {
        "jsCode": "const memories = $input.all();\nconst operation = $('Memory Operation Webhook').first().json.operation;\n\nlet response = {};\n\nif (operation === 'store') {\n  response = {\n    success: true,\n    memory_id: memories[0].json.id,\n    message: 'Memory stored successfully'\n  };\n} else if (operation === 'retrieve') {\n  response = {\n    success: true,\n    memories: memories.map(m => ({\n      content: m.json.content,\n      metadata: m.json.metadata,\n      created_at: m.json.created_at\n    })),\n    count: memories.length\n  };\n} else if (operation === 'search') {\n  response = {\n    success: true,\n    results: memories.map(m => ({\n      content: m.json.content,\n      metadata: m.json.metadata,\n      created_at: m.json.created_at,\n      similarity_score: 1 - m.json.similarity\n    })),\n    count: memories.length\n  };\n}\n\nreturn [{ json: response }];"
      },
      "id": "format-response",
      "name": "Format Response",
      "type": "n8n-nodes-base.code",
      "position": [1120, 300]
    }
  ],
  "connections": {
    "Memory Operation Webhook": {
      "main": [
        [
          {
            "node": "Operation Router",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Operation Router": {
      "main": [
        [
          {
            "node": "Generate Embedding",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Retrieve Memory",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Generate Search Embedding",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Generate Embedding": {
      "main": [
        [
          {
            "node": "Store Memory",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Store Memory": {
      "main": [
        [
          {
            "node": "Format Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Retrieve Memory": {
      "main": [
        [
          {
            "node": "Format Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Generate Search Embedding": {
      "main": [
        [
          {
            "node": "Semantic Search",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Semantic Search": {
      "main": [
        [
          {
            "node": "Format Response",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

## 4. Scalability Considerations

### 4.1 Horizontal Scaling Patterns

#### n8n Instance Distribution
```yaml
# docker-compose.yml for horizontal scaling
version: '3.8'
services:
  n8n-coordinator:
    image: n8nio/n8n:latest
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - WEBHOOK_URL=${WEBHOOK_BASE_URL}
      - N8N_DATABASE_TYPE=postgresdb
      - N8N_DATABASE_HOST=postgres
      - N8N_DATABASE_PORT=5432
      - N8N_DATABASE_DB=n8n_coordinator
      - N8N_DATABASE_USER=${POSTGRES_USER}
      - N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "5678:5678"
    volumes:
      - n8n_coordinator_data:/home/node/.n8n
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 1

  n8n-worker-conversation:
    image: n8nio/n8n:latest
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - WEBHOOK_URL=${WEBHOOK_BASE_URL}
      - N8N_DATABASE_TYPE=postgresdb
      - N8N_DATABASE_HOST=postgres
      - N8N_DATABASE_PORT=5432
      - N8N_DATABASE_DB=n8n_conversation
      - N8N_DATABASE_USER=${POSTGRES_USER}
      - N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}
      - EXECUTIONS_MODE=queue
      - QUEUE_BULL_REDIS_HOST=redis
    ports:
      - "5679:5678"
    volumes:
      - n8n_conversation_data:/home/node/.n8n
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 3

  n8n-worker-audio:
    image: n8nio/n8n:latest
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - WEBHOOK_URL=${WEBHOOK_BASE_URL}
      - N8N_DATABASE_TYPE=postgresdb
      - N8N_DATABASE_HOST=postgres
      - N8N_DATABASE_PORT=5432
      - N8N_DATABASE_DB=n8n_audio
      - N8N_DATABASE_USER=${POSTGRES_USER}
      - N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}
      - EXECUTIONS_MODE=queue
      - QUEUE_BULL_REDIS_HOST=redis
    ports:
      - "5680:5678"
    volumes:
      - n8n_audio_data:/home/node/.n8n
    depends_on:
      - postgres
      - redis
    deploy:
      replicas: 2

  nginx-load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - n8n-coordinator
      - n8n-worker-conversation
      - n8n-worker-audio
```

#### Load Balancer Configuration
```nginx
# nginx.conf
upstream n8n_coordinator {
    server n8n-coordinator:5678;
}

upstream n8n_conversation_workers {
    least_conn;
    server n8n-worker-conversation-1:5678;
    server n8n-worker-conversation-2:5678;
    server n8n-worker-conversation-3:5678;
}

upstream n8n_audio_workers {
    least_conn;
    server n8n-worker-audio-1:5678;
    server n8n-worker-audio-2:5678;
}

server {
    listen 80;
    server_name conversation-engine.local;

    # Route coordination requests
    location /webhook/coordinate/ {
        proxy_pass http://n8n_coordinator;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Route conversation processing
    location /webhook/agent/ {
        proxy_pass http://n8n_conversation_workers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Route audio processing
    location /webhook/audio/ {
        proxy_pass http://n8n_audio_workers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Health checks
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

### 4.2 Queue Management for Concurrent Conversations

#### Redis Queue Configuration
```yaml
# redis-queues.conf
# Conversation processing queue
queue:conversations:
  max_jobs: 100
  concurrency: 5
  retry_attempts: 3
  retry_delay: 5000  # 5 seconds

# Audio processing queue
queue:audio:
  max_jobs: 50
  concurrency: 3
  retry_attempts: 2
  retry_delay: 2000  # 2 seconds

# Priority queue for urgent tasks
queue:priority:
  max_jobs: 20
  concurrency: 10
  retry_attempts: 1
  retry_delay: 1000  # 1 second
```

#### Queue Management Workflow
```json
{
  "name": "Queue Management System",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "queue/manage",
        "responseMode": "responseNode"
      },
      "id": "queue-webhook",
      "name": "Queue Management Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "jsCode": "const request = $input.first().json;\nconst queueType = request.queue_type || 'conversations';\nconst priority = request.priority || 1;\nconst maxRetries = request.max_retries || 3;\n\n// Calculate queue assignment based on load\nconst queueKey = `queue:${queueType}:${priority}`;\nconst jobId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;\n\nreturn [{\n  json: {\n    queue_key: queueKey,\n    job_id: jobId,\n    payload: request.payload,\n    priority: priority,\n    max_retries: maxRetries,\n    created_at: new Date().toISOString()\n  }\n}];"
      },
      "id": "queue-assignment",
      "name": "Queue Assignment Logic",
      "type": "n8n-nodes-base.code",
      "position": [460, 300]
    },
    {
      "parameters": {
        "operation": "lpush",
        "key": "={{ $json.queue_key }}",
        "value": "={{ JSON.stringify($json) }}"
      },
      "id": "enqueue-job",
      "name": "Enqueue Job",
      "type": "n8n-nodes-base.redis",
      "position": [680, 300]
    },
    {
      "parameters": {
        "operation": "set",
        "key": "job:{{ $json.job_id }}:status",
        "value": "queued",
        "options": {
          "ttl": 3600
        }
      },
      "id": "set-job-status",
      "name": "Set Job Status",
      "type": "n8n-nodes-base.redis",
      "position": [900, 300]
    }
  ]
}
```

### 4.3 Resource Optimization Strategies

#### Memory Management
```yaml
# n8n environment variables for optimization
environment:
  - N8N_DEFAULT_BINARY_DATA_MODE=filesystem
  - N8N_BINARY_DATA_TTL=60  # 1 hour
  - N8N_BINARY_DATA_STORAGE_PATH=/tmp/n8n-binary-data
  - EXECUTIONS_DATA_PRUNE=true
  - EXECUTIONS_DATA_MAX_AGE=168  # 1 week
  - N8N_LOG_LEVEL=info
  - N8N_LOG_OUTPUT=file
  - N8N_METRICS=true
  - N8N_DIAGNOSTICS_ENABLED=false
```

#### Database Connection Pooling
```javascript
// PostgreSQL connection optimization
const dbConfig = {
  host: process.env.DB_HOST,
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: 20,  // Maximum pool size
  min: 5,   // Minimum pool size
  idle: 10000,  // Idle timeout (10 seconds)
  acquire: 30000,  // Acquire timeout (30 seconds)
  evict: 1000,  // Evict timeout (1 second)
  handleDisconnects: true,
  charset: 'utf8mb4',
  logging: process.env.NODE_ENV === 'development' ? console.log : false
};
```

#### Caching Strategy
```json
{
  "name": "Intelligent Caching System",
  "nodes": [
    {
      "parameters": {
        "operation": "get",
        "key": "cache:{{ $json.cache_key }}"
      },
      "id": "check-cache",
      "name": "Check Cache",
      "type": "n8n-nodes-base.redis",
      "position": [240, 300]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "leftValue": "={{ $json.value }}",
              "operation": "isEmpty"
            }
          ]
        }
      },
      "id": "cache-miss-check",
      "name": "Cache Miss Check",
      "type": "n8n-nodes-base.if",
      "position": [460, 300]
    },
    {
      "parameters": {
        "operation": "setex",
        "key": "cache:{{ $('Original Request').first().json.cache_key }}",
        "value": "={{ JSON.stringify($json) }}",
        "ttl": "={{ $('Original Request').first().json.cache_ttl || 300 }}"
      },
      "id": "store-in-cache",
      "name": "Store in Cache",
      "type": "n8n-nodes-base.redis",
      "position": [900, 400]
    }
  ]
}
```

### 4.4 Performance Monitoring Integration

#### Prometheus Metrics Collection
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
```

#### Custom Metrics Workflow
```json
{
  "name": "Performance Metrics Collection",
  "nodes": [
    {
      "parameters": {
        "triggerTimes": {
          "mode": "everyMinute"
        }
      },
      "id": "metrics-trigger",
      "name": "Metrics Collection Trigger",
      "type": "n8n-nodes-base.cron",
      "position": [240, 300]
    },
    {
      "parameters": {
        "operation": "llen",
        "key": "queue:conversations"
      },
      "id": "conversation-queue-length",
      "name": "Get Conversation Queue Length",
      "type": "n8n-nodes-base.redis",
      "position": [460, 200]
    },
    {
      "parameters": {
        "operation": "llen",
        "key": "queue:audio"
      },
      "id": "audio-queue-length",
      "name": "Get Audio Queue Length",
      "type": "n8n-nodes-base.redis",
      "position": [460, 300]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT COUNT(*) as active_conversations FROM conversations WHERE updated_at > NOW() - INTERVAL '5 minutes'"
      },
      "id": "active-conversations-count",
      "name": "Get Active Conversations",
      "type": "n8n-nodes-base.postgres",
      "position": [460, 400]
    },
    {
      "parameters": {
        "jsCode": "const conversationQueue = $('Get Conversation Queue Length').first().json.value;\nconst audioQueue = $('Get Audio Queue Length').first().json.value;\nconst activeConversations = $('Get Active Conversations').first().json.count;\nconst timestamp = Date.now();\n\nconst metrics = {\n  conversation_queue_length: conversationQueue,\n  audio_queue_length: audioQueue,\n  active_conversations: activeConversations,\n  timestamp: timestamp\n};\n\nreturn [{ json: metrics }];"
      },
      "id": "format-metrics",
      "name": "Format Metrics",
      "type": "n8n-nodes-base.code",
      "position": [680, 300]
    },
    {
      "parameters": {
        "url": "http://prometheus-pushgateway:9091/metrics/job/n8n-conversation-engine",
        "options": {
          "headers": {
            "Content-Type": "text/plain"
          }
        },
        "sendBody": true,
        "body": "conversation_queue_length {{ $json.conversation_queue_length }}\naudio_queue_length {{ $json.audio_queue_length }}\nactive_conversations {{ $json.active_conversations }}"
      },
      "id": "push-to-prometheus",
      "name": "Push to Prometheus",
      "type": "n8n-nodes-base.httpRequest",
      "position": [900, 300]
    }
  ]
}
```

## 5. Concrete Implementation

### 5.1 Environment Variable Management

#### Core Configuration
```bash
# .env file
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=conversation_engine
POSTGRES_USER=n8n_user
POSTGRES_PASSWORD=secure_password_123

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password_123

# n8n Configuration
N8N_BASIC_AUTH_ACTIVE=true
N8N_BASIC_AUTH_USER=admin
N8N_BASIC_AUTH_PASSWORD=n8n_admin_password
N8N_DATABASE_TYPE=postgresdb
N8N_DATABASE_HOST=${POSTGRES_HOST}
N8N_DATABASE_PORT=${POSTGRES_PORT}
N8N_DATABASE_DB=${POSTGRES_DB}
N8N_DATABASE_USER=${POSTGRES_USER}
N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}

# External Services
OPENAI_API_KEY=sk-your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
HA_TOKEN=your-home-assistant-long-lived-token

# Audio Services
WHISPER_SERVICE_URL=http://localhost:8000
ELEVENLABS_SERVICE_URL=https://api.elevenlabs.io

# Webhook Configuration
WEBHOOK_BASE_URL=https://your-domain.com
WEBHOOK_SECRET=your-webhook-secret-key

# Security
JWT_SECRET=your-jwt-secret-key
ENCRYPTION_KEY=your-32-byte-encryption-key

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_PASSWORD=grafana_admin_password
```

#### Docker Environment Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    environment:
      - N8N_BASIC_AUTH_ACTIVE=${N8N_BASIC_AUTH_ACTIVE}
      - N8N_BASIC_AUTH_USER=${N8N_BASIC_AUTH_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_BASIC_AUTH_PASSWORD}
      - WEBHOOK_URL=${WEBHOOK_BASE_URL}
      - N8N_DATABASE_TYPE=${N8N_DATABASE_TYPE}
      - N8N_DATABASE_HOST=${POSTGRES_HOST}
      - N8N_DATABASE_PORT=${POSTGRES_PORT}
      - N8N_DATABASE_DB=${POSTGRES_DB}
      - N8N_DATABASE_USER=${POSTGRES_USER}
      - N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
      - HA_TOKEN=${HA_TOKEN}
    ports:
      - "5678:5678"
    volumes:
      - n8n_data:/home/node/.n8n
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "${REDIS_PORT}:6379"
    volumes:
      - redis_data:/data

volumes:
  n8n_data:
  postgres_data:
  redis_data:
```

### 5.2 Security and Authentication Patterns

#### JWT Authentication Workflow
```json
{
  "name": "JWT Authentication System",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "auth/login",
        "responseMode": "responseNode"
      },
      "id": "login-webhook",
      "name": "Login Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "jsCode": "const { username, password } = $input.first().json;\nconst crypto = require('crypto');\n\n// Hash password with salt\nconst salt = process.env.PASSWORD_SALT;\nconst hashedPassword = crypto.createHash('sha256').update(password + salt).digest('hex');\n\nreturn [{\n  json: {\n    username,\n    hashed_password: hashedPassword\n  }\n}];"
      },
      "id": "hash-password",
      "name": "Hash Password",
      "type": "n8n-nodes-base.code",
      "position": [460, 300]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "SELECT id, username, role FROM users WHERE username = $1 AND password_hash = $2 AND active = true",
        "additionalFields": {
          "queryParameters": "{{ $json.username }},{{ $json.hashed_password }}"
        }
      },
      "id": "validate-credentials",
      "name": "Validate Credentials",
      "type": "n8n-nodes-base.postgres",
      "position": [680, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "leftValue": "={{ $json.length }}",
              "operation": "equal",
              "rightValue": 1
            }
          ]
        }
      },
      "id": "auth-check",
      "name": "Authentication Check",
      "type": "n8n-nodes-base.if",
      "position": [900, 300]
    },
    {
      "parameters": {
        "jsCode": "const jwt = require('jsonwebtoken');\nconst user = $input.first().json;\nconst secret = process.env.JWT_SECRET;\n\nconst payload = {\n  user_id: user.id,\n  username: user.username,\n  role: user.role,\n  iat: Math.floor(Date.now() / 1000),\n  exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60)  // 24 hours\n};\n\nconst token = jwt.sign(payload, secret);\n\nreturn [{\n  json: {\n    success: true,\n    token: token,\n    user: {\n      id: user.id,\n      username: user.username,\n      role: user.role\n    }\n  }\n}];"
      },
      "id": "generate-jwt",
      "name": "Generate JWT Token",
      "type": "n8n-nodes-base.code",
      "position": [1120, 200]
    }
  ]
}
```

#### API Security Middleware
```json
{
  "name": "API Security Middleware",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "secure/*",
        "responseMode": "responseNode"
      },
      "id": "secure-endpoint",
      "name": "Secure Endpoint",
      "type": "n8n-nodes-base.webhook",
      "position": [240, 300]
    },
    {
      "parameters": {
        "jsCode": "const authHeader = $input.first().json.headers.authorization;\nconst jwt = require('jsonwebtoken');\n\nif (!authHeader || !authHeader.startsWith('Bearer ')) {\n  return [{\n    json: {\n      error: 'Missing or invalid authorization header',\n      status: 401\n    }\n  }];\n}\n\nconst token = authHeader.substring(7);\nconst secret = process.env.JWT_SECRET;\n\ntry {\n  const decoded = jwt.verify(token, secret);\n  return [{\n    json: {\n      valid: true,\n      user: decoded,\n      original_request: $input.first().json\n    }\n  }];\n} catch (error) {\n  return [{\n    json: {\n      error: 'Invalid or expired token',\n      status: 401\n    }\n  }];\n}"
      },
      "id": "verify-jwt",
      "name": "Verify JWT Token",
      "type": "n8n-nodes-base.code",
      "position": [460, 300]
    },
    {
      "parameters": {
        "conditions": {
          "boolean": [
            {
              "leftValue": "={{ $json.valid }}",
              "operation": "equal",
              "rightValue": true
            }
          ]
        }
      },
      "id": "token-validation",
      "name": "Token Validation",
      "type": "n8n-nodes-base.if",
      "position": [680, 300]
    }
  ]
}
```

### 5.3 Complete Deployment Configuration

#### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - n8n-coordinator
      - n8n-worker
    restart: unless-stopped

  n8n-coordinator:
    image: n8nio/n8n:latest
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_BASIC_AUTH_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_BASIC_AUTH_PASSWORD}
      - WEBHOOK_URL=${WEBHOOK_BASE_URL}
      - N8N_DATABASE_TYPE=postgresdb
      - N8N_DATABASE_HOST=postgres
      - N8N_DATABASE_PORT=5432
      - N8N_DATABASE_DB=${POSTGRES_DB}
      - N8N_DATABASE_USER=${POSTGRES_USER}
      - N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}
      - EXECUTIONS_MODE=queue
      - QUEUE_BULL_REDIS_HOST=redis
      - QUEUE_BULL_REDIS_PASSWORD=${REDIS_PASSWORD}
      - N8N_METRICS=true
      - N8N_LOG_LEVEL=info
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
      - HA_TOKEN=${HA_TOKEN}
    volumes:
      - n8n_coordinator_data:/home/node/.n8n
      - ./workflows:/home/node/.n8n/workflows
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  n8n-worker:
    image: n8nio/n8n:latest
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=${N8N_BASIC_AUTH_USER}
      - N8N_BASIC_AUTH_PASSWORD=${N8N_BASIC_AUTH_PASSWORD}
      - WEBHOOK_URL=${WEBHOOK_BASE_URL}
      - N8N_DATABASE_TYPE=postgresdb
      - N8N_DATABASE_HOST=postgres
      - N8N_DATABASE_PORT=5432
      - N8N_DATABASE_DB=${POSTGRES_DB}
      - N8N_DATABASE_USER=${POSTGRES_USER}
      - N8N_DATABASE_PASSWORD=${POSTGRES_PASSWORD}
      - EXECUTIONS_MODE=queue
      - QUEUE_BULL_REDIS_HOST=redis
      - QUEUE_BULL_REDIS_PASSWORD=${REDIS_PASSWORD}
      - N8N_LOG_LEVEL=info
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY}
      - HA_TOKEN=${HA_TOKEN}
    volumes:
      - n8n_worker_data:/home/node/.n8n
      - ./workflows:/home/node/.n8n/workflows
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1.5G
          cpus: '0.8'
        reservations:
          memory: 512M
          cpus: '0.3'

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/01-init.sql
      - ./sql/extensions.sql:/docker-entrypoint-initdb.d/02-extensions.sql
      - ./backups:/backups
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  redis:
    image: redis:7-alpine
    command: >
      sh -c "redis-server
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru"
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1.5G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.2'

  whisper-service:
    build:
      context: ./services/whisper
      dockerfile: Dockerfile
    environment:
      - MODEL_SIZE=large-v3
      - CUDA_VISIBLE_DEVICES=0
      - BATCH_SIZE=16
    ports:
      - "8000:8000"
    volumes:
      - whisper_models:/app/models
      - audio_temp:/tmp/audio
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '2.0'
        reservations:
          memory: 4G
          cpus: '1.0'

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  n8n_coordinator_data:
  n8n_worker_data:
  postgres_data:
  redis_data:
  whisper_models:
  audio_temp:
  prometheus_data:
  grafana_data:
```

#### Health Check and Monitoring
```bash
#!/bin/bash
# health-check.sh

# Check n8n coordinator health
if ! curl -f http://localhost:5678/healthz > /dev/null 2>&1; then
    echo "n8n coordinator is down"
    exit 1
fi

# Check PostgreSQL health
if ! pg_isready -h localhost -p 5432 -U ${POSTGRES_USER} > /dev/null 2>&1; then
    echo "PostgreSQL is down"
    exit 1
fi

# Check Redis health
if ! redis-cli -h localhost -p 6379 -a ${REDIS_PASSWORD} ping > /dev/null 2>&1; then
    echo "Redis is down"
    exit 1
fi

# Check queue lengths
CONVERSATION_QUEUE=$(redis-cli -h localhost -p 6379 -a ${REDIS_PASSWORD} llen queue:conversations)
AUDIO_QUEUE=$(redis-cli -h localhost -p 6379 -a ${REDIS_PASSWORD} llen queue:audio)

if [ "$CONVERSATION_QUEUE" -gt 100 ]; then
    echo "Warning: Conversation queue is backing up ($CONVERSATION_QUEUE items)"
fi

if [ "$AUDIO_QUEUE" -gt 50 ]; then
    echo "Warning: Audio queue is backing up ($AUDIO_QUEUE items)"
fi

echo "All services healthy"
exit 0
```

## Implementation Benefits & Migration Path

### Benefits of n8n Architecture
1. **Scalability**: Horizontal scaling with independent workers
2. **Reliability**: Individual component failures don't crash system
3. **Maintainability**: Visual workflow management and debugging
4. **Flexibility**: Easy integration with new services and APIs
5. **Monitoring**: Built-in metrics and performance tracking
6. **Cost Efficiency**: Resource optimization and queue management

### Migration Strategy
1. **Phase 1**: Set up n8n infrastructure and basic workflows
2. **Phase 2**: Migrate conversation orchestration workflows
3. **Phase 3**: Implement audio processing pipelines
4. **Phase 4**: Add Home Assistant integration
5. **Phase 5**: Optimize performance and implement monitoring
6. **Phase 6**: Decommission threading-based system

This n8n-based architecture provides a robust, scalable foundation for multi-agent conversation engines while maintaining the interactive and engaging experience of the original system.
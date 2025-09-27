// MongoDB Initialization Script for Multi-Agent Conversation Engine

// Switch to the multi_agent_conversations database
db = db.getSiblingDB('multi_agent_conversations');

// Create application user
db.createUser({
  user: 'conversation_app',
  pwd: 'conv_app_pass_2024',
  roles: [
    {
      role: 'readWrite',
      db: 'multi_agent_conversations'
    }
  ]
});

// Create collections with validation schemas
db.createCollection('conversations', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['session_id', 'created_at', 'participants', 'messages'],
      properties: {
        session_id: {
          bsonType: 'string',
          description: 'Unique session identifier'
        },
        created_at: {
          bsonType: 'date',
          description: 'Conversation creation timestamp'
        },
        updated_at: {
          bsonType: 'date',
          description: 'Last update timestamp'
        },
        status: {
          bsonType: 'string',
          enum: ['active', 'paused', 'completed', 'archived'],
          description: 'Conversation status'
        },
        participants: {
          bsonType: 'array',
          description: 'List of conversation participants',
          items: {
            bsonType: 'object',
            required: ['participant_id', 'type', 'name'],
            properties: {
              participant_id: { bsonType: 'string' },
              type: {
                bsonType: 'string',
                enum: ['human', 'agent']
              },
              name: { bsonType: 'string' }
            }
          }
        },
        messages: {
          bsonType: 'array',
          description: 'Conversation messages',
          items: {
            bsonType: 'object',
            required: ['message_id', 'timestamp', 'speaker_id', 'content'],
            properties: {
              message_id: { bsonType: 'string' },
              timestamp: { bsonType: 'date' },
              speaker_id: { bsonType: 'string' },
              speaker_type: {
                bsonType: 'string',
                enum: ['human', 'agent']
              },
              content: {
                bsonType: 'object',
                required: ['text'],
                properties: {
                  text: { bsonType: 'string' }
                }
              }
            }
          }
        }
      }
    }
  }
});

// Create collection for long-term memory
db.createCollection('long_term_memory', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['session_id', 'content', 'timestamp', 'importance_score'],
      properties: {
        session_id: {
          bsonType: 'string',
          description: 'Session identifier'
        },
        content: {
          bsonType: 'string',
          description: 'Memory content'
        },
        timestamp: {
          bsonType: 'date',
          description: 'Memory creation timestamp'
        },
        importance_score: {
          bsonType: 'double',
          minimum: 0.0,
          maximum: 1.0,
          description: 'Memory importance score'
        },
        memory_type: {
          bsonType: 'string',
          enum: ['episodic', 'semantic', 'procedural', 'shared_episodic'],
          description: 'Type of memory'
        },
        agent_id: {
          bsonType: 'string',
          description: 'Agent associated with memory'
        },
        embedding: {
          bsonType: 'array',
          description: 'Vector embedding for similarity search'
        }
      }
    }
  }
});

// Create collection for character development
db.createCollection('character_development', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['character_id', 'session_id', 'personality_state', 'timestamp'],
      properties: {
        character_id: {
          bsonType: 'string',
          description: 'Character identifier'
        },
        session_id: {
          bsonType: 'string',
          description: 'Session identifier'
        },
        personality_state: {
          bsonType: 'object',
          description: 'Current personality state'
        },
        timestamp: {
          bsonType: 'date',
          description: 'State snapshot timestamp'
        }
      }
    }
  }
});

// Create indexes for performance optimization
db.conversations.createIndex({ 'session_id': 1 }, { unique: true });
db.conversations.createIndex({ 'created_at': -1 });
db.conversations.createIndex({ 'status': 1 });
db.conversations.createIndex({ 'participants.participant_id': 1 });
db.conversations.createIndex({ 'messages.timestamp': -1 });
db.conversations.createIndex({ 'metadata.topic_category': 1 });

// Long-term memory indexes
db.long_term_memory.createIndex({ 'session_id': 1 });
db.long_term_memory.createIndex({ 'timestamp': -1 });
db.long_term_memory.createIndex({ 'importance_score': -1 });
db.long_term_memory.createIndex({ 'memory_type': 1 });
db.long_term_memory.createIndex({ 'agent_id': 1 });

// Vector search index for memory embeddings (if using MongoDB Atlas)
// db.long_term_memory.createIndex({
//   'embedding': '2dsphere'
// });

// Character development indexes
db.character_development.createIndex({ 'character_id': 1, 'session_id': 1 });
db.character_development.createIndex({ 'timestamp': -1 });

// Create capped collection for real-time events
db.createCollection('real_time_events', {
  capped: true,
  size: 100000000, // 100MB
  max: 1000000     // 1M documents
});

db.real_time_events.createIndex({ 'timestamp': 1 });
db.real_time_events.createIndex({ 'event_type': 1 });

print('MongoDB initialization completed successfully');
print('Collections created: conversations, long_term_memory, character_development, real_time_events');
print('Indexes created for performance optimization');
print('Application user created: conversation_app');
import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import io from 'socket.io-client';
import { toast } from 'react-toastify';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [messages, setMessages] = useState([]);
  const socketRef = useRef(null);

  // Real-time data states
  const [agentStatuses, setAgentStatuses] = useState({});
  const [activeConflicts, setActiveConflicts] = useState([]);
  const [runningWorkflows, setRunningWorkflows] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState({});

  useEffect(() => {
    // Initialize WebSocket connection
    const initializeSocket = () => {
      const newSocket = io('http://localhost:8000', {
        transports: ['websocket'],
        upgrade: false,
        rememberUpgrade: false,
      });

      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setConnected(true);
        setReconnecting(false);
        toast.success('Connected to Multi-Agent System');
      });

      newSocket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        toast.error('Disconnected from Multi-Agent System');
      });

      newSocket.on('reconnect_attempt', () => {
        setReconnecting(true);
        toast.info('Reconnecting to Multi-Agent System...');
      });

      newSocket.on('reconnect', () => {
        setReconnecting(false);
        toast.success('Reconnected to Multi-Agent System');
      });

      // Agent-related events
      newSocket.on('agent_status_update', (data) => {
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent_id]: {
            ...prev[data.agent_id],
            ...data
          }
        }));
      });

      newSocket.on('agent_registered', (data) => {
        setAgentStatuses(prev => ({
          ...prev,
          [data.agent_id]: {
            ...data,
            status: 'available',
            last_updated: new Date().toISOString()
          }
        }));
        toast.success(`Agent ${data.agent_id} registered`);
      });

      newSocket.on('agent_unregistered', (data) => {
        setAgentStatuses(prev => {
          const newStatuses = { ...prev };
          delete newStatuses[data.agent_id];
          return newStatuses;
        });
        toast.info(`Agent ${data.agent_id} unregistered`);
      });

      // Conflict-related events
      newSocket.on('conflict_detected', (data) => {
        setActiveConflicts(prev => [...prev, data]);
        toast.warning(`Conflict detected: ${data.conflict_type}`);
      });

      newSocket.on('conflict_resolved', (data) => {
        setActiveConflicts(prev => prev.filter(c => c.conflict_id !== data.conflict_id));
        toast.success(`Conflict resolved: ${data.resolution_strategy}`);
      });

      newSocket.on('conflict_escalated', (data) => {
        toast.error(`Conflict escalated: ${data.reason}`);
      });

      // Workflow-related events
      newSocket.on('workflow_started', (data) => {
        setRunningWorkflows(prev => [...prev, data]);
        toast.info(`Workflow started: ${data.title}`);
      });

      newSocket.on('workflow_completed', (data) => {
        setRunningWorkflows(prev => prev.filter(w => w.workflow_id !== data.workflow_id));
        toast.success(`Workflow completed: ${data.title}`);
      });

      newSocket.on('workflow_failed', (data) => {
        setRunningWorkflows(prev => prev.filter(w => w.workflow_id !== data.workflow_id));
        toast.error(`Workflow failed: ${data.title}`);
      });

      newSocket.on('task_completed', (data) => {
        setRunningWorkflows(prev => prev.map(w =>
          w.workflow_id === data.workflow_id
            ? { ...w, completion_percentage: data.completion_percentage }
            : w
        ));
      });

      // System metrics events
      newSocket.on('system_metrics_update', (data) => {
        setSystemMetrics(data);
      });

      // Communication events
      newSocket.on('agent_message', (data) => {
        setMessages(prev => [...prev, {
          ...data,
          timestamp: new Date().toISOString(),
          id: Date.now()
        }]);
      });

      // Error handling
      newSocket.on('error', (error) => {
        console.error('WebSocket error:', error);
        toast.error(`WebSocket error: ${error.message}`);
      });

      socketRef.current = newSocket;
      setSocket(newSocket);
    };

    initializeSocket();

    // Cleanup on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  // API functions for sending events
  const sendMessage = (event, data) => {
    if (socket && connected) {
      socket.emit(event, data);
    } else {
      toast.error('Not connected to WebSocket');
    }
  };

  const joinRoom = (roomName) => {
    if (socket && connected) {
      socket.emit('join_room', { room: roomName });
    }
  };

  const leaveRoom = (roomName) => {
    if (socket && connected) {
      socket.emit('leave_room', { room: roomName });
    }
  };

  const subscribeToMetrics = () => {
    if (socket && connected) {
      socket.emit('subscribe_metrics');
    }
  };

  const unsubscribeFromMetrics = () => {
    if (socket && connected) {
      socket.emit('unsubscribe_metrics');
    }
  };

  const requestAgentStatus = () => {
    if (socket && connected) {
      socket.emit('request_agent_status');
    }
  };

  const requestActiveConflicts = () => {
    if (socket && connected) {
      socket.emit('request_active_conflicts');
    }
  };

  const requestRunningWorkflows = () => {
    if (socket && connected) {
      socket.emit('request_running_workflows');
    }
  };

  const clearMessages = () => {
    setMessages([]);
  };

  const value = {
    socket,
    connected,
    reconnecting,
    messages,
    agentStatuses,
    activeConflicts,
    runningWorkflows,
    systemMetrics,

    // Actions
    sendMessage,
    joinRoom,
    leaveRoom,
    subscribeToMetrics,
    unsubscribeFromMetrics,
    requestAgentStatus,
    requestActiveConflicts,
    requestRunningWorkflows,
    clearMessages,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
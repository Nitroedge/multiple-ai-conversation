import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { useWebSocket } from './WebSocketContext';

const AgentContext = createContext();

export const useAgent = () => {
  const context = useContext(AgentContext);
  if (!context) {
    throw new Error('useAgent must be used within an AgentProvider');
  }
  return context;
};

// Initial state
const initialState = {
  agents: {},
  activeConversations: {},
  workflows: {},
  conflicts: {},
  systemMetrics: {
    total_agents: 0,
    active_agents: 0,
    total_conflicts: 0,
    resolved_conflicts: 0,
    active_workflows: 0,
    completed_workflows: 0,
  },
  loading: {
    agents: false,
    workflows: false,
    conflicts: false,
    metrics: false,
  },
  errors: {},
};

// Action types
const ActionTypes = {
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',

  // Agents
  SET_AGENTS: 'SET_AGENTS',
  UPDATE_AGENT: 'UPDATE_AGENT',
  REMOVE_AGENT: 'REMOVE_AGENT',

  // Workflows
  SET_WORKFLOWS: 'SET_WORKFLOWS',
  ADD_WORKFLOW: 'ADD_WORKFLOW',
  UPDATE_WORKFLOW: 'UPDATE_WORKFLOW',
  REMOVE_WORKFLOW: 'REMOVE_WORKFLOW',

  // Conflicts
  SET_CONFLICTS: 'SET_CONFLICTS',
  ADD_CONFLICT: 'ADD_CONFLICT',
  UPDATE_CONFLICT: 'UPDATE_CONFLICT',
  REMOVE_CONFLICT: 'REMOVE_CONFLICT',

  // Conversations
  SET_CONVERSATIONS: 'SET_CONVERSATIONS',
  UPDATE_CONVERSATION: 'UPDATE_CONVERSATION',

  // Metrics
  SET_METRICS: 'SET_METRICS',
};

// Reducer
const agentReducer = (state, action) => {
  switch (action.type) {
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        loading: {
          ...state.loading,
          [action.payload.key]: action.payload.value,
        },
      };

    case ActionTypes.SET_ERROR:
      return {
        ...state,
        errors: {
          ...state.errors,
          [action.payload.key]: action.payload.error,
        },
      };

    case ActionTypes.CLEAR_ERROR:
      const newErrors = { ...state.errors };
      delete newErrors[action.payload.key];
      return {
        ...state,
        errors: newErrors,
      };

    case ActionTypes.SET_AGENTS:
      return {
        ...state,
        agents: action.payload,
      };

    case ActionTypes.UPDATE_AGENT:
      return {
        ...state,
        agents: {
          ...state.agents,
          [action.payload.agent_id]: {
            ...state.agents[action.payload.agent_id],
            ...action.payload,
          },
        },
      };

    case ActionTypes.REMOVE_AGENT:
      const newAgents = { ...state.agents };
      delete newAgents[action.payload.agent_id];
      return {
        ...state,
        agents: newAgents,
      };

    case ActionTypes.SET_WORKFLOWS:
      return {
        ...state,
        workflows: action.payload,
      };

    case ActionTypes.ADD_WORKFLOW:
      return {
        ...state,
        workflows: {
          ...state.workflows,
          [action.payload.workflow_id]: action.payload,
        },
      };

    case ActionTypes.UPDATE_WORKFLOW:
      return {
        ...state,
        workflows: {
          ...state.workflows,
          [action.payload.workflow_id]: {
            ...state.workflows[action.payload.workflow_id],
            ...action.payload,
          },
        },
      };

    case ActionTypes.REMOVE_WORKFLOW:
      const newWorkflows = { ...state.workflows };
      delete newWorkflows[action.payload.workflow_id];
      return {
        ...state,
        workflows: newWorkflows,
      };

    case ActionTypes.SET_CONFLICTS:
      return {
        ...state,
        conflicts: action.payload,
      };

    case ActionTypes.ADD_CONFLICT:
      return {
        ...state,
        conflicts: {
          ...state.conflicts,
          [action.payload.conflict_id]: action.payload,
        },
      };

    case ActionTypes.UPDATE_CONFLICT:
      return {
        ...state,
        conflicts: {
          ...state.conflicts,
          [action.payload.conflict_id]: {
            ...state.conflicts[action.payload.conflict_id],
            ...action.payload,
          },
        },
      };

    case ActionTypes.REMOVE_CONFLICT:
      const newConflicts = { ...state.conflicts };
      delete newConflicts[action.payload.conflict_id];
      return {
        ...state,
        conflicts: newConflicts,
      };

    case ActionTypes.SET_CONVERSATIONS:
      return {
        ...state,
        activeConversations: action.payload,
      };

    case ActionTypes.UPDATE_CONVERSATION:
      return {
        ...state,
        activeConversations: {
          ...state.activeConversations,
          [action.payload.session_id]: {
            ...state.activeConversations[action.payload.session_id],
            ...action.payload,
          },
        },
      };

    case ActionTypes.SET_METRICS:
      return {
        ...state,
        systemMetrics: {
          ...state.systemMetrics,
          ...action.payload,
        },
      };

    default:
      return state;
  }
};

export const AgentProvider = ({ children }) => {
  const [state, dispatch] = useReducer(agentReducer, initialState);
  const { agentStatuses, activeConflicts, runningWorkflows, systemMetrics } = useWebSocket();

  // Sync with WebSocket updates
  useEffect(() => {
    if (agentStatuses) {
      dispatch({ type: ActionTypes.SET_AGENTS, payload: agentStatuses });
    }
  }, [agentStatuses]);

  useEffect(() => {
    if (activeConflicts) {
      const conflictsObj = activeConflicts.reduce((acc, conflict) => {
        acc[conflict.conflict_id] = conflict;
        return acc;
      }, {});
      dispatch({ type: ActionTypes.SET_CONFLICTS, payload: conflictsObj });
    }
  }, [activeConflicts]);

  useEffect(() => {
    if (runningWorkflows) {
      const workflowsObj = runningWorkflows.reduce((acc, workflow) => {
        acc[workflow.workflow_id] = workflow;
        return acc;
      }, {});
      dispatch({ type: ActionTypes.SET_WORKFLOWS, payload: workflowsObj });
    }
  }, [runningWorkflows]);

  useEffect(() => {
    if (systemMetrics) {
      dispatch({ type: ActionTypes.SET_METRICS, payload: systemMetrics });
    }
  }, [systemMetrics]);

  // API functions
  const setLoading = (key, value) => {
    dispatch({ type: ActionTypes.SET_LOADING, payload: { key, value } });
  };

  const setError = (key, error) => {
    dispatch({ type: ActionTypes.SET_ERROR, payload: { key, error } });
  };

  const clearError = (key) => {
    dispatch({ type: ActionTypes.CLEAR_ERROR, payload: { key } });
  };

  // Agent API functions
  const registerAgent = async (agentData) => {
    setLoading('agents', true);
    clearError('agents');

    try {
      const response = await axios.post('/api/multi-agent/agents/register', agentData);
      toast.success(`Agent ${agentData.agent_id} registered successfully`);
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError('agents', errorMessage);
      toast.error(`Failed to register agent: ${errorMessage}`);
      throw error;
    } finally {
      setLoading('agents', false);
    }
  };

  const updateAgentStatus = async (agentId, statusData) => {
    try {
      const response = await axios.post(`/api/multi-agent/agents/${agentId}/status`, statusData);
      dispatch({ type: ActionTypes.UPDATE_AGENT, payload: { agent_id: agentId, ...statusData } });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      toast.error(`Failed to update agent status: ${errorMessage}`);
      throw error;
    }
  };

  const fetchActiveAgents = async () => {
    setLoading('agents', true);
    clearError('agents');

    try {
      const response = await axios.get('/api/multi-agent/agents/active');
      dispatch({ type: ActionTypes.SET_AGENTS, payload: response.data.agents });
      return response.data.agents;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError('agents', errorMessage);
      throw error;
    } finally {
      setLoading('agents', false);
    }
  };

  // Workflow API functions
  const createWorkflow = async (templateId, title, description, context = {}) => {
    setLoading('workflows', true);
    clearError('workflows');

    try {
      const response = await axios.post('/api/multi-agent/workflows/create', {
        template_id: templateId,
        title,
        description,
        context,
      });

      dispatch({ type: ActionTypes.ADD_WORKFLOW, payload: response.data });
      toast.success(`Workflow "${title}" created successfully`);
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError('workflows', errorMessage);
      toast.error(`Failed to create workflow: ${errorMessage}`);
      throw error;
    } finally {
      setLoading('workflows', false);
    }
  };

  const assignAgentsToWorkflow = async (workflowId, availableAgents) => {
    try {
      const response = await axios.post(`/api/multi-agent/workflows/${workflowId}/assign-agents`, {
        workflow_id: workflowId,
        available_agents: availableAgents,
      });

      dispatch({ type: ActionTypes.UPDATE_WORKFLOW, payload: { workflow_id: workflowId, assignments: response.data.assignments } });
      toast.success('Agents assigned to workflow successfully');
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      toast.error(`Failed to assign agents: ${errorMessage}`);
      throw error;
    }
  };

  const startWorkflow = async (workflowId) => {
    try {
      const response = await axios.post(`/api/multi-agent/workflows/${workflowId}/start`);
      dispatch({ type: ActionTypes.UPDATE_WORKFLOW, payload: { workflow_id: workflowId, status: 'started' } });
      toast.success('Workflow started successfully');
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      toast.error(`Failed to start workflow: ${errorMessage}`);
      throw error;
    }
  };

  const cancelWorkflow = async (workflowId, reason = 'User cancelled') => {
    try {
      const response = await axios.post(`/api/multi-agent/workflows/${workflowId}/cancel`, { reason });
      dispatch({ type: ActionTypes.UPDATE_WORKFLOW, payload: { workflow_id: workflowId, status: 'cancelled' } });
      toast.success('Workflow cancelled successfully');
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      toast.error(`Failed to cancel workflow: ${errorMessage}`);
      throw error;
    }
  };

  const fetchWorkflowTemplates = async () => {
    try {
      const response = await axios.get('/api/multi-agent/workflows/templates');
      return response.data.templates;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      toast.error(`Failed to fetch templates: ${errorMessage}`);
      throw error;
    }
  };

  // Conflict API functions
  const detectConflicts = async (context) => {
    setLoading('conflicts', true);
    clearError('conflicts');

    try {
      const response = await axios.post('/api/multi-agent/conflicts/detect', context);

      response.data.forEach(conflict => {
        dispatch({ type: ActionTypes.ADD_CONFLICT, payload: conflict });
      });

      if (response.data.length > 0) {
        toast.warning(`${response.data.length} conflict(s) detected`);
      }

      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError('conflicts', errorMessage);
      toast.error(`Failed to detect conflicts: ${errorMessage}`);
      throw error;
    } finally {
      setLoading('conflicts', false);
    }
  };

  const resolveConflict = async (conflictId, preferredStrategy = null) => {
    try {
      const payload = { conflict_id: conflictId };
      if (preferredStrategy) {
        payload.preferred_strategy = preferredStrategy;
      }

      const response = await axios.post(`/api/multi-agent/conflicts/${conflictId}/resolve`, payload);

      if (response.data.success) {
        dispatch({ type: ActionTypes.REMOVE_CONFLICT, payload: { conflict_id: conflictId } });
        toast.success(`Conflict resolved using ${response.data.strategy}`);
      } else {
        toast.error(`Failed to resolve conflict: ${response.data.reason}`);
      }

      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      toast.error(`Failed to resolve conflict: ${errorMessage}`);
      throw error;
    }
  };

  const fetchActiveConflicts = async () => {
    setLoading('conflicts', true);
    clearError('conflicts');

    try {
      const response = await axios.get('/api/multi-agent/conflicts/active');
      const conflictsObj = response.data.reduce((acc, conflict) => {
        acc[conflict.conflict_id] = conflict;
        return acc;
      }, {});

      dispatch({ type: ActionTypes.SET_CONFLICTS, payload: conflictsObj });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError('conflicts', errorMessage);
      throw error;
    } finally {
      setLoading('conflicts', false);
    }
  };

  // Metrics API functions
  const fetchSystemMetrics = async () => {
    setLoading('metrics', true);
    clearError('metrics');

    try {
      const response = await axios.get('/api/multi-agent/status');
      dispatch({ type: ActionTypes.SET_METRICS, payload: response.data });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message;
      setError('metrics', errorMessage);
      throw error;
    } finally {
      setLoading('metrics', false);
    }
  };

  const value = {
    // State
    ...state,

    // Actions
    registerAgent,
    updateAgentStatus,
    fetchActiveAgents,
    createWorkflow,
    assignAgentsToWorkflow,
    startWorkflow,
    cancelWorkflow,
    fetchWorkflowTemplates,
    detectConflicts,
    resolveConflict,
    fetchActiveConflicts,
    fetchSystemMetrics,

    // Utility
    setLoading,
    setError,
    clearError,
  };

  return (
    <AgentContext.Provider value={value}>
      {children}
    </AgentContext.Provider>
  );
};
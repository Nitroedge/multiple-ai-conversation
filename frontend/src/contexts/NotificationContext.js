import React, { createContext, useContext, useReducer } from 'react';

const NotificationContext = createContext();

export const useNotification = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};

// Initial state
const initialState = {
  notifications: [],
  unreadCount: 0,
  settings: {
    conflictAlerts: true,
    workflowUpdates: true,
    agentStatus: true,
    systemMetrics: false,
    soundEnabled: true,
    emailNotifications: false,
  },
};

// Action types
const ActionTypes = {
  ADD_NOTIFICATION: 'ADD_NOTIFICATION',
  MARK_AS_READ: 'MARK_AS_READ',
  MARK_ALL_AS_READ: 'MARK_ALL_AS_READ',
  REMOVE_NOTIFICATION: 'REMOVE_NOTIFICATION',
  CLEAR_ALL_NOTIFICATIONS: 'CLEAR_ALL_NOTIFICATIONS',
  UPDATE_SETTINGS: 'UPDATE_SETTINGS',
};

// Reducer
const notificationReducer = (state, action) => {
  switch (action.type) {
    case ActionTypes.ADD_NOTIFICATION:
      const newNotification = {
        id: Date.now() + Math.random(),
        timestamp: new Date().toISOString(),
        read: false,
        ...action.payload,
      };

      return {
        ...state,
        notifications: [newNotification, ...state.notifications],
        unreadCount: state.unreadCount + 1,
      };

    case ActionTypes.MARK_AS_READ:
      return {
        ...state,
        notifications: state.notifications.map(notification =>
          notification.id === action.payload.id
            ? { ...notification, read: true }
            : notification
        ),
        unreadCount: Math.max(0, state.unreadCount - 1),
      };

    case ActionTypes.MARK_ALL_AS_READ:
      return {
        ...state,
        notifications: state.notifications.map(notification => ({
          ...notification,
          read: true,
        })),
        unreadCount: 0,
      };

    case ActionTypes.REMOVE_NOTIFICATION:
      const notificationToRemove = state.notifications.find(n => n.id === action.payload.id);
      return {
        ...state,
        notifications: state.notifications.filter(notification => notification.id !== action.payload.id),
        unreadCount: notificationToRemove && !notificationToRemove.read
          ? Math.max(0, state.unreadCount - 1)
          : state.unreadCount,
      };

    case ActionTypes.CLEAR_ALL_NOTIFICATIONS:
      return {
        ...state,
        notifications: [],
        unreadCount: 0,
      };

    case ActionTypes.UPDATE_SETTINGS:
      return {
        ...state,
        settings: {
          ...state.settings,
          ...action.payload,
        },
      };

    default:
      return state;
  }
};

export const NotificationProvider = ({ children }) => {
  const [state, dispatch] = useReducer(notificationReducer, initialState);

  // Add notification
  const addNotification = (notification) => {
    // Check if notification type is enabled
    const { type } = notification;
    const settingsMap = {
      conflict: 'conflictAlerts',
      workflow: 'workflowUpdates',
      agent: 'agentStatus',
      system: 'systemMetrics',
    };

    const settingKey = settingsMap[type];
    if (settingKey && !state.settings[settingKey]) {
      return; // Skip notification if disabled
    }

    dispatch({ type: ActionTypes.ADD_NOTIFICATION, payload: notification });

    // Play sound if enabled
    if (state.settings.soundEnabled) {
      playNotificationSound(notification.priority);
    }
  };

  // Mark notification as read
  const markAsRead = (id) => {
    dispatch({ type: ActionTypes.MARK_AS_READ, payload: { id } });
  };

  // Mark all notifications as read
  const markAllAsRead = () => {
    dispatch({ type: ActionTypes.MARK_ALL_AS_READ });
  };

  // Remove notification
  const removeNotification = (id) => {
    dispatch({ type: ActionTypes.REMOVE_NOTIFICATION, payload: { id } });
  };

  // Clear all notifications
  const clearAllNotifications = () => {
    dispatch({ type: ActionTypes.CLEAR_ALL_NOTIFICATIONS });
  };

  // Update notification settings
  const updateSettings = (newSettings) => {
    dispatch({ type: ActionTypes.UPDATE_SETTINGS, payload: newSettings });
  };

  // Helper function to play notification sounds
  const playNotificationSound = (priority = 'normal') => {
    try {
      const audio = new Audio();

      switch (priority) {
        case 'critical':
          audio.src = '/sounds/critical.mp3';
          break;
        case 'high':
          audio.src = '/sounds/high.mp3';
          break;
        case 'normal':
        default:
          audio.src = '/sounds/notification.mp3';
          break;
      }

      audio.volume = 0.5;
      audio.play().catch(() => {
        // Ignore audio play errors (user interaction required)
      });
    } catch (error) {
      // Ignore sound errors
    }
  };

  // Helper functions for specific notification types
  const addConflictNotification = (conflict) => {
    addNotification({
      type: 'conflict',
      title: 'Conflict Detected',
      message: `${conflict.conflict_type} conflict detected between ${conflict.participants.length} agents`,
      priority: conflict.severity === 'critical' ? 'critical' : 'high',
      data: conflict,
      actions: [
        { label: 'View Details', action: 'view_conflict', data: conflict.conflict_id },
        { label: 'Resolve', action: 'resolve_conflict', data: conflict.conflict_id },
      ],
    });
  };

  const addWorkflowNotification = (workflow, status) => {
    const messages = {
      started: `Workflow "${workflow.title}" has started`,
      completed: `Workflow "${workflow.title}" completed successfully`,
      failed: `Workflow "${workflow.title}" failed`,
      cancelled: `Workflow "${workflow.title}" was cancelled`,
    };

    addNotification({
      type: 'workflow',
      title: 'Workflow Update',
      message: messages[status] || `Workflow "${workflow.title}" status changed`,
      priority: status === 'failed' ? 'high' : 'normal',
      data: workflow,
      actions: [
        { label: 'View Workflow', action: 'view_workflow', data: workflow.workflow_id },
      ],
    });
  };

  const addAgentNotification = (agent, status) => {
    const messages = {
      registered: `Agent ${agent.agent_id} has been registered`,
      unregistered: `Agent ${agent.agent_id} has been unregistered`,
      error: `Agent ${agent.agent_id} encountered an error`,
      maintenance: `Agent ${agent.agent_id} is under maintenance`,
    };

    addNotification({
      type: 'agent',
      title: 'Agent Status Update',
      message: messages[status] || `Agent ${agent.agent_id} status changed to ${status}`,
      priority: status === 'error' ? 'high' : 'normal',
      data: agent,
      actions: [
        { label: 'View Agent', action: 'view_agent', data: agent.agent_id },
      ],
    });
  };

  const addSystemNotification = (title, message, priority = 'normal', data = null) => {
    addNotification({
      type: 'system',
      title,
      message,
      priority,
      data,
    });
  };

  const value = {
    // State
    notifications: state.notifications,
    unreadCount: state.unreadCount,
    settings: state.settings,

    // Actions
    addNotification,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAllNotifications,
    updateSettings,

    // Specific notification helpers
    addConflictNotification,
    addWorkflowNotification,
    addAgentNotification,
    addSystemNotification,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};
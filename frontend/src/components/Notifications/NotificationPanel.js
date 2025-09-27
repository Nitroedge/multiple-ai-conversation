import React from 'react';
import {
  Popover,
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Button,
  Chip,
  Divider,
  Paper,
} from '@mui/material';
import {
  Warning as WarningIcon,
  AccountTree as WorkflowIcon,
  Group as AgentIcon,
  Computer as SystemIcon,
  Close as CloseIcon,
  MarkAsUnread as MarkReadIcon,
  Clear as ClearAllIcon,
} from '@mui/icons-material';
import { useNotification } from '../../contexts/NotificationContext';
import moment from 'moment';

const NotificationPanel = ({ anchorEl, open, onClose }) => {
  const {
    notifications,
    unreadCount,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearAllNotifications,
  } = useNotification();

  const getNotificationIcon = (type) => {
    switch (type) {
      case 'conflict':
        return <WarningIcon color="error" />;
      case 'workflow':
        return <WorkflowIcon color="primary" />;
      case 'agent':
        return <AgentIcon color="info" />;
      case 'system':
        return <SystemIcon color="secondary" />;
      default:
        return <SystemIcon />;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'normal':
      default:
        return 'default';
    }
  };

  const handleNotificationClick = (notification) => {
    if (!notification.read) {
      markAsRead(notification.id);
    }

    // Handle notification actions if any
    if (notification.actions && notification.actions.length > 0) {
      const primaryAction = notification.actions[0];
      // Implement action handling based on action.action type
      console.log('Notification action:', primaryAction);
    }
  };

  const handleMarkAllRead = () => {
    markAllAsRead();
  };

  const handleClearAll = () => {
    clearAllNotifications();
  };

  const recentNotifications = notifications.slice(0, 10); // Show only recent 10

  return (
    <Popover
      anchorEl={anchorEl}
      open={open}
      onClose={onClose}
      anchorOrigin={{
        vertical: 'bottom',
        horizontal: 'right',
      }}
      transformOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      PaperProps={{
        sx: {
          width: 400,
          maxHeight: 500,
          backgroundColor: 'background.paper',
          border: '1px solid rgba(255, 255, 255, 0.12)',
        },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Notifications
          {unreadCount > 0 && (
            <Chip
              label={unreadCount}
              size="small"
              color="primary"
              sx={{ ml: 1, height: 20 }}
            />
          )}
        </Typography>

        <IconButton size="small" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </Box>

      {/* Actions */}
      {notifications.length > 0 && (
        <Box
          sx={{
            px: 2,
            py: 1,
            borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
            display: 'flex',
            gap: 1,
          }}
        >
          {unreadCount > 0 && (
            <Button
              size="small"
              startIcon={<MarkReadIcon />}
              onClick={handleMarkAllRead}
            >
              Mark All Read
            </Button>
          )}

          <Button
            size="small"
            startIcon={<ClearAllIcon />}
            onClick={handleClearAll}
            color="error"
          >
            Clear All
          </Button>
        </Box>
      )}

      {/* Notifications List */}
      <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
        {recentNotifications.length === 0 ? (
          <Box
            sx={{
              p: 4,
              textAlign: 'center',
              color: 'text.secondary',
            }}
          >
            <Typography variant="body2">No notifications</Typography>
          </Box>
        ) : (
          <List sx={{ p: 0 }}>
            {recentNotifications.map((notification, index) => (
              <React.Fragment key={notification.id}>
                <ListItem
                  sx={{
                    alignItems: 'flex-start',
                    backgroundColor: notification.read
                      ? 'transparent'
                      : 'rgba(33, 150, 243, 0.08)',
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: notification.read
                        ? 'rgba(255, 255, 255, 0.04)'
                        : 'rgba(33, 150, 243, 0.12)',
                    },
                  }}
                  onClick={() => handleNotificationClick(notification)}
                >
                  <ListItemIcon sx={{ mt: 0.5 }}>
                    {getNotificationIcon(notification.type)}
                  </ListItemIcon>

                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography
                          variant="subtitle2"
                          sx={{
                            fontWeight: notification.read ? 500 : 600,
                            flexGrow: 1,
                          }}
                        >
                          {notification.title}
                        </Typography>

                        <Chip
                          label={notification.priority}
                          size="small"
                          color={getPriorityColor(notification.priority)}
                          sx={{ height: 18, fontSize: '0.7rem' }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography
                          variant="body2"
                          sx={{
                            color: 'text.secondary',
                            mt: 0.5,
                            fontWeight: notification.read ? 400 : 500,
                          }}
                        >
                          {notification.message}
                        </Typography>

                        <Typography
                          variant="caption"
                          sx={{
                            color: 'text.secondary',
                            mt: 0.5,
                            display: 'block',
                          }}
                        >
                          {moment(notification.timestamp).fromNow()}
                        </Typography>

                        {notification.actions && notification.actions.length > 0 && (
                          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                            {notification.actions.slice(0, 2).map((action, actionIndex) => (
                              <Button
                                key={actionIndex}
                                size="small"
                                variant="outlined"
                                sx={{ fontSize: '0.7rem', py: 0.5 }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  // Handle action
                                  console.log('Action clicked:', action);
                                }}
                              >
                                {action.label}
                              </Button>
                            ))}
                          </Box>
                        )}
                      </Box>
                    }
                  />

                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeNotification(notification.id);
                    }}
                    sx={{ mt: 0.5 }}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </ListItem>

                {index < recentNotifications.length - 1 && (
                  <Divider sx={{ mx: 2 }} />
                )}
              </React.Fragment>
            ))}
          </List>
        )}
      </Box>

      {/* Footer */}
      {notifications.length > 10 && (
        <Box
          sx={{
            p: 2,
            borderTop: '1px solid rgba(255, 255, 255, 0.12)',
            textAlign: 'center',
          }}
        >
          <Button size="small" color="primary">
            View All Notifications ({notifications.length})
          </Button>
        </Box>
      )}
    </Popover>
  );
};

export default NotificationPanel;
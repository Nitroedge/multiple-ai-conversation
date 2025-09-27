import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Box,
  Menu,
  MenuItem,
  Avatar,
  Divider,
  Chip,
  Tooltip,
  Button,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  AccountCircle as AccountIcon,
  Settings as SettingsIcon,
  Logout as LogoutIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useNotification } from '../../contexts/NotificationContext';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { useAgent } from '../../contexts/AgentContext';
import NotificationPanel from '../Notifications/NotificationPanel';

const Header = ({ onSidebarToggle }) => {
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const [profileAnchor, setProfileAnchor] = useState(null);
  const [infoAnchor, setInfoAnchor] = useState(null);

  const { unreadCount } = useNotification();
  const { connected, reconnecting, systemMetrics } = useWebSocket();
  const { fetchSystemMetrics, fetchActiveAgents, fetchActiveConflicts } = useAgent();

  const handleNotificationClick = (event) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationClose = () => {
    setNotificationAnchor(null);
  };

  const handleProfileClick = (event) => {
    setProfileAnchor(event.currentTarget);
  };

  const handleProfileClose = () => {
    setProfileAnchor(null);
  };

  const handleInfoClick = (event) => {
    setInfoAnchor(event.currentTarget);
  };

  const handleInfoClose = () => {
    setInfoAnchor(null);
  };

  const handleRefresh = async () => {
    try {
      await Promise.all([
        fetchSystemMetrics(),
        fetchActiveAgents(),
        fetchActiveConflicts(),
      ]);
    } catch (error) {
      console.error('Failed to refresh data:', error);
    }
  };

  const getConnectionStatus = () => {
    if (reconnecting) return { text: 'Reconnecting...', color: 'warning' };
    if (connected) return { text: 'Connected', color: 'success' };
    return { text: 'Disconnected', color: 'error' };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <>
      <AppBar
        position="static"
        elevation={0}
        sx={{
          backgroundColor: 'transparent',
          borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
          mb: 3,
        }}
      >
        <Toolbar sx={{ px: 0 }}>
          {/* Left side - Title and status */}
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Typography variant="h6" sx={{ mr: 3, fontWeight: 600 }}>
              Multi-Agent Coordination Dashboard
            </Typography>

            <Chip
              label={connectionStatus.text}
              color={connectionStatus.color}
              size="small"
              sx={{ mr: 2 }}
            />

            {systemMetrics && (
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip
                  label={`${systemMetrics.active_agents || 0} Agents`}
                  variant="outlined"
                  size="small"
                />
                <Chip
                  label={`${systemMetrics.active_workflows || 0} Workflows`}
                  variant="outlined"
                  size="small"
                />
                {systemMetrics.active_conflicts > 0 && (
                  <Chip
                    label={`${systemMetrics.active_conflicts} Conflicts`}
                    color="error"
                    size="small"
                  />
                )}
              </Box>
            )}
          </Box>

          {/* Right side - Actions */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {/* Refresh button */}
            <Tooltip title="Refresh Data">
              <IconButton
                onClick={handleRefresh}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  },
                }}
              >
                <RefreshIcon />
              </IconButton>
            </Tooltip>

            {/* System info */}
            <Tooltip title="System Information">
              <IconButton
                onClick={handleInfoClick}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  },
                }}
              >
                <InfoIcon />
              </IconButton>
            </Tooltip>

            {/* Notifications */}
            <Tooltip title="Notifications">
              <IconButton
                onClick={handleNotificationClick}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  },
                }}
              >
                <Badge badgeContent={unreadCount} color="error">
                  <NotificationsIcon />
                </Badge>
              </IconButton>
            </Tooltip>

            {/* Profile menu */}
            <Tooltip title="Profile & Settings">
              <IconButton
                onClick={handleProfileClick}
                color="inherit"
                sx={{
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                  },
                }}
              >
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
                  <AccountIcon />
                </Avatar>
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Notification Panel */}
      <NotificationPanel
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={handleNotificationClose}
      />

      {/* Profile Menu */}
      <Menu
        anchorEl={profileAnchor}
        open={Boolean(profileAnchor)}
        onClose={handleProfileClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          sx: {
            mt: 1,
            minWidth: 200,
            backgroundColor: 'background.paper',
            border: '1px solid rgba(255, 255, 255, 0.12)',
          },
        }}
      >
        <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid rgba(255, 255, 255, 0.12)' }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
            System Administrator
          </Typography>
          <Typography variant="caption" color="text.secondary">
            admin@multi-agent-system.local
          </Typography>
        </Box>

        <MenuItem onClick={handleProfileClose}>
          <SettingsIcon sx={{ mr: 2 }} />
          Settings
        </MenuItem>

        <Divider />

        <MenuItem onClick={handleProfileClose}>
          <LogoutIcon sx={{ mr: 2 }} />
          Logout
        </MenuItem>
      </Menu>

      {/* System Info Menu */}
      <Menu
        anchorEl={infoAnchor}
        open={Boolean(infoAnchor)}
        onClose={handleInfoClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          sx: {
            mt: 1,
            minWidth: 280,
            backgroundColor: 'background.paper',
            border: '1px solid rgba(255, 255, 255, 0.12)',
          },
        }}
      >
        <Box sx={{ px: 2, py: 1.5 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            System Information
          </Typography>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Version</Typography>
              <Typography variant="caption" color="primary.main">
                v2.0.0
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Status</Typography>
              <Typography variant="caption" color={connectionStatus.color + '.main'}>
                {connectionStatus.text}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Total Agents</Typography>
              <Typography variant="caption">
                {systemMetrics?.total_agents || 0}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Total Workflows</Typography>
              <Typography variant="caption">
                {systemMetrics?.total_workflows || 0}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Conflicts Resolved</Typography>
              <Typography variant="caption">
                {systemMetrics?.resolved_conflicts || 0}
              </Typography>
            </Box>
          </Box>

          <Button
            size="small"
            fullWidth
            sx={{ mt: 2 }}
            onClick={() => {
              handleInfoClose();
              window.open('/api/multi-agent/health', '_blank');
            }}
          >
            View Full Status
          </Button>
        </Box>
      </Menu>
    </>
  );
};

export default Header;
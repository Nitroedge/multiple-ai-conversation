import React from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Box,
  Typography,
  Chip,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Group as GroupIcon,
  Warning as ConflictIcon,
  AccountTree as WorkflowIcon,
  Chat as CommunicationIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAgent } from '../../contexts/AgentContext';
import { useWebSocket } from '../../contexts/WebSocketContext';

const drawerWidth = 280;
const collapsedWidth = 73;

const menuItems = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: DashboardIcon,
    path: '/dashboard',
    description: 'System overview and metrics',
  },
  {
    id: 'agents',
    label: 'Agent Coordination',
    icon: GroupIcon,
    path: '/agents',
    description: 'Manage and coordinate agents',
    badgeKey: 'active_agents',
  },
  {
    id: 'conflicts',
    label: 'Conflict Resolution',
    icon: ConflictIcon,
    path: '/conflicts',
    description: 'Resolve agent conflicts',
    badgeKey: 'active_conflicts',
    badgeColor: 'error',
  },
  {
    id: 'workflows',
    label: 'Collaboration Workflows',
    icon: WorkflowIcon,
    path: '/workflows',
    description: 'Manage collaboration workflows',
    badgeKey: 'active_workflows',
    badgeColor: 'primary',
  },
  {
    id: 'communication',
    label: 'Communication',
    icon: CommunicationIcon,
    path: '/communication',
    description: 'Agent communication and messaging',
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: AnalyticsIcon,
    path: '/analytics',
    description: 'Performance analytics and insights',
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: SettingsIcon,
    path: '/settings',
    description: 'System configuration and preferences',
  },
];

const Sidebar = ({ open, onToggle, currentPage, onPageChange }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { systemMetrics } = useAgent();
  const { connected, activeConflicts, runningWorkflows } = useWebSocket();

  const handleNavigation = (item) => {
    navigate(item.path);
    onPageChange(item.id);
  };

  const getBadgeValue = (badgeKey) => {
    switch (badgeKey) {
      case 'active_agents':
        return systemMetrics?.active_agents || Object.keys(systemMetrics?.agents || {}).length || 0;
      case 'active_conflicts':
        return activeConflicts?.length || Object.keys(systemMetrics?.conflicts || {}).length || 0;
      case 'active_workflows':
        return runningWorkflows?.length || Object.keys(systemMetrics?.workflows || {}).length || 0;
      default:
        return 0;
    }
  };

  const isActive = (path) => location.pathname === path;

  const SidebarContent = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <Box
        sx={{
          p: open ? 2 : 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: open ? 'space-between' : 'center',
          borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
        }}
      >
        {open && (
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
              Multi-Agent System
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: connected ? 'success.main' : 'error.main',
                  mr: 1,
                }}
              />
              <Typography variant="caption" color="text.secondary">
                {connected ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
          </Box>
        )}

        <IconButton
          onClick={onToggle}
          sx={{
            color: 'text.secondary',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
            },
          }}
        >
          {open ? <ChevronLeftIcon /> : <ChevronRightIcon />}
        </IconButton>
      </Box>

      {/* Navigation */}
      <List sx={{ flexGrow: 1, py: 1 }}>
        {menuItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);
          const badgeValue = item.badgeKey ? getBadgeValue(item.badgeKey) : 0;

          return (
            <Tooltip
              key={item.id}
              title={open ? '' : item.description}
              placement="right"
              arrow
            >
              <ListItem disablePadding sx={{ px: open ? 1 : 0.5 }}>
                <ListItemButton
                  onClick={() => handleNavigation(item)}
                  sx={{
                    borderRadius: 2,
                    mb: 0.5,
                    minHeight: 48,
                    backgroundColor: active
                      ? 'rgba(33, 150, 243, 0.12)'
                      : 'transparent',
                    color: active ? 'primary.main' : 'text.primary',
                    '&:hover': {
                      backgroundColor: active
                        ? 'rgba(33, 150, 243, 0.16)'
                        : 'rgba(255, 255, 255, 0.08)',
                    },
                    px: open ? 2 : 1,
                    justifyContent: open ? 'initial' : 'center',
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: open ? 2 : 'auto',
                      justifyContent: 'center',
                      color: 'inherit',
                    }}
                  >
                    <Icon />
                  </ListItemIcon>

                  {open && (
                    <>
                      <ListItemText
                        primary={item.label}
                        sx={{
                          '& .MuiListItemText-primary': {
                            fontSize: '0.9rem',
                            fontWeight: active ? 600 : 500,
                          },
                        }}
                      />

                      {item.badgeKey && badgeValue > 0 && (
                        <Chip
                          label={badgeValue}
                          size="small"
                          color={item.badgeColor || 'default'}
                          sx={{
                            height: 20,
                            fontSize: '0.75rem',
                            fontWeight: 600,
                          }}
                        />
                      )}
                    </>
                  )}
                </ListItemButton>
              </ListItem>
            </Tooltip>
          );
        })}
      </List>

      {/* System Status */}
      {open && (
        <>
          <Divider sx={{ mx: 2 }} />
          <Box sx={{ p: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              System Status
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption">Active Agents</Typography>
                <Typography variant="caption" color="primary.main">
                  {getBadgeValue('active_agents')}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption">Active Conflicts</Typography>
                <Typography
                  variant="caption"
                  color={getBadgeValue('active_conflicts') > 0 ? 'error.main' : 'text.secondary'}
                >
                  {getBadgeValue('active_conflicts')}
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption">Running Workflows</Typography>
                <Typography variant="caption" color="primary.main">
                  {getBadgeValue('active_workflows')}
                </Typography>
              </Box>
            </Box>
          </Box>
        </>
      )}
    </Box>
  );

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: open ? drawerWidth : collapsedWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: open ? drawerWidth : collapsedWidth,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: '1px solid rgba(255, 255, 255, 0.12)',
          transition: (theme) =>
            theme.transitions.create('width', {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.enteringScreen,
            }),
        },
      }}
    >
      <SidebarContent />
    </Drawer>
  );
};

export default Sidebar;
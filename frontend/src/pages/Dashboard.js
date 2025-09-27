import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  IconButton,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Button,
  Paper,
  Avatar,
} from '@mui/material';
import {
  Group as AgentIcon,
  Warning as ConflictIcon,
  AccountTree as WorkflowIcon,
  Chat as MessageIcon,
  TrendingUp as TrendingUpIcon,
  Speed as PerformanceIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts';
import { useAgent } from '../contexts/AgentContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useNotification } from '../contexts/NotificationContext';
import moment from 'moment';

// Sample data for charts (in real implementation, this would come from API)
const performanceData = [
  { time: '00:00', agents: 12, conflicts: 2, workflows: 5 },
  { time: '04:00', agents: 15, conflicts: 1, workflows: 8 },
  { time: '08:00', agents: 22, conflicts: 3, workflows: 12 },
  { time: '12:00', agents: 28, conflicts: 5, workflows: 15 },
  { time: '16:00', agents: 25, conflicts: 2, workflows: 18 },
  { time: '20:00', agents: 20, conflicts: 1, workflows: 14 },
];

const agentTypeData = [
  { name: 'Specialists', value: 45, color: '#2196f3' },
  { name: 'Generalists', value: 30, color: '#4caf50' },
  { name: 'Coordinators', value: 15, color: '#ff9800' },
  { name: 'Reviewers', value: 10, color: '#9c27b0' },
];

const workflowStatusData = [
  { status: 'Completed', count: 25, color: '#4caf50' },
  { status: 'Running', count: 8, color: '#2196f3' },
  { status: 'Failed', count: 3, color: '#f44336' },
  { status: 'Cancelled', count: 2, color: '#9e9e9e' },
];

const Dashboard = () => {
  const {
    systemMetrics,
    agents,
    workflows,
    conflicts,
    loading,
    fetchSystemMetrics,
    fetchActiveAgents,
  } = useAgent();

  const {
    connected,
    agentStatuses,
    activeConflicts,
    runningWorkflows,
    messages,
  } = useWebSocket();

  const { notifications } = useNotification();

  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Initial data fetch
    const loadDashboardData = async () => {
      try {
        await Promise.all([
          fetchSystemMetrics(),
          fetchActiveAgents(),
        ]);
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
      }
    };

    loadDashboardData();
  }, [fetchSystemMetrics, fetchActiveAgents]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await Promise.all([
        fetchSystemMetrics(),
        fetchActiveAgents(),
      ]);
    } catch (error) {
      console.error('Failed to refresh dashboard:', error);
    }
    setRefreshing(false);
  };

  // Calculate derived metrics
  const totalAgents = Object.keys(agentStatuses).length || systemMetrics?.total_agents || 0;
  const activeAgents = Object.values(agentStatuses).filter(a => a.status === 'available').length || systemMetrics?.active_agents || 0;
  const totalConflicts = activeConflicts?.length || Object.keys(conflicts).length || 0;
  const totalWorkflows = runningWorkflows?.length || Object.keys(workflows).length || 0;
  const recentMessages = messages?.slice(-5) || [];
  const recentNotifications = notifications?.slice(0, 5) || [];

  // Status card component
  const StatusCard = ({ title, value, subtitle, icon: Icon, color, trend }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography color="text.secondary" gutterBottom variant="overline">
              {title}
            </Typography>
            <Typography variant="h4" sx={{ fontWeight: 600, color }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography color="text.secondary" variant="body2">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Avatar sx={{ bgcolor: color, width: 56, height: 56 }}>
            <Icon />
          </Avatar>
        </Box>
        {trend && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            <TrendingUpIcon sx={{ color: 'success.main', mr: 0.5 }} />
            <Typography variant="caption" color="success.main">
              {trend}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
            Dashboard Overview
          </Typography>
          <Typography color="text.secondary">
            Real-time multi-agent system monitoring and coordination
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            label={connected ? 'Connected' : 'Disconnected'}
            color={connected ? 'success' : 'error'}
            variant="outlined"
          />
          <Button
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Status Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="Active Agents"
            value={activeAgents}
            subtitle={`${totalAgents} total registered`}
            icon={AgentIcon}
            color="primary.main"
            trend="+12% from yesterday"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="Active Conflicts"
            value={totalConflicts}
            subtitle={`${systemMetrics?.resolved_conflicts || 0} resolved today`}
            icon={ConflictIcon}
            color={totalConflicts > 0 ? "error.main" : "success.main"}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="Running Workflows"
            value={totalWorkflows}
            subtitle={`${systemMetrics?.completed_workflows || 0} completed`}
            icon={WorkflowIcon}
            color="info.main"
            trend="+8% efficiency"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatusCard
            title="Messages Today"
            value={messages?.length || 0}
            subtitle="Inter-agent communication"
            icon={MessageIcon}
            color="secondary.main"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Performance Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  System Performance (24h)
                </Typography>
                <IconButton size="small">
                  <MoreIcon />
                </IconButton>
              </Box>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="time" stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a1a',
                      border: '1px solid rgba(255,255,255,0.12)',
                      borderRadius: 8,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="agents"
                    stroke="#2196f3"
                    strokeWidth={3}
                    dot={{ fill: '#2196f3', strokeWidth: 2, r: 4 }}
                    name="Active Agents"
                  />
                  <Line
                    type="monotone"
                    dataKey="workflows"
                    stroke="#4caf50"
                    strokeWidth={3}
                    dot={{ fill: '#4caf50', strokeWidth: 2, r: 4 }}
                    name="Workflows"
                  />
                  <Line
                    type="monotone"
                    dataKey="conflicts"
                    stroke="#f44336"
                    strokeWidth={3}
                    dot={{ fill: '#f44336', strokeWidth: 2, r: 4 }}
                    name="Conflicts"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Agent Types Distribution */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Agent Types Distribution
              </Typography>

              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={agentTypeData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {agentTypeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>

              <Box sx={{ mt: 2 }}>
                {agentTypeData.map((item, index) => (
                  <Box key={index} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Box
                      sx={{
                        width: 12,
                        height: 12,
                        bgcolor: item.color,
                        borderRadius: '50%',
                        mr: 1,
                      }}
                    />
                    <Typography variant="body2" sx={{ flexGrow: 1 }}>
                      {item.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {item.value}%
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Recent Activity
              </Typography>

              <List dense>
                {recentNotifications.length > 0 ? (
                  recentNotifications.map((notification, index) => (
                    <ListItem key={index} sx={{ px: 0 }}>
                      <ListItemIcon>
                        {notification.type === 'conflict' && <ConflictIcon color="error" />}
                        {notification.type === 'workflow' && <WorkflowIcon color="primary" />}
                        {notification.type === 'agent' && <AgentIcon color="info" />}
                        {notification.type === 'system' && <PerformanceIcon color="secondary" />}
                      </ListItemIcon>
                      <ListItemText
                        primary={notification.title}
                        secondary={
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              {notification.message}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {moment(notification.timestamp).fromNow()}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))
                ) : (
                  <ListItem sx={{ px: 0 }}>
                    <ListItemText
                      primary="No recent activity"
                      secondary="System is running smoothly"
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Workflow Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
                Workflow Status
              </Typography>

              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={workflowStatusData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="status" stroke="#666" />
                  <YAxis stroke="#666" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a1a',
                      border: '1px solid rgba(255,255,255,0.12)',
                      borderRadius: 8,
                    }}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {workflowStatusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Loading overlay */}
      {(loading.metrics || loading.agents || refreshing) && (
        <LinearProgress
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 9999,
          }}
        />
      )}
    </Box>
  );
};

export default Dashboard;
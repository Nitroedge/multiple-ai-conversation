import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Avatar,
  LinearProgress,
  Tooltip,
  Menu,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Add as AddIcon,
  MoreVert as MoreIcon,
  Refresh as RefreshIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  Group as GroupIcon,
  Computer as ComputerIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import { useAgent } from '../contexts/AgentContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useNotification } from '../contexts/NotificationContext';
import moment from 'moment';

const AgentCoordination = () => {
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [registerDialogOpen, setRegisterDialogOpen] = useState(false);
  const [actionMenuAnchor, setActionMenuAnchor] = useState(null);
  const [menuAgentId, setMenuAgentId] = useState(null);

  const {
    agents,
    systemMetrics,
    loading,
    registerAgent,
    updateAgentStatus,
    fetchActiveAgents,
  } = useAgent();

  const { agentStatuses, connected } = useWebSocket();
  const { addAgentNotification } = useNotification();

  // Form state for new agent registration
  const [newAgent, setNewAgent] = useState({
    agent_id: '',
    capabilities: {
      domains: [],
      skills: [],
      languages: [],
    },
    specializations: [],
    max_concurrent_tasks: 3,
    priority_level: 'normal',
  });

  useEffect(() => {
    fetchActiveAgents();
  }, [fetchActiveAgents]);

  // Convert agents object to array for DataGrid
  const agentRows = Object.entries(agentStatuses).map(([agentId, agent]) => ({
    id: agentId,
    agent_id: agentId,
    status: agent.status || 'unknown',
    capabilities: agent.capabilities || {},
    performance_score: agent.performance_score || 0,
    last_updated: agent.last_updated || new Date().toISOString(),
    current_tasks: agent.current_tasks || 0,
    total_tasks: agent.total_tasks || 0,
    success_rate: agent.success_rate || 0,
    ...agent,
  }));

  const handleRegisterAgent = async () => {
    try {
      await registerAgent(newAgent);
      setRegisterDialogOpen(false);
      setNewAgent({
        agent_id: '',
        capabilities: {
          domains: [],
          skills: [],
          languages: [],
        },
        specializations: [],
        max_concurrent_tasks: 3,
        priority_level: 'normal',
      });
      addAgentNotification({ agent_id: newAgent.agent_id }, 'registered');
    } catch (error) {
      console.error('Failed to register agent:', error);
    }
  };

  const handleUpdateAgentStatus = async (agentId, status) => {
    try {
      await updateAgentStatus(agentId, { status });
      addAgentNotification({ agent_id: agentId }, status);
    } catch (error) {
      console.error('Failed to update agent status:', error);
    }
  };

  const handleActionMenuOpen = (event, agentId) => {
    setActionMenuAnchor(event.currentTarget);
    setMenuAgentId(agentId);
  };

  const handleActionMenuClose = () => {
    setActionMenuAnchor(null);
    setMenuAgentId(null);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'available':
        return 'success';
      case 'busy':
        return 'warning';
      case 'processing':
        return 'info';
      case 'error':
        return 'error';
      case 'offline':
        return 'default';
      case 'maintenance':
        return 'secondary';
      default:
        return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'available':
        return <CheckCircleIcon color="success" />;
      case 'busy':
      case 'processing':
        return <WarningIcon color="warning" />;
      case 'error':
        return <ErrorIcon color="error" />;
      default:
        return <ComputerIcon color="disabled" />;
    }
  };

  const columns = [
    {
      field: 'agent_id',
      headerName: 'Agent ID',
      width: 150,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Avatar sx={{ width: 32, height: 32, mr: 1, bgcolor: 'primary.main' }}>
            <GroupIcon />
          </Avatar>
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {params.value}
          </Typography>
        </Box>
      ),
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 120,
      renderCell: (params) => (
        <Chip
          label={params.value}
          color={getStatusColor(params.value)}
          size="small"
          icon={getStatusIcon(params.value)}
        />
      ),
    },
    {
      field: 'performance_score',
      headerName: 'Performance',
      width: 120,
      renderCell: (params) => (
        <Box sx={{ width: '100%' }}>
          <LinearProgress
            variant="determinate"
            value={params.value * 100}
            sx={{ mb: 0.5 }}
            color={params.value > 0.8 ? 'success' : params.value > 0.6 ? 'warning' : 'error'}
          />
          <Typography variant="caption">
            {Math.round(params.value * 100)}%
          </Typography>
        </Box>
      ),
    },
    {
      field: 'current_tasks',
      headerName: 'Current Tasks',
      width: 100,
      renderCell: (params) => (
        <Typography variant="body2">
          {params.value || 0} / {params.row.max_concurrent_tasks || 3}
        </Typography>
      ),
    },
    {
      field: 'success_rate',
      headerName: 'Success Rate',
      width: 100,
      renderCell: (params) => (
        <Typography variant="body2">
          {Math.round((params.value || 0) * 100)}%
        </Typography>
      ),
    },
    {
      field: 'last_updated',
      headerName: 'Last Updated',
      width: 150,
      renderCell: (params) => (
        <Typography variant="caption" color="text.secondary">
          {moment(params.value).fromNow()}
        </Typography>
      ),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 100,
      sortable: false,
      renderCell: (params) => (
        <IconButton
          size="small"
          onClick={(event) => handleActionMenuOpen(event, params.row.agent_id)}
        >
          <MoreIcon />
        </IconButton>
      ),
    },
  ];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
            Agent Coordination
          </Typography>
          <Typography color="text.secondary">
            Manage and monitor multi-agent system coordination
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            startIcon={<RefreshIcon />}
            onClick={fetchActiveAgents}
            disabled={loading.agents}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setRegisterDialogOpen(true)}
          >
            Register Agent
          </Button>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Total Agents
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                {agentRows.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Available
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                {agentRows.filter(a => a.status === 'available').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Busy/Processing
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                {agentRows.filter(a => ['busy', 'processing'].includes(a.status)).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Offline/Error
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                {agentRows.filter(a => ['offline', 'error'].includes(a.status)).length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Agents Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Active Agents
          </Typography>

          <Box sx={{ height: 600, width: '100%' }}>
            <DataGrid
              rows={agentRows}
              columns={columns}
              pageSize={10}
              rowsPerPageOptions={[10, 25, 50]}
              checkboxSelection
              disableSelectionOnClick
              loading={loading.agents}
              sx={{
                '& .MuiDataGrid-cell': {
                  borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
                },
                '& .MuiDataGrid-columnHeaders': {
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
                },
              }}
            />
          </Box>
        </CardContent>
      </Card>

      {/* Register Agent Dialog */}
      <Dialog
        open={registerDialogOpen}
        onClose={() => setRegisterDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Register New Agent</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Agent ID"
                value={newAgent.agent_id}
                onChange={(e) => setNewAgent({ ...newAgent, agent_id: e.target.value })}
                placeholder="e.g., agent_001"
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Priority Level</InputLabel>
                <Select
                  value={newAgent.priority_level}
                  onChange={(e) => setNewAgent({ ...newAgent, priority_level: e.target.value })}
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="normal">Normal</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="critical">Critical</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                type="number"
                label="Max Concurrent Tasks"
                value={newAgent.max_concurrent_tasks}
                onChange={(e) => setNewAgent({ ...newAgent, max_concurrent_tasks: parseInt(e.target.value) })}
                inputProps={{ min: 1, max: 10 }}
              />
            </Grid>

            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Specializations"
                value={newAgent.specializations.join(', ')}
                onChange={(e) => setNewAgent({
                  ...newAgent,
                  specializations: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                })}
                placeholder="e.g., analysis, coordination, review"
                helperText="Comma-separated list"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Capabilities
              </Typography>
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Domains"
                value={newAgent.capabilities.domains.join(', ')}
                onChange={(e) => setNewAgent({
                  ...newAgent,
                  capabilities: {
                    ...newAgent.capabilities,
                    domains: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                  }
                })}
                placeholder="e.g., technology, creative, analysis"
                helperText="Comma-separated list"
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Skills"
                value={newAgent.capabilities.skills.join(', ')}
                onChange={(e) => setNewAgent({
                  ...newAgent,
                  capabilities: {
                    ...newAgent.capabilities,
                    skills: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                  }
                })}
                placeholder="e.g., coding, writing, math"
                helperText="Comma-separated list"
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Languages"
                value={newAgent.capabilities.languages.join(', ')}
                onChange={(e) => setNewAgent({
                  ...newAgent,
                  capabilities: {
                    ...newAgent.capabilities,
                    languages: e.target.value.split(',').map(s => s.trim()).filter(s => s)
                  }
                })}
                placeholder="e.g., english, python, javascript"
                helperText="Comma-separated list"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRegisterDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleRegisterAgent}
            disabled={!newAgent.agent_id || loading.agents}
          >
            Register Agent
          </Button>
        </DialogActions>
      </Dialog>

      {/* Action Menu */}
      <Menu
        anchorEl={actionMenuAnchor}
        open={Boolean(actionMenuAnchor)}
        onClose={handleActionMenuClose}
      >
        <MenuItem onClick={() => {
          handleUpdateAgentStatus(menuAgentId, 'available');
          handleActionMenuClose();
        }}>
          <ListItemIcon>
            <StartIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Set Available</ListItemText>
        </MenuItem>

        <MenuItem onClick={() => {
          handleUpdateAgentStatus(menuAgentId, 'maintenance');
          handleActionMenuClose();
        }}>
          <ListItemIcon>
            <SettingsIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Set Maintenance</ListItemText>
        </MenuItem>

        <MenuItem onClick={() => {
          handleUpdateAgentStatus(menuAgentId, 'offline');
          handleActionMenuClose();
        }}>
          <ListItemIcon>
            <StopIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Set Offline</ListItemText>
        </MenuItem>

        <MenuItem onClick={() => {
          setSelectedAgent(menuAgentId);
          handleActionMenuClose();
        }}>
          <ListItemIcon>
            <EditIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText>Edit Agent</ListItemText>
        </MenuItem>
      </Menu>

      {/* Loading overlay */}
      {loading.agents && (
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

export default AgentCoordination;
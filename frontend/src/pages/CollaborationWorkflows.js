import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  IconButton,
  Chip,
  Button,
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  LinearProgress,
} from '@mui/material';
import {
  AccountTree as WorkflowIcon,
  Add as AddIcon,
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Pause as PauseIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Visibility as ViewIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useAgent } from '../contexts/AgentContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useNotification } from '../contexts/NotificationContext';

const CollaborationWorkflows = () => {
  const { workflows, loading, fetchWorkflows } = useAgent();
  const { connected, runningWorkflows } = useWebSocket();
  const { addNotification } = useNotification();

  const [openDialog, setOpenDialog] = useState(false);
  const [selectedWorkflow, setSelectedWorkflow] = useState(null);
  const [workflowData, setWorkflowData] = useState({
    name: '',
    type: '',
    description: '',
    agents: [],
    status: 'draft'
  });

  const workflowTypes = [
    'Pipeline Processing',
    'Parallel Processing',
    'Divide and Conquer',
    'Brainstorming',
    'Consensus Building',
    'Quality Assurance',
    'Research Analysis',
    'Creative Writing',
    'Problem Solving',
    'Data Analysis'
  ];

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const handleCreateWorkflow = () => {
    setSelectedWorkflow(null);
    setWorkflowData({
      name: '',
      type: '',
      description: '',
      agents: [],
      status: 'draft'
    });
    setOpenDialog(true);
  };

  const handleEditWorkflow = (workflow) => {
    setSelectedWorkflow(workflow);
    setWorkflowData(workflow);
    setOpenDialog(true);
  };

  const handleSaveWorkflow = async () => {
    try {
      // API call to save workflow would go here
      addNotification('success', 'Workflow saved successfully');
      setOpenDialog(false);
      fetchWorkflows();
    } catch (error) {
      addNotification('error', 'Failed to save workflow');
    }
  };

  const handleStartWorkflow = async (workflowId) => {
    try {
      // API call to start workflow would go here
      addNotification('success', 'Workflow started successfully');
    } catch (error) {
      addNotification('error', 'Failed to start workflow');
    }
  };

  const handleStopWorkflow = async (workflowId) => {
    try {
      // API call to stop workflow would go here
      addNotification('success', 'Workflow stopped successfully');
    } catch (error) {
      addNotification('error', 'Failed to stop workflow');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <StopIcon />;
      case 'paused': return <StartIcon />;
      default: return <StartIcon />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, color: 'text.primary' }}>
          Collaboration Workflows
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={fetchWorkflows}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={handleCreateWorkflow}
          >
            Create Workflow
          </Button>
        </Box>
      </Box>

      {/* Workflow Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Workflows
                  </Typography>
                  <Typography variant="h4" component="div">
                    {workflows?.length || 0}
                  </Typography>
                </Box>
                <WorkflowIcon sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Running
                  </Typography>
                  <Typography variant="h4" component="div" color="primary.main">
                    {runningWorkflows?.length || 0}
                  </Typography>
                </Box>
                <StartIcon sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Completed Today
                  </Typography>
                  <Typography variant="h4" component="div" color="success.main">
                    12
                  </Typography>
                </Box>
                <Chip label="✓" sx={{ fontSize: 20, bgcolor: 'success.main', color: 'white' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Success Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="success.main">
                    94%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={94}
                  sx={{ width: 40, height: 6, borderRadius: 3 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Workflows Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Workflow Management
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Agents</TableCell>
                  <TableCell>Progress</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {workflows?.map((workflow) => (
                  <TableRow key={workflow.id}>
                    <TableCell>{workflow.name}</TableCell>
                    <TableCell>{workflow.type}</TableCell>
                    <TableCell>
                      <Chip
                        label={workflow.status}
                        color={getStatusColor(workflow.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{workflow.agents?.length || 0}</TableCell>
                    <TableCell>
                      <LinearProgress
                        variant="determinate"
                        value={workflow.progress || 0}
                        sx={{ minWidth: 100 }}
                      />
                    </TableCell>
                    <TableCell>{workflow.created || 'N/A'}</TableCell>
                    <TableCell>
                      <IconButton
                        size="small"
                        onClick={() => handleEditWorkflow(workflow)}
                      >
                        <EditIcon />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => workflow.status === 'running'
                          ? handleStopWorkflow(workflow.id)
                          : handleStartWorkflow(workflow.id)
                        }
                      >
                        {getStatusIcon(workflow.status)}
                      </IconButton>
                      <IconButton size="small">
                        <ViewIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Create/Edit Workflow Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          {selectedWorkflow ? 'Edit Workflow' : 'Create New Workflow'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 2 }}>
            <TextField
              label="Workflow Name"
              value={workflowData.name}
              onChange={(e) => setWorkflowData({ ...workflowData, name: e.target.value })}
              fullWidth
            />
            <FormControl fullWidth>
              <InputLabel>Workflow Type</InputLabel>
              <Select
                value={workflowData.type}
                onChange={(e) => setWorkflowData({ ...workflowData, type: e.target.value })}
              >
                {workflowTypes.map((type) => (
                  <MenuItem key={type} value={type}>
                    {type}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Description"
              value={workflowData.description}
              onChange={(e) => setWorkflowData({ ...workflowData, description: e.target.value })}
              multiline
              rows={4}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveWorkflow} variant="contained">
            {selectedWorkflow ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CollaborationWorkflows;
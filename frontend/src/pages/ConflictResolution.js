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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Alert,
  LinearProgress,
  Tooltip,
  Avatar,
  AvatarGroup,
} from '@mui/material';
import {
  Warning as WarningIcon,
  CheckCircle as ResolvedIcon,
  HourglassEmpty as PendingIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as ResolveIcon,
  Refresh as RefreshIcon,
  Group as GroupIcon,
  Timeline as TimelineIcon,
  AccountTree as TreeIcon,
  Gavel as GavelIcon,
  ThumbUp as VoteIcon,
  Psychology as ExpertIcon,
  Shuffle as RandomIcon,
  Balance as CompromiseIcon,
  TrendingUp as EscalateIcon,
} from '@mui/icons-material';
import { DataGrid } from '@mui/x-data-grid';
import { useAgent } from '../contexts/AgentContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useNotification } from '../contexts/NotificationContext';
import moment from 'moment';

const ConflictResolution = () => {
  const [selectedConflict, setSelectedConflict] = useState(null);
  const [resolveDialogOpen, setResolveDialogOpen] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [detailsDialogOpen, setDetailsDialogOpen] = useState(false);

  const {
    conflicts,
    loading,
    fetchActiveConflicts,
    resolveConflict,
    detectConflicts,
  } = useAgent();

  const { activeConflicts, connected } = useWebSocket();
  const { addConflictNotification } = useNotification();

  useEffect(() => {
    fetchActiveConflicts();
  }, [fetchActiveConflicts]);

  // Convert conflicts object to array for DataGrid
  const conflictRows = Object.entries(conflicts).map(([conflictId, conflict]) => ({
    id: conflictId,
    conflict_id: conflictId,
    conflict_type: conflict.conflict_type || 'unknown',
    severity: conflict.severity || 'medium',
    status: conflict.status || 'detected',
    participants_count: conflict.participants?.length || 0,
    participants: conflict.participants || [],
    detected_at: conflict.detected_at || new Date().toISOString(),
    resolution_strategy: conflict.resolution_strategy,
    ...conflict,
  }));

  const handleResolveConflict = async () => {
    if (!selectedConflict) return;

    try {
      await resolveConflict(selectedConflict.conflict_id, selectedStrategy || null);
      setResolveDialogOpen(false);
      setSelectedConflict(null);
      setSelectedStrategy('');
    } catch (error) {
      console.error('Failed to resolve conflict:', error);
    }
  };

  const handleShowDetails = (conflict) => {
    setSelectedConflict(conflict);
    setDetailsDialogOpen(true);
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'high':
        return 'warning';
      case 'medium':
        return 'info';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'resolved':
        return 'success';
      case 'resolving':
        return 'warning';
      case 'escalated':
        return 'error';
      case 'detected':
      case 'analyzing':
      default:
        return 'primary';
    }
  };

  const getStrategyIcon = (strategy) => {
    switch (strategy) {
      case 'hierarchical':
        return <TreeIcon />;
      case 'consensus':
        return <GroupIcon />;
      case 'expertise_based':
        return <ExpertIcon />;
      case 'majority_vote':
        return <VoteIcon />;
      case 'compromise':
        return <CompromiseIcon />;
      case 'random':
        return <RandomIcon />;
      case 'escalation':
        return <EscalateIcon />;
      default:
        return <GavelIcon />;
    }
  };

  const getStrategyDescription = (strategy) => {
    const descriptions = {
      hierarchical: 'Highest priority agent wins',
      consensus: 'Seek agreement among all participants',
      expertise_based: 'Most qualified agent decides',
      majority_vote: 'Democratic decision making',
      round_robin: 'Take turns approach',
      compromise: 'Find middle ground solution',
      performance_based: 'Best performing agent wins',
      random: 'Random selection',
      escalation: 'Escalate to higher authority',
    };
    return descriptions[strategy] || 'Unknown strategy';
  };

  const columns = [
    {
      field: 'conflict_type',
      headerName: 'Type',
      width: 150,
      renderCell: (params) => (
        <Chip
          label={params.value.replace('_', ' ')}
          size="small"
          variant="outlined"
        />
      ),
    },
    {
      field: 'severity',
      headerName: 'Severity',
      width: 100,
      renderCell: (params) => (
        <Chip
          label={params.value}
          color={getSeverityColor(params.value)}
          size="small"
        />
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
          icon={
            params.value === 'resolved' ? <ResolvedIcon /> :
            params.value === 'resolving' ? <PendingIcon /> :
            <WarningIcon />
          }
        />
      ),
    },
    {
      field: 'participants',
      headerName: 'Participants',
      width: 150,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <AvatarGroup max={3} sx={{ mr: 1 }}>
            {params.value.map((participant, index) => (
              <Avatar key={index} sx={{ width: 24, height: 24, fontSize: '0.75rem' }}>
                {participant.agent_id[0]}
              </Avatar>
            ))}
          </AvatarGroup>
          <Typography variant="caption">
            {params.value.length} agents
          </Typography>
        </Box>
      ),
    },
    {
      field: 'detected_at',
      headerName: 'Detected',
      width: 150,
      renderCell: (params) => (
        <Typography variant="caption" color="text.secondary">
          {moment(params.value).fromNow()}
        </Typography>
      ),
    },
    {
      field: 'resolution_strategy',
      headerName: 'Strategy',
      width: 150,
      renderCell: (params) => (
        params.value ? (
          <Tooltip title={getStrategyDescription(params.value)}>
            <Chip
              icon={getStrategyIcon(params.value)}
              label={params.value.replace('_', ' ')}
              size="small"
              variant="outlined"
            />
          </Tooltip>
        ) : (
          <Typography variant="caption" color="text.secondary">
            Pending
          </Typography>
        )
      ),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 150,
      sortable: false,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="View Details">
            <IconButton
              size="small"
              onClick={() => handleShowDetails(params.row)}
            >
              <TimelineIcon />
            </IconButton>
          </Tooltip>

          {params.row.status !== 'resolved' && (
            <Tooltip title="Resolve Conflict">
              <IconButton
                size="small"
                color="primary"
                onClick={() => {
                  setSelectedConflict(params.row);
                  setResolveDialogOpen(true);
                }}
              >
                <ResolveIcon />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      ),
    },
  ];

  const resolutionStrategies = [
    'hierarchical',
    'consensus',
    'expertise_based',
    'majority_vote',
    'round_robin',
    'compromise',
    'performance_based',
    'random',
  ];

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 600, mb: 1 }}>
            Conflict Resolution
          </Typography>
          <Typography color="text.secondary">
            Detect and resolve conflicts between agents
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            startIcon={<RefreshIcon />}
            onClick={fetchActiveConflicts}
            disabled={loading.conflicts}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Active Conflicts
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                {conflictRows.filter(c => c.status !== 'resolved').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Critical Severity
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                {conflictRows.filter(c => c.severity === 'critical').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Resolved Today
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                {conflictRows.filter(c => c.status === 'resolved').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom variant="overline">
                Success Rate
              </Typography>
              <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                95%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Alerts for critical conflicts */}
      {conflictRows.filter(c => c.severity === 'critical' && c.status !== 'resolved').length > 0 && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="subtitle2">
            Critical conflicts detected! Immediate attention required.
          </Typography>
          <Typography variant="body2">
            {conflictRows.filter(c => c.severity === 'critical' && c.status !== 'resolved').length} critical conflicts need resolution.
          </Typography>
        </Alert>
      )}

      {/* Conflicts Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
            Active Conflicts
          </Typography>

          <Box sx={{ height: 600, width: '100%' }}>
            <DataGrid
              rows={conflictRows}
              columns={columns}
              pageSize={10}
              rowsPerPageOptions={[10, 25, 50]}
              checkboxSelection
              disableSelectionOnClick
              loading={loading.conflicts}
              sx={{
                '& .MuiDataGrid-cell': {
                  borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
                },
                '& .MuiDataGrid-columnHeaders': {
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderBottom: '1px solid rgba(255, 255, 255, 0.12)',
                },
                '& .MuiDataGrid-row': {
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.04)',
                  },
                },
              }}
            />
          </Box>
        </CardContent>
      </Card>

      {/* Resolve Conflict Dialog */}
      <Dialog
        open={resolveDialogOpen}
        onClose={() => setResolveDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Resolve Conflict</DialogTitle>
        <DialogContent>
          {selectedConflict && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                {selectedConflict.conflict_type.replace('_', ' ')} Conflict
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Severity
                  </Typography>
                  <Chip
                    label={selectedConflict.severity}
                    color={getSeverityColor(selectedConflict.severity)}
                  />
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Participants
                  </Typography>
                  <Typography variant="body2">
                    {selectedConflict.participants?.map(p => p.agent_id).join(', ')}
                  </Typography>
                </Grid>

                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Resolution Strategy
                  </Typography>
                  <FormControl fullWidth>
                    <InputLabel>Choose Strategy</InputLabel>
                    <Select
                      value={selectedStrategy}
                      onChange={(e) => setSelectedStrategy(e.target.value)}
                    >
                      <MenuItem value="">Auto-select best strategy</MenuItem>
                      {resolutionStrategies.map((strategy) => (
                        <MenuItem key={strategy} value={strategy}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {getStrategyIcon(strategy)}
                            <Box>
                              <Typography variant="body2">
                                {strategy.replace('_', ' ')}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {getStrategyDescription(strategy)}
                              </Typography>
                            </Box>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResolveDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleResolveConflict}
            disabled={loading.conflicts}
            startIcon={<ResolveIcon />}
          >
            Resolve Conflict
          </Button>
        </DialogActions>
      </Dialog>

      {/* Conflict Details Dialog */}
      <Dialog
        open={detailsDialogOpen}
        onClose={() => setDetailsDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>Conflict Details</DialogTitle>
        <DialogContent>
          {selectedConflict && (
            <Box sx={{ mt: 2 }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={8}>
                  <Typography variant="h6" gutterBottom>
                    Conflict Information
                  </Typography>

                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Conflict Type"
                        secondary={selectedConflict.conflict_type.replace('_', ' ')}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Severity"
                        secondary={
                          <Chip
                            label={selectedConflict.severity}
                            color={getSeverityColor(selectedConflict.severity)}
                            size="small"
                          />
                        }
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Status"
                        secondary={
                          <Chip
                            label={selectedConflict.status}
                            color={getStatusColor(selectedConflict.status)}
                            size="small"
                          />
                        }
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Detected At"
                        secondary={moment(selectedConflict.detected_at).format('YYYY-MM-DD HH:mm:ss')}
                      />
                    </ListItem>
                  </List>
                </Grid>

                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom>
                    Participants
                  </Typography>

                  <List dense>
                    {selectedConflict.participants?.map((participant, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <Avatar sx={{ width: 32, height: 32 }}>
                            {participant.agent_id[0]}
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText
                          primary={participant.agent_id}
                          secondary={`Priority: ${participant.priority}, Confidence: ${Math.round(participant.confidence * 100)}%`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Grid>

                {selectedConflict.resolution_result && (
                  <Grid item xs={12}>
                    <Typography variant="h6" gutterBottom>
                      Resolution Result
                    </Typography>

                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>View Resolution Details</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <pre style={{ fontSize: '0.875rem', overflow: 'auto' }}>
                          {JSON.stringify(selectedConflict.resolution_result, null, 2)}
                        </pre>
                      </AccordionDetails>
                    </Accordion>
                  </Grid>
                )}
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Loading overlay */}
      {loading.conflicts && (
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

export default ConflictResolution;
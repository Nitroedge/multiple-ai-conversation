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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  Speed as PerformanceIcon,
  Timeline as TimelineIcon,
  Assessment as ReportIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  DateRange as DateRangeIcon,
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
  ComposedChart,
  Legend,
} from 'recharts';
import { useAgent } from '../contexts/AgentContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import moment from 'moment';

const Analytics = () => {
  const { systemMetrics, agents, workflows } = useAgent();
  const { connected } = useWebSocket();

  const [selectedTab, setSelectedTab] = useState(0);
  const [timeRange, setTimeRange] = useState('24h');
  const [metricType, setMetricType] = useState('performance');

  // Sample analytics data (in real implementation, this would come from API)
  const performanceMetrics = [
    { time: '00:00', responseTime: 120, throughput: 450, errorRate: 2.1, cpuUsage: 45 },
    { time: '04:00', responseTime: 110, throughput: 520, errorRate: 1.8, cpuUsage: 42 },
    { time: '08:00', responseTime: 95, throughput: 680, errorRate: 1.2, cpuUsage: 55 },
    { time: '12:00', responseTime: 105, throughput: 720, errorRate: 1.5, cpuUsage: 62 },
    { time: '16:00', responseTime: 115, throughput: 650, errorRate: 2.0, cpuUsage: 58 },
    { time: '20:00', responseTime: 100, throughput: 580, errorRate: 1.6, cpuUsage: 48 },
  ];

  const agentEfficiencyData = [
    { name: 'Agent-Alpha', efficiency: 95, tasksCompleted: 142, avgResponseTime: 85 },
    { name: 'Agent-Beta', efficiency: 92, tasksCompleted: 138, avgResponseTime: 92 },
    { name: 'Agent-Gamma', efficiency: 88, tasksCompleted: 125, avgResponseTime: 98 },
    { name: 'Agent-Delta', efficiency: 91, tasksCompleted: 134, avgResponseTime: 88 },
    { name: 'Agent-Epsilon', efficiency: 87, tasksCompleted: 121, avgResponseTime: 102 },
  ];

  const workflowSuccessData = [
    { type: 'Pipeline', success: 94, total: 150 },
    { type: 'Parallel', success: 96, total: 125 },
    { type: 'Brainstorming', success: 89, total: 85 },
    { type: 'Analysis', success: 92, total: 110 },
    { type: 'QA Review', success: 98, total: 75 },
  ];

  const memoryUsageData = [
    { time: '00:00', working: 1.2, longTerm: 4.8, vector: 2.1, total: 8.1 },
    { time: '04:00', working: 1.1, longTerm: 4.9, vector: 2.2, total: 8.2 },
    { time: '08:00', working: 1.5, longTerm: 5.1, vector: 2.3, total: 8.9 },
    { time: '12:00', working: 1.8, longTerm: 5.3, vector: 2.4, total: 9.5 },
    { time: '16:00', working: 1.6, longTerm: 5.2, vector: 2.3, total: 9.1 },
    { time: '20:00', working: 1.3, longTerm: 5.0, vector: 2.2, total: 8.5 },
  ];

  const colors = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0'];

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600, color: 'text.primary' }}>
          System Analytics
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              label="Time Range"
            >
              <MenuItem value="1h">Last Hour</MenuItem>
              <MenuItem value="24h">Last 24h</MenuItem>
              <MenuItem value="7d">Last 7 days</MenuItem>
              <MenuItem value="30d">Last 30 days</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
          >
            Export Report
          </Button>
        </Box>
      </Box>

      {/* Key Metrics Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Avg Response Time
                  </Typography>
                  <Typography variant="h4" component="div" color="primary.main">
                    104ms
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    ↓ 12% from yesterday
                  </Typography>
                </Box>
                <PerformanceIcon sx={{ fontSize: 40, color: 'primary.main' }} />
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
                    System Throughput
                  </Typography>
                  <Typography variant="h4" component="div" color="success.main">
                    615/min
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    ↑ 8% from yesterday
                  </Typography>
                </Box>
                <TrendingUpIcon sx={{ fontSize: 40, color: 'success.main' }} />
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
                    Error Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="warning.main">
                    1.7%
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    ↓ 0.3% from yesterday
                  </Typography>
                </Box>
                <ReportIcon sx={{ fontSize: 40, color: 'warning.main' }} />
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
                    System Uptime
                  </Typography>
                  <Typography variant="h4" component="div" color="success.main">
                    99.8%
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    ↑ 0.1% from yesterday
                  </Typography>
                </Box>
                <TimelineIcon sx={{ fontSize: 40, color: 'success.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Analytics Tabs */}
      <Card>
        <CardContent>
          <Tabs value={selectedTab} onChange={handleTabChange} aria-label="analytics tabs">
            <Tab label="Performance" />
            <Tab label="Agent Efficiency" />
            <Tab label="Workflow Success" />
            <Tab label="Memory Usage" />
            <Tab label="Reports" />
          </Tabs>

          {/* Performance Tab */}
          <TabPanel value={selectedTab} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Typography variant="h6" sx={{ mb: 2 }}>System Performance Over Time</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <ComposedChart data={performanceMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="left" dataKey="throughput" fill="#2196f3" name="Throughput (req/min)" />
                    <Line yAxisId="right" type="monotone" dataKey="responseTime" stroke="#4caf50" name="Response Time (ms)" />
                    <Line yAxisId="right" type="monotone" dataKey="errorRate" stroke="#f44336" name="Error Rate (%)" />
                  </ComposedChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} lg={4}>
                <Typography variant="h6" sx={{ mb: 2 }}>CPU Usage Distribution</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={performanceMetrics.map((item, index) => ({
                        name: item.time,
                        value: item.cpuUsage,
                        fill: colors[index % colors.length]
                      }))}
                      cx="50%"
                      cy="50%"
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                      label={({ name, value }) => `${name}: ${value}%`}
                    >
                      {performanceMetrics.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Agent Efficiency Tab */}
          <TabPanel value={selectedTab} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Typography variant="h6" sx={{ mb: 2 }}>Agent Efficiency Comparison</Typography>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={agentEfficiencyData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="efficiency" fill="#2196f3" name="Efficiency %" />
                    <Bar dataKey="avgResponseTime" fill="#ff9800" name="Avg Response Time (ms)" />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={12} lg={4}>
                <Typography variant="h6" sx={{ mb: 2 }}>Agent Performance Table</Typography>
                <TableContainer component={Paper}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Agent</TableCell>
                        <TableCell>Tasks</TableCell>
                        <TableCell>Efficiency</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {agentEfficiencyData.map((agent) => (
                        <TableRow key={agent.name}>
                          <TableCell>{agent.name}</TableCell>
                          <TableCell>{agent.tasksCompleted}</TableCell>
                          <TableCell>
                            <Chip
                              label={`${agent.efficiency}%`}
                              color={agent.efficiency > 90 ? 'success' : 'warning'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Workflow Success Tab */}
          <TabPanel value={selectedTab} index={2}>
            <Typography variant="h6" sx={{ mb: 2 }}>Workflow Success Rates by Type</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={workflowSuccessData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" />
                <YAxis />
                <Tooltip
                  formatter={(value, name) => [
                    name === 'success' ? `${value}%` : value,
                    name === 'success' ? 'Success Rate' : 'Total Workflows'
                  ]}
                />
                <Legend />
                <Bar dataKey="success" fill="#4caf50" name="Success Rate %" />
                <Bar dataKey="total" fill="#2196f3" name="Total Workflows" />
              </BarChart>
            </ResponsiveContainer>
          </TabPanel>

          {/* Memory Usage Tab */}
          <TabPanel value={selectedTab} index={3}>
            <Typography variant="h6" sx={{ mb: 2 }}>Memory Usage Over Time (GB)</Typography>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={memoryUsageData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="working" stackId="1" stroke="#2196f3" fill="#2196f3" name="Working Memory" />
                <Area type="monotone" dataKey="longTerm" stackId="1" stroke="#4caf50" fill="#4caf50" name="Long-term Memory" />
                <Area type="monotone" dataKey="vector" stackId="1" stroke="#ff9800" fill="#ff9800" name="Vector Memory" />
              </AreaChart>
            </ResponsiveContainer>
          </TabPanel>

          {/* Reports Tab */}
          <TabPanel value={selectedTab} index={4}>
            <Typography variant="h6" sx={{ mb: 2 }}>Available Reports</Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Daily Performance Report
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Comprehensive daily overview of system performance, agent efficiency, and workflow success rates.
                    </Typography>
                    <Button variant="contained" startIcon={<DownloadIcon />}>
                      Download PDF
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Agent Efficiency Analysis
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Detailed analysis of individual agent performance with optimization recommendations.
                    </Typography>
                    <Button variant="contained" startIcon={<DownloadIcon />}>
                      Download CSV
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Memory Usage Report
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Memory consumption patterns and optimization opportunities across all memory layers.
                    </Typography>
                    <Button variant="contained" startIcon={<DownloadIcon />}>
                      Download Excel
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Custom Analytics Report
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Generate custom reports with specific metrics and time ranges tailored to your needs.
                    </Typography>
                    <Button variant="outlined" startIcon={<ReportIcon />}>
                      Configure Report
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Analytics;
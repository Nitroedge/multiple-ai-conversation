import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  IconButton,
  Button,
  Switch,
  FormControlLabel,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  Paper,
  Chip,
  Slider,
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
  Memory as MemoryIcon,
  Api as ApiIcon,
  Save as SaveIcon,
  Restore as RestoreIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { useNotification } from '../contexts/NotificationContext';

const Settings = () => {
  const { addNotification } = useNotification();

  const [selectedTab, setSelectedTab] = useState(0);
  const [openDialog, setOpenDialog] = useState(false);
  const [dialogType, setDialogType] = useState('');

  // General Settings
  const [generalSettings, setGeneralSettings] = useState({
    systemName: 'Multi-Agent Conversation Engine',
    maxConcurrentAgents: 50,
    defaultTimeout: 30,
    enableLogging: true,
    logLevel: 'info',
    autoSave: true,
    autoSaveInterval: 300,
  });

  // Security Settings
  const [securitySettings, setSecuritySettings] = useState({
    enableAuthentication: true,
    sessionTimeout: 3600,
    maxLoginAttempts: 5,
    enableRateLimit: true,
    rateLimitRequests: 1000,
    rateLimitWindow: 3600,
    enableEncryption: true,
  });

  // Notification Settings
  const [notificationSettings, setNotificationSettings] = useState({
    enableEmailNotifications: true,
    enableSmsNotifications: false,
    enablePushNotifications: true,
    notifyOnErrors: true,
    notifyOnConflicts: true,
    notifyOnWorkflowCompletion: true,
    notifyOnAgentStatus: false,
    emailAddress: 'admin@example.com',
    smsNumber: '',
  });

  // Memory Settings
  const [memorySettings, setMemorySettings] = useState({
    workingMemorySize: 1000,
    longTermMemoryRetention: 30,
    vectorMemoryDimensions: 1536,
    enableMemoryOptimization: true,
    memoryCleanupInterval: 3600,
    maxMemoryUsage: 85,
  });

  // API Keys
  const [apiKeys, setApiKeys] = useState([
    { id: 1, name: 'OpenAI API Key', service: 'OpenAI', masked: '••••••••••••sk-abc123', active: true },
    { id: 2, name: 'Claude API Key', service: 'Anthropic', masked: '••••••••••••sk-ant-456', active: true },
    { id: 3, name: 'ElevenLabs API Key', service: 'ElevenLabs', masked: '••••••••••••el-789def', active: false },
  ]);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleSaveSettings = async () => {
    try {
      // API call to save settings would go here
      addNotification('success', 'Settings saved successfully');
    } catch (error) {
      addNotification('error', 'Failed to save settings');
    }
  };

  const handleExportSettings = () => {
    const settings = {
      general: generalSettings,
      security: securitySettings,
      notifications: notificationSettings,
      memory: memorySettings,
    };

    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'system-settings.json';
    a.click();
    URL.revokeObjectURL(url);

    addNotification('success', 'Settings exported successfully');
  };

  const handleImportSettings = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const settings = JSON.parse(e.target.result);
            if (settings.general) setGeneralSettings(settings.general);
            if (settings.security) setSecuritySettings(settings.security);
            if (settings.notifications) setNotificationSettings(settings.notifications);
            if (settings.memory) setMemorySettings(settings.memory);
            addNotification('success', 'Settings imported successfully');
          } catch (error) {
            addNotification('error', 'Invalid settings file');
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
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
          System Settings
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
            onClick={handleImportSettings}
          >
            Import
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExportSettings}
          >
            Export
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSaveSettings}
          >
            Save All
          </Button>
        </Box>
      </Box>

      <Card>
        <CardContent>
          <Tabs value={selectedTab} onChange={handleTabChange} aria-label="settings tabs">
            <Tab label="General" />
            <Tab label="Security" />
            <Tab label="Notifications" />
            <Tab label="Memory" />
            <Tab label="API Keys" />
          </Tabs>

          {/* General Settings Tab */}
          <TabPanel value={selectedTab} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="System Name"
                  value={generalSettings.systemName}
                  onChange={(e) => setGeneralSettings({ ...generalSettings, systemName: e.target.value })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
                <TextField
                  label="Max Concurrent Agents"
                  type="number"
                  value={generalSettings.maxConcurrentAgents}
                  onChange={(e) => setGeneralSettings({ ...generalSettings, maxConcurrentAgents: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
                <TextField
                  label="Default Timeout (seconds)"
                  type="number"
                  value={generalSettings.defaultTimeout}
                  onChange={(e) => setGeneralSettings({ ...generalSettings, defaultTimeout: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth sx={{ mb: 3 }}>
                  <InputLabel>Log Level</InputLabel>
                  <Select
                    value={generalSettings.logLevel}
                    onChange={(e) => setGeneralSettings({ ...generalSettings, logLevel: e.target.value })}
                  >
                    <MenuItem value="debug">Debug</MenuItem>
                    <MenuItem value="info">Info</MenuItem>
                    <MenuItem value="warning">Warning</MenuItem>
                    <MenuItem value="error">Error</MenuItem>
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={
                    <Switch
                      checked={generalSettings.enableLogging}
                      onChange={(e) => setGeneralSettings({ ...generalSettings, enableLogging: e.target.checked })}
                    />
                  }
                  label="Enable Logging"
                  sx={{ mb: 2, display: 'block' }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={generalSettings.autoSave}
                      onChange={(e) => setGeneralSettings({ ...generalSettings, autoSave: e.target.checked })}
                    />
                  }
                  label="Auto Save"
                  sx={{ mb: 2, display: 'block' }}
                />
                <TextField
                  label="Auto Save Interval (seconds)"
                  type="number"
                  value={generalSettings.autoSaveInterval}
                  onChange={(e) => setGeneralSettings({ ...generalSettings, autoSaveInterval: parseInt(e.target.value) })}
                  fullWidth
                  disabled={!generalSettings.autoSave}
                />
              </Grid>
            </Grid>
          </TabPanel>

          {/* Security Settings Tab */}
          <TabPanel value={selectedTab} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={securitySettings.enableAuthentication}
                      onChange={(e) => setSecuritySettings({ ...securitySettings, enableAuthentication: e.target.checked })}
                    />
                  }
                  label="Enable Authentication"
                  sx={{ mb: 2, display: 'block' }}
                />
                <TextField
                  label="Session Timeout (seconds)"
                  type="number"
                  value={securitySettings.sessionTimeout}
                  onChange={(e) => setSecuritySettings({ ...securitySettings, sessionTimeout: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
                <TextField
                  label="Max Login Attempts"
                  type="number"
                  value={securitySettings.maxLoginAttempts}
                  onChange={(e) => setSecuritySettings({ ...securitySettings, maxLoginAttempts: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={securitySettings.enableRateLimit}
                      onChange={(e) => setSecuritySettings({ ...securitySettings, enableRateLimit: e.target.checked })}
                    />
                  }
                  label="Enable Rate Limiting"
                  sx={{ mb: 2, display: 'block' }}
                />
                <TextField
                  label="Rate Limit (requests)"
                  type="number"
                  value={securitySettings.rateLimitRequests}
                  onChange={(e) => setSecuritySettings({ ...securitySettings, rateLimitRequests: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                  disabled={!securitySettings.enableRateLimit}
                />
                <TextField
                  label="Rate Limit Window (seconds)"
                  type="number"
                  value={securitySettings.rateLimitWindow}
                  onChange={(e) => setSecuritySettings({ ...securitySettings, rateLimitWindow: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                  disabled={!securitySettings.enableRateLimit}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={securitySettings.enableEncryption}
                      onChange={(e) => setSecuritySettings({ ...securitySettings, enableEncryption: e.target.checked })}
                    />
                  }
                  label="Enable Data Encryption"
                  sx={{ mb: 2, display: 'block' }}
                />
              </Grid>
            </Grid>
          </TabPanel>

          {/* Notifications Settings Tab */}
          <TabPanel value={selectedTab} index={2}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Notification Channels</Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.enableEmailNotifications}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, enableEmailNotifications: e.target.checked })}
                    />
                  }
                  label="Email Notifications"
                  sx={{ mb: 2, display: 'block' }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.enableSmsNotifications}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, enableSmsNotifications: e.target.checked })}
                    />
                  }
                  label="SMS Notifications"
                  sx={{ mb: 2, display: 'block' }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.enablePushNotifications}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, enablePushNotifications: e.target.checked })}
                    />
                  }
                  label="Push Notifications"
                  sx={{ mb: 2, display: 'block' }}
                />
                <TextField
                  label="Email Address"
                  value={notificationSettings.emailAddress}
                  onChange={(e) => setNotificationSettings({ ...notificationSettings, emailAddress: e.target.value })}
                  fullWidth
                  sx={{ mb: 3 }}
                  disabled={!notificationSettings.enableEmailNotifications}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Notification Types</Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.notifyOnErrors}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, notifyOnErrors: e.target.checked })}
                    />
                  }
                  label="System Errors"
                  sx={{ mb: 2, display: 'block' }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.notifyOnConflicts}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, notifyOnConflicts: e.target.checked })}
                    />
                  }
                  label="Agent Conflicts"
                  sx={{ mb: 2, display: 'block' }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.notifyOnWorkflowCompletion}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, notifyOnWorkflowCompletion: e.target.checked })}
                    />
                  }
                  label="Workflow Completion"
                  sx={{ mb: 2, display: 'block' }}
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={notificationSettings.notifyOnAgentStatus}
                      onChange={(e) => setNotificationSettings({ ...notificationSettings, notifyOnAgentStatus: e.target.checked })}
                    />
                  }
                  label="Agent Status Changes"
                  sx={{ mb: 2, display: 'block' }}
                />
              </Grid>
            </Grid>
          </TabPanel>

          {/* Memory Settings Tab */}
          <TabPanel value={selectedTab} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Memory Configuration</Typography>
                <Typography gutterBottom>Working Memory Size (MB)</Typography>
                <Slider
                  value={memorySettings.workingMemorySize}
                  onChange={(e, value) => setMemorySettings({ ...memorySettings, workingMemorySize: value })}
                  min={100}
                  max={5000}
                  valueLabelDisplay="auto"
                  sx={{ mb: 3 }}
                />
                <Typography gutterBottom>Long-term Memory Retention (days)</Typography>
                <Slider
                  value={memorySettings.longTermMemoryRetention}
                  onChange={(e, value) => setMemorySettings({ ...memorySettings, longTermMemoryRetention: value })}
                  min={1}
                  max={365}
                  valueLabelDisplay="auto"
                  sx={{ mb: 3 }}
                />
                <TextField
                  label="Vector Memory Dimensions"
                  type="number"
                  value={memorySettings.vectorMemoryDimensions}
                  onChange={(e) => setMemorySettings({ ...memorySettings, vectorMemoryDimensions: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" sx={{ mb: 2 }}>Memory Optimization</Typography>
                <FormControlLabel
                  control={
                    <Switch
                      checked={memorySettings.enableMemoryOptimization}
                      onChange={(e) => setMemorySettings({ ...memorySettings, enableMemoryOptimization: e.target.checked })}
                    />
                  }
                  label="Enable Memory Optimization"
                  sx={{ mb: 2, display: 'block' }}
                />
                <TextField
                  label="Memory Cleanup Interval (seconds)"
                  type="number"
                  value={memorySettings.memoryCleanupInterval}
                  onChange={(e) => setMemorySettings({ ...memorySettings, memoryCleanupInterval: parseInt(e.target.value) })}
                  fullWidth
                  sx={{ mb: 3 }}
                />
                <Typography gutterBottom>Max Memory Usage (%)</Typography>
                <Slider
                  value={memorySettings.maxMemoryUsage}
                  onChange={(e, value) => setMemorySettings({ ...memorySettings, maxMemoryUsage: value })}
                  min={50}
                  max={95}
                  valueLabelDisplay="auto"
                  sx={{ mb: 3 }}
                />
              </Grid>
            </Grid>
          </TabPanel>

          {/* API Keys Tab */}
          <TabPanel value={selectedTab} index={4}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6">API Key Management</Typography>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={() => {
                  setDialogType('add');
                  setOpenDialog(true);
                }}
              >
                Add API Key
              </Button>
            </Box>
            <List>
              {apiKeys.map((key, index) => (
                <React.Fragment key={key.id}>
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle1">{key.name}</Typography>
                          <Chip
                            label={key.service}
                            size="small"
                            color="primary"
                          />
                          <Chip
                            label={key.active ? 'Active' : 'Inactive'}
                            size="small"
                            color={key.active ? 'success' : 'default'}
                          />
                        </Box>
                      }
                      secondary={key.masked}
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        size="small"
                        onClick={() => {
                          setDialogType('edit');
                          setOpenDialog(true);
                        }}
                      >
                        <EditIcon />
                      </IconButton>
                      <IconButton size="small">
                        <DeleteIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < apiKeys.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </TabPanel>
        </CardContent>
      </Card>

      {/* API Key Dialog */}
      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          {dialogType === 'add' ? 'Add API Key' : 'Edit API Key'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 2 }}>
            <TextField label="Key Name" fullWidth />
            <FormControl fullWidth>
              <InputLabel>Service</InputLabel>
              <Select defaultValue="">
                <MenuItem value="OpenAI">OpenAI</MenuItem>
                <MenuItem value="Anthropic">Anthropic</MenuItem>
                <MenuItem value="ElevenLabs">ElevenLabs</MenuItem>
                <MenuItem value="Other">Other</MenuItem>
              </Select>
            </FormControl>
            <TextField label="API Key" type="password" fullWidth />
            <FormControlLabel
              control={<Switch defaultChecked />}
              label="Active"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button variant="contained">
            {dialogType === 'add' ? 'Add' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Settings;
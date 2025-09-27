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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Paper,
  TextField,
  Divider,
  Avatar,
  Badge,
  Tab,
  Tabs,
  TabPanel,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Send as SendIcon,
  Group as GroupIcon,
  Person as PersonIcon,
  Notifications as NotificationIcon,
  Message as MessageIcon,
  Broadcast as BroadcastIcon,
  Reply as ReplyIcon,
  MoreVert as MoreIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import { useAgent } from '../contexts/AgentContext';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useNotification } from '../contexts/NotificationContext';
import moment from 'moment';

const Communication = () => {
  const { agents } = useAgent();
  const { connected, messages } = useWebSocket();
  const { addNotification } = useNotification();

  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [messageText, setMessageText] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [messageFilter, setMessageFilter] = useState('all');

  const [agentMessages, setAgentMessages] = useState([
    {
      id: 1,
      from: 'Agent-Alpha',
      to: 'Agent-Beta',
      content: 'Task coordination request for workflow #123',
      timestamp: moment().subtract(5, 'minutes').toISOString(),
      type: 'request',
      status: 'delivered'
    },
    {
      id: 2,
      from: 'Agent-Beta',
      to: 'Agent-Alpha',
      content: 'Acknowledged. Ready to proceed with task execution.',
      timestamp: moment().subtract(3, 'minutes').toISOString(),
      type: 'response',
      status: 'delivered'
    },
    {
      id: 3,
      from: 'System',
      to: 'All Agents',
      content: 'Memory cleanup scheduled for 2:00 AM UTC',
      timestamp: moment().subtract(1, 'hour').toISOString(),
      type: 'broadcast',
      status: 'delivered'
    }
  ]);

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
  };

  const handleSendMessage = () => {
    if (!messageText.trim() || !selectedAgent) return;

    const newMessage = {
      id: Date.now(),
      from: 'User',
      to: selectedAgent.name,
      content: messageText,
      timestamp: moment().toISOString(),
      type: 'message',
      status: 'sending'
    };

    setAgentMessages([newMessage, ...agentMessages]);
    setMessageText('');
    addNotification('success', 'Message sent successfully');
  };

  const handleBroadcastMessage = () => {
    if (!messageText.trim()) return;

    const newMessage = {
      id: Date.now(),
      from: 'User',
      to: 'All Agents',
      content: messageText,
      timestamp: moment().toISOString(),
      type: 'broadcast',
      status: 'sending'
    };

    setAgentMessages([newMessage, ...agentMessages]);
    setMessageText('');
    addNotification('success', 'Broadcast message sent');
  };

  const getMessageTypeColor = (type) => {
    switch (type) {
      case 'request': return 'primary';
      case 'response': return 'success';
      case 'broadcast': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'delivered': return 'success';
      case 'sending': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const filteredMessages = agentMessages.filter(msg => {
    const matchesSearch = msg.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         msg.from.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         msg.to.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesFilter = messageFilter === 'all' || msg.type === messageFilter;

    return matchesSearch && matchesFilter;
  });

  const TabPanel = ({ children, value, index, ...other }) => (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`communication-tabpanel-${index}`}
      aria-labelledby={`communication-tab-${index}`}
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
          Agent Communication
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Badge badgeContent={connected ? 'Online' : 'Offline'} color={connected ? 'success' : 'error'}>
            <ChatIcon />
          </Badge>
        </Box>
      </Box>

      {/* Communication Statistics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Messages
                  </Typography>
                  <Typography variant="h4" component="div">
                    {agentMessages.length}
                  </Typography>
                </Box>
                <MessageIcon sx={{ fontSize: 40, color: 'primary.main' }} />
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
                    Active Agents
                  </Typography>
                  <Typography variant="h4" component="div" color="success.main">
                    {agents?.filter(a => a.status === 'active').length || 0}
                  </Typography>
                </Box>
                <GroupIcon sx={{ fontSize: 40, color: 'success.main' }} />
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
                    Broadcasts Today
                  </Typography>
                  <Typography variant="h4" component="div" color="warning.main">
                    {agentMessages.filter(m => m.type === 'broadcast').length}
                  </Typography>
                </Box>
                <BroadcastIcon sx={{ fontSize: 40, color: 'warning.main' }} />
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
                    Response Rate
                  </Typography>
                  <Typography variant="h4" component="div" color="primary.main">
                    96%
                  </Typography>
                </Box>
                <ReplyIcon sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Communication Tabs */}
      <Card>
        <CardContent>
          <Tabs value={selectedTab} onChange={handleTabChange} aria-label="communication tabs">
            <Tab label="Message Center" />
            <Tab label="Direct Messages" />
            <Tab label="Broadcasts" />
            <Tab label="System Notifications" />
          </Tabs>

          {/* Message Center Tab */}
          <TabPanel value={selectedTab} index={0}>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                placeholder="Search messages..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                size="small"
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />
                }}
                sx={{ flexGrow: 1 }}
              />
              <Button
                variant="outlined"
                startIcon={<FilterIcon />}
                onClick={() => {
                  const filters = ['all', 'request', 'response', 'broadcast', 'error'];
                  const currentIndex = filters.indexOf(messageFilter);
                  setMessageFilter(filters[(currentIndex + 1) % filters.length]);
                }}
              >
                Filter: {messageFilter}
              </Button>
            </Box>

            <List>
              {filteredMessages.map((message, index) => (
                <React.Fragment key={message.id}>
                  <ListItem>
                    <ListItemIcon>
                      <Avatar sx={{ bgcolor: getMessageTypeColor(message.type) + '.main' }}>
                        {message.from.charAt(0)}
                      </Avatar>
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle2">
                            {message.from} → {message.to}
                          </Typography>
                          <Chip
                            label={message.type}
                            size="small"
                            color={getMessageTypeColor(message.type)}
                          />
                          <Chip
                            label={message.status}
                            size="small"
                            color={getStatusColor(message.status)}
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.primary" sx={{ mb: 0.5 }}>
                            {message.content}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {moment(message.timestamp).fromNow()}
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <IconButton size="small">
                        <MoreIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < filteredMessages.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </TabPanel>

          {/* Direct Messages Tab */}
          <TabPanel value={selectedTab} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" sx={{ mb: 2 }}>Select Agent</Typography>
                <List>
                  {agents?.map((agent) => (
                    <ListItem
                      key={agent.id}
                      button
                      selected={selectedAgent?.id === agent.id}
                      onClick={() => setSelectedAgent(agent)}
                    >
                      <ListItemIcon>
                        <Badge
                          color={agent.status === 'active' ? 'success' : 'default'}
                          variant="dot"
                        >
                          <Avatar>{agent.name?.charAt(0) || 'A'}</Avatar>
                        </Badge>
                      </ListItemIcon>
                      <ListItemText
                        primary={agent.name}
                        secondary={agent.status}
                      />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              <Grid item xs={12} md={8}>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  {selectedAgent ? `Message ${selectedAgent.name}` : 'Select an agent to send a message'}
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <TextField
                    placeholder="Type your message..."
                    value={messageText}
                    onChange={(e) => setMessageText(e.target.value)}
                    fullWidth
                    multiline
                    rows={3}
                    disabled={!selectedAgent}
                  />
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button
                      variant="contained"
                      startIcon={<SendIcon />}
                      onClick={handleSendMessage}
                      disabled={!selectedAgent || !messageText.trim()}
                    >
                      Send
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<BroadcastIcon />}
                      onClick={handleBroadcastMessage}
                      disabled={!messageText.trim()}
                    >
                      Broadcast
                    </Button>
                  </Box>
                </Box>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Broadcasts Tab */}
          <TabPanel value={selectedTab} index={2}>
            <Typography variant="h6" sx={{ mb: 2 }}>Broadcast Messages</Typography>
            <List>
              {agentMessages.filter(m => m.type === 'broadcast').map((message) => (
                <ListItem key={message.id}>
                  <ListItemIcon>
                    <BroadcastIcon color="warning" />
                  </ListItemIcon>
                  <ListItemText
                    primary={message.content}
                    secondary={`From: ${message.from} • ${moment(message.timestamp).fromNow()}`}
                  />
                </ListItem>
              ))}
            </List>
          </TabPanel>

          {/* System Notifications Tab */}
          <TabPanel value={selectedTab} index={3}>
            <Typography variant="h6" sx={{ mb: 2 }}>System Notifications</Typography>
            <List>
              {agentMessages.filter(m => m.from === 'System').map((message) => (
                <ListItem key={message.id}>
                  <ListItemIcon>
                    <NotificationIcon color="info" />
                  </ListItemIcon>
                  <ListItemText
                    primary={message.content}
                    secondary={moment(message.timestamp).fromNow()}
                  />
                </ListItem>
              ))}
            </List>
          </TabPanel>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Communication;
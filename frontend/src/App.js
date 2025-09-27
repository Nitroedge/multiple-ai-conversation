import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Components
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import Dashboard from './pages/Dashboard';
import AgentCoordination from './pages/AgentCoordination';
import ConflictResolution from './pages/ConflictResolution';
import CollaborationWorkflows from './pages/CollaborationWorkflows';
import Communication from './pages/Communication';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

// Contexts
import { WebSocketProvider } from './contexts/WebSocketContext';
import { AgentProvider } from './contexts/AgentContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#f50057',
      light: '#ff5983',
      dark: '#c51162',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
});

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentPage, setCurrentPage] = useState('dashboard');

  const handleSidebarToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <WebSocketProvider>
        <AgentProvider>
          <NotificationProvider>
            <Router>
              <Box sx={{ display: 'flex', minHeight: '100vh' }}>
                <Sidebar
                  open={sidebarOpen}
                  onToggle={handleSidebarToggle}
                  currentPage={currentPage}
                  onPageChange={setCurrentPage}
                />

                <Box
                  component="main"
                  sx={{
                    flexGrow: 1,
                    p: 3,
                    width: { sm: `calc(100% - ${sidebarOpen ? 280 : 73}px)` },
                    ml: { sm: `${sidebarOpen ? 280 : 73}px` },
                    transition: theme.transitions.create(['margin', 'width'], {
                      easing: theme.transitions.easing.sharp,
                      duration: theme.transitions.duration.leavingScreen,
                    }),
                  }}
                >
                  <Header onSidebarToggle={handleSidebarToggle} />

                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/agents" element={<AgentCoordination />} />
                    <Route path="/conflicts" element={<ConflictResolution />} />
                    <Route path="/workflows" element={<CollaborationWorkflows />} />
                    <Route path="/communication" element={<Communication />} />
                    <Route path="/analytics" element={<Analytics />} />
                    <Route path="/settings" element={<Settings />} />
                  </Routes>
                </Box>
              </Box>

              <ToastContainer
                position="top-right"
                autoClose={5000}
                hideProgressBar={false}
                newestOnTop={false}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="dark"
              />
            </Router>
          </NotificationProvider>
        </AgentProvider>
      </WebSocketProvider>
    </ThemeProvider>
  );
}

export default App;
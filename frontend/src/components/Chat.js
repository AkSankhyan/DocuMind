import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  IconButton,
  Avatar,
  Chip,
  CircularProgress,
  Divider,
  Card,
  CardContent,
  Alert,
  Tooltip,
  Fade,
  Zoom,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Send as SendIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
  FileUpload as UploadIcon,
  DeleteOutline as ClearIcon,
  History as HistoryIcon,
  ContentCopy as CopyIcon,
  Article as DocumentIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import { useFile } from '../contexts/FileContext';
import config from '../config';

// Message bubble animations
const messageVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.2 } },
};

// Typing indicator animation
const typingVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.3 } },
  exit: { opacity: 0, transition: { duration: 0.3 } },
};

// Dot animation for typing indicator
const dotVariants = {
  initial: { y: 0 },
  animate: {
    y: [0, -5, 0],
    transition: { duration: 0.6, repeat: Infinity, repeatType: 'loop' },
  },
};

const Chat = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { uploadedFile, collectionId, docStats, fileContent, fileType, chatHistory, addMessageToHistory, clearFile } = useFile();
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showDocumentInfo, setShowDocumentInfo] = useState(true);
  const endOfMessagesRef = useRef(null);
  const navigate = useNavigate();

  // Add an ID for the current loading session
  const [loadingSessionId, setLoadingSessionId] = useState(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (endOfMessagesRef.current) {
      endOfMessagesRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  // Handle submission of a new question — uses SSE streaming endpoint
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const questionText = input.trim();
    addMessageToHistory({ role: 'user', content: questionText });
    setInput('');

    const currentSessionId = Date.now().toString();
    setLoadingSessionId(currentSessionId);
    setIsLoading(true);
    setError(null);

    // Seed a streaming bot message placeholder
    const streamingMsg = { role: 'assistant', content: '', citations: [], _streaming: true };
    addMessageToHistory(streamingMsg);

    let accumulatedText = '';
    let finalCitations = [];

    try {
      const response = await fetch(`${config.API_BASE_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText, collection_id: collectionId || null }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const evt = JSON.parse(jsonStr);

            if (evt.token !== undefined) {
              accumulatedText += evt.token;
              // Update the last message (streaming placeholder) in-place
              addMessageToHistory({
                role: 'assistant',
                content: accumulatedText,
                citations: [],
                _streaming: true,
                _replace: true,       // signal context to replace last assistant msg
              });
            }

            if (evt.done) {
              finalCitations = evt.citations || [];
              // Finalize the message — remove _streaming flag, attach citations
              addMessageToHistory({
                role: 'assistant',
                content: accumulatedText,
                citations: finalCitations,
                _streaming: false,
                _replace: true,
              });
            }

            if (evt.error) {
              throw new Error(evt.error);
            }
          } catch (parseErr) {
            // skip malformed SSE lines
          }
        }
      }
    } catch (err) {
      console.error('Stream error:', err);
      const errorMessage = err.message || 'Failed to get an answer';
      setError(errorMessage);
      addMessageToHistory({
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMessage}`,
        isError: true,
        _replace: accumulatedText === '',  // only replace placeholder if nothing was streamed
      });
    } finally {
      setIsLoading(false);
      setLoadingSessionId(null);
    }
  };

  const handleClear = () => {
    clearFile();
    navigate('/upload');
  };

  const handleCopy = (text) => {
    navigator.clipboard.writeText(text);
  };

  const toggleDocumentInfo = () => {
    setShowDocumentInfo(!showDocumentInfo);
  };

  // If no file is uploaded, redirect to upload page
  if (!uploadedFile) {
    return (
      <Container maxWidth="md" sx={{ py: 6, textAlign: 'center' }}>
        <Card elevation={3} sx={{ borderRadius: 3, p: 4 }}>
          <CardContent>
            <Typography variant="h5" component="h1" gutterBottom>
              No Document Loaded
            </Typography>
            <Typography variant="body1" color="textSecondary" paragraph>
              Please upload a document first to start chatting
            </Typography>
            <Button
              variant="contained"
              color="primary"
              startIcon={<UploadIcon />}
              onClick={() => navigate('/upload')}
              sx={{ mt: 2 }}
            >
              Upload Document
            </Button>
          </CardContent>
        </Card>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 4, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 80px)' }}>
      {/* Document Info Header */}
      <Fade in={true} timeout={500}>
        <Paper elevation={3} sx={{ 
          p: 2, 
          display: 'flex', 
          alignItems: 'center',
          justifyContent: 'space-between',
          borderRadius: 2,
          mb: 2,
          bgcolor: 'background.paper',
          position: 'sticky',
          top: 0,
          zIndex: 1
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar sx={{ bgcolor: 'primary.main' }}>
              <DocumentIcon />
            </Avatar>
            <Box>
              <Typography variant="subtitle1" fontWeight={600} noWrap>
                {uploadedFile?.name}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                {uploadedFile?.size ? `${(uploadedFile.size / 1024).toFixed(1)} KB` : ''}
                {docStats?.numPages ? ` • ${docStats.numPages} Page(s)` : ''}
                {docStats?.numChunks ? ` • ${docStats.numChunks} Vector Chunks` : ''}
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Toggle Document Info">
              <IconButton onClick={toggleDocumentInfo} size="small">
                {showDocumentInfo ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Tooltip>
            <Tooltip title="Clear Chat">
              <IconButton 
                color="error" 
                onClick={handleClear} 
                size="small"
              >
                <ClearIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Paper>
      </Fade>

      {/* Chat Messages Area */}
      <Paper 
        elevation={3} 
        sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          flex: 1,
          borderRadius: 2,
          overflow: 'hidden',
          bgcolor: 'background.default',
        }}
      >
        <Box 
          sx={{ 
            flex: 1, 
            overflowY: 'auto', 
            p: 2,
            display: 'flex',
            flexDirection: 'column',
            gap: 2
          }}
        >
          <AnimatePresence mode="sync">
            {chatHistory.map((message, index) => (
              <motion.div
                key={`message-${index}`}
                variants={messageVariants}
                initial="hidden"
                animate="visible"
                exit="exit"
                style={{ 
                  alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: isMobile ? '90%' : '70%'
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    gap: 1,
                    flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                  }}
                >
                  <Avatar
                    sx={{
                      bgcolor: message.role === 'user' ? 'secondary.main' : 'primary.main',
                    }}
                  >
                    {message.role === 'user' ? <PersonIcon /> : <BotIcon />}
                  </Avatar>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      bgcolor: message.role === 'user' ? 'secondary.light' : 'background.paper',
                      position: 'relative',
                    }}
                  >
                    {message.role === 'assistant' && (
                      <Tooltip title="Copy to Clipboard">
                        <IconButton
                          size="small"
                          sx={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            opacity: 0.6,
                            '&:hover': { opacity: 1 },
                          }}
                          onClick={() => handleCopy(message.content)}
                        >
                          <CopyIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Box sx={{ pr: message.role === 'assistant' ? 4 : 0 }}>
                      {message.role === 'assistant' ? (
                        <Box className="markdown-body">
                          <ReactMarkdown>{message.content}</ReactMarkdown>
                          {message.citations && message.citations.length > 0 && (
                            <Box sx={{ mt: 2, pt: 1.5, borderTop: '1px solid #e2e8f0' }}>
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  fontWeight: 600, 
                                  color: 'primary.main', 
                                  display: 'flex', 
                                  alignItems: 'center', 
                                  gap: 0.5,
                                  mb: 0.8
                                }}
                              >
                                📚 Grounded Sources & Citations:
                              </Typography>
                              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.8 }}>
                                {message.citations.map((cite, cIdx) => (
                                  <Tooltip 
                                    key={cIdx} 
                                    title={
                                      <Box sx={{ p: 0.5 }}>
                                        <Typography variant="caption" sx={{ fontWeight: 600 }}>Excerpt:</Typography>
                                        <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>{cite.snippet}</Typography>
                                      </Box>
                                    } 
                                    arrow 
                                    placement="top"
                                  >
                                    <Chip
                                      size="small"
                                      label={`${cite.source || 'Doc'} | Page ${cite.page || 1}`}
                                      color="primary"
                                      variant="outlined"
                                      sx={{ fontSize: '0.75rem', fontWeight: 500 }}
                                    />
                                  </Tooltip>
                                ))}
                              </Box>
                            </Box>
                          )}
                        </Box>
                      ) : (
                        <Typography>{message.content}</Typography>
                      )}
                    </Box>
                  </Paper>
                </Box>
              </motion.div>
            ))}

            {/* Typing indicator - only show if loading and not just added a response */}
            {isLoading && (
              <motion.div
                key={`typing-indicator-${loadingSessionId}`}
                variants={typingVariants}
                initial="initial"
                animate="animate"
                exit="exit"
                style={{ alignSelf: 'flex-start' }}
              >
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'flex-start', 
                  gap: 2,
                  mt: 2 
                }}>
                  <Avatar sx={{ bgcolor: 'primary.main' }}>
                    <BotIcon />
                  </Avatar>
                  <Paper
                    elevation={1}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1
                    }}
                  >
                    <Typography>Thinking</Typography>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      {[0, 1, 2].map((i) => (
                        <motion.span
                          key={i}
                          variants={dotVariants}
                          animate="animate"
                          style={{ 
                            display: 'inline-block', 
                            width: '5px',
                            height: '5px',
                            borderRadius: '50%',
                            backgroundColor: 'currentColor', 
                          }}
                          transition={{ delay: i * 0.15 }}
                        >
                          .
                        </motion.span>
                      ))}
                    </Box>
                  </Paper>
                </Box>
              </motion.div>
            )}

            {error && (
              <Zoom in={true}>
                <Alert 
                  severity="error" 
                  sx={{ mt: 2, borderRadius: 2 }}
                  onClose={() => setError(null)}
                >
                  {error}
                </Alert>
              </Zoom>
            )}

            <div ref={endOfMessagesRef} />
          </AnimatePresence>
        </Box>

        <Divider />

        {/* Input area */}
        <Box 
          component="form" 
          onSubmit={handleSubmit} 
          sx={{ 
            p: 2, 
            bgcolor: 'background.paper',
            display: 'flex',
            gap: 1
          }}
        >
          <TextField
            fullWidth
            placeholder="Ask a question about your document..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            variant="outlined"
            disabled={isLoading}
            multiline
            maxRows={4}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
              }
            }}
          />
          <Button
            type="submit"
            variant="contained"
            color="primary"
            disabled={!input.trim() || isLoading}
            sx={{ 
              borderRadius: 2,
              minWidth: '50px',
              height: '56px'
            }}
          >
            {isLoading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default Chat; 
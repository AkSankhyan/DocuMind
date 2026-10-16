import React, { createContext, useState, useContext } from 'react';

const FileContext = createContext();

export const useFile = () => useContext(FileContext);

export const FileProvider = ({ children }) => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [collectionId, setCollectionId] = useState(null);
  const [docStats, setDocStats] = useState({ numPages: 0, numChunks: 0 });
  const [fileContent, setFileContent] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const clearFile = () => {
    setUploadedFile(null);
    setCollectionId(null);
    setDocStats({ numPages: 0, numChunks: 0 });
    setFileContent(null);
    setFileType(null);
    setChatHistory([]);
  };

  const addMessageToHistory = (message) => {
    if (message._replace) {
      // Replace the last assistant message in-place (used for streaming token updates)
      setChatHistory(prev => {
        const updated = [...prev];
        for (let i = updated.length - 1; i >= 0; i--) {
          if (updated[i].role === 'assistant') {
            updated[i] = { ...message };
            return updated;
          }
        }
        // No assistant message found — just append
        return [...prev, { ...message }];
      });
    } else {
      setChatHistory(prev => [...prev, message]);
    }
  };

  const value = {
    uploadedFile,
    setUploadedFile,
    collectionId,
    setCollectionId,
    docStats,
    setDocStats,
    fileContent,
    setFileContent,
    fileType,
    setFileType,
    isProcessing,
    setIsProcessing,
    chatHistory,
    setChatHistory,
    clearFile,
    addMessageToHistory
  };

  return <FileContext.Provider value={value}>{children}</FileContext.Provider>;
};
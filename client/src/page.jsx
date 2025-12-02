import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  FaSmile,
  FaSadTear,
  FaAngry,
  FaSkullCrossbones,
  FaHeart,
  FaSurprise,
  FaPaperPlane,
  FaSpinner,
  FaBrain,
  FaTrash,
} from "react-icons/fa";
import "./page.css";

const EmotionAnalyzer = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const emotionConfig = {
    joy: { icon: <FaSmile />, color: "#FFD700", gradient: "linear-gradient(135deg, #FFD700 0%, #FFA500 100%)", label: "Joy" },
    sadness: { icon: <FaSadTear />, color: "#4682B4", gradient: "linear-gradient(135deg, #4682B4 0%, #1E3A5F 100%)", label: "Sadness" },
    anger: { icon: <FaAngry />, color: "#DC143C", gradient: "linear-gradient(135deg, #DC143C 0%, #8B0000 100%)", label: "Anger" },
    fear: { icon: <FaSkullCrossbones />, color: "#9370DB", gradient: "linear-gradient(135deg, #9370DB 0%, #4B0082 100%)", label: "Fear" },
    love: { icon: <FaHeart />, color: "#FF69B4", gradient: "linear-gradient(135deg, #FF69B4 0%, #FF1493 100%)", label: "Love" },
    surprise: { icon: <FaSurprise />, color: "#FF8C00", gradient: "linear-gradient(135deg, #FF8C00 0%, #FF4500 100%)", label: "Surprise" },
  };

  const analyzeEmotion = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text to analyze!", {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });
      return;
    }

    setResult(null);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:4000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      const data = await response.json();

      if (data.error) {
        toast.error(data.error, {
          position: "top-right",
          autoClose: 3000,
        });
      } else {
        setResult(data);
        setHistory([{ text, emotion: data.emotion, confidence: data.confidence, timestamp: new Date() }, ...history.slice(0, 4)]);
        toast.success(`Emotion detected: ${data.emotion.toUpperCase()}!`, {
          position: "top-right",
          autoClose: 3000,
          icon: emotionConfig[data.emotion]?.icon,
        });
      }
    } catch (err) {
      toast.error("Server not reachable. Make sure Node & Flask are running.", {
        position: "top-right",
        autoClose: 5000,
      });
    }
    setLoading(false);
  };

  const clearText = () => {
    setText("");
    setResult(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      analyzeEmotion();
    }
  };

  return (
    <div className="emotion-page">
      <ToastContainer />
      
      {/* Left Side - Header & Input */}
      <motion.div
        className="header"
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="header-icon">
          <FaBrain className="brain-icon" />
        </div>
        <h1 className="emotion-title">Emotion Detection AI</h1>
        <p className="emotion-subtitle">Powered by Deep Learning CNN Model</p>
      </motion.div>

      <motion.div
        className="input-section"
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
      >
        <div className="textarea-wrapper">
          <textarea
            className="emotion-textarea"
            rows="6"
            placeholder="Type or paste your text here... (Press Ctrl+Enter to analyze)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          {text && (
            <motion.button
              className="clear-btn"
              onClick={clearText}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <FaTrash />
            </motion.button>
          )}
        </div>

        <div className="char-count">
          {text.length} characters
        </div>

        <motion.button
          onClick={analyzeEmotion}
          className="emotion-button"
          disabled={loading}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          {loading ? (
            <>
              <FaSpinner className="spinner" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <FaPaperPlane />
              <span>Analyze Emotion</span>
            </>
          )}
        </motion.button>
      </motion.div>

      {/* Right Side - Results & History */}
      <motion.div
        className="right-side"
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        {/* Result Section */}
        <AnimatePresence>
          {result && (
            <motion.div
              className="emotion-result-box"
              initial={{ opacity: 0, scale: 0.8, y: 50 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: 50 }}
              transition={{ duration: 0.5, type: "spring" }}
              style={{
                background: emotionConfig[result.emotion]?.gradient,
              }}
            >
              <motion.div
                className="emotion-icon-large"
                initial={{ rotate: 0 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
              >
                {emotionConfig[result.emotion]?.icon}
              </motion.div>

              <h2 className="emotion-label">
                {emotionConfig[result.emotion]?.label}
              </h2>

              <div className="confidence-section">
                <div className="confidence-label">Confidence Score</div>
                <div className="confidence-value">
                  {(result.confidence * 100).toFixed(2)}%
                </div>
                <motion.div
                  className="confidence-bar-bg"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 0.5 }}
                >
                  <motion.div
                    className="confidence-bar"
                    initial={{ width: 0 }}
                    animate={{ width: `${result.confidence * 100}%` }}
                    transition={{ duration: 1, delay: 0.2 }}
                  />
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* History Section */}
        {history.length > 0 && (
          <motion.div
            className="history-section"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <h3 className="history-title">Recent Analyses</h3>
            <div className="history-list">
              {history.map((item, index) => (
                <motion.div
                  key={index}
                  className="history-item"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="history-emotion-icon" style={{ color: emotionConfig[item.emotion]?.color }}>
                    {emotionConfig[item.emotion]?.icon}
                  </div>
                  <div className="history-content">
                    <div className="history-text">{item.text.substring(0, 50)}...</div>
                    <div className="history-meta">
                      <span className="history-emotion">{emotionConfig[item.emotion]?.label}</span>
                      <span className="history-confidence">{(item.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Footer */}
      <motion.div
        className="footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <p>Built with TensorFlow, Flask, Express & React</p>
      </motion.div>
    </div>
  );
};

export default EmotionAnalyzer;

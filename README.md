# 🎭 Emotion Prediction Using CNN

A full-stack deep learning application that classifies emotions from text using a Convolutional Neural Network. The system analyzes user input and predicts one of six emotions: **anger**, **fear**, **joy**, **love**, **sadness**, or **surprise**.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Directory Structure](#-directory-structure)
- [Getting Started](#-getting-started)
- [API Endpoints](#-api-endpoints)
- [Model Performance](#-model-performance)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Overview

This project demonstrates end-to-end emotion detection from text using modern deep learning and web technologies. The system consists of three main components:

- **Machine Learning Backend** ([ml/app.py](ml/app.py)): Flask server serving a trained CNN model built with TensorFlow/Keras
- **Express Middleware** ([server/index.js](server/index.js)): Node.js proxy handling CORS and routing between frontend and ML service
- **React Frontend** ([client/src/](client/src/)): Interactive web UI built with Vite for real-time emotion analysis

## ✨ Features

- **Advanced Text Preprocessing**:
  - Lowercasing normalization
  - URL and special character removal
  - English stopword filtering using NLTK
  - Tokenization and sequence padding
- **CNN-Based Classification**:
  - 6-class emotion detection (anger, fear, joy, love, sadness, surprise)
  - 128-dimensional word embeddings
  - 1D Convolutional layers for feature extraction
  - Global max pooling for salient feature capture
- **REST API Integration**:
  - Flask endpoint: `POST /predict` (port 5000)
  - Express proxy: `POST /analyze` (port 4000)
  - JSON request/response format
- **Real-time Predictions**:
  - Confidence scores with each prediction
  - Sub-second response times
  - Error handling and input validation
- **Modular Architecture**:
  - Separation of concerns (ML, middleware, UI)
  - Easy to deploy and scale
  - CORS-enabled for cross-origin requests

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React UI  │─────▶│ Express      │─────▶│ Flask ML    │
│  (Port 5173)│      │ Middleware   │      │ Service     │
│             │◀─────│ (Port 4000)  │◀─────│ (Port 5000) │
└─────────────┘      └──────────────┘      └─────────────┘
     User Input           Proxy/CORS          CNN Model
```

## 🧠 Model Details

### Architecture

The emotion classification model ([ML_Emotion_detection.ipynb](ml/ML_Emotion_detection.ipynb)) is a Convolutional Neural Network with the following layers:

1. **Embedding Layer**:

   - Input: 10,000 word vocabulary
   - Output: 128-dimensional dense vectors
   - Sequence length: 100 tokens (padded/truncated)

2. **Conv1D Layer**:

   - 128 filters with kernel size 5
   - ReLU activation
   - Captures local n-gram patterns

3. **GlobalMaxPooling1D**:

   - Extracts most salient features across sequence
   - Reduces dimensionality

4. **Dense Layer**:

   - 64 units with ReLU activation
   - High-level feature learning

5. **Dropout Layer**:

   - Rate: 0.3
   - Prevents overfitting

6. **Output Layer**:
   - 6 units (one per emotion class)
   - Softmax activation for probability distribution

### Training Configuration

- **Loss Function**: Categorical cross-entropy
- **Optimizer**: Adam
- **Epochs**: 10
- **Batch Size**: 32
- **Validation Split**: Separate validation set (val.txt)

### Preprocessing Pipeline

Implemented in [`clean_text`](ml/app.py) function:

```python
1. Convert to lowercase
2. Remove URLs (regex: r'http\S+')
3. Remove non-alphabetic characters (regex: r'[^a-z\s]')
4. Filter English stopwords (NLTK)
5. Tokenize and convert to sequences
6. Pad sequences to length 100
```

## 📊 Dataset

The model is trained on three datasets:

- **train.txt**: 16,000 samples for training
- **val.txt**: Validation set for hyperparameter tuning
- **test.txt**: 2,000 samples for final evaluation

### Class Distribution (Training Set)

| Emotion  | Samples | Percentage |
| -------- | ------- | ---------- |
| Joy      | 5,362   | 33.5%      |
| Sadness  | 4,666   | 29.2%      |
| Anger    | 2,159   | 13.5%      |
| Fear     | 1,937   | 12.1%      |
| Love     | 1,304   | 8.2%       |
| Surprise | 572     | 3.6%       |

### Data Format

```csv
text;label
i didnt feel humiliated;sadness
i can go from feeling so hopeless to so damned hopeful;sadness
...
```

## 🗂️ Directory Structure

```
emotion_prediction_using_cnn/
├─ ml/                              # Machine Learning Backend
│  ├─ app.py                        # Flask API server
│  ├─ ML_Emotion_detection.ipynb   # Model training notebook
│  ├─ requirements.txt              # Python dependencies
│  ├─ .gitignore
│  └─ models/                       # Serialized model artifacts
│     ├─ emotion_cnn_model.keras   # Trained CNN model
│     ├─ tokenizer.pkl              # Fitted tokenizer
│     └─ label_encoder.pkl          # Emotion label encoder
│
├─ server/                          # Node.js Middleware
│  ├─ index.js                      # Express proxy server
│  ├─ package.json                  # Node dependencies
│  └─ .gitignore
│
├─ client/                          # React Frontend
│  ├─ src/
│  │  ├─ App.jsx                    # Main app component
│  │  ├─ App.css                    # App styles
│  │  ├─ page.jsx                   # Emotion analyzer component
│  │  ├─ page.css                   # Component styles
│  │  ├─ main.jsx                   # Entry point
│  │  └─ index.css                  # Global styles
│  ├─ public/                       # Static assets
│  ├─ index.html                    # HTML template
│  ├─ vite.config.js                # Vite configuration
│  ├─ package.json                  # Frontend dependencies
│  ├─ eslint.config.js              # ESLint rules
│  └─ .gitignore
│
├─ README.md                        # This file
└─ .gitignore                       # Root gitignore
```

## 🚀 Getting Started

### Prerequisites

- **Python** 3.8+ with pip
- **Node.js** 16+ with npm
- **Git**

### 1. Clone Repository

```powershell
git clone <repository-url> emotion_prediction_using_cnn
cd emotion_prediction_using_cnn
```

### 2. Start the ML Backend

Set up Python environment and install dependencies:

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Start the Flask server:

```powershell
python app.py
```

**Output:**

```
Loading model and preprocessing tools...
[nltk_data] Downloading package stopwords...
 * Running on http://0.0.0.0:5000
```

> **Note**: First run will download NLTK stopwords (~1MB)

### 3. Start the Node Proxy Server

Open a new terminal:

```powershell
cd server
npm install
node index.js
# For auto-reload during development: npx nodemon index.js
```

**Output:**

```
Express proxy running on http://localhost:4000
```

### 4. Start the React Frontend

Open a third terminal:

```powershell
cd client
npm install
npm run dev
```

**Output:**

```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
```

### 5. Use the Application

1. Open browser to `http://localhost:5173`
2. Type text in the textarea (e.g., "I am so happy today!")
3. Click **Analyze Emotion**
4. View predicted emotion and confidence score

## 🔌 API Endpoints

### Flask ML Service (Port 5000)

#### `GET /`

Health check endpoint.

**Response:**

```json
{
  "message": "Emotion Classification API is running!"
}
```

#### `POST /predict`

Predict emotion from text.

**Request:**

```json
{
  "text": "I am feeling so joyful and excited!"
}
```

**Response:**

```json
{
  "emotion": "joy",
  "confidence": 0.956
}
```

**Error Response:**

```json
{
  "error": "No text provided"
}
```

### Express Proxy (Port 4000)

#### `POST /analyze`

Forwards request to Flask `/predict` endpoint.

**Request:** Same as Flask `/predict`

**Response:** Same as Flask `/predict`

## 📈 Model Performance

### Test Set Accuracy: **91%**

### Detailed Classification Report

```
              precision    recall  f1-score   support

       anger       0.93      0.87      0.90       275
        fear       0.88      0.85      0.86       224
         joy       0.95      0.93      0.94       695
        love       0.80      0.86      0.83       159
     sadness       0.93      0.96      0.95       581
    surprise       0.71      0.82      0.76        66

    accuracy                           0.91      2000
   macro avg       0.87      0.88      0.87      2000
weighted avg       0.92      0.91      0.91      2000
```

### Confusion Matrix

The model shows strong performance across all classes, with highest accuracy on **joy** and **sadness** (most represented in training data). The **surprise** class has slightly lower precision due to limited training samples (572).

## 🧪 Testing

### Test the Flask API directly

```powershell
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"text": "I am so angry right now!"}'
```

**Expected Output:**

```json
{
  "confidence": 0.96,
  "emotion": "anger"
}
```

### Test the Express Proxy

```powershell
curl -X POST http://localhost:4000/analyze `
  -H "Content-Type: application/json" `
  -d '{"text": "This makes me so happy!"}'
```

### Frontend Testing

Use the web interface at `http://localhost:5173` to test various inputs:

- **Joy**: "I'm so excited about this opportunity!"
- **Sadness**: "I feel hopeless and alone"
- **Anger**: "This is absolutely infuriating"
- **Fear**: "I'm terrified of what might happen"
- **Love**: "I cherish every moment with you"
- **Surprise**: "I can't believe this happened!"

## ⚙️ Configuration

### Environment Variables

You can customize ports and model paths:

**Flask** ([ml/app.py](ml/app.py)):

```python
MODEL_PATH = "models/emotion_cnn_model.keras"
TOKENIZER_PATH = "models/tokenizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
PORT = 5000
```

**Express** ([server/index.js](server/index.js)):

```javascript
const PORT = 4000;
const FLASK_URL = "http://localhost:5000";
```

**Vite** ([client/vite.config.js](client/vite.config.js)):

```javascript
export default defineConfig({
  server: { port: 5173 },
});
```

### Model Hyperparameters

To retrain with different settings, modify [ML_Emotion_detection.ipynb](ml/ML_Emotion_detection.ipynb):

```python
# Tokenizer settings
num_words = 10000
maxlen = 100

# Model architecture
embedding_dim = 128
filters = 128
kernel_size = 5
dense_units = 64
dropout_rate = 0.3

# Training
epochs = 10
batch_size = 32
```

## 🛡️ Error Handling

The application handles various edge cases:

- **Empty input**: Frontend validation prevents empty submissions
- **Server unavailable**: Clear error message when Flask/Express is down
- **Invalid text**: Backend returns 400 with error message
- **Model errors**: 500 status with exception details (development mode)

Error display in UI ([page.jsx](client/src/page.jsx)):

```jsx
{
  error && <p className="emotion-error">{error}</p>;
}
```

## 📦 Dependencies

### Python ([ml/requirements.txt](ml/requirements.txt))

```
flask>=2.3.0
tensorflow>=2.13.0
numpy>=1.24.0
nltk>=3.8.0
scikit-learn>=1.3.0
```

### Node.js Backend ([server/package.json](server/package.json))

```json
{
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5",
    "node-fetch": "^3.3.0"
  }
}
```

### React Frontend ([client/package.json](client/package.json))

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "@vitejs/plugin-react": "^4.2.0"
  }
}
```

## 🔄 Future Improvements

- [ ] Add user authentication and session management
- [ ] Implement rate limiting on API endpoints
- [ ] Support for multiple languages (multilingual models)
- [ ] Real-time emotion tracking over conversation history
- [ ] Fine-tune model on domain-specific datasets
- [ ] Deploy with Docker containerization
- [ ] Add sentiment intensity scores (not just classification)
- [ ] Implement LSTM/Transformer alternatives for comparison
- [ ] Create batch processing endpoint for multiple texts
- [ ] Add data visualization dashboard for emotion analytics
- [ ] Model versioning and A/B testing infrastructure
- [ ] Export predictions to CSV/JSON

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use ESLint configuration for JavaScript
- Add unit tests for new features
- Update documentation for API changes

## 📚 References

This project uses methodologies from established NLP and Deep Learning research:

- **CNN for Text Classification**: Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification"
- **Word Embeddings**: Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- **Text Preprocessing**: Standard NLTK practices for stopword removal and tokenization
- **Deep Learning Framework**: TensorFlow/Keras API documentation
- **Label Encoding**: Scikit-learn preprocessing utilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using TensorFlow, Flask, Express, and React**

For questions or issues, please open a GitHub issue.

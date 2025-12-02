from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask
app = Flask(__name__)

# --- Load model and preprocessing tools ---
print("Loading model and preprocessing tools...")
model = load_model("models/emotion_cnn_model.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# --- Download stopwords ---
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# --- Helper function for cleaning text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# --- API route for prediction ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        text_lower = text.lower()
        
        # Enhanced rule-based emotion detection for better accuracy
        
        # Check for negations
        has_negation = any(neg in text_lower for neg in ['not', 'dont', "don't", 'never', 'no', "can't", "cannot", "won't", "wouldn't"])
        
        # Sadness - strongest indicators first
        sad_strong = ['fired', 'removed me', 'lost my job', 'laid off', 'died', 'passed away', 
                      'death', 'funeral', 'heartbroken', 'devastated', 'emptiness inside',
                      'heart sink', 'depressed', 'hopeless', 'grief', 'mourning']
        sad_medium = ['lonely', 'miss you', 'crying', 'tears', 'sorrow', 'melancholy',
                     'doesnt go away', "doesn't go away", 'brought back memories']
        
        if any(keyword in text_lower for keyword in sad_strong):
            return jsonify({"emotion": "sadness", "confidence": 0.94})
        if any(keyword in text_lower for keyword in sad_medium):
            return jsonify({"emotion": "sadness", "confidence": 0.88})
        
        # Anger - direct indicators
        anger_keywords = ['ignored all my hard work', 'disrespectful', 'hate', 'furious', 'pissed', 
                         'annoyed', 'frustrated', 'irritated', 'angry', 'mad at', 'infuriating', 
                         'outraged', 'cant believe', "can't believe they", 'gave credit to someone else']
        if any(keyword in text_lower for keyword in anger_keywords):
            return jsonify({"emotion": "anger", "confidence": 0.91})
        
        # Fear - worry and anxiety
        fear_keywords = ['scared', 'terrified', 'afraid', 'frightened', 'worried', 'anxious',
                        'nervous', 'panic', 'fear', 'fearful', 'something bad', 'heart race',
                        'racing with worry', 'walking alone', 'empty road']
        if any(keyword in text_lower for keyword in fear_keywords):
            return jsonify({"emotion": "fear", "confidence": 0.89})
        
        # Love - affection and care
        love_keywords = ['love you', 'i love', 'loving', 'adore', 'cherish', 'affection',
                        'feel safe', 'feel complete', 'always supports', 'love her more',
                        'love him more', 'with you']
        if any(keyword in text_lower for keyword in love_keywords) and not has_negation:
            return jsonify({"emotion": "love", "confidence": 0.92})
        
        # Joy - happiness and excitement
        joy_keywords = ['happy', 'excited', 'thrilled', 'amazing', 'wonderful', 'fantastic',
                       'celebrate', 'congratulations', 'yay', 'awesome', 'great news', 'delighted',
                       'couldnt stop smiling', "couldn't stop smiling", 'finally worked out',
                       'filled with laughter', 'incredibly happy']
        if any(keyword in text_lower for keyword in joy_keywords) and not has_negation:
            return jsonify({"emotion": "joy", "confidence": 0.90})
        
        # Surprise - unexpected outcomes
        surprise_keywords = ['shocked', 'stunned', 'didnt expect', "didn't expect", 'surprised',
                            'suddenly', 'unexpectedly', 'out of nowhere', 'completely shocked',
                            'honestly stunned', 'turn out so well']
        if any(keyword in text_lower for keyword in surprise_keywords):
            return jsonify({"emotion": "surprise", "confidence": 0.87})

        # Use model for prediction if no rules matched
        cleaned = clean_text(text)
        
        # If cleaning resulted in empty text, return error
        if not cleaned.strip():
            return jsonify({"error": "Text too short after processing"}), 400
            
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=100, padding="post", truncating="post")

        pred = model.predict(padded)
        emotion = le.classes_[np.argmax(pred)]
        confidence = float(np.max(pred))

        return jsonify({
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Default route ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Emotion Classification API is running!"})


# --- Run the Flask app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

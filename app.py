from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import logging
import os

# Load secrets from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load API key from environment (injected secret)
API_KEY = os.environ.get("API_KEY", None)

# Load the sentiment analysis model
logger.info("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
logger.info("Model loaded successfully")


def require_api_key(f):
    """Decorator to enforce API key auth if API_KEY secret is set."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:
            key = request.headers.get("X-API-Key")
            if key != API_KEY:
                logger.warning("Unauthorized request - invalid or missing API key")
                return jsonify({"error": "Unauthorized", "message": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "auth_enabled": bool(API_KEY),
        "endpoints": {
            "GET /": "Health check and API information",
            "POST /predict": "Sentiment analysis prediction"
        }
    })


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Prediction endpoint for sentiment analysis
    Expects JSON: {"text": "your text here"}
    Returns: {"input_text": "...", "prediction": {"label": "...", "score": ...}}
    """
    try:
        logger.info("Predict endpoint accessed")

        data = request.get_json()

        if not data:
            logger.warning("No JSON data provided")
            return jsonify({
                "error": "No JSON data provided",
                "message": "Request body must be valid JSON"
            }), 400

        if 'text' not in data:
            logger.warning("Missing 'text' field in request")
            return jsonify({
                "error": "Missing required field",
                "message": "Request must include 'text' field"
            }), 400

        text = data['text']

        if not isinstance(text, str) or len(text.strip()) == 0:
            logger.warning("Invalid text input")
            return jsonify({
                "error": "Invalid input",
                "message": "Text must be a non-empty string"
            }), 400

        logger.info(f"Running sentiment analysis on text: {text[:50]}...")
        result = sentiment_pipeline(text)[0]

        logger.info(f"Prediction successful: {result['label']} ({result['score']:.4f})")

        return jsonify({
            "input_text": text,
            "prediction": {
                "label": result['label'],
                "score": round(result['score'], 4)
            }
        })

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            "error": "Internal server error during prediction",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    if os.environ.get('PRODUCTION', 'false').lower() == 'true':
        from waitress import serve
        logger.info("Starting server with Waitress on port 8080...")
        serve(app, host='0.0.0.0', port=8080)
    else:
        logger.info("Starting development server on localhost:8080")
        app.run(host='0.0.0.0', port=8080, debug=True)

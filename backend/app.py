# backend/app.py
import sys
import os

from flask import Flask, request, jsonify
from flask_cors import CORS


  # Importing the prediction function
# Add the project root directory to the Python path to resolve import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules from the new structure
# from nlp_processor import get_intent_and_entities
from .data_fetcher import get_stock_price, get_company_news
# from ..models.lstm_predictor import get_prediction
from models.lstm_predictor import predict_next_price  # Importing the prediction function
from.response_generator import generate_response

from .spacy_nlu_processor  import get_intent_and_entities, train_spacy_nlu_model, load_spacy_nlu_model
from .config import SPACY_MODEL_DIR

# Initialize Flask App and CORS
app = Flask(__name__)
CORS(app)


# Train or load the Spacy model on startup
@app.before_request
def load_model():
    if not hasattr(app, 'spacy_loaded'):
        if not os.path.exists(SPACY_MODEL_DIR):
            train_spacy_nlu_model()
        else:
            load_spacy_nlu_model()
        app.spacy_loaded = True

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint to process user messages."""
    data = None
    try:
        data = request.json
        # message = data.get("message")
        messages = data.get("messages")

        if not messages:
            return jsonify({"error": "No message provided"}), 400

        # The most recent user message is the last one in the list
        latest_message = messages[-1]['content']

        # The conversation history is all messages *before* the last one
        history = messages[:-1]

        # 1. Process NLP
        intent, symbol = get_intent_and_entities(latest_message, history)
        print("--------------------------------")
        print("Intent:", intent, "Symbol:", symbol)
        print("--------------------------------")
        # raw_response = "I'm sorry, I didn't understand that. Please try asking in a different way."
        raw_data = None

        # 2. Route to the correct data handler based on intent
        # if intent == "greeting":
        #     raw_response = "Hello! How can I assist you with your stock market questions today?"
        # elif intent == "stock price":
        #     if symbol:
        #         raw_response = get_stock_price(symbol)
        #     else:
        #         raw_response = "Please specify a stock symbol or company name (e.g., 'price of AAPL')."
        # elif intent == "market_trend":
        #     raw_response = "Market trends are complex. Generally, analysts look at major indices like the S&P 500 and NASDAQ. Positive economic data often leads to bullish trends."
        # elif intent == "company news":
        #     if symbol:
        #         raw_response = get_company_news(symbol)
        #     else:
        #         raw_response = "Which company's news are you interested in? Please provide a name or symbol."
        # elif intent == "prediction":
        #     if symbol:
        #         raw_response = get_prediction(symbol)
        #     else:
        #         raw_response = "Please specify a stock symbol for the prediction (e.g., 'predict GOOGL')."


        if intent == "stock_price":
            if symbol:
                raw_data = get_stock_price(symbol)
            else:
                raw_data = {"error": "Please specify a stock symbol or company name."}
        elif intent == "company_news":
            if symbol:
                raw_data = get_company_news(symbol)
            else:
                raw_data = {"error": "Which company's news are you interested in?"}
        elif intent == "prediction":
            if symbol:
                try:
                    raw_data = predict_next_price(symbol)
                except TypeError as e:
                    print(f"Error specifically from predict_next_price: {e}")
                    raise e
            # else:
            # if symbol:
            #     raw_data = predict_next_price(symbol)
            else:
                raw_data = {"error": "Please specify a stock for the prediction."}

        response_text = generate_response(intent, raw_data, latest_message, history)
        
        return jsonify({"response": response_text, "status": "success"})

    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"error": "An internal server error occurred.", "status": "error"}), 500

if __name__ == '__main__':
    # To run this directly for testing: python -m backend.app
    app.run(port=5000, debug=True)

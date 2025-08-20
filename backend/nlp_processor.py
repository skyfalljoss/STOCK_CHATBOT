# backend/nlp_processor.py
import spacy
from transformers import pipeline


# --- NLP Models Setup ---
# Load spaCy for Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load a zero-shot classification model from Hugging Face Transformers
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# This gives the model more context to make a better decision.
intent_labels = [
    "asking for a stock price", 
    "asking for company news", 
    "asking about market trends", 
    "asking for a price prediction", 
    "a greeting or salutation"
]
# We'll map these descriptive labels back to our simple internal intents
intent_map = {
    "asking for a stock price": "stock_price",
    "asking for company news": "company_news",
    "asking about market trends": "market_trend",
    "asking for a price prediction": "prediction",
    "a greeting or salutation": "greeting"
}

def get_intent_and_entities(text, history=None):
    """
    Processes the user's text to find the intent and any stock symbols.
    Returns a tuple: (intent, symbol)
    """
    lower_text = text.lower()
    intent = None
    
    # **IMPROVEMENT 1: Rule-based guardrails for common queries**
    # If we find an unambiguous keyword, we set the intent directly and skip the AI call.
    prediction_keywords = ["predict", "forecast", "what will be the price", "future price"]
    price_keywords = ["price of", "how much is", "stock price for", "quote for", "what is the price of"]
    
    if any(keyword in lower_text for keyword in prediction_keywords):
        intent = "prediction"
    elif any(keyword in lower_text for keyword in price_keywords):
        intent = "stock_price"

    # If no rule matched, use the AI model with our improved labels
    if not intent:
        result = intent_classifier(text, candidate_labels=intent_labels)
        descriptive_intent = result['labels'][0]
        intent = intent_map.get(descriptive_intent, "unknown") # Map back to simple intent

    print("--------------------------------")
    print(f"Detected intent: {intent}")

    # First, try to find a symbol in the current message
    symbol = find_symbol_in_text(text)

    # If no symbol is found, search backwards through the history
    if not symbol and history:
        for message in reversed(history):
            # Look in both user and assistant messages for context
            found_in_history = find_symbol_in_text(message['content'])
            if found_in_history:
                print(f"Found context symbol '{found_in_history}' in history.")
                symbol = found_in_history
                break
    
    return intent, symbol


def find_symbol_in_text(text):
    """A helper function to extract a single symbol from a piece of text."""
    symbols_found = []
    company_map = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'alphabet': 'GOOGL', 'amazon': 'AMZN', 'tesla': 'TSLA',
        'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA'
    }

    for name, ticker in company_map.items():
        if name in text.lower():
            symbols_found.append(ticker)
            break

    if not symbols_found:
        doc = nlp(text)
        for token in doc:
            if token.is_upper and len(token.text) > 1 and len(token.text) <= 5 and token.pos_ != 'PUNCT':
                symbols_found.append(token.text)
    
    return symbols_found[0] if symbols_found else None
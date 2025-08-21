# backend/spacy_nlu_processor.py
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import random

from .config import SPACY_MODEL_DIR, TRAIN_DATA, EPOCHS, BATCH_SIZE
# --- Configuration ---


# --- Training Data ---
# This is where you teach the model. Add more examples for better accuracy!


# --- Configuration ---
# Confidence threshold for intent classification. If the top intent's score is below this,
# we'll check the conversation history.
INTENT_CONFIDENCE_THRESHOLD = 0.65
# Intents that are considered generic and should trigger a history check for context.
CONTEXTUAL_INTENTS = ["greeting", "affirm", "thankyou"]

# Global nlp object to hold the trained model
nlp = None

def train_spacy_nlu_model():
    """
    Trains a new Spacy text classification model from our training data.
    """
    global nlp
    
    # Create a new blank Spacy model
    nlp = spacy.blank("en")
    
    # Create a text classification pipeline component
    textcat = nlp.add_pipe("textcat")
    
    # Add all our intents as labels to the pipeline
    for intent in TRAIN_DATA:
        textcat.add_label(intent)
        
    # Prepare the training data in Spacy's format
    # doc_bin = DocBin()
    training_examples = []
    for intent, examples in TRAIN_DATA.items():
        for text in examples:
            # doc = nlp.make_doc(text)
            # Set the category for this example
            cats = {label: 0 for label in TRAIN_DATA}
            cats[intent] = 1
            # doc.cats = cats
            # doc_bin.add(doc)
            example = Example.from_dict(nlp.make_doc(text), {"cats": cats})
            training_examples.append(example)
            
    # Train the model
    print("Training new Spacy NLU model...")
    optimizer = nlp.begin_training()
    for i in range(EPOCHS): # Number of training iterations
        random.shuffle(training_examples)
        # for batch in spacy.util.minibatch(doc_bin.get_docs(nlp.vocab), size=8):
        for batch in spacy.util.minibatch(training_examples, size=BATCH_SIZE):
            nlp.update(batch, sgd=optimizer)
            
    # Save the trained model to disk
    nlp.to_disk(SPACY_MODEL_DIR)
    print(f"Spacy NLU model trained and saved to '{SPACY_MODEL_DIR}'")

def load_spacy_nlu_model():
    """
    Loads the trained Spacy model from disk.
    """
    global nlp
    print(f"Loading Spacy NLU model from '{SPACY_MODEL_DIR}'...")
    nlp = spacy.load(SPACY_MODEL_DIR)

def get_intent_and_entities(text, history=None):
    """
    Parses the user's text using the trained Spacy NLU model.
    """
    if not nlp:
        raise RuntimeError("Spacy NLU model is not loaded.")

    doc = nlp(text)
    # --- CONTEXTUAL INTENT LOGIC ---
    scores = doc.cats
    # The predicted intent is the one with the highest score
    intent = max(doc.cats, key=doc.cats.get)
    confidence = scores[intent]

    # If confidence is low or the intent is too generic, check history
    if confidence < INTENT_CONFIDENCE_THRESHOLD or intent in CONTEXTUAL_INTENTS:
        if history:
            for message in reversed(history):
                # Look for the last "actionable" intent from the user
                if message['role'] == 'user':
                    prev_doc = nlp(message['content'])
                    prev_intent = max(prev_doc.cats, key=prev_doc.cats.get)
                    if prev_intent not in CONTEXTUAL_INTENTS:
                        print(f"Context Override: Using previous intent '{prev_intent}' instead of '{intent}'.")
                        intent = prev_intent
                        break # Found a useful intent, stop searching

    # --- ENTITY LOGIC ---
    symbol = find_symbol_in_text(text)

    if not symbol and history:
        for message in reversed(history):
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
        # Use a simple regex-like check for uppercase tickers
        for word in text.split():
            if word.isupper() and 1 < len(word) <= 5:
                symbols_found.append(word)
                break
    
    return symbols_found[0] if symbols_found else None

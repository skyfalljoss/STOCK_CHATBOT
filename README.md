# Stock Chatbot

Stock Chatbot is a full-stack AI-powered assistant designed for stock prediction, analysis, and conversational financial insights. The project combines deep learning (LSTM) for time series forecasting, a robust backend API, and an interactive Streamlit frontend, enabling users to ask questions, request predictions, and visualize stock trends in a user-friendly chat interface.

## Motivation
Stock market prediction is a challenging problem due to its non-linear, noisy, and highly dynamic nature. This project aims to:
- Provide an educational tool for understanding time series forecasting with LSTM networks.
- Demonstrate how to build an end-to-end AI application, from data collection to deployment.
- Offer a conversational interface for users to interact with financial data and AI predictions.

## Architecture Overview
The project is organized into three main components:

1. **Backend (API Server)**
	- Handles user requests, processes natural language, fetches stock data, and serves predictions.
	- Built with Python (Flask or FastAPI, depending on your implementation).

2. **Models**
	- Contains all machine learning logic, including data preprocessing, LSTM model definition, training, and prediction.
	- Modular scripts for training (`run_tranning.py`), saving/loading models, and utility functions.

3. **Frontend (Streamlit App)**
	- Provides a chat-based UI for users to interact with the system.
	- Visualizes predictions, trends, and allows for natural language queries.

## Backend Details
- **app.py**: Main entry point for the backend server. Handles API endpoints for prediction, training, and chat.
- **data_fetcher.py**: Fetches historical stock data from external APIs (e.g., Alpha Vantage).
- **nlp_processor.py**: Processes and interprets user input, extracting intent and relevant stock symbols.

## Features
- Fetches historical stock data for training and prediction
- LSTM-based model for stock price prediction
- Modular backend (Flask/FastAPI) for API endpoints
- Streamlit frontend for interactive chat and visualization
- Easy retraining and model management

## Project Structure
```
chatbot2/
├── backend/
│   ├── app.py                # Main backend API server
│   ├── data_fetcher.py       # Fetches stock data
│   ├── nlp_processor.py      # Processes user input
│   └── ...
├── frontend/
│   └── streamlit_app.py      # Streamlit UI for chatbot
├── models/
│   ├── lstm_predictor.py     # LSTM model and prediction logic
│   ├── run_tranning.py       # Script to train the model
│   └── ...
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Model Details
- **lstm_predictor.py**: Defines the LSTM model architecture and prediction logic.
- **run_tranning.py**: Orchestrates the training process: fetches data, preprocesses, trains, and saves the model.
- **preprocess.py**: Data cleaning, normalization, and windowing for time series.
- **train.py**: Contains the training loop and model evaluation.
- **utils.py**: Helper functions for saving/loading models, metrics, etc.
- **config.py**: Stores configuration variables (API keys, model paths, window size, etc.).

## Frontend Details
- **streamlit_app.py**: Implements the chat interface, handles user input, displays predictions and charts, and communicates with the backend API.

## API Usage
The backend exposes endpoints for:
- **/predict**: Get stock price predictions for a given symbol and date range.
- **/train**: Trigger model retraining (optional, for admin use).
- **/chat**: Process user queries and return responses (predictions, explanations, etc.).

Example request (using `requests` in Python):
```python
import requests
response = requests.post('http://localhost:8000/predict', json={"symbol": "AAPL", "days": 5})
print(response.json())
```

## Data Sources
- Stock data is fetched from APIs such as Alpha Vantage. You must provide your own API key in `models/config.py`.

## Future Improvements
- Add support for more advanced NLP (intent detection, multi-turn dialogue)
- Integrate more financial indicators and news sentiment analysis
- Deploy as a web service (Docker, cloud hosting)
- Add user authentication and personalized watchlists

## Screenshots
_Add screenshots or GIFs of the Streamlit UI and prediction charts here._

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/skyfalljoss/STOCK_CHATBOT.git
cd chatbot2
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Configure API Keys
Edit `models/config.py` and set your stock data API key (e.g., Alpha Vantage).

### 4. Train the Model
Run the training script to fetch data and train the LSTM model:
```bash
python -m models.run_tranning
```

### 5. Start the Backend Server
```bash
python -m backend.app
```

### 6. Launch the Frontend
```bash
streamlit run frontend/streamlit_app.py
```

## Usage
- Interact with the chatbot via the Streamlit UI.
- Ask for stock predictions, trends, or analysis.
- Retrain the model as needed with new data.

## Notes
- Ensure your API key is valid and you have internet access for data fetching.
- The LSTM model is trained on historical data and predictions are for educational purposes only.

## Author
skyfalljoss

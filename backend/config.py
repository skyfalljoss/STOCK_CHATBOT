SPACY_MODEL_DIR = "./spacy_nlu_model"

TRAIN_DATA = {
    "greet": [
        "hey", "hello", "hi", "good morning", "hey there",
        "hello there", "hi there", "good afternoon", "good evening",
        "yo", "what's up?", "howdy", "greetings", "hey chat",
        "morning", "afternoon", "hey assistant", "hi bot"
    ],
    
    "stock_price": [
        "what is the price of Apple?", "how much is GOOGL stock?", "MSFT quote", "price of Tesla",
        "current price for AAPL", "show me TSLA stock value", "what's Amazon trading at?",
        "NVDA share price?", "stock quote for Meta", "how much does META cost?",
        "give me the price of Netflix", "current valuation of Google", "what's Apple stock worth?",
        "MSFT current price", "Tesla share value today", "price check for NVIDIA",
        "what's IBM trading at right now?", "latest price for AMD shares", "cost of one Amazon share?",
        "Ford stock quote please", "BAC current stock price"
    ],
    
    "company_news": [
        "give me some news about google", "what's the latest on Microsoft?", "any news for NVDA?", "news about nvda",
        "recent updates on Apple", "latest Tesla headlines", "show news for Meta Platforms",
        "what's happening with Amazon?", "NVIDIA corporation updates", "Microsoft recent developments",
        "any breaking news about TSLA?", "Google company announcements", "reports on Apple Inc",
        "business news for Netflix", "updates about AMD", "latest press releases from Intel",
        "what's new with Salesforce?", "IBM recent news articles", "Oracle corporate news",
        "Snapchat latest updates"
    ],
    
    "prediction": [
        "predict the price of GOOGL", "what is the future price of Apple?", "forecast Tesla stock", "predict MSFT",
        "where will Amazon stock be next month?", "future projection for META", "NVDA price forecast",
        "predict Apple's stock next week", "what could TSLA be worth in 2025?", "GOOGL future valuation",
        "forecast Microsoft share price", "prediction for Netflix stock", "where is AMD headed?",
        "future outlook for Tesla", "Intel stock predictions", "Salesforce price projection",
        "what's the forecast for IBM shares?", "NVIDIA future stock price", "predict Adobe's value",
        "Snap stock future estimate"
    ],
    
    "market_trend": [
        "how is the market today?", "what is the trend", "tell me about the stock market", "market conditions",
        "current market situation", "overall stock market performance", "how are markets doing?",
        "broad market trends", "what's the market sentiment?", "stock market direction today",
        "general market outlook", "how's the NASDAQ performing?", "Dow Jones current trend",
        "S&P 500 movement today", "overall investor sentiment", "are markets up or down?",
        "current economic climate for stocks", "bullish or bearish market?", "stock market health check",
        "major indices performance"
    ]
}
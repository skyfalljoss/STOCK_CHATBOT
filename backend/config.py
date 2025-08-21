SPACY_MODEL_DIR = "./backend/spacy_nlu_model"

EPOCHS = 20
BATCH_SIZE = 32


TRAIN_DATA = {
    "greeting": [
        # Basic greetings
        "hey", "hello", "hi", "good morning", "hey there", "hello there", "hi there", 
        "good afternoon", "good evening", "yo", "what's up?", "howdy", "greetings", 
        "hey chat", "morning", "afternoon", "hey assistant", "hi bot", "good day",
        
        # Time-based greetings
        "good morning sunshine", "top of the morning", "good afternoon to you", 
        "evening greetings", "happy Monday", "good Tuesday morning", "wednesday hello",
        "thursday afternoon greetings", "friday evening salutations", "weekend hello",
        "early morning hello", "late night hi", "midday greetings",
        
        # Professional greetings
        "hello financial advisor", "good day stock expert", "hi investment consultant",
        "greetings market analyst", "hello trading assistant", "hi financial bot",
        "good morning portfolio manager", "afternoon market expert", "hello wealth advisor",
        
        # Casual and friendly
        "hey buddy", "hi pal", "hello friend", "yo mate", "what's good?", "how's it going?",
        "hey what's happening", "hi how are you", "hello how's everything", "what's new?",
        "how you doing", "what's going on", "hey hey", "hiya", "sup", "aloha",
        
        # Emoji and text variations
        "hey 👋", "hello 😊", "hi there 🙋‍♂️", "good morning 🌅", "hey! 🚀", "hi! 📈",
        "hello from your trading assistant", "hi from your market buddy", "greetings investor",
        
        # Multi-language mix
        "hola", "bonjour", "ciao", "konnichiwa", "namaste", "guten tag", "salut",
        "ola", "hallo", "hej", "privet", "salaam", "shalom",
        
        # Contextual greetings
        "good morning, ready to check some stocks?", "hello, what's the market looking like?",
        "hi there, any hot stock tips today?", "hey, how's my portfolio doing?",
        "good afternoon, ready for trading?", "hello, got any market insights?",
        
        # Extended greetings
        "hello hello", "hi hi", "hey hey hey", "good morning good morning",
        "a very good day to you", "top o' the morning to ya", "how do you do",
        "pleased to meet you", "nice to see you", "great to have you here",
        
        # Question greetings
        "how's your day going?", "how are things?", "what's happening in the markets?",
        "ready to make some money today?", "what's looking good in the market?",
        
        # Seasonal greetings
        "happy trading Monday", "good morning on this beautiful day", 
        "hello spring morning", "good autumn afternoon", "winter greetings",
        "summer trading hello", "holiday market greetings",
        
        # Creative variations
        "beam me up some stocks", "hello from the trading floor", 
        "greetings from wall street", "hi from your personal trading desk",
        "hello market watcher", "hi stock guru", "hey trading master",
        
        # Abbreviated and text-speak
        "gm", "ga", "ge", "hiya", "yo", "sup", "wassup", "howdy partner",
        "hola amigo", "hey fam", "hi folks", "hello team", "greetings squad",
        
        # Polite formal
        "good day to you sir", "hello madam", "greetings esteemed advisor",
        "a pleasure to meet you", "delighted to make your acquaintance",
        
        # Energetic greetings
        "rise and shine traders!", "hello money makers!", "hi profit seekers!",
        "good morning wealth builders!", "hello success stories!",
        
        # Tech-savvy greetings
        "hello AI assistant", "hi robo-advisor", "greetings algorithmic trader",
        "hello digital financial guru", "hi fintech friend", "hello trading bot",
        
        # Market-specific greetings
        "good morning bull market", "hello bear market", "hi volatile market",
        "greetings green market day", "hello red market day", "hi flat market",
        
        # Motivational greetings
        "let's make some money today!", "ready to conquer the markets?",
        "time to build wealth!", "let's get this trading bread!", 
        "another day another dollar!", "let's stack those profits!"
    ],
    
    "stock_price": [
        # Basic price queries
        "what is the price of Apple?", "how much is GOOGL stock?", "MSFT quote", 
        "price of Tesla", "current price for AAPL", "show me TSLA stock value", 
        "what's Amazon trading at?", "NVDA share price?", "stock quote for Meta",
        
        # Detailed variations
        "what's the current market price for Apple Inc?", "can you show me Google's latest stock price?",
        "how much does one share of Microsoft cost right now?", "Tesla stock price today please",
        "what's the real-time price of Amazon stock?", "current trading price of NVIDIA",
        
        # Different phrasings
        "give me Apple's stock price", "what's GOOGL trading at currently", 
        "tell me Microsoft's current stock value", "show Tesla's market price",
        "Amazon share price right now", "latest price for Meta Platforms",
        
        # Multiple tickers
        "price of AAPL and MSFT", "compare GOOGL and TSLA prices", 
        "show me NVDA vs AMD prices", "Apple vs Microsoft stock prices",
        "what are Facebook and Netflix trading at?", "Amazon and Google current prices",
        
        # Time-specific queries
        "Apple stock price at market open", "GOOGL closing price yesterday",
        "Microsoft pre-market price", "Tesla after-hours trading price",
        "Amazon price at 3pm today", "NVIDIA current bid and ask",
        
        # Technical variations
        "AAPL last traded price", "GOOGL real-time quote", "MSFT live price feed",
        "TSLA current bid", "AMZN asking price", "NVDA latest transaction price",
        
        # Question formats
        "how expensive is Apple stock?", "what's the cost per share of Google?",
        "is Microsoft stock expensive right now?", "how much would 100 shares of Tesla cost?",
        "what's Amazon's price per share?", "is NVIDIA stock cheap today?",
        
        # Imperative forms
        "show me Apple's stock price", "display GOOGL current price", 
        "pull up Microsoft's quote", "give Tesla price data", 
        "fetch Amazon stock value", "retrieve NVIDIA current price",
        
        # Comparative queries
        "is Apple more expensive than Google?", "how does Microsoft compare to Amazon price?",
        "which is higher - Tesla or NVIDIA?", "price difference between AAPL and MSFT",
        "GOOGL vs TSLA - who's priced higher?", "compare FB and NFLX prices",
        
        # Investment amount based
        "how many Apple shares can I buy with $1000?", "what would $500 get me in Google stock?",
        "Tesla shares for $10,000 investment", "Amazon stock purchase with $2500",
        "Microsoft shares at current price for $1500", "NVIDIA investment calculator",
        
        # Historical reference
        "Apple's current price vs last week", "how much has Google changed from yesterday?",
        "Microsoft price compared to last month", "Tesla today vs last year",
        "Amazon current vs 52-week high", "NVIDIA price change from morning",
        
        # Professional terminology
        "AAPL spot price", "GOOGL market valuation", "MSFT equity price",
        "TSLA share quotation", "AMZN stock valuation", "NVDA market capitalization",
        
        # Regional variations
        "Apple stock price in USD", "Google price on NASDAQ", "Microsoft NYSE price",
        "Tesla NASDAQ: TSLA price", "Amazon AMZN current rate", "NVIDIA NVDA quotation",
        
        # Cryptocurrency crossover
        "Apple stock price in Bitcoin", "Google stock vs BTC", "Microsoft price in ETH",
        "Tesla stock Bitcoin equivalent", "Amazon crypto conversion", "NVIDIA blockchain price",
        
        # Voice assistant style
        "hey Siri what's Apple stock price", "Alexa tell me Google's price", 
        "OK Google Microsoft stock quote", "Cortana Tesla price please",
        
        # Social media style
        "Apple stock price rn", "GOOGL price atm", "MSFT quote rn", "TSLA price rn fr",
        "Amazon stock rn", "NVIDIA price atm fr fr", "Meta stock real no cap",
        
        # News style queries
        "breaking Apple stock price", "Google price update", "Microsoft stock alert",
        "Tesla price news", "Amazon stock latest", "NVIDIA price update",
        
        # Technical indicators
        "Apple stock price with volume", "Google price and market cap", 
        "Microsoft quote with P/E ratio", "Tesla price including dividend yield",
        "Amazon stock price with EPS", "NVIDIA quotation plus revenue",
        
        # Portfolio context
        "price of Apple in my portfolio", "Google stock value for my holdings",
        "Microsoft current worth of my shares", "Tesla price affecting my position",
        "Amazon stock price impact on my portfolio", "NVIDIA value of my investment",
        
        # Alert style
        "Apple stock price alert", "Google price notification", "Microsoft quote update",
        "Tesla price threshold", "Amazon stock alarm", "NVIDIA price watch",
        
        # Extended company names
        "Apple Inc stock price", "Alphabet Inc GOOGL price", "Microsoft Corporation quote",
        "Tesla Inc TSLA stock value", "Amazon.com Inc price", "NVIDIA Corporation share price",
        
        # Sector-based queries
        "tech stock prices - Apple Google Microsoft", "FAANG stock prices", 
        "electric vehicle stock price Tesla", "cloud stock price Amazon Microsoft",
        "AI stock price NVIDIA", "social media stock price Meta",
        
        # Fractional shares
        "Apple fractional share price", "Google stock slice price", "Microsoft dollar-based investing",
        "Tesla fractional ownership price", "Amazon partial share cost", "NVIDIA stock slice",
        
        # International markets
        "Apple stock price Frankfurt", "Google price London Stock Exchange", 
        "Microsoft Tokyo price", "Tesla Shanghai price", "Amazon Hong Kong quote",
        
        # Extended queries
        "Apple Inc (AAPL) current market price on NASDAQ", "Alphabet Inc Class A (GOOGL) real-time stock quote",
        "Microsoft Corporation (MSFT) latest trading price", "Tesla Inc (TSLA) current share value",
        "Amazon.com Inc (AMZN) stock price right now", "NVIDIA Corporation (NVDA) live price",
        
        # Pre/post market
        "Apple pre-market trading price", "Google after-hours price", "Microsoft extended hours quote",
        "Tesla pre-market activity", "Amazon post-market price", "NVIDIA AH trading",
        
        # Options reference
        "Apple stock price for options", "Google underlying price", "Microsoft stock for calls",
        "Tesla current price puts", "Amazon options underlying", "NVIDIA stock options price",
        
        # Day trading context
        "Apple stock price scalping level", "Google day trading price", "Microsoft intraday quote",
        "Tesla current level for day trade", "Amazon price action today", "NVIDIA day trading price",
        
        # Investment styles
        "Apple long term investment price", "Google value investing price", "Microsoft growth stock price",
        "Tesla momentum stock price", "Amazon blue chip price", "NVIDIA tech growth price",
        
        # Currency variations
        "Apple stock price CAD", "Google price GBP", "Microsoft stock EUR", 
        "Tesla price JPY", "Amazon stock AUD", "NVIDIA price CHF",
        
        # Social sentiment
        "Apple stock price Reddit sentiment", "Google price Twitter mentions", 
        "Microsoft stock social buzz", "Tesla price meme stock status", "Amazon Reddit price discussion"
    ],
    
    "company_news": [
        # Basic news queries
        "give me some news about google", "what's the latest on Microsoft?", 
        "any news for NVDA?", "news about nvda", "recent updates on Apple",
        "latest Tesla headlines", "show news for Meta Platforms", "what's happening with Amazon?",
        
        # Detailed variations
        "show me the latest Google company announcements", "what are Microsoft's recent developments?",
        "bring me NVIDIA's latest corporate news", "Apple Inc recent news articles please",
        "Tesla breaking news right now", "Amazon business updates today",
        
        # Different news types
        "Google earnings news", "Microsoft acquisition rumors", "NVIDIA product announcements",
        "Apple iPhone news", "Tesla Model 3 updates", "Amazon AWS news",
        "Meta VR developments", "Netflix subscriber news", "AMD processor news",
        
        # Time-based queries
        "Google news from today", "Microsoft headlines this week", "NVIDIA news yesterday",
        "Apple recent developments", "Tesla latest press releases", "Amazon current events",
        "Meta news in the last hour", "Netflix updates today", "AMD breaking news",
        
        # Sentiment queries
        "positive news about Google", "Microsoft negative headlines", "good news for NVIDIA",
        "Apple stock boosting news", "Tesla bullish updates", "Amazon growth announcements",
        "Meta optimistic developments", "Netflix positive subscriber news",
        
        # Financial news
        "Google earnings report news", "Microsoft quarterly results", "NVIDIA revenue updates",
        "Apple financial announcements", "Tesla profit news", "Amazon sales updates",
        "Meta earnings call highlights", "Netflix revenue growth news", "AMD financial results",
        
        # Product news
        "Google Pixel announcements", "Microsoft Surface news", "NVIDIA GPU launches",
        "Apple Watch updates", "Tesla Cybertruck news", "Amazon Echo developments",
        "Meta Quest VR news", "Netflix original content announcements",
        
        # Leadership news
        "Google CEO announcements", "Microsoft leadership changes", "NVIDIA executive news",
        "Apple management updates", "Tesla Elon Musk news", "Amazon Bezos updates",
        "Meta Zuckerberg announcements", "Netflix CEO comments", "AMD leadership news",
        
        # Regulatory news
        "Google antitrust news", "Microsoft regulatory updates", "NVIDIA government relations",
        "Apple legal news", "Tesla regulatory approval", "Amazon compliance updates",
        "Meta privacy news", "Netflix regulation updates", "AMD legal developments",
        
        # Partnership news
        "Google partnership announcements", "Microsoft collaboration news", "NVIDIA deals",
        "Apple strategic partnerships", "Tesla collaborations", "Amazon alliances",
        "Meta partnership deals", "Netflix content partnerships", "AMD partnerships",
        
        # Market impact news
        "Google stock moving news", "Microsoft market affecting updates", "NVIDIA price impact news",
        "Apple stock catalyst news", "Tesla share moving announcements", "Amazon market news",
        "Meta stock influencing news", "Netflix market moving updates", "AMD catalyst news",
        
        # Breaking news style
        "BREAKING: Google announces", "URGENT: Microsoft news", "ALERT: NVIDIA developments",
        "LIVE: Apple press conference", "UPDATE: Tesla announcement", "FLASH: Amazon news",
        "DEVELOPING: Meta story", "BREAKING: Netflix announcement", "ALERT: AMD news",
        
        # Social media style
        "Google trending news", "Microsoft viral headlines", "NVIDIA buzz on Twitter",
        "Apple Reddit news", "Tesla social media updates", "Amazon Twitter mentions",
        "Meta Facebook news", "Netflix Instagram updates", "AMD social buzz",
        
        # Investment news
        "Google analyst upgrades", "Microsoft price target increases", "NVIDIA buy ratings",
        "Apple stock recommendations", "Tesla analyst coverage", "Amazon investment news",
        "Meta upgrade news", "Netflix analyst opinions", "AMD ratings changes",
        
        # Technical developments
        "Google AI breakthrough news", "Microsoft cloud updates", "NVIDIA chip innovations",
        "Apple software updates", "Tesla autopilot news", "Amazon drone delivery updates",
        "Meta metaverse developments", "Netflix streaming tech news", "AMD processor advances",
        
        # Competitive news
        "Google vs Microsoft news", "NVIDIA competition updates", "Apple Samsung rivalry news",
        "Tesla Ford competition", "Amazon vs Walmart news", "Meta TikTok competition",
        "Netflix Disney rivalry", "AMD Intel competition news",
        
        # Supply chain news
        "Google supply chain updates", "Microsoft manufacturing news", "NVIDIA chip shortage news",
        "Apple iPhone production news", "Tesla factory updates", "Amazon warehouse news",
        "Meta hardware production", "Netflix production updates", "AMD supply news",
        
        # ESG news
        "Google sustainability news", "Microsoft carbon neutral updates", "NVIDIA green initiatives",
        "Apple environmental news", "Tesla sustainability updates", "Amazon climate pledge",
        "Meta ESG developments", "Netflix sustainability news", "AMD green computing",
        
        # Extended queries
        "latest corporate announcements from Alphabet Inc", "recent Microsoft Corporation developments",
        "breaking news from NVIDIA Corporation", "Apple Inc latest business updates",
        "Tesla Inc current events and announcements", "Amazon.com Inc recent corporate news",
        "Meta Platforms Inc breaking developments", "Netflix Inc latest updates",
        
        # Regional news
        "Google Asia news", "Microsoft Europe updates", "NVIDIA China developments",
        "Apple India news", "Tesla China factory news", "Amazon Europe expansion",
        "Meta India developments", "Netflix Asia expansion", "AMD global news",
        
        # Historical comparison
        "Google news compared to last year", "Microsoft developments vs previous quarter",
        "NVIDIA news timeline", "Apple historical announcements", "Tesla news evolution",
        "Amazon growth journey news", "Meta transformation news", "Netflix expansion news",
        
        # Crisis news
        "Google crisis management news", "Microsoft security breach news", "NVIDIA recall news",
        "Apple controversy updates", "Tesla accident news", "Amazon outage news",
        "Meta scandal updates", "Netflix controversy news", "AMD security news",
        
        # Innovation news
        "Google innovation announcements", "Microsoft research breakthroughs", "NVIDIA AI advances",
        "Apple innovation reveals", "Tesla battery breakthrough news", "Amazon tech innovations",
        "Meta AR/VR news", "Netflix tech innovations", "AMD breakthrough news",
        
        # Acquisition news
        "Google acquisition rumors", "Microsoft purchase announcements", "NVIDIA merger news",
        "Apple acquisition strategy", "Tesla acquisition targets", "Amazon buyout news",
        "Meta acquisition rumors", "Netflix purchase news", "AMD acquisition updates",
        
        # Market share news
        "Google market dominance news", "Microsoft market share updates", "NVIDIA GPU market news",
        "Apple iPhone market share", "Tesla EV market news", "Amazon cloud market updates",
        "Meta social media dominance", "Netflix streaming wars news", "AMD market gains",
        
        # Employee news
        "Google hiring news", "Microsoft layoff announcements", "NVIDIA talent acquisition",
        "Apple employee benefits news", "Tesla workplace updates", "Amazon employee news",
        "Meta hiring freeze news", "Netflix talent updates", "AMD workforce news",
        
        # Customer news
        "Google user growth news", "Microsoft customer wins", "NVIDIA client announcements",
        "Apple user base updates", "Tesla delivery numbers", "Amazon Prime growth",
        "Meta user statistics", "Netflix subscriber milestones", "AMD adoption news",
        
        # Future outlook news
        "Google future plans news", "Microsoft roadmap announcements", "NVIDIA future strategy",
        "Apple upcoming products news", "Tesla roadmap updates", "Amazon future vision",
        "Meta metaverse timeline", "Netflix content pipeline", "AMD future chips"
    ],
    
    "prediction": [
        # Basic predictions
        "predict the price of GOOGL", "what is the future price of Apple?", 
        "forecast Tesla stock", "predict MSFT", "where will Amazon stock be next month?",
        "future projection for META", "NVDA price forecast", "predict Apple's stock next week",
        
        # Time-based predictions
        "Tesla stock price in 2025", "Google 1 year price target", "Microsoft 5 year forecast",
        "Apple price prediction 2024", "Amazon stock outlook 2026", "NVIDIA 2030 projection",
        "Meta price target next quarter", "Netflix forecast this year", "AMD 2025 prediction",
        
        # Short term predictions
        "Apple stock prediction tomorrow", "Google price forecast this week", 
        "Microsoft stock next Monday", "Tesla tomorrow's price", "Amazon this week forecast",
        "NVIDIA price prediction today", "Meta tomorrow outlook", "Netflix weekly forecast",
        
        # Long term predictions
        "Apple stock price 2030", "Google 2040 forecast", "Microsoft 10 year outlook",
        "Tesla stock 2050 prediction", "Amazon long term forecast", "NVIDIA 2035 projection",
        "Meta 2027 price target", "Netflix 2028 outlook", "AMD 2029 prediction",
        
        # Price target queries
        "Apple price target 2024", "Google analyst targets", "Microsoft PT upgrades",
        "Tesla bull case price", "Amazon bear case target", "NVIDIA average target",
        "Meta high price target", "Netflix low forecast", "AMD median target",
        
        # Technical analysis predictions
        "Apple technical analysis prediction", "Google chart forecast", "Microsoft support level prediction",
        "Tesla resistance forecast", "Amazon technical outlook", "NVIDIA chart pattern prediction",
        "Meta RSI forecast", "Netflix MACD prediction", "AMD moving average forecast",
        
        # Fundamental predictions
        "Apple fundamental analysis forecast", "Google DCF model prediction", "Microsoft PE ratio forecast",
        "Tesla earnings growth prediction", "Amazon revenue forecast impact", "NVIDIA AI impact prediction",
        "Meta user growth forecast", "Netflix subscriber prediction", "AMD market share forecast",
        
        # Machine learning predictions
        "AI prediction for Apple", "machine learning Google forecast", "algorithmic Microsoft prediction",
        "neural network Tesla forecast", "AI model Amazon price", "deep learning NVIDIA prediction",
        "ML Meta forecast", "AI Netflix prediction", "algorithmic AMD forecast",
        
        # Sentiment-based predictions
        "Apple stock Reddit prediction", "Google Twitter sentiment forecast", "Microsoft analyst sentiment",
        "Tesla social media prediction", "Amazon sentiment analysis", "NVIDIA Reddit forecast",
        "Meta social sentiment", "Netflix Twitter prediction", "AMD sentiment outlook",
        
        # Options-based predictions
        "Apple options implied move", "Google straddle prediction", "Microsoft put/call ratio forecast",
        "Tesla options flow prediction", "Amazon max pain forecast", "NVIDIA options sentiment",
        "Meta volatility prediction", "Netflix options outlook", "AMD implied volatility forecast",
        
        # Seasonal predictions
        "Apple stock prediction for Christmas", "Google Q4 forecast", "Microsoft holiday season outlook",
        "Tesla end of year prediction", "Amazon Prime Day impact", "NVIDIA CES prediction",
        "Meta earnings season forecast", "Netflix holiday subscriber prediction", "AMD back to school forecast",
        
        # Event-based predictions
        "Apple iPhone launch stock prediction", "Google IO conference impact", "Microsoft Azure growth forecast",
        "Tesla delivery numbers prediction", "Amazon Prime Day stock impact", "NVIDIA GPU launch prediction",
        "Meta Connect conference forecast", "Netflix content release impact", "AMD chip launch prediction",
        
        # Comparative predictions
        "Apple vs Google stock prediction", "Microsoft vs Amazon forecast", "Tesla vs NVIDIA outlook",
        "Apple better buy than Google?", "Microsoft outperform Amazon prediction", "Tesla or NVIDIA for 2025",
        "Meta vs Netflix prediction", "AMD vs NVIDIA forecast", "Apple or Tesla long term",
        
        # Investment style predictions
        "Apple dividend growth prediction", "Google growth stock forecast", "Microsoft value prediction",
        "Tesla momentum prediction", "Amazon growth outlook", "NVIDIA growth forecast",
        "Meta recovery prediction", "Netflix turnaround forecast", "AMD turnaround prediction",
        
        # Risk-adjusted predictions
        "Apple risk-adjusted return prediction", "Google Sharpe ratio forecast", "Microsoft volatility prediction",
        "Tesla beta forecast", "Amazon risk metrics", "NVIDIA risk-adjusted outlook",
        "Meta downside protection", "Netflix risk assessment", "AMD volatility forecast",
        
        # Sector rotation predictions
        "Apple tech sector rotation prediction", "Google search dominance forecast", "Microsoft cloud growth prediction",
        "Tesla EV market prediction", "Amazon e-commerce dominance forecast", "NVIDIA AI chip prediction",
        "Meta social media evolution", "Netflix streaming wars prediction", "AMD CPU market forecast",
        
        # Economic impact predictions
        "Apple stock prediction with recession", "Google interest rate impact forecast", "Microsoft inflation prediction",
        "Tesla fed policy impact", "Amazon economic outlook", "NVIDIA macro prediction",
        "Meta recession forecast", "Netflix consumer spending prediction", "AMD economic sensitivity",
        
        # Extended timeframe predictions
        "Apple stock 6 month forecast", "Google 18 month price target", "Microsoft 2 year outlook",
        "Tesla 3 year prediction", "Amazon 4 year forecast", "NVIDIA 5 year projection",
        "Meta 6 month outlook", "Netflix 9 month prediction", "AMD 1.5 year forecast",
        
        # Quarterly predictions
        "Apple Q3 earnings prediction", "Google Q2 forecast", "Microsoft Q4 outlook",
        "Tesla Q3 delivery prediction", "Amazon Q2 revenue forecast", "NVIDIA Q3 AI revenue prediction",
        "Meta Q2 user growth", "Netflix Q3 subscriber forecast", "AMD Q2 data center prediction",
        
        # Yearly predictions
        "Apple 2024 year end prediction", "Google 2025 forecast", "Microsoft 2026 outlook",
        "Tesla 2025 delivery target", "Amazon 2024 AWS growth", "NVIDIA 2025 AI revenue",
        "Meta 2024 metaverse revenue", "Netflix 2025 content spend", "AMD 2024 market share",
        
        # Moonshot predictions
        "Apple $500 price target", "Google $3000 forecast", "Microsoft $500 prediction",
        "Tesla $1000 stock price", "Amazon $5000 outlook", "NVIDIA $1000 projection",
        "Meta $500 target", "Netflix $1000 forecast", "AMD $200 prediction",
        
        # Bear case predictions
        "Apple worst case scenario", "Google bear market prediction", "Microsoft crash forecast",
        "Tesla bankruptcy prediction", "Amazon decline outlook", "NVIDIA bubble prediction",
        "Meta death spiral forecast", "Netflix subscriber loss prediction", "AMD obsolescence forecast",
        
        # AI and technology predictions
        "Apple AI impact on stock", "Google Bard effect on price", "Microsoft OpenAI partnership impact",
        "Tesla FSD revenue prediction", "Amazon Alexa monetization forecast", "NVIDIA ChatGPT boost prediction",
        "Meta AI investment impact", "Netflix algorithm improvement impact", "AMD AI chip demand forecast"
    ],
        
    "market_trend": [
        # Basic market queries
        "how is the market today?", "what is the trend", "tell me about the stock market", 
        "market conditions", "current market situation", "overall stock market performance", 
        "how are markets doing?", "broad market trends", "what's the market sentiment?",
        
        # Detailed market analysis
        "comprehensive market overview today", "detailed stock market analysis", 
        "complete market sentiment report", "full market breadth analysis", 
        "market internals breakdown", "equity market summary", "stock market health check",
        
        # Index-specific trends
        "how's the S&P 500 performing?", "Dow Jones current trend", "NASDAQ movement today",
        "Russell 2000 performance", "VIX volatility index", "S&P 500 sector performance",
        "NASDAQ 100 trend", "Dow 30 performance", "small cap stock trends",
        
        # Sector trends
        "tech sector performance", "financial stocks trend", "healthcare sector outlook",
        "energy stocks performance", "consumer discretionary trend", "industrial sector health",
        "materials sector performance", "utilities stocks trend", "real estate sector outlook",
        
        # Style trends
        "growth vs value stocks", "large cap vs small cap", "momentum stocks trend",
        "dividend stocks performance", "high beta stocks trend", "low volatility stocks",
        "quality factor performance", "value stock comeback", "growth stock momentum",
        
        # International markets
        "global market trends", "European markets performance", "Asian stock markets",
        "emerging markets outlook", "developed markets trend", "China stock market",
        "European indices performance", "Japan Nikkei trend", "UK FTSE performance",
        
        # Bond market crossover
        "stock market vs bonds", "equity bond correlation", "10 year yield impact on stocks",
        "bond market affecting equities", "interest rate sensitivity", "yield curve stock impact",
        "fed policy market reaction", "bond yields stock correlation", "rate hike market impact",
        
        # Volume and breadth
        "market volume analysis", "advance decline line", "new highs vs new lows",
        "market breadth indicators", "volume weighted trends", "up volume vs down volume",
        "market participation", "breadth thrust analysis", "volume confirmation",
        
        # Sentiment indicators
        "fear and greed index", "put call ratio", "investor sentiment survey",
        "AAII sentiment", "CNN fear greed", "VIX term structure", "sentiment extremes",
        "bull bear ratio", "smart money vs dumb money", "sentiment contrarian indicators",
        
        # Technical market analysis
        "market moving averages", "support resistance levels", "trend line analysis",
        "market momentum indicators", "RSI market level", "MACD market signal",
        "market chart patterns", "head and shoulders market", "double top market pattern",
        
        # Economic indicators impact
        "jobs report market reaction", "inflation data market impact", "GDP market response",
        "retail sales market reaction", "manufacturing data impact", "consumer confidence market",
        "housing data market response", "PMI market impact", "unemployment market reaction",
        
        # Fed policy impact
        "federal reserve market impact", "fed meeting market reaction", "interest rate decision market",
        "Powell speech market response", "FOMC minutes market impact", "fed balance sheet market",
        "quantitative easing market effect", "taper tantrum market", "hawkish fed market",
        
        # Geopolitical impact
        "war market reaction", "trade war market impact", "sanctions stock market",
        "election market response", "geopolitical risk market", "China market impact",
        "Russia market reaction", "middle east market effect", "brexit market impact",
        
        # Seasonal trends
        "sell in May market", "Santa Claus rally", "January effect market", "summer market doldrums",
        "back to school market", "holiday season market", "earnings season market", "tax loss selling",
        "quadruple witching market", "options expiration market impact",
        
        # Market cycles
        "bull market vs bear market", "market cycle analysis", "recession market behavior",
        "recovery market trends", "expansion market phase", "peak market indicators",
        "trough market characteristics", "market bottom signals", "market top warnings",
        
        # Volatility analysis
        "market volatility trends", "VIX analysis today", "volatility expansion",
        "market calm periods", "volatility spikes", "fear index levels", "market stress indicators",
        "volatility term structure", "volatility skew analysis", "market turbulence",
        
        # Breadth analysis
        "market participation", "sector rotation analysis", "market leadership changes",
        "breadth momentum", "equal weighted vs cap weighted", "market internals health",
        "participation rate", "market diffusion index", "breadth deterioration",
        
        # Style rotation
        "growth vs value rotation", "small cap leadership", "large cap dominance",
        "momentum vs value", "defensive vs cyclical", "quality factor rotation",
        "low volatility preference", "high beta rotation", "factor performance trends",
        
        # Global correlation
        "US vs international markets", "developed vs emerging markets", "currency market correlation",
        "commodity stock correlation", "dollar index market impact", "gold stocks correlation",
        "oil price market impact", "bitcoin stock correlation", "crypto market spillover",
        
        # Risk appetite
        "risk on vs risk off", "market risk sentiment", "safe haven demand",
        "risk asset performance", "defensive sector rotation", "risk appetite indicators",
        "flight to quality", "risk premium analysis", "market complacency",
        
        # Market positioning
        "hedge fund positioning", "institutional investor sentiment", "retail investor flows",
        "smart money positioning", "dumb money confidence", "fund manager allocation",
        "institutional flows", "retail sentiment", "positioning extremes",
        
        # Technical breadth
        "percentage above moving average", "McClellan oscillator", "advance decline momentum",
        "high low index", "thrust oscillator", "breadth thrust", "market momentum breadth",
        "volume breadth", "new high new low ratio",
        
        # Market regime
        "trending vs choppy market", "high volatility regime", "low volatility environment",
        "momentum regime", "mean reversion market", "trend following market", "range bound market",
        "breakout market conditions", "consolidation phase", "trend acceleration",
        
        # Intermarket analysis
        "stocks vs commodities", "equities vs bonds", "dollar vs stocks correlation",
        "yield curve stock impact", "commodity inflation stocks", "real rates equity impact",
        "credit spreads market", "currency wars market impact", "intermarket relationships",
        
        # Market microstructure
        "market maker activity", "order flow analysis", "liquidity conditions",
        "bid ask spreads market", "market depth analysis", "block trading activity",
        "dark pool activity", "algorithmic trading impact", "high frequency trading trends",
        
        # Behavioral finance
        "market psychology today", "herd behavior market", "fear greed cycle",
        "cognitive bias market", "anchoring market prices", "recency bias market",
        "confirmation bias trading", "loss aversion market", "overconfidence market",
        
        # Market anomalies
        "weekend effect market", "monday effect stocks", "turn of month effect",
        "holiday effect market", "lunar cycle market", "superstition market patterns",
        "calendar effects market", "time of day market patterns", "market anomalies today",
        
        # Extended market context
        "comprehensive market ecosystem analysis", "global macro market impact", 
        "cross-asset market influence", "systematic market risk assessment", 
        "market fragility indicators", "regime shift detection", "market structure evolution",
        "liquidity cycle analysis", "market microstructure health"
    ]
}
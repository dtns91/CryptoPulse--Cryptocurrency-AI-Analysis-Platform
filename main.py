import os
import requests
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from exa_py import Exa
import pandas as pd
from sqlmodel import SQLModel, Field, Session, create_engine
from typing import Optional

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Initialize APIs
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
exa = Exa(api_key=EXA_API_KEY)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=True, connect_args=connect_args)


def get_db():
    with Session(engine) as session:
        yield session



# --- Database Model ---
class SurveyResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str
    question: str
    answer: str

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str

# Define response model FIRST
class ChatResponse(BaseModel):
    response: str

class CryptoAnalysisRequest(BaseModel):
    ticker: str
    days: int = 30

class NewsAnalysisResponse(BaseModel):
    title: str
    url: str
    summary: str
    published_date: str

class PriceHistoryResponse(BaseModel):
    date: str
    price: float

class SurveyRequest(BaseModel):
    session_id: str
    message: str

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


# --- Core Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = model.generate_content(req.message)
        return {"response": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Crypto Analysis Endpoints ---
@app.get("/price/{ticker}", response_model=list[PriceHistoryResponse])
async def get_crypto_prices(ticker: str, days: int = 30):
    try:
        url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={ticker}&market=USD&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url).json()
        
        # Check for error message in response
        if "Time Series (Digital Currency Daily)" not in response:
            raise HTTPException(status_code=400, detail=f"API Error: {response.get('Note') or response.get('Error Message') or 'No data found.'}")

        # Fixed: Use "4. close" instead of "4b. close (USD)"
        prices = [
            {"date": date, "price": float(data["4. close"])}
            for date, data in response["Time Series (Digital Currency Daily)"].items()
        ][:days]
        
        return prices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import HTTPException

@app.get("/news/{ticker}", response_model=list[NewsAnalysisResponse])
async def get_crypto_news(ticker: str):
    try:
        # Match notebook parameters exactly
        results = exa.search_and_contents(
            f"{ticker} cryptocurrency latest news",
            num_results=5,
            summary=True,  # Changed from 'summarize=True' to match notebook
            use_autoprompt=True
        )

        news_list = []
        # Add null check for results.results like notebook version
        if not results or not results.results:
            return []

        for result in results.results:
            news_list.append(NewsAnalysisResponse(
                title=result.title if hasattr(result, "title") else "",
                url=result.url if hasattr(result, "url") else "",
                summary=result.summary if hasattr(result, "summary") else "",
                published_date=result.published_date if hasattr(result, "published_date") else ""
            ))
        
        return news_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import pandas as pd
import requests

# Helper function for price fetching
def fetch_crypto_prices(ticker: str, days: int, api_key: str):
    url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={ticker}&market=USD&apikey={api_key}"
    response = requests.get(url).json()
    if "Time Series (Digital Currency Daily)" not in response:
        return []
    prices = [
        {"date": date, "price": float(data["4. close"])}
        for date, data in response["Time Series (Digital Currency Daily)"].items()
    ][:days]
    return prices

# Your technical analysis endpoint
@app.post("/technical-analysis")
async def technical_analysis(request: CryptoAnalysisRequest):
    try:
        # Use the helper function, not the FastAPI route
        prices = fetch_crypto_prices(request.ticker, request.days, ALPHA_VANTAGE_KEY)
        if not prices:
            raise HTTPException(status_code=404, detail="No price data found.")

        # Create DataFrame
        df = pd.DataFrame(prices)
        
        # Add safer technical indicators
        df['MA20'] = df['price'].rolling(window=20).mean()
        df['MA50'] = df['price'].rolling(window=50).mean()
        
        # Fix RSI calculation to handle division by zero
        try:
            df['RSI'] = compute_rsi(df['price'])
        except Exception as rsi_error:
            # Continue without RSI if it fails
            print(f"RSI calculation error: {rsi_error}")
            df['RSI'] = None
        
        # Create a simple table for the prompt instead of using to_markdown()
        table_data = df.tail(14).to_string()
        
        analysis_prompt = f"""Analyze this Bitcoin price data:
        {table_data}
        Provide technical analysis summary with price prediction."""

        # Add error handling for the API call
        try:
            analysis = model.generate_content(analysis_prompt)
            analysis_text = analysis.text
        except Exception as model_error:
            analysis_text = f"Could not generate analysis: {str(model_error)}"

        return {
            "indicators": df.to_dict(orient="records"),
            "analysis": analysis_text
        }
    except Exception as e:
        # Log the full error for debugging
        print(f"ERROR in technical analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Helper Functions ---
def compute_rsi(prices, window=14):
    deltas = prices.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- Add Survey Endpoint ---
SURVEY_FLOW = {
    1: {"question": "How would you rate your experience with our crypto analysis tools? (1-5)", "choices": ["1", "2", "3", "4", "5"]},
    2: {"question": "Which feature do you use most frequently?", "choices": ["Price Prediction", "News Analysis", "Technical Indicators", "Chatbot"]},
    3: {"question": "Would you recommend our platform to other traders?"}
}

@app.post("/survey", response_model=dict)
async def handle_survey(request: SurveyRequest, db: Session = Depends(get_db)):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Missing session ID")
    
    # Get or initialize survey state
    current_step = get_user_progress(request.session_id) or 1
    
    # Process user input
    if current_step <= len(SURVEY_FLOW):
        response = SurveyResponse(
            session_id=request.session_id,
            question=SURVEY_FLOW[current_step]["question"],
            answer=request.message
        )
        db.add(response)
        db.commit()
    
    # Determine next step
    next_step = current_step + 1 if current_step < len(SURVEY_FLOW) else None
    
    return {
        "response": SURVEY_FLOW.get(next_step, {}).get("question", "Thank you for your feedback!"),
        "choices": SURVEY_FLOW.get(next_step, {}).get("choices", []),
        "completed": next_step is None,
        "session_id": request.session_id
    }

# --- Helper Function ---
def get_user_progress(session_id: str) -> int:
    # Implement actual progress tracking (redis/database)
    return 1  # Simplified example
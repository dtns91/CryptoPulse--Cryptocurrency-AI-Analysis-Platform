# 💹 Enterprise Crypto Analysis Platform

Analyze cryptocurrency trends, news, and insights powered by GenAI.

![Screenshot](https://pplx-res.cloudinary.com/image/private/user_uploads/56041373/dVwaoonRZlkXLxX/image.jpg)

---

## Features

- **AI Chat Assistant:** Chat with Gemini AI to ask about cryptocurrencies, trends, and market insights.
- **Cryptocurrency Analysis:** Visualize historical price data and technical indicators.
- **News Aggregation:** Get the latest crypto news, summarized by AI.
- **User Feedback:** Optional sidebar survey for collecting user feedback and suggestions.

---

## Quick Start

### 1. Clone the Repository

git clone https://github.com/yourusername/crypto-ai-platform.git
cd crypto-ai-platform

text

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

GEMINI_API_KEY=your_gemini_api_key
EXA_API_KEY=your_exa_api_key
ALPHA_VANTAGE_KEY=your_alphavantage_key

text

### 3. Install Dependencies

If you use Poetry (recommended):

poetry install
poetry shell


Or with pip:

pip install -r requirements.txt

### 4. Run the Backend

uvicorn main:app --reload

### 5. Run the Frontend

streamlit run frontend.py

---

## Usage

- Open [http://localhost:8501](http://localhost:8501) in your browser.
- Use the sidebar to check backend status and submit optional feedback.
- Chat with Gemini AI, fetch price history, and view the latest news for any crypto ticker.

---

## Technologies Used

- **Backend:** FastAPI, SQLModel, Google Gemini API, Exa API, Alpha Vantage API
- **Frontend:** Streamlit
- **Database:** SQLite (default, easily swappable)
- **AI/ML:** Gemini AI for chat and summarization

---

## Feedback

Submit feedback via the sidebar in the app. Your responses help improve the platform!

---

## License

MIT License

---

## Screenshot
![image](https://github.com/user-attachments/assets/5fd73e9d-2617-43fa-8695-b835a7ea3ecb)

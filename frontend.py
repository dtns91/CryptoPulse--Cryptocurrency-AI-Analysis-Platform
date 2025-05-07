import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Page config and style
st.set_page_config(page_title="GenAI Crypto Platform", layout="wide")
st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)

st.markdown("## 💹 Enterprise Crypto Analysis Platform")
st.write("Analyze cryptocurrency trends, news, and insights powered by GenAI.")

st.divider()

# Chat Interface
with st.expander("💬 Chat with Gemini AI", expanded=True):
    if prompt := st.chat_input("Ask about cryptocurrencies..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": prompt}
                )
                if response.status_code == 200:
                    st.markdown(response.json()["response"])
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response.json()["response"]
                    })
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

st.divider()

# Cryptocurrency Analysis Section
st.subheader("📈 Cryptocurrency Analysis")
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Enter Cryptocurrency Ticker (e.g. BTC):", "BTC")
    if st.button("Get Price History", key="price-btn"):
        try:
            prices = requests.get(f"{BACKEND_URL}/price/{ticker}").json()
            if prices:
                # Convert to DataFrame for better charting
                df = pd.DataFrame(prices)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values("date")
                st.write(f"### {ticker.upper()} Price History")
                st.line_chart(df, x="date", y="price", use_container_width=True)
            else:
                st.warning("No price data available")
        except Exception as e:
            st.error(f"Error fetching prices: {str(e)}")

with col2:
    if st.button("Get Latest News", key="news-btn"):
        try:
            news = requests.get(f"{BACKEND_URL}/news/{ticker}").json()
            if news:
                st.write("### Latest News")
                for article in news:
                    with st.expander(article["title"]):
                        st.markdown(f"**Published**: {article['published_date']}")
                        st.markdown(article["summary"])
                        st.markdown(f"[Read more]({article['url']})")
            else:
                st.info("No news found.")
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")

st.divider()

# # Technical Analysis Section
# if st.button("Run Technical Analysis", key="ta-btn"):
#     try:
#         analysis = requests.post(
#             f"{BACKEND_URL}/technical-analysis",
#             json={"ticker": ticker, "days": 30}
#         ).json()
#         with st.expander("📊 Technical Indicators"):
#             df = pd.DataFrame(analysis["indicators"])
#             df['date'] = pd.to_datetime(df['date'])
#             df = df.sort_values("date")
#             st.line_chart(df, x="date", y="price", use_container_width=True)
#         with st.expander("🤖 AI Analysis Summary"):
#             st.markdown(analysis["analysis"])
#     except Exception as e:
#         st.error(f"Analysis error: {str(e)}")

# ... (existing imports and code)

with st.sidebar:

    st.caption("Powered by GenAI & Streamlit")

    # --- Optional Survey Section ---
    st.markdown("### 📝 Feedback (Optional)")
    with st.form(key="feedback_form"):
        satisfaction = st.radio(
            "How satisfied are you with the app?",
            ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"],
            index=1,
        )
        feedback_text = st.text_area("Additional comments (optional):")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            # Compose a single message for the survey endpoint
            survey_message = f"Satisfaction: {satisfaction}. Comments: {feedback_text}"
            # Use a session_id or generate a random one if not logged in
            import uuid
            session_id = st.session_state.get("session_id") or str(uuid.uuid4())
            st.session_state["session_id"] = session_id

            try:
                response = requests.post(
                    f"{BACKEND_URL}/survey",
                    json={
                        "session_id": session_id,
                        "message": survey_message,
                    },
                    timeout=10,
                )
                if response.status_code == 200:
                    st.success("Thank you for your feedback!")
                else:
                    st.warning("Could not submit feedback. Please try again.")
            except Exception as e:
                st.error(f"Error submitting feedback: {e}")

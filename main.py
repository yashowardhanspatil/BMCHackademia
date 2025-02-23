import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

# Load dataset
df = pd.read_csv("merged_data.csv")  # Replace with actual dataset file

# Create LangChain agent for CSV Analysis
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# Define Static Information
STATIC_INFO = """
We evaluated multiple classification models on a dataset with 14,876 samples to compare their accuracy, precision, and recall.

Top-Performing Models (100% Accuracy):
- Random Forest (Accuracy: 1.000)
- Gradient Boosting (Accuracy: 1.000)
- AdaBoost (Accuracy: 1.000)
- XGBoost (Accuracy: 1.000)
- LightGBM (Accuracy: 1.000)
- Decision Tree (Accuracy: 1.000)

High-Performing Models (Above 99% Accuracy):
- K-Nearest Neighbors (Accuracy: 0.9996)
  - Precision (Class 1): 0.93
  - Recall (Class 1): 1.00
- Naïve Bayes (Accuracy: 0.9972)
  - Precision (Class 1): 0.67
  - Recall (Class 1): 1.00
"""

# Create a Prompt Template for Static Info
prompt = PromptTemplate.from_template(
    "You are a helpful assistant that strictly answers based on the given information:\n{info}\n\nUser: {question}\nAssistant:"
)

# Define a Runnable Chain for Static Chatbot
chat_chain = RunnableMap({"question": lambda x: x, "info": lambda _: STATIC_INFO}) | prompt | llm

# Streamlit Layout
st.set_page_config(page_title="AI Chatbot & Data Analysis", layout="wide")
st.title("📊 Data Insights & AI Chatbot 🤖")

# Power BI Dashboard in Main Layout
st.subheader("📈 Power BI Dashboard")
power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiODU3ZTQ3ZWItNWFmMi00MDA2LThkZWUtZTE0ZTBjNjY5NGE5IiwidCI6IjhjNzhjMTIyLWY3ODEtNDUwMC05YzJhLWY2NDVhNzYyODFmNSJ9"
st.markdown(
    f'<iframe title="Power BI Report" width="100%" height="600px" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>',
    unsafe_allow_html=True
)

# Dataset Preview
st.subheader("📜 Dataset Preview")
st.dataframe(df.head())

# Dropdown for Chatbot Selection
chatbot_choice = st.selectbox("Choose Chatbot:", ["📌 Static Info Bot", "📊 CSV Analysis Bot"])

# Predefined Questions (Scrollable)
static_questions = [
    "Which model has the highest accuracy?",
    "What are the precision and recall values for Naïve Bayes?",
    "How does Gradient Boosting perform?",
    "What is the accuracy of the Decision Tree?",
    "Show a comparison between Random Forest and XGBoost.",
]

csv_questions = [
    "What is the average value of column X?",
    "How many unique categories are in column Y?",
    "Show me the top 5 rows sorted by column Z.",
    "Find the highest value in column A.",
    "Get the number of missing values in the dataset.",
]

# Select question from a scrollable list
st.subheader("💡 Suggested Questions")

if chatbot_choice == "📌ML Info Bot":
    selected_question = st.selectbox("Select a predefined question:", ["Type your own"] + static_questions)

elif chatbot_choice == "📊 CSV Analysis Bot":
    selected_question = st.selectbox("Select a predefined question:", ["Type your own"] + csv_questions)

# User Query Input
user_query = ""
if selected_question and selected_question != "Type your own":
    user_query = selected_question
else:
    user_query = st.text_input("Ask a question:")

if chatbot_choice == "📌 Static Info Bot":
    if user_query:
        with st.spinner("🤖 Thinking... Please wait"):
            try:
                response = chat_chain.invoke({"question": user_query})
                clean_response = response.content
                st.success("### 🤖 AI Response:")
                st.write(clean_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

elif chatbot_choice == "📊 CSV Analysis Bot":
    if user_query:
        with st.spinner("📊 Analyzing data... Please wait"):
            try:
                response = agent.run(user_query)
                st.success("### 📊 Data Analysis Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("📌 Built with Streamlit, Power BI & LangChain")

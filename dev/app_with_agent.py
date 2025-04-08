import streamlit as st
import pandas as pd
import openai
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.agents import create_openai_functions_agent

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]


# Helper function to load data (cached)
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df


# Function to perform the dynamic action (like calculating profit margin)
def calculate_profit_margin(df):
    try:
        df["Profit Margin"] = df["Profit"] / df["Sales"] * 100
        return df[["Region", "Category", "Profit Margin"]]
    except Exception as e:
        return str(e)


# LangChain tools setup: A tool to calculate the profit margin
tools = [
    Tool(
        name="Calculate Profit Margin",
        func=calculate_profit_margin,
        description="Given a dataframe, calculate profit margin = (Profit / Sales) * 100.",
    ),
]

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.5, openai_api_key=openai.api_key)

# Set up the LangChain Agent with tools and OpenAI LLM
agent = initialize_agent(
    tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Streamlit file uploader
uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load the uploaded data
    df = load_data(uploaded_file)

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    date_range = st.sidebar.date_input(
        "Order Date Range", [df["Order Date"].min(), df["Order Date"].max()]
    )
    regions = st.sidebar.multiselect(
        "Select Region(s)", df["Region"].unique(), default=df["Region"].unique()
    )
    categories = st.sidebar.multiselect(
        "Select Category", df["Category"].unique(), default=df["Category"].unique()
    )

    # Filter the data based on the user input
    mask = (
        (df["Order Date"] >= pd.to_datetime(date_range[0]))
        & (df["Order Date"] <= pd.to_datetime(date_range[1]))
        & (df["Region"].isin(regions))
        & (df["Category"].isin(categories))
    )
    filtered_df = df[mask]

    # Display dashboard metrics and charts
    st.markdown("### 📈 Key Metrics")
    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Profit"].sum()
    total_orders = filtered_df["Order ID"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Total Profit", f"${total_profit:,.2f}")
    col3.metric("Total Orders", total_orders)

    # AI Summary
    st.markdown("### 🧠 AI-Generated Summary")
    summary_prompt = f"""
    You are a data analyst. Provide a short summary (4-5 sentences) of the following sales data:
    - Total sales: ${total_sales:,.2f}
    - Total profit: ${total_profit:,.2f}
    - Total orders: {total_orders}
    - Selected regions: {", ".join(regions)}
    - Selected categories: {", ".join(categories)}
    - Date range: {date_range[0]} to {date_range[1]}
    Include insights about trends or particularly notable performance or anomalies.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize sales data for dashboards.",
                },
                {"role": "user", "content": summary_prompt},
            ],
        )
        st.markdown(
            f"<div style='background-color:#e6f7e6; padding: 1em; border-radius: 10px;'>{response.choices[0].message['content']}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error generating summary: {e}")

    # Chatbot - Ask about Profit Margin or any other metric
    st.markdown("### 💬 Ask AI About the Data")
    user_question = st.text_input(
        "Ask a question (e.g., Calculate profit margin by region):"
    )

    if user_question:
        try:
            # Use LangChain agent to calculate profit margin based on user question
            result = agent.run(user_question)
            st.write("Agent Response:", result)
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("Please upload a CSV file to get started.")

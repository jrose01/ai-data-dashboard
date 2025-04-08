import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os

# ---------------------
# SETUP
# ---------------------
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Custom Sales Dashboard with AI Insights")

# OpenAI setup
openai.api_key = st.secrets["OPENAI_API_KEY"]  # OR: os.getenv("OPENAI_API_KEY")

# ---------------------
# FILE UPLOAD
# ---------------------
st.sidebar.header("📁 Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, encoding="ISO-8859-1")
    st.success("File uploaded successfully.")
else:
    st.info("Using default sample dataset.")

    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv"
        df = pd.read_csv(url, encoding="ISO-8859-1", parse_dates=["Order Date"])
        return df

    df = load_data()

# ---------------------
# CHECK REQUIRED COLUMNS
# ---------------------
required_cols = [
    "Order Date",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
    "Profit",
    "Order ID",
]
if not all(col in df.columns for col in required_cols):
    st.error(f"Dataset must contain the following columns: {', '.join(required_cols)}")
    st.stop()

# Ensure datetime format
df["Order Date"] = pd.to_datetime(df["Order Date"])

# ---------------------
# SIDEBAR FILTERS
# ---------------------
st.sidebar.header("🔍 Filters")
min_date, max_date = df["Order Date"].min(), df["Order Date"].max()
date_range = st.sidebar.date_input("Order Date Range", [min_date, max_date])

regions = st.sidebar.multiselect(
    "Select Region(s)", options=df["Region"].unique(), default=df["Region"].unique()
)
categories = st.sidebar.multiselect(
    "Select Category", options=df["Category"].unique(), default=df["Category"].unique()
)

# Filter data
mask = (
    (df["Order Date"] >= pd.to_datetime(date_range[0]))
    & (df["Order Date"] <= pd.to_datetime(date_range[1]))
    & (df["Region"].isin(regions))
    & (df["Category"].isin(categories))
)
filtered_df = df[mask]

# ---------------------
# key performance indicators
# ---------------------
st.markdown("### 📈 Key Peformance Indicators")
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Total Profit", f"${total_profit:,.2f}")
col3.metric("Total Orders", total_orders)

# ---------------------
# VISUALIZATIONS
# ---------------------
st.markdown("### 📊 Sales & Profit Over Time")
sales_trend = filtered_df.groupby("Order Date")[["Sales", "Profit"]].sum().reset_index()
fig_trend = px.line(
    sales_trend, x="Order Date", y=["Sales", "Profit"], title="Sales & Profit Over Time"
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("### 🧩 Sales by Category and Sub-Category")
cat_sales = (
    filtered_df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
)
fig_bar = px.bar(
    cat_sales,
    x="Sub-Category",
    y="Sales",
    color="Category",
    title="Sales by Category/Sub-Category",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("### 🌍 Regional Sales Distribution")
region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
fig_pie = px.pie(region_sales, names="Region", values="Sales", title="Sales by Region")
st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------
# OPENAI-GENERATED SUMMARY
# ---------------------
st.markdown("### 🤖 AI-Generated Summary")
summary_prompt = f"""
You are a data analyst. Provide a short summary (2-3 sentences) of the following filtered sales data:
- Total sales: ${total_sales:,.2f}
- Total profit: ${total_profit:,.2f}
- Total orders: {total_orders}
- Selected regions: {", ".join(regions)}
- Selected categories: {", ".join(categories)}
- Date range: {date_range[0]} to {date_range[1]}

Include observations about trends or notable insights.
"""

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You summarize sales data for business dashboards.",
            },
            {"role": "user", "content": summary_prompt},
        ],
        temperature=0.5,
    )
    summary = response["choices"][0]["message"]["content"]
    st.success(summary)
except Exception as e:
    st.error(f"Error generating summary: {e}")

# ---------------------
# CHATBOT ASSISTANT
# ---------------------
st.markdown("### 💬 Ask AI About the Data")
user_question = st.text_input(
    "Ask a question (e.g., 'Which sub-category had the lowest profit?')"
)

if user_question:
    csv_preview = filtered_df.head(300).to_csv(index=False)
    chat_prompt = f"""
You are a helpful data analyst. The user is asking about sales data. Answer the following question based on this CSV data:

{csv_preview}

Question: {user_question}
"""

    try:
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You analyze and answer questions about sales data from CSV.",
                },
                {"role": "user", "content": chat_prompt},
            ],
            temperature=0.3,
        )
        answer = chat_response["choices"][0]["message"]["content"]
        st.info(answer)
    except Exception as e:
        st.error(f"Error in chatbot response: {e}")

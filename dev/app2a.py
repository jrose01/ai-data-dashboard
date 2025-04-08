import streamlit as st
import pandas as pd
import plotly.express as px
import openai
from openai import OpenAI

# Set Streamlit page layout
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Custom Sales Dashboard with AI Insights")

# Get your OpenAI key from .streamlit/secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI()

# CSV Upload
st.sidebar.header("📁 Upload Your CSV")
st.sidebar.file_uploader("Choose a CSV file", type=["csv"])


@st.cache_data
def load_data(dataset):
    df = pd.read_csv(dataset, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


if uploaded_file:
    try:
        df = load_data(uploaded_file)
        # Required columns
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
            st.error(
                "Dataset must contain the following columns: "
                + ", ".join(required_cols)
            )
            st.stop()

        # Parse date
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

        # Sidebar Filters
        st.sidebar.header("🔍 Filters")
        min_date, max_date = df["Order Date"].min(), df["Order Date"].max()
        date_range = st.sidebar.date_input("Order Date Range", [min_date, max_date])

        regions = st.sidebar.multiselect(
            "Select Region(s)", df["Region"].unique(), default=df["Region"].unique()
        )
        categories = st.sidebar.multiselect(
            "Select Category", df["Category"].unique(), default=df["Category"].unique()
        )

        # Filtered data
        mask = (
            (df["Order Date"] >= pd.to_datetime(date_range[0]))
            & (df["Order Date"] <= pd.to_datetime(date_range[1]))
            & (df["Region"].isin(regions))
            & (df["Category"].isin(categories))
        )
        filtered_df = df[mask]

        # KPIs
        st.markdown("### 📈 Key Metrics")
        total_sales = filtered_df["Sales"].sum()
        total_profit = filtered_df["Profit"].sum()
        total_orders = filtered_df["Order ID"].nunique()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Total Profit", f"${total_profit:,.2f}")
        col3.metric("Total Orders", total_orders)

        # Visualizations
        st.markdown("### 📊 Sales & Profit Over Time")
        sales_trend = (
            filtered_df.groupby("Order Date")[["Sales", "Profit"]].sum().reset_index()
        )
        fig_trend = px.line(
            sales_trend,
            x="Order Date",
            y=["Sales", "Profit"],
            title="Sales & Profit Over Time",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("### 🧩 Sales by Category and Sub-Category")
        cat_sales = (
            filtered_df.groupby(["Category", "Sub-Category"])["Sales"]
            .sum()
            .reset_index()
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
        fig_pie = px.pie(
            region_sales, names="Region", values="Sales", title="Sales by Region"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # OpenAI Summary
        st.markdown("###  AI-Generated Summary")
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

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize sales data for business dashboards.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.3,
        )
        summary = response.choices[0].message.content
        st.markdown(
            f"<div style='background-color:#e6f7e6; padding: 1em; border-radius: 10px;'>{summary}</div>",
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Error: {e}")

# Chat Assistant
st.markdown("### 💬 Ask AI About the Data")
user_question = st.chat_input("Ask a question about the uploaded data:")

if user_question:
    try:
        # Reduce size for efficiency — you can tune this
        csv_preview = filtered_df.head(50).to_csv(index=False)

        chat_prompt = f"""
        You are a helpful data analyst. Answer the user's question using the sales data provided below.
        Be concise, clear, and use numerical evidence where possible.

        Question:
        {user_question}

        Here are the first few rows of the filtered dataset (CSV format):
        {csv_preview}
        """

        chat_response = client.chat.completions.create(
            model="gpt-4o-mini",  # change model here if needed
            messages=[
                {
                    "role": "system",
                    "content": "You analyze and answer questions about tabular sales data.",
                },
                {"role": "user", "content": chat_prompt},
            ],
            temperature=0.3,
        )

        answer = chat_response.choices[0].message.content
        st.info(answer)

    except Exception as e:
        st.error(f"Chatbot error: {e}")

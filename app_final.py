import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set page config
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("📊 Custom Sales Dashboard with AI Insights")


@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    return df


def filter_data(df, date_range, regions, categories):
    mask = (
        (df["Order Date"] >= pd.to_datetime(date_range[0]))
        & (df["Order Date"] <= pd.to_datetime(date_range[1]))
        & (df["Region"].isin(regions))
        & (df["Category"].isin(categories))
    )
    return df[mask]


def display_dashboard(filtered_df):
    st.markdown("### 📈 Key Metrics")
    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Profit"].sum()
    total_orders = filtered_df["Order ID"].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${total_sales:,.2f}")
    col2.metric("Total Profit", f"${total_profit:,.2f}")
    col3.metric("Total Orders", total_orders)

    st.markdown("### 📊 Sales & Profit Over Time")
    sales_trend = (
        filtered_df.groupby("Order Date")[["Sales", "Profit"]].sum().reset_index()
    )
    fig_trend = px.line(sales_trend, x="Order Date", y=["Sales", "Profit"])
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### 🧩 Sales by Category and Sub-Category")
    cat_sales = (
        filtered_df.groupby(["Category", "Sub-Category"])["Sales"].sum().reset_index()
    )
    fig_bar = px.bar(cat_sales, x="Sub-Category", y="Sales", color="Category")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🌍 Regional Sales Distribution")
    region_sales = filtered_df.groupby("Region")["Sales"].sum().reset_index()
    fig_pie = px.pie(region_sales, names="Region", values="Sales")
    st.plotly_chart(fig_pie, use_container_width=True)

    return total_sales, total_profit, total_orders


def generate_summary(
    total_sales, total_profit, total_orders, date_range, regions, categories
):
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize sales data for dashboards.",
                },
                {"role": "user", "content": summary_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"


def chat_about_data(filtered_df):
    st.markdown("### 💬 Ask AI About the Data")
    user_question = st.chat_input("Ask a question about the uploaded data:")

    if user_question:
        try:
            csv_preview = filtered_df.head(50).to_csv(index=False)
            chat_prompt = f"""
            You are a helpful data analyst. Answer the user's question using the data below.

            Question:
            {user_question}

            Here are the first few rows of the filtered dataset:
            {csv_preview}
            """

            chat_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You answer questions about CSV sales data.",
                    },
                    {"role": "user", "content": chat_prompt},
                ],
                temperature=0.3,
            )
            st.info(chat_response.choices[0].message.content)
        except Exception as e:
            st.error(f"Chatbot error: {e}")


# ---- Main App Flow ----

uploaded_file = st.sidebar.file_uploader("📁 Upload a CSV file", type=["csv"])

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
            st.error("Missing required columns.")
            st.stop()

        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        min_date, max_date = df["Order Date"].min(), df["Order Date"].max()
        date_range = st.sidebar.date_input("Order Date Range", [min_date, max_date])
        regions = st.sidebar.multiselect(
            "Select Region(s)", df["Region"].unique(), default=df["Region"].unique()
        )
        categories = st.sidebar.multiselect(
            "Select Category", df["Category"].unique(), default=df["Category"].unique()
        )

        # Filter and Display
        filtered_df = filter_data(df, date_range, regions, categories)
        total_sales, total_profit, total_orders = display_dashboard(filtered_df)

        # AI Summary
        st.markdown("### 🧠 AI-Generated Summary")
        summary = generate_summary(
            total_sales, total_profit, total_orders, date_range, regions, categories
        )
        st.markdown(
            f"<div style='background-color:#e6f7e6; padding: 1em; border-radius: 10px;'>{summary}</div>",
            unsafe_allow_html=True,
        )

        # Chatbot
        chat_about_data(filtered_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

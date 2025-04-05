import os
import io
import contextlib
import re
from dotenv import find_dotenv, load_dotenv

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import sklearn as sk

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Title
st.title("AI-Powered Data Analysis Assistant")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
question = st.text_input("What would you like to know about your data?")

# Initialize OpenAI LLM
template = """
You are a Python data analysis assistant. A user has uploaded the following dataset sample:

{data_sample}

The user asked: "{question}"

Please determine the most appropriate data analysis (e.g., regression, classification, survival analysis), 
explain the method, and generate Python code using the uploaded dataset, which is already loaded in a variable 
called `df`. 

Using the uploaded dataset called df, to suggest descriptive statistics and when approapriate, 
please provide suggestions for statistical significance tests. Only include the explanation and the code 
in the suggestions output.

Make sure the when you run the code that you interpret the result using print() or st.write(). Round numeric 
output to 2 decimal places.

"""
prompt = PromptTemplate(input_variables=["data_sample", "question"], template=template)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_chain = prompt | llm

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Fix Arrow serialization issues
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.write("Preview of your data:")
    st.dataframe(df.head())
    st.write(f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    if st.checkbox("Show all columns"):
        st.write("Columns:", list(df.columns))

    if st.checkbox("Show full dataset"):
        st.dataframe(df)

    if st.checkbox("Show summary statistics"):
        st.write(df.describe(include="all"))

    if question and st.button("Generate Analysis"):
        with st.spinner("Working on it..."):
            data_sample = df.head(10).to_csv(index=False)
            result = llm_chain.invoke(
                {"data_sample": data_sample, "question": question}
            )

        st.markdown("### Suggested Analysis & Code")
        output_text = result.content if hasattr(result, "content") else result
        st.markdown(output_text if output_text else "*No response received from LLM*")
        st.session_state.generated_code = output_text

# Code Execution
if "generated_code" in st.session_state:
    if st.button("Run the code"):
        result = st.session_state.generated_code

        # Extract Python code block from the result
        code_match = re.search(r"```python(.*?)```", result, re.DOTALL)
        code_to_run = code_match.group(1).strip() if code_match else result

        st.markdown("### Code Output")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                exec(
                    code_to_run, {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
                )
                output = f.getvalue()
                st.text(output)
            except Exception as e:
                st.error(f"Error running code: {e}")

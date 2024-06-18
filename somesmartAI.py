import streamlit as st
import pandas as pd
from transformers import TapexTokenizer, BartForConditionalGeneration
from io import StringIO

# Load TAPEX model (within Streamlit script)
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

st.title("Table Question Answering")

# Text Input for Table Data
table_data = st.text_area("Enter your table data (comma or tab-separated):")

if table_data:
    # Attempt to Read Table
    try:
        table = pd.read_csv(StringIO(table_data))  # Try reading as CSV
    except pd.errors.ParserError:
        try:
            table = pd.read_csv(StringIO(table_data), sep="\t")  # Try tab-separated
        except pd.errors.ParserError:
            st.error("Invalid table format. Please use comma or tab-separated values.")
            table = None

    # If Table Read Successfully
    if table is not None:
        st.subheader("Your Table:")
        st.table(table)  

        # Question Answering Loop
        while True:
            query = st.text_input("Ask a question about the table (or type 'exit' to quit):")

            if query.lower() == "exit":
                break

            # Generate Answer (with error handling)
            try:
                encoding = tokenizer(table=table, query=query, return_tensors="pt")
                outputs = model.generate(**encoding)
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                st.write(f"\nAnswer: {answer[0]}\n")
            except Exception as e:
                st.error(f"Error generating answer: {e}")

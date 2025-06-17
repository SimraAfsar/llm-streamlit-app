import streamlit as st
import openai
import os
import pandas as pd
from datasets import load_from_disk

# Load API Key from Streamlit secrets
api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)

# Load local dataset from disk
try:
    dataset_dict = load_dataset()("question_answering")
    dataset = dataset_dict["train"]  # access the 'train' split
    small_dataset = dataset.select(range(1000))

except Exception as e:
    st.error("Failed to load dataset. Make sure 'question_answering' folder exists.")
    st.stop()

# Function to retrieve relevant context
def search_dataset(query, dataset):
    results = []
    for example in dataset:
        try:
            question = example.get('question', '')
            context = example.get('context', '')

            if query.lower() in question.lower():
                results.append(context)
        except Exception:
            continue
    return results[:1]

# Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

# Title and input
st.title("🧠 Simra's LLM Assistant")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    retrieved_context = search_dataset(question, small_dataset)

    if retrieved_context:
        full_prompt = f"Context: {retrieved_context[0]}\n\nQuestion: {question}"
    else:
        full_prompt = question

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}]
        )
        st.session_state.llm_response = response.choices[0].message.content
    except Exception as e:
        st.error("LLM request failed. Check your API key and model access.")
        st.stop()

# Display LLM output
if st.session_state.llm_response:
    st.write("### LLM Response:")
    st.write(st.session_state.llm_response)

    rating = st.slider("Rate the Response (1-5)", 1, 5)
    comment = st.text_input("Leave a Comment about the Response")

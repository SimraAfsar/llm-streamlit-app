import streamlit as st
import openai
import os
import pandas as pd
from datasets import load_from_disk

# Load API Key from Streamlit secrets
api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)

# Log and Load dataset from local directory
try:
    dataset = load_from_disk("question_answering")
    small_dataset = dataset.select(range(1000))
    st.success("Dataset loaded successfully.")
    st.info("Subset used: Custom local ELOQUENCE subset")
except Exception:
    st.error("Failed to load dataset. Make sure the 'question_answering' folder exists in the same directory.")
    st.stop()

# Function to find relevant context
def search_dataset(query, dataset):
    results = []
    for example in dataset:
        try:
            question = example.get("question")
            context = example.get("context")

            if question and context and query.lower() in question.lower():
                results.append(context)
        except Exception:
            continue
    return results[:1]


# Initialise session state
if "history" not in st.session_state:
    st.session_state.history = []

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

# UI input
st.title("Simra's LLM Assistant")
question = st.text_input("Enter your question:")

if st.button("Ask") and question:
    context = search_dataset(question, small_dataset)
    prompt = f"Context: {context[0]}\n\nQuestion: {question}" if context else question

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        st.session_state.llm_response = response.choices[0].message.content
    except Exception:
        st.error("LLM request failed. Check your API key and model access.")
        st.stop()

# Display response
if st.session_state.llm_response:
    st.subheader("LLM Response")
    st.write(st.session_state.llm_response)

    rating = st.slider("Rate the Response (1-5)", 1, 5)
    comment = st.text_input("Leave a Comment about the Response")

    if st.button("Submit Evaluation"):
        st.session_state.history.append({
            "Prompt": question,
            "Response": st.session_state.llm_response,
            "Rating": rating,
            "Comment": comment
        })
        st.success("Your evaluation has been submitted.")

# Show history
if st.session_state.history:
    st.subheader("Evaluation History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    avg = df["Rating"].mean()
    st.write(f"Average Rating: {avg:.2f}")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download History as CSV", csv, "evaluation_history.csv", "text/csv")

import streamlit as st
import openai
import os
import pandas as pd
from datasets import load_dataset

# Load API Key from Streamlit secrets
api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)

# Load dataset from JSON file (must be in the same folder as this app)
try:
    dataset = load_dataset("json", data_files="eloquence_data.json")["train"]
    small_dataset = dataset.select(range(1000))
except Exception as e:
    st.error("❌ Failed to load dataset. Make sure 'eloquence_data.json' is present in the same folder.")
    st.stop()

# Search function
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

# Session state init
if "history" not in st.session_state:
    st.session_state.history = []

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

# UI
st.title("🧠 Simra's LLM Assistant")
question = st.text_input("Enter your question:")

if st.button("Ask") and question.strip():
    retrieved_context = search_dataset(question, small_dataset)

    full_prompt = (
        f"Context: {retrieved_context[0]}\n\nQuestion: {question}"
        if retrieved_context else question
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": full_prompt}]
        )
        st.session_state.llm_response = response.choices[0].message.content
    except Exception:
        st.error("❌ LLM request failed. Check your API key and internet connection.")
        st.stop()

# Display LLM output
if st.session_state.llm_response:
    st.write("### 🤖 LLM Response:")
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
        st.success("✅ Your evaluation has been submitted.")

# Show history
if st.session_state.history:
    st.write("### 📊 Evaluation History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    avg_rating = df["Rating"].mean()
    st.write(f"**Average Rating:** {avg_rating:.2f}")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Evaluation History as CSV",
        data=csv,
        file_name="evaluation_history.csv",
        mime="text/csv"
    )

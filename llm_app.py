import streamlit as st
import openai
import os
import pandas as pd
from datasets import load_from_disk

# Load API Key from Streamlit secrets
api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)

# Load local dataset from folder
try:
    dataset = load_from_disk("question_answering")  # Make sure this folder is in the same directory as your script
    small_dataset = dataset.select(range(1000))
except Exception as e:
    st.error("Failed to load dataset. Make sure the 'question_answering' folder exists in the same directory.")
    st.stop()

# Function to retrieve relevant context
def search_dataset(query, dataset):
    results = []
    for example in dataset:
        try:
            question = example["question"]
            context = example["context"]
            if query.lower() in question.lower():
                results.append(context)
        except Exception:
            continue
    return results[:1]

# Streamlit session state setup
if "history" not in st.session_state:
    st.session_state.history = []

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

# UI
st.title("Simra's LLM Assistant")
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

# Show LLM response
if st.session_state.llm_response:
    st.write("LLM Response:")
    st.write(st.session_state.llm_response)

    rating = st.slider("Rate the Response (1-5)", 1, 5)
    comment = st.text_input("Leave a Comment about the Response")

    if st.button("Submit Evaluation"):
        st.session_state.history.append({
            'Prompt': question,
            'Response': st.session_state.llm_response,
            'Rating': rating,
            'Comment': comment
        })
        st.success("Your evaluation has been submitted.")

# Show evaluation history
if st.session_state.history:
    st.write("Evaluation History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    avg_rating = df['Rating'].mean()
    st.write(f"Average Rating so far: {avg_rating:.2f}")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Evaluation History as CSV",
        data=csv,
        file_name='evaluation_history.csv',
        mime='text/csv'
    )

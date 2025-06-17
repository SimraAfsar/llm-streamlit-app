import streamlit as st
import openai
import os
import pandas as pd
from datasets import load_dataset

# Load API Key from secrets
api_key = st.secrets["api_key"]
client = openai.OpenAI(api_key=api_key)

from datasets import load_from_disk
dataset = load_from_disk("question_answering")
small_dataset = dataset.select(range(1000))

# Search function with fixed iteration
def search_dataset(query, dataset):
    results = []
    for i in range(len(dataset)):
        example = dataset[i]
        if query.lower() in example['question'].lower():
            results.append(example['context'])
    return results[:1]

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

# UI
st.title("🧠 Simra's LLM Assistant")

question = st.text_input("Enter your question:")

if st.button("Ask"):
    retrieved_context = search_dataset(question, small_dataset)

    if retrieved_context:
        full_prompt = f"Context: {retrieved_context[0]}\n\nQuestion: {question}"
    else:
        full_prompt = question

    # Get LLM response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}]
    )

    st.session_state.llm_response = response.choices[0].message.content

# Display response
if st.session_state.llm_response:
    st.write("### LLM Response:")
    st.write(st.session_state.llm_response)

    rating = st.slider('Rate the Response (1-5)', 1, 5)
    comment = st.text_input('Leave a Comment about the Response')

    if st.button('Submit Evaluation'):
        st.session_state.history.append({
            'Prompt': question,
            'Response': st.session_state.llm_response,
            'Rating': rating,
            'Comment': comment
        })
        st.success('Your evaluation has been submitted.')

# Show evaluation history
if st.session_state.history:
    st.write("### Evaluation History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    avg_rating = df['Rating'].mean()
    st.write(f"**Average Rating so far:** {avg_rating:.2f}")

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Evaluation History as CSV",
        data=csv,
        file_name='evaluation_history.csv',
        mime='text/csv'
    )

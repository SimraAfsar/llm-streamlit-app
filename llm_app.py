import streamlit as st
import openai
import os
import pandas as pd

#Load API Keys
api_key = st.secrets["api_key"]
client = openai.OpenAI(
    api_key=api_key
)

from datasets import load_dataset
dataset = load_dataset("squad")
small_dataset = dataset['train'].select(range(1000))  

def search_dataset(query, dataset):
    results = []
    for example in dataset:
        if query.lower() in example['question'].lower():
            results.append(example['context'])
    return results[:1]

if "history" not in st.session_state:
    st.session_state.history = []

if "llm_response" not in st.session_state:
    st.session_state.llm_response = ""

st.title("🧠 Simra's LLM Assistant")

if "history" not in st.session_state:
    st.session_state.history = [] 

# Text input box
question = st.text_input("Enter your question:")

# Button
if st.button("Ask"):
    # Search the dataset for related context
    retrieved_context = search_dataset(question, dataset)

    # Check if any context is found
    if retrieved_context:
        full_prompt = f"Context: {retrieved_context[0]}\n\nQuestion: {question}"
    else:
        full_prompt = question  # fallback if no match

    # LLM call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    
    # Save the response
    st.session_state.llm_response = response.choices[0].message.content

# Now display it ONLY if it exists
if st.session_state.llm_response:
    st.write("### LLM Response:")
    st.write(st.session_state.llm_response)


    # Rating slider
    rating = st.slider('Rate the Response (1-5)', 1, 5)

    if "llm_response" not in st.session_state:
        st.session_state.llm_response = None

    # Comment box
    comment = st.text_input('Leave a Comment about the Response')

    # Save interaction when user clicks a button
    if st.button('Submit Evaluation'):
        st.session_state.history.append({
            'Prompt': question,
            'Response': llm_response,
            'Rating': rating,
            'Comment': comment
        })
        st.success('Your evaluation has been submitted.')

# Show history if available
if st.session_state.history:
    st.write("### Evaluation History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    # Show average rating
    avg_rating = df['Rating'].mean()
    st.write(f"**Average Rating so far:** {avg_rating:.2f}")

    # Option to download history
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Evaluation History as CSV",
        data=csv,
        file_name='evaluation_history.csv',
        mime='text/csv',
    )


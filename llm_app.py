import streamlit as st
import openai
import os


api_key = st.secrets["api_key"]
client = openai.OpenAI(
    api_key=api_key
)

st.title("🧠 Ask the LLM")

# Text input box
question = st.text_input("Enter your question:")

# Button
if st.button("Ask"):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": question}
        ]
    )

    st.write(response.choices[0].message.content)

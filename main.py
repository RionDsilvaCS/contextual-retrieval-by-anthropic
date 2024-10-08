import streamlit as st
import requests
import os
import time
import json
from dotenv import load_dotenv
load_dotenv()

st.title("RAG with Contextual Retrieval")
st.markdown("Implemented in Llama-index 🦙")
st.markdown("Link to the Anthropic [blog post](https://www.anthropic.com/news/contextual-retrieval)")

# Streaming response from API call
def response_generator(query):
    url = os.getenv("API_URL")
    s = requests.Session()
    data = {"query": query}
    with s.post(url, data=json.dumps(data), headers=None, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                yield str(line.decode('utf-8')) + "\n"

# Function for first None request
def fake_data():
    _LOREM_IPSUM = "Hi !!! I am your personal recipe assistant. How can i help you ?"
    for word in _LOREM_IPSUM.split(" "):
        yield word + " "
        time.sleep(0.05)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Saving messages state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Type your questions here !!!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

# Input passed to API
with st.chat_message("assistant"):
    if prompt is not None:
        response = st.write_stream(response_generator(str(prompt)))
    else:
        response = st.write_stream(fake_data)

st.session_state.messages.append({"role": "assistant", "content": response})

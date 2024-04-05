import streamlit as st
import ollama

st.title("💬 llama2 (7B) Chatbot")

# Initialize the selected option and document in session state if not already done
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None
if 'uploaded_document' not in st.session_state:
    st.session_state['uploaded_document'] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Add a sidebar with a drop-down menu for options
selected_option = st.sidebar.selectbox(
    "Choose an option:",
    ("Risk Analysis", "Financial Sentiment Analysis", "Financial NER", "Financial Visual Data Analysis", "DocQA")
)

# If "DocQA" is selected, provide an option to upload a document
if selected_option == "DocQA":
    doc = st.sidebar.file_uploader("Upload a document for analysis", type=['pdf', 'docx', 'txt'])
    if doc is not None:
        st.session_state['uploaded_document'] = doc
        st.session_state["messages"].append({"role": "assistant", "content": "Document uploaded successfully."})

# Define a dictionary mapping options to models
select_dict = {
    "Risk Analysis": 'mistral-risk',
    "Financial Sentiment Analysis": "fin-sentiment",
    "Financial NER": "mistral-NER",
    "Financial Visual Data Analysis": "mistral:instruct",
    "DocQA": "mistral:instruct"
}

# Check if the option has just been selected or changed, and update the session state
if st.session_state['selected_option'] != selected_option:
    st.session_state['selected_option'] = selected_option
    st.session_state["messages"].append({"role": "assistant", "content": f"Selected ChatBot: {selected_option}", "type": "subheader"})

# Write Message History
for msg in st.session_state.messages:
    if 'type' in msg and msg['type'] == 'subheader':
        st.subheader(msg["content"])
    elif msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# Generator for Streaming Tokens
def generate_response(model):
   # thinking_message = st.empty()  # Create an empty element to hold the "Thinking..." message
   # thinking_message.write("Thinking...")
    response = ollama.chat(model=model, stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

# Chat input for user messages
if prompt := st.chat_input():
    model = select_dict[selected_option]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    #st.chat_message("assistant", avatar="🤖").write("Dummy Response")
    stop_button = st.button("Stop Generating")
    if stop_button:
        st.stop()
    else:
        st.chat_message("assistant", avatar="🤖").write_stream(generate_response(model))
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})
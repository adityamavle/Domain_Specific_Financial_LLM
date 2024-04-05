import streamlit as st
import ollama

st.title("💬 llama2 (7B) Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = None

# Function to add a message to the chat
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖").write(content)

# Add a sidebar with a drop-down menu for options
selected_option = st.sidebar.selectbox(
    "Choose an option:",
    ("Risk Analysis", "Financial Sentiment Analysis", "Financial NER", "Financial Visual Data Analysis"),
    index=0
)
select_dict = {
    "Risk Analysis":'fin-risk',
    "Financial Sentiment Analysis":"fin-sentiment",
    "Financial NER":"fin-NER",
    "Financial Visual Data Analysis":"llama2"}
# If a new option is selected, add a subheader to the chat
if selected_option and selected_option != st.session_state["selected_option"]:
    st.session_state["selected_option"] = selected_option
    add_message("assistant", f"*{selected_option}*")

# Write Message History
def write_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            # Check if the message is a subheader and should be bold
            if msg["content"].startswith("*") and msg["content"].endswith("*"):
                st.subheader(msg["content"].strip("*"))
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

write_messages()

# Generator for Streaming Tokens
def generate_response(model):
    response = ollama.chat(model=model, stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

# Chat input for user messages
if prompt := st.chat_input():
    model = select_dict[selected_option]
    add_message("user", prompt)
    # Generate the response and add only the final message to the chat
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response(model))
    # for _ in generate_response():
    #     pass  # The response is being built in the background
    # Once the response is complete, append it to the chat
    add_message("assistant", st.session_state["full_message"])
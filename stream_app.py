import streamlit as st
import ollama

st.title("💬 llama2 (7B) Chatbot")

# Initialize the selected option in session state if not already done
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Add a sidebar with a drop-down menu for options
selected_option = st.sidebar.selectbox(
    "Choose an option:",
    ("Risk Analysis", "Financial Sentiment Analysis", "Financial NER", "Financial Visual Data Analysis")
)
select_dict = {
    "Risk Analysis":'fin-risk',
    "Financial Sentiment Analysis":"fin-sentiment",
    "Financial NER":"fin-NER",
    "Financial Visual Data Analysis":"llama2"}
# If the option has just been selected or changed, update the session state
if st.session_state['selected_option'] != selected_option:
    st.session_state['selected_option'] = selected_option
    st.session_state["messages"].append({"role": "assistant", "content": f"You have selected {selected_option}"})

# If an option is selected, display the section header and all messages below it
if st.session_state['selected_option']:
    st.subheader(st.session_state['selected_option'])
    
    ### Write Message History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

## Generator for Streaming Tokens
def generate_response(model):
    response = ollama.chat(model=model, stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

if prompt := st.chat_input():
    model = select_dict[selected_option]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response(model))
    # for token in generate_response():
    #     pass  # The tokens are being added to 'full_message'
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})
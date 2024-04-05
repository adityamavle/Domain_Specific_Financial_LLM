import streamlit as st
import ollama

st.title("💬 llama2 (7B) Chatbot")

# Initialize session state variables if they don't exist
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None
if 'uploaded_document' not in st.session_state:
    st.session_state['uploaded_document'] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "full_message" not in st.session_state:
    st.session_state["full_message"] = ""
if "stop_pressed" not in st.session_state:
    st.session_state["stop_pressed"] = False

# Function to reset the stop flag before each new message generation
def reset_stop_flag():
    st.session_state["stop_pressed"] = False

# Define a dictionary mapping options to models
select_dict = {
    "Risk Analysis": 'mistral-risk',
    "Financial Sentiment Analysis": "fin-sentiment",
    "Financial NER": "mistral-NER",
    "Financial Visual Data Analysis": "mistral:instruct",
    "DocQA": "mistral:instruct"
}

selected_option = st.sidebar.selectbox(
    "Choose an option:",
    options=list(select_dict.keys()),
    on_change=reset_stop_flag
)

# Define the generator function for message generation
def generate_response(model):
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
    st.session_state["full_message"] = ""  # Reset the full_message before generating a new one
    response_placeholder = st.empty()  # Placeholder to display response

    stop_button = st.button("Stop Generating")
    if stop_button:
        st.session_state["stop_pressed"] = True  # Set the flag when button is pressed

    for token in generate_response(model):  # Generate and display response in real-time
        if st.session_state["stop_pressed"]:
            break  # Exit loop if stop button was pressed
        response_placeholder.text(st.session_state["full_message"])  # Update placeholder with the current message

    # Once out of the loop, optionally display the (partial) response in a new area/widget
    st.text_area("Generated Response:", value=st.session_state["full_message"], height=150, disabled=True)
    st.session_state["stop_pressed"] = False  # Reset the stop flag for 
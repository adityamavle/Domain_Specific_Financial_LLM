import os
import tempfile
import streamlit as st
# from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="DocQA")

def display_messages():
    st.subheader("Chat")
    for i, msg_obj in enumerate(st.session_state["messages"]):
        role = msg_obj["role"]
        content = msg_obj["content"]
        if role == "user":
            st.write(f"🧑‍💻 User: {content}")
        elif role == "assistant":
            st.write(f"🤖 Assistant: {content}")
        elif role == "info":
            st.info(content)
        elif role == "warning":
            st.warning(content)
        elif role == "error":
            st.error(content)
        elif role == "success":
            st.success(content)
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append({"role": "user", "content": user_text})
        st.session_state["messages"].append({"role": "assistant", "content": agent_text})

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def page():
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


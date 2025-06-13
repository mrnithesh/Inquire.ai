import streamlit as st
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage

from src.config.constants import CHROMA_DIR
from src.utils.document_utils import (
    create_project,
    list_projects,
    load_and_chunk,
    get_vectorstore,
    add_docs_to_vectorstore
)
from src.models.agent import get_agent_executor

def main():
    st.set_page_config(
        page_title="Research Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🔍"
    )
    st.header("🤖 Research Assistant")

    # Project selection/creation
    st.sidebar.header("Projects")
    projects = list_projects()
    project = st.sidebar.selectbox("Select Project", projects + ["+ Create new project"])
    if project == "+ Create new project":
        new_project = st.sidebar.text_input("New project name")
        if st.sidebar.button("Create") and new_project:
            create_project(new_project)
            st.rerun()
        st.stop()
    else:
        st.session_state["project"] = project

    # Upload documents
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        if "project" not in st.session_state:
            st.sidebar.error("Please select or create a project first.")
        else:
            project = st.session_state["project"]
            for uploaded_file in uploaded_files:
                suffix = uploaded_file.name.split(".")[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    docs = load_and_chunk(tmp.name, "pdf" if suffix == "pdf" else "txt")
                    add_docs_to_vectorstore(project, docs)
                os.unlink(tmp.name)
            st.sidebar.success("Documents processed and added to project.")
            # Optionally clear chat history when new docs are added
            project_key = f"chat_history_{project}"
            if project_key in st.session_state:
                del st.session_state[project_key]

    # Chat history state
    project_key = f"chat_history_{st.session_state.get('project', 'default')}"
    if project_key not in st.session_state:
        st.session_state[project_key] = []

    st.write(f"Ask questions about your documents in '{st.session_state.get('project', 'N/A')}'")

    # Display chat history
    for message in st.session_state[project_key]:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)

    query = st.chat_input("Enter your question")

    if query:
        if "project" not in st.session_state:
            st.error("Please select or create a project first.")
        else:
            project = st.session_state["project"]
            vectorstore = get_vectorstore(project)
            agent_invoker = get_agent_executor(vectorstore)

            st.chat_message("user").write(query)
            
            # Add user message to chat history as proper LangChain Message object
            user_message = HumanMessage(content=query)
            st.session_state[project_key].append(user_message)
            
            with st.spinner("Thinking..."):
                try:
                    # Use our custom invoker function that properly formats the query
                    result = agent_invoker(query, st.session_state[project_key])
                    
                    # Extract the assistant's response
                    if "messages" in result:
                        assistant_messages = [msg for msg in result["messages"] if msg.type == "ai"]
                        if assistant_messages:
                            answer = assistant_messages[-1].content
                        else:
                            answer = "I couldn't generate a proper response."
                    
                    st.chat_message("assistant").write(answer)
                    
                    # Add assistant response to chat history
                    assistant_message = AIMessage(content=answer)
                    st.session_state[project_key].append(assistant_message)
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state[project_key].append(AIMessage(content=f"I encountered an error: {error_msg}"))

if __name__ == "__main__":
    main() 
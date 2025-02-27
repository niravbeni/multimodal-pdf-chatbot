"""
Utility helper functions for the Streamlit app
"""
import os
import uuid
import traceback
import streamlit as st
import joblib
from langchain.schema.document import Document
from langchain.schema import HumanMessage

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    temp_dir = "./temp_pdf_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def ensure_document_format(items):
    """
    Ensure that items are in the correct Document format
    """
    documents = []
    for item in items:
        if isinstance(item, Document):
            # Already a Document object
            documents.append(item)
        elif hasattr(item, "page_content") and hasattr(item, "metadata"):
            # Object with Document-like attributes
            documents.append(Document(
                page_content=item.page_content,
                metadata=item.metadata
            ))
        elif isinstance(item, dict) and "page_content" in item:
            # Dictionary with Document structure
            documents.append(Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {})
            ))
        elif isinstance(item, str):
            # Simple string content
            documents.append(Document(
                page_content=item,
                metadata={}
            ))
        else:
            # Try to convert to string
            try:
                content = str(item)
                documents.append(Document(
                    page_content=content,
                    metadata={}
                ))
            except:
                pass  # Skip items that can't be converted
    
    return documents

def load_preprocessed_data(file_path):
    """
    Load preprocessed data from joblib file with flexible format handling
    
    Returns:
    - documents: list of Document objects
    - metadata: dictionary with additional info
    """
    try:
        # Get file size for logging
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        
        # Load the data
        data = joblib.load(file_path)
        
        documents = []
        summaries = []
        metadata = {"file_size_mb": file_size}
        
        # Handle different possible formats of the data
        
        # Case 1: Dictionary with documents and possibly summaries
        if isinstance(data, dict):
            
            # Try to get documents from 'documents' key
            if "documents" in data:
                raw_docs = data["documents"]
                
                # Convert to Document objects
                for i, doc in enumerate(raw_docs):
                    # If already a Document
                    if isinstance(doc, Document):
                        documents.append(doc)
                    # If has page_content attribute (Document-like)
                    elif hasattr(doc, "page_content"):
                        documents.append(Document(
                            page_content=doc.page_content,
                            metadata=getattr(doc, "metadata", {})
                        ))
                    # If it's a string
                    elif isinstance(doc, str):
                        documents.append(Document(
                            page_content=doc,
                            metadata={"index": i}
                        ))
                    # Try to convert to string
                    else:
                        try:
                            documents.append(Document(
                                page_content=str(doc),
                                metadata={"index": i}
                            ))
                        except:
                            # Skip if can't convert
                            pass
            
            # Try to get summaries from 'summaries' key
            if "summaries" in data:
                summaries = data["summaries"]
            
            # Extract any other metadata
            for key in data:
                if key not in ["documents", "summaries"]:
                    metadata[key] = data[key]
        
        # Case 2: List of documents
        elif isinstance(data, list):
            
            # Try to convert each item to a Document
            for i, item in enumerate(data):
                if isinstance(item, Document):
                    documents.append(item)
                elif hasattr(item, "page_content"):
                    documents.append(Document(
                        page_content=item.page_content,
                        metadata=getattr(item, "metadata", {})
                    ))
                elif isinstance(item, str):
                    documents.append(Document(
                        page_content=item,
                        metadata={"index": i}
                    ))
                else:
                    try:
                        documents.append(Document(
                            page_content=str(item),
                            metadata={"index": i}
                        ))
                    except:
                        pass
        
        # Case 3: Object with documents attribute
        elif hasattr(data, "documents"):
            
            raw_docs = data.documents
            # Convert to Document objects
            for i, doc in enumerate(raw_docs):
                if isinstance(doc, Document):
                    documents.append(doc)
                elif hasattr(doc, "page_content"):
                    documents.append(Document(
                        page_content=doc.page_content,
                        metadata=getattr(doc, "metadata", {})
                    ))
                elif isinstance(doc, str):
                    documents.append(Document(
                        page_content=doc,
                        metadata={"index": i}
                    ))
                else:
                    try:
                        documents.append(Document(
                            page_content=str(doc),
                            metadata={"index": i}
                        ))
                    except:
                        pass
            
            # Try to get summaries
            if hasattr(data, "summaries"):
                summaries = data.summaries
        
        # Add summaries to metadata
        metadata["summaries"] = summaries
        metadata["doc_count"] = len(documents)
        metadata["sum_count"] = len(summaries)
        
        return documents, metadata
        
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        return [], {"error": error_msg, "traceback": traceback.format_exc()}

def ensure_chroma_directory():
    """Ensure the ChromaDB directory exists"""
    chroma_dir = "./chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir

def display_conversation(memory):
    """Display the conversation history using Streamlit's chat components"""
    # Get messages
    messages = memory.chat_memory.messages
    if not messages:  # If no messages, return early
        return
    
    # Define custom avatars with correct paths
    user_avatar = "images/user.png"
    assistant_avatar = "images/ideo.png"
    
    # Create a container for the messages
    with st.container():
        # If chat history is disabled, only show the latest exchange
        if not st.session_state.show_chat_history:
            if len(messages) >= 2:  # Make sure we have at least one exchange
                with st.chat_message("user", avatar=user_avatar):
                    st.markdown(messages[-2].content)
                with st.chat_message("assistant", avatar=assistant_avatar):
                    st.markdown(messages[-1].content)
            return
        
        # Otherwise, show full history with pagination
        msg_count = len(messages)
        max_displayed = st.session_state.get("max_displayed_messages", 100)
        
        # Get the last N messages and reverse them
        start_idx = max(0, msg_count - max_displayed)
        messages_to_display = messages[start_idx:]
        
        # Display messages in reverse order (newest first)
        for message in reversed(messages_to_display):
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            avatar = user_avatar if role == "user" else assistant_avatar
            with st.chat_message(role, avatar=avatar):
                st.markdown(message.content)
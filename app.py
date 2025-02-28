"""
Multimodal PDF Chat - Main Application
"""
import sys
import os
import time
import shutil

# SQLite fix for ChromaDB
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

import uuid
import warnings
import traceback
import logging
import joblib
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from utils.rag_chain import (
    parse_docs, build_prompt, get_conversational_rag_chain,
    create_multimodal_retriever, create_fallback_retriever
)

# Add the project directory to the Python path to fix import issues
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.helpers import save_uploaded_file, ensure_document_format
from utils.helpers import load_preprocessed_data, ensure_chroma_directory, display_conversation
from utils.html_templates import inject_css, bot_template, user_template
from utils.document_processor import (
    UNSTRUCTURED_AVAILABLE, process_pdfs_with_unstructured, 
    get_pdf_text_fallback, summarize_elements, process_fallback_documents,
    generate_missing_summaries
)

# Suppress warnings (this won't affect the PyTorch warnings but helps with other libraries)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Workaround for PyTorch error in Streamlit file watcher
# This prevents Streamlit from watching PyTorch modules which can cause errors
import streamlit.watcher.path_watcher
original_watch_dir = streamlit.watcher.path_watcher.watch_dir

def patched_watch_dir(path, *args, **kwargs):
    if "torch" in path or "_torch" in path or "site-packages" in path:
        # Skip watching PyTorch-related directories
        return None
    return original_watch_dir(path, *args, **kwargs)

streamlit.watcher.path_watcher.watch_dir = patched_watch_dir

# Load environment variables
load_dotenv()  # Keep this for local development

# Get OpenAI API key from Streamlit secrets or environment variable
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml or .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set for OpenAI client

# Constants for preloaded collection
PREPROCESSED_DATA_PATH = "./preprocessed_data"
PREPROCESSED_COLLECTION_FILE = os.path.join(PREPROCESSED_DATA_PATH, "primary_collection.joblib")

# Check if preprocessed_data directory exists
if not os.path.exists(PREPROCESSED_DATA_PATH):
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
    print(f"Created preprocessed data directory at {PREPROCESSED_DATA_PATH}")

# Check if collection file exists
if not os.path.exists(PREPROCESSED_COLLECTION_FILE):
    print(f"Warning: Preprocessed collection file not found at {PREPROCESSED_COLLECTION_FILE}")
    print("You'll need to run the preprocessing script first or switch to upload mode.")

# Cache expensive operations
@st.cache_resource
def get_openai_model(model_name):
    """Cache the OpenAI model to avoid reloading it"""
    return ChatOpenAI(model=model_name, temperature=0.2)

@st.cache_resource
def get_embeddings():
    """Cache the embeddings model to avoid reloading it"""
    return OpenAIEmbeddings()

def parse_docs(docs):
    """Split base64-encoded images and texts (like the example)"""
    b64_images = []
    texts = []
    for doc in docs:
        # Check if this is a base64 image
        if isinstance(doc, str):
            try:
                import base64
                base64.b64decode(doc)
                b64_images.append(doc)
            except Exception:
                pass  # Not a valid base64 string
        else:
            texts.append(doc)
    
    return {"images": b64_images, "texts": texts}

def build_prompt(kwargs):
    """Build a prompt that includes text context and images (like the example)"""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    chat_history = kwargs.get("chat_history", [])

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            if hasattr(text_element, 'text'):
                context_text += text_element.text + "\n\n"
            elif hasattr(text_element, 'page_content'):
                context_text += text_element.page_content + "\n\n"
            else:
                context_text += str(text_element) + "\n\n"

    # Format chat history
    chat_history_text = ""
    if chat_history:
        for msg in chat_history:
            role = "Human" if msg.type == "human" else "Assistant"
            chat_history_text += f"{role}: {msg.content}\n"

    # Construct prompt with context (like the example)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.
    
    Context: {context_text}
    
    {chat_history_text}
    
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    # Add images to the prompt if available (like the example)
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def get_conversational_rag_chain(retriever, model):
    """Create the RAG chain for conversation"""
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.chat_memory.messages
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
    
    return chain, memory

def handle_userinput(user_question, rag_chain, memory):
    """Process the user question and update the chat"""
    if not user_question:
        return
    
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke(user_question)
            
            # Update memory
            memory.chat_memory.add_user_message(user_question)
            memory.chat_memory.add_ai_message(response)
            
            # Rerun to show updated conversation
            st.rerun()
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)

def load_preloaded_collection():
    """
    Load the preprocessed collection from the joblib file and create a retriever
    """
    # Check if the joblib file exists
    if not os.path.exists(PREPROCESSED_COLLECTION_FILE):
        st.error(f"Preprocessed collection file not found at {PREPROCESSED_COLLECTION_FILE}")
        return None
    
    try:
        with st.status("Loading preprocessed collection...") as status:
            # Use our flexible data loading function
            status.update(label="Loading data from file...")
            documents, metadata = load_preprocessed_data(PREPROCESSED_COLLECTION_FILE)
            
            # Check if we have documents but no summaries
            summaries = metadata.get("summaries", [])
            
            if documents and (not summaries or len(summaries) == 0):
                status.update(label="No summaries found. Generating summaries...")
                model = get_openai_model("gpt-4o-mini")
                summaries = generate_missing_summaries(documents, model)
            
            # Handle mismatch between documents and summaries
            elif documents and summaries and len(documents) != len(summaries):
                if len(documents) > len(summaries):
                    # Generate missing summaries
                    status.update(label=f"Generating {len(documents) - len(summaries)} missing summaries...")
                    model = get_openai_model("gpt-4o-mini")
                    missing_summaries = generate_missing_summaries(documents[len(summaries):], model)
                    summaries.extend(missing_summaries)
                else:
                    # Trim excess summaries
                    status.update(label="Trimming excess summaries...")
                    summaries = summaries[:len(documents)]
            
            # Initialize embeddings
            status.update(label="Initializing embeddings...")
            embeddings = get_embeddings()
            
            # Create a unique ChromaDB collection for this session
            status.update(label="Setting up vector store...")
            chroma_dir = ensure_chroma_directory()
            collection_name = f"preloaded_rag_{uuid.uuid4().hex[:8]}"
            
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=chroma_dir
            )
            
            store = InMemoryStore()
            id_key = "doc_id"
            
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=store,
                id_key=id_key,
            )
            
            # Add documents and summaries to the retriever
            if documents and summaries:
                doc_ids = [str(uuid.uuid4()) for _ in documents]
                summary_docs = [
                    Document(page_content=summary, metadata={id_key: doc_ids[i]})
                    for i, summary in enumerate(summaries)
                ]
                
                # Add documents to vectorstore
                status.update(label=f"Adding {len(summary_docs)} documents to vectorstore...")
                retriever.vectorstore.add_documents(summary_docs)
                
                # Add original documents to docstore
                status.update(label="Adding documents to docstore...")
                retriever.docstore.mset(list(zip(doc_ids, documents)))
                
                # Create the conversation chain
                status.update(label="Setting up conversation chain...")
                model = get_openai_model("gpt-4o-mini")
                chain, memory = get_conversational_rag_chain(retriever, model)
                
                status.update(label=f"Successfully loaded {len(documents)} documents!", state="complete")
                return chain, memory
            else:
                status.update(label="Error: No documents or summaries found in preprocessed data", state="error")
                return None
            
    except Exception as e:
        error_msg = f"Error loading preprocessed collection: {str(e)}"
        st.error(error_msg)
        return None

def initialize_session_state():
    """Initialize all session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.mode = "preloaded"
        st.session_state.rag_chain = None
        st.session_state.memory = None
        st.session_state.processing_complete = False
        st.session_state.max_displayed_messages = 10
        st.session_state.mode_changed = False
        st.session_state.show_chat_history = True
        # Add variables to store preloaded mode state
        st.session_state.preloaded_chain = None
        st.session_state.preloaded_memory = None
        st.session_state.preloaded_complete = False
        # Add variables to store upload mode state
        st.session_state.upload_chain = None
        st.session_state.upload_memory = None
        st.session_state.upload_complete = False
        # Add variables to store uploaded files state
        st.session_state.uploaded_files = None
        st.session_state.files_processed = False
        # Add variable to track if we should clear files
        st.session_state.clear_files = False
        st.session_state.show_success = False

def on_file_upload():
    """Callback for file uploader changes"""
    if "pdf_uploader" in st.session_state:
        pdf_docs = st.session_state.pdf_uploader
        if pdf_docs is not None:
            st.session_state.uploaded_files = pdf_docs
            if not pdf_docs:  # Empty list means files were removed
                st.session_state.files_processed = False

def handle_mode_change():
    """Handle mode change with proper state management"""
    new_mode = "preloaded" if st.session_state.mode_radio == "Use Trend Report Collection" else "upload"
    
    if new_mode != st.session_state.mode:
        if new_mode == "upload":
            # Restore upload mode state
            st.session_state.rag_chain = st.session_state.upload_chain
            st.session_state.memory = st.session_state.upload_memory
            st.session_state.processing_complete = st.session_state.upload_complete
        else:
            # Save upload mode state before switching
            st.session_state.upload_chain = st.session_state.rag_chain
            st.session_state.upload_memory = st.session_state.memory
            st.session_state.upload_complete = st.session_state.processing_complete
            # Restore preloaded mode state
            st.session_state.rag_chain = st.session_state.preloaded_chain
            st.session_state.memory = st.session_state.preloaded_memory
            st.session_state.processing_complete = st.session_state.preloaded_complete
        
        st.session_state.mode = new_mode
        st.session_state.mode_changed = True

def inject_css():
    """Inject custom CSS styles"""
    return """
        <style>
            /* Style for processed documents state */
            .processed-docs .stTextInput input {
                border: 2px solid #28a745 !important;
                background-color: #f8fff9 !important;
            }
            
            /* Make sidebar text smaller */
            .css-163ttbj {  /* sidebar */
                font-size: 0.8rem;
            }
            
            /* Adjust sidebar header sizes */
            .css-zt5igj {  /* sidebar headers */
                font-size: 1rem;
            }
            
            /* Reduce spacing in sidebar */
            .css-1544g2n {  /* sidebar padding */
                padding: 1rem 0.5rem;
            }
            
            /* Adjust success message size in sidebar */
            .sidebar .stSuccess {
                font-size: 0.8rem;
                padding: 0.5rem;
            }
            
            /* Basic chat message styling */
            [data-testid="stChatMessage"] {
                padding: 2rem 0;
            }
            
            /* Style user messages */
            [data-testid="stChatMessage"][data-chat-message-user-name="user"] {
                background-color: white;
            }
            
            /* Style assistant messages */
            [data-testid="stChatMessage"][data-chat-message-user-name="assistant"] {
                background-color: #f7f7f8;
                position: relative;
            }
            
            /* Add padding to chat message container */
            .stChatMessage {
                padding-left: 2rem !important;
                padding-right: 2rem !important;
                display: flex !important;
                gap: 1.5rem !important;  /* Add gap between avatar and content */
            }
            
            /* Adjust the avatar container */
            .stChatMessage > div:first-child {
                position: relative !important;
                left: 0 !important;
            }
            
            /* Adjust the content container spacing */
            .stChatMessageContent {
                flex: 1 !important;
                padding-left: 1rem !important;
                padding-right: 2rem !important;
            }
        </style>
    """

def cleanup_temp_files(pdf_paths):
    """Clean up temporary PDF files after processing"""
    for path in pdf_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            st.warning(f"Could not remove temporary file {path}: {e}")

def cleanup_old_collections():
    """Clean up ChromaDB collections older than 24 hours"""
    chroma_dir = ensure_chroma_directory()
    current_time = time.time()
    
    try:
        for collection_dir in os.listdir(chroma_dir):
            collection_path = os.path.join(chroma_dir, collection_dir)
            if os.path.isdir(collection_path):
                # Check if directory is older than 24 hours
                if current_time - os.path.getctime(collection_path) > 86400:  # 24 hours in seconds
                    shutil.rmtree(collection_path)
    except Exception as e:
        st.warning(f"Could not clean up old collections: {e}")

def main():
    # Add cleanup of old collections at start
    cleanup_old_collections()
    
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="Multimodal PDF Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Check if we need to rerun due to mode change
    if st.session_state.mode_changed:
        st.session_state.mode_changed = False
        st.rerun()
    
    # Add CSS for styling
    st.markdown(inject_css(), unsafe_allow_html=True)
    
    # Display the header
    st.header("📚 Multimodal PDF LLM")

    # Mode selection in sidebar - do this before main content
    with st.sidebar:
        st.subheader("Mode Selection")
        
        # Add radio with key and on_change callback
        st.radio(
            "Choose mode:",
            ["Use Preloaded Collection", "Upload Your Own PDFs"],
            key="mode_radio",
            index=0 if st.session_state.mode == "preloaded" else 1,
            on_change=handle_mode_change
        )

        # Clear all messages button
        if st.button("Clear All Messages"):
            if "memory" in st.session_state and st.session_state.memory:
                # Only clear the chat messages
                st.session_state.memory.chat_memory.messages = []
                st.rerun()

    # Display appropriate status message based on mode
    if not st.session_state.processing_complete:
        if st.session_state.mode == "preloaded":
            st.info("Please click 'Load Preloaded Collection' in the sidebar to get started.")
        else:
            st.info("Please upload and process your PDFs in the sidebar to get started.")
    else:
        st.success("You can now ask questions about your documents.")

    # Container for chat interface
    chat_container = st.container()
    
    with chat_container:
        # Add class to form based on processing state
        form_class = "processed-docs" if st.session_state.processing_complete else ""
        
        # Chat input form at the top with dynamic class
        with st.form(key='chat_form', clear_on_submit=True):
            # Add the class to a div wrapper
            st.markdown(f'<div class="{form_class}">', unsafe_allow_html=True)
            
            cols = st.columns([8, 1])
            with cols[0]:
                user_question = st.text_input(
                    "Ask a question about your documents:",
                    key="user_question",
                    label_visibility="collapsed"
                )
            
            with cols[1]:
                submit_button = st.form_submit_button(
                    "Send",
                    use_container_width=True
                )
            
            # Close the div wrapper
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle form submission
            if submit_button and user_question:
                if st.session_state.rag_chain and st.session_state.memory:
                    handle_userinput(user_question, st.session_state.rag_chain, st.session_state.memory)
                elif not st.session_state.processing_complete:
                    if st.session_state.mode == "preloaded":
                        st.warning("Please load the preloaded collection first!")
                    else:
                        st.warning("Please upload and process your PDFs first!")
        
        # Display conversation history below the input form
        if st.session_state.memory and st.session_state.processing_complete:
            display_conversation(st.session_state.memory)

    with st.sidebar:
        # Message display settings
        st.subheader("Display Settings")
        
        # Add chat history toggle with immediate rerun
        show_history = st.toggle("Chat History", value=st.session_state.show_chat_history)
        if show_history != st.session_state.show_chat_history:
            st.session_state.show_chat_history = show_history
            st.rerun()
        
        # Only show max messages slider if chat history is enabled
        if show_history:
            max_messages = st.slider("Max messages to display", 2, 100, 10)
            if max_messages != st.session_state.max_displayed_messages:
                st.session_state.max_displayed_messages = max_messages
                st.rerun()
        
        # Only show upload section if in upload mode
        if st.session_state.mode == "upload":
            st.subheader("Document Processing")
            
            # Create a container for file uploader
            uploader_container = st.container()
            
            with uploader_container:
                # File uploader with callback
                pdf_docs = st.file_uploader(
                    "Upload your PDFs here and click on **Process Documents** \n \n Processing is done in a single batch.", 
                    accept_multiple_files=True,
                    type=["pdf"],
                    key="pdf_uploader",
                    on_change=on_file_upload
                )
                
                # Show files if they exist (using session state)
                if st.session_state.uploaded_files:
                    st.write("Currently loaded files:")
                    for file in st.session_state.uploaded_files:
                        st.write(f"- {file.name}")
            
            process_button = st.button("Process Documents")
            
            # Show success message after process button
            if st.session_state.show_success and st.session_state.processing_complete:
                st.success(f"Successfully processed your documents!")
            
            if process_button and pdf_docs:
                try:
                    # Show loading message without spinner
                    loading_message = st.info("Please wait. This may take a few minutes 🙂")
                    
                    # Save uploaded files
                    pdf_paths = [save_uploaded_file(pdf) for pdf in pdf_docs]
                    st.write("📄 Processing files:", ", ".join(path.split('/')[-1] for path in pdf_paths))
                    
                    # Configure models - use cached versions
                    model = get_openai_model("gpt-4o-mini")
                    embeddings = get_embeddings()
                    
                    # Process documents without duplicate spinner
                    unstructured_status = "✅" if UNSTRUCTURED_AVAILABLE else "❌"
                    st.write(f"Unstructured API: {unstructured_status}")
                    
                    if UNSTRUCTURED_AVAILABLE:
                        try:
                            st.write("🔄 Using Unstructured for processing...")
                            texts, tables, images = process_pdfs_with_unstructured(pdf_paths)
                            
                            if texts or tables or images:
                                st.write(f"Extracted: {len(texts)} texts, {len(tables)} tables, {len(images)} images")  # Debug info
                                # Summarize elements
                                text_summaries, table_summaries, image_summaries = summarize_elements(
                                    texts, tables, images, model
                                )
                                
                                # Create retriever and chain
                                retriever = create_multimodal_retriever(
                                    texts, tables, images,
                                    text_summaries, table_summaries, image_summaries,
                                    embeddings
                                )
                            else:
                                st.warning("No content extracted. Falling back to PyPDF2.")
                                documents = get_pdf_text_fallback(pdf_paths)
                                text_chunks, summaries = process_fallback_documents(documents, model)
                                retriever = create_fallback_retriever(text_chunks, summaries, embeddings)
                        except Exception as e:
                            st.warning(f"Error with Unstructured processing: {str(e)}. Falling back to PyPDF2.")
                            documents = get_pdf_text_fallback(pdf_paths)
                            text_chunks, summaries = process_fallback_documents(documents, model)
                            retriever = create_fallback_retriever(text_chunks, summaries, embeddings)
                    else:
                        st.info("Using PyPDF2 for text extraction.")
                        documents = get_pdf_text_fallback(pdf_paths)
                        text_chunks, summaries = process_fallback_documents(documents, model)
                        retriever = create_fallback_retriever(text_chunks, summaries, embeddings)
                    
                    # Create RAG chain
                    rag_chain, memory = get_conversational_rag_chain(retriever, model)
                    
                    # Save to both current and upload mode states
                    st.session_state.rag_chain = rag_chain
                    st.session_state.memory = memory
                    st.session_state.processing_complete = True
                    st.session_state.upload_chain = rag_chain
                    st.session_state.upload_memory = memory
                    st.session_state.upload_complete = True
                    
                    # Cleanup
                    cleanup_temp_files(pdf_paths)
                    
                    # Clear loading message before rerun
                    loading_message.empty()
                    st.session_state.files_processed = True
                    st.session_state.show_success = True
                    st.rerun()
                    
                except Exception as e:
                    cleanup_temp_files(pdf_paths)  # Clean up even if processing fails
                    error_msg = f"Error processing documents: {str(e)}"
                    st.error(error_msg)
                    
        elif st.session_state.mode == "preloaded":
            st.subheader("Preloaded Collection")
            
            # Information about the collection
            preprocessed_exists = os.path.exists(PREPROCESSED_COLLECTION_FILE)
            if not preprocessed_exists:
                st.error(f"Collection file not found at: {PREPROCESSED_COLLECTION_FILE}")
                st.write("Please ensure your preprocessed data is in the correct location:")
                st.code(f"./preprocessed_data/primary_collection.joblib")
            
            # Button to load the preloaded collection
            load_button = st.button("Load Preloaded Collection")
            
            # Show success message after load button
            if st.session_state.processing_complete:
                st.success(f"Successfully loaded preloaded collection!")
            
            if load_button:
                # Show loading message without spinner
                loading_message = st.info("Please wait. This may take a few minutes 🙂")
                
                result = load_preloaded_collection()
                if result:
                    rag_chain, memory = result
                    # Update both current and preloaded states
                    st.session_state.rag_chain = rag_chain
                    st.session_state.memory = memory
                    st.session_state.processing_complete = True
                    st.session_state.preloaded_chain = rag_chain
                    st.session_state.preloaded_memory = memory
                    st.session_state.preloaded_complete = True
                    
                    # Clear loading message before rerun
                    loading_message.empty()
                    st.rerun()

        # Add disclaimer at the bottom of sidebar
        st.markdown("<br>" * 3, unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size: 0.7rem; color: #666; margin-top: 3rem;'>
        <p><em>This application uses ChatGPT and can make mistakes. <br>OpenAI doesn't use IDEO workspace data to train its models.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
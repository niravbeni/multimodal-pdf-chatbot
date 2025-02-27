import streamlit as st
import os
import uuid
import base64
import warnings
import traceback
import logging
import joblib
from io import BytesIO
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever

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

# Import Unstructured
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except Exception as e:
    UNSTRUCTURED_AVAILABLE = False
    print(f"Unstructured not available: {e}")

# Load environment variables
load_dotenv()

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

# Load and encode images at module level
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

try:
    bot_img = get_base64_encoded_image("images/ideo.png")
    user_img = get_base64_encoded_image("images/user.png")
except Exception as e:
    bot_img = ""
    user_img = ""
    print(f"Error loading chat icons: {str(e)}")

# HTML Templates
css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    position: relative;
}
.chat-message.user {
    background-color: #0095ff20  /* Streamlit info blue with transparency */
}
.chat-message.bot {
    background-color: #E6E6FA  /* light purple */
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #000;  /* changed text color to black for better contrast */
}
.debug-info {
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
    overflow-x: auto;
}
</style>
"""

bot_template = f"""
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{bot_img}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
"""

user_template = f"""
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_img}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
"""

# Add debug mode
DEBUG_MODE = False

def debug(message):
    """Print debug messages if debug mode is on"""
    if DEBUG_MODE:
        st.markdown(f'<div class="debug-info">{message}</div>', unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location and return the path"""
    temp_dir = "./temp_pdf_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def process_pdfs_with_unstructured(pdf_paths):
    """Process PDFs using Unstructured following the example approach"""
    all_texts = []
    all_tables = []
    all_images = []
    
    with st.status("Processing PDFs...") as status:
        for i, pdf_path in enumerate(pdf_paths):
            status.update(label=f"Processing PDF {i+1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
            
            try:
                # Extract content using the same parameters as the example
                chunks = partition_pdf(
                    filename=pdf_path,
                    infer_table_structure=True,
                    strategy="hi_res",
                    extract_image_block_types=["Image"],
                    extract_image_block_to_payload=True,
                    chunking_strategy="by_title",
                    max_characters=10000,
                    combine_text_under_n_chars=2000,
                    new_after_n_chars=6000,
                )
                
                # Separate chunks by type exactly like the example
                pdf_tables = []
                pdf_texts = []
                
                for chunk in chunks:
                    if "Table" in str(type(chunk)):
                        pdf_tables.append(chunk)
                    if "CompositeElement" in str(type(chunk)):
                        pdf_texts.append(chunk)
                
                # Get images from CompositeElements like the example
                pdf_images = []
                for chunk in chunks:
                    if "CompositeElement" in str(type(chunk)):
                        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                            chunk_els = chunk.metadata.orig_elements
                            for el in chunk_els:
                                if "Image" in str(type(el)):
                                    if hasattr(el, 'metadata') and hasattr(el.metadata, 'image_base64'):
                                        pdf_images.append(el.metadata.image_base64)
                
                # Add source information for display purposes only (not modifying original chunks)
                debug(f"Extracted from {os.path.basename(pdf_path)}: {len(pdf_texts)} texts, {len(pdf_tables)} tables, {len(pdf_images)} images")
                
                # Add to the overall collections
                all_texts.extend(pdf_texts)
                all_tables.extend(pdf_tables)
                all_images.extend(pdf_images)
                
            except Exception as e:
                error_msg = f"Error processing {os.path.basename(pdf_path)}: {str(e)}"
                debug(f"{error_msg}\n{traceback.format_exc()}")
                st.error(error_msg)
        
        status.update(label=f"PDF processing complete! Extracted {len(all_texts)} text chunks, {len(all_tables)} tables, {len(all_images)} images.", state="complete")
    
    return all_texts, all_tables, all_images

def get_pdf_text_fallback(pdf_paths):
    """Fallback method using PyPDF2"""
    from PyPDF2 import PdfReader
    
    all_docs = []
    with st.status("Processing PDFs using fallback method...") as status:
        for i, pdf_path in enumerate(pdf_paths):
            status.update(label=f"Processing PDF {i+1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
            
            try:
                pdf_reader = PdfReader(pdf_path)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1
                            }
                        )
                        all_docs.append(doc)
            except Exception as e:
                st.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
    
    return all_docs

def summarize_elements(texts, tables, images, model):
    """Generate summaries for elements just like the example code"""
    text_summaries = []
    table_summaries = []
    image_summaries = []
    
    with st.status("Generating summaries...") as status:
        # Text and table summary prompt (same as example)
        prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        # Summarize texts
        if texts:
            status.update(label=f"Summarizing {len(texts)} text chunks...")
            for text in texts:
                try:
                    summary = summarize_chain.invoke(text)
                    text_summaries.append(summary)
                except Exception as e:
                    debug(f"Error summarizing text: {str(e)}")
        
        # Summarize tables
        if tables:
            status.update(label=f"Summarizing {len(tables)} tables...")
            # Get HTML representation of tables like the example
            tables_html = [table.metadata.text_as_html for table in tables if hasattr(table, 'metadata') and hasattr(table.metadata, 'text_as_html')]
            
            for table_html in tables_html:
                try:
                    summary = summarize_chain.invoke(table_html)
                    table_summaries.append(summary)
                except Exception as e:
                    debug(f"Error summarizing table: {str(e)}")
        
        # Summarize images
        if images:
            status.update(label=f"Analyzing {len(images)} images...")
            # Image summary prompt (similar to example)
            img_prompt_template = """Describe the image in detail. For context, 
                            the image is part of a PDF document. Be specific about 
                            any graphs, tables, or visual elements."""
            
            for i, image in enumerate(images):
                try:
                    messages = [
                        (
                            "user",
                            [
                                {"type": "text", "text": img_prompt_template},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                                },
                            ],
                        )
                    ]
                    
                    img_prompt = ChatPromptTemplate.from_messages(messages)
                    img_chain = img_prompt | model | StrOutputParser()
                    summary = img_chain.invoke("")
                    image_summaries.append(summary)
                except Exception as e:
                    debug(f"Error summarizing image {i}: {str(e)}")
        
        debug(f"Generated summaries: {len(text_summaries)} texts, {len(table_summaries)} tables, {len(image_summaries)} images")
        status.update(label="All summaries generated!", state="complete")
    
    return text_summaries, table_summaries, image_summaries

def process_fallback_documents(documents, model):
    """Process documents from the fallback method"""
    text_chunks = []
    summaries = []
    
    with st.status("Processing document content...") as status:
        # Text splitter for long documents
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Split long documents into chunks
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            text_chunks.extend(chunks)
        
        # Summarize chunks
        prompt_text = """
        You are an assistant tasked with summarizing text.
        Give a concise summary of the text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Text chunk: {element}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        for chunk in text_chunks:
            try:
                summary = summarize_chain.invoke(chunk.page_content)
                summaries.append(summary)
            except Exception as e:
                debug(f"Error summarizing chunk: {str(e)}")
    
    return text_chunks, summaries

def ensure_chroma_directory():
    """Ensure the ChromaDB directory exists"""
    chroma_dir = "./chroma_db"
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir

def create_multimodal_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries, embeddings):
    """Create a MultiVectorRetriever like the example code but with persistent storage"""
    # Create a directory for ChromaDB
    chroma_dir = ensure_chroma_directory()
    
    # Clear any existing collections by using a unique collection name
    collection_name = f"multimodal_rag_{uuid.uuid4().hex[:8]}"
    
    # Create vectorstore and docstore with persistent path
    vectorstore = Chroma(
        collection_name=collection_name, 
        embedding_function=embeddings,
        persist_directory=chroma_dir
    )
    
    store = InMemoryStore()
    id_key = "doc_id"
    
    # Create retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    # Add texts (exactly like the example)
    if texts and text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) 
            for i, summary in enumerate(text_summaries)
        ]
        
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))
    
    # Add tables (exactly like the example)
    if tables and table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]}) 
            for i, summary in enumerate(table_summaries)
        ]
        
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))
    
    # Add images (exactly like the example)
    if images and image_summaries:
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) 
            for i, summary in enumerate(image_summaries)
        ]
        
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, images)))
    
    return retriever

def create_fallback_retriever(text_chunks, summaries, embeddings):
    """Create retriever from fallback processed documents with persistent storage"""
    # Create a directory for ChromaDB
    chroma_dir = ensure_chroma_directory()
    
    # Use unique collection name to avoid conflicts
    collection_name = f"fallback_rag_{uuid.uuid4().hex[:8]}"
    
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
    
    doc_ids = [str(uuid.uuid4()) for _ in range(len(summaries))]
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) 
        for i, summary in enumerate(summaries)
    ]
    
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, text_chunks[:len(summaries)])))
    
    return retriever

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

def load_preprocessed_data(file_path, debug=False):
    """
    Load preprocessed data from joblib file with flexible format handling
    
    Returns:
    - documents: list of Document objects
    - metadata: dictionary with additional info
    """
    try:
        # Get file size for logging
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if debug:
            st.write(f"Loading file: {file_path} ({file_size:.2f} MB)")
        
        # Load the data
        data = joblib.load(file_path)
        
        documents = []
        summaries = []
        metadata = {"file_size_mb": file_size}
        
        # Handle different possible formats of the data
        
        # Case 1: Dictionary with documents and possibly summaries
        if isinstance(data, dict):
            if debug:
                st.write(f"Loaded dictionary with keys: {list(data.keys())}")
            
            # Try to get documents from 'documents' key
            if "documents" in data:
                raw_docs = data["documents"]
                if debug:
                    st.write(f"Found {len(raw_docs)} documents in 'documents' key")
                
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
                if debug:
                    st.write(f"Found {len(summaries)} summaries in 'summaries' key")
            
            # Extract any other metadata
            for key in data:
                if key not in ["documents", "summaries"]:
                    metadata[key] = data[key]
        
        # Case 2: List of documents
        elif isinstance(data, list):
            if debug:
                st.write(f"Loaded list with {len(data)} items")
            
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
            if debug:
                st.write("Loaded object with 'documents' attribute")
            
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
                if debug:
                    st.write(f"Found {len(summaries)} summaries in 'summaries' attribute")
        
        # Add summaries to metadata
        metadata["summaries"] = summaries
        metadata["doc_count"] = len(documents)
        metadata["sum_count"] = len(summaries)
        
        if debug:
            st.write(f"Successfully loaded {len(documents)} documents and {len(summaries)} summaries")
        
        return documents, metadata
        
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        if debug:
            st.error(error_msg)
            st.write(traceback.format_exc())
        return [], {"error": error_msg, "traceback": traceback.format_exc()}

def generate_missing_summaries(documents, model):
    """
    Generate summaries for documents that don't have them
    """
    summaries = []
    with st.status("Generating missing summaries...") as status:
        # Create a summary prompt
        prompt_text = """
        You are an assistant tasked with summarizing text.
        Give a concise summary of the text.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Text: {element}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_text)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        
        total = len(documents)
        for i, doc in enumerate(documents):
            status.update(label=f"Generating summary {i+1}/{total}...")
            try:
                # Get the content from the document
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif hasattr(doc, 'text'):
                    content = doc.text
                else:
                    content = str(doc)
                
                # Limit content length if needed
                content = content[:5000]  # Limit to 5000 chars to avoid token limits
                
                summary = summarize_chain.invoke(content)
                summaries.append(summary)
            except Exception as e:
                debug(f"Error generating summary: {str(e)}")
                summaries.append("Summary not available.")
        
        status.update(label=f"Generated {len(summaries)} summaries", state="complete")
    return summaries

def load_preloaded_collection(model_name="gpt-4o-mini"):
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
            documents, metadata = load_preprocessed_data(PREPROCESSED_COLLECTION_FILE, debug=DEBUG_MODE)
            # Load the preprocessed data
            preprocessed_data = joblib.load(PREPROCESSED_COLLECTION_FILE)
            
            # Debug the structure of the preprocessed data
            if DEBUG_MODE:
                st.write("Preprocessed data keys:", list(preprocessed_data.keys()) if isinstance(preprocessed_data, dict) else "Not a dictionary")
                st.write("Preprocessed data type:", type(preprocessed_data))
            
            # Try to extract documents and summaries with flexible handling
            if isinstance(preprocessed_data, dict):
                # Standard dictionary format
                documents = preprocessed_data.get("documents", [])
                summaries = preprocessed_data.get("summaries", [])
            elif hasattr(preprocessed_data, "documents") and hasattr(preprocessed_data, "summaries"):
                # Object with attributes
                documents = preprocessed_data.documents
                summaries = preprocessed_data.summaries
            elif isinstance(preprocessed_data, (list, tuple)) and len(preprocessed_data) >= 2:
                # Tuple/list format with documents, summaries
                documents = preprocessed_data[0]
                summaries = preprocessed_data[1]
            else:
                # Try to use the data directly if it looks like a list of documents
                if isinstance(preprocessed_data, list):
                    documents = preprocessed_data
                    summaries = []
                else:
                    documents = []
                    summaries = []
                    
            # Convert to lists if they're not already
            if not isinstance(documents, list):
                documents = [documents]
            if not isinstance(summaries, list):
                summaries = [summaries]
                
            # Debug document and summary counts
            if DEBUG_MODE:
                st.write(f"Documents type: {type(documents)}, count: {len(documents)}")
                st.write(f"Summaries type: {type(summaries)}, count: {len(summaries)}")
                if len(documents) > 0:
                    st.write(f"First document type: {type(documents[0])}")
            
            # Ensure documents are in the correct format
            documents = ensure_document_format(documents)
            
            if DEBUG_MODE:
                st.write(f"After formatting: {len(documents)} documents")
            
            # Check if we have documents but no summaries
            if documents and (not summaries or len(summaries) == 0):
                status.update(label="No summaries found. Generating summaries...")
                model = ChatOpenAI(model=model_name, temperature=0.2)
                summaries = generate_missing_summaries(documents, model)
            
            # Handle mismatch between documents and summaries
            elif documents and summaries and len(documents) != len(summaries):
                debug(f"Warning: Document count ({len(documents)}) doesn't match summary count ({len(summaries)})")
                
                if len(documents) > len(summaries):
                    # Generate missing summaries
                    status.update(label=f"Document count ({len(documents)}) > summary count ({len(summaries)}). Generating {len(documents) - len(summaries)} missing summaries...")
                    model = ChatOpenAI(model=model_name, temperature=0.2)
                    missing_summaries = generate_missing_summaries(documents[len(summaries):], model)
                    summaries.extend(missing_summaries)
                else:
                    # Trim excess summaries
                    status.update(label=f"Summary count ({len(summaries)}) > document count ({len(documents)}). Trimming excess summaries.")
                    summaries = summaries[:len(documents)]
            
            status.update(label=f"Loaded {len(documents)} documents from preprocessed collection")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create a unique ChromaDB collection for this session
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
                status.update(label=f"Adding {len(documents)} documents to docstore...")
                retriever.docstore.mset(list(zip(doc_ids, documents)))
                
                # Create the conversation chain
                status.update(label="Setting up conversation chain...")
                model = ChatOpenAI(model=model_name, temperature=0.2)
                chain, memory = get_conversational_rag_chain(retriever, model)
                
                status.update(label=f"Successfully loaded {len(documents)} documents from preprocessed collection", state="complete")
                return chain, memory
            else:
                status.update(label="Error: No documents or summaries found in preprocessed data", state="error")
                return None
            
    except Exception as e:
        error_msg = f"Error loading preprocessed collection: {str(e)}"
        st.error(error_msg)
        debug(f"{error_msg}\n{traceback.format_exc()}")
        return None

def parse_docs(docs):
    """Split base64-encoded images and texts (like the example)"""
    b64_images = []
    texts = []
    for doc in docs:
        # Check if this is a base64 image
        if isinstance(doc, str):
            try:
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

def display_conversation(memory):
    """Display the conversation history"""
    # Create a container for the chat messages
    chat_container = st.container()
    
    # Calculate number of messages to skip if message limit is reached
    msg_count = len(memory.chat_memory.messages)
    max_displayed = st.session_state.get("max_displayed_messages", 100)
    skip_count = max(0, msg_count - max_displayed)
    
    # Display the conversation
    with chat_container:
        for i, message in enumerate(memory.chat_memory.messages):
            # Skip messages if we have too many
            if i < skip_count and max_displayed < msg_count:
                continue
                
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

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
            
            # Display conversation
            display_conversation(memory)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            debug(f"{error_msg}\n{traceback.format_exc()}")
            st.error(error_msg)

def main():
    # Set up Streamlit page
    st.set_page_config(
        page_title="Multimodal PDF Chat",
        page_icon="📚",
        layout="wide"
    )
    
    # Add CSS for styling
    st.write(css, unsafe_allow_html=True)
    
    st.header("📚 Multimodal PDF LLM")
    
    # Initialize session state
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "max_displayed_messages" not in st.session_state:
        st.session_state.max_displayed_messages = 10
    if "mode" not in st.session_state:
        st.session_state.mode = "preloaded"  # Default to preloaded mode
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Create sidebar for PDF upload and settings
    with st.sidebar:
        st.subheader("Mode Selection")
        
        # Add toggle between modes
        mode = st.radio(
            "Choose mode:",
            ["Use Preloaded Collection", "Upload Your Own PDFs"],
            index=0 if st.session_state.mode == "preloaded" else 1
        )
        
        # Update session state based on selection
        new_mode = "preloaded" if mode == "Use Preloaded Collection" else "upload"
        
        # Reset processing state if mode changes
        if new_mode != st.session_state.mode:
            st.session_state.processing_complete = False
            st.session_state.rag_chain = None
            st.session_state.memory = None
        
        st.session_state.mode = new_mode
        
        # Toggle debug mode
        global DEBUG_MODE
        DEBUG_MODE = st.checkbox("Debug Mode", value=DEBUG_MODE)
        
        # Clear all messages button
        if st.button("Clear All Messages"):
            if "memory" in st.session_state and st.session_state.memory:
                # Clear the memory but keep the conversation chain
                st.session_state.memory.chat_memory.messages = []
                st.rerun()
        
        # Message display settings
        st.subheader("Display Settings")
        max_messages = st.slider("Max messages to display", 2, 100, 10)
        if max_messages != st.session_state.max_displayed_messages:
            st.session_state.max_displayed_messages = max_messages
            st.rerun()
        
        # Model selection
        model_options = ["gpt-4o-mini", "gpt-3.5-turbo"]
        selected_model = st.selectbox("Select model:", model_options)
        
        # API key input as fallback if not in .env
        if not api_key:
            api_key = st.text_input("Enter OpenAI API Key (or set in .env file)", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.success("API key loaded from .env file ✅")
        
        # Only show upload section if in upload mode
        if st.session_state.mode == "upload":
            st.subheader("Document Processing")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", 
                accept_multiple_files=True,
                type=["pdf"]
            )
            
            process_button = st.button("Process Documents")
            
            if process_button and pdf_docs and api_key:
                try:
                    # Save uploaded files
                    pdf_paths = [save_uploaded_file(pdf) for pdf in pdf_docs]
                    
                    # Configure models
                    model = ChatOpenAI(model=selected_model, temperature=0.2)
                    embeddings = OpenAIEmbeddings()
                    
                    # First try with Unstructured (following the example code pattern)
                    if UNSTRUCTURED_AVAILABLE:
                        texts, tables, images = process_pdfs_with_unstructured(pdf_paths)
                        
                        if texts or tables or images:
                            # Summarize elements
                            text_summaries, table_summaries, image_summaries = summarize_elements(
                                texts, tables, images, model
                            )
                            
                            # Create retriever
                            retriever = create_multimodal_retriever(
                                texts, tables, images, 
                                text_summaries, table_summaries, image_summaries, 
                                embeddings
                            )
                        else:
                            st.warning("No content extracted with Unstructured. Falling back to PyPDF2.")
                            # Fallback to PyPDF2
                            documents = get_pdf_text_fallback(pdf_paths)
                            text_chunks, summaries = process_fallback_documents(documents, model)
                            retriever = create_fallback_retriever(text_chunks, summaries, embeddings)
                    else:
                        st.warning("Unstructured not available. Using PyPDF2 for text extraction.")
                        # Fallback to PyPDF2
                        documents = get_pdf_text_fallback(pdf_paths)
                        text_chunks, summaries = process_fallback_documents(documents, model)
                        retriever = create_fallback_retriever(text_chunks, summaries, embeddings)
                    
                    # Create RAG chain
                    rag_chain, memory = get_conversational_rag_chain(retriever, model)
                    
                    # Save to session state
                    st.session_state.rag_chain = rag_chain
                    st.session_state.memory = memory
                    st.session_state.processing_complete = True
                    
                    st.success(f"Successfully processed {len(pdf_docs)} documents!")
                    
                except Exception as e:
                    error_msg = f"Error processing documents: {str(e)}"
                    if DEBUG_MODE:
                        error_msg += f"\n\n{traceback.format_exc()}"
                    st.error(error_msg)
            elif process_button and not api_key:
                st.error("Please provide an OpenAI API key in the .env file or in the sidebar.")
            elif process_button and not pdf_docs:
                st.error("Please upload at least one PDF document.")
        
        # Preloaded mode - add a button to load the collection
        elif st.session_state.mode == "preloaded":
            st.subheader("Preloaded Collection")
            
            # Information about the collection
            preprocessed_exists = os.path.exists(PREPROCESSED_COLLECTION_FILE)
            if preprocessed_exists:
                # Try to get some basic info about the collection
                try:
                    data = joblib.load(PREPROCESSED_COLLECTION_FILE)
                    file_size = os.path.getsize(PREPROCESSED_COLLECTION_FILE) / (1024 * 1024)  # Size in MB
                    
                    st.write(f"📚 Collection file found: {PREPROCESSED_COLLECTION_FILE}")
                    st.write(f"📊 File size: {file_size:.2f} MB")
                    
                    if isinstance(data, dict):
                        doc_count = len(data.get("documents", []))
                        sum_count = len(data.get("summaries", []))
                        emb_count = len(data.get("embeddings", []))
                        
                        st.write(f"📑 Contains:")
                        st.write(f"- {doc_count} documents")
                        st.write(f"- {sum_count} summaries")
                        st.write(f"- {emb_count} embeddings")
                    else:
                        st.write("📑 Collection format: Non-standard (will attempt to extract documents)")
                except Exception as e:
                    st.write("📚 Preloaded collection file exists, but couldn't read contents.")
                    if DEBUG_MODE:
                        st.error(f"Error inspecting collection: {str(e)}")
            else:
                st.error(f"Collection file not found at: {PREPROCESSED_COLLECTION_FILE}")
                st.write("Please ensure your preprocessed data is in the correct location:")
                st.code(f"./preprocessed_data/primary_collection.joblib")
                
            # Button to load the preloaded collection
            load_button = st.button("Load Preloaded Collection")
            
            if load_button and api_key:
                # Load the preloaded collection
                result = load_preloaded_collection(selected_model)
                
                if result:
                    rag_chain, memory = result
                    st.session_state.rag_chain = rag_chain
                    st.session_state.memory = memory
                    st.session_state.processing_complete = True
                    st.success("Successfully loaded preloaded collection!")
            elif load_button and not api_key:
                st.error("Please provide an OpenAI API key in the .env file or in the sidebar.")
    
    # Chat interface
    user_question = st.text_input("Ask a question about your documents:")
    
    # Status message depending on the mode and processing state
    if not st.session_state.processing_complete:
        if st.session_state.mode == "preloaded":
            st.info("Please click 'Load Preloaded Collection' in the sidebar to get started.")
        else:
            st.info("Please upload and process your PDFs in the sidebar to get started.")
    
    # Handle user input if processing is complete
    if user_question and st.session_state.rag_chain and st.session_state.memory:
        handle_userinput(user_question, st.session_state.rag_chain, st.session_state.memory)
    elif user_question and not st.session_state.processing_complete:
        if st.session_state.mode == "preloaded":
            st.warning("Please load the preloaded collection first!")
        else:
            st.warning("Please upload and process your PDFs first!")
    elif st.session_state.memory and st.session_state.processing_complete:
        # Just display the conversation if no new question
        display_conversation(st.session_state.memory)

if __name__ == "__main__":
    main()
"""
RAG Chain Creation Functions
"""
import os
import uuid
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from utils.helpers import ensure_chroma_directory

def parse_docs(docs):
    """Split base64-encoded images and texts"""
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
    """Build a prompt that includes text context and images"""
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

    # Construct prompt with context
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and images.
    
    Context: {context_text}
    
    {chat_history_text}
    
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    # Add images to the prompt if available
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

def create_multimodal_retriever(texts, tables, images, text_summaries, table_summaries, image_summaries, embeddings):
    """Create a MultiVectorRetriever for multimodal content"""
    # Create a unique ChromaDB collection for this session
    chroma_dir = ensure_chroma_directory()
    collection_name = f"multimodal_rag_{uuid.uuid4().hex[:8]}"
    
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
    
    # Add texts
    if texts and text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(text_summaries)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, texts)))
    
    # Add tables
    if tables and table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]})
            for i, summary in enumerate(table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))
    
    # Add images
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
    """Create a MultiVectorRetriever for fallback text-only content"""
    # Create a unique ChromaDB collection
    chroma_dir = ensure_chroma_directory()
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
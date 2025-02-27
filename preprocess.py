import os
import uuid
import traceback
import chromadb
from unstructured.partition.pdf import partition_pdf
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema.document import Document
import joblib  # for efficient serialization
import dotenv
from PyPDF2 import PdfReader  # as a fallback

# Load environment variables
dotenv.load_dotenv()

# Constants
PRELOADED_PDF_DIRECTORY = "./preloaded_pdfs_test"
PREPROCESSED_DATA_PATH = "./preprocessed_data"

class PrimaryCollectionPreprocessor:
    def __init__(self, model_name="gpt-4o-mini"):
        # Ensure output directory exists
        os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)
        
        # Initialize embedding and chat models
        self.embeddings = OpenAIEmbeddings()
        self.model = ChatOpenAI(model=model_name, temperature=0.2)
    
    def extract_pdf_text_fallback(self, pdf_path):
        """
        Fallback method to extract text using PyPDF2
        """
        documents = []
        try:
            print(f"Using fallback method for {os.path.basename(pdf_path)}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            doc = Document(
                                page_content=text.strip(),
                                metadata={
                                    "source": os.path.basename(pdf_path),
                                    "page": page_num + 1
                                }
                            )
                            documents.append(doc)
                    except Exception as page_error:
                        print(f"Error extracting text from page {page_num} of {pdf_path}: {page_error}")
        except Exception as e:
            print(f"Error processing {pdf_path} with PyPDF2: {e}")
        
        print(f"Fallback extracted {len(documents)} chunks from {os.path.basename(pdf_path)}")
        return documents
    
    def process_pdf_with_unstructured(self, pdf_path):
        """
        Process a single PDF using Unstructured with fallback to PyPDF2
        Returns a list of text chunks with metadata
        """
        try:
            print(f"Processing {os.path.basename(pdf_path)} with Unstructured...")
            
            # Extract content using Unstructured
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=2000,
                new_after_n_chars=1500
            )
            
            # Prepare documents with metadata
            documents = []
            for element in elements:
                if hasattr(element, 'text') and element.text.strip():
                    doc = Document(
                        page_content=element.text.strip(),
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "type": str(type(element))
                        }
                    )
                    documents.append(doc)
            
            print(f"Extracted {len(documents)} chunks from {os.path.basename(pdf_path)}")
            
            # If no documents extracted, use fallback
            if not documents:
                print(f"No content extracted with Unstructured from {os.path.basename(pdf_path)}. Trying fallback.")
                documents = self.extract_pdf_text_fallback(pdf_path)
            
            return documents
        
        except Exception as e:
            print(f"Error processing {os.path.basename(pdf_path)} with Unstructured: {e}")
            print(traceback.format_exc())
            
            # Fallback to PyPDF2
            return self.extract_pdf_text_fallback(pdf_path)
    
    def summarize_documents(self, documents):
        """
        Generate summaries for documents
        """
        print(f"Generating summaries for {len(documents)} documents...")
        summaries = []
        for i, doc in enumerate(documents):
            # Print progress
            if i % 10 == 0:
                print(f"Summarizing document {i+1}/{len(documents)}...")
                
            # Create a summary prompt
            summary_prompt = f"""
            Provide a concise, informative summary of the following text.
            Focus on the key points and main ideas.
            
            Text:
            {doc.page_content[:1000]}  # Limit to first 1000 characters
            """
            
            try:
                summary = self.model.invoke(summary_prompt).content
                summaries.append(summary)
            except Exception as e:
                print(f"Error generating summary for document {i+1}: {e}")
                print(traceback.format_exc())
                # Add a placeholder summary to maintain alignment with documents
                summaries.append("Summary not available due to processing error.")
        
        print(f"Generated {len(summaries)} summaries")
        return summaries
    
    def generate_embeddings(self, documents):
        """
        Generate embeddings for documents with better error handling
        """
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings_list = []
        
        for i, doc in enumerate(documents):
            # Print progress
            if i % 10 == 0:
                print(f"Embedding document {i+1}/{len(documents)}...")
                
            try:
                embedding = self.embeddings.embed_query(doc.page_content)
                embeddings_list.append(embedding)
            except Exception as e:
                print(f"Error generating embedding for document {i+1}: {e}")
                print(traceback.format_exc())
                # Add a zero vector to maintain alignment with documents
                embeddings_list.append([0.0] * 1536)  # OpenAI embeddings are 1536 dimensions
        
        print(f"Generated {len(embeddings_list)} embeddings")
        return embeddings_list
    
    def verify_data_alignment(self, documents, summaries, embeddings):
        """
        Verify that all three lists are the same length
        """
        doc_len = len(documents)
        sum_len = len(summaries)
        emb_len = len(embeddings)
        
        print(f"Data verification: {doc_len} documents, {sum_len} summaries, {emb_len} embeddings")
        
        if doc_len != sum_len or doc_len != emb_len:
            print("WARNING: Data misalignment detected!")
            print(f"Documents: {doc_len}, Summaries: {sum_len}, Embeddings: {emb_len}")
            
            # Truncate to the minimum length to ensure alignment
            min_len = min(doc_len, sum_len, emb_len)
            documents = documents[:min_len]
            summaries = summaries[:min_len]
            embeddings = embeddings[:min_len]
            
            print(f"Truncated to {min_len} items for alignment")
        
        return documents, summaries, embeddings
    
    def preprocess_collection(self):
        """
        Preprocess all PDFs in the primary collection
        """
        # Ensure the directories exist
        os.makedirs(PRELOADED_PDF_DIRECTORY, exist_ok=True)
        
        # Get all PDF files
        pdf_files = [
            f for f in os.listdir(PRELOADED_PDF_DIRECTORY) 
            if f.lower().endswith('.pdf')
        ]
        
        print(f"Found {len(pdf_files)} PDF files in {PRELOADED_PDF_DIRECTORY}")
        
        if not pdf_files:
            print(f"No PDF files found in {PRELOADED_PDF_DIRECTORY}! Please add some PDF files.")
            return
        
        # Process PDFs
        all_documents = []
        processed_files = []
        failed_files = []
        
        for pdf_file in pdf_files:
            print(f"\nProcessing {pdf_file} ({len(processed_files) + 1}/{len(pdf_files)})")
            pdf_path = os.path.join(PRELOADED_PDF_DIRECTORY, pdf_file)
            
            try:
                # Process individual PDF
                pdf_documents = self.process_pdf_with_unstructured(pdf_path)
                
                if pdf_documents:
                    all_documents.extend(pdf_documents)
                    processed_files.append(pdf_file)
                    print(f"Successfully processed {pdf_file}. Extracted {len(pdf_documents)} chunks.")
                else:
                    print(f"No documents extracted from {pdf_file}")
                    failed_files.append(pdf_file)
            except Exception as e:
                print(f"Failed to process {pdf_file}: {e}")
                print(traceback.format_exc())
                failed_files.append(pdf_file)
        
        print(f"\nDocument extraction complete. Processed {len(processed_files)}/{len(pdf_files)} files.")
        print(f"Total documents extracted: {len(all_documents)}")
        
        if not all_documents:
            print("No documents were extracted from any PDFs. Cannot continue.")
            return
        
        # Generate summaries
        summaries = self.summarize_documents(all_documents)
        
        # Create embeddings
        embeddings = self.generate_embeddings(all_documents)
        
        # Verify data alignment
        all_documents, summaries, embeddings = self.verify_data_alignment(
            all_documents, summaries, embeddings
        )
        
        # Prepare data for saving
        preprocessed_data = {
            "documents": all_documents,
            "summaries": summaries,
            "embeddings": embeddings
        }
        
        # Save preprocessed data
        output_path = os.path.join(PREPROCESSED_DATA_PATH, "primary_collection.joblib")
        print(f"\nSaving preprocessed data to {output_path}...")
        
        try:
            joblib.dump(preprocessed_data, output_path)
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"Successfully saved preprocessed data ({file_size:.2f} MB)")
            print(f"Contains {len(all_documents)} documents, {len(summaries)} summaries, {len(embeddings)} embeddings")
        except Exception as e:
            print(f"Error saving preprocessed data: {e}")
            print(traceback.format_exc())
        
        # Print summary
        print("\nPreprocessing Summary:")
        print(f"- Total PDF files: {len(pdf_files)}")
        print(f"- Successfully processed: {len(processed_files)}")
        print(f"- Failed to process: {len(failed_files)}")
        if failed_files:
            print(f"- Failed files: {', '.join(failed_files)}")

def main():
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in .env file")
        return
    
    print("Starting preprocessing of PDF collection...")
    
    # Initialize and run preprocessor
    preprocessor = PrimaryCollectionPreprocessor()
    preprocessor.preprocess_collection()
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
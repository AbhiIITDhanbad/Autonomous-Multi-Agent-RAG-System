from pathlib import Path
import sys
import os
from typing import List , Tuple , Dict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM , OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate , PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from datasets import Dataset
import numpy as np
import json , time
# Configuration
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "llama3.2:1b"
EMBEDDINGS = OllamaEmbeddings(model=EMBEDDING_MODEL)

CHROMA_PATH = Path(__file__).parent.parent / "data" / "vector_db"

class Simple_RAG:
    def __init__(self, files_data):
        self.files_data = files_data
        
        v_db= r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\data\vector_db"
        if not os.path.exists(v_db):
            # 1. Load & Split Documents (Run Once)
            print("starting splitting...")
            self.all_splits = self.doc_load_and_split(files_data)
            print("spliting done")   
                # 2. Create Vector Store (Run Once)
            print("creating vector store")
            self.vector_store = self.create_vector_store(self.all_splits)
        else:
            try:
                self.vector_store = Chroma(
                embedding_function=EMBEDDINGS,
                collection_name="knowledge_base",
                persist_directory=str(v_db)
                )
            except:
                print("Error adjusting existing vectordb")
            
        
        # 3. Create RAG Chain (Run Once & Store in memory)
        self.chain = self.create_rag_chain(self.vector_store)
        
    def doc_load_and_split(self, files_data):
        """Load and split documents from file paths."""
        # FIX: Increased chunk_size to 600 to capture full paragraphs/context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      
            chunk_overlap=60
        )
        all_splits = []

        for entry in files_data:
            file_path = entry["path"]
            custom_metadata = entry["metadata"]
            
            print(f"Loading {file_path}...")
            
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                raw_documents = loader.load()
                
                # Add custom metadata
                for doc in raw_documents:
                    doc.metadata.update(custom_metadata)
                
                splits = splitter.split_documents(raw_documents)
                all_splits.extend(splits)
                print(f"  → Created {len(splits)} chunks")
                
            except Exception as e:
                print(f"  ❌ Error loading {file_path}: {e}")
        
        print(f"Total chunks created: {len(all_splits)}")
        return all_splits
    
    def create_vector_store(self, splits):
        """Create Chroma vector store from document splits."""
        if not splits:
            raise ValueError("No documents to create vector store!")
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=EMBEDDINGS,
            collection_name="knowledge_base",
            persist_directory=str(CHROMA_PATH)
        )
        
        print(f"✅ Vector store created with {len(splits)} documents (Indexed once)")
        return vector_store
    
    def create_rag_chain(self, vector_store):
        """Create RAG chain."""
        # Initialize LLM
        llm = OllamaLLM(model=LLM_MODEL, temperature=0.0)
        
        # Prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the provided context.
        
        Context:
        {context}
        
        Question: {input}
        
        If the answer is not in the context, say "I don't know".
        """)
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    def ask_question(self, question: str):
        """Ask a question using the RAG chain."""
        print(f"🤔 Question: {question}")
        
        # FIX: Directly use the self.chain created in __init__
        # No more re-indexing!
        result = self.chain.invoke({"input": question})
        
        return result

            
def load_golden_dataset(dataset_path: Path) -> List[Dict]:
    """Load and validate golden dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    
    print(f"✅ Loaded {len(data)} samples from golden dataset")
    return data


def run_rag_pipeline(rag_system: Simple_RAG, questions: List[str]) -> Tuple[List[str], List[List[str]], List[float]]:
    """Run questions through RAG system and collect outputs."""
    answers = []
    contexts_list = []
    latencies = []
    
    print(f"🔍 Running {len(questions)} queries through RAG pipeline...")
    
    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] Processing: {question}...")
        start_time = time.time()
        try:
            result = rag_system.ask_question(question)
            answer = result.get('answer', '')
            print("answer: ", answer)
            context_docs = result.get('context', [])
            contexts = []
            
            if context_docs:
                first_doc = context_docs[0]
                if hasattr(first_doc, 'page_content'):
                    contexts = [doc.page_content for doc in context_docs]
                elif isinstance(first_doc, str):
                    contexts = context_docs
                else:
                    contexts = [str(c) for c in context_docs]
            print(contexts)
            answers.append(answer)
            contexts_list.append(contexts)
            latencies.append(time.time() - start_time)
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            answers.append("Error processing query")
            contexts_list.append([""]) 
            latencies.append(0.0)
    
    return answers, contexts_list, latencies

def prepare_ragas_dataset(questions, answers, contexts_list, ground_truths) -> Dataset:
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths 
    }
    return Dataset.from_dict(dataset_dict)

def main():
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    FILES_DATA = [
        {"path": str(DATA_DIR / "ai_infra_compute.txt"), "metadata": {"source": "technical_report"}},
        {"path": str(DATA_DIR / "emerging_ai_hardware.txt"), "metadata": {"source": "article"}},
        {"path": str(DATA_DIR / "enterprise_ai.txt"), "metadata": {"source": "market_report"}},
        {"path": str(DATA_DIR / "llm_training_alignment.txt"), "metadata": {"source": "research_summary"}},
        {"path": str(DATA_DIR / "rag_and_evaluation.txt"), "metadata": {"source": "financial_report"}}
    ]
    BASE_DIR = Path(__file__).parent.parent
    GOLDEN_DATASET_PATH = BASE_DIR / "benchmark" / "golden_dataset.json"
    rag=Simple_RAG(FILES_DATA)
    golden_data= load_golden_dataset(GOLDEN_DATASET_PATH)
    print("dataset loaded")
    questions = [item["question"] for item in golden_data]
    ground_truths = [item["ground_truth"] for item in golden_data]
    answers, contexts_list, latencies = run_rag_pipeline(rag, questions)
    dataset = prepare_ragas_dataset(questions, answers, contexts_list, ground_truths)
    
    print("ragas level dataset>>>>>>>")

    print(dataset)
    print("*"*50)
    print("baseline average latency : ", (np.sum(latencies)/len(latencies)))
        

if __name__ == "__main__":
    main()
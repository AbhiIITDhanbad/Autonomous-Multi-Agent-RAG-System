# updated script
"""
Document Retrieval Agent: Retrieves relevant documents using enhanced hybrid search.
Combines vector similarity, BM25, query expansion, and reranking.
"""
import sys
import os
import asyncio
from typing import Dict, Any, List
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama  # Required for query expansion

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state
# FIX: Import the new enhanced retriever factory
from retrieval.enhanced_hybrid_search import create_enhanced_retriever


class DocRetrievalAgent(BaseAgent):
    """
    Document Retrieval Agent using Enhanced Hybrid Search.
    
    Responsibilities:
    1. Load and manage vector database
    2. Perform query expansion (via LLM)
    3. Hybrid search (BM25 + Semantic)
    4. Reranking and filtering
    5. Provide document context
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize DocRetrievalAgent."""
        super().__init__(
            name="doc_retrieval_agent",
            description="Retrieves documents using enhanced hybrid search",
            timeout_seconds=300,
            max_retries=2
        )
        
        self.config = config or {}
        self.retriever = None
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the enhanced hybrid retriever."""
        try:
            # 1. Get configuration
            retrieval_config = self.config.get("retrieval", {})
            
            # Path to vector DB
            vector_db_path = Path(__file__).parent.parent.parent / "data" / "vector_db"
            
            if not vector_db_path.exists():
                # Fallback for testing environments
                vector_db_path = Path("data/vector_db")
                if not vector_db_path.exists():
                    print(f"⚠️ Vector DB path not found: {vector_db_path}")
            
            # 2. Initialize LLM for Query Expansion
            # We reuse the default model config for the expansion task
            llm_config = self.config.get("llm", {})
            model_name = llm_config.get("default_model", "llama3.1:8b")
            
            query_expansion_llm = ChatOllama(
                model=model_name,
                temperature=0.0
            )
            
            # 3. Enhanced Search Configuration
            hybrid_config = retrieval_config.get("hybrid_search", {})
            
            # Map config to enhanced retriever parameters
            enhanced_config = {
                "bm25_weight": hybrid_config.get("bm25_weight", 0.25),
                "semantic_weight": hybrid_config.get("semantic_weight", 0.65),
                "reranker_weight": 0.10,  # Default for enhanced
                "k": 15,                  # Fetch more for reranking (candidate pool)
                "final_k": retrieval_config.get("k", 5), # Final number to return
                "enable_query_expansion": True,
                "enable_reranking": True,
                "enable_metadata_boosting": True,
                "embedding_model": retrieval_config.get("embeddings", {}).get("model", "nomic-embed-text:latest")
            }
            
            # 4. Create Retriever
            self.retriever = create_enhanced_retriever(
                vector_db_path=vector_db_path,
                llm=query_expansion_llm,
                config=enhanced_config
            )
            
            print(f"✅ DocRetrievalAgent initialized with Enhanced Hybrid Search")
            print(f"   Config: Weights(BM25={enhanced_config['bm25_weight']}, Sem={enhanced_config['semantic_weight']}), Final K={enhanced_config['final_k']}")
            
        except Exception as e:
            print(f"❌ Failed to initialize retriever: {e}")
            import traceback
            traceback.print_exc()
            self.retriever = None
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents using enhanced search.
        """
        print(f"🔍 DocRetrievalAgent searching for relevant documents...")
        
        if not self.retriever:
            error_msg = "Retriever not initialized"
            print(f"❌ {error_msg}")
            state = add_error_to_state(state, self.name, error_msg)
            return state
        
        try:
            # Get query from state
            query = state.get("query", "")
            
            if not query:
                print("⚠️  No query found in state")
                state["retrieved_documents"] = []
                return state
            
            # Note: Enhanced retriever handles query expansion internally, 
            # so we pass the raw query (or slightly modified one) directly.
            
            # Perform Enhanced Retrieval (AWAIT is critical here)
            documents = await self.retriever.retrieve(query)
            
            # Update state
            state["retrieved_documents"] = documents
            print("Retreived_documents...")
            print(documents)
            print("*"*50)
            # Log retrieval
            print(f"✅ Retrieved {len(documents)} documents")
            
            # Show sample
            if documents:
                first_doc = documents[0]
                preview = first_doc.page_content[:100].replace('\n', ' ')
                print(f"   Top result: {preview}...")
                if first_doc.metadata:
                    print(f"   Metadata: {first_doc.metadata}")
            
            # Store comprehensive retrieval metadata
            if state.get("retrieval_metadata") is None:
                state["retrieval_metadata"] = {}
            
            stats = self.retriever.get_stats()
            
            state["retrieval_metadata"].update({
                "retrieval_method": "enhanced_hybrid",
                "total_documents_retrieved": len(documents),
                "expansion_rate": stats.get("expansion_rate", 0),
                "avg_score": stats.get("avg_final_score", 0),
                "retriever_stats": stats
            })
            
            return state
            
        except Exception as e:
            error_msg = f"Document retrieval failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            state = add_error_to_state(state, self.name, error_msg)
            state["retrieved_documents"] = []
            return state

# Test function
async def test_doc_retrieval_agent():
    """Test the DocRetrievalAgent."""
    print("🧪 Testing DocRetrievalAgent (Enhanced)...")
    
    # Initialize agent
    # Mimic structure from agents.yaml
    config = {
        "llm": {"default_model": "llama3.1:8b"},
        "retrieval": {
            "k": 3,
            "hybrid_search": {
                "bm25_weight": 0.25,
                "semantic_weight": 0.65
            }
        }
    }
    
    agent = DocRetrievalAgent(config)
    
    if not agent.retriever:
        print("❌ Agent failed to initialize retriever. Check paths.")
        return

    # Create test state
    state = {
        "query": "RAG adoption in enterprise 2023",
        "user_id": "test_user",
        "query_analysis": {},
        "intent": "research",
        "retrieved_documents": None,
        "errors": [],
        "agent_timestamps": {},
        "execution_path": [],
        "retrieval_metadata": None
    }
    
    # Execute
    result = await agent.execute(state)
    
    # Check results
    if result.get("retrieved_documents"):
        print(f"\n✅ Test Successful. Docs found: {len(result['retrieved_documents'])}")
        if result.get("retrieval_metadata"):
            print(f"📊 Stats: {result['retrieval_metadata']['retriever_stats']}")
    else:
        print("\n❌ Test Failed. No docs retrieved.")

if __name__ == "__main__":
    asyncio.run(test_doc_retrieval_agent())
# """
# Document Retrieval Agent: Retrieves relevant documents using hybrid search.
# Combines vector similarity search with BM25 keyword search for better recall.
# """
# import sys , os
# import asyncio
# from typing import Dict, Any, List
# from pathlib import Path

# from langchain_chroma import Chroma
# from langchain_ollama.embeddings import OllamaEmbeddings
# from langchain_core.documents import Document

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from agents.base_agent import BaseAgent
# from graph.state import AgentState, add_error_to_state
# from retrieval.hybrid_search import create_hybrid_retriever


# class DocRetrievalAgent(BaseAgent):
#     """
#     Document Retrieval Agent using hybrid search.
    
#     Responsibilities:
#     1. Load and manage vector database
#     2. Perform hybrid search (BM25 + semantic)
#     3. Filter and rank documents
#     4. Provide document context for synthesis
#     """
    
#     def __init__(self, config: Dict[str, Any] = None):
#         """Initialize DocRetrievalAgent."""
#         super().__init__(
#             name="doc_retrieval_agent",
#             description="Retrieves documents using hybrid search",
#             timeout_seconds=300,
#             max_retries=2
#         )
        
#         self.config = config or {}
#         self.retriever = None
#         self._initialize_retriever()
    
#     def _initialize_retriever(self):
#         """Initialize the hybrid retriever."""
#         try:
#             # Get configuration
#             retrieval_config = self.config.get("doc_retrieval_agent", {})
#             vector_store_config = retrieval_config.get("vector_store", {})
            
#             # Path to vector DB
#             vector_db_path = Path(__file__).parent.parent.parent / "data" / "vector_db"
            
#             if not vector_db_path.exists():
#                 raise FileNotFoundError(f"Vector DB not found at {vector_db_path}")
            
#             # Hybrid search configuration
#             hybrid_config = retrieval_config.get("hybrid_search", {})
#             bm25_weight = hybrid_config.get("bm25_weight", 0.25)
#             semantic_weight = hybrid_config.get("semantic_weight", 0.75)
#             k = retrieval_config.get("k", 3)
            
#             # Create hybrid retriever
#             self.retriever = create_hybrid_retriever(
#                 vector_db_path=vector_db_path,
#                 bm25_weight=bm25_weight,
#                 semantic_weight=semantic_weight,
#                 k=k
#             )
            
#             print(f"✅ DocRetrievalAgent initialized with hybrid search")
#             print(f"   Config: BM25={bm25_weight}, Semantic={semantic_weight}, k={k}")
            
#         except Exception as e:
#             print(f"❌ Failed to initialize retriever: {e}")
#             self.retriever = None
    
#     async def _execute_impl(self, state: AgentState) -> AgentState:
#         """
#         Retrieve relevant documents based on query and analysis.
        
#         Args:
#             state: Current AgentState
            
#         Returns:
#             Updated AgentState with retrieved documents
#         """
#         print(f"🔍 DocRetrievalAgent searching for relevant documents...")
        
#         if not self.retriever:
#             error_msg = "Retriever not initialized"
#             print(f"❌ {error_msg}")
#             state = add_error_to_state(state, self.name, error_msg)
#             return state
        
#         try:
#             # Get query from state
#             query = state.get("query", "")
            
#             if not query:
#                 print("⚠️  No query found in state")
#                 state["retrieved_documents"] = []
#                 return state
            
#             # Check if we need to modify query based on analysis
#             enhanced_query = self._enhance_query(query, state)
            
#             # Perform hybrid search
#             documents = self.retriever.retrieve(enhanced_query)
            
#             # Apply metadata filtering if needed
#             filtered_docs = self._filter_documents(documents, state)
            
#             # Update state
#             state["retrieved_documents"] = filtered_docs
            
#             # Log retrieval
#             print(f"✅ Retrieved {len(filtered_docs)} documents")
            
#             # Show sample of retrieved content
#             if filtered_docs:
#                 for i, doc in enumerate(filtered_docs[:2]):  # First 2 docs
#                     content_preview = doc.page_content[:80].replace('\n', ' ')
#                     print(f"  {i+1}. {content_preview}...")
            
#             # Store retrieval metadata
#             # if "retrieval_metadata" not in state:
#             #     state["retrieval_metadata"] = {}
#             if state.get("retrieval_metadata") is None:
#                 state["retrieval_metadata"] = {}
            
#             state["retrieval_metadata"].update({
#                 "retrieval_method": "hybrid",
#                 "query_used": enhanced_query,
#                 "total_documents_retrieved": len(documents),
#                 "documents_after_filtering": len(filtered_docs),
#                 "retriever_stats": self.retriever.get_stats()
#             })
            
#             return state
            
#         except Exception as e:
#             error_msg = f"Document retrieval failed: {str(e)}"
#             print(f"❌ {error_msg}")
#             state = add_error_to_state(state, self.name, error_msg)
#             state["retrieved_documents"] = []
#             return state
    
#     def _enhance_query(self, query: str, state: AgentState) -> str:
#         """
#         Enhance query based on analysis for better retrieval.
        
#         Args:
#             query: Original query
#             state: Current AgentState
            
#         Returns:
#             Enhanced query
#         """
#         # Basic enhancement - can be improved
#         enhanced = query
        
#         # Add intent information if available
#         intent = state.get("intent", "")
#         if intent:
#             # Add intent-specific keywords
#             intent_keywords = {
#                 "comparison": "compare comparison vs versus difference",
#                 "research": "research study findings report",
#                 "analysis": "analyze analysis trend pattern",
#                 "synthesis": "synthesize combine summary overview"
#             }
#             if intent in intent_keywords:
#                 enhanced = f"{enhanced} {intent_keywords[intent]}"
        
#         # Add key entities from analysis if available
#         analysis = state.get("query_analysis", {})
#         metadata = analysis.get("metadata", {})
#         key_entities = metadata.get("key_entities", [])
        
#         if key_entities:
#             entities_str = " ".join(key_entities)
#             enhanced = f"{enhanced} {entities_str}"
        
#         print(f"   Enhanced query: {enhanced[:100]}...")
#         return enhanced
    
#     def _filter_documents(self, documents: List[Document], state: AgentState) -> List[Document]:
#         """
#         Filter documents based on metadata and relevance.
        
#         Args:
#             documents: Retrieved documents
#             state: Current AgentState
            
#         Returns:
#             Filtered documents
#         """
#         if not documents:
#             return []
        
#         # Get analysis for filtering criteria
#         analysis = state.get("query_analysis", {})
#         metadata = analysis.get("metadata", {})
        
#         filtered_docs = []
        
#         for doc in documents:
#             # Check metadata if available
#             doc_metadata = doc.metadata or {}
            
#             # Apply time sensitivity filter if needed
#             time_sensitivity = metadata.get("time_sensitivity", "low")
#             if time_sensitivity == "high" and "date" in doc_metadata:
#                 # In Phase 1, we don't have dates in metadata
#                 # This is a placeholder for future implementation
#                 pass
            
#             # Apply domain filter if available
#             domains = metadata.get("domains", [])
#             if domains and "domain" in doc_metadata:
#                 doc_domain = doc_metadata.get("domain", "").lower()
#                 if doc_domain and not any(domain.lower() in doc_domain for domain in domains):
#                     continue  # Skip if domain doesn't match
            
#             filtered_docs.append(doc)
        
#         # If filtering removed all docs, return original list
#         if not filtered_docs and documents:
#             print("⚠️  Filtering removed all documents, using unfiltered results")
#             return documents
        
#         return filtered_docs
    
#     async def _handle_timeout(self, state: AgentState) -> AgentState:
#         """Handle timeout gracefully."""
#         print("⏰ DocRetrievalAgent timed out, returning empty results")
#         state["retrieved_documents"] = []
#         state["retrieval_metadata"] = {
#             "retrieval_method": "timeout_fallback",
#             "error": f"Timeout after {self.timeout_seconds} seconds"
#         }
#         return state


# # Test function for the agent
# async def test_doc_retrieval_agent():
#     """Test the DocRetrievalAgent."""
#     print("🧪 Testing DocRetrievalAgent...")
    
#     # Initialize agent with minimal config
#     config = {
#         "retrieval": {
#             "k": 3,
#             "hybrid_search": {
#                 "bm25_weight": 0.3,
#                 "semantic_weight": 0.7
#             }
#         }
#     }
    
#     agent = DocRetrievalAgent(config)
    
#     # Create a test state
#     state = {
#             "query": "Compare NVIDIA H200 and AMD MI300X for AI workloads",
#             "user_id": "test_user",
#             "query_analysis": {
#                 "intent": "comparison",
#                 "metadata": {
#                     "key_entities": ["NVIDIA", "AMD", "H200", "MI300X"],
#                     "domains": ["hardware", "AI"],
#                     "time_sensitivity": "medium"
#                 }
#             },
#             "intent": "comparison",
#             "decomposed_tasks": [],
#             "retrieved_documents": None,
#             "research_results": None,
#             "fact_check_results": None,
#             "intermediate_answers": None,
#             "final_answer": None,
#             "confidence_score": None,
#             "citations": None,
#             "retrieval_sources": [],
#             "errors": [],
#             "agent_timestamps": {},   # <--- REQUIRED: Missing in original code
#             "execution_path": [],     # <--- REQUIRED: Missing in original code
#             "latency_per_agent": {},
#             "token_usage": {},
#             "retrieval_metadata": None
#         }
    
#     # Execute agent
#     result = await agent.execute(state)
    
#     # Print results
#     if result["retrieved_documents"]:
#         docs = result["retrieved_documents"]
#         print(f"✅ Retrieved {len(docs)} documents")
        
#         for i, doc in enumerate(docs[:2]):  # Show first 2
#             print(f"\n📄 Document {i+1}:")
#             print(f"   Content: {doc.page_content[:150]}...")
#             if doc.metadata:
#                 print(f"   Metadata: {list(doc.metadata.keys())}")
#     else:
#         print("❌ No documents retrieved")
    
#     # Print retrieval metadata
#     if "retrieval_metadata" in result:
#         print(f"\n📊 Retrieval Metadata:")
#         if result["retrieval_metadata"] is not None:
#             for key, value in result["retrieval_metadata"].items():
#                 if key != "retriever_stats":
#                     print(f"  {key}: {value}")
#         else:
#             print(f"{result["retrieval_metadata"]} is NoneType","\n")
#     return result



# if __name__ == "__main__":
#     asyncio.run(test_doc_retrieval_agent())

"""
Enhanced Hybrid Retrieval with Reranking, Query Expansion, and Metadata Boosting.
Week 4: Goal of >10% recall improvement.
"""
import sys
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("⬇️ Downloading NLTK 'stopwords' data...")
    nltk.download('stopwords')

class EnhancedHybridSearchRetriever:
    """
    Enhanced hybrid search with:
    1. Query expansion using LLM
    2. Cross-encoder reranking
    3. Metadata boosting (recency, source type)
    4. Diversity-aware retrieval
    5. Adaptive weight tuning
    """
    
    def __init__(
        self,
        vector_store: Chroma,
        llm=None,  # For query expansion
        bm25_weight: float = 0.25,
        semantic_weight: float = 0.75,
        reranker_weight: float = 0.10,
        k: int = 10,  # Retrieve more for reranking
        final_k: int = 5,  # Return after reranking
        enable_query_expansion: bool = True,
        enable_reranking: bool = True,
        enable_metadata_boosting: bool = True
    ):
        """
        Initialize enhanced hybrid retriever.
        
        Args:
            vector_store: Chroma vector store
            llm: Language model for query expansion
            bm25_weight: Weight for BM25 scores
            semantic_weight: Weight for semantic scores
            reranker_weight: Weight for reranker scores
            k: Initial retrieval count
            final_k: Final documents to return
        """
        self.vector_store = vector_store
        self.llm = llm
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.reranker_weight = reranker_weight
        self.k = k
        self.final_k = final_k
        
        # Features
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking
        self.enable_metadata_boosting = enable_metadata_boosting
        
        # Initialize components
        self.bm25_index = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Metadata field weights
        self.metadata_weights = {
            'recency': 0.3,  # Recent documents get boost
            'source_type': {
                'technical_spec': 0.9,
                'academic_paper': 0.95,
                'company_website': 0.85,
                'news_article': 0.75,
                'industry_report': 0.8,
                'blog': 0.6
            }
        }
        
        # Query expansion cache
        self.query_expansion_cache = {}
        
        # Statistics
        self.retrieval_stats = {
            "total_queries": 0,
            "with_expansion": 0,
            "with_reranking": 0,
            "recall_improvements": [],
            "avg_final_score": 0.0
        }
        
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 index with enhanced preprocessing."""
        try:
            collection = self.vector_store._collection
            if collection is None:
                raise ValueError("Vector store collection is empty")
            
            results = collection.get(include=["documents", "metadatas"])
            
            if not results["documents"]:
                print("⚠️  No documents found in vector store")
                return
            
            self.documents = []
            doc_texts = []
            
            for i, doc_text in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                document = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                self.documents.append(document)
                doc_texts.append(doc_text)
            
            # Enhanced tokenization with stemming and stopword removal
            tokenized_docs = [self._enhanced_tokenize(doc) for doc in doc_texts]
            
            # Create BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            
            print(f"✅ Enhanced BM25 initialized with {len(self.documents)} documents")
            
        except Exception as e:
            print(f"❌ Failed to initialize BM25: {e}")
            self.bm25_index = None
    
    def _enhanced_tokenize(self, text: str) -> List[str]:
        """
        Enhanced tokenization with:
        - Lowercasing
        - Stopword removal
        - Stemming
        - Special character handling
        - Number preservation
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Process tokens
        enhanced_tokens = []
        for token in tokens:
            # Remove very short tokens
            if len(token) < 2:
                continue
            
            # Keep numbers
            if token.isdigit():
                enhanced_tokens.append(token)
                continue
            
            # Remove stopwords
            if token in self.stop_words:
                continue
            
            # Remove non-alphanumeric (except hyphens in compound words)
            if not re.match(r'^[a-zA-Z0-9-]+$', token):
                continue
            
            # Apply stemming
            stemmed = self.stemmer.stem(token)
            if len(stemmed) > 1:
                enhanced_tokens.append(stemmed)
        
        return enhanced_tokens
    
    async def _expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand query using LLM to generate synonyms and related terms.
        
        Returns:
            Tuple of (expanded_query, expansion_terms)
        """
        if not self.enable_query_expansion or not self.llm:
            return query, []
        
        # Check cache
        if query in self.query_expansion_cache:
            self.retrieval_stats["with_expansion"] += 1
            return self.query_expansion_cache[query]
        
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query expansion expert.
Return a valid JSON object with synonyms and technical terms.
Example: {{ "expanded_query": "original query term1 term2", "expansion_terms": ["term1", "term2"] }}
Do not add any text before or after the JSON."""),
                ("human", "{query}")
            ])
            
            chain = prompt | self.llm
            result = await chain.ainvoke({"query": query})
            
            if isinstance(result.content, str):
                import json
                content = result.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                try:
                    expansion_data = json.loads(content)
                    expanded_query = expansion_data.get("expanded_query", query)
                    expansion_terms = expansion_data.get("expansion_terms", [])
                except json.JSONDecodeError:
                    print(f"⚠️ JSON Decode Error in query expansion. Using original query.")
                    expanded_query = query
                    expansion_terms = []
            else:
                expanded_query = query
                expansion_terms = []
            
            self.query_expansion_cache[query] = (expanded_query, expansion_terms)
            self.retrieval_stats["with_expansion"] += 1
            
            print(f"🔍 Query expanded: {len(expansion_terms)} terms added")
            return expanded_query, expansion_terms
            
        except Exception as e:
            print(f"⚠️  Query expansion failed: {e}")
            return query, []
    
    def _bm25_search(self, query: str, expansion_terms: List[str] = None) -> List[Tuple[Document, float]]:
        """Perform BM25 search with query expansion."""
        if not self.bm25_index or not self.documents:
            return []
        
        # Combine query with expansion terms
        search_terms = [query]
        if expansion_terms:
            search_terms.extend(expansion_terms)
        
        # Search for each term and combine scores
        all_scores = {}
        
        for term in search_terms:
            term_tokens = self._enhanced_tokenize(term)
            if not term_tokens:
                continue
            
            scores = self.bm25_index.get_scores(term_tokens)
            
            for i, score in enumerate(scores):
                if score > 0:
                    if i not in all_scores:
                        all_scores[i] = 0.0
                    all_scores[i] = max(all_scores[i], score)  # Max pooling
        
        # Convert to list of (document, score)
        bm25_results = []
        for doc_idx, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            if len(bm25_results) >= self.k * 2:  # Get more for reranking
                break
            bm25_results.append((self.documents[doc_idx], score))
        
        return bm25_results
    
    def _semantic_search(self, query: str) -> List[Tuple[Document, float]]:
        """Perform semantic vector search."""
        try:
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, 
                k=self.k * 2  # Get more for reranking
            )
            return results
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return []
    
    def _calculate_metadata_score(self, doc: Document) -> float:
        """Calculate metadata-based score boost."""
        if not self.enable_metadata_boosting:
            return 1.0
        
        metadata = doc.metadata or {}
        total_score = 0.0
        weight_sum = 0.0
        
        # Recency boost
# Recency boost
        if 'date' in metadata and metadata['date'] and metadata['date'] != "unknown":
            try:
                # Handle YYYY-MM-DD format commonly used in metadata
                date_str = metadata['date'].replace('Z', '+00:00')
                if len(date_str) == 10: # YYYY-MM-DD
                    doc_date = datetime.strptime(date_str, "%Y-%m-%d")
                else:
                    doc_date = datetime.fromisoformat(date_str)
                
                days_old = (datetime.now() - doc_date).days
                # ... (rest of logic remains same)
                
                # Exponential decay: newer = higher score
                if days_old <= 7:  # Last week
                    recency_score = 1.0
                elif days_old <= 30:  # Last month
                    recency_score = 0.9
                elif days_old <= 90:  # Last quarter
                    recency_score = 0.7
                elif days_old <= 365:  # Last year
                    recency_score = 0.5
                else:
                    recency_score = 0.3
                
                total_score += recency_score * self.metadata_weights['recency']
                weight_sum += self.metadata_weights['recency']
            except:
                pass
        
        # Source type boost
        if 'source_type' in metadata:
            source_type = metadata['source_type']
            source_weights = self.metadata_weights['source_type']
            
            if source_type in source_weights:
                source_score = source_weights[source_type]
                total_score += source_score * 0.2  # Fixed weight for source
                weight_sum += 0.2
        
        # Normalize
        if weight_sum > 0:
            return total_score / weight_sum
        return 1.0
    
    def _rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Simple cross-encoder style reranking.
        In production, use a trained cross-encoder model.
        """
        if not self.enable_reranking or not documents:
            return [(doc, 0.5) for doc in documents]
        
        try:
            # Simple TF-IDF based reranking as placeholder
            # In Phase 2, replace with actual cross-encoder
            query_terms = set(self._enhanced_tokenize(query))
            
            reranked = []
            for doc in documents:
                doc_terms = set(self._enhanced_tokenize(doc.page_content))
                
                # Jaccard similarity
                if query_terms and doc_terms:
                    intersection = len(query_terms.intersection(doc_terms))
                    union = len(query_terms.union(doc_terms))
                    similarity = intersection / union if union > 0 else 0
                else:
                    similarity = 0
                
                # Boost for query terms in document
                reranked.append((doc, similarity))
            
            # Sort by similarity
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            self.retrieval_stats["with_reranking"] += 1
            return reranked
            
        except Exception as e:
            print(f"⚠️  Reranking failed: {e}")
            return [(doc, 0.5) for doc in documents]
    
    def _calculate_hybrid_score(self, doc_info: Dict, query: str) -> float:
        """Calculate enhanced hybrid score with multiple factors."""
        
        # Base hybrid score
        hybrid_score = (
            doc_info["bm25_score"] * self.bm25_weight +
            doc_info["semantic_score"] * self.semantic_weight +
            doc_info["reranker_score"] * self.reranker_weight
        )
        
        # Apply metadata boost
        metadata_boost = self._calculate_metadata_score(doc_info["document"])
        boosted_score = hybrid_score * metadata_boost
        
        # Apply length penalty (avoid very short/long docs)
        doc_length = len(doc_info["document"].page_content.split())
        if doc_length < 50:  # Too short
            length_penalty = 0.8
        elif doc_length > 1000:  # Too long
            length_penalty = 0.9
        else:
            length_penalty = 1.0
        
        final_score = boosted_score * length_penalty
        
        return min(1.0, final_score)  # Cap at 1.0
    
    async def retrieve(self, query: str) -> List[Document]:
        """
        Enhanced hybrid retrieval with query expansion and reranking.
        
        Args:
            query: Original search query
            
        Returns:
            List of ranked documents
        """
        self.retrieval_stats["total_queries"] += 1
        
        print(f"🔍 Enhanced hybrid search for: {query[:80]}...")
        
        # Step 1: Query Expansion
        expanded_query, expansion_terms = await self._expand_query(query)
        
        # Step 2: Parallel retrieval
        bm25_results = self._bm25_search(expanded_query, expansion_terms)
        semantic_results = self._semantic_search(expanded_query)
        
        # Step 3: Combine and deduplicate
        all_docs = {}
        
        # Add BM25 docs
        for doc, score in bm25_results:
            doc_id = hash(doc.page_content[:200])  # Better ID
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "document": doc,
                    "bm25_score": score,
                    "semantic_score": 0.0,
                    "reranker_score": 0.5  # Default
                }
        
        # Add semantic docs
        for doc, score in semantic_results:
            doc_id = hash(doc.page_content[:200])
            if doc_id not in all_docs:
                all_docs[doc_id] = {
                    "document": doc,
                    "bm25_score": 0.0,
                    "semantic_score": score,
                    "reranker_score": 0.5
                }
            else:
                # Update semantic score if higher
                all_docs[doc_id]["semantic_score"] = max(
                    all_docs[doc_id]["semantic_score"], score
                )
        
        # Step 4: Rerank top candidates
        top_candidates = list(all_docs.values())[:self.k * 3]  # Top for reranking
        documents_for_reranking = [info["document"] for info in top_candidates]
        
        reranked_results = self._rerank_with_cross_encoder(
            expanded_query, documents_for_reranking
        )
        
        # Update reranker scores
        reranked_dict = {hash(doc.page_content[:200]): score for doc, score in reranked_results}
        for doc_id, doc_info in all_docs.items():
            if doc_id in reranked_dict:
                doc_info["reranker_score"] = reranked_dict[doc_id]
        
        # Step 5: Calculate final hybrid scores
        scored_docs = []
        for doc_info in all_docs.values():
            # Normalize scores
            bm25_norm = self._normalize_score(doc_info["bm25_score"], 0, 10)
            semantic_norm = self._normalize_score(doc_info["semantic_score"], 0, 1)
            reranker_norm = self._normalize_score(doc_info["reranker_score"], 0, 1)
            
            # Update with normalized scores
            doc_info["bm25_score"] = bm25_norm
            doc_info["semantic_score"] = semantic_norm
            doc_info["reranker_score"] = reranker_norm
            
            # Calculate hybrid score
            hybrid_score = self._calculate_hybrid_score(doc_info, query)
            doc_info["hybrid_score"] = hybrid_score
            
            scored_docs.append(doc_info)
        
        # Step 6: Sort by hybrid score
        scored_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Step 7: Apply diversity (avoid similar documents)
        final_docs = self._apply_diversity(scored_docs)
        
        # Update statistics
        if scored_docs:
            self.retrieval_stats["avg_final_score"] = float(np.mean(
                [d["hybrid_score"] for d in scored_docs[:self.final_k]]
            ))
        
        # Return final documents
        result_docs = [d["document"] for d in final_docs[:self.final_k]]
        
        # Log results
        print(f"✅ Enhanced retrieval: {len(result_docs)} documents")
        if result_docs:
            top_score = final_docs[0]["hybrid_score"]
            print(f"   Top score: {top_score:.3f}")
            print(f"   Components - BM25: {final_docs[0]['bm25_score']:.3f}, "
                  f"Semantic: {final_docs[0]['semantic_score']:.3f}, "
                  f"Reranker: {final_docs[0]['reranker_score']:.3f}")
        
        return result_docs
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        if max_val == min_val:
            return 0.5
        normalized = (score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def _apply_diversity(self, scored_docs: List[Dict]) -> List[Dict]:
        """
        Apply diversity to avoid similar documents in top results.
        Simple cosine similarity based deduplication.
        """
        if len(scored_docs) <= self.final_k:
            return scored_docs
        
        selected = []
        selected_content = []
        
        for doc_info in scored_docs:
            content = doc_info["document"].page_content
            
            # Check similarity with already selected documents
            too_similar = False
            for selected_content_item in selected_content:
                # Simple Jaccard similarity
                words1 = set(content.lower().split()[:50])
                words2 = set(selected_content_item.lower().split()[:50])
                
                if words1 and words2:
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.7:  # Too similar threshold
                        too_similar = True
                        break
            
            if not too_similar:
                selected.append(doc_info)
                selected_content.append(content)
            
            if len(selected) >= self.final_k:
                break
        
        # If we don't have enough diverse docs, add remaining high-scoring ones
        if len(selected) < self.final_k:
            remaining = [d for d in scored_docs if d not in selected]
            selected.extend(remaining[:self.final_k - len(selected)])
        
        return selected
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = self.retrieval_stats.copy()
        
        if stats["total_queries"] > 0:
            stats["expansion_rate"] = stats["with_expansion"] / stats["total_queries"]
            stats["reranking_rate"] = stats["with_reranking"] / stats["total_queries"]
        
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.retrieval_stats = {
            "total_queries": 0,
            "with_expansion": 0,
            "with_reranking": 0,
            "recall_improvements": [],
            "avg_final_score": 0.0
        }

# Factory function
def create_enhanced_retriever(
    vector_db_path: Path,
    llm=None,
    config: Dict[str, Any] = None
) -> EnhancedHybridSearchRetriever:
    """
    Create enhanced hybrid retriever.
    
    Args:
        vector_db_path: Path to Chroma vector database
        llm: Language model for query expansion
        config: Configuration dictionary
        
    Returns:
        EnhancedHybridSearchRetriever instance
    """
    config = config or {}
    
    # Initialize embeddings
    embedding_model = config.get("embedding_model", "nomic-embed-text:latest")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Load vector store
    vector_store = Chroma(
        persist_directory=str(vector_db_path),
        embedding_function=embeddings,
        collection_name="knowledge_base"
    )
    
    # Create enhanced retriever
    retriever = EnhancedHybridSearchRetriever(
        vector_store=vector_store,
        llm=llm,
        bm25_weight=config.get("bm25_weight", 0.25),
        semantic_weight=config.get("semantic_weight", 0.65),
        reranker_weight=config.get("reranker_weight", 0.10),
        k=config.get("k", 10),
        final_k=config.get("final_k", 5),
        enable_query_expansion=config.get("enable_query_expansion", True),
        enable_reranking=config.get("enable_reranking", True),
        enable_metadata_boosting=config.get("enable_metadata_boosting", True)
    )
    
    return retriever

# Test function
async def test_enhanced_retrieval():
    """Test enhanced hybrid retrieval."""
    print("🧪 Testing Enhanced Hybrid Retrieval")
    print("=" * 60)
    
    from langchain_ollama import ChatOllama
    
    # Initialize LLM for query expansion
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
    
    # Path to vector DB
    vector_db_path = Path(__file__).parent.parent.parent / "data" / "vector_db"
    
    if not vector_db_path.exists():
        print(f"❌ Vector DB not found at {vector_db_path}")
        return
    
    # Create enhanced retriever
    retriever = create_enhanced_retriever(
        vector_db_path=vector_db_path,
        llm=llm,
        config={
            "bm25_weight": 0.25,
            "semantic_weight": 0.75,
            "reranker_weight": 0.1,
            "k": 15,
            "final_k": 5
        }
    )
    
    # Test queries
    test_queries = [

        "RAG adoption in enterprise earnings calls 2023",
        "LLM deployment challenges and solutions",
        "Compare NVIDIA H200 and AMD MI300X performance",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 40)
        
        results = await retriever.retrieve(query)
        
        print(f"📄 Retrieved {len(results)} documents")
        
        for i, doc in enumerate(results[:3]):  # Show first 3
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i+1}. {content_preview}...")
            
            if doc.metadata:
                source_type = doc.metadata.get("source_type", "unknown")
                date = doc.metadata.get("date", "unknown")
                print(f"     Source: {source_type}, Date: {date}")
    
    # Print statistics
    print(f"\n📊 Enhanced Retrieval Statistics:")
    stats = retriever.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    import asyncio 
    asyncio.run(test_enhanced_retrieval())                                     
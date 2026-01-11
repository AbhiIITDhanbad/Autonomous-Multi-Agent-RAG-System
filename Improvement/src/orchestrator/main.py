#!/usr/bin/env python3
"""
Main Orchestrator for Phase 1 Multi-Agent RAG System.
Executes the complete LangGraph workflow including:
1. Query Analyzer
2. Research Agent (Tavily)
3. Doc Retrieval Agent (Enhanced Hybrid)
4. Fact Verification Agent
5. Synthesis Agent
"""
import os
import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import time
import json
# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.workflow import MultiAgentGraph
from graph.state import get_state_summary
 

class Phase1Orchestrator:
    """
    Orchestrator for the Phase 1 System using LangGraph.
    Manages configuration, workflow initialization, and test execution.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.workflow = None
        self._initialize_workflow()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults."""
        default_config = {
            "llm": {
                "default_model": "llama3.2:1b", 
                "research_model": "llama3.1:8b",
                "synthesis_model": "llama3.1:8b",
                "temperature": 0.0
            },
            "agents": {
                "timeout_seconds": 600,
                "max_retries": 2
            },
            "workflow": {
                "max_total_timeout": 600,
                "use_langgraph": True
            },
            "retrieval": {
                "k": 5,
                "hybrid_search": {
                    "bm25_weight": 0.25,
                    "semantic_weight": 0.75
                }
            },
            "fact_verification": {
                "contradiction_threshold": 0.7,
                "min_sources": 2
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "enable_streaming": True
            },
            "evaluation": {
                "golden_queries_path": "data/golden_queries.json",
                "metrics": ["faithfulness", "answer_relevance", "context_recall"]
            }
        }
            
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    
                    # Recursive update helper
                    def deep_update(target, source):
                        for key, value in source.items():
                            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                                deep_update(target[key], value)
                            else:
                                target[key] = value
                    
                    if loaded_config:
                        deep_update(default_config, loaded_config)
                        print(f"✅ Configuration loaded from {config_path}")
            except Exception as e:
                print(f"⚠️ Error loading config from {config_path}: {e}")
                print("Using default configuration.")
        else:
            print("ℹ️ No config file found. Using default configuration.")
        
        return default_config
    
    def _initialize_workflow(self):
        """Initialize the LangGraph workflow."""
        print("\n🔄 Initializing LangGraph workflow...")
        try:
            self.workflow = MultiAgentGraph(self.config)
            print("✅ Workflow initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize workflow: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    async def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a query through the complete system."""
        print(f"\n🔍 Processing: {query[:80]}...")
        
        if not self.workflow:
            return {"error": "Workflow not initialized"}
        
        # Process using LangGraph
        result = await self.workflow.process_query(query, user_id)
        
        # Safe extraction helpers
        research_res = result.get("research_results") or {}
        docs = result.get("retrieved_documents") or []
        citations = result.get("citations") or []
        errors = result.get("errors") or []
        
        # Extract Verification Data
        verified_facts = result.get("verified_facts") or []
        contradictions = result.get("contradiction_report") or []

        # Create response
        response = {
            "query": query,
            "answer": result.get("final_answer"),
            "confidence": result.get("confidence_score", 0),
            "intent": result.get("intent"),
            "execution_path": result.get("execution_path", []),
            
            # Detailed Statistics
            "stats": {
                "research_findings": len(research_res.get("findings", [])),
                "retrieved_documents": len(docs),
                "citations_count": len(citations),
                "verified_facts_count": len(verified_facts),
                "contradictions_found": len(contradictions)
            },
            
            "errors": errors,
            "state_summary": get_state_summary(result)
        }
        
        return response
    
    async def run_golden_queries_test(self):
        """Run tests with golden queries."""
        # golden_queries = [
        #     {
        #         "query": "Source 1 says quantum supremacy was achieved in 2019 with a 53-qubit processor. Source 2 claims it hasn't been practically achieved yet. Cross-reference these with at least three authoritative sources (e.g., Nature, IBM, Google) to determine the consensus.",
        #         "type": "complex_comparison"
        #     },
        #     {
        #         "query": "List the primary data sources and APIs that a 'Competitive Intelligence Dashboard' for the Cloud AI market would need to monitor, based on the methodologies described in Gartner's 2023 Market Guide.",
        #         "type": "research_simple"
        #     }
        # ]
        BASE_DIR = Path(__file__).parent.parent.parent
        GOLDEN_DATASET_PATH = BASE_DIR / "benchmarks" / "golden_dataset.json"
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
        # golden_data= load_golden_dataset(GOLDEN_DATASET_PATH)
        # print("dataset loaded")
        # golden_queries = [item["question"] for item in golden_data]
        print("\n🧪 RUNNING INTEGRATION TESTS")
        print("=" * 60)
        
        results = []
        answers = {}
        for i, test in enumerate(golden_data, 1):
            if i==10 :
                start = time.perf_counter()
                print(f"\nTest {i}: {test['question']}...")
                print("-" * 40)
                
                result = await self.process_query(test["question"])
                stats = result.get("stats", {})
                
                # Evaluate result
                evaluation = {
                    "test_id": i,
                    "query": test["question"],
                    "has_answer": bool(result.get("answer")),
                    "confidence": result.get("confidence", 0),
                    "execution_success": len(result.get("errors", [])) == 0,
                    "facts_verified": stats.get("verified_facts_count", 0),
                    "contradictions": stats.get("contradictions_found", 0),
                    "path": result.get("execution_path", [])
                }
                end = time.perf_counter()
                # print("*"*50)
                # print(evaluation)
                # print("*"*50)
                results.append(evaluation)
                print(f"total time to execute the query is {end - start}")
                print(f"✅ Answer generated: {bool(result['answer'])}")
                print(f"📊 Confidence: {result['confidence']:.2f}")
                print(f"⚖️  Verification: {stats.get('verified_facts_count')} facts, {stats.get('contradictions_found')} contradictions")
                
                # FIX: If synthesis_agent is missing from path (due to override), append it for display if answer exists
                # path_display = result['execution_path']
                # if result['answer'] and "synthesis_agent" not in path_display:
                #     path_display.append("synthesis_agent (inferred)")
                path_display = list(result['execution_path']) # Create a copy
                if result['answer'] and "synthesis_agent" not in path_display:
                    path_display.append("synthesis_agent (inferred)")
                    
                print(f"🚶 Path: {' → '.join(path_display)}")

                if result.get("errors"):
                    print(f"⚠️  Errors encountered: {len(result['errors'])}")
                    for err in result['errors']:
                        print(f"   - {err.get('agent', 'unknown')}: {err.get('error')}")
                
                if result["answer"]:
                    print("*"*50)
                    print(f"\n📝 answer: {result['answer']}.")
                    print("*"*50)
                    answers[test["question"]] = result['answer']
                else: print("no answer generated")

                if i==10:
                    print("*"*50)
                    print(answers)
                    print("-"*50)    
        return results

    def print_system_status(self):
        """Print detailed system status."""
        print("\n" + "="*60)
        print("🤖 PHASE 1 MULTI-AGENT RAG SYSTEM - COMPLETE")
        print("="*60)
        if self.workflow:
            info = self.workflow.get_graph_info()
            print(f"Status: {info.get('status', 'Unknown')}")
            print(f"Active Agents: {', '.join(info.get('agents', []))}")
        else:
            print("Status: Workflow Not Initialized")
        print("="*60)

async def main():
    """Main entry point for Phase 1 complete system."""
    # Initialize orchestrator
    # Adjust path to where your config is actually stored
    config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
    
    orchestrator = Phase1Orchestrator(config_path)
    
    # Print Status
    orchestrator.print_system_status()
    
    # Run integration tests
    await orchestrator.run_golden_queries_test()


    
    print("\n" + "=" * 60)
    print("✅ SYSTEM EXECUTION COMPLETE")
    print()

if __name__ == "__main__":
    asyncio.run(main())
# """
# Updated Main orchestrator for Phase 1 Multi-Agent RAG System.
# Now with complete agent suite and LangGraph workflow.
# """
# import os
# import asyncio
# import sys
# from pathlib import Path
# from typing import Optional, Dict, Any
# import yaml

# # Add src to path
# src_path = Path(__file__).parent.parent
# sys.path.append(str(src_path))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from graph.workflow import MultiAgentGraph
# from graph.state import get_state_summary

# class Phase1Orchestrator:
#     """Complete Phase 1 orchestrator with all components."""
    
#     def __init__(self, config_path: Optional[Path] = None):
#         self.config = self._load_config(config_path)
#         self.workflow = None
#         self._initialize_workflow()
        
#     def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
#         """Load configuration from YAML file."""
#         default_config = {
#             "llm": {
#                 "default_model": "llama3.2:1b", 
#                 "research_model": "llama3.1:8b",
#                 "synthesis_model": "llama3.1:8b",
#                 "temperature": 0.0
#             },
#             "agents": {
#                 "timeout_seconds": 300,
#                 "max_retries": 2
#             },
#             "workflow": {
#                 "max_total_timeout": 120,
#                 "use_langgraph": True
#             },
#             "retrieval": {
#                 "k": 5,
#                 "hybrid_search": {
#                     "bm25_weight": 0.25,
#                     "semantic_weight": 0.75
#                 }
#             },
#             "api": {
#                 "host": "0.0.0.0",
#                 "port": 8000,
#                 "enable_streaming": True
#             },
#             "evaluation": {
#                 "golden_queries_path": "data/golden_queries.json",
#                 "metrics": ["faithfulness", "answer_relevance", "context_recall"]
#             }
#         }
            
        
#         if config_path and config_path.exists():
#             with open(config_path, 'r') as f:
#                 loaded_config = yaml.safe_load(f)
#                 import copy
#                 def deep_update(target, source):
#                     for key, value in source.items():
#                         if key in target and isinstance(target[key], dict) and isinstance(value, dict):
#                             deep_update(target[key], value)
#                         else:
#                             target[key] = value
                
#                 deep_update(default_config, loaded_config)
        
#         return default_config
    
#     def _initialize_workflow(self):
#         """Initialize the LangGraph workflow."""
#         print("🔄 Initializing LangGraph workflow...")
#         self.workflow = MultiAgentGraph(self.config)
#         print("✅ Workflow initialized")
    
#     async def process_query(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
#         """Process a query through the complete system."""
#         print(f"\n🔍 Processing: {query[:80]}...")
        
#         if not self.workflow:
#             return {"error": "Workflow not initialized"}
        
#         # Process using LangGraph
#         result = await self.workflow.process_query(query, user_id)
        
#         # Safe extraction helpers
#         research_res = result.get("research_results") or {}
#         docs = result.get("retrieved_documents") or []
#         citations = result.get("citations") or []
#         errors = result.get("errors") or []

#         # Create response
#         response = {
#             "query": query,
#             "answer": result.get("final_answer"),
#             "confidence": result.get("confidence_score", 0),
#             "intent": result.get("intent"),
#             "execution_path": result.get("execution_path", []),
#             # Safe access to nested dicts
#             "research_findings": len(research_res.get("findings", [])),
#             "retrieved_documents": len(docs),
#             "citations": len(citations),
#             "errors": errors,  # FIX: Return list, not len(errors)
#             "state_summary": get_state_summary(result)
#         }
        
#         return response
    
#     async def run_golden_queries_test(self):
#         """Run tests with golden queries."""
#         golden_queries = [
#             {
#                 "query": "Compare NVIDIA's H200, AMD's MI300X, and Google's TPU v5e for large language model training in terms of performance (TFLOPS), memory bandwidth, and cost-per-hour on AWS as of Q1 2024.",
#                 "expected_aspects": ["TFLOPS", "memory bandwidth", "cost"]
#             }
#         ]
        
#         print("\n🧪 RUNNING GOLDEN QUERIES TEST")
#         print("=" * 60)
        
#         results = []
#         for i, test in enumerate(golden_queries, 1):
#             print(f"\nTest {i}: {test['query'][:80]}...")
#             print("-" * 40)
            
#             result = await self.process_query(test["query"])
            
#             # Evaluate result
#             evaluation = {
#                 "test_id": i,
#                 "query": test["query"],
#                 "has_answer": bool(result.get("answer")),
#                 "confidence": result.get("confidence", 0),
#                 "execution_success": len(result.get("errors", [])) == 0, # Now this works because 'errors' is a list
#                 "research_used": result.get("research_findings", 0) > 0,
#                 "documents_used": result.get("retrieved_documents", 0) > 0
#             }
            
#             results.append(evaluation)
            
#             print(f"✅ Answer generated: {bool(result['answer'])}")
#             print(f"📊 Confidence: {result['confidence']:.2f}")
#             print(f"🚶 Path: {' → '.join(result['execution_path'])}")
            
#             if result["answer"]:
#                 print(f"📝 Preview: {result['answer'][:150]}...")
        
#         return results
    
#     def print_system_status(self):
#         """Print detailed system status."""
#         print("\n" + "="*60)
#         print("🤖 PHASE 1 MULTI-AGENT RAG SYSTEM - COMPLETE")
#         print("="*60)
#         # (Simplified for brevity)
#         print("System Initialized.")

# async def main():
#     """Main entry point for Phase 1 complete system."""
#     print("🤖 Phase 1 Multi-Agent RAG System - COMPLETE")
#     print("=" * 60)
    
#     # Initialize orchestrator
#     config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
#     orchestrator = Phase1Orchestrator(config_path)
    
#     # Run golden queries test
#     print("\n🧪 RUNNING INTEGRATION TESTS")
#     print("=" * 60)
    
#     await orchestrator.run_golden_queries_test()
    
#     print("\n" + "=" * 60)
#     print("✅ PHASE 1 DEVELOPMENT COMPLETE")

# if __name__ == "__main__":
#     asyncio.run(main())
# """
# Main orchestrator for Phase 1 Multi-Agent RAG System.
# Now with QueryAnalyzer, ResearchAgent, and DocRetrievalAgent.
# """
# import os
# import asyncio
# import sys
# from pathlib import Path
# from typing import Optional, Dict, Any
# import yaml


# # Add src to path
# src_path = Path(__file__).parent.parent
# sys.path.append(str(src_path))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# from graph.state import AgentState, create_initial_state, add_error_to_state
# from agents.query_analyzer import QueryAnalyzerAgent
# from agents.research_agent import ResearchAgent
# from agents.doc_retrieval_agent import DocRetrievalAgent
# # SynthesisAgent will be added in next step
# from agents.synthesis_agent import SynthesisAgent


# class MultiAgentOrchestrator:
#     """Orchestrates the multi-agent workflow."""
    
#     def __init__(self, config_path: Optional[Path] = None):
#         self.config = self._load_config(config_path)
#         self.agents: Dict[str, Any] = {}
#         self._initialize_agents()
#         self.workflow_stage = "initialized"
        
#     def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
#         """Load configuration from YAML file."""
#         default_config = {
#             "llm": {"default_model": "llama3.2:1b", "temperature": 0.0},
#             "agents": {},
#             "workflow": {"max_total_timeout": 120},
#             "retrieval": {"k": 5},
#             "logging": {"level": "INFO"}
#         }
        
#         if config_path and config_path.exists():
#             with open(config_path, 'r') as f:
#                 loaded_config = yaml.safe_load(f)
#                 # Merge with defaults
#                 default_config.update(loaded_config)
        
#         return default_config
    
#     def _initialize_agents(self):
#         """Initialize all agents."""
#         print("🤖 Initializing agents...")
        
#         # Initialize QueryAnalyzerAgent
#         try:
#             self.agents["query_analyzer"] = QueryAnalyzerAgent(self.config)
#             print("  ✅ QueryAnalyzerAgent initialized")
#         except Exception as e:
#             print(f"  ❌ Failed to initialize QueryAnalyzerAgent: {e}")
#             self.agents["query_analyzer"] = None
        
#         # Initialize ResearchAgent
#         try:
#             self.agents["research_agent"] = ResearchAgent(self.config)
#             print("  ✅ ResearchAgent initialized")
#         except Exception as e:
#             print(f"  ❌ Failed to initialize ResearchAgent: {e}")
#             self.agents["research_agent"] = None
        
#         # Initialize DocRetrievalAgent
#         try:
#             self.agents["doc_retrieval_agent"] = DocRetrievalAgent(self.config)
#             print("  ✅ DocRetrievalAgent initialized")
#         except Exception as e:
#             print(f"  ❌ Failed to initialize DocRetrievalAgent: {e}")
#             self.agents["doc_retrieval_agent"] = None
        
#         # Placeholder for other agents
#         try:
#             self.agents["synthesis_agent"] = SynthesisAgent(self.config)
#             print("  ✅ SynthesisAgent initialized")
#         except Exception as e:
#             print(f"  ❌ Failed to initialize SynthesisAgent: {e}")
#             self.agents["synthesis_agent"] = None
#         self.agents["fact_check_agent"] = None
        
#         active_agents = len([a for a in self.agents.values() if a is not None])
#         print(f"  📊 Total active agents: {active_agents}/5")
    
#     async def process_query(self, query: str, user_id: Optional[str] = None) -> AgentState:
#         """
#         Process a query through the multi-agent workflow.
        
#         Args:
#             query: User query
#             user_id: Optional user identifier
            
#         Returns:
#             Final AgentState with answer
#         """
#         print(f"\n🔍 Processing query: {query}")
#         print("=" * 60)
        
#         # Create initial state
#         state = create_initial_state(query, user_id)
#         self.workflow_stage = "started"
        
#         try:
#             # Step 1: Query Analysis
#             print("\n1️⃣  Step 1: Query Analysis")
#             print("-" * 40)
            
#             if self.agents["query_analyzer"]:
#                 state = await self.agents["query_analyzer"].execute(state)
#                 self.workflow_stage = "analysis_complete"
                
#                 # Print analysis summary
#                 if hasattr(self.agents["query_analyzer"], 'get_analysis_summary'):
#                     summary = self.agents["query_analyzer"].get_analysis_summary(state)
#                     print(summary)
#             else:
#                 print("⚠️  QueryAnalyzer not available, using simple fallback")
#                 state["intent"] = "simple"
#                 state["routing_decision"] = "parallel"
            
#             # Step 2: Determine routing
#             print(f"\n2️⃣  Step 2: Routing Decision")
#             print("-" * 40)
            
#             routing = state.get("routing_decision", "parallel")
#             print(f"  Routing: {routing}")
            
#             # Step 3: Execute based on routing
#             if routing == "research_only":
#                 state = await self._execute_research_only(state)
#             elif routing == "retrieval_only":
#                 state = await self._execute_retrieval_only(state)
#             else:  # parallel or sequential
#                 state = await self._execute_parallel_workflow(state, routing)
            
#             self.workflow_stage = "processing_complete"
            
#             # Step 4: Generate final answer
#             print(f"\n4️⃣  Step 4: Generate Final Answer")
#             print("-" * 40)
#             state = await self._generate_final_answer(state)
            
#             print("\n✅ Query processing complete")
#             return state
            
#         except Exception as e:
#             print(f"❌ Error in workflow: {e}")
#             import traceback
#             traceback.print_exc()
            
#             state["errors"].append({
#                 "stage": self.workflow_stage,
#                 "error": str(e),
#                 "traceback": traceback.format_exc()
#             })
            
#             # Fallback to simple answer
#             state["final_answer"] = self._generate_error_answer(state, e)
#             state["confidence_score"] = 0.1
            
#             return state
    
#     async def _execute_research_only(self, state: AgentState) -> AgentState:
#         """Execute research-only workflow."""
#         print("  Route: Research only workflow")
        
#         # Execute research agent if available
#         if self.agents["research_agent"]:
#             state = await self.agents["research_agent"].execute(state)
#         else:
#             print("  ⚠️  ResearchAgent not available")
        
#         return state
    
#     async def _execute_retrieval_only(self, state: AgentState) -> AgentState:
#         """Execute retrieval-only workflow."""
#         print("  Route: Retrieval only workflow")
        
#         # Execute document retrieval agent if available
#         if self.agents["doc_retrieval_agent"]:
#             state = await self.agents["doc_retrieval_agent"].execute(state)
#         else:
#             print("  ⚠️  DocRetrievalAgent not available")
        
#         return state
    
#     async def _execute_parallel_workflow(self, state: AgentState, routing: str) -> AgentState:
#         """Execute parallel workflow (research and retrieval in parallel)."""
#         print(f"  Route: {routing.capitalize()} workflow")
        
#         # Create tasks for parallel execution
#         tasks = []
        
#         # Research task
#         if self.agents["research_agent"]:
#             research_task = self.agents["research_agent"].execute(state.copy())
#             tasks.append(("research", research_task))
        
#         # Retrieval task
#         if self.agents["doc_retrieval_agent"]:
#             retrieval_task = self.agents["doc_retrieval_agent"].execute(state.copy())
#             tasks.append(("retrieval", retrieval_task))
        
#         if not tasks:
#             print("  ⚠️  No agents available for parallel execution")
#             return state
        
#         # Execute in parallel
#         print(f"  🚀 Executing {len(tasks)} agents in parallel...")
#         results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
#         # Merge results
#         for (agent_name, _), result in zip(tasks, results):
#             if isinstance(result, Exception):
#                 print(f"  ❌ {agent_name} failed: {result}")
#                 state = add_error_to_state(state, agent_name, str(result))
#             elif isinstance(result, dict):
#                 # Merge research results
#                 if agent_name == "research" and "research_results" in result:
#                     state["research_results"] = result["research_results"]
                
#                 # Merge retrieved documents
#                 if agent_name == "retrieval" and "retrieved_documents" in result:
#                     state["retrieved_documents"] = result["retrieved_documents"]
                
#                 # Merge any errors
#                 if "errors" in result:
#                     state["errors"].extend(result["errors"])
        
#         return state
    
#     async def _generate_final_answer(self, state: AgentState) -> AgentState:
#         """Generate final answer from available information."""
        
#         # Check what information we have
#         has_research = state.get("research_results") is not None
#         has_documents = bool(state.get("retrieved_documents"))
        
#         # Generate answer based on available information
#         if has_research and has_documents:
#             answer = self._generate_combined_answer(state)
#             confidence = 0.7
#         elif has_research:
#             answer = self._generate_research_based_answer(state)
#             confidence = 0.6
#         elif has_documents:
#             answer = self._generate_document_based_answer(state)
#             confidence = 0.5
#         else:
#             answer = self._generate_fallback_answer(state)
#             confidence = 0.3
        
#         state["final_answer"] = answer
#         state["confidence_score"] = confidence
        
#         print(f"  📝 Generated answer (confidence: {confidence:.1%})")
#         print(f"  Answer preview: {answer[:150]}...")
        
#         return state
    
#     def _generate_combined_answer(self, state: AgentState) -> str:
#         """Generate answer combining research and documents."""
#         query = state.get("query", "")
        
#         research_info = ""
#         if state.get("research_results"):
#             findings = state["research_results"].get("findings", [])
#             if findings:
#                 research_info = "Research findings:\n"
#                 for i, finding in enumerate(findings[:3]):  # First 3 findings
#                     research_info += f"{i+1}. {finding.get('content', '')[:100]}...\n"
        
#         document_info = ""
#         if state.get("retrieved_documents"):
#             docs = state["retrieved_documents"]
#             document_info = "Relevant documents:\n"
#             for i, doc in enumerate(docs[:3]):  # First 3 docs
#                 document_info += f"{i+1}. {doc.page_content[:100]}...\n"
        
#         answer = f"""Based on my analysis of your query: "{query}"

# {research_info}

# {document_info}

# Note: The synthesis agent is not yet implemented. This is a placeholder combining research findings and document retrieval results.

# Once the synthesis agent is implemented (Week 2, Day 10), it will properly combine these sources into a coherent answer with citations."""
        
#         return answer
    
#     def _generate_research_based_answer(self, state: AgentState) -> str:
#         """Generate answer based only on research."""
#         query = state.get("query", "")
        
#         research_info = ""
#         if state.get("research_results"):
#             findings = state["research_results"].get("findings", [])
#             if findings:
#                 research_info = "Based on my research:\n"
#                 for i, finding in enumerate(findings[:3]):
#                     research_info += f"• {finding.get('content', '')[:150]}...\n"
        
#         answer = f"""Regarding your query about "{query}":

# {research_info}

# Note: This answer is based on research findings only. Document retrieval was not performed or returned no results."""
        
#         return answer
    
#     def _generate_document_based_answer(self, state: AgentState) -> str:
#         """Generate answer based only on documents."""
#         query = state.get("query", "")
        
#         document_info = ""
#         if state.get("retrieved_documents"):
#             docs = state["retrieved_documents"]
#             document_info = "Based on the available documents:\n"
#             for i, doc in enumerate(docs[:3]):
#                 document_info += f"• {doc.page_content[:150]}...\n"
        
#         answer = f"""For your query: "{query}":

# {document_info}

# Note: This answer is based on document retrieval only. Research findings were not available."""
        
#         return answer
    
#     def _generate_fallback_answer(self, state: AgentState) -> str:
#         """Generate fallback answer when no information is available."""
#         query = state.get("query", "")
#         intent = state.get("intent", "unknown")
        
#         templates = {
#             "comparison": f"I need to compare multiple items in your query: '{query}'. However, I couldn't retrieve sufficient information to provide a comparison. The multi-agent system is still under development.",
#             "research": f"I attempted to research your query: '{query}', but couldn't gather sufficient information. The research workflow is being improved.",
#             "analysis": f"Your analysis query: '{query}' requires detailed examination. The analytical components are currently being implemented.",
#             "synthesis": f"Your synthesis query requires combining multiple sources. This capability is being developed and will be available soon.",
#             "unknown": f"I've received your query: '{query}'. The multi-agent system is under construction and will be able to answer this soon."
#         }
        
#         return templates.get(intent, templates["unknown"])
    
#     def _generate_error_answer(self, state: AgentState, error: Exception) -> str:
#         """Generate an error message answer."""
#         return f"I encountered an error while processing your query: '{state.get('query', '')}'. Error: {str(error)[:100]}... The system is still under development."
    
#     def get_system_info(self) -> Dict[str, Any]:
#         """Get system information and statistics."""
#         agent_stats = {}
#         for name, agent in self.agents.items():
#             if agent and hasattr(agent, 'get_stats'):
#                 agent_stats[name] = agent.get_stats()
#             elif agent:
#                 agent_stats[name] = {"status": "initialized"}
#             else:
#                 agent_stats[name] = {"status": "not_implemented"}
        
#         return {
#             "system": "Phase 1 Multi-Agent RAG",
#             "status": "development",
#             "phase": 1,
#             "workflow_stage": self.workflow_stage,
#             "agents": agent_stats,
#             "config": {
#                 "llm_model": self.config.get("llm", {}).get("default_model"),
#                 "workflow_timeout": self.config.get("workflow", {}).get("max_total_timeout"),
#                 "retrieval_k": self.config.get("retrieval", {}).get("k", 5)
#             }
#         }
    
#     def print_detailed_status(self):
#         """Print detailed system status."""
#         info = self.get_system_info()
        
#         print("\n" + "="*60)
#         print("🤖 MULTI-AGENT SYSTEM STATUS")
#         print("="*60)
        
#         print(f"System: {info['system']} (Phase {info['phase']})")
#         print(f"Status: {info['status'].upper()}")
#         print(f"Stage: {info['workflow_stage']}")
        
#         print("\n🔧 AGENTS STATUS:")
#         for agent_name, stats in info["agents"].items():
#             status = stats.get("status", "unknown")
#             if "total_executions" in stats:
#                 executions = stats["total_executions"]
#                 success_rate = stats.get("success_rate", 0) * 100
#                 latency = stats.get("average_latency", 0)
#                 print(f"  • {agent_name}: ✅ Active (executions: {executions}, success: {success_rate:.1f}%, latency: {latency:.2f}s)")
#             elif status == "initialized":
#                 print(f"  • {agent_name}: ⚡ Initialized")
#             elif status == "not_implemented":
#                 print(f"  • {agent_name}: ⏳ Not implemented")
#             else:
#                 print(f"  • {agent_name}: ❓ {status}")
        
#         print("\n⚙️  CONFIGURATION:")
#         for key, value in info["config"].items():
#             print(f"  • {key}: {value}")
        
#         print("\n📊 NEXT AGENT TO IMPLEMENT: SynthesisAgent")
#         print("="*60)


# async def main():
#     """Main entry point for testing."""
#     print("🤖 Phase 1 Multi-Agent RAG Orchestrator")
#     print("=" * 60)
    
#     # Initialize orchestrator
#     config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
#     orchestrator = MultiAgentOrchestrator(config_path)
    
#     # Display system status
#     orchestrator.print_detailed_status()
    
#     # Test with sample queries
#     test_queries = [
#         "Compare NVIDIA's H200 and AMD's MI300X for AI workloads",
#         "What are the main technical approaches to in-context learning improvement?",
#         "How has Salesforce's AI strategy evolved since 2023?",
#         "Chart the R&D spending of Apple, Google, and Microsoft"
#     ]
    
#     print("\n🧪 RUNNING TEST QUERIES")
#     print("=" * 60)
    
#     for i, query in enumerate(test_queries, 1):
#         print(f"\nTest {i}: {query}")
#         print("-" * 40)
        
#         result = await orchestrator.process_query(query)
        
#         print(f"\n📝 RESULT:")
#         print(f"Answer: {result.get('final_answer', 'No answer generated')[:200]}...")
#         print(f"Confidence: {result.get('confidence_score', 'N/A')}")
#         print(f"Intent: {result.get('intent', 'N/A')}")
#         print(f"Tasks: {len(result.get('decomposed_tasks', []))}")
        
#         # Show research results if available
#         if result.get("research_results"):
#             findings = result["research_results"].get("findings", [])
#             print(f"Research Findings: {len(findings)}")
        
#         # Show retrieved documents if available
#         if result.get("retrieved_documents"):
#             print(f"Retrieved Documents: {len(result['retrieved_documents'])}")
        
#         if result.get("errors"):
#             print(f"Errors: {len(result['errors'])} occurred")
        
#         print("-" * 40)
    
#     # Final status
#     print("\n" + "=" * 60)
#     print("✅ Phase 1 orchestrator test complete")
#     orchestrator.print_detailed_status()
    
#     print("\n📋 WEEK 1 PROGRESS:")
#     print("✅ QueryAnalyzerAgent - Complete")
#     print("✅ ResearchAgent - Complete (mock implementation)")
#     print("✅ DocRetrievalAgent - Complete (with hybrid search)")
#     print("⏳ SynthesisAgent - Next to implement")
#     print("⏳ FactCheckAgent - To implement")
#     print("⏳ LangGraph workflow - Week 3")
#     print("=" * 60)


# if __name__ == "__main__":
#     asyncio.run(main())
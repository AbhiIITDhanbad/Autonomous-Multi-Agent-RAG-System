"""
LangGraph Workflow for Multi-Agent RAG System.
Implements the orchestration using LangGraph with conditional routing.
"""
import sys
import os
import asyncio
from typing import Dict, Any, Literal, Optional
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import AgentState
from agents.query_analyzer import QueryAnalyzerAgent
from agents.research_agent import ResearchAgent
from agents.doc_retrieval_agent import DocRetrievalAgent
from agents.synthesis_agent import SynthesisAgent
from agents.fact_verification_agent import FactVerificationAgent
 

class MultiAgentGraph:
    """
    LangGraph implementation of the multi-agent workflow.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LangGraph workflow."""
        self.config = config or {}
        self.graph = None
        self.checkpointer = None
        self.agents = {}
        
        self._initialize_agents()
        self._build_graph()
        
    def _initialize_agents(self):
        """Initialize all agents."""
        print("🤖 Initializing agents for LangGraph...")
        
        # Initialize QueryAnalyzerAgent
        self.agents["query_analyzer"] = QueryAnalyzerAgent(self.config)
        
        # Initialize ResearchAgent
        self.agents["research_agent"] = ResearchAgent(self.config)
        
        # Initialize DocRetrievalAgent
        self.agents["doc_retrieval_agent"] = DocRetrievalAgent(self.config)

        self.agents["fact_verification_agent"] = FactVerificationAgent(self.config)
        
        # Initialize SynthesisAgent
        self.agents["synthesis_agent"] = SynthesisAgent(self.config)
        
        print(f"✅ All {len(self.agents)} agents initialized")
    
    def _build_graph(self):
            """Build the LangGraph state machine."""
            print("🔄 Building LangGraph workflow...")
            
            # Create the graph
            workflow = StateGraph(AgentState)
            
            # Add nodes (agents)
            workflow.add_node("query_analyzer", self._run_query_analyzer)
            workflow.add_node("research_agent", self._run_research_agent)
            workflow.add_node("doc_retrieval_agent", self._run_doc_retrieval_agent)
            workflow.add_node("fact_verification_agent", self._run_fact_verification_agent)
            workflow.add_node("synthesis_agent", self._run_synthesis_agent)
            
            # Set entry point
            workflow.set_entry_point("query_analyzer")
            
            # Add conditional edges based on routing decision
            workflow.add_conditional_edges(
                "query_analyzer",
                self._route_after_analysis,
                {
                    "research_only": "research_agent",
                    "retrieval_only": "doc_retrieval_agent",
                    "parallel": "parallel_branch",
                    # FIX: Map sequential to parallel_branch so both agents run (Phase 1 simplification)
                    "sequential": "parallel_branch"
                }
            )
            
            # Add parallel branch
            workflow.add_node("parallel_branch", self._run_parallel_branch)
            
            # Edges from research agent
            workflow.add_edge("research_agent", "fact_verification_agent")
            
            # Edges from retrieval agent
            workflow.add_edge("doc_retrieval_agent", "fact_verification_agent")
            
            # Edge from parallel branch to synthesis
            workflow.add_edge("parallel_branch", "fact_verification_agent")
            workflow.add_edge("fact_verification_agent", "synthesis_agent")
            
            # Set end point
            workflow.add_edge("synthesis_agent", END)
            
            # Add memory for conversation
            memory = InMemorySaver()
            
            # Compile the graph
            # FIX: Removed interrupts so the graph runs start-to-finish automatically
            self.graph = workflow.compile(checkpointer=memory)
            
            print(f"print {self.graph}","\n")
    
    async def _run_query_analyzer(self, state: AgentState) -> AgentState:
        """Run query analyzer agent."""
        print(f"\n🔍 [Graph] Running Query Analyzer...")
        result = await self.agents["query_analyzer"].execute(state)
        print(f"✅ [Graph] Query analysis complete")
        return result
    
    async def _run_research_agent(self, state: AgentState) -> AgentState:
        """Run research agent."""
        print(f"\n🌐 [Graph] Running Research Agent...")
        result = await self.agents["research_agent"].execute(state)
        print(f"✅ [Graph] Research complete: {len(result.get('research_results', {}).get('findings', []))} findings")
        return result
    
    async def _run_doc_retrieval_agent(self, state: AgentState) -> AgentState:
        """Run document retrieval agent."""
        print(f"\n📚 [Graph] Running Document Retrieval Agent...")
        result = await self.agents["doc_retrieval_agent"].execute(state)
        docs_count = len(result.get("retrieved_documents", [])) if result.get("retrieved_documents") else 0
        print(f"✅ [Graph] Document retrieval complete: {docs_count} documents")
        return result
    
    async def _run_fact_verification_agent(self, state: AgentState) -> AgentState:
        """Run fact verification agent."""
        print(f"\n ⚖️ [Graph] Running Fact Verification Agent...")
        result = await self.agents["fact_verification_agent"].execute(state)
        
        # Log quick stats
        contradictions = result.get("contradiction_report", [])
        verified = result.get("verified_facts", [])
        print(f"✅ [Graph] Verification complete: {len(verified)} facts checked, {len(contradictions)} contradictions found")
        
        return result
    
    async def _run_synthesis_agent(self, state: AgentState) -> AgentState:
        """Run synthesis agent."""
        print(f"\n🧠 [Graph] Running Synthesis Agent...")
        result = await self.agents["synthesis_agent"].execute(state)
        confidence = result.get("confidence_score", 0)
        print(f"✅ [Graph] Synthesis complete: Confidence={confidence}")
        return result
    
    async def _run_parallel_branch(self, state: AgentState) -> AgentState:
        """Run research and retrieval in parallel."""
        print(f"\n⚡ [Graph] Running Parallel Branch...")
        
        import asyncio
        
        # Create copies of state for parallel execution
        # Note: dict.copy() is shallow. Mutable items (lists/dicts) inside are shared.
        state_copy1 = state.copy()
        state_copy2 = state.copy()
        
        # Run agents in parallel
        results = await asyncio.gather(
            self.agents["research_agent"].execute(state_copy1),
            self.agents["doc_retrieval_agent"].execute(state_copy2),
            return_exceptions=True
        )
        
        # Merge results
        for result in results:
            if not isinstance(result, Exception):
                # FIX: Only update if key exists AND has value.
                # Otherwise, one agent's None value overwrites the other agent's valid data.
                if result.get("research_results"):
                    state["research_results"] = result["research_results"]
                
                if result.get("retrieved_documents"):
                    state["retrieved_documents"] = result["retrieved_documents"]
                
                # FIX: Do NOT manually merge 'errors' here to avoid duplication
                # (Agents already appended to the shared error list)
        
        print(f"✅ [Graph] Parallel branch complete")
        return state
    
    def _route_after_analysis(self, state: AgentState) -> Literal["research_only", "retrieval_only", "parallel", "sequential"]:
        """Determine routing after query analysis."""
        routing = state.get("routing_decision", "parallel")
        
        # Override based on analysis if needed
        analysis = state.get("query_analysis", {}) or {} # Handle None
        metadata = analysis.get("metadata", {})
        
        # Check if external data is required
        requires_external = metadata.get("requires_external_data", True)
        
        if not requires_external:
            return "retrieval_only"
        
        print(f"🔄 [Graph] Routing decision: {routing}")
        return routing
    
    async def process_query(self, query: str, user_id: Optional[str] = None) -> AgentState:
        """
        Process a query through the LangGraph workflow.
        """
        print(f"\n🚀 Processing query via LangGraph: {query[:80]}...")
        print("=" * 60)
        
        # Create initial state
        initial_state = {
            "query": query,
            "user_id": user_id,
            "query_analysis": None,
            "decomposed_tasks": None,
            "intent": None,
            "routing_decision": None,
            "research_results": None,
            "retrieved_documents": None,
            "fact_check_results": None,
            "intermediate_answers": None,
            "final_answer": None,
            "confidence_score": None,
            "citations": None,
            "agent_timestamps": {},
            "retrieval_sources": [],
            "errors": [],
            "execution_path": [],
            "latency_per_agent": {},
            "token_usage": {},
            "retrieval_metadata": None,
            "verified_facts": None,
            "contradiction_report": None,
            "verification_confidence": None
        }
        
        try:
            # Execute the graph
            config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}
            result = await self.graph.ainvoke(initial_state, config)
            
            print(f"\n✅ Graph execution complete")
            print(f"📊 Final confidence: {result.get('confidence_score', 0)}")
            print(f"🚶 Execution path: {' → '.join(result.get('execution_path', []))}")
            
            return result
            
        except Exception as e:
            print(f"❌ Graph execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            initial_state["errors"].append({
                "stage": "graph_execution",
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            initial_state["final_answer"] = f"Graph execution failed: {str(e)[:100]}..."
            initial_state["confidence_score"] = 0.0
            
            return initial_state
    
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph structure."""
        if not self.graph:
            return {"status": "not_compiled"}
        
        nodes = list(self.graph.nodes)
        
        return {
            "status": "compiled",
            "nodes": nodes,
            "agents": list(self.agents.keys()),
            "has_checkpointer": self.graph.checkpointer is not None
        }


# Test function
def test_langgraph_workflow():
    """Test the LangGraph workflow."""
    print("🧪 Testing LangGraph Workflow")
    print("=" * 60)
    
    # Initialize workflow
    workflow = MultiAgentGraph()
    
    # Get graph info
    info = workflow.get_graph_info()
    print(f"📊 Graph Info: {info['status']}")
    print(f"   Nodes: {', '.join(info['nodes'])}")
    print(f"   Agents: {', '.join(info['agents'])}")
    
    # Test queries
    test_queries = [
        "Find the technical specification document for Tesla's Dojo D1 chip, identify its theoretical peak FP32 performance, and verify if this claim was benchmarked in an independent analysis by MLPerf.",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query[:80]}...")
        print("-"*60)
        
        result = workflow.process_query(query)
        
        print(f"\n📝 RESULT:")
        result = asyncio.run(result)
        # FIX: Safe access to final_answer to prevent TypeError
        final_answer = result.get('final_answer') or "No answer generated"
        print(f"Answer: {final_answer}...")
        
        print(f"Confidence: {result.get('confidence_score', 'N/A')}")
        print(f"Intent: {result.get('intent', 'N/A')}")
        
        # Show agent execution
        if result.get("execution_path"):
            print(f"Execution: {' → '.join(result['execution_path'])}")
        
        # Show errors if any
        if result.get("errors"):
            print(f"Errors: {len(result['errors'])}")
            for e in result['errors']:
                print(f"  - {e.get('agent', 'System')}: {e.get('error')}")
    
    print(f"\n{'='*60}")
    print("✅ LangGraph workflow test complete")


if __name__ == "__main__":
    test_langgraph_workflow()


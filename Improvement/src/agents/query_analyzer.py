"""
Query Analyzer Agent: Analyzes and decomposes complex queries.
First agent in the multi-agent workflow.
"""
"""
Query Analyzer Agent: Analyzes and decomposes complex queries.
First agent in the multi-agent workflow.
"""
import sys
import os
import json
import asyncio
from typing import Dict, Any, List,  Literal
from datetime import datetime
import json
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser # Added import
from pydantic import BaseModel, Field
from agents.base_agent import BaseAgent
from graph.state import AgentState, QueryAnalysisResult, add_error_to_state


class Task(BaseModel):
    task: str = Field(description="Specific actionable task description")
    # STRICT VALIDATION: LLM will error if it uses an undefined agent
    agent: Literal["research_agent", "doc_retrieval_agent"] 
    priority: int = Field(description="Priority level, e.g., 1")
    expected_output: str = Field(description="What this task should produce")

class Metadata(BaseModel):
    complexity: Literal["simple", "moderate", "complex"]
    requires_external_data: bool
    time_sensitivity: Literal["high", "medium", "low"]
    expected_output_format: Literal["table", "paragraph", "list", "json"]

class QueryAnalysis(BaseModel):
    """Schema for decomposing business queries."""
    intent: Literal["research", "comparison", "analysis", "synthesis"]
    tasks: List[Task]
    metadata: Metadata
    routing_decision: Literal["research_only", "retrieval_only", "parallel"]

class QueryAnalyzerAgent(BaseAgent):
    """
    Analyzes user queries to determine intent and decompose into sub-tasks.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize QueryAnalyzerAgent."""
        super().__init__(
            name="query_analyzer",
            description="Analyzes and decomposes complex queries",
            timeout_seconds=300,
            max_retries=2
        )
        
        self.config = config or {}
        self._initialize_llm()
        self._initialize_prompt()
        self._initialize_chain()
        
    # def _initialize_llm(self):
    #     """Initialize the LLM for this agent."""
    #     llm_config = self.config.get("llm", {})
    #     model = llm_config.get("default_model", "llama3.1:8b")
    #     temperature = llm_config.get("temperature", 0.0)
        
    #     self.llm = OllamaLLM(
    #         model=model,
    #         temperature=temperature,
    #         format="json"  # FIX: Explicitly request JSON mode from Ollama
    #     )
    #     print(f"✅ QueryAnalyzer LLM initialized: {model}")
    def _initialize_llm(self):
            """Initialize the LLM for this agent."""
            llm_config = self.config.get("llm", {})
            model = llm_config.get("query_analyzer_model", "llama3.1:8b")
            
            # --- FIX: Handle dictionary configuration for temperature ---
            temp_config = llm_config.get("temperature", 0.0)
            if isinstance(temp_config, dict):
                # Query Analyzer should use low temperature (analysis)
                temperature = temp_config.get("analysis", temp_config.get("default", 0.0))
            else:
                temperature = float(temp_config)
            # ------------------------------------------------------------
            
            self.llm = OllamaLLM(
                model=model,
                temperature=temperature,
                format="json"
            )
            print(f"✅ QueryAnalyzer LLM initialized: {model}")
    
#     def _initialize_prompt(self):
#         """Initialize the prompt template."""
#         # FIX: Define Data Sources explicitly so the LLM knows what is 'Internal' vs 'External'
#         self.prompt_template = ChatPromptTemplate.from_messages([
#             ("system", """You are a competitive intelligence analyst expert at decomposing complex business queries.

# ### AVAILABLE DATA SOURCES:
# 1. **Internal Knowledge Base (agent: doc_retrieval_agent)**
#    *Use this for specific technical specs and historical reports already indexed.*
#    - **Corporates:** Quarterly earnings calls for **Microsoft, Google, and Salesforce** (Q4 2022 - Q1 2024).
#    - **Hardware:** Technical specifications for **NVIDIA H200, AMD MI300X, and Google TPU v5e**.
#    - **Research:** Academic papers on **In-Context Learning** (OpenAI, Google DeepMind, Anthropic) post-Jan 2023.

# 2. **Web Research (agent: research_agent)**
#    *Use this for live data, news, or topics NOT listed above.*
#    - Recent news (last 7 days), real-time stock prices, or general market trends.
#    - Information on companies NOT in the knowledge base (e.g., Apple, Meta, Amazon).
#    - General technology concepts or competitor comparisons not covered by internal specs.

# ### ROUTING LOGIC:
# - **retrieval_only**: Query asks ONLY about the specific internal topics listed above (e.g., "What is the memory bandwidth of H200?").
# - **research_only**: Query asks about topics NOT in the internal base (e.g., "What is Apple's current stock price?").
# - **parallel**: Query requires BOTH internal specs and external market context (e.g., "Compare NVIDIA H200 [Internal] with the new Blackwell chip rumors [External]").

# ### OUTPUT SCHEMA (Strict JSON):
# {{
#     "intent": "research|comparison|analysis|synthesis",
#     "tasks": [
#         {{
#             "task": "specific actionable task description",
#             "agent": "research_agent|doc_retrieval_agent",
#             "priority": 1,
#             "expected_output": "what this task should produce"
#         }}
#     ],
#     "metadata": {{
#         "complexity": "simple|moderate|complex",
#         "requires_external_data": true,
#         "time_sensitivity": "high|medium|low",
#         "expected_output_format": "table|paragraph|list|json"
#     }},
#     "routing_decision": "research_only|retrieval_only|parallel"
# }}

# Do not add any text outside the JSON object.
# """),
#             ("human", "Analyze this query: {query}")
#         ])
#         self.prompt_template = ChatPromptTemplate.from_messages([
#             ("system", """You are a competitive intelligence analyst expert at decomposing complex business queries.

# ### AVAILABLE DATA SOURCES:
# 1. **Internal Knowledge Base (agent: doc_retrieval_agent)**
#    *Use this for specific technical specs and historical reports already indexed.*
#    - **Corporates:** Quarterly earnings calls for **Microsoft, Google, and Salesforce** (Q4 2022 - Q1 2024).
#    - **Hardware:** Technical specifications for **NVIDIA H200, AMD MI300X, and Google TPU v5e**.
#    - **Research:** Academic papers on **In-Context Learning** (OpenAI, Google DeepMind, Anthropic) post-Jan 2023.

# 2. **Web Research (agent: research_agent)**
#    *Use this for live data, news, or topics NOT listed above.*
#    - Recent news (last 7 days), real-time stock prices, or general market trends.
#    - Information on companies NOT in the knowledge base (e.g., Apple, Meta, Amazon).
#    - General technology concepts or competitor comparisons not covered by internal specs.

# ### ROUTING LOGIC:
# - **retrieval_only**: Query asks ONLY about the specific internal topics listed above.
# - **research_only**: Query asks about topics NOT in the internal base.
# - **parallel**: Query requires BOTH internal specs and external market context.

# ### OUTPUT FORMATTING:
# You must strictly follow this schema:
# {format_instructions}
# """),
#             ("human", "Analyze this query: {query}")
#         ])
    def _initialize_prompt(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a competitive intelligence analyst expert at decomposing complex business queries.

### AVAILABLE DATA SOURCES:
1. **Internal Knowledge Base (agent: doc_retrieval_agent)**
   - Quarterly earnings calls: Microsoft, Google, Salesforce (Q4 2022 - Q1 2024).
   - Hardware Specs: NVIDIA H200, AMD MI300X, Google TPU v5e.
   - Research: Academic papers on In-Context Learning post-Jan 2023.

2. **Web Research (agent: research_agent)**
   - Recent news, live data, companies NOT in the KB (Apple, Meta, Amazon).
   - General tech concepts.
   - Anything outside to **Internal Knowledge Base**
### ROUTING LOGIC:
- **retrieval_only**: Internal topics only.
- **research_only**: External topics only.
- **parallel**: Both.

### OUTPUT INSTRUCTIONS:
You must output a single valid JSON object that strictly adheres to the schema below. 
**IMPORTANT:** Do NOT return the schema definition/rules. Return only the actual JSON data containing the 'intent', 'tasks', 'metadata', and 'routing_decision' fields based on the user's query.

Schema format:
{format_instructions}
"""),
    ("human", "Analyze this query: {query}")
])
        
    
    # def _initialize_chain(self):
    #     """Initialize the LLM chain."""
    #     # FIX: Use a simple LCEL chain: Prompt -> LLM -> JSON Parser
    #     # This is much more reliable than 'create_agent' for pure structured output
    #     parser = JsonOutputParser()
    #     self.chain = self.prompt_template | self.llm | parser
    
    def _initialize_chain(self):
        """Initialize the LLM chain with Pydantic validation."""
        # 1. Setup Pydantic Parser
        self.parser = PydanticOutputParser(pydantic_object=QueryAnalysis)

        # 2. Build Chain: Prompt -> LLM -> Pydantic Validation -> Dict Conversion
        # The RunnableLambda converts the strict Pydantic object back to a standard Python Dict
        self.chain = (
            self.prompt_template | self.llm | self.parser | RunnableLambda(lambda x: x.model_dump())
        )


    # async def _execute_impl(self, state: AgentState) -> AgentState:
    #     print(f"🔍 QueryAnalyzer analyzing: {state['query'][:50]}...")
        
    #     try:
    #         # Use ainvoke for async execution
    #         result = await self.chain.ainvoke({"query": state["query"]})
            
    #         # 1. Validate if we got a dictionary back
    #         if not result or not isinstance(result, dict):
    #             print("⚠️ LLM returned invalid or empty JSON")
    #             return await self._provide_fallback_analysis(state, "Empty LLM result")
    #         else:
    #             print("SUCCESS")
    #             print(result)
    #             print("-"*50)
    #         # 2. Explicitly map keys to the state (matching test_query_analyzer expectation)
    #         state["query_analysis"] = result
    #         state["decomposed_tasks"] = result.get("tasks", [])
    #         state["intent"] = result.get("intent", "research")
    #         state["routing_decision"] = result.get("routing_decision", "parallel")
            
    #         print(f"✅ Success: Intent={state['intent']}, Tasks={len(state['decomposed_tasks'])}")
    #         return state
            
    #     except Exception as e:
    #         print(f"❌ Analysis error: {str(e)}")
    #         return await self._provide_fallback_analysis(state, str(e))
    async def _execute_impl(self, state: AgentState) -> AgentState:
        print(f"🔍 QueryAnalyzer analyzing: {state['query'][:50]}...")
        
        try:
            # Pass format_instructions to the chain
            result = await self.chain.ainvoke({
                "query": state["query"],
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Validation is now handled by Pydantic inside the chain.
            # If we reach here, 'result' is guaranteed to be a valid Dictionary matching the schema.
            
            print("SUCCESS")
            print(result)
            print("-"*50)

            # Map keys to state
            state["query_analysis"] = result
            state["decomposed_tasks"] = result.get("tasks", [])
            state["intent"] = result.get("intent", "research")
            state["routing_decision"] = result.get("routing_decision", "parallel")
            
            print(f"✅ Success: Intent={state['intent']}, Tasks={len(state['decomposed_tasks'])}")
            return state
            
        except Exception as e:
            print(f"❌ Analysis error (Validation Failed): {str(e)}")
            # Pydantic validation errors will be caught here
            return await self._provide_fallback_analysis(state, str(e))

    def _validate_analysis_result(self, result: Dict) -> bool:
        """Validate the structure of analysis result."""
        required_keys = ["intent", "tasks", "metadata"]
        
        if not isinstance(result, dict):
            print(f"❌ Result is not a dictionary: {type(result)}")
            return False

        for key in required_keys:
            if key not in result:
                print(f"❌ Missing key in analysis: {key}")
                return False
        
        if not isinstance(result["tasks"], list):
            print("❌ Tasks must be a list")
            return False
            
        return True
    
    async def _provide_fallback_analysis(self, state: AgentState, error: str) -> AgentState:
        """Provide a fallback analysis when the main analysis fails."""
        print("🔄 Using fallback query analysis...")
        
        # Simple keyword-based fallback analysis
        query = state["query"].lower()
        
        if any(word in query for word in ["compare", "vs", "versus", "difference"]):
            intent = "comparison"
        elif any(word in query for word in ["research", "find", "search", "look up"]):
            intent = "research"
        else:
            intent = "analysis"
        
        tasks = [{
            "task": f"Research information about: {state['query']}",
            "agent": "research",
            "priority": 1,
            "expected_output": "Key findings summary"
        }]
        
        fallback_result = {
            "intent": intent,
            "tasks": tasks,
            "metadata": {
                "complexity": "moderate",
                "requires_external_data": True,
                "time_sensitivity": "medium",
                "expected_output_format": "paragraph"
            },
            "routing_decision": "sequential"
        }
        
        state["query_analysis"] = fallback_result
        state["decomposed_tasks"] = tasks
        state["intent"] = intent
        state["routing_decision"] = "sequential"
        
        return state

# Test function for the agent
async def test_query_analyzer():
    """Test the QueryAnalyzerAgent."""
    print("🧪 Testing QueryAnalyzerAgent...")
    
    # Initialize agent
    agent = QueryAnalyzerAgent()
    
    # Test queries
    test_queries = [
        "Compare NVIDIA H200 and AMD MI300X for AI workloads",
        "Chart the quarterly mentions of RAG in earnings calls of Microsoft, Google, and Salesforce",
        "How has the focus of Meta's AI research publications shifted between computer vision and NLP?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query[:80]}...")
        print("-"*60)
        
        state = {
            "query": query,
            "query_analysis": None,
            "decomposed_tasks": None,
            "intent": None,
            "routing_decision": None,
            "errors": [],
            "agent_timestamps": {},   # Added missing required field
            "execution_path": []      # Added missing required field
        }
        
        try:
            result = await agent.execute(state)
            
            if result["query_analysis"]:
                analysis = result["query_analysis"]
                print(f"✅ Intent: {analysis['intent']}")
                print(f"✅ Tasks: {len(analysis['tasks'])}")
                
                if analysis['tasks']:
                    print(f"📋 Sample task: {analysis['tasks'][0]['task']}")
            else:
                print("❌ No analysis generated")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    asyncio.run(test_query_analyzer())

"""
Synthesis Agent: Combines verified facts, research, and documents into final answer.
Fourth core agent in the multi-agent workflow.
"""
import sys
import os
import asyncio
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , JsonOutputParser
from langchain_core.runnables import RunnableLambda
from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state

# --- Data Models ---

class SynthesisResult(BaseModel):
    answer: str = Field(description="The comprehensive, detailed synthesized answer")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used")
    contradictions: List[str] = Field(default_factory=list, description="Contradictions addressed")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")

class SynthesisAgent(BaseAgent):
    """
    Synthesizes information into final answer, prioritizing verified facts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="synthesis_agent",
            description="Synthesizes information into final answer",
            timeout_seconds=300,
            max_retries=2
        )
        self.config = config or {}
        self._initialize_llm()
        self._initialize_prompts()
        
    def _initialize_llm(self):
        llm_config = self.config.get("llm", {})
        # IMPROVEMENT: Default to a stronger model for synthesis
        model = llm_config.get("synthesis_model", "llama3.1:8b")
        
        temp = llm_config.get("temperature", 0.1)
        if isinstance(temp, dict): temp = temp.get("default", 0.1)
        
        self.llm = OllamaLLM(model=model, temperature=float(temp), format="json")
        print(f"✅ SynthesisAgent LLM initialized: {model}")
    
    def _initialize_prompts(self):
        # IMPROVEMENT: Detailed, intent-aware prompt
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Lead Technical Analyst.
Synthesize a comprehensive answer based on the provided VERIFIED FACTS and SOURCES.

### INPUT CONTEXT:
- **User Query**: {query}
- **Intent**: {query_intent}
- **Guidance**: {intent_guidance}

### SOURCE DATA:
1. **VERIFIED FACTS** (True Data):
{verified_facts_summary}

2. **Contradictions**:
{contradiction_report}

3. **Raw Research**:
{raw_data_summary}

### OUTPUT FORMAT:
Return a single JSON object. Do not include markdown formatting or explanations.
Follow this exact structure:
{{
    "answer": "Detailed answer text here...",
    "confidence": 0.9,
    "citations": [
        {{"source_id": "doc_1", "content_referenced": "...", "credibility": 0.9}}
    ],
    "contradictions": ["List any conflicts found"],
    "assumptions": ["List any assumptions made"]
}}

### REQUIREMENTS:
1. **Detail**: Provide a detailed, multi-paragraph analysis.
2. **Data**: Include specific numbers/metrics from the Verified Facts.
3. **Accuracy**: If facts conflict, acknowledge the discrepancy.
### CRITICAL RULES:
1. If the retrieved documents do not contain the specific metric, you MUST state "Information not found in context."
2. DO NOT estimate or hallucinate future projections unless explicitly stated in the source text.             
             
             """),


            ("human", "Synthesize the final answer now.")
        ])

    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Diagnostic override to catch errors that BaseAgent might swallow.
        """
        try:
            return await self._execute_impl(state)
        except Exception as e:
            print(f"❌ CRITICAL ERROR in SynthesisAgent: {e}")
            traceback.print_exc()
            state["final_answer"] = f"Agent Crashed: {str(e)}"
            state["confidence_score"] = 0.0
            state["errors"].append({"agent": self.name, "error": str(e)})
            return state

    async def _execute_impl(self, state: AgentState) -> AgentState:
        print(f"🧠 SynthesisAgent synthesizing final answer...")
        
        try:
            inputs = self._prepare_synthesis_inputs(state)
            
            if not self._has_sufficient_information(inputs):
                print("⚠️  Insufficient information")
                return await self._provide_minimal_answer(state, inputs)
            
            # Setup Parser
            self.parser = PydanticOutputParser(pydantic_object=SynthesisResult)
            
            # Inject format instructions and intent guidance
            # inputs["format_instructions"] = self.parser.get_format_instructions()
            inputs["intent_guidance"] = self._get_intent_guidance(inputs["query_intent"])
            
            # Run Chain
            print(f"⏳ Invoking LLM (Intent: {inputs['query_intent']})...")
            chain = (
                self.synthesis_prompt | self.llm | JsonOutputParser()
            )
            
            result = await chain.ainvoke(inputs)
            
            # Process Result
            synthesis_result = SynthesisResult(**result)
            
            # Confidence Calculation
            up_conf = state.get("verification_confidence") or 0.5
            final_conf = (up_conf * 0.4) + (synthesis_result.confidence * 0.6)
            
            # Update State
            state["final_answer"] = synthesis_result.answer
            state["confidence_score"] = float(f"{final_conf:.2f}")
            state["citations"] = synthesis_result.citations
            
            if state.get("intermediate_answers") is None:
                state["intermediate_answers"] = {}
            state["intermediate_answers"]["synthesis"] = synthesis_result.model_dump()
            
            print(f"✅ Synthesis complete. Confidence: {state['confidence_score']}")
            return state
            
        except Exception as e:
            print(f"❌ Logic Error in _execute_impl: {e}")
            traceback.print_exc()
            return await self._provide_fallback_answer(state)
    
    def _get_intent_guidance(self, intent: str) -> str:
        """Provide specific prompting instructions based on intent."""
        guidance = {
            "comparison": "Create a side-by-side analysis. Highlight strengths and weaknesses of each entity. Use specific metrics to contrast.",
            "technical_spec": "List specifications clearly. Group technical details by category (e.g., Memory, Performance, Power).",
            "research": "Provide a narrative summary of findings. Connect disparate facts into a cohesive story.",
            "summary": "Focus on the main points. Be concise but complete.",
            "unknown": "Provide a general, comprehensive answer covering all available facts."
        }
        return guidance.get(intent, guidance["unknown"])

    def _prepare_synthesis_inputs(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        
        # Extract Analysis Context
        analysis = state.get("query_analysis") or {}
        intent = analysis.get("intent", "unknown")
        # Handle cases where analysis might be None or missing metadata
        meta = analysis.get("metadata") or {}
        complexity = meta.get("complexity", "moderate")
        
        # 1. Facts
        facts = state.get("verified_facts") or []
        fact_summary = ""
        if facts:
            fact_summary = f"Verified Facts ({len(facts)} items):\n"
            for i, f in enumerate(facts):
                status = f.get("verification_status", "unknown")
                content = f.get("content", "")
                fact_summary += f"{i+1}. [{status}] {content}\n"
        else:
            fact_summary = "No verified facts available."
        
        # 2. Contradictions
        reports = state.get("contradiction_report") or []
        con_summary = "None" if not reports else str(reports)
        
        # 3. Raw Data (Backup)
        findings = (state.get("research_results") or {}).get("findings", [])
        docs = state.get("retrieved_documents") or []
        
        raw_summary = ""
        if findings:
            raw_summary += f"\nResearch Findings ({len(findings)}):\n"
            for f in findings[:3]:
                raw_summary += f"- {f.get('content', '')[:150]}...\n"
        
        if docs:
            raw_summary += f"\nDocuments ({len(docs)}):\n"
            for d in docs[:3]:
                raw_summary += f"- {d.page_content[:150]}...\n"
        
        if not raw_summary:
            raw_summary = "No raw data available."
        
        return {
            "query": query,
            "query_intent": intent,
            "query_complexity": complexity,
            "verified_facts_summary": fact_summary,
            "contradiction_report": con_summary,
            "raw_data_summary": raw_summary
        }
    
    def _has_sufficient_information(self, inputs: Dict[str, Any]) -> bool:
        # Check lengths of summaries (rough heuristic)
        has_facts = "No verified facts" not in inputs["verified_facts_summary"]
        has_raw = "No raw data" not in inputs["raw_data_summary"]
        return has_facts or has_raw

    async def _provide_minimal_answer(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
        state["final_answer"] = f"Insufficient data for: {inputs['query']}"
        state["confidence_score"] = 0.1
        return state

    async def _provide_fallback_answer(self, state: AgentState) -> AgentState:
        state["final_answer"] = "Error generating answer. Please check logs."
        state["confidence_score"] = 0.0
        return state

# --- DIAGNOSTIC TEST FUNCTION ---
async def test_synthesis_agent():
    print("🧪 Testing SynthesisAgent (Enhanced Mode)...")
    
    # Initialize with stronger model config if possible, else defaults apply
    config = {
        "llm": {"synthesis_model": "llama3.1:8b", "temperature": 0.1}
    }
    agent = SynthesisAgent(config)
    
    # Robust Test State
    state = {
        "query": "Compare NVIDIA H200 and AMD MI300X",
        "query_analysis": {
            "intent": "comparison", # Crucial for triggering the right prompt path
            "metadata": {"complexity": "high"}
        },
        "verified_facts": [
            {"content": "NVIDIA H200 has 141GB HBM3e memory with 4.8TB/s bandwidth.", "verification_status": "verified"},
            {"content": "AMD MI300X has 192GB HBM3 memory with 5.3TB/s bandwidth.", "verification_status": "verified"},
            {"content": "H200 FP8 performance is approx 3,958 TFLOPS.", "verification_status": "verified"},
            {"content": "MI300X FP8 performance is approx 2,614 TFLOPS.", "verification_status": "verified"}
        ],
        "contradiction_report": [],
        "verification_confidence": 0.9,
        "research_results": {"findings": []},
        "retrieved_documents": [],
        "errors": [],
        "final_answer": None,
        "intermediate_answers": {},
        "agent_timestamps": {},
    }
    
    print("\n▶️  Executing Agent...")
    result = await agent.execute(state)
    
    print("\n🔍  RESULT ANALYSIS:")
    answer = result.get('final_answer')
    
    if answer:
        print(f"✅ Answer Generated:\n{'-'*60}\n{answer}\n{'-'*60}")
        print(f"📊 Confidence: {result.get('confidence_score')}")
    else:
        print("❌ Answer is NONE.")

if __name__ == "__main__":
    asyncio.run(test_synthesis_agent())

# """
# Synthesis Agent: Combines verified facts, research, and documents into final answer.
# Fourth core agent in the multi-agent workflow.
# """
# import sys
# import os
# import asyncio
# import traceback
# from typing import Dict, Any, List, Optional
# from datetime import datetime
# from pydantic import BaseModel, Field

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser , PydanticOutputParser
# from langchain_core.runnables import RunnableLambda
# from base_agent import BaseAgent
# from graph.state import AgentState, add_error_to_state

# # --- DIAGNOSTIC FIX: Define Model Locally ---
# # class SynthesisResult(BaseModel):
# #     answer: str = Field(description="The synthesized answer")
# #     confidence: float = Field(description="Confidence score between 0.0 and 1.0")
# #     citations: List[Dict[str, Any]] = Field(default_factory=list)
# #     contradictions: List[str] = Field(default_factory=list)
# #     assumptions: List[str] = Field(default_factory=list)

# # In agents/synthesis_agent.py

# class SynthesisResult(BaseModel):
#     answer: str = Field(description="The synthesized answer")
#     confidence: float = Field(description="Confidence score between 0.0 and 1.0")
#     citations: List[Dict[str, Any]] = Field(default_factory=list)
#     # CHANGE: Allow Dict or str for contradictions
#     contradictions: List[Any] = Field(default_factory=list) 
#     assumptions: List[str] = Field(default_factory=list)

# class SynthesisAgent(BaseAgent):
#     """
#     Synthesizes information into final answer, prioritizing verified facts.
#     """
    
#     def __init__(self, config: Dict[str, Any] = None):
#         super().__init__(
#             name="synthesis_agent",
#             description="Synthesizes information into final answer",
#             timeout_seconds=300,
#             max_retries=2
#         )
#         self.config = config or {}
#         self._initialize_llm()
#         self._initialize_prompts()
        
#     def _initialize_llm(self):
#         llm_config = self.config.get("llm", {})
#         model = llm_config.get("synthesis_model", "llama3.1:8b")
#         temp = llm_config.get("temperature", 0.1)
#         # Handle dict config
#         if isinstance(temp, dict): temp = temp.get("default", 0.1)
        
#         self.llm = OllamaLLM(model=model, temperature=float(temp), format="json")
#         print(f"✅ SynthesisAgent LLM initialized: {model}")
    
#     def _initialize_prompts(self):
#         self.synthesis_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a synthesis expert.
# INPUTS:
# 1. Query: {query}
# 2. Facts: {verified_facts_summary}
# 3. Contradictions: {contradiction_report}
# 4. Raw Data: {raw_data_summary}

# ### OUTPUT INSTRUCTIONS:
# You must output a single valid JSON object that strictly adheres to the schema below. 
# **IMPORTANT:** Do NOT return the schema definition/rules. Return only the actual JSON data containing the 'answer' , 'confidence' , 'citations' , 'contradictions' , 'assumptions' fields.

# Schema format:
# {format_instructions}"""),
#             ("human", "Synthesize the answer.")
#         ])

    
#     async def execute(self, state: AgentState) -> AgentState:
#         """
#         Diagnostic override to catch errors that BaseAgent might swallow.
#         """
#         print(f"DEBUG: Entering SynthesisAgent.execute manually")
#         try:
#             # Directly call impl to bypass potential middleware crashes
#             return await self._execute_impl(state)
#         except Exception as e:
#             print(f"❌ CRITICAL ERROR in SynthesisAgent: {e}")
#             traceback.print_exc()
            
#             # Return state with error info
#             state["final_answer"] = f"Agent Crashed: {str(e)}"
#             state["confidence_score"] = 0.0
#             state["errors"].append({"agent": self.name, "error": str(e)})
#             return state

#     async def _execute_impl(self, state: AgentState) -> AgentState:
#         print(f"🧠 SynthesisAgent synthesizing final answer...")
        
#         try:
#             inputs = self._prepare_synthesis_inputs(state)
            
#             if not self._has_sufficient_information(inputs):
#                 print("⚠️  Insufficient information")
#                 return await self._provide_minimal_answer(state, inputs)
            
#             # Run Chain
#             print("⏳ Invoking LLM...")
#             # parser = JsonOutputParser()
#             # chain = self.synthesis_prompt | self.llm | parser
#             self.parser = PydanticOutputParser(pydantic_object = SynthesisResult)
#             chain = (
#                 self.synthesis_prompt | self.llm | self.parser | RunnableLambda(lambda x: x.model_dump())
#             )
#             inputs["format_instructions"] = self.parser.get_format_instructions()
#             result = await chain.ainvoke(inputs)
            
#             # Process Result
#             synthesis_result = SynthesisResult(
#                 answer=result.get("answer", "No answer generated"),
#                 confidence=result.get("confidence", 0.5),
#                 citations=result.get("citations", []),
#                 contradictions=result.get("contradictions", []),
#                 assumptions=result.get("assumptions", [])
#             )
            
#             # Calc Confidence
#             up_conf = state.get("verification_confidence") or 0.5
#             final_conf = (up_conf * 0.4) + (synthesis_result.confidence * 0.6)
            
#             # Update State
#             state["final_answer"] = synthesis_result.answer
#             state["confidence_score"] = float(f"{final_conf:.2f}")
#             state["citations"] = synthesis_result.citations
            
#             if state.get("intermediate_answers") is None:
#                 state["intermediate_answers"] = {}
#             state["intermediate_answers"]["synthesis"] = synthesis_result.model_dump()
            
#             print(f"✅ Synthesis complete. Confidence: {state['confidence_score']}")
#             return state
            
#         except Exception as e:
#             print(f"❌ Logic Error in _execute_impl: {e}")
#             traceback.print_exc()
#             return await self._provide_fallback_answer(state)
    
#     def _prepare_synthesis_inputs(self, state: AgentState) -> Dict[str, Any]:
#         query = state.get("query", "")
        
#         # 1. Facts
#         facts = state.get("verified_facts") or []
#         fact_summary = f"Facts ({len(facts)}):\n" + "\n".join([f"- {f.get('content','')}" for f in facts])
        
#         # 2. Contradictions
#         reports = state.get("contradiction_report") or []
#         con_summary = "None" if not reports else str(reports)
        
#         # 3. Raw Data
#         findings = (state.get("research_results") or {}).get("findings", [])
#         docs = state.get("retrieved_documents") or []
#         raw_summary = f"Research: {len(findings)} items, Docs: {len(docs)} items."
        
#         return {
#             "query": query,
#             "verified_facts_summary": fact_summary,
#             "contradiction_report": con_summary,
#             "raw_data_summary": raw_summary
#         }
    
#     def _has_sufficient_information(self, inputs: Dict[str, Any]) -> bool:
#         # Check lengths of summaries
#         return len(inputs["verified_facts_summary"]) > 20 or len(inputs["raw_data_summary"]) > 30

#     async def _provide_minimal_answer(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
#         state["final_answer"] = f"Insufficient data for: {inputs['query']}"
#         state["confidence_score"] = 0.1
#         return state

#     async def _provide_fallback_answer(self, state: AgentState) -> AgentState:
#         state["final_answer"] = "Error generating answer."
#         state["confidence_score"] = 0.0
#         return state

# # --- DIAGNOSTIC TEST FUNCTION ---
# async def test_synthesis_agent():
#     print("🧪 Testing SynthesisAgent (Diagnostic Mode)...")
#     agent = SynthesisAgent()
    
#     # Robust Test State
#     state = {
#         "query": "Compare NVIDIA H200 and AMD MI300X",
#         "verified_facts": [
#             {"content": "NVIDIA H200 has 141GB memory", "verification_status": "verified"},
#             {"content": "AMD MI300X has 192GB memory", "verification_status": "verified"}
#         ],
#         "contradiction_report": [],
#         "verification_confidence": 0.9,
#         "research_results": {"findings": []},
#         "retrieved_documents": [],
#         "query_analysis": {},
#         "errors": [],
#         "final_answer": None,
#         "intermediate_answers": {},
#         "agent_timestamps": {},  # Required by BaseAgent
#     }
    
#     print("\n▶️  Executing Agent...")
#     result = await agent.execute(state)
    
#     print("*"*50)
#     print(result)
#     print("*"*50)
#     print("\n🔍  RESULT ANALYSIS:")
#     answer = result.get('final_answer')
    
#     if answer:
#         print(f"✅ Answer Generated: {answer[:100]}...")
#         print(f"📊 Confidence: {result.get('confidence_score')}")
#     else:
#         print("❌ Answer is NONE.")
#         print("dumping result keys:", result.keys())
#         # Check for BaseAgent error format
#         if result.get(f"{agent.name}_result"):
#             print(f"⚠️  Agent Error Found: {result[f'{agent.name}_result']}")
#         if result.get("errors"):
#             print(f"⚠️  Errors List: {result['errors']}")

# if __name__ == "__main__":
#     asyncio.run(test_synthesis_agent())
# """
# Synthesis Agent: Combines research findings and retrieved documents into final answer.
# Fourth core agent in the multi-agent workflow.
# """
# import sys
# import os
# import asyncio
# from typing import Dict, Any, List
# from datetime import datetime

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from agents.base_agent import BaseAgent
# from graph.state import AgentState, add_error_to_state, SynthesisResult

# class SynthesisAgent(BaseAgent):
#     """
#     Synthesizes research findings and retrieved documents into final answer.
#     """
    
#     def __init__(self, config: Dict[str, Any] = None):
#         """Initialize SynthesisAgent."""
#         super().__init__(
#             name="synthesis_agent",
#             description="Synthesizes information into final answer",
#             timeout_seconds=300,
#             max_retries=2
#         )
        
#         self.config = config or {}
#         self._initialize_llm()
#         self._initialize_prompts()
        
#     def _initialize_llm(self):
#         """Initialize the LLM for synthesis."""
#         llm_config = self.config.get("llm", {})
#         model = llm_config.get("synthesis_model", "llama3.1:8b")
        
#         # --- FIX: Handle dictionary configuration for temperature ---
#         temp_config = llm_config.get("temperature", 0.1)
#         if isinstance(temp_config, dict):
#             temperature = temp_config.get("synthesis", temp_config.get("default", 0.1))
#         else:
#             temperature = float(temp_config)
#         # ------------------------------------------------------------
        
#         self.llm = OllamaLLM(
#             model=model,
#             temperature=temperature,
#             format="json"
#         )
#         print(f"✅ SynthesisAgent LLM initialized: {model}")
    
#     def _initialize_prompts(self):
#         """Initialize prompt templates."""
#         # Main synthesis prompt
        
#         self.synthesis_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a competitive intelligence synthesis expert.
# Your task is to combine research findings with document retrieval results to create a comprehensive, accurate answer.

# ## INPUTS AVAILABLE:
# 1. User Query: {query}
# 2. Research Findings: {research_summary}
# 3. Retrieved Documents: {documents_summary}
# 4. Query Analysis: {analysis_summary}

# ## OUTPUT FORMAT (STRICT JSON):
# {{
#     "answer": "Comprehensive answer synthesizing all available information. Include specific facts, numbers, and dates when available.",
#     "confidence": 0.0-1.0,
#     "citations": [
#         {{
#             "source_type": "research|document",
#             "source_id": "reference to source",
#             "content_referenced": "specific fact or quote",
#             "credibility": 0.0-1.0
#         }}
#     ],
#     "contradictions": ["List any contradictory information found"],
#     "assumptions": ["List assumptions made in the answer"],
#     "answer_format": "paragraph|table|list|json"
# }}

# ## SYNTHESIS GUIDELINES:
# 1. Prioritize information from research findings (external data)
# 2. Support with specific details from documents
# 3. Acknowledge contradictions or gaps in information
# 4. Rate confidence based on information quality and completeness
# 5. Include citations for key facts
# 6. Format answer based on query requirements

# Generate a synthesis that directly answers the user's query."""),
#             ("human", "Please synthesize the information into a final answer.")
#         ])
        
#         # Prompt for generating citations
#         self.citation_prompt = ChatPromptTemplate.from_messages([
#             ("system", """Extract key facts from the following content and create citations.
# For each significant fact, create a citation entry with:
# - Source type (research/document)
# - Specific content referenced
# - Credibility score (based on source quality)
# - Reference to original source"""),
#             ("human", "Content to cite:\n{content}")
#         ])
    
#     async def _execute_impl(self, state: AgentState) -> AgentState:
#         """
#         Synthesize information into final answer.
        
#         Args:
#             state: Current AgentState
            
#         Returns:
#             Updated AgentState with final answer
#         """
#         print(f"🧠 SynthesisAgent synthesizing final answer...")
        
#         try:
#             # Prepare inputs for synthesis
#             inputs = self._prepare_synthesis_inputs(state)
            
#             # Check if we have enough information
#             if not self._has_sufficient_information(inputs):
#                 print("⚠️  Insufficient information for synthesis")
#                 return await self._provide_minimal_answer(state, inputs)
            
#             # Create synthesis chain
#             parser = JsonOutputParser()
#             chain = self.synthesis_prompt | self.llm | parser
            
#             # Execute synthesis
#             result = await chain.ainvoke({
#                 "query": inputs["query"],
#                 "research_summary": inputs["research_summary"],
#                 "documents_summary": inputs["documents_summary"],
#                 "analysis_summary": inputs["analysis_summary"]
#             })
            
#             # Validate result
#             if not self._validate_synthesis_result(result):
#                 raise ValueError("Invalid synthesis result structure")
            
#             # Create structured synthesis result
#             synthesis_result = SynthesisResult(
#                 answer=result["answer"],
#                 confidence=result["confidence"],
#                 citations=result["citations"],
#                 contradictions=result.get("contradictions", []),
#                 assumptions=result.get("assumptions", [])
#             )
            
#             # Update state
#             state["final_answer"] = synthesis_result.answer
#             state["confidence_score"] = synthesis_result.confidence
#             state["citations"] = synthesis_result.citations
#             state["intermediate_answers"] = {
#                 "synthesis": synthesis_result.model_dump()
#             }
            
#             # Add any contradictions to fact_check_results
#             if synthesis_result.contradictions:
#                 if state.get("fact_check_results")  is None:
#                     state["fact_check_results"] = []
#                 for contradiction in synthesis_result.contradictions:
#                     state["fact_check_results"].append({
#                         "type": "contradiction",
#                         "description": contradiction,
#                         "agent": self.name,
#                         "timestamp": datetime.now().isoformat()
#                     })
            
#             print(f"✅ Synthesis complete: Confidence={synthesis_result.confidence:.2f}")
#             print(f"   Answer preview: {synthesis_result.answer[:150]}...")
            
#             return state
            
#         except Exception as e:
#             error_msg = f"Synthesis failed: {str(e)}"
#             print(f"❌ {error_msg}")
#             state = add_error_to_state(state, self.name, error_msg)
            
#             # Provide fallback answer
#             return await self._provide_fallback_answer(state)
    
#     def _prepare_synthesis_inputs(self, state: AgentState) -> Dict[str, Any]:
#         """Prepare inputs for synthesis."""
#         query = state.get("query", "")
        
#         # Summarize research findings
#         research_summary = "No research findings available."
#         if state.get("research_results"):
#             research_data = state["research_results"]
#             findings = research_data.get("findings", [])
#             if findings:
#                 research_summary = f"Research Findings ({len(findings)}):\n"
#                 for i, finding in enumerate(findings[:5]):  # First 5 findings
#                     content = finding.get("content", "")[:200]
#                     source = finding.get("source_type", "unknown")
#                     credibility = finding.get("credibility", 0.5)
#                     research_summary += f"{i+1}. [{source}, cred={credibility:.2f}] {content}\n"
        
#         # Summarize retrieved documents
#         documents_summary = "No documents retrieved."
#         if state.get("retrieved_documents"):
#             docs = state["retrieved_documents"]
#             if docs:
#                 documents_summary = f"Retrieved Documents ({len(docs)}):\n"
#                 for i, doc in enumerate(docs[:5]):  # First 5 documents
#                     content = doc.page_content[:200]
#                     if doc.metadata:
#                         source = doc.metadata.get("source", "document")
#                     else:
#                         source = "document"
#                     documents_summary += f"{i+1}. [{source}] {content}\n"
        
#         # Summarize query analysis
#         analysis_summary = "No query analysis available."
#         if state.get("query_analysis"):
#             analysis = state["query_analysis"]
#             intent = analysis.get("intent", "unknown")
#             complexity = analysis.get("metadata", {}).get("complexity", "unknown")
#             tasks = len(analysis.get("tasks", []))
#             analysis_summary = f"Query Analysis: Intent={intent}, Complexity={complexity}, Tasks={tasks}"
        
#         return {
#             "query": query,
#             "research_summary": research_summary,
#             "documents_summary": documents_summary,
#             "analysis_summary": analysis_summary
#         }
    
#     def _has_sufficient_information(self, inputs: Dict[str, Any]) -> bool:
#         """Check if we have sufficient information for synthesis."""
#         has_research = "No research findings" not in inputs["research_summary"]
#         has_documents = "No documents retrieved" not in inputs["documents_summary"]
        
#         # We need at least one source of information
#         return has_research or has_documents
    
#     def _validate_synthesis_result(self, result: Dict) -> bool:
#         """Validate synthesis result structure."""
#         required_keys = ["answer", "confidence", "citations"]
        
#         if not isinstance(result, dict):
#             return False
        
#         for key in required_keys:
#             if key not in result:
#                 return False
        
#         # Validate types
#         if not isinstance(result["answer"], str):
#             return False
#         if not isinstance(result["confidence"], (int, float)):
#             return False
#         if not isinstance(result["citations"], list):
#             return False
        
#         return True
    
#     async def _provide_minimal_answer(self, state: AgentState, inputs: Dict[str, Any]) -> AgentState:
#         """Provide a minimal answer when information is insufficient."""
#         print("🔄 Providing minimal answer due to insufficient information")
        
#         answer = f"I couldn't find sufficient information to fully answer: '{inputs['query']}'\n\n"
#         answer += "The system retrieved limited information. For a comprehensive answer, please:\n"
#         answer += "1. Try rephrasing your query\n"
#         answer += "2. Ask about more specific topics\n"
#         answer += "3. Check back later as the knowledge base expands"
        
#         state["final_answer"] = answer
#         state["confidence_score"] = 0.2
#         state["citations"] = []
        
#         return state
    
#     async def _provide_fallback_answer(self, state: AgentState) -> AgentState:
#         """Provide fallback answer when synthesis fails."""
#         print("🔄 Using fallback synthesis...")
        
#         query = state.get("query", "")
        
#         # Create simple fallback answer
#         fallback_answer = f"Based on available information about '{query}':\n\n"
        
#         if state.get("research_results"):
#             findings = state["research_results"].get("findings", [])
#             if findings:
#                 fallback_answer += "Research indicates:\n"
#                 for finding in findings[:3]:
#                     fallback_answer += f"• {finding.get('content', '')[:150]}...\n"
        
#         if state.get("retrieved_documents"):
#             docs = state["retrieved_documents"]
#             if docs:
#                 fallback_answer += "\nDocuments suggest:\n"
#                 for doc in docs[:3]:
#                     fallback_answer += f"• {doc.page_content[:150]}...\n"
        
#         fallback_answer += "\nNote: This is a fallback answer due to synthesis limitations."
        
#         state["final_answer"] = fallback_answer
#         state["confidence_score"] = 0.4
#         state["citations"] = []
        
#         return state

# # Test function
# async def test_synthesis_agent():
#     """Test the SynthesisAgent."""
#     print("🧪 Testing SynthesisAgent...")
    
#     # Initialize agent
#     agent = SynthesisAgent()
    
#     # Create test state with mock data
#     state = {
#         "query": "Compare NVIDIA H200 and AMD MI300X for AI workloads",
#         "query_analysis": {
#             "intent": "comparison",
#             "metadata": {
#                 "complexity": "complex",
#                 "expected_output_format": "table"
#             },
#             "tasks": [
#                 {"task": "Compare GPU specifications", "agent": "research"},
#                 {"task": "Retrieve relevant documents", "agent": "doc_retrieval"}
#             ]
#         },
#         "research_results": {
#             "findings": [
#                 {
#                     "content": "NVIDIA H200 offers 1,979 TFLOPS FP8 performance",
#                     "source_type": "technical_spec",
#                     "credibility": 0.95
#                 },
#                 {
#                     "content": "AMD MI300X provides 1,835 TFLOPS with 5.2 TB/s memory bandwidth",
#                     "source_type": "technical_spec",
#                     "credibility": 0.95
#                 }
#             ]
#         },
#         "retrieved_documents": [
#             type('Document', (), {
#                 'page_content': 'H200 priced at $98.32/hour on AWS p5 instances',
#                 'metadata': {'source': 'AWS pricing docs'}
#             })(),
#             type('Document', (), {
#                 'page_content': 'MI300X available on Azure at $89.75/hour',
#                 'metadata': {'source': 'Azure documentation'}
#             })()
#         ],
#         "errors": [],
#         "agent_timestamps": {},
#         "execution_path": [],
#         "decomposed_tasks": None,
#         "intent": None,
#         "routing_decision": None,
#         "fact_check_results": None,
#         "intermediate_answers": None,
#         "final_answer": None,
#         "confidence_score": None,
#         "citations": None,
#         "retrieval_sources": [],
#         "latency_per_agent": {},
#         "token_usage": {},
#         "retrieval_metadata": None
#     }

    
#     # Execute agent
#     result = await agent.execute(state)
    
#     # Print results
#     if result.get("final_answer"):
#         print(f"✅ Final answer generated (confidence: {result.get('confidence_score', 0):.2f})")
#         print(f"📝 Answer preview: {result['final_answer'][:200]}...")
        
#         if result.get("citations"):
#             print(f"📚 Citations: {len(result['citations'])}")
#     else:
#         print("❌ No final answer generated")
    
#     # Print agent statistics
#     print(f"\n📊 Agent Statistics:")
#     stats = agent.get_stats()
#     for key, value in stats.items():
#         print(f"  {key}: {value}")

# if __name__ == "__main__":
#     asyncio.run(test_synthesis_agent())





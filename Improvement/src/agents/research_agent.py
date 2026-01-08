"""
Research Agent: Gathers external information via web/search APIs.
Uses Tavily API for real web research.
"""
import os
import sys
import asyncio
import json
from typing import Dict, Any, List , Literal
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel , Field
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser , PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from tavily import TavilyClient

from agents.base_agent import BaseAgent
from graph.state import AgentState, add_error_to_state, ResearchResult

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# class instruction(BaseModel):
#     {{
#     "findings": [
#         {{
#             "content": "Detailed finding with specific facts, numbers, and dates extracted from search results",
#             "source_type": "technical_spec|earnings_call|research_paper|news_article|industry_report|blog",
#             "recency": "2024|2023|2022|older",
#             "credibility": 0.0-1.0,
#             "key_metrics": ["metric1", "metric2"],
#             "source_title": "Original article title",
#             "source_url": "Actual URL from search results"
#         }}
#     ],
#     "summary": "Brief summary of research findings",
#     "confidence": 0.0-1.0,
#     "gaps_identified": ["list of information not found"],
#     "recommended_next_steps": ["suggestions for further research"]
# }}
class Finding(BaseModel):
    content: str = Field(
        ..., 
        description="Detailed finding with specific facts, numbers, and dates extracted from search results"
    )
    source_type: Literal[
        "technical_spec", 
        "earnings_call", 
        "research_paper", 
        "news_article", 
        "industry_report", 
        "blog"
    ]
    recency: Literal["2025","2024", "2023", "2022", "older"]
    credibility: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Score between 0.0 and 1.0 representing source credibility"
    )
    key_metrics: List[str] = Field(default_factory=list)
    source_title: str
    source_url: str  # You can change this to HttpUrl if you want strict URL validation

class ResearchReport(BaseModel):
    findings: List[Finding]
    summary: str = Field(..., description="Brief summary of research findings")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence score of the research"
    )
    gaps_identified: List[str] = Field(
        default_factory=list, 
        description="List of information that were not found"
    )
    recommended_next_steps: List[str] = Field(
        default_factory=list, 
        description="Suggestions for further research"
    )


class ResearchAgent(BaseAgent):
    """
    Research Agent for gathering external information using Tavily API.
    
    Responsibilities:
    1. Web search via Tavily API
    2. Source credibility scoring
    3. Time-based filtering (recency matters)
    4. Structured research output
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ResearchAgent."""
        super().__init__(
            name="research_agent",
            description="Gathers external information via web/search",
            timeout_seconds=600,
            max_retries=2
        )
        
        self.config = config or {}
        self._initialize_tavily_client()
        self._initialize_llm()
        self._initialize_prompts()
        
    def _initialize_tavily_client(self):
        """Initialize Tavily API client."""
        try:
            # Get API key from environment variable
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY not found in environment variables")
            
            self.tavily_client = TavilyClient(api_key=api_key)
            print("✅ Tavily API client initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Tavily client: {e}")
            self.tavily_client = None
    
    def _initialize_llm(self):
        """Initialize the LLM for research tasks."""
        llm_config = self.config.get("llm", {})
        model = llm_config.get("research_model", "nemotron-mini:4b")
        
        # Handle dictionary configuration for temperature
        temp_config = llm_config.get("temperature", 0.0)
        if isinstance(temp_config, dict):
            temperature = temp_config.get("default", 0.0)
        else:
            temperature = float(temp_config)
        
        self.llm = OllamaLLM(
            model=model,
            temperature=temperature,
            timeout=self.timeout_seconds
        )
        print(f"✅ ResearchAgent LLM initialized: {model}")
    
    def _initialize_prompts(self):
        """Initialize prompt templates."""
        # Main research prompt for analyzing search results
#         self.research_analysis_prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are a competitive intelligence research assistant.
# Analyze the provided web search results and extract structured findings.

# ## OUTPUT FORMAT (STRICT JSON):
# You must output a single valid JSON object that strictly adheres to the schema below. 
# **IMPORTANT:** Do NOT return the schema definition/rules. Return only the actual JSON data containing the 'intent', 'tasks', 'metadata', and 'routing_decision' fields based on the user's query.

# Schema format:
# {format_instructions}

# Now, analyze the following search results:"""),
#             ("human", "Search Query: {query}\n\nSearch Results:\n{search_results}")
#         ])
        self.research_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a competitive intelligence research assistant.
Analyze the provided web search results and extract structured findings.

## OUTPUT FORMAT (STRICT JSON):
    {{
    "findings": [
        {{
            "content": "Detailed finding with specific facts, numbers, and dates extracted from search results",
            "source_type": "technical_spec|earnings_call|research_paper|news_article|industry_report|blog",
            "recency": "2024|2023|2022|older",
            "credibility": 0.0-1.0,
            "key_metrics": ["metric1", "metric2"],
            "source_title": "Original article title",
            "source_url": "Actual URL from search results"
        }}
    ],
    "summary": "Brief summary of research findings",
    "confidence": 0.0-1.0,
    "gaps_identified": ["list of information not found"],
    "recommended_next_steps": ["suggestions for further research"]
}}

Now, analyze the following search results:"""),
            ("human", "Search Query: {query}\n\nSearch Results:\n{search_results}")
        ])
        
        # Source credibility assessment prompt
        self.credibility_prompt = ChatPromptTemplate.from_messages([
            ("system", """Assess the credibility of a source based on:
1. Domain authority (.edu, .gov, official company site, news outlet, blog)
2. Content quality (cited sources, detailed analysis vs. opinion)
3. Author/reputation (academic, industry expert, journalist)
4. Recency (current year = high, >2 years = low for time-sensitive topics)

Rate credibility from 0.0 (not credible) to 1.0 (highly credible).

Respond with a JSON containing credibility score and reasoning."""),
            ("human", "Source URL: {url}\nContent Excerpt: {content}\nDate: {date}")
        ])
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Execute research using Tavily API.
        
        Args:
            state: Current AgentState
            
        Returns:
            Updated AgentState with research findings
        """
        print(f"🔍 ResearchAgent conducting research with Tavily API...")
        
        try:
            # Check if Tavily client is available
            if not self.tavily_client:
                raise ValueError("Tavily API client not available")
            
            # Get query and analysis from state
            query = state.get("query", "")
            analysis = state.get("query_analysis", {})
            
            if not query:
                print("⚠️  No query found for research")
                return state
            
            # Extract research tasks from analysis
            research_tasks = self._extract_research_tasks(analysis, query)
            
            if not research_tasks:
                print("⚠️  No research tasks identified")
                return state
            
            print(f"   Found {len(research_tasks)} research tasks")
            
            # Execute research tasks
            all_findings = []
            all_sources = []
            
            for task in research_tasks:
                print(f"   Researching: {task[:60]}...")
                
                # Perform web search using Tavily API
                search_results = await self._perform_web_search(task)
                
                if search_results:
                    # Analyze search results with LLM
                    findings = await self._analyze_search_results(task, search_results)
                    all_findings.extend(findings)
                    
                    # Extract sources from findings
                    for finding in findings:
                        source = {
                            "type": finding.get("source_type", "unknown"),
                            "recency": finding.get("recency", "unknown"),
                            "credibility": finding.get("credibility", 0.5),
                            "url": finding.get("source_url", ""),
                            "title": finding.get("source_title", ""),
                            "key_points": finding.get("key_metrics", [])
                        }
                        all_sources.append(source)
                    
                    print(f"   Found {len(findings)} findings for this task")
                else:
                    print(f"   No search results found for: {task}")
            
            # If no findings, try broader search
            if not all_findings and research_tasks:
                print("   Trying broader search...")
                broad_results = await self._perform_web_search(query)
                if broad_results:
                    findings = await self._analyze_search_results(query, broad_results)
                    all_findings.extend(findings)
            
            # Create structured research result
            if all_findings:
                research_result = ResearchResult(
                    findings=all_findings,
                    sources=all_sources,
                    credibility_score=self._calculate_overall_credibility(all_sources),
                    recency_score=self._calculate_recency_score(all_sources)
                )
                
                # Update state
                state["research_results"] = {
                    "findings": all_findings,
                    "sources": all_sources,
                    "credibility_score": research_result.credibility_score,
                    "recency_score": research_result.recency_score,
                    "total_findings": len(all_findings),
                    "search_method": "tavily_api"
                }
                
                print(f"✅ Research completed: {len(all_findings)} findings, {len(all_sources)} sources")
                print(f"   Credibility: {research_result.credibility_score:.2f}, Recency: {research_result.recency_score:.2f}")
                
                # Log sample finding
                if all_findings:
                    sample = all_findings[0]["content"][:100]
                    print(f"   Sample finding: {sample}...")
            else:
                print("⚠️  No research findings generated")
                state["research_results"] = {
                    "findings": [],
                    "sources": [],
                    "credibility_score": 0.0,
                    "recency_score": 0.0,
                    "total_findings": 0,
                    "search_method": "tavily_api",
                    "no_results": True
                }
            
            return state
            
        except Exception as e:
            error_msg = f"Research execution failed: {str(e)}"
            print(f"❌ {error_msg}")
            state = add_error_to_state(state, self.name, error_msg)
            
            # Provide minimal fallback information
            state["research_results"] = {
                "findings": [],
                "sources": [],
                "credibility_score": 0.0,
                "recency_score": 0.0,
                "total_findings": 0,
                "error": str(e),
                "search_method": "failed"
            }
            return state
    
    async def _perform_web_search(self, query: str) -> List[Dict]:
        """
        Perform web search using Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            # Configure search parameters
            search_params = {
                "query": query,
                "max_results": 10,  # Maximum results to return
                "include_raw_content": True,  # Include full content
                "search_depth": "advanced",  # Advanced search for better results
                "include_answer": False,  # We'll analyze ourselves
                "include_images": False,
                "include_image_descriptions": False,
            }
            
            # Perform search
            print(f"   Searching Tavily for: {query[:50]}...")
            response = self.tavily_client.search(**search_params)
            
            # Extract results
            results = response.get("results", [])
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "raw_content": result.get("raw_content", ""),
                }
                processed_results.append(processed_result)
            
            print(f"   Found {len(processed_results)} search results")
            return processed_results
            
        except Exception as e:
            print(f"   Tavily search failed: {e}")
            return []
    def _clean_json_output(self, text: str) -> str:
            """Strip markdown code blocks to extract raw JSON."""
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return text.strip()
    
    async def _analyze_search_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """
        Analyze search results and extract structured findings.
        
        Args:
            query: Original search query
            search_results: List of search results from Tavily
            
        Returns:
            List of structured findings
        """
        try:
            # Format search results for the LLM
            formatted_results = ""
            for i, result in enumerate(search_results[:5]):  # Limit to top 5 for token efficiency
                formatted_results += f"\n--- Result {i+1} ---\n"
                formatted_results += f"Title: {result.get('title', 'N/A')}\n"
                formatted_results += f"URL: {result.get('url', 'N/A')}\n"
                formatted_results += f"Content: {result.get('content', 'N/A')[:500]}...\n"
            
            # parser = PydanticOutputParser(pydantic_object=ResearchReport)

            # Create analysis chain
            # chain = self.research_analysis_prompt | self.llm | parser | RunnableLambda(lambda x: x.model_dump())
            # chain = self.research_analysis_prompt | self.llm | JsonOutputParser()
            
            # Analyze with LLM
            # analysis_result = await chain.ainvoke({
            #     "query": query,
            #     "search_results": formatted_results,
            #     "format_instructions": parser.get_format_instructions()
            # })
            # analysis_result = await chain.ainvoke({
            #     "query": query,
            #     "search_results": formatted_results
            # })
            chain = self.research_analysis_prompt | self.llm 
            raw_response = await chain.ainvoke({
                "query": query, 
                "search_results": formatted_results
            })
            
            # Clean the output
            clean_text = self._clean_json_output(raw_response)
            
            # Parse manually
            try:
                analysis_result = json.loads(clean_text)
            except:
                # Fallback: try to find list in text if full JSON fails
                return []
            
            # Extract findings
            findings = analysis_result.get("findings", [])
            
            # Enhance findings with credibility assessment
            enhanced_findings = []
            for finding in findings:
                # Assess credibility
                credibility = await self._assess_source_credibility(
                    finding.get("source_url", ""),
                    finding.get("content", ""),
                    finding.get("recency", "unknown")
                )
                
                enhanced_finding = finding.copy()
                enhanced_finding["credibility"] = credibility
                enhanced_findings.append(enhanced_finding)
            
            return enhanced_findings
            
        except Exception as e:
            print(f"   Search analysis failed: {e}")
            return []
    
    async def _assess_source_credibility(self, url: str, content: str, date: str) -> float:
        """
        Assess the credibility of a source.
        
        Args:
            url: Source URL
            content: Content excerpt
            date: Date information
            
        Returns:
            Credibility score (0.0 to 1.0)
        """
        try:
            # Simple heuristic-based credibility assessment
            credibility = 0.5  # Default
            
            # Check domain
            if any(domain in url for domain in [".edu", ".gov", ".ac."]):
                credibility += 0.3
            elif any(domain in url for domain in ["research", "arxiv", "ncbi", "scholar"]):
                credibility += 0.2
            elif any(domain in url for domain in ["blog", "medium", "substack"]):
                credibility -= 0.1
            
            # Check recency
            if "2024" in date:
                credibility += 0.1
            elif "2023" in date:
                credibility += 0.05
            
            # Check content length (longer content often more credible)
            if len(content) > 200:
                credibility += 0.1
            elif len(content) < 50:
                credibility -= 0.1
            
            # Ensure score is within bounds
            return max(0.0, min(1.0, credibility))
            
        except:
            return 0.5  # Default if assessment fails
    
    def _extract_research_tasks(self, analysis: Dict, query: str) -> List[str]:
        """Extract research tasks from query analysis."""
        tasks = []
        
        # Get tasks from analysis if available
        if analysis and "tasks" in analysis:
            for task in analysis["tasks"]:
                if task.get("agent") == "research":
                    tasks.append(task.get("task", ""))
        
        if len(tasks):
            print(f"The tasks are: {tasks}")
        else:
            print("No tasks found!!!\nGoing to default mode\n")
            # Simple keyword-based task extraction
            tasks.append(f"Search for: {query}")
        
        return tasks
    
    def _calculate_overall_credibility(self, sources: List[Dict]) -> float:
        """Calculate overall credibility score from sources."""
        if not sources:
            return 0.0
        
        credibility_scores = [s.get("credibility", 0.0) for s in sources]
        return sum(credibility_scores) / len(credibility_scores)
    
    def _calculate_recency_score(self, sources: List[Dict]) -> float:
        """Calculate recency score from sources."""
        if not sources:
            return 0.0
        
        recency_map = {"2024": 1.0, "2023": 0.8, "2022": 0.6, "2021": 0.4, "2020": 0.2, "older": 0.1}
        
        scores = []
        for source in sources:
            recency = source.get("recency", "older")
            # Extract year from recency string
            for year in ["2024", "2023", "2022", "2021", "2020"]:
                if year in recency:
                    score = recency_map.get(year, 0.1)
                    scores.append(score)
                    break
            else:
                scores.append(recency_map.get("older", 0.1))
        
        return sum(scores) / len(scores)
    
    async def perform_real_research(self, query: str, max_results: int = 10) -> Dict:
        """
        Perform real research using Tavily API.
        
        Args:
            query: Research query
            max_results: Maximum number of results
            
        Returns:
            Research results
        """
        print(f"🔍 Performing real research for: {query}")
        
        try:
            if not self.tavily_client:
                return {"error": "Tavily API not available"}
            
            # Perform search
            search_results = await self._perform_web_search(query)
            
            if not search_results:
                return {"query": query, "findings": [], "total_findings": 0}
            
            # Analyze results
            findings = await self._analyze_search_results(query, search_results)
            
            return {
                "query": query,
                "search_results": search_results,
                "findings": findings,
                "total_findings": len(findings),
                "method": "tavily_api",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"query": query, "error": str(e), "method": "tavily_api_failed"}


# Test function
async def test_research_agent():
    """Test the ResearchAgent with Tavily API."""
    print("🧪 Testing ResearchAgent with Tavily API...")
    
    # Check for API key
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found in environment variables")
        print("   Please set TAVILY_API_KEY in your .env file")
        return
    
    # Initialize agent
    agent = ResearchAgent()
    
    # Test with a real search query
    test_query = "What are the key technical specifications of the Nvidia Blackwell B200 GPU?"
    
    print(f"\n🔍 Test Query: {test_query}")
    print("-" * 40)
    
    # Perform real research
    result = await agent.perform_real_research(test_query, max_results=5)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Search completed: {result['total_findings']} findings")
        
        if result["findings"]:
            for i, finding in enumerate(result["findings"][:3]):  # Show first 3
                print(f"\n📄 Finding {i+1}:")
                print(f"   Content: {finding.get('content', '')[:150]}...")
                print(f"   Source: {finding.get('source_url', 'N/A')}")
                print(f"   Credibility: {finding.get('credibility', 0.0):.2f}")
                print(f"   Recency: {finding.get('recency', 'unknown')}")
    
    # Test statistics
    print(f"\n📊 Agent Statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_research_agent())

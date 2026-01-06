"""
Module: base_agent.py
"""
"""
Base Agent class for all agents in the multi-agent system.
"""
import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
import json
from pathlib import Path
import traceback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableSerializable

from graph.state import AgentState, update_state_timestamp, add_error_to_state



class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    All agents must implement the `execute` method.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        timeout_seconds: int = 100,
        max_retries: int = 2,
        config_path: Optional[Path] = None
    ):
        self.name = name
        self.description = description
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.config = self._load_config(config_path) if config_path else {}
        
        # LLM components (to be initialized by subclasses)
        self.llm = None
        self.prompt = None
        self.chain = None
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.average_latency = 0.0
        self.errors = []
        
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load agent configuration from YAML/JSON file."""
        if not config_path.exists():
            return {}
        
        if config_path.suffix == ".json":
            with open(config_path, 'r') as f:
                return json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute the agent's task asynchronously.
        
        Args:
            state: Current AgentState
            
        Returns:
            Updated AgentState
        """
        start_time = datetime.now()
        self.total_executions += 1
        
        try:
            # Update timestamp
            state = update_state_timestamp(state, self.name)
            
            # Execute with timeout
            result_state = await asyncio.wait_for(
                self._execute_impl(state),
                timeout=self.timeout_seconds
            )
            
            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds()
            self._update_latency_stats(latency)
            
            # Store latency in state
            if "latency_per_agent" not in result_state:
                result_state["latency_per_agent"] = {}
            result_state["latency_per_agent"][self.name] = latency
            
            self.successful_executions += 1
            return result_state
            
        except asyncio.TimeoutError:
            error_msg = f"{self.name} timed out after {self.timeout_seconds} seconds"
            self.errors.append({"type": "timeout", "message": error_msg})
            state = add_error_to_state(state, self.name, error_msg)
            return await self._handle_timeout(state)
            
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            self.errors.append({"type": "exception", "message": error_msg, "traceback": traceback.format_exc()})
            state = add_error_to_state(state, self.name, error_msg, traceback.format_exc())
            return await self._handle_error(state, e)
    
    @abstractmethod
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """
        Agent-specific implementation.
        Subclasses must implement this method.
        
        Args:
            state: Current AgentState
            
        Returns:
            Updated AgentState
        """
        pass
    
    async def _handle_timeout(self, state: AgentState) -> AgentState:
        """
        Handle timeout gracefully.
        Can be overridden by subclasses.
        """
        # Add placeholder result to allow pipeline to continue
        state[f"{self.name}_result"] = {
            "status": "timeout",
            "error": f"Agent timed out after {self.timeout_seconds} seconds",
            "partial_data": None
        }
        return state
    
    async def _handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """
        Handle execution errors gracefully.
        Can be overridden by subclasses.
        """
        state[f"{self.name}_result"] = {
            "status": "error",
            "error": str(error),
            "traceback": traceback.format_exc()
        }
        return state
    
    def _update_latency_stats(self, new_latency: float):
        """Update average latency statistics."""
        if self.average_latency == 0:
            self.average_latency = new_latency
        else:
            # Exponential moving average
            self.average_latency = 0.9 * self.average_latency + 0.1 * new_latency
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        success_rate = (self.successful_executions / self.total_executions 
                       if self.total_executions > 0 else 0)
        
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": success_rate,
            "average_latency": self.average_latency,
            "timeout_seconds": self.timeout_seconds,
            "error_count": len(self.errors)
        }
    
    def validate_input(self, state: AgentState) -> bool:
        """
        Validate input state before execution.
        Can be overridden by subclasses.
        """
        # Basic validation - check if query exists
        if "query" not in state or not state["query"]:
            return False
        return True
    
    def validate_output(self, state: AgentState) -> bool:
        """
        Validate output state after execution.
        Can be overridden by subclasses.
        """
        # Basic validation - check if agent added its result
        return True
    
    def create_prompt_template(self, template: str) -> ChatPromptTemplate:
        """Helper to create prompt template."""
        return ChatPromptTemplate.from_template(template)
    
    def create_json_chain(self, prompt: ChatPromptTemplate, llm) -> RunnableSerializable:
        """Create a chain that outputs JSON."""
        return prompt | llm | JsonOutputParser()
    
    def create_text_chain(self, prompt: ChatPromptTemplate, llm) -> RunnableSerializable:
        """Create a chain that outputs text."""
        return prompt | llm | StrOutputParser()
    
    def reset_stats(self):
        """Reset agent statistics."""
        self.total_executions = 0
        self.successful_executions = 0
        self.average_latency = 0.0
        self.errors = []
        print(f"📊 Reset statistics for {self.name}")


class SequentialAgent(BaseAgent):
    """Agent that executes tasks sequentially."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = []
    
    def add_task(self, task_func, *args, **kwargs):
        """Add a task to be executed sequentially."""
        self.tasks.append((task_func, args, kwargs))
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Execute tasks sequentially."""
        for task_func, args, kwargs in self.tasks:
            try:
                state = await task_func(state, *args, **kwargs)
            except Exception as e:
                state = add_error_to_state(state, self.name, f"Task failed: {str(e)}")
                break
        return state


class ParallelAgent(BaseAgent):
    """Agent that executes tasks in parallel."""
    
    def __init__(self, *args, max_workers: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_workers = max_workers
        self.tasks = []
    
    def add_task(self, task_func, *args, **kwargs):
        """Add a task to be executed in parallel."""
        self.tasks.append((task_func, args, kwargs))
    
    async def _execute_impl(self, state: AgentState) -> AgentState:
        """Execute tasks in parallel."""
        import asyncio
        
        async def execute_task(task_func, args, kwargs):
            try:
                return await task_func(state, *args, **kwargs)
            except Exception as e:
                return add_error_to_state(state, self.name, f"Parallel task failed: {str(e)}")
        
        # Execute tasks in parallel
        tasks = [execute_task(func, args, kwargs) for func, args, kwargs in self.tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results (simple implementation - can be customized)
        for result in results:
            if not isinstance(result, Exception):
                # Merge state updates
                for key, value in result.items():
                    if key not in ["agent_timestamps", "execution_path"]:
                        state[key] = value
        
        return state


# Utility function to create agent instance
def create_agent(agent_class, config: Dict[str, Any] = None, **kwargs):
    """Helper function to create an agent instance."""
    try:
        agent = agent_class(config=config, **kwargs)
        print(f"✅ Created agent: {agent.name}")
        return agent
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None

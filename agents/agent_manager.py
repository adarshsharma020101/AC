 # agents/agent_manager.py

import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio

class AgentManager:
    """
    Coordinates interactions between multiple agents in the system.
    Manages agent lifecycle, communication, and orchestrates complex workflows.
    """
    
    def __init__(self):
        self.agents = {}
        self.logger = logging.getLogger(__name__)
        self.active_workflows = {}
        self.workflow_counter = 0
        
    def register_agent(self, agent_id: str, agent_instance: Any) -> None:
        """Register a new agent with the manager"""
        if agent_id in self.agents:
            self.logger.warning(f"Agent with ID {agent_id} already registered. Overwriting.")
        self.agents[agent_id] = agent_instance
        self.logger.info(f"Agent {agent_id} registered successfully")
        
    def get_agent(self, agent_id: str) -> Any:
        """Retrieve an agent by ID"""
        if agent_id not in self.agents:
            self.logger.error(f"Agent {agent_id} not found")
            return None
        return self.agents[agent_id]
    
    def list_agents(self) -> List[str]:
        """List all registered agent IDs"""
        return list(self.agents.keys())
    
    async def execute_workflow(self, workflow_steps: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Execute a sequence of agent operations as a workflow
        
        Args:
            workflow_steps: List of tuples containing (agent_id, params)
            
        Returns:
            Dictionary containing the results of the workflow execution
        """
        self.workflow_counter += 1
        workflow_id = f"workflow_{self.workflow_counter}"
        self.active_workflows[workflow_id] = {"status": "running", "steps": []}
        
        results = {}
        
        for step_num, (agent_id, params) in enumerate(workflow_steps):
            try:
                step_id = f"{workflow_id}_step_{step_num}"
                self.active_workflows[workflow_id]["steps"].append({
                    "step_id": step_id,
                    "agent_id": agent_id,
                    "status": "running"
                })
                
                agent = self.get_agent(agent_id)
                if not agent:
                    raise ValueError(f"Agent {agent_id} not found")
                
                # Pass any previous results to the agent
                params["previous_results"] = results
                
                # Execute agent operation
                self.logger.info(f"Executing {agent_id} as step {step_num} in {workflow_id}")
                if hasattr(agent, 'run_async'):
                    step_result = await agent.run_async(**params)
                else:
                    step_result = agent.run(**params)
                
                # Update results
                results[agent_id] = step_result
                
                # Update workflow status
                self.active_workflows[workflow_id]["steps"][-1]["status"] = "completed"
            
            except Exception as e:
                self.logger.error(f"Error in workflow {workflow_id}, step {step_num} ({agent_id}): {str(e)}")
                self.active_workflows[workflow_id]["steps"][-1]["status"] = "failed"
                self.active_workflows[workflow_id]["steps"][-1]["error"] = str(e)
                self.active_workflows[workflow_id]["status"] = "failed"
                raise
        
        self.active_workflows[workflow_id]["status"] = "completed"
        return results
    
    def execute_workflow_sync(self, workflow_steps: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Synchronous version of execute_workflow for compatibility"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.execute_workflow(workflow_steps))
        finally:
            loop.close()
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow"""
        if workflow_id not in self.active_workflows:
            self.logger.warning(f"Workflow {workflow_id} not found")
            return {"status": "not_found"}
        return self.active_workflows[workflow_id]

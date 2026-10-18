"""The OKF agent: a LangGraph graph that answers from the OKF MCP server, with Postgres memory.

    from okf_agent import OKFAgent
    agent = OKFAgent.start()
    print(agent.ask("What score did GPT-4 get on the bar exam?", thread_id="chat-1").answer)
"""

from .tracing import tracing_status                  # first: loads .env so LANGSMITH_* is seen
from .agent import AGENT_NAME, OKFAgent
from .config import AgentConfig
from .graph import build_graph, looks_like_followup
from .state import AgentReply, AgentState

__all__ = ["OKFAgent", "AgentConfig", "AgentReply", "AgentState", "build_graph", "looks_like_followup", "AGENT_NAME", "tracing_status"]

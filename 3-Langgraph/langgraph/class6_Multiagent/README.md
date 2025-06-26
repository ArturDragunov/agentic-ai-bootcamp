# LangGraph Multiagent Network Orchestration Tutorial

## Overview
This tutorial demonstrates **network orchestration** in LangGraph, where specialized agents collaborate by transferring control between each other using `Command` objects and tool-based handoffs.

## Key Concepts

### Network Orchestration
- Agents can delegate tasks to other specialized agents
- Uses `Command(goto="agent_name")` for explicit routing
- Tool-based handoffs enable seamless collaboration

### Implementation Pattern
```python
def agent_function(state: MessagesState) -> Command[Literal["next_agent", "__end__"]]:
  # Agent logic with tool binding
  ai_msg = llm.bind_tools([transfer_tool]).invoke(messages)
  
  if ai_msg.tool_calls:
    # Transfer to another agent
    return Command(goto="next_agent", update={"messages": [ai_msg, tool_msg]})
  
  # Continue or end
  return {"messages": [ai_msg]}
```

## Examples

### 1. Math Collaboration (Addition + Multiplication)
- **Addition Expert**: Handles addition, transfers to multiplication expert when needed
- **Multiplication Expert**: Handles multiplication, can transfer back to addition expert
- **Orchestration**: Bidirectional handoffs based on task requirements

### 2. Research + Visualization Pipeline
- **Researcher Agent**: Uses Tavily search tool for data gathering
- **Chart Generator Agent**: Uses Python REPL for data visualization
- **Orchestration**: Linear pipeline with "FINAL ANSWER" termination condition

## Core Components

### Agent Definition
```python
@tool
def transfer_to_agent():
  """Transfer control to specialized agent"""
  return

def agent_node(state: MessagesState) -> Command:
  system_prompt = "You are [specialization]. You can transfer to [other_agent]."
  messages = [{"role": "system", "content": system_prompt}] + state["messages"]
  
  ai_msg = llm.bind_tools([transfer_tool]).invoke(messages)
  # Handle tool calls and routing logic
```

### Graph Construction
```python
workflow = StateGraph(MessagesState)
workflow.add_node("agent1", agent1_function)
workflow.add_node("agent2", agent2_function)
workflow.add_edge(START, "agent1")
app = workflow.compile()
```

### Termination Logic
```python
def get_next_node(last_message: BaseMessage, goto: str):
  if "FINAL ANSWER" in last_message.content:
    return END
  return goto
```

## Key Features

- **Dynamic Routing**: Agents decide when to transfer control
- **Stateful Handoffs**: Complete message history preserved across transfers
- **Tool Integration**: Each agent has specialized tools (search, code execution)
- **Flexible Termination**: Agents can signal completion with "FINAL ANSWER"

## Setup Requirements
```python
# Required packages
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
```

## Usage Pattern
1. Define specialized agents with specific tools
2. Create transfer tools for inter-agent communication
3. Implement routing logic with `Command` objects
4. Build StateGraph with agent nodes
5. Execute with user query

This pattern enables complex workflows where multiple AI agents collaborate autonomously, each contributing their specialized capabilities to solve comprehensive tasks.
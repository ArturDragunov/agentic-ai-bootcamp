# Multi-Agent System with LangGraph

## Overview
This tutorial demonstrates building a multi-agent system using LangGraph that orchestrates specialized agents through a supervisor pattern. The system handles user requests by routing them to appropriate agents based on task requirements.

## Architecture

### Agents
- **Supervisor Agent**: Routes tasks to appropriate workers or finishes the conversation
- **Researcher Agent**: Handles internet searches using Tavily search tool
- **Coder Agent**: Executes Python code using a REPL environment

### Flow
1. User input → Supervisor Agent
2. Supervisor decides: `researcher`, `coder`, or `FINISH`
3. Chosen agent executes task
4. Agent returns to Supervisor
5. Process repeats until `FINISH`

## Key Components

### State Management
```python
class State(MessagesState):
  next: str
```
- Inherits from LangGraph's `MessagesState` (conversation history)
- Adds `next` field for routing logic

### Routing Logic
```python
class Router(TypedDict):
  next: Literal['researcher', 'coder', 'FINISH']
```
- Structured output for supervisor decisions
- Uses LLM with structured output for routing

### Agent Implementation
- **Supervisor**: Uses structured output to determine next agent
- **Research Agent**: `create_react_agent` with search tools
- **Coder Agent**: `create_react_agent` with Python REPL tool

## Tools Used
- **Tavily Search**: Internet research capabilities
- **Python REPL**: Code execution environment
- **ChatGroq**: LLM provider (DeepSeek model)

## Key Features
- Hierarchical agent orchestration
- Automatic routing based on task type
- Conversation history preservation
- Specialized agent capabilities
- Command-based state transitions
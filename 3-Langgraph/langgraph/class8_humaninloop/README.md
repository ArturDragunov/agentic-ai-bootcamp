# Human-in-the-Loop with LangGraph

## Overview
This tutorial demonstrates implementing human-in-the-loop workflows using LangGraph, allowing users to control and approve agent actions before execution.

## Key Concepts

### Two Implementation Approaches

#### 1. Custom Human-in-Loop
- Manual tool invocation with user prompts
- Custom `invoke_tool()` function with permission checks
- Direct user input via `input()` prompts

#### 2. LangGraph Built-in Human-in-Loop
- Uses `interrupt_before=["tools"]` parameter
- Pauses execution before tool calls
- Allows state inspection and modification

## Core Components

### State Management
```python
class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]
```
- Uses `operator.add` for message accumulation
- Preserves conversation history across interactions

### Memory and Persistence
```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
```

**Memory Types:**
- `MemorySaver`: In-memory (lost on restart)
- `FileSystemCheckpointer`: Persistent file storage
- `SQLiteCheckpointer`: Database storage

### Interruption Workflow
```python
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])
```

1. Graph pauses before tool execution
2. Human inspects proposed tool call via `get_state()`
3. Human approves/rejects or modifies state
4. Resume with `invoke(None, config)` or custom messages

## Key Features

### StateSnapshot Components
- **values**: Current state data (messages, variables)
- **next**: Upcoming nodes to execute
- **config**: Thread ID and checkpoint metadata  
- **metadata**: Execution details and step information
- **tasks**: Pending work items
- **interrupts**: Active pause points

### Human Control Options
1. **Approve/Reject**: Continue or stop tool execution
2. **State Modification**: Update messages before resuming
3. **Custom Responses**: Provide manual answers via `ToolMessage`

## Example Usage

```python
# Initial request - pauses before tool
response = app.invoke({"messages": [HumanMessage("GDP of China?")]}, config)

# Inspect state
snapshot = app.get_state(config)
tool_call = snapshot.values["messages"][-1].tool_calls[0]

# Option 1: Resume normally
response = app.invoke(None, config)

# Option 2: Provide custom response
app.update_state(config, {"messages": [
  ToolMessage(content="Custom answer", tool_call_id=tool_call["id"])
]})
```

## Thread Management
- Each conversation uses unique `thread_id`
- Persistent across sessions with proper checkpointer
- Enables multi-user scenarios with isolated states
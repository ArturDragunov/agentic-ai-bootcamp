# %%
print("all ok")

# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain_groq import ChatGroq

# %%
llm=ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# %%
llm.invoke("What is the capital of France?")

# %%
from langchain_core.tools import tool

# %%
from langchain_community.tools.tavily_search import TavilySearchResults

# %%
@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y

# %%
multiply.invoke({"x":2, "y":3}) # input to a tool should be a dictionary. String is allowed only if a tool takes single parameter as input

# %%
@tool
def search(query: str):
    """search the web for a query and return the results"""
    tavily=TavilySearchResults()
    result=tavily.invoke(query)
    return f"Result for {query} is: \n{result}"
    

# %%
print(search.invoke({"query":"What is the capital of France?"}))

# %%
tools = [multiply, search]

# %%
tools

# %%
llm_with_tools=llm.bind_tools(tools)

# %%
result=llm_with_tools.invoke("what is current gdp of india?")

# %%
result.content

# %%
result.tool_calls

# %%
result.tool_calls[0]["name"] # [0] to get out of list, and ["name"] to get the name of the tool

# %%
result.tool_calls[0]["args"]

# %%
type(result.tool_calls[0]["args"])

# %%
tool_mapping={tool.name:tool for tool in tools} # tool_mapping is a dictionary of tool names and tools

# %%
tool_mapping

# %%
tool_mapping["search"] # giving tool name as a key, we get the tool object as a value

# %%
#manually i am passing here
tool_mapping["search"].invoke({"query":"What is the capital of india?"})

# %%

tool_mapping[result.tool_calls[0]["name"]].invoke(result.tool_calls[0]["args"])
# tool_mapping[tool_name].invoke(tool_query)
# tool_mapping["search"].invoke({"query":"What is the capital of india?"})


# %%
from typing import TypedDict, Sequence, Annotated

# %%
import operator

# %%
from langchain_core.messages import BaseMessage

# %% [markdown]
# In this example, operator.add is a function from Python's operator module that performs addition (+).
# Here it's used as a reducer function in the Annotated type hint. When multiple values are assigned to the messages field, operator.add will concatenate/merge them together (since adding sequences concatenates them in Python).
# So if you have:
# 
# messages = [msg1, msg2]
# Then add messages = [msg3, msg4]
# 
# The reducer will do [msg1, msg2] + [msg3, msg4] = [msg1, msg2, msg3, msg4]
# This is commonly used in state management frameworks like LangGraph where you want to accumulate messages rather than replace them.

# %%
class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[Sequence[BaseMessage],operator.add]

# %%
# state={"messages":["hi","hello","how are you?"]} # messages will be appended in the same list

# %%
def invoke_model(state:AgentState):
    messages=state["messages"]
    question=messages[-1] # last (and the only existing) message in state is the question. There's no loop, so we don't need to provide entire conversation history.
    response=llm_with_tools.invoke(question)
    return {"messages":[response]} # state appended with the response
    

# %%
def router(state:AgentState):
    tool_calls=state["messages"][-1].tool_calls # last message comes from invoke_model
    if len(tool_calls)>0:
        return "tool" #key name
    else:
        return "end" #key name

# %%
def invoke_tool(state:AgentState):
    tool_details=state["messages"][-1].tool_calls # if we got here, we know for sure that the last message is a tool call
    
    if tool_details is None:
        return Exception("No tool calls found in the last message.")
    
    print(f"Seleted tool: {tool_details[0]['name']}") # we print tool name from tool details
    
    if tool_details[0]["name"]=="search":
        response=input(prompt=f"[yes/no] do you want to continue with this expensive web search")
        if response.lower()=="no":
            print("web search discarded by the user. exiting gracefully")
            raise Exception("Web search discarded by the user.")
            
    
    response=tool_mapping[tool_details[0]["name"]].invoke(tool_details[0]["args"]) # we invoke the search tool
    return {"messages":[response]} # once again we added a response to the state in a dictionary format
    

# %%
tools

# %% [markdown]
# for which tool money might be requied: search tool
# 
# should we take pemission from human(user) before proceding with the taviley tool call?

# %%
from langgraph.graph import StateGraph, START,END

# %%
graph=StateGraph(AgentState)

# %%
graph.add_node("ai_assistant", invoke_model)

# %% [markdown]
# ##### eariler we were using the tool node from list of tool
# ##### but now we have crate tool invoke(custom funtion) -> tool invoke is executed by a human and not LLM
# ##### why we are doing it: as a user if we want to take a authority to which i need to give permission for execution 

# %%
graph.add_node("tool", invoke_tool)
# now, we don't append tool into Node. We create custom logic for tool if we want to have human-in-loop -> Human decides whether they want to trigger the tool or not

# %%
graph.add_conditional_edges("ai_assistant",
                            router,
                            {
                                "tool":"tool", ##with the key tool which value is associated <tool>.
                                  # If LLM returns tool, we go to tool. otherwise, end. tool here is both the node name and the returned value from router
                                "end":END
                            }
                            )

# %%
graph.add_edge("tool", END) # from tool we always end the process

# %%
graph.set_entry_point("ai_assistant") # or you could write it as graph.add_edge(START, "ai_assistant")

# %%
app=graph.compile()

# %%

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

# %%
app.invoke({"messages":["What is the current gdp of the india?"]})

# %%
app.invoke({"messages":["What is the multiplication of 5 and 20?"]})

# %%
app.invoke({"messages":["What is the current weather in india delhi?"]})

# %%
app.invoke({"messages":["what is a latest news of bengaluru?"]})

# %%
app.invoke({"messages":["what is a latest news of delhi?"]})

# %% [markdown]
# 

# %% [markdown]
# ## Langgraph inbuilt human in loop

# %%
tools

# %%
from langgraph.prebuilt import ToolNode, tools_condition # in-built tool router for ReAct flow
tool_node=ToolNode(tools)

# %%
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# %%
llm_with_tools=llm.bind_tools(tools)

# %%
def ai_assistant(state:AgentState):
    response=llm_with_tools.invoke(state["messages"]) 
    # we are providing the whole messages state to the LLM because in ReAct LLM needs to see both the question and the tool output
    return {"messages":[response]}

# %%
from langgraph.checkpoint.memory import MemorySaver

# %%
memory=MemorySaver()

# %%
graph_builder=StateGraph(AgentState)

# %%
graph_builder.add_node("ai_assistant", ai_assistant)

# %%
graph_builder.add_node("tools", tool_node)

# %% [markdown]
# 

# %%
graph_builder.add_edge(START,"ai_assistant")

# %%
graph_builder.add_conditional_edges("ai_assistant",
                                    tools_condition # tools condition is a function that returns a dictionary with the key "tool" and the value is the tool to use
                                    # {
                                    #     "tool":"tool", # if the tool is selected, we go to the tool node
                                    #     "end":END # if the tool is not selected, we go to the end node
                                    # }
                                    )

# tools condition is the analogy to our custom router
# def router(state:AgentState):
    # tool_calls=state["messages"][-1].tool_calls # last message comes from invoke_model
    # if len(tool_calls)>0:
    #     return "tool" #key name
    # else:
    #     return "end" #key name                    
    # 
# graph_builder.add_conditional_edges("ai_assistant",
#                                     router,  
#                                      {
#                                          "tool":"tool", # if the tool is selected, we go to the tool node
#                                          "end":END # if the tool is not selected, we go to the end node
#                                      }
#                                     )                

# %%
graph_builder.add_edge("tools", "ai_assistant") # this edge creates a loop -> we provide a tool call to the ai_assistant node for summarization

# %% [markdown]
# # State vs Memory in LangGraph
# 
# ## Overview
# 
# **State** is what your LLM sees during execution - it's the current conversation context that gets passed between nodes. In your example, the `messages` field in `AgentState` contains the conversation history.
# 
# **Memory** is LangGraph's persistence mechanism that saves the entire graph execution state across different invocations.
# 
# ## When You Need Memory
# 
# ### 1. Human-in-the-loop workflows
# ```python
# app2=graph_builder.compile(checkpointer=memory,interrupt_before=["tools"])
# ```
# - Graph pauses before executing tools
# - Human can review and approve/reject tool calls
# - Memory preserves the exact state where it paused
# - You can resume with `app2.invoke(None, config)` or modify the state
# 
# ### 2. Long-running conversations
# - Memory persists across multiple `invoke()` calls
# - Each thread maintains its own state via `thread_id`
# - State survives server restarts, crashes, etc.
# 
# ### 3. Multi-step workflows with breaks
# - When you need to pause execution and resume later
# - When external systems need time to process
# - When you want to inspect/modify state mid-execution
# 
# ## Key Differences
# 
# | Aspect | State | Memory |
# |--------|-------|--------|
# | **Lifetime** | Only during single `invoke()` call | Across multiple `invoke()` calls |
# | **Persistence** | Gets reset each time | Survives program restarts |
# | **Visibility** | LLM can see during execution | Enables pausing/resuming workflows |
# | **Modification** | Changes lost after invoke | Allows state inspection and modification |
# 
# ## Example Use Case
# 
# ```python
# # First call - graph pauses before tools
# response = app2.invoke({"messages":[HumanMessage("What is the current gdp of the china?")]}, config=config)
# 
# # Human reviews tool call and decides to proceed
# response = app2.invoke(None, config)  # Resume from where it paused
# 
# # Or human can modify the state before resuming
# app2.update_state(config, {"messages": new_message})
# response = app2.invoke(None, config)
# ```
# 
# Without memory, you'd lose the entire conversation context and tool call state between these operations. Memory makes the graph "remember" where it was and what it was doing.

# %% [markdown]
# # Memory Storage Options in LangGraph
# 
# ## Overview
# 
# Yes, with memory you can save the entire conversation state to hard drive and restore it later. Here's how it works:
# 
# ## Memory Storage Options
# 
# ### 1. **MemorySaver** (In-Memory)
# ```python
# from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()
# ```
# - Stores state in RAM
# - Lost when program restarts
# - Fastest option
# 
# ### 2. **FileSystemCheckpointer** (Hard Drive)
# ```python
# from langgraph.checkpoint import FileSystemCheckpointer
# memory = FileSystemCheckpointer("path/to/checkpoints")
# ```
# - Saves state to files on hard drive
# - Survives program restarts
# - Persistent across sessions
# 
# ### 3. **SQLiteCheckpointer** (Database)
# ```python
# from langgraph.checkpoint import SQLiteCheckpointer
# memory = SQLiteCheckpointer("conversations.db")
# ```
# - Stores in SQLite database
# - Good for production use
# - Supports concurrent access
# 
# ## Example: File-Based Memory
# 
# ```python
# from langgraph.checkpoint import FileSystemCheckpointer
# 
# # Save to hard drive
# memory = FileSystemCheckpointer("./conversation_logs")
# 
# # Compile with persistent memory
# app = graph.compile(checkpointer=memory)
# 
# # Use thread_id to organize conversations
# config = {"configurable": {"thread_id": "user_123"}}
# 
# # First conversation - saves to disk
# response = app.invoke({"messages": ["Hello"]}, config=config)
# 
# # Later, even after restarting the program...
# # Load the same conversation
# response = app.invoke({"messages": ["How are you?"]}, config=config)
# # The agent remembers the previous "Hello" message!
# ```
# 
# ## What Gets Saved
# 
# The memory stores:
# - Complete conversation history
# - Current graph state
# - Tool call results
# - Any custom state variables
# - Execution metadata
# 
# ## Use Cases
# 
# 1. **Long-running conversations** across multiple sessions
# 2. **Debugging** - inspect saved states later
# 3. **Audit trails** - keep logs of all interactions
# 4. **Resume workflows** - pick up where you left off
# 5. **Multi-user systems** - each user gets their own persistent thread
# 
# So yes, you can essentially create "logs" that persist the entire conversation state and restore them later, just like saving a game and loading it back!

# %%
app2=graph_builder.compile(checkpointer=memory,interrupt_before=["tools"])
# interrupt_before=["tools"] means that we will interrupt the graph before the tools node
# pass this parameter to the compile function
# it will interrupt the graph before calling ANY TOOL, and we then can take the snapshot, see which tool was called and then decide what to do
# if search was suggested, then we can ask user if they want to continue with the search. Otherwise, we can just continue with the process

# %%

from IPython.display import Image, display
display(Image(app2.get_graph().draw_mermaid_png()))

# %% [markdown]
# ![alt text](<Screenshot 2025-07-04 173517.png>)

# %%
config={"configurable":{"thread_id":"1"}} # this is for memory. Every new conversation has a new thread_id

# %%
from langchain_core.messages import HumanMessage
response=app2.invoke({"messages":[HumanMessage("What is the current gdp of China?")]},config=config)

# %%
response

# %% [markdown]
# # Understanding StateSnapshot in LangGraph
# 
# ## What is StateSnapshot?
# 
# A `StateSnapshot` is a complete snapshot of your graph's execution state at a specific moment. It's like taking a "screenshot" of everything that's happening in your graph right now.
# 
# ## Breaking Down Your Snapshot
# 
# ### 1. **values** - Current State Data
# ```python
# values={'messages': [HumanMessage(...), AIMessage(...)]}
# ```
# - Contains your actual state data (messages, variables, etc.)
# - This is what your LLM sees and works with
# 
# ### 2. **next** - What's Coming Next
# ```python
# next=('tools',)
# ```
# - Shows which node(s) will execute next
# - In your case, the graph is paused before the "tools" node
# - This is why you can intercept and modify before tool execution
# 
# ### 3. **config** - Thread/Checkpoint Info
# ```python
# config={'configurable': {
#     'thread_id': '1', 
#     'checkpoint_ns': '', 
#     'checkpoint_id': '1f058eaa-a538-6d7d-8001-64b642e5df76'
# }}
# ```
# - Identifies which conversation thread this belongs to
# - Contains checkpoint ID for memory persistence
# 
# ### 4. **metadata** - Execution Details
# ```python
# metadata={
#     'source': 'loop',
#     'writes': {'ai_assistant': {'messages': [...]}},
#     'step': 1,
#     'parents': {},
#     'thread_id': '1'
# }
# ```
# - Shows which node wrote what data
# - Execution step number
# - Parent relationships
# 
# ### 5. **tasks** - Pending Work
# ```python
# tasks=(PregelTask(id='...', name='tools', path=('__pregel_pull', 'tools'), ...))
# ```
# - Shows what tasks are waiting to be executed
# - In your case, the "tools" task is pending
# 
# ### 6. **interrupts** - Pause Points
# ```python
# interrupts=()
# ```
# - Shows if execution was interrupted
# - Empty means no active interrupts
# 
# ## Why This Matters
# 
# 1. **Debugging**: You can inspect exactly what state the graph is in
# 2. **Human-in-the-loop**: You can see what tool is about to be called
# 3. **State Modification**: You can change the state before resuming
# 4. **Audit Trail**: You can track how the conversation evolved
# 
# ## Common Operations
# 
# ```python
# # Get current state
# snapshot = app2.get_state(config)
# 
# # Access messages
# messages = snapshot.values["messages"]
# 
# # Check what's next
# next_node = snapshot.next[0]  # 'tools'
# 
# # Modify state before resuming
# app2.update_state(config, {"messages": new_messages})
# 
# # Resume execution
# app2.invoke(None, config)
# ```
# 
# The snapshot is essentially your "save point" - it captures everything you need to understand where you are and what's about to happen next!

# %%
snapshot=app2.get_state(config) # complete configuration of your current state

# %%
snapshot

# %%
snapshot.next # from state we can get the detail of the next call -> should we proceed or should we end

# %%
last_message=snapshot.values["messages"][-1]

# %%
last_message

# %%
tool_details=last_message.tool_calls

# %%
tool_details

# %%
tool_details[0]["name"]

# %%
if tool_details[0]["name"]== "search":
    user_input=input(prompt=f"[yes/no] do you want to continue with {tool_details[0]['name']}?").lower()
    if user_input=="no":
        print("web tool discarded")
        raise Exception("Web tool discarded by the user.")
    else: # user_input=="yes"
        response=app2.invoke(None,config) # we proceed with the process
        print(response)
else:
    response=app2.invoke(None,config) # if tool was not search, we proceed with the process without asking for user's permission
    print(response)

# %%
{'messages': [HumanMessage(content='What is the current gdp of the china?', additional_kwargs={}, response_metadata={}), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'nmmwmjaqc', 'function': {'arguments': '{"query":"current GDP of China"}', 'name': 'search'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 149, 'prompt_tokens': 176, 'total_tokens': 325, 'completion_time': 0.609416334, 'prompt_time': 0.01099666, 'queue_time': 0.05363433, 'total_time': 0.620412994}, 'model_name': 'deepseek-r1-distill-llama-70b', 'system_fingerprint': 'fp_1bbe7845ec', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run--35ca1cc3-7893-40af-b047-adfd3d49ee23-0', tool_calls=[{'name': 'search', 'args': {'query': 'current GDP of China'}, 'id': 'nmmwmjaqc', 'type': 'tool_call'}], usage_metadata={'input_tokens': 176, 'output_tokens': 149, 'total_tokens': 325}), ToolMessage(content="Result for current GDP of China is: \n[{'title': 'China GDP - Worldometer', 'url': 'https://www.worldometers.info/gdp/china-gdp/', 'content': 'Nominal (current) Gross Domestic Product (GDP) of China is $17,794,800,000,000 (USD) as of 2023. · Real GDP (constant, inflation adjusted) of China reached', 'score': 0.92507464}, {'title': 'China: GDP at current prices 1985-2030 - Statista', 'url': 'https://www.statista.com/statistics/263770/gross-domestic-product-gdp-of-china/', 'content': 'In 2024, the gross domestic product (GDP) of China amounted to around 18.7 trillion U.S. dollars. In comparison to the GDP of the other BRIC', 'score': 0.87113565}, {'title': 'China GDP - Trading Economics', 'url': 'https://tradingeconomics.com/china/gdp', 'content': '##### Members\\n\\n##### \\n\\n# China GDP\\n\\n## The Gross Domestic Product (GDP) in China was worth 17794.78 billion US dollars in 2023, according to official data from the World Bank. The GDP value of China represents 16.88 percent of the world economy. source: World Bank [...] ### GDP in China is expected to reach 18542.00 USD Billion by the end of 2025, according to Trading Economics global macro models and analysts expectations. In the long-term, the China GDP is projected to trend around 19284.00 USD Billion in 2026 and 20094.00 USD Billion in 2027, according to our econometric models. [...] | Related | Last | Previous | Unit | Reference |\\n| --- | --- | --- | --- | --- |\\n| Full Year GDP Growth | 5.00 | 5.40 | percent | Dec 2024 |\\n| GDP | 17794.78 | 17881.78 | USD Billion | Dec 2023 |\\n| GDP Growth Rate YoY | 5.40 | 5.40 | percent | Mar 2025 |\\n| GDP Growth Rate | 1.20 | 1.60 | percent | Mar 2025 |\\n| GDP per Capita | 12175.20 | 11555.93 | USD | Dec 2023 |\\n| GDP per Capita PPP | 22137.60 | 21011.62 | USD | Dec 2023 |', 'score': 0.8690161}, {'title': 'China GDP | Historical Chart & Data - Macrotrends', 'url': 'https://www.macrotrends.net/global-metrics/countries/chn/china/gdp-gross-domestic-product', 'content': 'China GDP for 2023 was 17.795 trillion US dollars, a 0.49% decline from 2022. · China GDP for 2022 was 17.882 trillion US dollars, a 0.34% increase from 2021.', 'score': 0.8282873}, {'title': 'National Bureau of Statistics of China - National Data', 'url': 'https://data.stats.gov.cn/english/easyquery.htm?cn=B01', 'content': '4Q 2023 ; Gross Domestic Product, Current Quarter(100 million yuan), 318758.0, 373726.2, 341758.0, 328837.6 ; Gross Domestic Product, Accumulated(100 million yuan)', 'score': 0.81770587}]", name='search', tool_call_id='nmmwmjaqc'), AIMessage(content='The current GDP of China is approximately **$17.7 trillion USD** as of 2023, according to data from Worldometer and Trading Economics. This represents about **16.88% of the world economy**. For more detailed and up-to-date information, you can refer to sources like Worldometer, Statista, or Trading Economics.', additional_kwargs={}, response_metadata={'token_usage': {'completion_tokens': 324, 'prompt_tokens': 997, 'total_tokens': 1321, 'completion_time': 1.523101661, 'prompt_time': 0.084892204, 'queue_time': 0.057106376000000014, 'total_time': 1.607993865}, 'model_name': 'deepseek-r1-distill-llama-70b', 'system_fingerprint': 'fp_1bbe7845ec', 'finish_reason': 'stop', 'logprobs': None}, id='run--945b4539-2780-4c57-b5ac-3f3d89e15325-0', usage_metadata={'input_tokens': 997, 'output_tokens': 324, 'total_tokens': 1321})]}

# %%
app2.invoke(None,config) #none means resume the last state or resume the process -> we didn't wait for human input -> we resume the process with None

# %% [markdown]
# ### How to allow a user to provide input text into the conversation

# %%
response=app2.invoke({"messages":[HumanMessage("What is the current gdp of the japan?")]},config=config)

# %%
response # we stopped on AIMessage which prepared query for search tool: "current GDP of Japan"

# %%
snapshot=app2.get_state(config)

# %%
snapshot.next

# %%
last_message=snapshot.values["messages"][-1]

# %%
last_message.tool_calls

# %%
tool_call_id=last_message.tool_calls[0]["id"]

# %%
tool_call_id

# %%
from langchain_core.messages import AIMessage,ToolMessage

# %%
new_message=[
    ToolMessage(content="according to the latest data 4.1 trillion USD",tool_call_id=tool_call_id), # this is provided answer by user using UI
    # we need to save the answer under tool id
    AIMessage(content="GDP is 4.1 Trillion USD.")
    
]

# %% [markdown]
# ## Update state with a Human Message! We can write it as ToolMessage or HumanMessage or whatever Message we want.

# %%
app2.update_state(config, # conversation memory
                  {
                      "messages":new_message # we update state with new message - human message
                   }
                  )

# %%
app2.get_state(config).values["messages"][-1]

# %%
app2.invoke(None,config=config) # Now, tool message is taken by a human -> we skip tavily search

# %%
app2.invoke({"messages":[HumanMessage("What is the current gdp of the japan?")]},config=config)

# %% [markdown]
# Assignment for the multiagent:
# deadline till next friday: 11PM

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
parallization



# %%
print("all ok")

# %%
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# %% [markdown]
# ### Load the model

# %%
from langchain_groq import ChatGroq


# %%
llm=ChatGroq(model_name="deepseek-r1-distill-llama-70b")

# %%
llm.invoke("hi")

# %%
print(llm.invoke("hi").content)

# %%
# import operator
# from typing import List
# from langgraph.graph.message import add_messages
# from pydantic import BaseModel , Field
# from typing import TypedDict, Annotated, Sequence
# from langchain_core.messages import BaseMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.prebuilt import ToolNode

# %%
def call_model(state: MessagesState): # MessagesState is same as AgentState which we defined last lesson manually
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]} # new message combination was added to the state

# %%
# HumanMessage("hi how are you?")

# %%
# HumanMessage(["hi how are you?"])

# %% [markdown]
# ### this code is only for the testing

# %%

state={"messages":["hi hello how are you?"]}
call_model(state)

# %%
# from langchain_core.messages import AnyMessage
# # class MessagesState(TypedDict):
# #     messages: Annotated[list[AnyMessage], add_messages]


# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]

# %% [markdown]
# ## Design a simple workflow without tool calling/

# %% [markdown]
# #### ✅ What is StateGraph in LangGraph?
# 
# StateGraph is a class provided by LangGraph (a framework by LangChain) that allows you to:
# - Define a stateful computation graph, where each node is a step (e.g., a function, chain, or model).
# - Manage and persist state between nodes.
# - Handle branching, looping, and conditional logic.
# - Compose LLM-based workflows in a modular, structured way.
# 
# #### 🧠 Why and When to Use StateGraph
# - Model Stateful, Multi-Step Workflows: You want to pass state (a dictionary-like object) between steps. Each step can read/write/update the state.

# %%
workflow=StateGraph(MessagesState)

# %%
workflow.add_node("mybot",call_model) # mybot is the custom node name, which will be visible in diagram.
# call_model is the function that will be called when the node is executed

# %%
workflow.add_edge(START,"mybot")

# %%
workflow.add_edge("mybot",END) # here we say that mybot is the last node which finishes the workflow

# %%
app=workflow.compile()

# %%
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

# %%
input={"messages":["hi hello how are you?"]}

# %%
app.invoke(input)

# %%
# see output from individual nodes
for output in app.stream(input):
    for key,value in output.items():
        print(f"Output from {key} Node")
        print("_______")
        print(value)
        print("\n")

# %% [markdown]
# ### this is a workflow with tool calling

# %%
# from langchain_core.tools import tool
# tool decorator from langchain automatically makes a tool object out of a function. It has same methods as other langchain objects
@tool
def search(query:str):
    """this is my custom tool for searching a weather"""
    if "delhi" in query.lower():
        return "the temp is 45 degree and sunny"
    return "the temp is 25 degree and cloudy"

# %% [markdown]
# ## testing a tool

# %%
search.invoke("what is a temperature in kashmir?")

# %%
search.invoke("what is a temperature in delhi?")

# %%
search

# %%
llm.invoke("what is a weather in delhi?")

# %% [markdown]
# ### Binding a tool to the LLM
# 
# ### Special Note: use strong model for agentic workflow since opensource model might not give you the correct output

# %%
tools=[search]

# %%
llm_with_tool=llm.bind_tools(tools) 
# bind_tools takes a list of tools and returns a new LLM object with the tools bound to it.
# now, when you invoke the llm_with_tool, it will use the tools if they are needed.
# It has description of the tools in the prompt.

# %% [markdown]
# ### testig my llm_with_tool

# %%
response=llm_with_tool.invoke("what is a weather is delhi?")
# """this is my custom tool for searching a weather""" is the description of the tool.
# LLM has description of the tools in the prompt.
# It will redirect the query to the tool.

# %%
response

# %%
response.content

# %%
response.tool_calls

# %%
def call_model(state:MessagesState):
    question=state["messages"]
    print('call_model question', question)
    response=llm_with_tool.invoke(question)
    return {"messages":[response]}

# %% [markdown]
# ### Testing code

# %%
# input={"messages":["what is a weather in delhi?"]}
input={"messages":["how are you?"]}

# %%
response=call_model(input)

# %%
response["messages"][-1].content # depending on the input we either call LLM or a tool

# %%
response["messages"][-1].tool_calls

# %% [markdown]
# ### here my router function
# 
# #### now whatever will come from call_model router funtion will redirect this to the appropriate tool

# %%
def router_function(state:MessagesState):
    message=state["messages"]
    last_message=message[-1]
    if last_message.tool_calls: # if tool_calls exists, we need to call the tool
        return "tools"
    return END # if no tool_calls, LLM was already called and we can end the workflow
    

# %%
tools

# %%
#!!! MANDATORY STEP !!!
tool_node=ToolNode(tools) # we convert a tool into a node to be able to add it to the langgraph workflow


# %%
tool_node

# %%
workflow2=StateGraph(MessagesState)

# %% [markdown]
# workflow2.add_edge(START,"llmwithtool") is same as workflow.set_entry_point("Supervisor")
# you can use either one or another syntax! 
# 
# Langgraph knows which node is first because first node:
# - Has no inbound edges (i.e., nothing leads into it),
# - Is uniquely identifiable as the starting point.
# 
# IMPORTANT! LLM decides (it's triggered), whether it should use a provided tool or generate an answer on its own.
# 
# Each time the workflow runs:
# - The llm_with_tool.invoke() is called inside your call_model() function.
# - The LLM receives the user's query (from state["messages"]).
# - It decides on its own whether it needs to invoke a tool.
# 
# Important: This behavior depends on the prompt format and the model’s capability to suggest tool usage (enabled by .bind_tools(tools)).
# 
# Internally, LLM outputs special format based on the langchain prompt (this prompt instructs that llm has tools to use and it needs to output special format if it thinks we need to use tool!!)

# %%
workflow2.add_node("llmwithtool",call_model)

workflow2.add_node("mytools",tool_node) # tool itself

workflow2.add_edge(START,"llmwithtool") # from start we move the query to the llmwithtool node



# one conditional edge = one if-else statement
workflow2.add_conditional_edges("llmwithtool",
                                router_function, # if router outputs tools, we go to the tool node, otherwise we go to the END node
                                {"tools":"mytools", # mytools is the name of the tool node
                                 END:END}) 

# %%
app2=workflow2.compile()

# %%
from IPython.display import Image, display
display(Image(app2.get_graph().draw_mermaid_png()))

# %%
response=app2.invoke({"messages":["what is a weather in bengraluru?"]})

# %%
response["messages"][-1].content

# %%
app2.invoke({"messages":["what is a weather in delhi?"]})

# %% [markdown]
# ### use good resoning based model

# %%
app2.invoke({"messages":["hi how are you?"]})

# %%
workflow2.add_edge("mytools","llmwithtool") # new edge back from tool node to llmwithtool node (cycle)
# now, llm will receive the result of the tool call as input and evaluate it whether it is satisfied with the result or not.

# %%
app3=workflow2.compile()

# %% [markdown]
# This is a ReAct architecture which we built by simply adding one more edge from tool to LLM

# %%
from IPython.display import Image, display
display(Image(app3.get_graph().draw_mermaid_png()))

# %%
# .stream method allows you to get an output from each node
for output in app3.stream({"messages":["what is a weather in new delhi?"]}):
    for key,value in output.items():
        print(f"here is output from {key}") # node name
        print("_______")
        print(value) # value (answer) generated from the node
        print("\n")
    

# %%
"what is a weather in delhi can you tell me some good hotel for staying in north delhi"

# %%
from langgraph.checkpoint.memory import MemorySaver

# %%
memory=MemorySaver() # created a memory saver object

# %%
workflow3=StateGraph(MessagesState)

workflow3.add_node("llmwithtool",call_model)

workflow3.add_node("mytools",tool_node)

workflow3.add_edge(START,"llmwithtool")

workflow3.add_conditional_edges("llmwithtool",
                                router_function,
                                {"tools":"mytools",
                                 END:END})

workflow3.add_edge("mytools","llmwithtool")

# %%
app4=workflow3.compile(checkpointer=memory) # in-build memory to save the state of the workflow
# this way llm will understand the steps and the context better

# %%
from IPython.display import Image, display
display(Image(app4.get_graph().draw_mermaid_png()))

# %%
config={"configurable": {"thread_id": "1"}} # every chat is treated as a separate thread

# %% [markdown]
# using this configuration we will be able to save the entire configuration using this specific thread!!
# 
# so, if one user has multiple queries, we will remember it by using a single thread

# %% [markdown]
# #### Config in LangGraph
# 
# **Config** is how you pass runtime configuration to your graph execution. The key use case is **thread management** for conversational AI:
# 
# - `thread_id`: Creates separate conversation threads - each thread maintains its own state/memory
# - Different thread_ids = completely separate conversations
# - Same thread_id = continues the existing conversation with previous context
# 
# Other config uses:
# - Model parameters (temperature, etc.)
# - Custom configuration values your nodes need
# - Debugging settings
# 
# ## Stream Mode
# 
# **`stream_mode`** controls what data you get back during streaming execution:
# 
# - `"values"`: Returns the **full state** after each node executes
# - `"updates"`: Returns only the **changes/updates** each node made  
# - `"debug"`: Returns detailed execution info (useful for debugging)
# 
# In your example with `stream_mode="values"`, you'll get the complete graph state (including all accumulated messages) after each step, not just the incremental changes.
# 
# **Quick comparison:**
# - `"values"` → Full state snapshots
# - `"updates"` → Just what changed
# - `"debug"` → Execution metadata
# 
# Most common is `"values"` for chat applications since you want to see the full conversation state.

# %%
events=app4.stream(
    {"messages":["what is a weather in new delhi?"]},config=config,stream_mode="values"
    )

# %% [markdown]
# Use stream, when you want to see every step. Use invoke when you want to see the final output

# %%
for event in events:
    event["messages"][-1].pretty_print()
    

# %%
events=app4.stream(
    {"messages":["what is a weather in indore?"]},config=config,stream_mode="values"
    )

# %%
for event in events:
    event["messages"][-1].pretty_print()

# %%
config

# %%
memory.get(config)

# %%
events=app4.stream(
    {"messages":["in which city the temp was 25 degree?"]},config=config,stream_mode="values"
    )

# %%
for event in events:
    event["messages"][-1].pretty_print() # it's able to answer it because of the conversation memory. No tool is being called!!!

# %%
memory.get(config) # the entire conversation under config key was saved

# %% [markdown]
# good suggestion is to sustain last 5-10 conversations



# %%
print("all ok")

# %%
from dotenv import load_dotenv

# %%
load_dotenv()

# %%
from langchain_groq import ChatGroq

# %%
llm=ChatGroq(model="deepseek-r1-distill-llama-70b")

# %%
print(llm.invoke("What is the capital of France?").content)

# %%
print(llm.invoke("What is the capital of india tell me in detail?").content)

# %%
import os
from langchain_community.tools.tavily_search import TavilySearchResults
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
search_tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

# %%
search_tool.invoke("What is the capital of France?")

# %%
my_code = """
x=10
y=x+10
print(y)
"""

# %%
from langchain_experimental.utilities import PythonREPL

# %%
repl=PythonREPL()

# %%
repl.run(my_code) # provide code in string format

# %%
repl.invoke(my_code) # repl is not a langchain tool

# %%
from typing import Annotated # with this Annotated we can add metadata to the function parameters
from langchain_core.tools import tool
@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate output."]):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str

# %%
python_repl_tool.invoke("x=10\ny=x+10\nprint(y)") # now we can use repl in langchain syntax with invoke

# %%
python_repl_tool.invoke(my_code)

# %% [markdown]
# ### WE HAVE TWO SUB AGENT 
# 1. RESEARCHER- internet
# 2. CODER- executing the code

# %%
members=["researcher","coder"] # tavily is researcher and repl is a coder

# %%
members

# %%
options = members+["FINISH"]

# %%
options

# %%
from typing import Literal

# %%
from typing_extensions import TypedDict

# %% [markdown]
# ## There is no routing logic
# ### it is simply going to return the next candidate(next_agent)
# ### this next is containig the next candidate name

# %%
class Router(TypedDict):
    next: Literal['researcher', 'coder', 'FINISH']

# %%
from langgraph.graph import MessagesState,StateGraph,START, END

# %% [markdown]
# #### this is a messagesstate which we are loading from the langgraph(inbuilt message state)

# %%
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# %% [markdown]
# ##### this is how my state will be looking like for SUPERVISOR
# 
# Both Router and State have field next. But Router we need for structured output! And returned value will update the State in next

# %%
class State(MessagesState): # MessagesState is a base class for the state of the agent taken from langgraph. It already has a field for messages.
    # now we only need to add other fields we want. So, instead of creating from scratch, you can inherit from MessagesState.
    next:str
state={"messages": ["hi"], "next": "researcher"} # state will consist of messages (inherited from MessagesState) and next state to go to
#messages is a list of conversation history

# %%
system_prompt = f""""
You are a supervisor, tasked with managing a conversation between the following workers: {members}. 
Given the following user request, respond with the worker to act next. 
Each worker will perform a task and respond with their results and status. 
When finished, respond with FINISH.
"""

# %%
# system_prompt = f""""
# You are a supervisor, tasked with managing a conversation between the following workers: {members}. 
# Given the following user request, respond with the worker to act next. 
# Each worker will perform a task and respond with their results and status. 
# When finished, respond with FINISH.
# **Strict Guidelines:**
# if there is any common messages like hi, hello, how are you, greetings etc then,respond with FINISH.
# """

# %% [markdown]
# #### you can try out with this prompt also

# %%
system_prompt = f"""
You are a supervisor managing a task delegation system with the following workers: {members}.

Your job is to decide which worker should act next based on the user’s input.

Guidelines:
- Carefully read the user’s message.
- If the message clearly requires a specific action (e.g., search, compute, rewrite), assign it to the appropriate worker.
- If the message is general, conversational, or does **not** require any specific action, immediately respond with `FINISH`.
- Do **not** invent tasks or assign actions unless the message clearly demands it.

Each worker will return results after completing their task.
Once all necessary tasks are completed, end the flow by responding with `FINISH`.

Be strict — if the message is casual, rhetorical, or lacks a clear task, reply with `FINISH`.
"""


# %%
print(system_prompt)

# %%
messages = [{"role": "system", "content": system_prompt},] + state["messages"]

# %%
messages

# %% [markdown]
# [{'role': 'system',
#   'content': '"\nYou are a supervisor, tasked with managing a conversation between the following workers: [\'researcher\', \'coder\']. \nGiven the following user request, respond with the worker to act next. \nEach worker will perform a task and respond with their results and status. \nWhen finished, respond with FINISH.\n**Strict Guidelines:**\nif there is any common messages like hi, hello, how are you, greetings etc then,respond with FINISH.\n'},
#  'hi']

# %%
llm_with_structure_output=llm.with_structured_output(Router)

# %%
messages

# %% [markdown]
# Below is example of what input supervisor gets and what output it produces

# %%
llm_with_structure_output.invoke(messages)

# %% [markdown]
# #### This is my all three agents

# %%
from langgraph.types import Command

# %%
def supervisor_agent(state:State)->Command[Literal['researcher', 'coder', '__end__']]: # three options for finish
    
    # supervisor agent reads the conversation and decides which agent to go to next
    messages = [{"role": "system", "content": system_prompt},] + state["messages"]
    
    # llm will return a dictionary with the key "next" and the value is the next agent to go to or FINISH
    # {next: 'researcher'}
    llm_with_structure_output=llm.with_structured_output(Router)
    
    response=llm_with_structure_output.invoke(messages)
    
    #this is my response {'next': 'researcher'} -> caused by Router class
    # we then pass this "next" element to update "next" State
    
    #this is my next worker agent
    goto=response["next"]
    
    print("**********BELOW IS MY GOTO***************")
    
    print(goto)
    
    if goto == "FINISH": # if llm returns FINISH, we go to the end of the graph
        goto=END
    
    # class State(MessagesState):
    #   next:str
    # output of the state: state={"messages": ["hi"], "next": "researcher"}
    
    return Command(goto=goto, update={"next":goto}) # update next comes from State class

# %%
from langgraph.prebuilt import create_react_agent

# %%
from langchain_core.messages import AIMessage, HumanMessage

# %%
def research_agent(state: State) -> Command[Literal["supervisor"]]: # researchers ONLY LOOKS IN THE INTERNET, so available tool is search_tool
    
    research_agent = create_react_agent(llm, tools=[search_tool], prompt="You are a researcher. DO NOT do any math.")
    
    result=research_agent.invoke(state)
    
    return Command(
        update={
            "messages": [
                # research agent reads conversation (messages state) and appends its own message to it mentioning its name
                HumanMessage(content=result["messages"][-1].content, name="researcher") 
            ]
        },
        goto="supervisor", # NO MATTER WHAT THE RESULT IS, WE GO TO THE SUPERVISOR
    )
    

# %%
def coder_agent(state:State)->Command[Literal['supervisor']]: # using Command we do hand-off -> passing the state to the next agent
    code_agent=create_react_agent(llm,tools=[python_repl_tool], prompt="You are a coder. DO NOT do any research.")
    result=code_agent.invoke(state) # code agent reads the whole state (messages + authors) and returns a new state with the new message (python code)
    
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder") 
            ]
        },
        goto="supervisor",
    )
    

# %% [markdown]
# #### this is my orchestration flow with langgraph

# %%
graph=StateGraph(State)

# %%
graph.add_node("supervisor", supervisor_agent)

# %%
graph.add_node("researcher", research_agent)

# %%
graph.add_node("coder", coder_agent)

# %%
graph.add_edge(START, "supervisor") # the rest edges are added automatically using Command.goto

# %%
app=graph.compile()

# %%
from IPython.display import display,Image

# %%
display(Image(app.get_graph().draw_mermaid_png()))

# %% [markdown]
# subgraph is the concept of hierarchical graphs. You have multiple layers, and each layer is a subgraph. So, if you want to see outputs of 
# subgraphs, you need to mention that

# %% [markdown]
# ![alt text](photo_2025-07-03_14-11-28.jpg)

# %%
for s in app.stream({"messages": [("user", "What's the square root of 42?")]}, subgraphs=True):
    print(s)
    print("**********BELOW IS MY STATE***************")

# %%
result=app.invoke({"messages": [("user", "what is an efficent python code to get prime number?")]}, subgraphs=True)

# %%
result=app.ainvoke({"messages": [("user", "what is an efficent python code to get prime number?")]}, subgraphs=True)



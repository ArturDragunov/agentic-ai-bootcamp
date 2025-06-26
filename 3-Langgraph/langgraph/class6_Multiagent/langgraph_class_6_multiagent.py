# %%
from dotenv import load_dotenv

# %%
load_dotenv()

# %%
import os
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# %%
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=os.getenv("LLM_MODEL"))

# %%
llm.invoke("hi hello how are you?")

# %%
from langgraph.types import Command

# %%
from langgraph.prebuilt import create_react_agent # we can build react agent using the prebuilt function! no need to do everything manually

# %%
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     Name:str
#     age:int
#     DOB:int

# %%
def add_number(state):
    result=state["num1"]+state["num2"]
    print(f"addition is {result}")
    return Command(goto="multiply",update={"sum":result}) # update is a dictionary of key-value pairs that will be used to update the state
# goto is the name of the next node to go to

# Using Command we will be able to switch to multiple agents

# %%
state={"num1":10,"num2":20}

# %%
add_number(state)

# %% [markdown]
# ### Creating one dummy multiagent
# 
# it is for network/collab multiagent

# %%
from langchain_core.tools import tool

# %%
@tool
def transfer_to_multiplication_expert():
    """Ask multiplication agent for help"""
    return

# %%
@tool
def transfer_to_addition_expert():
    """Ask addition agent for help"""
    return

# %%
llm_with_tool=llm.bind_tools([transfer_to_addition_expert])

# %%
response=llm_with_tool.invoke("hi")

# %%
response.content

# %%
response.tool_calls

# %%
response=llm_with_tool.invoke("what is 2+2?")

# %%
response.content

# %%
response.tool_calls

# %%
system_prompt = (
        "You are an addition expert, you can ask the multiplication expert for help with multiplication."
        "Always do your portion of calculation before the handoff."
    )

# %%
messages = [{"role": "system", "content": system_prompt}] + ["can you tell me the addition of 2 and 2?"]

# %%
messages

# %% [markdown]
# [{'role': 'system',
#   'content': 'You are an addition expert, you can ask the multiplication expert for help with multiplication.Always do your portion of calculation before the handoff.'},
#  'can you tell me the addition of 2 and 2?']

# %%
from typing_extensions import Literal
from langgraph.graph import MessagesState,StateGraph, START,END
##Agent1
def additional_expert(state:MessagesState)-> Command[Literal["multiplication_expert", "__end__"]]: # we either go to multiplication_expert or end the conversation
    
    system_prompt = (
        "You are an addition expert, you can ask the multiplication expert for help with multiplication."
        "Always do your portion of calculation before the handoff." # handoff is delegation
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    
    ai_msg = llm.bind_tools([transfer_to_multiplication_expert]).invoke(messages)
    
    
    if len(ai_msg.tool_calls) > 0: # if tool was invoked, then we need to return the tool message
        tool_call_id = ai_msg.tool_calls[-1]["id"]

        #meta information 
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred to multiplication expert",
            "tool_call_id": tool_call_id,
        }
        
        return Command(
            goto="multiplication_expert", update={"messages": [ai_msg, tool_msg]} # if tool was triggered, then we go to multiplication_expert
        )
    return {"messages": [ai_msg]} # otherwise, return llm message

# %%
##Agent2
def multiplication_expert(state:MessagesState)-> Command[Literal["additional_expert", "__end__"]]: # we either go to additional_expert or end the conversation
    
    system_prompt = (
        "You are a multiplication expert, you can ask an addition expert for help with addition. "
        "Always do your portion of calculation before the handoff."
    )
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    ai_msg = llm.bind_tools([transfer_to_addition_expert]).invoke(messages)
    
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred to addition expert",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="additional_expert", update={"messages": [ai_msg, tool_msg]})
    return {"messages": [ai_msg]}

# %%
graph=StateGraph(MessagesState) # workflow=StateGraph(AgentState) -> we always start our stategraph with that

# %%
graph.add_node("additional_expert",additional_expert)
graph.add_node("multiplication_expert",multiplication_expert)

# %%
graph.add_edge(START, "additional_expert") # we start from the additional_expert node
# this dummy example is collaborative (network) agent

# %%
app=graph.compile() # we don't need to mention conditional edges here, because we already did that in the graph construction using Command

# %%
app

# %%
app.invoke({"messages":[("user","what's (3 + 5) * 12. Provide me the output")]})

# %% [markdown]
# ## With realtime tool - Networking Agent

# %%
import os
from langchain_community.tools.tavily_search import TavilySearchResults
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
search_tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

# %%
search_tool.invoke("who is a current pm of uk?")

# %%
from langchain_experimental.utilities import PythonREPL # given code in the form of string, it will execute it and return the result

# %%
repl=PythonREPL()

# %%
code = """
x = 5
y = x * 2
print(y)
"""

# %%
repl.run(code)

# %%
from typing import Annotated

# %%
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
    

# %%
python_repl_tool

# %%
print(python_repl_tool.invoke(code))

# %%
def make_system_prompt(instruction:str)->str:
    return  (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows when to stop."
        f"\n{instruction}"
    )

# %%
make_system_prompt("You can only do research. You are working with a chart generator colleague.")

# %% [markdown]
# "You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK, another assistant with different tools  will help where you left off. Execute what you can to make progress. If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop.\nYou can only do research. You are working with a chart generator colleague."

# %%
from langchain_core.messages import BaseMessage, HumanMessage
# BaseMessage is any user or AI message

# %%
def get_next_node(last_message:BaseMessage, goto:str):
    if "FINAL ANSWER" in last_message.content: # LLM sees the state and if it thinks that the work is done, it will return FINAL ANSWER which triggers the END node
        # Any agent decided the work is done
        return END
    return goto

# %%
#agent1 - itself is react agent
def research_node(state:MessagesState)->Command[Literal["chart_generator", END]]:
    research_agent=create_react_agent(
        llm,
        tools=[search_tool],
        prompt=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague."
    ), 
        )
    
    result=research_agent.invoke(state)
    goto=get_next_node(result["messages"][-1],"chart_generator") # go either to chart_generator or END
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="researcher") # we define a human message with researcher name and last result output
    return Command(update={"messages": result["messages"]},goto=goto) # update state and then go to the next node (chart_generator or END)

# %%
#agent2
def chart_node(state:MessagesState)-> Command[Literal["researcher", END]]:
    chart_agent=create_react_agent(
        llm, # this is reach Agent. First, we get AIMessage we chooses to use the Tool. Then we get ToolMessage. And then we get HumanMessage.
        tools=[python_repl_tool], # we use python repl tool to execute code
        prompt=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague."
    ),
        )
    result=chart_agent.invoke(state) # we pass entire state to the agent - all the messages. SO, we want the last message to be from Human for AI to work.  
    goto=get_next_node(result["messages"][-1],"researcher")
    result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="chart_generator") # we are simply rewriting the last message with the name of the agent
    return Command(update={"messages": result["messages"]},goto=goto)

# %%
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_edge(START, "researcher")
app = workflow.compile()

# %%
workflow.compile()

# %%
app.invoke({"messages": [("user","get the UK's GDP over the past 3 years, then make a line chart of it.Once you make the chart, finish.")],})
# 1 function execution (researcher or chart_generator) creates 3 messages: First is AIMessage, then ToolMessage (these 2 come from create_react_agent), and then HumanMessage.
# AI Agent visualized the chart itself



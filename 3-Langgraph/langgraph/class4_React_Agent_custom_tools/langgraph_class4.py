# %%
print("all ok")

# %%
from langchain_groq import ChatGroq
llm = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    temperature=0
)
response=llm.invoke("what is length of wall of china?")

# %%
response

# %%
response.content

# %% [markdown]
# ## this is my custom tools

# %%
from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The product of a and b.
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of a and b.
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """
    Divide two integers.

    Args:
        a (int): The numerator.
        b (int): The denominator (must not be 0).

    Returns:
        float: The result of division.
    """
    if b == 0:
        raise ValueError("Denominator cannot be zero.")
    return a / b

# %% [markdown]
# ## importing the inbuilt tool
# 
# Tavily give the raw result while DuckDuckGo gives the summary your search results

# %%
from langchain_community.tools import DuckDuckGoSearchRun
search=DuckDuckGoSearchRun()

# %%
search.invoke("what is the latest update on iphone17 release?")

# %%
# !pip install duckduckgo_search


# %%
tools=[multiply, add, divide, search]

# %%
llm_with_tools=llm.bind_tools(tools)

# %%
response=llm_with_tools.invoke("hi")

# %%
response.content

# %% [markdown]
# tool calls shows which tools were called for solution of the query. When you ask LLM a task which requires tool calling, you won't see any response as such. Instead you will see the tool calling. LLM structures your input in a proper way to match your tool, then it provides arguments into the tool. ***BUT THE TOOL IS NOT YET CALLED. YOU NEED TO ORCHESTRATE IT USING LANGGRAPH*** For a response, you need to forward the output of the tool calling back to the LLM.

# %%
response.tool_calls

# %%
response=llm_with_tools.invoke("what is 2+2?")

# %%
response.content

# %%
response.tool_calls

# %%
response=llm_with_tools.invoke("what is 10/2?")

# %%
response.content

# %%
response.tool_calls

# %%
response=llm_with_tools.invoke("what is a current age of the TATA Group?")

# %%
response

# %%
response.content

# %%
response.tool_calls

# %%
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# %%
SYSTEM_PROMPT="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs." # system prompt is based on the tasks LLM is supposed to perform



# %%
user_query=["tell me what is 2+2"]

# %%
[SYSTEM_PROMPT]+user_query

# %%
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState,StateGraph, END, START
def function_1(state:MessagesState):
    
    user_question=state["messages"]
    
    input_question = [SYSTEM_PROMPT]+user_question # user_question is a list of messages because of the MessagesState, so we need to add the system prompt to the beginning of the list
    
    response = llm_with_tools.invoke(input_question)
    
    return {
        "messages":[response] # langgraph automatically adds the response to the messages list
    }
    
    

# %%
builder=StateGraph(MessagesState)

# %%
builder.add_node("llm_decision_step",function_1) # you define a function that takes the query, triggers LLM and then LLM decides whether to use tools or to answer it directly

# %%
tools # the list of tools that the agent can use

# %%
from langgraph.prebuilt import ToolNode
builder.add_node("tools",ToolNode(tools)) # to use tools in the graph you need to convert it into toolnode

# %%
builder.add_edge(START,"llm_decision_step") # starting node

# %%
from langgraph.prebuilt import tools_condition 
builder.add_conditional_edges(
    "llm_decision_step",
    tools_condition,
)
# instead of describing each tool in conditional edge separately, 
# and defining a custom router function,
# we can use tools_condition which will automatically check if
#  the tool is available and return the appropriate edge

# %%
builder.add_edge("tools","llm_decision_step") # edge goes from tools to llm_decision_step providing the output of the tool to the llm

# %%
react_graph=builder.compile()

# %%
from IPython.display import Image, display
display(Image(react_graph.get_graph().draw_mermaid_png()))

# %%
# message=[HumanMessage(content="What is 2 times of narendramodi's age given the fact that today is 20th June 2025?")] # user message (query) should be in the form of a list
message=[HumanMessage(content="What is 2 times of narendramodi's age?")] # user message (query) should be in the form of a list

# %%
react_graph.invoke({"messages":message}) # sometimes it does calculations itself, sometimes it triggers multiplication tool

# %%
message = [HumanMessage(content="How much is the net worth of Elon Musk, and divide it by 2?")]

# %%
react_graph.invoke({"messages":message})

# %%
message = [HumanMessage(content="What is the speed of light in m/s and multiply it by 10?")]

# %%
response=react_graph.invoke({"messages":message})

# %%
response["messages"][-1].content

# %%
for m in response["messages"]:
    m.pretty_print() # inbuild pretty print function

# Answer or Final Answer is the way how LLM indicates that it has finished its task

# %%
import yfinance as yf

# %%
@tool # ALWAYS USE @tool decorator to define a tool
def get_stock_price(ticker:str)->str:
    """
    Fetches the previous closing price of a given stock ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA', 'NIFTY.BO').

    Returns:
        str: A message with the stock's previous closing price.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get('previousClose')
        if price is None:
            return f"Could not fetch price for ticker '{ticker}'."
        return f"The last closing price of {ticker.upper()} was ${price:.2f}."
    except Exception as e:
        return f"An error occurred while fetching stock data: {str(e)}"
     
    

# %%
get_stock_price.invoke("AAPL")

# %%
get_stock_price.invoke("TSLA")

# %%
tools

# %%
tools.append(get_stock_price)

# %%
llm_with_tools=llm.bind_tools(tools)

# %%
response=llm_with_tools.invoke("can you give me a latest stock price of adani greens?")

# %%
response.content # empty because LLM decided to use the tool instead of answering the question directly

# %%
response.tool_calls # LLM prepared the input for the tool

# %%
SYSTEM_PROMPT = SystemMessage(
    content="You are a helpful assistant tasked with using search, the yahoo finance tool and performing arithmetic on a set of inputs."
)
def function_1(state:MessagesState):
    
    user_question=state["messages"]
    
    input_question = [SYSTEM_PROMPT]+user_question
    
    response = llm_with_tools.invoke(input_question)
    
    return {
        "messages":[response]
    }

# %%
workflow = StateGraph(MessagesState) # we are creating a state graph with the MessagesState as input
workflow.add_node("llm_decision_step", function_1)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "llm_decision_step")
workflow.add_conditional_edges(
    "llm_decision_step",
    tools_condition,
)
workflow.add_edge("tools", "llm_decision_step") # cycle (output from the tool goes directly to the llm)
react_graph2 = workflow.compile()

# %%
display(Image(react_graph2.get_graph(xray=True).draw_mermaid_png()))

# %%
messages = [HumanMessage(content="add 1000 in the current stock price of Apple.")]
messages = react_graph2.invoke({"messages": messages})

# %%
for m in messages['messages']:
    m.pretty_print()

# %%
messages = [HumanMessage(content="can you give me 2 times of current stock price of Apple with the latest news of the Apple.")]
messages = react_graph2.invoke({"messages": messages})

# %%
for m in messages['messages']:
    m.pretty_print()

# %%
next class:
1. human in loop
2. agentic RAG
3. multiagent system(collabaration and supervisor)
4. some mislionoius concept of the langgraph: sub graph, paralle l execution etc

# %%
Assignment: # design a trip to any place worldwide
    
AI Travel Agent & Expense Planner(Purpose: Trip planning for any city worldwide with Realtime data.")

• Real-time weather information
• Top attractions and activities
• Hotel cost calculation (per day × total days)
• Currency conversion to user's native currency
• Complete itinerary generation
• Total expense calculation
• generate a summary of the entire output

user_input
  |
search attraction and activity
1. search attracation
2. search restaurant
3. search activity # what can I do in the city
4. search transportation
  |
search weather forcasting
1. get current weather
2. get weather forcast
  |
search hotel costs
1. search hotel
2. estimate the hotel cost
3. budget_range
  |
calculate total cost
1. add
2. multiply
3. calculated total cost
4. calcualte the daily budget # calculate the daily budget based on the total cost and the number of days
    | 
currency_conversion
1. get exchnage rate
2. convert currency
    | 
itinerary generation (a planned route or journey)
1. get day plan
2. crete full itinerary 
    |
create Trip Summary
    |
Retun complete travel plan

Note: if you know the OOPS then design this entire system using object and class in modular fashion. Each tool is a class with methods inside.


deadline is till next friday 9PM IST


 everyone you can submit the assignments in this form. MAke sure to have one GitHub link and put all the assignments there https://forms.gle/g8RZ4qx8yvNcih4B7    
    



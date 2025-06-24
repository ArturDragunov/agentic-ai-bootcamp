# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Config the model

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-05-20')
output=model.invoke("hi")
print(output.content)

# %% [markdown]
# ### Config the embedding model

# %%
import os
os.environ['HF_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# %%
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
len(embeddings.embed_query("hi"))

# %% [markdown]
# ## lets take a data embedd it and store in VDB

# %%
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
loader=DirectoryLoader("../data2",glob="./*.txt",loader_cls=TextLoader) 
# parent directory is ../data2, glob is the pattern to match the files,
#  loader_cls is the class to use to load the files

# %%
docs=loader.load()

# %%
docs

# %%
docs[0].page_content

# %%
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

# %%
new_docs=text_splitter.split_documents(documents=docs)

# %%
new_docs # chunks of data

# %%
doc_string=[doc.page_content for doc in new_docs] # we dropped the metadata

# %%
doc_string[:5]

# %%
len(doc_string)

# %%
db=Chroma.from_documents(new_docs,embeddings) # in-memory vector DB

# %%
retriever=db.as_retriever(search_kwargs={"k": 3})

# %%
retriever.invoke("industrial growth of usa?")

# %% [markdown]
# ## creation of pydantic class

# %%
import operator
from typing import List
from pydantic import BaseModel , Field
from langchain.prompts import PromptTemplate
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph,END

# %%
from typing import Literal
class TopicSelectionParser(BaseModel):
  Topic: Literal["USA", "Not Related"] = Field(description="selected topic")
  Reasoning: str = Field(description='Reasoning behind topic selection')
# I would do it vice versa -> first Reasoning then Topic. Less probability for halusination    


# %%
from langchain.output_parsers import PydanticOutputParser

# %%
parser=PydanticOutputParser(pydantic_object=TopicSelectionParser)

# %% [markdown]
# Pydantic's PydanticOutputParser.get_format_instructions() always returns JSON format instructions by default. It's not configurable to other formats.
# The reason is JSON because:
#
# Pydantic models are designed around JSON schema - they naturally serialize to/from JSON
#
#
# ...
#
# The output should be formatted as a valid JSON instance that conforms to the JSON schema below.
#
# As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"], "type": "object"}
# the object {"foo": ["bar", "baz"]} is a valid instance of the schema. Alternatively, the object {"foo": ["bar"]} is also a valid instance of the schema. Alternatively, the object {"foo": ["bar", "baz", "qux"]} is also a valid instance of the schema.
#
# Here is the output schema:
# {"properties": {"Topic": {"title": "Topic", "description": "selected topic", "type": "string"}, "Reasoning": {"title": "Reasoning", "description": "Reasoning behind topic selection", "type": "string"}}, "required": ["Topic", "Reasoning"], "type": "object"}

# %%
from pprint import pprint
pprint(parser.get_format_instructions())

# %%
'The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"Topic": {"description": "selected topic", "title": "Topic", "type": "string"}, "Reasoning": {"description": "Reasoning behind topic selection", "title": "Reasoning", "type": "string"}}, "required": ["Topic", "Reasoning"]}\n```'

# %% [markdown]
# ### this below agentstate is just for the explanation like how state works step by step

# %%
Agentstate={}

# %%
Agentstate["messages"]=[] # we created a key and assigned an empty list to it


# %%
Agentstate # this is how agent state will look like at the beginning

# %%
Agentstate["messages"].append("hi how are you?")

# %%
Agentstate

# %%
Agentstate["messages"].append("what are you doing?")

# %%
Agentstate

# %%
Agentstate["messages"].append("i hope everything fine")

# %%
Agentstate # the current agent state is three messages inside


# %%
Agentstate["messages"][-1]

# %%
Agentstate["messages"][0]


# %% [markdown]
# ### this agentstate class you need to inside the stategraph

# %% [markdown]
# AgentState is a TypedDict, not a regular class. TypedDict is just a type hint - it doesn't create a constructor.
#
# Thus, instead of state = AgentState(messages=state["messages"]) we call state = {"messages":"hi"}
#
# Why TypedDict instead of regular class?
#
# - Performance - no object overhead, just a dict
# - JSON serialization - dicts serialize naturally
# - LangGraph compatibility - expects dict-like state
# - Immutability patterns - easier to create new state versions
#
# LangGraph works best with TypedDict because it's designed around dict-based state management. The TypedDict gives you type safety without the class overhead.
# Think of AgentState as a "shape definition" rather than a "constructor blueprint."

# %%
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
# sequence of base messages and operator.add is a function that adds the messages to the sequence.
# key here is messages. Sequence[BaseMessage] is ['hi how are you?','what are you doing?','i hope everything fine']
# and operator.add is the same as list append

# It's the same as {"messages": [response.Topic]} -> it will be still of AgentState type


# %%
state={"messages":["hi"]}

# %%
state="hi"

# %%
# class TopicSelectionParser(BaseModel):
#     Topic:str=Field(description="selected topic")
#     Reasoning:str=Field(description='Reasoning behind topic selection')
# parser=PydanticOutputParser(pydantic_object=TopicSelectionParser)
# parser.get_format_instructions()
from langchain_core.messages import AIMessage
def function_1(state:AgentState): # state should be of AgentState type
    
    question=state["messages"][-1].content # we take latest message (recent) -> this is a BaseMessage object
    
    print("Question",question)
    
    template="""
    Your task is to classify the given user query into one of the following categories: [USA,Not Related]. 
    Only respond with the category name and nothing else.

    User query: {question} # You're inserting a BaseMessage object here
    {format_instructions}
    """
    
    prompt= PromptTemplate(
        template=template,
        input_variable=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()} # parser was defined outside of the function and it's a PydanticOutputParser
    ) # partial variables are those which enter the prompt but they are not part of the user query
    
    
    chain= prompt | model | parser 
    # we pass the question to the prompt, then the llm model takes the prompt and outputs the response,
		# and then this response is evaluated by parser to have the right format
    response = chain.invoke({"question":question})
    
    print("Parsed response:", response)
    
    return {"messages": [AIMessage(content=response.Topic)]} # now it's like a new message in the state

# %% [markdown]
# even though I'm not appending new message directly to state, langgraph does it automatically:
#
# The magic happens in LangGraph's state management, not in your function.
# Here's the flow:
#
# Your function returns a new dict:
# return {"messages": [AIMessage(content=response.Topic)]}
#
# LangGraph receives this and sees:
#
# Key: "messages"
# Current state: {"messages": [msg1, msg2, msg3]}
# New value: [AIMessage(content=response.Topic)]
#
#
# LangGraph looks at the AgentState definition:
# messages: Annotated[Sequence[BaseMessage], operator.add]
#
# 	                                          ^^^^^^^^^^^^
# 																						
# 	                                          This tells LangGraph HOW to merge
#
# LangGraph automatically applies operator.add:
#  LangGraph does this internally:
# state["messages"] = state["messages"] + [AIMessage(content=response.Topic)]
#
#
# You're not doing the appending - LangGraph is! The operator.add in the type annotation is an instruction to LangGraph's state management system.
#
# If you had used operator.replace instead, it would overwrite the entire messages list rather than append.
# This is LangGraph's "reducer" pattern - you return partial state updates, and LangGraph merges them according to the annotations.

# %%
from langchain_core.messages import HumanMessage

state = {"messages": [HumanMessage(content="what is a today weather?")]}

# %%
function_1(state)

# %%
state = {"messages": [HumanMessage(content="what is a GDP of usa??")]}

# %%
function_1(state)


# %%
# class TopicSelectionParser(BaseModel):
#     Topic:str=Field(description="selected topic")
#     Reasoning:str=Field(description='Reasoning behind topic selection')

# %%
def router(state:AgentState): 
# router is for conditional edges -> depending on the input,
# it will select the next appropriate node
    print("-> ROUTER ->")
    
    last_message=state["messages"][-1].content # Now extract content from AIMessage
    print("last_message:", last_message)
    
    if "usa" in last_message.lower():
        return "RAG Call"
    else:
        return "LLM Call"


# %% [markdown]
# workflow.add_conditional_edges(
#     "Supervisor",
#     router,
#     {
#         "RAG Call": "RAG"(function_2),
#         "LLM Call": "LLM"(function_3),
#     }
# )

# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
# RAG Function
def function_2(state:AgentState):
    print("-> RAG Call ->")
    
    question = state["messages"][-2].content 
    # question is the second last message. AI answer is the last message
    
    prompt=PromptTemplate(
        template="""You are an assistant for question-answering tasks.
                  Use the following pieces of retrieved context to answer the question.
									If you don't know the answer, just say that you don't know.
									Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
        
        input_variables=['context', 'question']
    )
# retriever is from RAG db. It searches for similar outputs in db based on question and returns the context    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return  {"messages": [AIMessage(content=result)]}


# %%
# LLM Function
def function_3(state:AgentState):
    print("-> LLM Call ->")
    question = state["messages"][-2].content
    
    # Normal LLM call
    complete_query = "Anwer the follow question with you knowledge of the real world. Following is the user question: " + question
    response = model.invoke(complete_query)
    return {"messages": [AIMessage(content=response.content)]}


# %%
from langgraph.graph import StateGraph,END

# %% [markdown]
# First, you create a workflow, and then you create nodes and edges of the workflow

# %%
workflow=StateGraph(AgentState)

# %%
workflow.add_node("Supervisor",function_1) # function 1 will be supervisor -> it selects RAG or LLM

# %%
workflow.add_node("RAG",function_2)

# %%
workflow.add_node("LLM",function_3)

# %%
workflow.set_entry_point("Supervisor") # start of the graph

# %% [markdown]
# Supervisor (function_1) → router() → RAG or LLM

# %%
# to connect nodes, we need edges. HERE we use conditional edges which go FROM router to RAG or LLM
# depending on the input, it will select the next appropriate node
# Router returns a string (it's an if/else condition). # depending on string, we either use RAG or LLM
# We connect Supervisor to router and depending on router output we trigger RAG or LLM
workflow.add_conditional_edges(
    "Supervisor",  # FROM this node
    router,        # Use this function to decide
    {              # Map router output to next nodes
        "RAG Call": "RAG",  # If router returns "RAG Call", go to RAG node
        "LLM Call": "LLM",  # If router returns "LLM Call", go to LLM node
    }
)
# Router is the decision logic, not a processing step.

# %%
workflow.add_edge("RAG",END) # END is a special node that indicates the end of the workflow.
workflow.add_edge("LLM",END) # from langgraph.graph import END

# %%
app = workflow.compile()

# %%
workflow.compile()


# %%
from langchain_core.messages import HumanMessage

state = {"messages": [HumanMessage(content="what is the weather today")]}
# state = AgentState(messages=["hi"]) # this is wrong!!! It won't work. AgentState is a TypedDict, not a regular class. TypedDict is just a type hint - it doesn't create a constructor.

# %%
app.invoke(state)

# %%
state = {"messages": [HumanMessage(content="what is a gdp of usa?")]}

# %%
app.invoke(state)

# %%
state = {"messages": [HumanMessage(content="can you tell me the industrial growth of world's most powerful economy?")]}

# %%
state = {"messages": [HumanMessage(content="can you tell me the industrial growth of world's poor economy?")]}

# %%
result=app.invoke(state)

# %%
result["messages"][-1]

# %%

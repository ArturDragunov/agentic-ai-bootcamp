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

# %% [markdown]
# ### Config the embedding model

# %%
import os
os.environ['HF_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# %% [markdown]
# ## lets take a data embedd it and store in VDB

# %%
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
loader=DirectoryLoader("../data2",glob="./*.txt",loader_cls=TextLoader) 
docs=loader.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
new_docs=text_splitter.split_documents(documents=docs)
doc_string=[doc.page_content for doc in new_docs] # we dropped the metadata


# %%
db=Chroma.from_documents(new_docs,embeddings) # in-memory vector DB

# %%
retriever=db.as_retriever(search_kwargs={"k": 3})

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
from dotenv import load_dotenv
load_dotenv()

# %%
from typing import Literal
class TopicSelectionParser(BaseModel):
  # to decrease halucination, first do reasoning and then topic
  Reasoning: str = Field(description='Reasoning behind topic selection')
  Topic: Literal["USA", "Web", "Not Related"] = Field(description="selected topic")


# %%
from langchain.output_parsers import PydanticOutputParser
parser=PydanticOutputParser(pydantic_object=TopicSelectionParser)


# %%
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# %%
from langchain_core.messages import AIMessage
def function_1(state:AgentState): # state should be of AgentState type
    
    question=state["messages"][-1].content # we take latest message (recent) -> this is a BaseMessage object
    
    print("Question",question)
    
    template="""
    Your task is to classify the given user query into one of the following categories: [USA, Web, Not Related]. 
    Only respond with the category name and nothing else. Respond USA if the query is about the US economy 
    (available period is Y2024 and future overview until Y2030),
    respond Web if the query requires to look for the latest information in the internet,
    respond Not Related if the query is not related to USA or Web and you can answer it using your knowledge.

    User query: {question}
    {format_instructions}
    """
    
    prompt= PromptTemplate(
        template=template,
        input_variable=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()} # parser was defined outside of the function and it's a PydanticOutputParser
    ) # partial variables are those which enter the prompt but they are not part of the chain invoke
    
    
    chain= prompt | model | parser 
    # we pass the question to the prompt, then the llm model takes the prompt and outputs the response,
		# and then this response is evaluated by parser to have the right format
    response = chain.invoke({"question":question})
    
    print("Parsed response:", response)
    
    return {"messages": [AIMessage(content=response.Topic)]} # now it's like a new message in the state


# %%
def router(state:AgentState): 
# router is for conditional edges -> depending on the input,
# it will select the next appropriate node
    print("-> ROUTER ->")
    
    last_message=state["messages"][-1].content # Now extract content from AIMessage
    print("last_message:", last_message)
    
    if "usa" in last_message.lower():
        return "RAG Call"
    elif "web" in last_message.lower():
        return "Web Call"
    else:
        return "LLM Call"


# %%
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# %%
# RAG Function
def function_2(state:AgentState):
    print("-> RAG Call ->")
    
    question = state["messages"][0].content 
    
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
    question = state["messages"][0].content
    
    # Normal LLM call
    complete_query = "Anwer the follow question with you knowledge of the real world. Following is the user question: " + question
    response = model.invoke(complete_query)
    return {"messages": [AIMessage(content=response.content)]}


# %%
from langchain_community.tools.tavily_search import TavilySearchResults
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
tool=TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
def format_tavily_results(results):
  return "\n\n".join([
    f"Title: {result['title']}\nContent: {result['content']}" 
    for result in results
  ])
# Web Call Function
def function_4(state:AgentState):
	print("-> Web Call ->")
	question = state["messages"][0].content
	web_search = format_tavily_results(tool.invoke({"query": question, "max_results": 5}))
	prompt=PromptTemplate(
	template="""You are an assistant for question-answering tasks.
						Use the following pieces of the latest fetched web news to answer the question.
						If you don't know the answer, just say that you don't know.
						Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",

	input_variables=['context', 'question'])
	# Normal LLM call
	chain = prompt | model
	response = chain.invoke({"question": question, "context": web_search})
	return {"messages": [AIMessage(content=response.content)]}


# %%
class ValidationParser(BaseModel):
  # to decrease halucination, first do reasoning and then topic
  Reasoning: str = Field(description='Reasoning behind validation output')
  Validation: Literal["Pass", "Fail"] = Field(description="validation based on question and answer")

validation_parser=PydanticOutputParser(pydantic_object=ValidationParser)


# %%
# Validation Function
def function_5(state:AgentState):
	print("-> Validation Call ->")
	question = state["messages"][0].content
	answer = state["messages"][-1].content
	# first is question, then supervisor output, then answer
	prompt=PromptTemplate(
	template="""You are a validator of a question-answering task.
Your task is to validate the answer based on the given question and to
classify it into one of the following categories: [Pass, Fail]. 
Classify Pass if:
 - the answer is logically correct and the question is related to the answer.
Classify Fail if:
 - the answer does not logically match the question.
Only respond with the category name and nothing else.

{format_instructions}
\nQuestion: {question} \nAnswer: {answer} \nClassification:""",
	input_variables=['answer', 'question'],
	partial_variables={"format_instructions": validation_parser.get_format_instructions()})
	chain = prompt | model | validation_parser
	response = chain.invoke({'question': question, 'answer': answer})
	return {"messages": [AIMessage(content=response.Validation)]}


# %%
def validation_router(state: AgentState):
  print("-> VALIDATION ROUTER ->")
  
  last_message = state["messages"][-1].content
  print("Validation result:", last_message)
  
  if "pass" in last_message.lower():
    return "END"
  else:
    return "SUPERVISOR"  # Go back to supervisor for retry


# %%
from langgraph.graph import StateGraph,END

# %%
workflow=StateGraph(AgentState) # we tell langgraph that we are dealing with AgentState
workflow.add_node("Supervisor",function_1)
workflow.add_node("RAG",function_2)
workflow.add_node("LLM",function_3)
workflow.add_node("Web",function_4)
workflow.add_node("Validation",function_5)

# %%
# Connect worker nodes to Validation
workflow.add_edge("RAG", "Validation")
workflow.add_edge("LLM", "Validation") 
workflow.add_edge("Web", "Validation")

# %%
workflow.set_entry_point("Supervisor") # start of the graph

# %% [markdown]
# one conditional edge -> one if else function

# %%
# First conditional edge: Supervisor to workers
workflow.add_conditional_edges(
  "Supervisor",  # FROM this node
  router,        # Use this function to decide
  {              # Map router output to next nodes
    "RAG Call": "RAG",  # If router returns "RAG Call", go to RAG node
    "LLM Call": "LLM",  # If router returns "LLM Call", go to LLM node
    "Web Call": "Web",  # If router returns "Web Call", go to Web node
  }
)

# Second conditional edge: Validation to END or back to Supervisor
workflow.add_conditional_edges(
  "Validation",
  validation_router,
  {
    "END": END,           # If validation passes, end workflow
    "SUPERVISOR": "Supervisor"  # If validation fails, go back to supervisor
  }
)

# %%
app = workflow.compile()

# %%
workflow.compile()


# %%
# Print the actual graph structure
from pprint import pprint
pprint(app.get_graph().edges)

# %%
from langchain_core.messages import HumanMessage

state = {"messages": [HumanMessage(content="what is the weather today")]}

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

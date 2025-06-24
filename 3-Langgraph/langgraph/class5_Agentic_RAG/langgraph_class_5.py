# %%
from dotenv import load_dotenv

# %%
load_dotenv()

# %% [markdown]
# # Step 1 - Data Ingest

# %%
import os
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# %%
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model=os.getenv("LLM_MODEL"))

# %%
llm.invoke("hello how are you my friend?")

# %%
from langchain_openai import OpenAIEmbeddings

# %%
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)


# %%
embeddings.embed_query("hello how are you my friend?")

# %%
len(embeddings.embed_query("hello how are you my firend?"))

# %%
url= "https://lilianweng.github.io/posts/2023-06-23-agent/" 

# %%
from langchain_community.document_loaders import WebBaseLoader

# %%
web_loader=WebBaseLoader(url)

# %%
data=web_loader.load()

# %%
data

# %%
data[0].metadata

# %%
len(data[0].metadata["description"])

# %%
data[0].metadata["description"]

# %%
data[0].page_content

# %%
len(data[0].page_content)

# %%
urls=[
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

# %%
docs=[WebBaseLoader(url).load() for url in urls]

# %%
docs # we loaded data from two urls. We can have multiple docs like that.
# docs is a list of lists. Each list is a document. We need to preprocess it.

# %%
docs_list=[item for sublist in docs for item in sublist] # flatten the list of lists into a single list to parse it further into split_documents

# %%
docs_list

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# %%
text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100,chunk_overlap=25) 
# This line creates a text splitter that:
# Uses OpenAI's tiktoken tokenizer to determine how many tokens are in each chunk (instead of using character counts).
# Splits text into chunks of ~100 tokens, not characters.
# Ensures 25 tokens of overlap between each chunk (so important context is preserved across chunks).

# Why is this useful?
# LLMs process input in tokens, not characters. Token-based splitting ensures that:
# You don’t accidentally cut off semantic meaning mid-token.
# This is especially important when using OpenAI models, since token count determines cost and performance.

# %%
doc_splits=text_splitter.split_documents(docs_list)

# %%
doc_splits

# %%
from langchain_community.vectorstores import Chroma

# %%
vectorstore=Chroma.from_documents(
    documents=doc_splits, # we create an in-memory vector store (we first split the documents into chunks and then we embed them)
    collection_name="rag-chrome", # we name the collection
    embedding=embeddings # openai embeddings
)

# %% [markdown]
# # Step 1 - Data Retriever 
# Get input query, embed it and look for similarities in vector DB

# %%
retriever=vectorstore.as_retriever()
retriever.invoke("what is agent and prompt?")

# %%
from langchain.tools.retriever import create_retriever_tool # you can create a @tool by wrapping retriever into a function OR you can use the function directly
retriever_tool=create_retriever_tool(
    retriever,
    "retriever_blog_post",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.You are a specialized assistant. Use the 'retriever_tool' **only** when the query explicitly relates to LangChain blog data. For all other queries, respond directly without using any tool. For simple queries like 'hi', 'hello', or 'how are you', provide a normal response.",
    )

# %%
tools=[retriever_tool]

# %%
from langgraph.prebuilt import ToolNode
retriever_node=ToolNode(tools) # we always need to wrap tools in a ToolNode

# %%
from typing import Annotated,Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# %%
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# %%
from langchain import hub
hub.pull("rlm/rag-prompt").pretty_print() # we pull the prompt from the hub

# %% [markdown]
# More detailed LLM_Decision_Maker BUT we can use a simple one with the same output

# %%
# def LLM_Decision_Maker(state:AgentState):
#     print("----CALL LLM_DECISION_MAKER----")
#     message=state["messages"]
    
#     llm_with_tool=llm.bind_tools(tools)
#     if len(message)>1: # then we already did one cycle and we came back for query rewriting
#         last_message=message[-1] # we take last message here, because last message IS REWRITTEN QUERY from rewrite function!!!
#         question=last_message.content
        
#         prompt=PromptTemplate(
#         template="""You are a helpful assistant whatever question has been asked to find out that in the given question and answer.
#                         Here is the question:{question}
#                         """,
#                         input_variables=["question"]
#                         )
#         chain=prompt | llm_with_tool
#         response=chain.invoke({"question":question})
#         return {"messages":[response]}
#     else: # first cycle of the workflow - we just started the conversation
#         response=llm_with_tool.invoke(message) # based on the query, LLM has tools and it decides if it needs to use them or not
#         return {"messages":[response]}

# %%
from langchain_core.prompts import PromptTemplate
def LLM_Decision_Maker(state:AgentState):
    print("----CALL LLM_DECISION_MAKE----")
    llm_with_tool=llm.bind_tools(tools)
    message=state["messages"]
    last_message=message[-1]
    question=last_message.content
    response=llm_with_tool.invoke(question)
    return {"messages":[response]}

# %%
from pydantic import BaseModel, Field
from typing import Literal
class grade(BaseModel): # pydantic class
    binary_score: Literal["yes", "no"] = Field(description="Relevance score 'yes' or 'no'") # Field is the place where you write down description

# %%
#we use it for type of hinting
def grade_documents(state:AgentState)->Literal["generator", "rewriter"]: # return either one or another 
    print("----CALLING GRADE FOR CHECKING RELEVANCY----")
    llm_with_structure_op=llm.with_structured_output(grade) # with_structured_output - Model wrapper that returns outputs formatted to match the given schema.
    
    prompt=PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
                    Here is the document: {context}
                    Here is the user’s question: {question}
                    If the document talks about or contains information related to the user’s question, mark it as relevant. 
                    Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
                    input_variables=["context", "question"]
                    )
     
    chain=prompt|llm_with_structure_op
     
     
    message=state['messages']
    
    last_message = message[-1]
    
    question = message[0].content # initial user query
    
    docs = last_message.content # retrieved docs which are in the last message state (previous step was RAG)
    
    scored_result=chain.invoke({"question": question, "context": docs})
    
    score=scored_result.binary_score # we get the score from the pydantic class - we defined it inside the grade class
    # we know for sure that we get output in this format BECAUSE we told llm to return structured output
     
    if score=="yes":
        print("----DECISION: DOCS ARE RELEVANT----")
        return "generator" # these exact return names will be used as references later in the conditional edge construction
    else:
        print("----DECISION: DOCS ARE NOT RELEVANT----")
        return "rewriter"

# %%
def generate(state:AgentState):
  print("----RAG OUTPUT GENERATE----")
  
  message=state["messages"]
  question=message[0].content # user question
  
  last_message = message[-1] # these are extracted documents from RAG
  docs = last_message.content # we get here ONLY if grade_documents returned "generator" (said YES)
  
  prompt=hub.pull("rlm/rag-prompt")
  
  rag_chain=prompt | llm
  
  response=rag_chain.invoke({"context": docs, "question": question})
  
  print(f"this is my response:{response}")
  
  return {"messages": [response]}

# %%
from langchain_core.messages import HumanMessage
def rewrite(state:AgentState):
    print("----TRANSFORM QUERY----")
    message=state["messages"]
    
    question=message[0].content # first message - user query
    
    input= [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning. 
                    Here is the initial question: {question} 
                    Formulate an improved question: """)
       ]

    response=llm.invoke(input)
    
    return {"messages": [response]} # HERE WE RETURN THE REWRITTEN QUERY - and this query already goes to the very beginning of the workflow
    

# %%
from langgraph.graph import END, StateGraph, START
workflow=StateGraph(AgentState)
workflow.add_node("LLM Decision Maker",LLM_Decision_Maker)
workflow.add_node("Vector Retriever",retriever_node) # embed user query and find relevant docs
workflow.add_node("Output Generator",generate) # generate answer based on relevant docs
workflow.add_node("Query Rewriter",rewrite) # rewrite query to be more specific

# %%
from langgraph.prebuilt import tools_condition
workflow.add_edge(START,"LLM Decision Maker")
workflow.add_conditional_edges("LLM Decision Maker",
                               tools_condition,
                               {"tools":"Vector Retriever",
                                END:END
                                })
# tools_condition - inbuild tool condition instead of your custom router function
# it either ends the conversation or goes to the vector retriever node
workflow.add_conditional_edges("Vector Retriever",
                               grade_documents,
                               {"generator":"Output Generator", # "generator" and "rewriter" are outputs from the grade_documents function
                                "rewriter":"Query Rewriter"
                                })
# here, if tools_condition decided to go to Vector Retriever, it will choose either to 
# generate the output OR to rewrite the query
workflow.add_edge("Output Generator",END)
# if we got output generator, then we are done
workflow.add_edge("Query Rewriter","LLM Decision Maker")
# if we got query rewriter, then we go to the very first node
app=workflow.compile()

# %%
app

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
app.invoke({"messages":["what is LLM Powered Autonomous Agents defined in LangChain blog?"]})
# with complex queries it can even split the query into multiple retrievals based on the topics asked.

# %%
app.invoke({"messages":["hi how are you gpt?"]})

# %%
app.invoke({"messages":["what is a capital of india?"]})

# %%
question="can you explain me what is a task decomposition and why Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks?"
app.invoke({"messages":[question]})

# %% [markdown]
# # Homework assignment

# %% [markdown]
# 1. take multiple data sources and complex data sources
# 2. strict your prompting in every node based on your desired output
# 3. check question and generated docuemnt(k=keep more than 5 with strict prompting according to your usecase)

# %% [markdown]
# what is a main aim of this agentic rag?
# answer: to provide more security and more validation and more determenstic flow to the RAG

# %% [markdown]
# ## As a homework now try out this one: Corrective RAG (Type of the Agentic RAG)
# 
# in this architecutre you are correcting the infomation from any other data source if retriever is not able to generate a correct output(in this architecture the external data source is web search)

# %% [markdown]
# ![image.png](attachment:image.png)



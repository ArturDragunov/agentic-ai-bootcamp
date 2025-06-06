#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from dotenv import load_dotenv
load_dotenv()


# In[ ]:


os.getenv("LANGCHAIN_PROJECT")


# In[ ]:


os.getenv("LANGCHAIN_API_KEY")


# In[4]:


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY") # keep these exact names as they would be looked for in langchain
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

## Langsmith Tracking And Tracing
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true" # this is for langsmith tracing


# In[ ]:


from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-4.1-nano-2025-04-14") # cost optimized model for real time applications
# llm=ChatOpenAI(model="gpt-4.1-mini-2025-04-14") # for more complex tasks
print(llm)



# In[ ]:


result=llm.invoke("What is Agentic AI")
print(result)


# In[ ]:


print(result.content)


# In[ ]:


from langchain_groq import ChatGroq
model=ChatGroq(model="qwen-qwq-32b") # reasoning model - we see it chain of thought
model.invoke("Hi My name is Krish").content


#  LangChain internally handles string formatting using the Jinja-like templating system. You’re not building the string yourself — you're passing a template object with variable placeholders. LangChain will later inject the value of "input" when you invoke the chain:
# 
#  ("user","{input}") -> This is just a tuple of two strings — plain Python. LangChain stores this structure inside a ChatPromptTemplate.
# 
#  LangChain parses your message string and searches for placeholders like {input}, then fills them in with values from the dictionary you pass during .invoke(...).
# 

# In[ ]:


### Prompt Engineering
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert AI Engineer. Provide me answer based on the question"), # system prompt definition
        ("user","{input}") # user prompt. System prompt is used to define the behavior of the model (provided in the background). User prompt is the input to the model.
    ]
)
prompt



# String PromptTemplate :
# These prompt templates are used to format a single string, and generally are used for simpler inputs. For example, a common way to construct and use a PromptTemplate is as follows:
# 
# From <https://python.langchain.com/docs/concepts/prompt_templates/> 
# from langchain_core.prompts import PromptTemplate
# 
# prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
# 
# prompt_template.invoke({"topic": "cats"})
# 
# ChatPromptTemplate:
# These prompt templates are used to format a list of messages. These "templates" consist of a list of templates themselves. For example, a common way to construct and use a ChatPromptTemplate is as follows:
# 
# from langchain_core.prompts import ChatPromptTemplate
# 
# prompt_template = ChatPromptTemplate([
#     ("system", "You are a helpful assistant"),
#     ("user", "Tell me a joke about {topic}")
# ])
# 
# prompt_template.invoke({"topic": "cats"})
# 
# In the above example, this ChatPromptTemplate will construct two messages when called. The first is a system message, that has no variables to format. The second is a HumanMessage, and will be formatted by the topic variable the user passes in.

# In[ ]:


from langchain_groq import ChatGroq
model=ChatGroq(model="gemma2-9b-it")
model


# In[ ]:


### chaining - the idea of langchain is to chain the components together
chain=prompt|model
chain


# In[ ]:


response=chain.invoke({"input":"Can you tell me something about Langsmith"})
print(response.content)


# In[ ]:


### OutputParser - display the output in a specific format
from langchain_core.output_parsers import StrOutputParser

output_parser=StrOutputParser()

chain=prompt|model|output_parser

response=chain.invoke({"input":"Can you tell me about Langsmith"})
print(response)


# In[ ]:


# there are multiple output parsers depending on your needs
from langchain_core.output_parsers import JsonOutputParser

output_parser=JsonOutputParser()
output_parser.get_format_instructions() 
# 'Return a JSON object.' -> this is nothing more than a string which you could add to your system or input prompts


# String Output Parser mostly works for chaining purposes. You can add it directly to the chain.
# 
# But for other parsers, we need to use ChatPromptTemplate.

# In[25]:


### OutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

output_parser=JsonOutputParser()

prompt=PromptTemplate( # Answer the user query - your prompt. format_instruction and query are placeholders for the input variables
    template="Answer the user query \n {format_instruction}\n {query}\n ", # instructions for the output parser + what context you want to add
    input_variables=["query"], # these elements (query and format_instruction) will then be fed into template
    partial_variables={"format_instruction":output_parser.get_format_instructions()},
)

prompt=PromptTemplate(
    template="Answer the user query \n Return a JSON object\n {query}\n ", # -> this is the same as the previous prompt!
    input_variables=["query"],
)


# In[ ]:


prompt


# In[ ]:


chain=prompt|model|output_parser
response=chain.invoke({"query":"Can you tell me about Langsmith?"})
print(response)



# In[ ]:


### Assisgnment ---Chatprompttemplate

# Alternative way to do the same thing using ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate([
    ("system",f"You are an expert AI Engineer. Provide me answer based on the question. {output_parser.get_format_instructions()}"),
    ("user", "{query}")
])
chain = prompt_template | model | output_parser
response = chain.invoke({"query": "Can you tell me about Langsmith?"})
print(response)


### Same done by Krish Naik

# from langchain_core.prompts import ChatPromptTemplate

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","You are an expert AI Engineer.Provide the response in json.Provide me answer based on the question"),
#         ("user","{input}")
#     ]
# )

# chain=prompt|model|output_parser
# response=chain.invoke({"input":"Can you tell me about Langsmith?"})
# print(response)


# In[ ]:


from langchain_core.output_parsers import XMLOutputParser
XML_output_parser=XMLOutputParser()

prompt_template = ChatPromptTemplate([
    ("system",f"You are an expert AI Engineer. Provide me answer based on the question. {XML_output_parser.get_format_instructions()}"),
    ("user", "{query}")
])
chain = prompt_template | model | XML_output_parser
response = chain.invoke({"query": "Can you tell me about Langsmith?"})
print(response)


# ### Assigments: https://python.langchain.com/docs/how_to/#prompt-templates

# In[ ]:


### OutputParser XML format
from langchain_core.output_parsers import XMLOutputParser
output_parser=XMLOutputParser()
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert AI Engineer.<response><answer>Your answer here</answer></response>.Provide me answer based on the question"),
        ("user","{input}")
    ]
)
prompt


# In[ ]:


### OutputParser
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate

output_parser=XMLOutputParser()

prompt=PromptTemplate(
    template="Answer the user query \n {format_instruction}\n {query}\n ",
    input_variables=["query"],
    partial_variables={"format_instruction":output_parser.get_format_instructions()},
)
prompt


# In[ ]:


chain=prompt|model
response=chain.invoke({"query":"Can you tell me about Langsmith?"})
print(response)


# In[ ]:


##output parser
#from langchain_core.output_parsers import XMLOutputParser
from langchain.output_parsers.xml import XMLOutputParser

# XML Output Parser
output_parser = XMLOutputParser()

# Prompt that instructs the model to return XML
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond in this XML format: <response><answer>Your answer here</answer></response>"),
    ("human", "{input}")
])

# Build the chain
chain = prompt | model

# Run the chain
#response = chain.invoke({"input": "What is LangChain?"})

raw_output =chain.invoke({"input": "What is LangChain?"})

# Print result
print(raw_output)


# In[ ]:


## With Pydantic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(temperature=0.7) # by default it is gpt-3.5-turbo


# Define your desired data structure. BaseModel is like a data class. Field is like a field in a data class.
# You can combine output parsers with the fields you want this parser to have using pydantic package
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke."

# Set up a parser + inject instructions into the prompt template -> instructions of the fields LLM should return as an output
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})


# In[ ]:


### Without Pydantic
joke_query = "Tell me a joke ."
model = ChatOpenAI(temperature=0.7, model="gpt-4.1-nano-2025-04-14")

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})


# In[ ]:


actor_query = "Generate the shortened filmography for Tom Hanks."

output = model.invoke(
    f"""{actor_query}
Please enclose the movies in <movie></movie> tags"""
)

print(output.content)


# In[ ]:


from langchain.output_parsers import YamlOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


model = ChatOpenAI(temperature=0.5, model="gpt-4.1-nano-2025-04-14")

# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke. "

# Set up a parser + inject instructions into the prompt template.
parser = YamlOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

chain.invoke({"query": joke_query})


# ### Assisgment:
# Create a simple assistant that uses any LLM and should be pydantic, when we ask about any product it should give you two information product Name, product details tentative price in USD (integer). use chat Prompt Template.
# 

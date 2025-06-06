#!/usr/bin/env python
# coding: utf-8

# ### Assisgment:
# 
# Create a simple assistant that uses any LLM and should be pydantic, when we ask about any product it should give you following information: product Name, product details tentative price in USD (integer). use chat Prompt Template.
# 

# In[2]:


from duckduckgo_search import DDGS

def search_product_on_web(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if results:
            return "\n".join([res['body'] for res in results])
        return "No relevant results found."


# In[ ]:


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

model = ChatOpenAI(temperature=0.1, model = "gpt-4.1-mini-2025-04-14")


class ProductDetails(BaseModel):
    name: str = Field(description="inquired product name")
    details: str = Field(description="details related to the inquired product")
    price: int = Field(description="tentative price of the inquired product in USD provided as an integer")


product_query = "Tell me about the fastest version of BMW model 3 available on the market"
web_data = search_product_on_web(product_query)

parser = JsonOutputParser(pydantic_object=ProductDetails)

prompt = PromptTemplate(
    template="Answer the user query: <{query}> using the following web search results <{web_data}>.\n{format_instructions}\n",
    input_variables=["query", "web_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

response = chain.invoke({
    "query": product_query,
    "web_data": web_data
})
print(response)


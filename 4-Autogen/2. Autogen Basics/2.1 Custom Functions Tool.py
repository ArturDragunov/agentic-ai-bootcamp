import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")


model_client=OpenAIChatCompletionClient(model=os.getenv("LLM_MODEL"),api_key=api_key)

def reverse_string(text: str) -> str:
    '''
    Reverse the given text

    input:str

    output:str

    The reverse string is returned.
    '''
    return text[::-1]

# similar wrapping as with @tool from langchain.
#  you cannot use @FunctionTool as a decorator in this case. Here's why:
# FunctionTool is not designed as a decorator - it's a class that takes a 
# function as a parameter in its constructor. The current syntax is correct
#  you're creating a FunctionTool instance and passing your function to it
reverse_tool = FunctionTool(reverse_string,description='A tool to reverse a string')



agent = AssistantAgent(
    name="ReverseStringAgent",
    model_client= model_client,
    # system prompt is saved to memory
    system_message='You are a helpful assistant that can reverse string using reverse_string tool. Give the result with summary',
    
    # multiple tools can be passed to the agent. We need them to have BaseTool inheritance (we can get it from FunctionTool)
    tools=[reverse_tool], 

# If True, the agent will make another model inference using the tool call and result to generate a response. 
# If False, the tool call result will be returned as the response. By default, if output_content_type is set, 
# this will be True; if output_content_type is not set, this will be False.
# in other words, with reflect_on_tool_use=True, the answer will be a reflection on the tool call results (LLM says that tool X returned Y,
# it evaluates the tool output, thinks about the relevance of the output, tries something else if needed)
# Otherwise, we receive a pure tool output (no additional reflection from LLM)
    reflect_on_tool_use=True
)

async def main(): 
    # you can run agents inside the async code routine -> we need some function to run it
    # in jupyter notebook you can run without def main()
    result = await agent.run(task = 'Reverse the string "Hello, World!"')

    print(result.messages[-1].content)
    # print(result)

if (__name__ == "__main__"):
    asyncio.run(main())

    # print(reverse_string("Hello, World!"))
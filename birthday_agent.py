from mcp.server.fastmcp import FastMCP
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Birthday")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
API_VERSION = "2024-05-01-preview"
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

model_client = AzureOpenAIChatCompletionClient(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION,
    model=AZURE_DEPLOYMENT_NAME
)   


@mcp.tool()
async def send_birthday_wish() -> str:
    birthday_agent = AssistantAgent(
    "BirthdayEmailAssistant",
    description="An assistant that specializes in drafting customized birthday emails.",
    model_client=model_client,
    system_message = '''You are an expert birthday email assistant. Your task is to help draft customized birthday emails using the provided email templates. Ensure that the messages are warm, professional, and tailored to the recipient's preferences.'''
    )
    response = await birthday_agent.on_messages([TextMessage(content=f"Create a customized birthday wish email.", source="user")], cancellation_token=None)
    return response.chat_message.content


if __name__ == "__main__":
    mcp.run(transport="stdio")
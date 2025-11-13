from fastapi import FastAPI
import uvicorn
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination    
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
API_VERSION = "2024-05-01-preview"
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

text_mention_termination = TextMentionTermination("TERMINATE")
max_message_termination = MaxMessageTermination(max_messages = 10)
termination = text_mention_termination | max_message_termination

model_client = AzureOpenAIChatCompletionClient(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION,
    model=AZURE_DEPLOYMENT_NAME
)   

classification_agent = AssistantAgent(
    "EmailClassificationAgent",
    description='''You classify the input mail whether the mail falls in Category 0. Category 1 or Promotional''',
    model_client= model_client,
    system_message='''You are an expert who classifies mails based on 3 categories and returns the category name as output.
    The categories are mentioned below:
    1. Category 0: Mails regarding Birthday or Anniversary wishes,
    2. Category 1: Mails regarding any sort of approvals.,
    3. Promotional: Mails that are refer any promotional advertisement.
    
    Make sure that the priority Category 1 mails is highest and that of Promotional mails is lowest. 
    So first check if mail falls under Category 1, then check if it falls under Category 0 and then check for Promotional.'''
)

birthday_agent = AssistantAgent(
    "BirthdayEmailAssistant",
    description="An assistant that specializes in drafting customized birthday emails.",
    model_client=model_client,
    system_message = '''You are an expert birthday email assistant. Your task is to help draft customized birthday emails using the provided email templates. Ensure that the messages are warm, professional, and tailored to the recipient's preferences.'''
)

actionable_agent = AssistantAgent(
    "ActionableEmailAssistant", 
    description="An assistant that specializes in drafting actionable emails.",
    model_client=model_client,
    system_message = '''You are an expert actionable email assistant. Your task is to help draft emails that include clear calls to action and are tailored to the recipient's needs.'''
)



team = SelectorGroupChat(
    [birthday_agent, actionable_agent],
    model_client=model_client,
    termination_condition=termination
)

@app.post("/categorize_email/")
async def categorize_email(subject: str, body: str):
    response = await classification_agent.on_messages([TextMessage(content=f"Respond with category name with small intent about {body} explaining why you think that mail is of that specific category. ", source="user")], cancellation_token=None)
    return {"email": response.chat_message.content}


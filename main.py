from fastapi import FastAPI
import uvicorn
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console    
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings 
from langchain_pinecone import Pinecone
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import pinecone
import asyncio
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec

load_dotenv()

app = FastAPI()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
API_VERSION = "2024-05-01-preview"
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
endpoint = "https://in03-583f886391f254a.serverless.aws-eu-central-1.cloud.zilliz.com"
token = "9b27dfcdd4a6c842c7fa56c4f034b442915c0ae5c0c0d483a5300adbd14687a377c050acbe602eca96a0a5219db776363b48d0a7"

custom_prompt = ChatPromptTemplate.from_template("""
You are a reasoning AI assistant with access to multiple MCP tools:
- Birthday: for writing birthday emails
- Actionable: for drafting actionable emails
                                                 
Your task is to classify incoming emails into one of the following categories:
                                                 
Use tools when needed, and clearly show your reasoning steps.
""")

model_client = AzureOpenAIChatCompletionClient(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=API_VERSION,
    model=AZURE_DEPLOYMENT_NAME
)

# async def main():
#     client = MultiServerMCPClient(
#         {
#             "Birthday": {
#                 "command": "python",
#                 "args": ["capstone_customized_emails.birthday_agent"],
#                 "transport": "stdio"
#             }
#         }
#     )

#     tools = await client.get_tools()

#     agent = create_agent(
#         model = model_client,
#         tools = tools
#     )


embeddings = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-small",
    model="text-embedding-3-small",
    openai_api_type="azure",
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version="2023-05-15",
    chunk_size=2048
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_message_termination = MaxMessageTermination(max_messages = 10)
termination = text_mention_termination | max_message_termination

# db_client = MilvusClient(
#   uri= endpoint,
#   token= token)

   

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
    So first check if mail falls under Category 1, then check if it falls under Category 0 and then check for Promotional.
    If mail falls under Category 1, {actionable_agent} will draft the mail.'''
)

birthday_agent = AssistantAgent(
    "BirthdayEmailAssistant",
    description="An assistant that specializes in drafting customized birthday emails.",
    model_client=model_client,
    system_message = '''You are an expert birthday email assistant. Your task is to help draft customized birthday emails using the provided email templates. Ensure that the messages are warm, professional, and tailored to the recipient's preferences.
    Make sure to respond with agent name at the end of the mail.'''
)

actionable_agent = AssistantAgent(
    "ActionableEmailAssistant", 
    description="An assistant that specializes in drafting actionable emails.",
    model_client=model_client,
    system_message = '''You are an expert actionable email assistant. The emails are further categorized into 4 categories.
    1. Approval mails: Mails that require some sort of approval from the recipient. 
    2. Asset mails: Mails regarding assets related information.
    3. Leave mails: Mails related to leave requests or approvals.
    4. Meeting mails: Mails concerning meeting schedules or invitations.

    Your task is to ask user if user wants to approve or deny and draft actionable emails accordingly using the provided email templates.
    Make sure to mention category in which mail falls in the beginning of the mail.
    Ensure that the messages are clear, concise, and prompt the recipient to take the desired action.'''
)

def setup_pinecone():
    pc = pinecone.Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
    return pc

def split_text(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = text_splitter.split_text(texts)
    return text_splitter, chunks

def create_index(pc):
    index_name = "emp01-arjun"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=1536, metric="cosine", spec = ServerlessSpec( cloud = "aws", region = "us-east-1"))
    return index_name

def store_vectors(text_splitter, texts, index_name, pc):
    index = pc.Index(host = "https://capstone-email-embeddings-nosu4uj.svc.aped-4627-b74a.pinecone.io")
    chunk =  text_splitter.create_documents(texts)

    # vectorstore = index.upsert(chunk)
    vectorstore = Pinecone.from_documents(
        documents=chunk,
        embedding=embeddings,
        index_name=index_name
    )
    
    return vectorstore

def define_llm():
    llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
    )
    return llm

def create_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})
    return retriever

def create_chain(llm, retriever):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return chain

team = SelectorGroupChat(
    [classification_agent, birthday_agent, actionable_agent],
    model_client=model_client,
    termination_condition=termination
)

@app.post("/get_email/")
async def get_email(subject: str, body: str):
    pc = setup_pinecone()
    text_splitter, chunks = split_text(body)
    index_name = create_index(pc)
    vector_store = store_vectors(text_splitter, chunks, index_name, pc)
    return {"message": f"Email processed and vectors stored successfully.{vector_store}"}

@app.post("/query/")
async def query(query: str):
    vector_store = Pinecone.from_existing_index(
        index_name="emp02-sophia",
        embedding=embeddings
    )
    llm = define_llm()
    retriever = create_retriever(vector_store)
    chain = create_chain(llm, retriever)
    result = chain.invoke(query)
    return {'result':result}

    # print(len(vector))
    # return {"result":[{"email":texts},{"vector": vector}]}
    # response = await team.on_messages([TextMessage(content=f"Draft a customized email for the mail with subject {subject} and body {body}.", source="user")], cancellation_token=None)
    # return {"email": texts}

@app.post("/categorize_email/")
async def categorize_email(subject: str, body: str):
    task = body
    agent = UserProxyAgent("user_proxy", input_func=input)

    team1 = RoundRobinGroupChat([actionable_agent, agent], termination_condition=termination)

    stream = team1.run_stream(task= task)
    return {"response": await Console(stream)}

    # async for event in team.run_stream(task = task):
    #     print(event)
    #     return {"event": event}
    # print(team.run_stream(task = task))
    # agent = UserProxyAgent("user_proxy")
    # response = await asyncio.create_task(
    #     agent.on_messages( 
    #          [TextMessage (content = await agent.on_messages( 
    #             messages = ([TextMessage (content = task , source="user").content]) ,
    #             cancellation_token=None) , source ="user" )],
    #             cancellation_token=None)
    # )
    # return {"response": response}
        # if event == "prompt":
        #     return {"response": event}
        # elif event == "response":
        #     return {"response": event}
    # return {"response":await Console(team.run_stream(task = task))}
    # return {"email": response.chat_message.content}

@app.post("/actionable_email/")
async def actionable_email(subject: str, body: str):
    response = await actionable_agent.on_messages([TextMessage(content=f"Respond with category name with small intent about {subject} and {body} explaining why you think that mail is of that specific category. ", source="user")], cancellation_token=None)
    return {"email": response.chat_message.content}


# if __name__:
#     asyncio.run(main())
    # uvicorn.run(app, host="
# db_client.create_collection(
#     collection_name="demo_aarti",
#     dimension=768,  # The vectors we will use in this demo have 768 dimensions
# )


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
from langchain_core.runnables import RunnableLambda
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_litellm import ChatLiteLLMRouter, ChatLiteLLM
from langchain.agents import create_agent
import pinecone
import asyncio
import os
from dotenv import load_dotenv
from pinecone import ServerlessSpec
from pydantic import BaseModel
import requests 
import litellm
from litellm import embedding, Router
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections, utility
import google.generativeai as genai
# from src.config_loader import MILVUS_CONFIG

load_dotenv()
litellm._turn_on_debug()
app = FastAPI()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
API_VERSION = "2024-05-01-preview"
AZURE_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

custom_prompt = ChatPromptTemplate.from_template("""
You are a reasoning AI assistant with access to multiple MCP tools:
- Birthday: for writing birthday emails
- Actionable: for drafting actionable emails
                                                 
Your task is to classify incoming emails into one of the following categories:
                                                 
Use tools when needed, and clearly show your reasoning steps.
""")

model_client = litellm.LiteLLM()
# model_client = AzureOpenAIChatCompletionClient(
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_KEY,
#     api_version=API_VERSION,
#     model=AZURE_DEPLOYMENT_NAME
# )

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


# embeddings = AzureOpenAIEmbeddings(
#     deployment="text-embedding-3-small",
#     model="text-embedding-3-small",
#     openai_api_type="azure",
#     openai_api_key=AZURE_OPENAI_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     openai_api_version="2023-05-15",
#     chunk_size=2048
# )

def get_embeddings(text):
    embeddings = embedding(
        model = "azure/text-embedding-3-small",
        api_key= AZURE_OPENAI_KEY,
        api_base= AZURE_OPENAI_ENDPOINT,
        api_version= API_VERSION,
        input= text
    )
    return embeddings
# text_mention_termination = TextMentionTermination("approve")
max_message_termination = MaxMessageTermination(max_messages = 5)
termination = max_message_termination

# db_client = MilvusClient(
#   uri= endpoint,
#   token= token)

def approved():
    # response = requests.get("http://127.0.0.1:8000/user_approval/")
    return "approved" 
   
user_agent = UserProxyAgent("user_proxy", input_func=input)

classification_agent = AssistantAgent(
    "EmailClassificationAgent",
    description='''You classify the input mail whether the mail falls in Category 0. Category 1 or Promotional''',
    model_client= model_client,
    system_message='''You are an expert who classifies mails based on 3 categories and returns the category name as output.
    The categories are mentioned below:
    1. Category 0: Mails regarding Birthday or Anniversary wishes,
    2. Category 1: Mails regarding Approvals reagarding any leaves, meetings, assets, etc.
    3. Promotional: Mails that are refer any promotional advertisement.
    
    Make sure that the priority Category 1 mails is highest and that of Promotional mails is lowest. 
    So first check if mail falls under Category 1, then check if it falls under Category 0 and then check for Promotional.
    '''
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

    Your task is to ask user if user wants to approve or deny and accordingly using the provided email templates.
    Make sure to mention category in which mail falls in the beginning of the mail.
    Ensure that the messages are clear, concise, and prompt the recipient to take the desired action.'''
)

def setup_pinecone():
    pc = pinecone.Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
    # pinecone.init(api_key = os.getenv("PINECONE_API_KEY"))
    # Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    # index_name = "emp02-sophia"    
    return pc

def split_text(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    chunks = text_splitter.split_text(texts)
    return text_splitter, chunks

def create_index(pc):
    index_name = "emp01"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=1536, metric="cosine", spec = ServerlessSpec( cloud = "aws", region = "us-east-1"))
    return index_name

def store_vectors(text_splitter, texts, index_name, pc):
    index = pc.Index(host = "https://capstone-email-embeddings-nosu4uj.svc.aped-4627-b74a.pinecone.io")
    chunk =  text_splitter.create_documents(texts)

    # vectorstore = index.upsert(chunk)
    vectorstore = Pinecone.from_documents(
        # documents=chunk,
        embedding=get_embeddings(texts),
        index_name=index_name
    )
    
    return vectorstore

def define_llm(query):
    # llm = AzureChatOpenAI(
    # openai_api_key=AZURE_OPENAI_KEY,
    # azure_deployment=AZURE_DEPLOYMENT_NAME,
    # api_version="2024-05-01-preview",
    # temperature=0,
    # max_tokens=None,
    # timeout=None,
    # max_retries=2
    # )
    llm = [
    {"model_name": "gpt-4o",
    "litellm_params":{
        "model": AZURE_DEPLOYMENT_NAME,
        "api_key": AZURE_OPENAI_KEY,
        "api_version": API_VERSION,
        "api_base": AZURE_OPENAI_ENDPOINT
        }}]
    
    return llm

def create_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})
    return retriever

def create_chain(llm, retriever):
    # RunnableLambda(llm)
    litellm_router = Router(model_list = llm)
    llm = ChatLiteLLMRouter(router=litellm_router, model_name= "gpt-4o", temperature=0.1)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return chain

team = SelectorGroupChat(
    [classification_agent, birthday_agent, actionable_agent, user_agent],
    model_client=model_client,
    termination_condition=termination
)

@app.get("/user_approval/")
async def get_approval():
    return {"response": "approved"}

@app.post("/get_email/")
async def get_email(subject: str, body: str):
    pc = setup_pinecone()
    text_splitter, chunks = split_text(body)
    index_name = create_index(pc)
    vector_store = store_vectors(text_splitter, chunks, index_name, pc)
    return {"message": f"Email processed and vectors stored successfully.{vector_store}"}

# @app.post("/query/")
# async def query(query: str):
#     vector_store = Pinecone.from_existing_index(
#         index_name="emp02-sophia",
#         embedding=embeddings
#     )
#     llm = define_llm()
#     retriever = create_retriever(vector_store)
#     chain = create_chain(llm, retriever)
#     result = chain.invoke(query)
#     return {'result':result}

    # print(len(vector))
    # return {"result":[{"email":texts},{"vector": vector}]}
    # response = await team.on_messages([TextMessage(content=f"Draft a customized email for the mail with subject {subject} and body {body}.", source="user")], cancellation_token=None)
    # return {"email": texts}

class BodyRequest(BaseModel):
    body: str

@app.post("/categorize_email/")
async def categorize_email(req: BodyRequest):
    task = req.body
    termination = MaxMessageTermination(max_messages=4)

    # team1 = RoundRobinGroupChat([actionable_agent, birthday_agent], termination_condition=termination)
    
    stream = team.run_stream(task= task)
    response = await Console(stream)
    print(response)
    return {"response": response}

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

#####################################################################################
#front end
#####################################################################################
class EmailRequest(BaseModel):
    # email_id:str
    subject:str
    body:str

@app.post("/get_email/")
async def get_email(req: EmailRequest):
    # email_id = req.email_id
    subject = req.subject
    body = req.body
    print(body)
    pc = setup_pinecone()
    text_splitter, chunks = split_text(body)
    index_name = create_index(pc)
    vector_store = store_vectors(text_splitter, chunks, index_name, pc)
    task = body
    agent = UserProxyAgent("user_proxy", input_func=input)
    # team1 = RoundRobinGroupChat([birthday_agent,actionable_agent, agent], termination_condition=termination)
    stream = team.run_stream(task= task)
    result = await Console(stream)
    final_message = result.messages[1].content #for bday
    # final_message = result.messages[0].content
    return {"result": final_message}

    # # If result is a list of messages:
    # if isinstance(result, list):
    #     final_message = result[-1]["content"]   # last message from agent
    # else:
    #     final_message = result("content", result)
    # return {"result": final_message}

class QueryRequest(BaseModel):
    query: str

# @app.post("/query/")
# async def query(req: QueryRequest):
def query(query):
    # query=req.query
    query = query
    embeddings = AzureOpenAIEmbeddings(openai_api_key=AZURE_OPENAI_KEY)
    vector_store = Pinecone.from_existing_index(
        index_name="emp02-sophia",
        embedding=embeddings
    )
    # retriever = vector_store.as_retriever()
    # llm_router =ChatLiteLLM(model_name="gpt-4o")
    # qa_chain = RetrievalQA.from_chain_type(llm_router, retriever)
    # response = qa_chain.run(query)
    llm = define_llm(vector_store)
    retriever = create_retriever(vector_store)
    chain = create_chain(llm, retriever)
    result = chain.invoke(query)
    # # result = get_embedding(query)
    return {'result':result}

def connect_milvus():
    connections.connect(
        alias="default",
        host=os.getenv("MILVUS_HOST"),
        port=os.getenv("MILVUS_PORT")
    )

def create_collection():
    connect_milvus()
 
    collection_name = os.getenv("MILVUS_COLLECTION_NAME")
    dimension = os.getenv("MILVUS_DIMENSION")
 
    fields = [
        FieldSchema(name="emial_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        
        FieldSchema(name="mail", dtype=DataType.VARCHAR, max_length=50),
        # FieldSchema(name="category1", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]
 
    schema = CollectionSchema(fields)
    collection = Collection(name=collection_name, schema=schema)
 
    print(f"Created Milvus collection: {collection_name}")
    return collection


def delete_collection():
    connect_milvus()
    collection_name = os.getenv("MILVUS_COLLECTION_NAME")
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
    else:
        print("Collection does not exist")

def get_collection():
    connect_milvus()
    collection_name = os.getenv("MILVUS_COLLECTION_NAME")
 
    try:
        collection = Collection(collection_name)
        collection.load()
        print(f"Collection '{collection_name}' loaded into memory")
        return collection
    except Exception:
        print("Collection not found, creating new...")
        return create_collection()
    
def insert_vector():
    connect_milvus()
    collection = get_collection()
 
    # collection_name = MILVUS_CONFIG["collection_name"]
    # collection = Collection(collection_name)
    mail = [
        [{"subject": "Request for Approval Monthly Budget"},
        {"body": "Dear David,\n\nI hope this message finds you well. I would like to request your approval for the monthly department budget attached below.\n\nRegards,\nSophia"}],
        [{"subject": "Meeting Request for Project Review"},
        {"body": "Dear David,\n\nCould we schedule a meeting this week to review the current status of Project Orion? Please share a convenient time.\n\nBest regards,\nDavid"}],
        [{"subject": "Leave Application"},
        {"body": "Dear David,\n\nI would like to request leave from [start date] to [end date] due to [reason]. Kindly approve my leave request.\n\nSincerely,\nSophia"}],
        [{"subject": "Submission of Report"},
        {"body": "Dear David,\n\nPlease find attached the finalized report for your review.\n\nRegards,\nDavid"}],
        [{"subject": "Confirmation of Attendance"},
        {"body": "Dear David,\n\nI confirm my attendance for the meeting scheduled on [date].\n\nBest regards,\nSophia"}],
        [{"subject": "Request for Document Verification"},
        {"body": "Dear David,\n\nKindly verify the attached documents at your earliest convenience.\n\nThank you,\nDavid"}],
        [{"subject": "Follow-up on Invoice Approval"},
        {"body": "Dear David,\n\nThis is a gentle reminder regarding the pending invoice approval. Please let me know if any additional information is needed.\n\nRegards,\nSophia"}],
        [{"subject": "Project Update – Q4"},
        {"body": "Dear Team,\n\nPlease find the project update for Q4 attached below.\n\nSincerely,\nDavid"}], 
        [{"subject": "Request for Clarification"},
        {"body": "Dear David,\n\nCould you please clarify the requirements mentioned in the recent document shared?\n\nBest,\nSophia"}],
        [{"subject": "Notice of Policy Update"},
        {"body": "Dear Team,\n\nPlease be informed that the company policy on [topic] has been updated.\n\nRegards,\nDavid"}]
    ]
        
    
    embedding = get_embeddings(mail)
    # embeddings = embedding(
    #     model = "text-embedding-3-small",
    #     input = mail
    # )

    data = [
        [mail],
        [embedding]
    ]
 
    collection.insert(data)
    print("Data Inserted")
    # print(f"Inserted vector for email_id={email_id}")

def create_index():
    connect_milvus()
    col = Collection(os.getenv("MILVUS_COLLECTION_NAME"))
 
    print("Creating index on embedding...")
 
    col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
        }
    )
 
    print("Index created successfully!")

def get_google_embedding(text):
    GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=(GOOGLE_KEY))
    response = genai.embed_content(
        model = "text-embedding-004",
        content = text,
    )
    return response['embedding']
if __name__ == "__main__":
    embeddings = get_google_embedding("Dear David,\n\nI hope this message finds you well. I would like to request your approval for the monthly department budget attached below.\n\nRegards,\nSophia")
    data = [
        [1234],
        ["David, I request your approval.Regards, Sophia"],
        [embedding]
    ]
    connect_milvus()
    collection = get_collection()
    collection.insert(data)
    print("Data Inserted")
# from src.milvus_client import connect_milvus, create_collection
 
# connect_milvus()
# create_collection()

# vector_store.py
# from config_loader import MILVUS_CONFIG, EMBEDDING_CONFIG
# from milvus_client import connect_milvus
# from embedder_google import embed_text
# from pymilvus import (
#     Collection, FieldSchema, CollectionSchema, DataType
# )
 
# from config_loader import MILVUS_CONFIG
 
# collection_name = MILVUS_CONFIG["collection_name"]
# dimension = MILVUS_CONFIG["dimension"]   # Example: 768 for Google models
 
 
# # ----------------------------------------------------
# # STEP 1 – Connect to Milvus
# # ----------------------------------------------------
# def get_collection():
#     connect_milvus()
 
#     # If collection exists, load and return it
#     try:
#         collection = Collection(collection_name)
#         collection.load()
#         return collection
#     except:
#         pass
 
#     # Create fields
#     email_id = FieldSchema(
#         name="email_id",
#         dtype=DataType.INT64,
#         is_primary=True,
#         auto_id=False
#     )
 
#     embedding = FieldSchema(
#         name="embedding",
#         dtype=DataType.FLOAT_VECTOR,
#         dim=dimension
#     )
 
#     schema = CollectionSchema(
#         fields=[email_id, embedding],
#         description="Email Vector Collection"
#     )
 
#     # Create new collection
#     collection = Collection(
#         name=collection_name,
#         schema=schema
#     )
 
#     print("Milvus collection created:", collection_name)
#     return collection
 
 
# # ----------------------------------------------------
# # STEP 2 – Insert embedding into Milvus
# # ----------------------------------------------------
# def insert_vector(email_id, text):
#     collection = get_collection()
 
#     # Get embedding from Google
#     vector = embed_text(text)
 
#     collection.insert([[email_id], [vector]])
#     collection.flush()
 
#     print(f"Inserted vector for email ID {email_id}")
 
 
# # ----------------------------------------------------
# # STEP 3 – Search similar emails
# # ----------------------------------------------------
# def search_vector(query_text, top_k=3):
#     collection = get_collection()
 
#     query_vector = embed_text(query_text)
 
#     search_params = {
#         "metric_type": MILVUS_CONFIG["metric_type"],
#         "params": {"nprobe": 10}
#     }
 
#     results = collection.search(
#         data=[query_vector],
#         anns_field="embedding",
#         param=search_params,
#         limit=top_k
#     )
 
#     return results[0]

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, connections
from src.config_loader import MILVUS_CONFIG
from src.embedder_google import embed_text
 
def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_CONFIG["host"],
        port=MILVUS_CONFIG["port"]
    )
 
def create_collection():
    connect_milvus()
 
    collection_name = MILVUS_CONFIG["collection_name"]
    dimension = MILVUS_CONFIG["dimension"]
 
    fields = [
        FieldSchema(name="email_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        
        FieldSchema(name="super_category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="sub_category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
    ]
 
    schema = CollectionSchema(fields)
    collection = Collection(name=collection_name, schema=schema)
 
    print(f"Created Milvus collection: {collection_name}")
    return collection


def get_collection():
    """
    Loads Milvus collection (creates if missing).
    """
    connect_milvus()
    collection_name = MILVUS_CONFIG["collection_name"]
 
    try:
        collection = Collection(collection_name)
        collection.load()
        print(f"Collection '{collection_name}' loaded into memory")
        return collection
    except Exception:
        print("Collection not found, creating new...")
        return create_collection()
 
def insert_vector(email_id, text, super_category, sub_category):
    connect_milvus()
    collection = get_collection()
 
    # collection_name = MILVUS_CONFIG["collection_name"]
    # collection = Collection(collection_name)
 
    embedding = embed_text(text)
 
    data = [
        [email_id],
        # [embedding],
        [super_category],
        [sub_category],
        [embedding]
    ]
 
    collection.insert(data)
    print(f"Inserted vector for email_id={email_id}")
 
 
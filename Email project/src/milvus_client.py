from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from src.config_loader import MILVUS_CONFIG
 
def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_CONFIG["host"],
        port=MILVUS_CONFIG["port"]
    )
    print("Connected to Milvus")
 
def create_collection():
    connect_milvus()
 
    collection_name = MILVUS_CONFIG["collection_name"]
    dimension = MILVUS_CONFIG["dimension"]
 
    fields = [
        FieldSchema(name="email_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        FieldSchema(name="super_category", dtype=DataType.VARCHAR, max_length=30),
        FieldSchema(name="sub_category", dtype=DataType.VARCHAR, max_length=30)
    ]
 
    schema = CollectionSchema(fields)
    collection = Collection(name=collection_name, schema=schema)
 
    print(f"Created Milvus Collection: {collection_name}")
    return collection
 
 
def get_collection():
    connect_milvus()
    col = Collection(MILVUS_CONFIG["collection_name"])
    col.load()
    print("Collection loaded into memory")
    return col
 
 
def create_index():
    connect_milvus()
    col = Collection(MILVUS_CONFIG["collection_name"])
 
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

if __name__ == "__main__":
    connect_milvus()
    collection = get_collection()
    collection.load()
    print("collection loaded into memory")
 



# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
# from src.config_loader import load_config, MILVUS_CONFIG
# CONFIG = load_config()
# MILVUS_CONFIG = CONFIG["milvus"]
 
# def connect_milvus():
#     connections.connect(
#         alias="default",
#         host=MILVUS_CONFIG["host"],
#         port=MILVUS_CONFIG["port"],
#         user=MILVUS_CONFIG["user"],
#         password=MILVUS_CONFIG["password"],
#         secure=MILVUS_CONFIG["secure"]
#     )
#     print("Connected to Milvus")
 
# def create_collection():
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="email_id", dtype=DataType.INT64),
#         FieldSchema(name="super_category", dtype=DataType.VARCHAR, max_length=30),
#         FieldSchema(name="sub_category", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_CONFIG["dimension"])
#     ]
 
#     schema = CollectionSchema(fields)
#     collection = Collection(MILVUS_CONFIG["collection_name"], schema)
#     print("Created Milvus Collection:", MILVUS_CONFIG["collection_name"])
#     return collection
 
# def get_collection():
#     connect_milvus()
#     collection = Collection(MILVUS_CONFIG["collection_name"])

#     collection.load()
#     print("collection loaded into memory")
#     # return Collection(MILVUS_CONFIG["collection_name"])
#     return collection

# def create_index():
#     from pymilvus import Collection, Index
 
#     collection = Collection(MILVUS_CONFIG["collection_name"])
#     collection.load()
 
#     print("Creating index on embedding...")
 
#     collection.create_index(
#         field_name="embedding",
#         index_params={
#             "index_type": "IVF_FLAT",
#             "metric_type": "COSINE",
#             "params": {"nlist": 128}
#         }
#     )
 
#     print("Index created successfully!")

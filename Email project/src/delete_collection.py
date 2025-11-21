from pymilvus import connections, utility
from config_loader import MILVUS_CONFIG
 
connections.connect(
    alias="default",
    host=MILVUS_CONFIG["host"],
    port=MILVUS_CONFIG["port"]
)
 
collection_name = MILVUS_CONFIG["collection_name"]
 
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Deleted collection: {collection_name}")
else:
    print("Collection does not exist")
 
import yaml
import os
from dotenv import load_dotenv
 
load_dotenv()
 
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        raw = f.read()
        for key, value in os.environ.items():
            raw = raw.replace("${" + key + "}", value)
        return yaml.safe_load(raw)
 
config = load_config()
 
MILVUS_CONFIG = config["milvus"]
DB_CONFIG = config["database"] #if "database" in config else {"name": "emails.db"}
CATEGORIES = config["categories"]
# EMBEDDING_MODEL = config["embeddings"]["model"]

# import yaml
 
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)
 
# MILVUS_CONFIG = config["milvus"]
EMBEDDING_CONFIG = config["embeddings"]
# DB_CONFIG = config["database"] if "database" in config else {}
 
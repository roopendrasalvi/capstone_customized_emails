# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# os.environ["TRANSFORMERS_NO_PYTORCH"] = "1"

# from sentence_transformers import SentenceTransformer
# import numpy as np
# from src.config_loader import EMBEDDING_MODEL
 
# model = SentenceTransformer(EMBEDDING_MODEL)
 
# def embed_text(text):
    # return model.encode(text).astype(np.float32).tolist()

# import os
# from transformers import AutoTokenizer, AutoModel
# import numpy as np
# import requests
 
# # Disable all deep-learning frameworks (TF, Torch, Flax)
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# os.environ["TRANSFORMERS_NO_PYTORCH"] = "1"
 
# from dotenv import load_dotenv
# load_dotenv()
 
# HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
 
# def embed_text(text: str):
#     """
#     Generate embeddings using HuggingFace API (no TensorFlow / PyTorch required)
#     """
#     api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
 
#     headers = {"Authorization": f"Bearer {HF_API_KEY}"}
 
#     response = requests.post(api_url, headers=headers, json={"inputs": text})
 
#     if response.status_code != 200:
#         print("❌ HuggingFace Error:", response.text)
#         return None
 
#     embedding = response.json()
#     return np.array(embedding).flatten().tolist()

# import os
# from transformers import AutoTokenizer, AutoModel
# import numpy as np
# import requests
 
# # Disable all deep-learning frameworks (TF, Torch, Flax)
# os.environ["TRANSFORMERS_NO_TF"] = "1"
# os.environ["TRANSFORMERS_NO_FLAX"] = "1"
# os.environ["TRANSFORMERS_NO_PYTORCH"] = "1"
 
# from dotenv import load_dotenv
# load_dotenv()
 
# HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
 
# def embed_text(text: str):
#     """
#     Generate embeddings using HuggingFace API (no TensorFlow / PyTorch required)
#     """
#     api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
 
#     headers = {"Authorization": f"Bearer {HF_API_KEY}"}
 
#     response = requests.post(api_url, headers=headers, json={"inputs": text})
 
#     if response.status_code != 200:
#         print("❌ HuggingFace Error:", response.text)
#         return None
 
#     embedding = response.json()
#     return np.array(embedding).flatten().tolist()
 
# import os
# import requests
# import numpy as np
# from dotenv import load_dotenv
 
# load_dotenv()
 
# HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
 
# API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
 
# def embed_text(text: str):
#     """
#     Generate embeddings using the NEW HuggingFace Router API.
#     No TensorFlow / PyTorch required.
#     """
 
#     headers = {
#         "Authorization": f"Bearer {HF_API_KEY}",
#         "Content-Type": "application/json"
#     }
 
#     payload = {
#         "inputs": text,
#         "parameters": {"truncate": True}
#     }
 
#     response = requests.post(API_URL, headers=headers, json=payload)
 
#     if response.status_code != 200:
#         print("❌ HF Error:", response.text)
#         return None
 
#     output = response.json()
 
#     # HF returns list of lists → we flatten to 384-dim
#     if isinstance(output, list):
#         return np.array(output[0]).tolist()
 
#     return None


# import os
# import requests
# import numpy as np
# from dotenv import load_dotenv
 
# load_dotenv()
 
# HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
 
# # NEW HuggingFace Router API (works in 2025)
# API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
 
# def embed_text(text: str):
#     """
#     Generate embeddings using HuggingFace Router API.
#     No tensorflow/pytorch needed.
#     """
 
#     headers = {
#         "Authorization": f"Bearer {HF_API_KEY}",
#         "Content-Type": "application/json"
#     }
 
#     payload = {
#         "inputs": text,
#         "parameters": {"truncate": True}
#     }
 
#     response = requests.post(API_URL, headers=headers, json=payload)
#     print("\n HF RAW RESPONSE: ", response.text,"\n")
 
#     if response.status_code != 200:
#         print("❌ HF Error:", response.text)
#         return None  # Failure → return None → Milvus will reject
 
#     output = response.json()

#     if "embeddings" in output:
#         emb = output["embeddings"][0]
#         return np.array(emb).tolist()
#     print("HF Error: No embeddings returned")
#     return None

 
#     # HF returns list of lists → convert to 384-dim list
#     # if isinstance(output, list) and len(output) > 0:
#     #     return np.array(output[0]).tolist()
 
#     # return None


import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai
 
load_dotenv()
 
# AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=(GOOGLE_KEY))
MODEL_NAME = "text-embedding-004"
# OPENAI_URL = "https://api.openai.com/v1/embeddings"
 
def embed_text(text: str):
    response = genai.embed_content(
        model = MODEL_NAME,
        content = text,
    )
    return response['embedding']
    # url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/embeddings?api-version={AZURE_VERSION}"
              
    # headers = {
    #     "Content-Type": "application/json",
    #     # "api-key": AZURE_KEY,
    #     "Authorization" : f"Bearer{OPENAI_KEY}"
    # }
 
    # payload = {
    #     "input": text,
    #     "model": "text-embedding-3-small"
    # }

    # response = requests.post(OPENAI_URL, headers=headers, json=payload)
 
    # if response.status_code != 200:
    #     print("❌ Azure Error:", response.text)
    #     return None
 
    # data = response.json()
    # embedding = data["data"][0]["embedding"]
 
    # return embedding
 
# from src.db_client import insert_email
# from src.vector_store import insert_vector
# from src.embedder_google import embed_text
 
# def ingest_email(sender, subject, body, super_category, sub_category):
#     """
#     1. Insert metadata into SQLite
#     2. Create embedding
#     3. Insert embedding + metadata into Milvus
#     """
 
#     # 1 — Insert into SQLite
#     email_id = insert_email(sender, subject, body, super_category, sub_category)
 
#     # 2 — Create text for embedding
#     text = subject + " " + body
#     embedding = embed_text(text)
 
#     # 3 — Insert vector + categories into Milvus
#     insert_vector(email_id, embedding, super_category, sub_category)
 
#     print(f"Email {email_id} inserted successfully with categories ({super_category}, {sub_category}).")
 
 
# src/email_ingest.py
 
from src.db_client import insert_email
from src.vector_store import insert_vector
 
 
def ingest_email(sender, subject, body, super_category, sub_category):
    """
    1. Insert metadata into SQLite
    2. Create text for embedding
    3. Insert text + metadata into Milvus
    """
    
    # 1 — Insert metadata into SQLite
    email_id = insert_email(sender, subject, body, super_category, sub_category)
 
    # 2 — Create text for embedding (subject + body)
    text = subject + " " + body
 
    # 3 — Insert into Milvus (vector_store creates embedding internally)
    insert_vector(email_id, text, super_category, sub_category)
 
    print(f"Email {email_id} inserted successfully with categories ({super_category}, {sub_category})")
 
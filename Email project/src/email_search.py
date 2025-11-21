# from src.milvus_client import get_collection
# from src.embedder_google import embed_text
# from src.db_client import get_email
 
# def find_similar(query):
#     collection = get_collection()
#     query_emb = embed_text(query)
 
#     results = collection.search(
#         data=[query_emb],
#         anns_field="embedding",
#         limit=1,
#         param={"metric_type": "COSINE", "params": {"ef": 50}},
#         output_fields=["email_id", "super_category", "sub_category"]
#     )
 
#     if not results:
#         return None
 
#     match = results[0][0]
#     email_id = match.entity.get("email_id")
#     super_category = match.entity.get("super_category")
#     sub_category = match.entity.get("sub_category")
 
#     email = get_email(email_id)
#     return super_category, sub_category, email

# email_search.py
 
# from vector_store import search_vector
# from db_client import get_email
 
# def find_similar_emails(query):
#     results = search_vector(query)
 
#     matches = []
#     for hit in results:
#         email_id = hit.id
#         similarity = hit.distance
#         sender, subject, body, super_category, sub_category = get_email(email_id)
 
#         matches.append({
#             "email_id": email_id,
#             "similarity": similarity,
#             "sender": sender,
#             "subject": subject,
#             "body": body,
#             "super_category": super_category,
#             "sub_category": sub_category
#         })
 
#     return matches
 

from src.embedder_google import embed_text
from src.milvus_client import get_collection
from src.db_client import get_email
 
def find_similar_email(query):
    collection = get_collection()
 
    query_vec = embed_text(query)
 
    results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {}},
        limit=5
    )
    return results
 
    # hits = results[0]
 
    # for hit in hits:
    #     print("Score:", hit.distance)
    #     print("Email:", get_email(hit.id))
 
# from src.db_client import init_db
# from src.email_ingest import ingest_email
 
# init_db()
 
# # Category 0
# ingest_email("hr@mail.com", "Happy Birthday!", "Wishing you an amazing year!", "category0", "birthday")
# ingest_email("hr@mail.com", "Happy Work Anniversary!", "Congrats on your work anniversary!", "category0", "work_anniversary")
 
# # Category 1
# ingest_email("it@mail.com", "Laptop Issue", "Laptop not working", "category1", "asset")
# ingest_email("employee@mail.com", "Need Leave", "I need 2 days leave", "category1", "leave")
# ingest_email("manager@mail.com", "Meeting Request", "Schedule a meeting tomorrow", "category1", "meeting")
# ingest_email("admin@mail.com", "Request Approval", "Please approve my request", "category1", "approval")
 
# # Promotional
# ingest_email("marketing@mail.com", "Special Offer!", "Promo email for new product", "promotional", "promotional")

# 
from src.db_client import init_db
from src.email_ingest import ingest_email

init_db()
ingest_email(
    sender="hr@test.com",
    subject="Happy Birthday John!",
    body="Wishing you a wonderful year ahead.",
    super_category="category0",
    sub_category="birthday"
)
 
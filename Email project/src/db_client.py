import sqlite3
from src.config_loader import DB_CONFIG
 
DB_NAME = DB_CONFIG["name"]
 
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
 
    cur.execute("""
    CREATE TABLE IF NOT EXISTS emails (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender TEXT,
        subject TEXT,
        body TEXT,
        super_category TEXT,
        sub_category TEXT
    );
    """)
 
    conn.commit()
    conn.close()
    print("SQLite DB ready.")
 
def insert_email(sender, subject, body, super_category, sub_category):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
 
    cur.execute("""
        INSERT INTO emails (sender, subject, body, super_category, sub_category)
        VALUES (?, ?, ?, ?, ?)
    """, (sender, subject, body, super_category, sub_category))
 
    email_id = cur.lastrowid
    conn.commit()
    conn.close()
    return email_id
 
def get_email(email_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT sender, subject, body, super_category, sub_category FROM emails WHERE id=?", (email_id,))
    result = cur.fetchone()
    conn.close()
    return result

if __name__ == "__main__":
    init_db()
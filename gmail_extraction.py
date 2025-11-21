import base64
import time
import json
import asyncio
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
from io import BytesIO
from email.utils import parseaddr, parsedate_to_datetime
from email.message import EmailMessage
from email.header import decode_header
from base64 import urlsafe_b64decode
from email import message_from_bytes
import webbrowser

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import html2text
from src.utils.s3_utility import upload_any_file_to_s3
from src.utils.logger import setup_logger

load_dotenv()


# ==================== Configuration Manager ====================

class ConfigManager:
    """Centralized configuration management from YAML"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load and parse YAML configuration"""
        config_path = os.getenv("CONFIG_PATH", "config/gmail_extraction_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Substitute environment variables
        self._substitute_env_vars()
    
    def _substitute_env_vars(self):
        """Replace ${VAR_NAME} with environment variables"""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                return os.getenv(var_name, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(v) for v in value]
            return value
        
        self.config = substitute_value(self.config)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation (e.g., 'database.host')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_all(self, section: str) -> Dict:
        """Get entire configuration section"""
        return self.config.get(section, {})


# Initialize config manager
config = ConfigManager()


# ==================== Enums ====================

class EmailCategory(str, Enum):
    WORK = "work"
    PERSONAL = "personal"
    SPAM = "spam"
    IMPORTANT = "important"


# ==================== Models ====================

class MessageAttachment(BaseModel):
    filename: str
    mimeType: str
    size: Optional[int] = None
    data: Optional[str] = None
    s3_url: Optional[str] = None
    stored_in_s3: bool = False


class MessageResponse(BaseModel):
    id: str
    from_email: Optional[str] = None
    from_name: Optional[str] = None
    subject: Optional[str] = None
    snippet: str
    body: Optional[str] = None
    date: Optional[str] = None
    attachments: List[MessageAttachment] = []
    has_attachments: bool = False


class EmailListResponse(BaseModel):
    messages: List[MessageResponse]
    count: int
    total_attachments: int = 0


class LabelResponse(BaseModel):
    id: str
    name: str
    type: str


class DraftResponse(BaseModel):
    id: str
    subject: str
    to: str


class FilterResponse(BaseModel):
    id: str
    criteria: Dict
    action: Dict


# ==================== PostgreSQL Database Setup ====================

class PostgreSQLDB:
    def __init__(self):
        # Load from config
        db_config = config.get_all('database')
        
        self.db_host = db_config.get('host', 'localhost')
        self.db_name = db_config.get('name', 'gmail_oauth')
        self.db_user = db_config.get('user', 'postgres')
        self.db_password = db_config.get('password', 'password')
        self.db_port = db_config.get('port', '5432')
        
        pool_config = db_config.get('pool', {})
        min_conn = pool_config.get('min_connections', 2)
        max_conn = pool_config.get('max_connections', 10)
        self.logger = setup_logger()
        self.connection_pool = pool.SimpleConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name
        )
        self.init_db()

    def get_conn(self):
        return self.connection_pool.getconn()

    def put_conn(self, conn):
        self.connection_pool.putconn(conn)

    def init_db(self):
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_credentials (
                    id SERIAL PRIMARY KEY,
                    user_email VARCHAR(255) UNIQUE NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT,
                    token_expiry TIMESTAMP,
                    token_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_user_email ON user_credentials(user_email);
            """)
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.put_conn(conn)

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        conn = self.get_conn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(query, params or ())
            return cursor.fetchall()
        finally:
            cursor.close()
            self.put_conn(conn)

    def execute_single(self, query: str, params: tuple = None) -> Optional[Dict]:
        conn = self.get_conn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(query, params or ())
            return cursor.fetchone()
        finally:
            cursor.close()
            self.put_conn(conn)

    def execute_update(self, query: str, params: tuple = None) -> int:
        conn = self.get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.rowcount
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.put_conn(conn)


db = PostgreSQLDB()


# ==================== Gmail OAuth Manager ====================

class GmailOAuthManager:
    def __init__(self):
        # Load from config
        oauth_config = config.get_all('oauth').get('google', {})
        
        self.GOOGLE_CLIENT_ID = oauth_config.get('client_id')
        self.GOOGLE_CLIENT_SECRET = oauth_config.get('client_secret')
        self.GOOGLE_REDIRECT_URI = oauth_config.get('redirect_uri')
        self.SCOPES = oauth_config.get('scopes', ["https://www.googleapis.com/auth/gmail.modify"])
        self.AUTH_URI = oauth_config.get('auth_uri')
        self.TOKEN_URI = oauth_config.get('token_uri')
        self.logger = setup_logger()
        
        # Load manager config
        manager_config = config.get_all('gmail_manager')
        cache_ttl = manager_config.get('cache_ttl_minutes', 30)
        rate_limit = manager_config.get('rate_limit_calls_per_second', 5)
        
        self.cache = {}
        self.cache_ttl = timedelta(minutes=cache_ttl)
        self.rate_limit_calls_per_second = rate_limit
        self.rate_limit_call_times = []

    # ==================== OAuth Methods ====================

    def get_authorization_url(self, state: str = None) -> tuple:
        """Generate Google OAuth authorization URL"""
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": self.GOOGLE_CLIENT_ID,
                    "client_secret": self.GOOGLE_CLIENT_SECRET,
                    "auth_uri": self.AUTH_URI,
                    "token_uri": self.TOKEN_URI,
                    "redirect_uris": [self.GOOGLE_REDIRECT_URI]
                }
            },
            scopes=self.SCOPES,
            state=state
        )
        flow.redirect_uri = self.GOOGLE_REDIRECT_URI
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            prompt='consent',
            include_granted_scopes='true'
        )
        return authorization_url, state

    def exchange_code_for_credentials(self, code: str) -> Credentials:
        """Exchange authorization code for credentials"""
        flow = Flow.from_client_config(
            {
                "installed": {
                    "client_id": self.GOOGLE_CLIENT_ID,
                    "client_secret": self.GOOGLE_CLIENT_SECRET,
                    "auth_uri": self.AUTH_URI,
                    "token_uri": self.TOKEN_URI,
                    "redirect_uris": [self.GOOGLE_REDIRECT_URI]
                }
            },
            scopes=self.SCOPES
        )
        flow.redirect_uri = self.GOOGLE_REDIRECT_URI
        flow.fetch_token(code=code)
        return flow.credentials

    def credentials_to_dict(self, credentials: Credentials) -> dict:
        """Convert credentials to dictionary for storage"""
        return {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes,
            'expiry': credentials.expiry.isoformat() if credentials.expiry else None
        }

    def dict_to_credentials(self, cred_dict: dict) -> Credentials:
        """Convert dictionary to credentials object"""
        credentials = Credentials(
            token=cred_dict.get('token'),
            refresh_token=cred_dict.get('refresh_token'),
            token_uri=cred_dict.get('token_uri'),
            client_id=cred_dict.get('client_id'),
            client_secret=cred_dict.get('client_secret'),
            scopes=cred_dict.get('scopes')
        )
        if cred_dict.get('expiry'):
            credentials.expiry = datetime.fromisoformat(cred_dict['expiry'])
        return credentials

    def refresh_token_if_needed(self, credentials: Credentials) -> Credentials:
        """Refresh token if expired"""
        if credentials.expired and credentials.refresh_token:
            request = Request()
            credentials.refresh(request)
        return credentials

    def store_credentials(self, user_email: str, credentials: Credentials) -> bool:
        """Store user credentials in database"""
        try:
            cred_dict = self.credentials_to_dict(credentials)
            cred_json = json.dumps(cred_dict)
            existing = db.execute_single(
                "SELECT id FROM user_credentials WHERE user_email = %s", (user_email,)
            )
            if existing:
                db.execute_update(
                    """
                    UPDATE user_credentials 
                    SET access_token = %s, 
                        refresh_token = %s, 
                        token_json = %s, 
                        token_expiry = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_email = %s
                    """,
                    (
                        credentials.token,
                        credentials.refresh_token,
                        cred_json,
                        credentials.expiry,
                        user_email
                    )
                )
            else:
                db.execute_update(
                    """
                    INSERT INTO user_credentials 
                    (user_email, access_token, refresh_token, token_json, token_expiry)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        user_email,
                        credentials.token,
                        credentials.refresh_token,
                        cred_json,
                        credentials.expiry
                    )
                )
            return True
        except psycopg2.Error as e:
            self.logger.error(f"Error storing credentials: {str(e)}")
            return False

    def get_credentials_from_db(self, user_email: str) -> Optional[Credentials]:
        """Retrieve user credentials from database"""
        try:
            result = db.execute_single(
                "SELECT token_json FROM user_credentials WHERE user_email = %s", (user_email,)
            )
            if not result:
                return None
            cred_dict = json.loads(result['token_json'])
            credentials = self.dict_to_credentials(cred_dict)
            credentials = self.refresh_token_if_needed(credentials)
            return credentials
        except Exception as e:
            self.logger.error(f"Error retrieving credentials: {str(e)}")
            return None

    # ==================== Rate Limiting ====================

    def _apply_rate_limit(self):
        """Apply rate limiting to API calls"""
        now = time.time()
        self.rate_limit_call_times = [t for t in self.rate_limit_call_times if now - t < 1]
        if len(self.rate_limit_call_times) >= self.rate_limit_calls_per_second:
            sleep_time = 1 - (now - self.rate_limit_call_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.rate_limit_call_times.append(now)

    # ==================== Gmail API Methods ====================

    def _get_gmail_service(self, credentials: Credentials):
        """Build and return Gmail API service"""
        credentials = self.refresh_token_if_needed(credentials)
        return build('gmail', 'v1', credentials=credentials)

    def _extract_message_body(self, payload: Dict) -> str:
        """Extract message body from email payload"""
        try:
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data', '')
                        if data:
                            return base64.urlsafe_b64decode(data).decode('utf-8')
                    elif part['mimeType'] == 'text/html':
                        data = part['body'].get('data', '')
                        if data:
                            html_content = base64.urlsafe_b64decode(data).decode('utf-8')
                            h = html2text.HTML2Text()
                            h.ignore_links = False
                            return h.handle(html_content)
            else:
                data = payload['body'].get('data', '')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8')
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting message body: {str(e)}")
            return ""

    def _get_attachment_data(self, service, user_id: str, msg_id: str, attachment_id: str) -> Optional[str]:
        """Fetch attachment data from Gmail API"""
        try:
            self._apply_rate_limit()
            attachment = service.users().messages().attachments().get(
                userId=user_id, messageId=msg_id, id=attachment_id
            ).execute()
            return attachment.get('data')
        except Exception as e:
            self.logger.error(f"Error fetching attachment {attachment_id}: {str(e)}")
            return None

    def _extract_attachments(self, service, user_id: str, msg_id: str, payload: Dict) -> List[Dict]:
        """Extract all attachments from email payload"""
        attachments = []

        def process_parts(parts):
            for part in parts:
                if 'parts' in part:
                    process_parts(part['parts'])

                filename = part.get('filename', '')
                mime_type = part.get('mimeType', '')
                body = part.get('body', {})
                attachment_id = body.get('attachmentId')
                size = body.get('size', 0)

                if filename and attachment_id:
                    data = self._get_attachment_data(service, user_id, msg_id, attachment_id)
                    if data:
                        attachments.append({
                            "filename": filename,
                            "mimeType": mime_type,
                            "size": size,
                            "data": data,
                            "attachmentId": attachment_id
                        })

        if 'parts' in payload:
            process_parts(payload['parts'])

        return attachments

    async def _upload_attachment_to_s3(
        self,
        attachment: Dict,
        sender_email: str,
        email_date: str,
        user_email: str
    ) -> Optional[str]:
        """Upload a single attachment to S3"""
        try:
            filename = attachment.get('filename', 'unnamed_attachment')
            mime_type = attachment.get('mimeType', 'application/octet-stream')
            data_base64 = attachment.get('data', '')

            if not data_base64:
                self.logger.error(f"No data for attachment {filename}")
                return None

            try:
                decoded_data = base64.urlsafe_b64decode(data_base64)
            except Exception as e:
                self.logger.error(f"Error decoding attachment {filename}: {str(e)}")
                return None

            sender_folder = sender_email.replace('@', '_at_').replace('.', '_').replace('/', '_').replace('\\', '_')
            folder_path = f"email-attachments/{user_email}/{email_date}/{sender_folder}"

            file_bytes = BytesIO(decoded_data)
            file_bytes.seek(0)

            from starlette.datastructures import UploadFile as StarletteUploadFile
            from starlette.datastructures import Headers

            headers = Headers({'content-type': mime_type})

            upload_file = StarletteUploadFile(
                filename=filename,
                file=file_bytes,
                headers=headers
            )

            s3_url = await upload_any_file_to_s3(
                user_id=user_email,
                file=upload_file,
                folder_name=folder_path
            )

            self.logger.info(f"Successfully uploaded {filename} to S3: {s3_url}")
            return s3_url

        except Exception as e:
            self.logger.error(f"Error uploading attachment {attachment.get('filename')}: {str(e)}")
            return None

    async def get_message_details(
        self,
        credentials: Credentials,
        message_id: str,
        user_email: str,
        upload_to_s3: bool = False
    ) -> Dict:
        """Get detailed message information including body and attachments"""
        self._apply_rate_limit()
        service = self._get_gmail_service(credentials)

        try:
            message = service.users().messages().get(
                userId='me', id=message_id, format='full'
            ).execute()

            payload = message['payload']
            headers = payload.get('headers', [])

            headers_dict = {}
            for header in headers:
                headers_dict[header['name']] = header['value']

            email_data = {
                'id': message['id'],
                'threadId': message.get('threadId'),
                'labelIds': message.get('labelIds', []),
                'snippet': message.get('snippet', ''),
                'internalDate': message.get('internalDate'),
                'headers': headers_dict,
                'body': self._extract_message_body(payload),
                'attachments': []
            }

            raw_attachments = self._extract_attachments(service, 'me', message_id, payload)

            if raw_attachments and upload_to_s3:
                from_header = headers_dict.get('From', 'unknown')
                sender_name, sender_email = parseaddr(from_header)
                if not sender_email:
                    sender_email = 'unknown_sender'

                date_header = headers_dict.get('Date')
                try:
                    if date_header:
                        email_date_obj = parsedate_to_datetime(date_header)
                        email_date = email_date_obj.strftime('%Y-%m-%d')
                    else:
                        email_date = datetime.now().strftime('%Y-%m-%d')
                except Exception:
                    email_date = datetime.now().strftime('%Y-%m-%d')

                processed_attachments = []
                for attachment in raw_attachments:
                    s3_url = await self._upload_attachment_to_s3(
                        attachment=attachment,
                        sender_email=sender_email,
                        email_date=email_date,
                        user_email=user_email
                    )

                    attachment_info = {
                        'filename': attachment['filename'],
                        'mimeType': attachment['mimeType'],
                        'size': attachment.get('size', 0),
                        'stored_in_s3': s3_url is not None
                    }

                    if s3_url:
                        attachment_info['data'] = s3_url
                        attachment_info['s3_url'] = s3_url
                    else:
                        attachment_info['data'] = attachment['data']

                    processed_attachments.append(attachment_info)

                email_data['attachments'] = processed_attachments
            else:
                email_data['attachments'] = raw_attachments

            return email_data

        except Exception as e:
            self.logger.error(f"Error getting message details for {message_id}: {str(e)}")
            raise

    def list_messages(
        self,
        credentials: Credentials,
        query: str = "",
        max_results: int = 10
    ) -> List[Dict]:
        """List messages matching the query"""
        self._apply_rate_limit()
        service = self._get_gmail_service(credentials)

        try:
            results = service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
            return results.get('messages', [])
        except Exception as e:
            self.logger.error(f"Error listing messages: {str(e)}")
            return []

    def mark_as_read(self, credentials: Credentials, message_id: str) -> bool:
        """Mark a message as read"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            service.users().messages().modify(
                userId='me', id=message_id, body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            self.logger.error(f"Error marking message as read: {str(e)}")
            return False

    # ==================== Send Email ====================

    async def send_email(self, credentials: Credentials, recipient: str, subject: str, message: str) -> dict:
        """Send email"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            message_obj = EmailMessage()
            message_obj.set_content(message)
            message_obj['To'] = recipient
            message_obj['Subject'] = subject

            encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
            create_message = {'raw': encoded_message}

            send_message = await asyncio.to_thread(
                service.users().messages().send(userId="me", body=create_message).execute
            )
            return {"status": "success", "message_id": send_message['id']}
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    # ==================== Draft Management ====================

    async def create_draft(self, credentials: Credentials, recipient: str, subject: str, message: str) -> dict:
        """Create draft email"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            message_obj = EmailMessage()
            message_obj.set_content(message)
            message_obj['To'] = recipient
            message_obj['Subject'] = subject

            encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
            create_message = {'raw': encoded_message}

            draft = await asyncio.to_thread(
                service.users().drafts().create(userId="me", body={'message': create_message}).execute
            )
            return {"status": "success", "draft_id": draft['id']}
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def list_drafts(self, credentials: Credentials) -> list:
        """List all drafts"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            results = await asyncio.to_thread(
                service.users().drafts().list(userId="me").execute
            )
            drafts = results.get('drafts', [])

            draft_list = []
            for draft in drafts:
                draft_id = draft['id']
                draft_data = await asyncio.to_thread(
                    service.users().drafts().get(userId="me", id=draft_id).execute
                )

                message = draft_data.get('message', {})
                headers = message.get('payload', {}).get('headers', [])

                subject = next(
                    (header['value'] for header in headers if header['name'].lower() == 'subject'),
                    'No Subject'
                )
                to = next(
                    (header['value'] for header in headers if header['name'].lower() == 'to'),
                    'No Recipient'
                )

                draft_list.append({'id': draft_id, 'subject': subject, 'to': to})

            return draft_list
        except HttpError as error:
            return []

    # ==================== Trash Management ====================

    async def trash_email(self, credentials: Credentials, email_id: str) -> str:
        """Move email to trash"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().messages().trash(userId="me", id=email_id).execute
            )
            return "Email moved to trash successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    # ==================== Label Management ====================

    async def list_labels(self, credentials: Credentials) -> list:
        """List all labels"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            results = await asyncio.to_thread(
                service.users().labels().list(userId="me").execute
            )
            labels = results.get('labels', [])

            label_list = []
            for label in labels:
                label_list.append({
                    'id': label['id'],
                    'name': label['name'],
                    'type': label['type']
                })

            return label_list
        except HttpError as error:
            return []

    async def create_label(self, credentials: Credentials, name: str) -> dict:
        """Create new label"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            label_object = {
                'name': name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }

            created_label = await asyncio.to_thread(
                service.users().labels().create(userId="me", body=label_object).execute
            )

            return {
                'status': 'success',
                'label_id': created_label['id'],
                'name': created_label['name']
            }
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def apply_label(self, credentials: Credentials, email_id: str, label_id: str) -> str:
        """Apply label to email"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={'addLabelIds': [label_id]}
                ).execute
            )
            return "Label applied successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    async def remove_label(self, credentials: Credentials, email_id: str, label_id: str) -> str:
        """Remove label from email"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={'removeLabelIds': [label_id]}
                ).execute
            )
            return "Label removed successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    async def rename_label(self, credentials: Credentials, label_id: str, new_name: str) -> dict:
        """Rename label"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            label = await asyncio.to_thread(
                service.users().labels().get(userId="me", id=label_id).execute
            )

            label['name'] = new_name

            updated_label = await asyncio.to_thread(
                service.users().labels().update(
                    userId="me",
                    id=label_id,
                    body=label
                ).execute
            )

            return {
                'status': 'success',
                'label_id': updated_label['id'],
                'name': updated_label['name']
            }
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def delete_label(self, credentials: Credentials, label_id: str) -> str:
        """Delete label"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().labels().delete(userId="me", id=label_id).execute
            )
            return "Label deleted successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    async def search_by_label(self, credentials: Credentials, label_id: str) -> list:
        """Search emails by label"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            query = f"label:{label_id}"

            response = await asyncio.to_thread(
                service.users().messages().list(userId="me", q=query).execute
            )

            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = await asyncio.to_thread(
                    service.users().messages().list(
                        userId="me",
                        q=query,
                        pageToken=page_token
                    ).execute
                )
                messages.extend(response['messages'])

            return messages
        except HttpError as error:
            return []

    # ==================== Filter Management ====================

    async def list_filters(self, credentials: Credentials) -> list:
        """List all filters"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            results = await asyncio.to_thread(
                service.users().settings().filters().list(userId="me").execute
            )
            return results.get('filter', [])
        except HttpError as error:
            return []

    async def create_filter(
        self,
        credentials: Credentials,
        from_email: str = None,
        to_email: str = None,
        subject: str = None,
        query: str = None,
        has_attachment: bool = None,
        exclude_chats: bool = None,
        size_comparison: str = None,
        size: int = None,
        add_label_ids: list = None,
        remove_label_ids: list = None,
        forward_to: str = None
    ) -> dict:
        """Create email filter"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            criteria = {}
            if from_email:
                criteria['from'] = from_email
            if to_email:
                criteria['to'] = to_email
            if subject:
                criteria['subject'] = subject
            if query:
                criteria['query'] = query
            if has_attachment is not None:
                criteria['hasAttachment'] = has_attachment
            if exclude_chats is not None:
                criteria['excludeChats'] = exclude_chats
            if size_comparison and size:
                criteria['sizeComparison'] = size_comparison
                criteria['size'] = size

            action = {}
            if add_label_ids:
                action['addLabelIds'] = add_label_ids
            if remove_label_ids:
                action['removeLabelIds'] = remove_label_ids
            if forward_to:
                action['forward'] = forward_to

            filter_object = {'criteria': criteria, 'action': action}

            created_filter = await asyncio.to_thread(
                service.users().settings().filters().create(
                    userId="me",
                    body=filter_object
                ).execute
            )

            return {'status': 'success', 'filter_id': created_filter['id']}
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def delete_filter(self, credentials: Credentials, filter_id: str) -> str:
        """Delete filter"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().settings().filters().delete(
                    userId="me",
                    id=filter_id
                ).execute
            )
            return "Filter deleted successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    # ==================== Folder Management ====================

    async def create_folder(self, credentials: Credentials, name: str) -> dict:
        """Create folder (label)"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            label_object = {
                'name': name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show',
                'type': 'user'
            }

            created_label = await asyncio.to_thread(
                service.users().labels().create(userId="me", body=label_object).execute
            )

            return {
                'status': 'success',
                'folder_id': created_label['id'],
                'name': created_label['name']
            }
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def move_to_folder(self, credentials: Credentials, email_id: str, folder_id: str) -> str:
        """Move email to folder"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={'addLabelIds': [folder_id], 'removeLabelIds': ['INBOX']}
                ).execute
            )
            return "Email moved to folder successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    async def list_folders(self, credentials: Credentials) -> list:
        """List all folders"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            results = await asyncio.to_thread(
                service.users().labels().list(userId="me").execute
            )
            labels = results.get('labels', [])

            folders = [
                {'id': label['id'], 'name': label['name']}
                for label in labels
                if label['type'] == 'user'
            ]

            return folders
        except HttpError as error:
            return []

    # ==================== Archive Management ====================

    async def archive_email(self, credentials: Credentials, email_id: str) -> str:
        """Archive email"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={'removeLabelIds': ['INBOX']}
                ).execute
            )
            return "Email archived successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    async def batch_archive(self, credentials: Credentials, query: str, max_emails: int = 100) -> dict:
        """Batch archive emails"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)

            response = await asyncio.to_thread(
                service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=max_emails
                ).execute
            )

            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            if not messages:
                return {
                    'status': 'success',
                    'archived_count': 0,
                    'message': 'No emails found.'
                }

            archived_count = 0
            for msg in messages:
                try:
                    await asyncio.to_thread(
                        service.users().messages().modify(
                            userId="me",
                            id=msg['id'],
                            body={'removeLabelIds': ['INBOX']}
                        ).execute
                    )
                    archived_count += 1
                except Exception as e:
                    self.logger.error(f"Error archiving email: {str(e)}")

            return {
                'status': 'success',
                'archived_count': archived_count,
                'total_found': len(messages)
            }
        except HttpError as error:
            return {"status": "error", "error_message": str(error)}

    async def list_archived(self, credentials: Credentials, max_results: int = 50) -> list:
        """List archived emails"""
        query = "-in:inbox"
        return await self.search_emails(credentials, query, max_results)

    async def restore_to_inbox(self, credentials: Credentials, email_id: str) -> str:
        """Restore email to inbox"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            await asyncio.to_thread(
                service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={'addLabelIds': ['INBOX']}
                ).execute
            )
            return "Email restored to inbox successfully."
        except HttpError as error:
            return f"An error occurred: {str(error)}"

    # ==================== Search ====================

    async def search_emails(self, credentials: Credentials, query: str, max_results: int = 50) -> list:
        """Search emails using Gmail syntax"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)

            response = await asyncio.to_thread(
                service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=max_results
                ).execute
            )

            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response and len(messages) < max_results:
                page_token = response['nextPageToken']
                response = await asyncio.to_thread(
                    service.users().messages().list(
                        userId='me',
                        q=query,
                        pageToken=page_token,
                        maxResults=max_results - len(messages)
                    ).execute
                )
                if 'messages' in response:
                    messages.extend(response['messages'])

            result_messages = []
            for msg in messages:
                msg_data = await asyncio.to_thread(
                    service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='metadata',
                        metadataHeaders=['Subject', 'From', 'Date']
                    ).execute
                )

                headers = msg_data.get('payload', {}).get('headers', [])

                subject = next(
                    (header['value'] for header in headers if header['name'].lower() == 'subject'),
                    'No Subject'
                )
                sender = next(
                    (header['value'] for header in headers if header['name'].lower() == 'from'),
                    'Unknown Sender'
                )
                date = next(
                    (header['value'] for header in headers if header['name'].lower() == 'date'),
                    ''
                )

                result_messages.append({
                    'id': msg['id'],
                    'threadId': msg['threadId'],
                    'subject': subject,
                    'from': sender,
                    'date': date,
                    'snippet': msg_data.get('snippet', '')
                })

            return result_messages

        except HttpError as error:
            return []

    # ==================== Read Email ====================

    async def get_unread_emails(self, credentials: Credentials) -> list:
        """Get unread emails"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            query = 'in:inbox is:unread category:primary'

            response = service.users().messages().list(userId='me', q=query).execute()
            messages = []
            if 'messages' in response:
                messages.extend(response['messages'])

            while 'nextPageToken' in response:
                page_token = response['nextPageToken']
                response = service.users().messages().list(
                    userId='me',
                    q=query,
                    pageToken=page_token
                ).execute()
                messages.extend(response['messages'])

            return messages

        except HttpError as error:
            return []

    async def read_email(self, credentials: Credentials, email_id: str) -> dict:
        """Read email content"""
        try:
            self._apply_rate_limit()
            service = self._get_gmail_service(credentials)
            msg = service.users().messages().get(
                userId="me", id=email_id, format='raw'
            ).execute()

            raw_data = msg['raw']
            decoded_data = urlsafe_b64decode(raw_data)
            mime_message = message_from_bytes(decoded_data)

            body = None
            if mime_message.is_multipart():
                for part in mime_message.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode()
                        break
            else:
                body = mime_message.get_payload(decode=True).decode()

            email_metadata = {
                'content': body,
                'subject': self._decode_header_helper(mime_message.get('subject', '')),
                'from': mime_message.get('from', ''),
                'to': mime_message.get('to', ''),
                'date': mime_message.get('date', '')
            }

            await asyncio.to_thread(self.mark_as_read, credentials, email_id)
            return email_metadata

        except HttpError as error:
            return {}

    async def open_email(self, email_id: str) -> str:
        """Open email in browser"""
        try:
            url = f"https://mail.google.com/#all/{email_id}"
            webbrowser.open(url, new=0, autoraise=True)
            return "Email opened in browser."
        except Exception as error:
            return f"An error occurred: {str(error)}"

    @staticmethod
    def _decode_header_helper(header: str) -> str:
        """Decode MIME header"""
        decoded_parts = decode_header(header)
        decoded_string = ''
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                decoded_string += part.decode(encoding or 'utf-8')
            else:
                decoded_string += part
        return decoded_string


# ==================== FastAPI Application ====================

def get_app_config():
    """Get app configuration from YAML"""
    return config.get_all('app')

app_config = get_app_config()

app = FastAPI(
    title=app_config.get('title', 'Gmail OAuth Integration with MCP Features'),
    description=app_config.get('description', 'Complete Gmail OAuth with all MCP features integrated'),
    version=app_config.get('version', '2.0.0')
)

gmail_manager = GmailOAuthManager()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": config.get('prompts.authentication_required', 'Gmail OAuth Integration API with Complete MCP Features'),
        "version": app_config.get('version', '2.0.0'),
        "features": {
            "authentication": ["/auth/google/login", "/auth/google/callback"],
            "emails": ["/api/emails", "/api/emails/unread", "/api/emails/search"],
            "drafts": ["/api/drafts"],
            "labels": ["/api/labels"],
            "filters": ["/api/filters"],
            "folders": ["/api/folders"],
            "archive": ["/api/archive"],
            "email_actions": ["/api/emails/{id}/mark-read", "/api/emails/{id}/trash", "/api/emails/{id}/read"]
        }
    }

@app.get("/auth/google/authorization-url")
async def get_authorization_url(
    user_email: str = Query(..., description="User's email address for reference")
):
    """Get Google OAuth authorization URL for user to authenticate"""
    try:
        authorization_url, state = gmail_manager.get_authorization_url()
        return {
            "status": "success",
            "authorization_url": authorization_url,
            "state": state,
            "user_email": user_email,
            "message": "Visit the authorization_url in your browser to authenticate. You will be redirected to /auth/google/callback?code=...&state=..."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate authorization URL: {str(e)}")

@app.get("/auth/google/login")
async def google_login():
    """Initiate Google OAuth flow"""
    authorization_url, state = gmail_manager.get_authorization_url()
    return RedirectResponse(authorization_url)


@app.get("/auth/google/callback")
async def google_callback(code: str = Query(None), state: str = Query(None)):
    """Handle Google OAuth callback"""
    try:
        if code is None:
            raise HTTPException(
                status_code=400,
                detail="Missing authorization code from Google OAuth redirect"
            )

        credentials = gmail_manager.exchange_code_for_credentials(code)
        credentials = gmail_manager.refresh_token_if_needed(credentials)

        service = build('gmail', 'v1', credentials=credentials)
        profile = service.users().getProfile(userId='me').execute()
        user_email = profile.get('emailAddress', 'unknown@gmail.com')

        success = gmail_manager.store_credentials(user_email, credentials)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store credentials"
            )

        return {
            "status": "success",
            "user_email": user_email,
            "message": "Successfully authenticated with Gmail"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")


# ==================== Email Endpoints ====================

@app.get("/api/emails", response_model=EmailListResponse)
async def get_emails(
    user_email: str = Query(..., description="User's email address"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of emails to fetch"),
    store_to_s3: bool = Query(False, description="Upload attachments to S3")
):
    """Fetch user's emails with optional S3 attachment storage"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"User credentials not found for {user_email}. Please authenticate first."
        )

    message_list = gmail_manager.list_messages(credentials, max_results=max_results)

    messages = []
    total_attachments = 0

    for msg in message_list:
        try:
            details = await gmail_manager.get_message_details(
                credentials=credentials,
                message_id=msg['id'],
                user_email=user_email,
                upload_to_s3=store_to_s3
            )

            from_header = details.get('headers', {}).get('From', '')
            sender_name, sender_email = parseaddr(from_header)

            attachments = [
                MessageAttachment(
                    filename=att['filename'],
                    mimeType=att['mimeType'],
                    size=att.get('size'),
                    data=att.get('data'),
                    s3_url=att.get('s3_url'),
                    stored_in_s3=att.get('stored_in_s3', False)
                ) for att in details.get("attachments", [])
            ]

            total_attachments += len(attachments)

            messages.append(MessageResponse(
                id=details.get('id'),
                from_email=sender_email or from_header,
                from_name=sender_name,
                subject=details.get('headers', {}).get('Subject'),
                snippet=details.get('snippet', ''),
                body=details.get('body'),
                date=details.get('headers', {}).get('Date'),
                attachments=attachments,
                has_attachments=len(attachments) > 0
            ))
        except Exception as e:
            self.logger.error(f"Error processing message {msg['id']}: {str(e)}")
            continue

    return EmailListResponse(
        messages=messages,
        count=len(messages),
        total_attachments=total_attachments
    )


@app.get("/api/emails/unread", response_model=EmailListResponse)
async def get_unread_emails(
    user_email: str = Query(..., description="User's email address"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of emails to fetch"),
    store_to_s3: bool = Query(False, description="Upload attachments to S3")
):
    """Fetch unread emails with optional S3 attachment storage"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"User credentials not found for {user_email}. Please authenticate first."
        )

    message_list = gmail_manager.list_messages(
        credentials,
        query="is:unread",
        max_results=max_results
    )

    messages = []
    total_attachments = 0

    for msg in message_list:
        try:
            details = await gmail_manager.get_message_details(
                credentials=credentials,
                message_id=msg['id'],
                user_email=user_email,
                upload_to_s3=store_to_s3
            )

            from_header = details.get('headers', {}).get('From', '')
            sender_name, sender_email = parseaddr(from_header)

            attachments = [
                MessageAttachment(
                    filename=att['filename'],
                    mimeType=att['mimeType'],
                    size=att.get('size'),
                    data=att.get('data'),
                    s3_url=att.get('s3_url'),
                    stored_in_s3=att.get('stored_in_s3', False)
                ) for att in details.get("attachments", [])
            ]

            total_attachments += len(attachments)

            messages.append(MessageResponse(
                id=details.get('id'),
                from_email=sender_email or from_header,
                from_name=sender_name,
                subject=details.get('headers', {}).get('Subject'),
                snippet=details.get('snippet', ''),
                body=details.get('body'),
                date=details.get('headers', {}).get('Date'),
                attachments=attachments,
                has_attachments=len(attachments) > 0
            ))
        except Exception as e:
            self.logger.error(f"Error processing message {msg['id']}: {str(e)}")
            continue

    return EmailListResponse(
        messages=messages,
        count=len(messages),
        total_attachments=total_attachments
    )


@app.get("/api/emails/search", response_model=EmailListResponse)
async def search_emails(
    user_email: str = Query(..., description="User's email address"),
    query: str = Query(..., description="Gmail search query"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of emails to fetch"),
    store_to_s3: bool = Query(False, description="Upload attachments to S3")
):
    """Search emails with optional S3 attachment storage"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(
            status_code=404,
            detail=f"User credentials not found for {user_email}. Please authenticate first."
        )

    message_list = gmail_manager.list_messages(
        credentials,
        query=query,
        max_results=max_results
    )

    messages = []
    total_attachments = 0

    for msg in message_list:
        try:
            details = await gmail_manager.get_message_details(
                credentials=credentials,
                message_id=msg['id'],
                user_email=user_email,
                upload_to_s3=store_to_s3
            )

            from_header = details.get('headers', {}).get('From', '')
            sender_name, sender_email = parseaddr(from_header)

            attachments = [
                MessageAttachment(
                    filename=att['filename'],
                    mimeType=att['mimeType'],
                    size=att.get('size'),
                    data=att.get('data'),
                    s3_url=att.get('s3_url'),
                    stored_in_s3=att.get('stored_in_s3', False)
                ) for att in details.get("attachments", [])
            ]

            total_attachments += len(attachments)

            messages.append(MessageResponse(
                id=details.get('id'),
                from_email=sender_email or from_header,
                from_name=sender_name,
                subject=details.get('headers', {}).get('Subject'),
                snippet=details.get('snippet', ''),
                body=details.get('body'),
                date=details.get('headers', {}).get('Date'),
                attachments=attachments,
                has_attachments=len(attachments) > 0
            ))
        except Exception as e:
            self.logger.error(f"Error processing message {msg['id']}: {str(e)}")
            continue

    return EmailListResponse(
        messages=messages,
        count=len(messages),
        total_attachments=total_attachments
    )


@app.get("/api/emails/{email_id}/read")
async def read_email(
    email_id: str,
    user_email: str = Query(..., description="User's email address")
):
    """Read email content"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    email_content = await gmail_manager.read_email(credentials, email_id)
    return {"status": "success", "email": email_content}


@app.post("/api/emails/{message_id}/mark-read")
async def mark_email_read(
    message_id: str,
    user_email: str = Query(..., description="User's email address")
):
    """Mark an email as read"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail=f"User credentials not found for {user_email}")

    success = gmail_manager.mark_as_read(credentials, message_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to mark email as read")

    return {"status": "success", "message": f"Email {message_id} marked as read"}


@app.post("/api/emails/{email_id}/trash")
async def trash_email(
    email_id: str,
    user_email: str = Query(...),
):
    """Trash email"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.trash_email(credentials, email_id)
    return {"status": "success", "message": result}


# ==================== Send Email ====================

@app.post("/api/emails/send")
async def send_email(
    user_email: str = Query(...),
    recipient: str = Query(...),
    subject: str = Query(...),
    message: str = Query(...)
):
    """Send email"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.send_email(credentials, recipient, subject, message)
    return result


# ==================== Draft Endpoints ====================

@app.get("/api/drafts")
async def get_drafts(user_email: str = Query(...)):
    """List drafts"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    drafts = await gmail_manager.list_drafts(credentials)
    return {"status": "success", "drafts": drafts, "count": len(drafts)}


@app.post("/api/drafts")
async def create_draft(
    user_email: str = Query(...),
    recipient: str = Query(...),
    subject: str = Query(...),
    message: str = Query(...)
):
    """Create draft"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.create_draft(credentials, recipient, subject, message)
    return result


# ==================== Label Endpoints ====================

@app.get("/api/labels")
async def get_labels(user_email: str = Query(...)):
    """List labels"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    labels = await gmail_manager.list_labels(credentials)
    return {"status": "success", "labels": labels, "count": len(labels)}


@app.post("/api/labels")
async def create_label(
    user_email: str = Query(...),
    name: str = Query(...)
):
    """Create label"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.create_label(credentials, name)
    return result


@app.put("/api/labels/{label_id}")
async def update_label(
    label_id: str,
    user_email: str = Query(...),
    new_name: str = Query(...)
):
    """Rename label"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.rename_label(credentials, label_id, new_name)
    return result


@app.delete("/api/labels/{label_id}")
async def delete_label(
    label_id: str,
    user_email: str = Query(...)
):
    """Delete label"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.delete_label(credentials, label_id)
    return {"status": "success", "message": result}


@app.post("/api/emails/{email_id}/apply-label")
async def apply_label(
    email_id: str,
    user_email: str = Query(...),
    label_id: str = Query(...)
):
    """Apply label to email"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.apply_label(credentials, email_id, label_id)
    return {"status": "success", "message": result}


@app.post("/api/emails/{email_id}/remove-label")
async def remove_label(
    email_id: str,
    user_email: str = Query(...),
    label_id: str = Query(...)
):
    """Remove label from email"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.remove_label(credentials, email_id, label_id)
    return {"status": "success", "message": result}


@app.get("/api/labels/{label_id}/search")
async def search_by_label(
    label_id: str,
    user_email: str = Query(...)
):
    """Search emails by label"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    messages = await gmail_manager.search_by_label(credentials, label_id)
    return {"status": "success", "messages": messages, "count": len(messages)}


# ==================== Filter Endpoints ====================

@app.get("/api/filters")
async def get_filters(user_email: str = Query(...)):
    """List filters"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    filters = await gmail_manager.list_filters(credentials)
    return {"status": "success", "filters": filters, "count": len(filters)}


@app.post("/api/filters")
async def create_filter(
    user_email: str = Query(...),
    from_email: Optional[str] = Query(None),
    to_email: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    add_labels: Optional[List[str]] = Query(None),
    remove_labels: Optional[List[str]] = Query(None),
):
    """Create filter"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.create_filter(
        credentials,
        from_email=from_email,
        to_email=to_email,
        subject=subject,
        add_label_ids=add_labels,
        remove_label_ids=remove_labels
    )
    return result


@app.delete("/api/filters/{filter_id}")
async def delete_filter(
    filter_id: str,
    user_email: str = Query(...)
):
    """Delete filter"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.delete_filter(credentials, filter_id)
    return {"status": "success", "message": result}


# ==================== Folder Endpoints ====================

@app.get("/api/folders")
async def get_folders(user_email: str = Query(...)):
    """List folders"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    folders = await gmail_manager.list_folders(credentials)
    return {"status": "success", "folders": folders, "count": len(folders)}


@app.post("/api/folders")
async def create_folder(
    user_email: str = Query(...),
    name: str = Query(...)
):
    """Create folder"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.create_folder(credentials, name)
    return result


@app.post("/api/emails/{email_id}/move-to-folder")
async def move_to_folder(
    email_id: str,
    user_email: str = Query(...),
    folder_id: str = Query(...)
):
    """Move email to folder"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.move_to_folder(credentials, email_id, folder_id)
    return {"status": "success", "message": result}


# ==================== Archive Endpoints ====================

@app.get("/api/archive")
async def get_archived_emails(
    user_email: str = Query(...),
    max_results: int = Query(50, ge=1, le=100)
):
    """List archived emails"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    archived = await gmail_manager.list_archived(credentials, max_results)
    return {"status": "success", "archived_emails": archived, "count": len(archived)}


@app.post("/api/emails/{email_id}/archive")
async def archive_email(
    email_id: str,
    user_email: str = Query(...)
):
    """Archive single email"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.archive_email(credentials, email_id)
    return {"status": "success", "message": result}


@app.post("/api/archive/batch")
async def batch_archive(
    user_email: str = Query(...),
    query: str = Query(...),
    max_emails: int = Query(100, ge=1, le=500)
):
    """Batch archive emails"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.batch_archive(credentials, query, max_emails)
    return result


@app.post("/api/emails/{email_id}/restore")
async def restore_email(
    email_id: str,
    user_email: str = Query(...)
):
    """Restore archived email"""
    credentials = gmail_manager.get_credentials_from_db(user_email)
    if not credentials:
        raise HTTPException(status_code=404, detail="User not authenticated")

    result = await gmail_manager.restore_to_inbox(credentials, email_id)
    return {"status": "success", "message": result}


if __name__ == "__main__":
    import uvicorn
    
    app_config = config.get_all('app')
    host = app_config.get('host', '0.0.0.0')
    port = app_config.get('port', 8000)
    log_level = app_config.get('log_level', 'info')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level
    )

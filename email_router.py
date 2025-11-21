"""
Email Extraction Router
Handles email extraction from Gmail and Outlook with attachment processing
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging
# from src.extraction.email_extraction.outlook_extraction import OutlookEmailExtractor        # Import Gmail manager
from src.extraction.email_extraction.gmail_extraction import gmail_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class EmailProvider(str, Enum):
    """Supported email providers"""
    GMAIL = "gmail"
    OUTLOOK = "outlook"


class AttachmentInfo(BaseModel):
    """Model for attachment information"""
    filename: str
    size: int
    content_type: str
    s3_url: Optional[str] = None


class EmailExtractionRequest(BaseModel):
    """Request model for email extraction"""
    user_email: EmailStr = Field(..., description="User's email address")
    provider: EmailProvider = Field(..., description="Email provider (gmail or outlook)")
    # Add filter parameters
    sender_email: Optional[str] = Field(None, description="Filter by sender email")
    subject_keyword: Optional[str] = Field(None, description="Filter by subject keyword")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    folder: Optional[str] = Field("inbox", description="Email folder to search in")

class EmailExtractionResponse(BaseModel):
    """Response model for email extraction"""
    status: str
    message: str
    provider: str
    user_email: str
    extraction_date: str
    emails_processed: int
    attachments_extracted: int
    attachments: List[AttachmentInfo]
    execution_time: float


@router.post("/extract", response_model=EmailExtractionResponse)
async def extract_emails(request: EmailExtractionRequest):
    """
    Extract emails and attachments from Gmail or Outlook for a specific date.
    
    This endpoint:
    1. Connects to the specified email provider
    2. Retrieves emails from the specified date (defaults to today)
    3. Extracts attachments if requested
    4. Uploads attachments to S3 with organized folder structure
    
    Returns:
        EmailExtractionResponse with details about extracted emails and attachments
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting email extraction for {request.user_email} from {request.provider.value}")
        

        now = datetime.now(timezone.utc)
        start_date = request.start_date or now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = request.end_date or now.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Route to appropriate provider
        if request.provider == EmailProvider.GMAIL:
            result = await _extract_gmail_attachments(
                user_email=request.user_email,
                start_date=start_date,
                end_date=end_date,
                include_attachments=True,
                sender_email=request.sender_email,
                subject_keyword=request.subject_keyword,
                folder=request.folder
            )
        elif request.provider == EmailProvider.OUTLOOK:
            result = await _extract_outlook_attachments(
                user_email=request.user_email,
                sender_email=request.sender_email,
                subject_keyword=request.subject_keyword,
                start_date=start_date,
                end_date=end_date,
                folder=request.folder
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported email provider: {request.provider}"
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return EmailExtractionResponse(
            status="success",
            message=f"Successfully extracted emails from {request.provider.value}",
            provider=request.provider.value,
            user_email=request.user_email,
            extraction_date=end_date.isoformat(),
            emails_processed=result["emails_processed"],
            attachments_extracted=result["attachments_extracted"],
            attachments=result["attachments"],
            execution_time=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting emails: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract emails: {str(e)}"
        )


async def _extract_gmail_attachments(
    user_email: str,
    start_date: datetime,
    end_date: datetime,
    include_attachments: bool,
    sender_email: Optional[str] = None,
    subject_keyword: Optional[str] = None,
    folder: str = "inbox"
) -> Dict[str, Any]:
    """
    Extract attachments from Gmail using existing Gmail API endpoints.
    
    Uses the Gmail OAuth Manager and its existing API methods.
    """
    try:
        # Get user credentials
        credentials = gmail_manager.get_credentials_from_db(user_email)
        if not credentials:
            raise HTTPException(
                status_code=404,
                detail=f"Gmail credentials not found for {user_email}. Please authenticate first."
            )
        
        date_str = start_date.strftime("%Y/%m/%d")
        end_date_str = (end_date + timedelta(days=1)).strftime("%Y/%m/%d")
        
        query_parts = [f"after:{date_str}", f"before:{end_date_str}", "has:attachment"]
        
        # Add sender filter
        if sender_email:
            query_parts.append(f"from:{sender_email}")
        
        # Add subject filter
        if subject_keyword:
            query_parts.append(f"subject:{subject_keyword}")
            
        # Add folder filter if not inbox
        if folder and folder.lower() != "inbox":
            query_parts.append(f"in:{folder}")
        
        query = " ".join(query_parts)
        print(f"Full Query is: {query}")
        # Use existing list_messages method to get emails with attachments
        message_list = gmail_manager.list_messages(
            credentials=credentials,
            query=query,
            max_results=100
        )

        print("message list is: ", message_list)
        attachments = []
        emails_processed = 0
        attachments_extracted = 0
        
        # Process each email
        for msg in message_list:
            try:
                # Get message details with attachment processing
                details = await gmail_manager.get_message_details(
                    credentials=credentials,
                    message_id=msg['id'],
                    user_email=user_email,
                    upload_to_s3=include_attachments
                )
                
                emails_processed += 1
                
                # Extract attachment information
                if include_attachments and details.get('attachments'):
                    for att in details['attachments']:
                        if att.get('s3_url') or att.get('stored_in_s3'):
                            attachments.append(AttachmentInfo(
                                filename=att['filename'],
                                size=att.get('size', 0),
                                content_type=att['mimeType'],
                                s3_url=att.get('s3_url') or att.get('data')
                            ))
                            attachments_extracted += 1
                
            except Exception as e:
                logger.error(f"Error processing Gmail message {msg['id']}: {str(e)}")
                continue
        
        logger.info(f"Gmail extraction completed: {emails_processed} emails, {attachments_extracted} attachments")
        
        return {
            "emails_processed": emails_processed,
            "attachments_extracted": attachments_extracted,
            "attachments": attachments
        }
        
    except Exception as e:
        logger.error(f"Gmail extraction error: {str(e)}")
        raise

async def _extract_outlook_attachments(
    user_email: str,
    sender_email: Optional[str] = None,
    subject_keyword: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    folder: str = "inbox"
) -> Dict[str, Any]:
    """
    Extract attachments from Outlook using OutlookEmailExtractor class.
    
    Uses optimized filtering to get only emails with attachments that match criteria.
    """
    try:        
        # Initialize extractor with user email
        extractor = OutlookEmailExtractor(
            user_email=user_email,
            config_path="config/outlook_config.yaml"
        )
        
        if not start_date or not end_date:
            now = datetime.now(timezone.utc)
            start_date = start_date or now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = end_date or now.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Get emails using optimized filter
        emails = extractor.list_emails_with_filter(
            folder=folder,
            sender_email=sender_email,
            subject_keyword=subject_keyword,
            start_date=start_date,
            end_date=end_date,
            has_attachments=True,
            limit=30
        )
        
        logger.info(f"Found {len(emails)} Outlook emails matching criteria")
    
        attachments = []
        emails_processed = 0
        attachments_extracted = 0
        
        # Process each email
        for email in emails:
            try:
                emails_processed += 1
                
                # Extract and save attachments
                saved_attachments = await extractor.extract_and_save_all_attachments(
                    message_id=email['id']
                )
                
                # Add to results
                for att in saved_attachments:
                    if att.s3_url:
                        attachments.append(AttachmentInfo(
                            filename=att.name,
                            size=att.size,
                            content_type=att.content_type,
                            s3_url=att.s3_url
                        ))
                        attachments_extracted += 1
                
            except Exception as e:
                logger.error(f"Error processing Outlook email {email.get('id')}: {str(e)}")
                continue
        
        logger.info(f"Outlook extraction completed: {emails_processed} emails, {attachments_extracted} attachments")
        
        return {
            "emails_processed": emails_processed,
            "attachments_extracted": attachments_extracted,
            "attachments": attachments
        }
        
    except Exception as e:
        logger.error(f"Outlook extraction error: {str(e)}")
        raise



@router.get("/providers")
async def get_supported_providers():
    """Get list of supported email providers"""
    return {
        "providers": [provider.value for provider in EmailProvider],
        "description": {
            "gmail": "Google Gmail with OAuth authentication",
            "outlook": "Microsoft Outlook with OAuth authentication"
        }
    }


@router.get("/status/{user_email}")
async def check_authentication_status(
    user_email: EmailStr,
    provider: EmailProvider = Query(..., description="Email provider to check")
):
    """
    Check if user is authenticated with the specified email provider.
    
    Returns authentication status and details.
    """
    try:
        if provider == EmailProvider.GMAIL:            
            credentials = gmail_manager.get_credentials_from_db(user_email)
            is_authenticated = credentials is not None
            
            if is_authenticated:
                # Check if token is expired
                import json
                cred_info = {
                    "authenticated": True,
                    "email": user_email,
                    "provider": "gmail",
                    "token_valid": not credentials.expired if hasattr(credentials, 'expired') else True
                }
            else:
                cred_info = {
                    "authenticated": False,
                    "email": user_email,
                    "provider": "gmail",
                    "message": "No credentials found. Please authenticate first."
                }
                
        elif provider == EmailProvider.OUTLOOK:            
            try:
                extractor = OutlookEmailExtractor(
                    user_email=user_email,
                    config_path="outlook_config.yaml"
                )
                # If initialization succeeds and can get token, user is authenticated
                token = extractor._get_access_token()
                
                cred_info = {
                    "authenticated": True,
                    "email": user_email,
                    "provider": "outlook",
                    "token_valid": True
                }
            except Exception as e:
                cred_info = {
                    "authenticated": False,
                    "email": user_email,
                    "provider": "outlook",
                    "message": f"Authentication failed: {str(e)}"
                }
        
        return cred_info
        
    except Exception as e:
        logger.error(f"Error checking authentication status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check authentication status: {str(e)}"
        )
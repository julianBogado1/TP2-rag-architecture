from datetime import datetime
from pydantic import BaseModel


class RequestContext(BaseModel):
    user_id: str
    raw_prompt: str
    timestamp: datetime
    session_id: str

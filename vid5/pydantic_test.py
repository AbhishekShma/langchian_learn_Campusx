from pydantic import BaseModel,EmailStr
from typing import Optional

class User(BaseModel):
    id : int
    name : Optional[str] = None
    email : Optional[EmailStr] = None

user1 = User(id=1)

print(user1)
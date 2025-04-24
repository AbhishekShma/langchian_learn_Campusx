from pydantic import BaseModel,EmailStr, Field
from typing import Optional


def func():
    ...
class User(BaseModel):
    id : int = Field(default=1, description="name of the user")
    name : Optional[str] = None
    email : Optional[EmailStr] = None

user1 = User(id=1)

print(user1)


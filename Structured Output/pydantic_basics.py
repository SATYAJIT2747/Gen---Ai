from pydantic import BaseModel , EmailStr , Field
from typing import Optional
class User(BaseModel):
    name : str = 'John Doe'
    age :Optional[int] = Field(default=34, ge=0 , le=100)
    email : EmailStr
user1 = User(name="Alice Smith" , age=28 , email="abc@gmail.com")
print(user1)
# build a schema using pydantic
from pydantic import BaseModel


class NewUser(BaseModel):
    email: str
    name: str
    question: str

    class Config:
        orm_mode = True
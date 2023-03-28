from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

from database import Base


class NewUser(Base):
    __tablename__ = 'new_users'
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String)
    name = Column(String)
    question = Column(String)


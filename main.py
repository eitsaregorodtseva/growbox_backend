from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os

from predict_model import read_imagefile, prepareFile
from schema import NewUser
import crud
import models
import schema
from database import SessionLocal, engine

# from predict_model import prepareFile

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/new_users/")
async def save_new_user(new_user: NewUser, db: Session = Depends(get_db)):
    new_user = crud.create_user(db, new_user)
    return new_user


@app.get("/new_users/")
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return users


@app.post("/class_image/")
async def predict_image_class(file: UploadFile):
    # print(file[0])
    # extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    # if not extension:
    #     return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    # print(image)
    prediction = prepareFile(image)
    # prediction = "123"
    return prediction
    # return {"filename": "Start"}

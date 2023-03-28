from sqlalchemy.orm import Session

import models
import schema


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.NewUser).offset(skip).limit(limit).all()


def create_user(db: Session, user: schema.NewUser):
    db_user = models.NewUser(email=user.email, name=user.name, question=user.question)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

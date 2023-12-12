from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from generate_face import detect_and_assign_id

class FaceModel(BaseModel):
    name: Optional[str] = None
    url: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/faces/")
async def create_face(face: FaceModel):
    generated_face_id, associated_name = detect_and_assign_id(face.url, face.name)
    return {"id": generated_face_id, "name": associated_name}

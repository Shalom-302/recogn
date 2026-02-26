from pydantic import BaseModel
from typing import List, Optional

class FaceAnalysis(BaseModel):
    age: int
    gender: str
    dominant_emotion: str
    dominant_race: str

class SearchResult(BaseModel):
    person_name: str
    distance: float

class MsgResponse(BaseModel):
    message: str


class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str



class Base64Request(BaseModel):
    img_base64: str
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from fastapi import FastAPI
from pydantic import BaseModel
from symptom_prompt import analyze_symptom

app = FastAPI()

class SymptomInput(BaseModel):
    symptoms: str
    severity: int
    trimester: str


@app.get("/")
def root():
    return {"message": "Sakhi Symptom Checker API is running"}

@app.post("/analyze-symptom")
async def analyze_symptom_route(input: SymptomInput):
   
    result = analyze_symptom(
        symptoms=input.symptoms,
        severity=input.severity,
        trimester=input.trimester
    )
    return {"result": result}
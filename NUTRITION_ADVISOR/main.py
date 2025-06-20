from fastapi import FastAPI
from pydantic import BaseModel
from .nutrition_rag2 import query_nutrition_advice

app = FastAPI()

class NutritionInput(BaseModel):
    query: str



@app.get("/")
def root():
    return { "message": "Sakhi Nutrition API is running"}

@app.post("/query-nutrition")
async def query_nutrition_route(input: NutritionInput):
    result = query_nutrition_advice(input.query)
    return {"result": result}
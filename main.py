# Run App
# uvicorn main:app --host 0.0.0.0 --port 8000

# Setup
# curl http://localhost:6333/collections
# curl -X PUT http://localhost:6333/collections/visits -H "Content-Type: application/json" -d "{\"vectors\":{\"size\":384,\"distance\":\"Cosine\"}}"
# curl -X PUT http://localhost:6333/collections/visits -H "Content-Type: application/json" -d "{\"vectors\":{\"size\":768,\"distance\":\"Cosine\"}}"
# curl -X POST "http://localhost:8000/upsert" -H "Content-Type: application/json" -d "{\"visits\":[{\"id\":1,\"description\":\"Visit by John to HQ in June\",\"start_date\":\"2024-06-15\",\"location\":\"HQ\"},{\"id\":2,\"description\":\"Safety Training event with Alice\",\"start_date\":\"2024-05-20\",\"location\":\"Building A\"}]}"
# curl -X POST "http://localhost:8000/upsert" -H "Content-Type: application/json" -d "{\"visits\":[{\"id\":1,\"description\":\"Visit by John to HQ in June\",\"start_date\":\"2024-06-15\",\"location\":\"HQ\"},{\"id\":2,\"description\":\"Safety Training event with Alice\",\"start_date\":\"2024-05-20\",\"location\":\"Building A\"},{\"id\":3,\"description\":\"Team meeting and project kickoff\",\"start_date\":\"2024-07-01\",\"location\":\"Building B\"},{\"id\":4,\"description\":\"Quarterly financial review\",\"start_date\":\"2024-06-20\",\"location\":\"HQ\"},{\"id\":5,\"description\":\"Customer visit and product demo\",\"start_date\":\"2024-07-10\",\"location\":\"Building C\"},{\"id\":6,\"description\":\"Maintenance and safety inspection\",\"start_date\":\"2024-05-25\",\"location\":\"Building A\"},{\"id\":7,\"description\":\"Staff onboarding orientation\",\"start_date\":\"2024-06-30\",\"location\":\"HQ\"},{\"id\":8,\"description\":\"Annual company retreat\",\"start_date\":\"2024-07-15\",\"location\":\"Resort X\"},{\"id\":9,\"description\":\"Vendor contract negotiation\",\"start_date\":\"2024-06-10\",\"location\":\"Building B\"},{\"id\":10,\"description\":\"Health and safety workshop\",\"start_date\":\"2024-05-30\",\"location\":\"Building C\"}]}"

# Delete
# curl -X DELETE http://localhost:6333/collections/visits

# Searching
# GET http://localhost:8000/search?q=John visits last June&location=HQ&top_k=3
# curl -G "http://localhost:8000/search" --data-urlencode "q=John visits last June" --data-urlencode "location=HQ" --data-urlencode "top_k=3"
# curl -G "http://localhost:8000/search" --data-urlencode "q=John and alice" --data-urlencode "top_k=3"
# curl -G "http://localhost:8000/search" --data-urlencode "q=building A visits" --data-urlencode "top_k=3"
# curl -G "http://localhost:8000/search" --data-urlencode "q=building A" --data-urlencode "top_k=3"
# curl -G "http://localhost:8000/search" --data-urlencode "q=visits in building A" --data-urlencode "top_k=3"
# curl -G "http://localhost:8000/search" --data-urlencode "q=location building A" --data-urlencode "top_k=3"


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import httpx
from typing import List

app = FastAPI()
model = SentenceTransformer("all-mpnet-base-v2")
QDRANT_URL = "http://localhost:6333"


class Visit(BaseModel):
    id: int
    description: str
    start_date: str
    location: str


class UpsertRequest(BaseModel):
    visits: List[Visit]


@app.post("/upsert")
async def upsert_visits(req: UpsertRequest):
    points = []
    for visit in req.visits:
        text_to_embed = (
            f"{visit.description} Location: {visit.location} Date: {visit.start_date}"
        )
        vector = model.encode(text_to_embed).tolist()
        points.append(
            {
                "id": visit.id,
                "vector": vector,
                "payload": {
                    "description": visit.description,
                    "start_date": visit.start_date,
                    "location": visit.location,
                },
            }
        )

    async with httpx.AsyncClient() as client:
        res = await client.put(
            f"{QDRANT_URL}/collections/visits/points", json={"points": points}
        )
        if res.status_code != 200:
            print("Qdrant error response:", await res.aread())
            raise HTTPException(
                status_code=500, detail="Failed to upsert points to Qdrant"
            )
    return {"upserted": len(points)}


@app.get("/search")
async def search(q: str, location: str = None, top_k: int = 5):  # type: ignore
    try:
        query_vector = model.encode(q).tolist()

        filter_payload = {}
        if location:
            filter_payload = {
                "must": [{"key": "location", "match": {"value": location}}]
            }

        body = {
            "vector": query_vector,
            "top": top_k,
            "with_payload": True,
        }
        if filter_payload:
            body["filter"] = filter_payload

        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{QDRANT_URL}/collections/visits/points/search", json=body
            )
            if res.status_code != 200:
                raise HTTPException(status_code=500, detail="Qdrant search failed")

            result = res.json()
            return result.get("result", [])
    except Exception as e:
        print(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Embedding failure exception")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

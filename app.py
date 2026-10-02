
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import run_travel_agent

app = FastAPI(
    title="AI Travel Planner Agent",
    description="Travel Planner using LangChain, LangGraph and Gemini",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class TravelRequest(BaseModel):

    origin: str = Field(
        default="",
        description="Starting city (optional)"
    )

    destination: str = Field(
        ...,
        min_length=2,
        description="Travel destination"
    )

    budget: str = Field(
        ...,
        description="Total budget in USD"
    )

    duration: str = Field(
        ...,
        description="Trip duration, e.g. 4 days"
    )

    interests: str = Field(
        default="sightseeing, food, culture",
        description="Travel interests"
    )


@app.get("/")
def home():

    return {
        "message": "AI Travel Planner Agent is running",
        "framework": "LangChain + LangGraph",
        "llm": "Google Gemini",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():

    return {
        "status": "success",
        "message": "Agent is running",
        "framework": "LangChain + LangGraph"
    }


@app.post("/api/plan")
def create_travel_plan(request: TravelRequest):

    if not os.getenv("GOOGLE_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail="AQ.Ab8RN6I7VRlNZg2v800CeJAfn7qETKdi4p1KN-KBt86VoXJT4A"
        )

    if not os.getenv("TAVILY_API_KEY"):

        raise HTTPException(
            status_code=500,
            detail="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"
        )

    try:

        result = run_travel_agent(
            request.model_dump()
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(error)}"
        )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )

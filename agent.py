import os
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import travel_tools


# Conversation state
class TravelState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Connect tools to Gemini
llm_with_tools = llm.bind_tools(travel_tools)


# Agent node
def agent_node(state: TravelState):
    system_prompt = """
You are an AI Travel Planner Agent.

Your task is to create a useful travel plan based on the user's request.

Instructions:
1. Use the flight search tool to find approximate flight options.
2. Use the hotel search tool to find accommodation options.
3. Use the places tool to suggest attractions and activities.
4. Use the weather tool to check destination weather.
5. Use the tools when relevant, then prepare a clear travel plan.
6. Show all estimated costs in USD.
7. Clearly mention that flight and hotel prices are estimates and must be verified.
8. Do not claim that bookings have been made.
9. If the origin city is missing, mention that flight estimates cannot be specific.
10. Keep the plan practical and organize it by day.
"""

    messages = [
        HumanMessage(content=system_prompt),
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Build LangGraph workflow
workflow = StateGraph(TravelState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(travel_tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

travel_graph = workflow.compile()


# Main function called by app.py
def run_travel_agent(request_data: dict):
    origin = request_data.get("origin") or "Not provided"
    destination = request_data["destination"]
    budget = request_data["budget"]
    duration = request_data["duration"]
    interests = request_data.get(
        "interests",
        "sightseeing, food, culture"
    )

    user_message = f"""
Create a travel plan with these details:

Starting city: {origin}
Destination: {destination}
Total budget: {budget} USD
Trip duration: {duration}
Interests: {interests}

Use the available tools to research the destination and prepare
a day-by-day travel itinerary. Include estimated costs and
practical travel tips.
"""

    result = travel_graph.invoke({
        "messages": [HumanMessage(content=user_message)]
    })

    final_message = result["messages"][-1].content

    return {
        "status": "success",
        "destination": destination,
        "plan": final_message
    }

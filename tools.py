
import os
import requests

from tavily import TavilyClient
from langchain_core.tools import tool


# Tavily API Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily = (
    TavilyClient(api_key=TAVILY_API_KEY)
    if TAVILY_API_KEY else None
)


# Common Web Search Function
def web_search(query, max_results=5):

    if tavily is None:
        raise RuntimeError("TAVILY_API_KEY is missing")

    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True
    )

    return {
        "answer": response.get("answer"),
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content", "")[:1200]
            }
            for item in response.get("results", [])
        ]
    }


# 1. FLIGHT SEARCH AGENT
@tool
def search_flights(
    origin: str,
    destination: str,
    duration: str,
    budget: str
) -> dict:
    """Search approximate flight options between two cities.
    Include airlines, approximate USD prices and flight duration."""

    query = f"""
    Find flight options from {origin} to {destination}.
    Duration: {duration}.
    Budget: {budget} USD.

    Find:
    - Airlines
    - Approximate round-trip prices
    - Flight duration
    - Direct flight options

    Prefer USD prices.
    """

    return {
        "agent": "Flight Search Agent",
        "data": web_search(query)
    }


# 2. HOTEL SEARCH AGENT
@tool
def search_hotels(
    destination: str,
    duration: str,
    budget: str
) -> dict:
    """Search hotels, approximate prices, ratings
    and accommodation locations."""

    query = f"""
    Find hotels in {destination} for {duration}.
    Total trip budget: {budget} USD.

    Find:
    - Budget-friendly hotels
    - Mid-range hotels
    - Approximate nightly prices
    - Ratings
    - Locations

    Prefer USD prices.
    """

    return {
        "agent": "Hotel Search Agent",
        "data": web_search(query)
    }


# 3. PLACES SEARCH AGENT
@tool
def search_places(
    destination: str,
    duration: str,
    interests: str
) -> dict:
    """Find tourist attractions, restaurants,
    activities and places to visit."""

    query = f"""
    Find attractions, restaurants and activities
    in {destination}.

    Duration: {duration}
    Interests: {interests}

    Include:
    - Popular tourist attractions
    - Local experiences
    - Famous restaurants
    - Activities
    """

    return {
        "agent": "Places Agent",
        "data": web_search(query, 7)
    }


# 4. WEATHER AGENT
@tool
def get_weather(destination: str) -> dict:
    """Get current weather and a seven-day forecast
    for the destination using Open-Meteo."""

    # Find destination coordinates
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geo_response = requests.get(
        geo_url,
        params={
            "name": destination,
            "count": 1,
            "language": "en",
            "format": "json"
        },
        timeout=20
    )

    geo_response.raise_for_status()

    locations = geo_response.json().get(
        "results", []
    )

    if not locations:
        return {
            "error": f"Location not found: {destination}"
        }

    location = locations[0]

    # Get weather forecast
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_response = requests.get(
        weather_url,
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m"
            ),
            "daily": (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "forecast_days": 7,
            "timezone": "auto"
        },
        timeout=20
    )

    weather_response.raise_for_status()

    data = weather_response.json()

    return {
        "agent": "Weather Agent",
        "location": location.get("name"),
        "country": location.get("country"),
        "current": data.get("current", {}),
        "forecast": data.get("daily", {})
    }


# LANGCHAIN TOOL REGISTRY
travel_tools = [
    search_flights,
    search_hotels,
    search_places,
    get_weather
]

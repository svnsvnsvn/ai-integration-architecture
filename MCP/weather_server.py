"""
Weather Data MCP Server
A simple Model Context Protocol server that provides weather data to AI models
with proper security controls and rate limiting.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import httpx
import json
from dataclasses import dataclass
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Weather MCP Server",
    description="MCP server providing secure weather data access",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# Rate limiting storage
rate_limits = defaultdict(list)
RATE_LIMIT_REQUESTS = 10  # requests per minute per client
RATE_LIMIT_WINDOW = 60  # seconds

# Weather API configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

@dataclass
class WeatherData:
    """Weather data structure"""
    city: str
    country: str
    temperature: float
    feels_like: float
    humidity: int
    pressure: int
    description: str
    wind_speed: float
    timestamp: datetime

class WeatherRequest(BaseModel):
    """Weather request model"""
    city: str = Field(..., min_length=1, max_length=100)
    country_code: Optional[str] = Field(None, min_length=2, max_length=2)
    units: str = Field("metric", pattern="^(metric|imperial|kelvin)$")

class WeatherResponse(BaseModel):
    """Weather response model"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime

def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit"""
    now = time.time()
    client_requests = rate_limits[client_ip]
    
    # Remove old requests outside the window
    rate_limits[client_ip] = [req_time for req_time in client_requests 
                              if now - req_time < RATE_LIMIT_WINDOW]
    
    # Check if limit exceeded
    if len(rate_limits[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    rate_limits[client_ip].append(now)
    return True

def validate_api_key() -> bool:
    """Validate that OpenWeather API key is configured"""
    return OPENWEATHER_API_KEY is not None and len(OPENWEATHER_API_KEY) > 0

async def get_weather_data(city: str, country_code: Optional[str] = None, units: str = "metric") -> WeatherData:
    """Fetch weather data from OpenWeather API"""
    if not validate_api_key():
        raise HTTPException(status_code=500, detail="Weather API key not configured")
    
    # Construct location query
    location = city
    if country_code:
        location = f"{city},{country_code}"
    
    # Sanitize inputs
    location = location.replace(" ", "+")
    
    url = f"{OPENWEATHER_BASE_URL}/weather"
    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
        "units": units
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return WeatherData(
                city=data["name"],
                country=data["sys"]["country"],
                temperature=data["main"]["temp"],
                feels_like=data["main"]["feels_like"],
                humidity=data["main"]["humidity"],
                pressure=data["main"]["pressure"],
                description=data["weather"][0]["description"],
                wind_speed=data["wind"]["speed"],
                timestamp=datetime.now()
            )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="City not found")
        elif e.response.status_code == 401:
            raise HTTPException(status_code=500, detail="Invalid API key")
        else:
            raise HTTPException(status_code=500, detail="Weather service error")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Weather service timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Weather MCP Server",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }

@app.post("/weather", response_model=WeatherResponse)
async def get_weather(request: WeatherRequest, req: Request):
    """Get weather data for a city"""
    client_ip = req.client.host
    
    # Rate limiting
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )
    
    try:
        weather_data = await get_weather_data(
            city=request.city,
            country_code=request.country_code,
            units=request.units
        )
        
        return WeatherResponse(
            success=True,
            data={
                "city": weather_data.city,
                "country": weather_data.country,
                "temperature": weather_data.temperature,
                "feels_like": weather_data.feels_like,
                "humidity": weather_data.humidity,
                "pressure": weather_data.pressure,
                "description": weather_data.description,
                "wind_speed": weather_data.wind_speed,
                "units": request.units
            },
            timestamp=weather_data.timestamp
        )
    except HTTPException:
        raise
    except Exception as e:
        return WeatherResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now()
        )

@app.get("/health")
async def health_check():
    """Health check with API key validation"""
    return {
        "status": "healthy",
        "api_key_configured": validate_api_key(),
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

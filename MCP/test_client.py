"""
Weather MCP Client
Simple client to test the Weather MCP Server
"""

import asyncio
import httpx
import json
from typing import Optional

class WeatherMCPClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    async def get_weather(self, city: str, country_code: Optional[str] = None, units: str = "metric"):
        """Get weather data from MCP server"""
        url = f"{self.base_url}/weather"
        
        payload = {
            "city": city,
            "units": units
        }
        
        if country_code:
            payload["country_code"] = country_code
            
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None
    
    async def health_check(self):
        """Check server health"""
        url = f"{self.base_url}/health"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Health check failed: {e}")
                return None

async def main():
    """Demo the weather MCP client"""
    client = WeatherMCPClient()
    
    print("=== Weather MCP Server Demo ===\n")
    
    # Health check
    print("1. Checking server health...")
    health = await client.health_check()
    if health:
        print(f"[OK] Server Status: {health['status']}")
        print(f"[OK] API Key Configured: {health['api_key_configured']}")
    else:
        print("[ERROR] Server health check failed")
        return
    
    print("\n" + "="*50 + "\n")
    
    # Test weather requests
    test_cities = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU"),
        ("InvalidCity", None)  # This should fail
    ]
    
    for city, country in test_cities:
        print(f"2. Getting weather for {city}" + (f", {country}" if country else "") + "...")
        
        result = await client.get_weather(city, country)
        
        if result and result.get("success"):
            data = result["data"]
            print(f"[OK] {data['city']}, {data['country']}")
            print(f"   Temperature: {data['temperature']}°C (feels like {data['feels_like']}°C)")
            print(f"   Conditions: {data['description']}")
            print(f"   Humidity: {data['humidity']}%")
            print(f"   Wind Speed: {data['wind_speed']} m/s")
        else:
            if result:
                print(f"[ERROR] Error: {result.get('error', 'Unknown error')}")
            else:
                print("[ERROR] Failed to get weather data")
        
        print()
    
    print("=== Rate Limiting Test ===\n")
    print("3. Testing rate limiting (sending multiple requests quickly)...")
    
    # Test rate limiting
    for i in range(12):  # Should hit rate limit after 10
        result = await client.get_weather("London", "GB")
        if result and result.get("success"):
            print(f"[OK] Request {i+1}: Success")
        else:
            print(f"[ERROR] Request {i+1}: Rate limited or error")
        
        if i == 10:  # Give a small break
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())

# Learning Model Context Protocol (MCP) - Weather API Demo

This is my attempt at understanding how Model Context Protocol works by building a simple weather API server. It's a FastAPI server that fetches weather data. 

## What I Built

- **Basic API Server**: FastAPI server that calls OpenWeatherMap
- **Rate Limiting**: Stops you from hitting the API too much (10 requests/minute)
- **Input Validation**: Makes sure city names aren't weird or malicious
- **Error Handling**: Tries not to crash when things go wrong
- **Simple Testing**: Basic test script to see if it works

## What I Learned

- **API Integration**: How to call external APIs properly with timeouts
- **FastAPI Basics**: Creating endpoints with validation using Pydantic
- **Security Basics**: Rate limiting, input sanitization, not exposing secrets
- **Environment Variables**: Keeping API keys out of code
- **Error Handling**: Making sure errors don't expose internal details

## Challenges I Faced

- **Rate Limiting**: Figuring out how to implement this without a database was tricky
- **Error Handling**: Balancing helpful error messages vs security
- **API Keys**: Learning proper environment variable management
- **Validation**: Making sure user input doesn't break things

## How to Run This

### You'll Need
1. Python 3.8+ 
2. An OpenWeatherMap API key (free at [openweathermap.org](https://openweathermap.org/api))

### Setup
```bash
# 1. Create a .env file with your API key
echo "OPENWEATHER_API_KEY=your_key_here" > .env

# 2. Install the packages
pip install fastapi uvicorn httpx python-dotenv pydantic

# 3. Start the server
python weather_server.py

# 4. Test it (in another terminal)
python test_client.py
```

## What the API Does

### GET /weather (with city parameter)
Give it a city name, get back weather data.

**Example:**
```bash
curl "http://localhost:8000/weather?city=London"
```

**You'll get back something like:**
```json
{
    "success": true,
    "data": {
        "city": "London",
        "temperature": 15.2,
        "description": "light rain",
        "humidity": 78
    }
}
```

### Other endpoints:
- `GET /health` - Check if the server is working
- `GET /` - Basic info about the API

## What I'm Still Learning

- **Real MCP Protocol**: This is just a REST API, not true MCP yet
- **Better Error Handling**: Could be more sophisticated
- **Database Integration**: Currently just uses memory for rate limiting
- **Monitoring**: No real observability beyond basic logs
- **Testing**: Could use proper unit tests

## Honest Assessment

This is a learning project that demonstrates:
- Basic API development with security considerations
- External API integration (OpenWeatherMap)
- Input validation and rate limiting
- Environment variable management

It's NOT:
❌ A production ready system
❌ True MCP protocol implementation (yet)
❌ Highly optimized or scalable
❌ Comprehensive in error handling

But it's a foundation for understanding how these systems work.
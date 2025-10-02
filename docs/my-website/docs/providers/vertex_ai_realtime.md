# Vertex AI Realtime API

## Overview

LiteLLM now supports Vertex AI's Realtime (Live) API through the unified realtime interface, providing seamless integration with Google's Live API service.

| Property | Details |
|----------|---------|
| Description | Vertex AI Realtime API provides real-time, bidirectional communication with Google's Live API service |
| Provider Route on LiteLLM | `vertex_ai/` |
| Link to Provider Doc | [Vertex AI Live API ↗](https://cloud.google.com/vertex-ai/generative-ai/docs/live-api) |
| Base URL | `wss://{region}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent` |
| Supported Operations | [`/v1/realtime`](#unified-realtime-api), [`/vertex_ai/live`](#passthrough-websocket) |

## Supported Models

- `gemini-2.0-flash-live-preview-04-09` - Latest Gemini Live model with real-time capabilities
- `gemini-2.0-flash-live-preview` - Gemini Live model preview
- Other Vertex AI Live models as they become available

## Unified Realtime API

Use the standard `/v1/realtime` endpoint with Vertex AI models:

```python
import asyncio
import websockets
import json

async def test_vertex_ai_realtime():
    uri = "ws://localhost:4000/v1/realtime?model=vertex_ai/gemini-2.0-flash-live-preview-04-09"
    
    async with websockets.connect(uri) as websocket:
        # Send session configuration
        session_config = {
            "type": "session.update",
            "session": {
                "max_tokens": 1000,
                "temperature": 0.8,
                "system_instruction": "You are a helpful AI assistant."
            }
        }
        await websocket.send(json.dumps(session_config))
        
        # Send text input
        text_input = "Hello, how are you?"
        await websocket.send(text_input)
        
        # Receive responses
        async for message in websocket:
            response = json.loads(message)
            print(f"Received: {response}")

# Run the test
asyncio.run(test_vertex_ai_realtime())
```

## Passthrough WebSocket

For direct access to Vertex AI Live API features, use the passthrough WebSocket endpoint:

```python
import asyncio
import websockets
import json

async def test_vertex_ai_passthrough():
    uri = "ws://localhost:4000/vertex_ai/live?model=gemini-2.0-flash-live-preview-04-09"
    
    async with websockets.connect(uri) as websocket:
        # Send Vertex AI Live API format messages directly
        setup_message = {
            "setup": {
                "model": "models/gemini-2.0-flash-live-preview-04-09",
                "generationConfig": {
                    "maxOutputTokens": 1000,
                    "temperature": 0.8
                }
            }
        }
        await websocket.send(json.dumps(setup_message))
        
        # Send input
        input_message = {
            "input": {
                "text": "Hello, how are you?"
            }
        }
        await websocket.send(json.dumps(input_message))
        
        # Receive responses
        async for message in websocket:
            response = json.loads(message)
            print(f"Received: {response}")

# Run the test
asyncio.run(test_vertex_ai_passthrough())
```

## Authentication

### Environment Variables

Set up your Vertex AI credentials:

```bash
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### Headers

You can also pass credentials via headers:

```python
headers = {
    "x-goog-user-project": "your-project-id",
    "x-goog-vertex-location": "us-central1"
}
```

## Configuration

### Model Configuration

```yaml
model_list:
  - model_name: gemini-2.0-flash-live-preview-04-09
    litellm_params:
      model: vertex_ai/gemini-2.0-flash-live-preview-04-09
      vertex_project: your-project-id
      vertex_location: us-central1
```

### Session Configuration

The Vertex AI Live API requires session configuration. You can provide this via:

1. **Default Configuration**: LiteLLM provides sensible defaults
2. **Custom Configuration**: Pass via session.update message
3. **Query Parameters**: Set via URL parameters

```python
# Custom session configuration
session_config = {
    "type": "session.update",
    "session": {
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "system_instruction": "You are a helpful AI assistant specialized in coding.",
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
}
```

## Supported Features

### Text Input/Output
- Real-time text generation
- Streaming responses
- Session management

### Audio Input/Output
- Audio buffer management
- Real-time audio processing
- Multiple audio formats support

### Safety and Moderation
- Built-in safety settings
- Content filtering
- Harm category blocking

### Tools and Functions
- Function calling support
- Tool integration
- Custom tool definitions

## Response Format

### Session Created
```json
{
    "type": "session.created",
    "session": {
        "id": "session-123",
        "model": "gemini-2.0-flash-live-preview-04-09",
        "created_at": 1234567890
    }
}
```

### Content Delta
```json
{
    "type": "content.delta",
    "delta": {
        "type": "text",
        "text": "Hello! How can I help you today?"
    }
}
```

### Audio Delta
```json
{
    "type": "content.delta",
    "delta": {
        "type": "audio",
        "audio": "base64_audio_data"
    }
}
```

### Error Response
```json
{
    "type": "error",
    "error": {
        "message": "Invalid request",
        "code": "INVALID_REQUEST"
    }
}
```

### Response Done
```json
{
    "type": "response.done",
    "response": {
        "id": "response-123",
        "model": "gemini-2.0-flash-live-preview-04-09",
        "created": 1234567890
    }
}
```

## Limitations

1. **Regional Availability**: Vertex AI Live API is only available in specific regions
2. **Model Support**: Limited to Gemini Live models
3. **Rate Limits**: Subject to Vertex AI rate limits
4. **Authentication**: Requires Google Cloud authentication
5. **WebSocket Only**: Real-time features require WebSocket connections

## Error Handling

Common error scenarios and solutions:

### Authentication Errors
```json
{
    "type": "error",
    "error": {
        "message": "Authentication error: Vertex AI project ID is required",
        "code": "AUTH_ERROR"
    }
}
```

### Model Not Found
```json
{
    "type": "error",
    "error": {
        "message": "Model not found: gemini-2.0-flash-live-preview-04-09",
        "code": "MODEL_NOT_FOUND"
    }
}
```

### Rate Limit Exceeded
```json
{
    "type": "error",
    "error": {
        "message": "Rate limit exceeded",
        "code": "RATE_LIMIT_EXCEEDED"
    }
}
```

## Best Practices

1. **Session Management**: Always send session configuration before sending inputs
2. **Error Handling**: Implement proper error handling for WebSocket connections
3. **Resource Cleanup**: Ensure proper WebSocket connection cleanup
4. **Authentication**: Use environment variables for sensitive credentials
5. **Monitoring**: Monitor connection status and implement reconnection logic

## Examples

### Complete Example with Error Handling

```python
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def vertex_ai_realtime_chat():
    uri = "ws://localhost:4000/v1/realtime?model=vertex_ai/gemini-2.0-flash-live-preview-04-09"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to Vertex AI Realtime API")
            
            # Send session configuration
            session_config = {
                "type": "session.update",
                "session": {
                    "max_tokens": 1000,
                    "temperature": 0.8,
                    "system_instruction": "You are a helpful AI assistant."
                }
            }
            await websocket.send(json.dumps(session_config))
            logger.info("Session configuration sent")
            
            # Send user input
            user_input = "Hello! Can you help me with Python programming?"
            await websocket.send(user_input)
            logger.info(f"User input sent: {user_input}")
            
            # Receive and process responses
            async for message in websocket:
                try:
                    response = json.loads(message)
                    logger.info(f"Received response: {response}")
                    
                    if response.get("type") == "content.delta":
                        print(f"AI: {response['delta']['text']}", end="", flush=True)
                    elif response.get("type") == "response.done":
                        print("\n[Response completed]")
                        break
                    elif response.get("type") == "error":
                        logger.error(f"Error: {response['error']}")
                        break
                        
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse message: {message}")
                    
    except websockets.exceptions.ConnectionClosed as e:
        logger.error(f"Connection closed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(vertex_ai_realtime_chat())
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if the LiteLLM proxy is running
2. **Authentication Failed**: Verify your Google Cloud credentials
3. **Model Not Found**: Ensure the model name is correct and available
4. **WebSocket Errors**: Check network connectivity and firewall settings

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

Test the connection before sending real requests:

```python
from litellm.realtime_api.main import _realtime_health_check

async def test_connection():
    try:
        await _realtime_health_check(
            model="vertex_ai/gemini-2.0-flash-live-preview-04-09",
            custom_llm_provider="vertex_ai",
            api_key=None,
            api_base=None
        )
        print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {e}")
```

"""
Vertex AI Realtime API configuration and transformation classes.

This module provides the configuration and transformation logic for Vertex AI's
Realtime (Live) API, enabling it to work through LiteLLM's unified realtime interface.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.realtime import (
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class VertexAIRealtimeConfig(BaseRealtimeConfig, VertexBase):
    """
    Configuration class for Vertex AI Realtime (Live) API.
    
    This class handles the transformation between OpenAI-style realtime requests
    and Vertex AI Live API format, including authentication and URL construction.
    """

    def __init__(self):
        super().__init__()
        self.llm_provider = "vertex_ai"
        self.vertex_project = None
        self.vertex_location = None

    def validate_environment(
        self, headers: dict, model: str, api_key: Optional[str] = None
    ) -> dict:
        """
        Validate and prepare headers for Vertex AI Live API.
        
        Args:
            headers: Incoming request headers
            model: Model name (e.g., "gemini-2.0-flash-live-preview-04-09")
            api_key: Optional API key (not used for Vertex AI)
            
        Returns:
            dict: Headers prepared for Vertex AI Live API
        """
        # Extract project and location from headers or use defaults
        project_id = headers.get("x-goog-user-project") or self.vertex_project
        location = headers.get("x-goog-vertex-location") or self.vertex_location
        
        if not project_id:
            raise ValueError("Vertex AI project ID is required. Set x-goog-user-project header or VERTEXAI_PROJECT environment variable.")
        
        # Prepare headers for Vertex AI Live API
        vertex_headers = {
            "Content-Type": "application/json",
            "x-goog-user-project": project_id,
        }
        
        if location:
            vertex_headers["x-goog-vertex-location"] = location
            
        # Forward any other x-goog-* headers
        for key, value in headers.items():
            if key.lower().startswith("x-goog-") and key not in vertex_headers:
                vertex_headers[key] = value
                
        return vertex_headers

    def get_complete_url(
        self, api_base: Optional[str], model: str, api_key: Optional[str] = None
    ) -> str:
        """
        Get the complete WebSocket URL for Vertex AI Live API.
        
        Args:
            api_base: Base URL (not used for Vertex AI)
            model: Model name
            api_key: API key (not used for Vertex AI)
            
        Returns:
            str: Complete WebSocket URL for Vertex AI Live API
        """
        # Determine the region based on the model
        location = self.get_vertex_region(
            vertex_region=self.vertex_location or "us-central1",
            model=model,
        )
        
        # Construct the host based on location
        if location == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{location}-aiplatform.googleapis.com"
            
        # Return the WebSocket URL for Vertex AI Live API
        return f"wss://{host}/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: Optional[str] = None,
    ) -> List[str]:
        """
        Transform OpenAI-style realtime request to Vertex AI Live API format.
        
        Args:
            message: The incoming message (JSON string)
            model: Model name
            session_configuration_request: Optional session configuration
            
        Returns:
            List[str]: Transformed messages for Vertex AI Live API
        """
        messages: List[str] = []
        
        # Try to parse as JSON first
        try:
            json_message = json.loads(message)
            
            # Handle session configuration
            if "type" in json_message and json_message["type"] == "session.update":
                session_config = self._transform_session_config(
                    json_message.get("session", {}), model
                )
                messages.append(json.dumps({"setup": session_config}))
                return messages

            # Handle input audio buffer
            if (
                "type" in json_message
                and json_message["type"] == "input_audio_buffer.append"
            ):
                audio_data = json_message.get("audio", "")
                realtime_input = {
                    "audio": {
                        "mimeType": self._get_audio_mime_type(),
                        "data": audio_data
                    }
                }
            else:
                # Handle other JSON messages as text
                realtime_input = {"text": message}
        except json.JSONDecodeError:
            # If not JSON, treat as plain text input
            realtime_input = {"text": message}

        # Create the realtime input message
        realtime_message = {
            "input": realtime_input
        }
        
        messages.append(json.dumps(realtime_message))
        return messages

    def transform_realtime_response(
        self,
        message: Union[str, bytes],
        model: str,
        logging_obj: Any,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        """
        Transform Vertex AI Live API response to OpenAI-style format.
        
        Args:
            message: Response message from Vertex AI Live API
            model: Model name
            logging_obj: Logging object
            realtime_response_transform_input: Input for response transformation
            
        Returns:
            RealtimeResponseTypedDict: Transformed response
        """
        try:
            if isinstance(message, bytes):
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = str(message)
                
            json_message = json.loads(message_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            return {
                "type": "error",
                "error": {
                    "message": f"Failed to parse response: {str(e)}",
                    "code": "parse_error"
                }
            }

        # Handle different types of responses from Vertex AI Live API
        if "setup" in json_message:
            # Session setup response
            return {
                "type": "session.created",
                "session": {
                    "id": json_message.get("setup", {}).get("sessionId", ""),
                    "model": model,
                    "created_at": json_message.get("setup", {}).get("createdAt", 0)
                }
            }
        elif "output" in json_message:
            # Content output response
            output = json_message["output"]
            
            if "text" in output:
                return {
                    "type": "content.delta",
                    "delta": {
                        "type": "text",
                        "text": output["text"]
                    }
                }
            elif "audio" in output:
                return {
                    "type": "content.delta",
                    "delta": {
                        "type": "audio",
                        "audio": output["audio"]
                    }
                }
        elif "error" in json_message:
            # Error response
            return {
                "type": "error",
                "error": {
                    "message": json_message["error"].get("message", "Unknown error"),
                    "code": json_message["error"].get("code", "unknown_error")
                }
            }
        elif "done" in json_message:
            # Completion response
            return {
                "type": "response.done",
                "response": {
                    "id": json_message.get("responseId", ""),
                    "model": model,
                    "created": json_message.get("createdAt", 0)
                }
            }

        # Default fallback
        return {
            "type": "content.delta",
            "delta": {
                "type": "text",
                "text": message_str
            }
        }

    def requires_session_configuration(self) -> bool:
        """
        Check if session configuration is required.
        
        Returns:
            bool: True if session configuration is required
        """
        return True

    def session_configuration_request(self, model: str) -> Optional[str]:
        """
        Get the default session configuration request.
        
        Args:
            model: Model name
            
        Returns:
            Optional[str]: Session configuration request JSON
        """
        config = {
            "model": f"models/{model}",
            "generationConfig": {
                "maxOutputTokens": 8192,
                "temperature": 0.7,
                "topP": 0.8,
                "topK": 40
            },
            "systemInstruction": {
                "parts": [{"text": "You are a helpful AI assistant."}]
            },
            "tools": [],
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        return json.dumps({"setup": config})

    def _transform_session_config(self, session_config: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Transform OpenAI-style session configuration to Vertex AI format.
        
        Args:
            session_config: OpenAI-style session configuration
            model: Model name
            
        Returns:
            Dict[str, Any]: Vertex AI format session configuration
        """
        vertex_config = {
            "model": f"models/{model}",
            "generationConfig": {},
            "systemInstruction": {"parts": []},
            "tools": [],
            "safetySettings": []
        }
        
        # Map OpenAI parameters to Vertex AI format
        if "max_tokens" in session_config:
            vertex_config["generationConfig"]["maxOutputTokens"] = session_config["max_tokens"]
            
        if "temperature" in session_config:
            vertex_config["generationConfig"]["temperature"] = session_config["temperature"]
            
        if "top_p" in session_config:
            vertex_config["generationConfig"]["topP"] = session_config["top_p"]
            
        if "top_k" in session_config:
            vertex_config["generationConfig"]["topK"] = session_config["top_k"]
            
        if "system_instruction" in session_config:
            vertex_config["systemInstruction"]["parts"] = [
                {"text": session_config["system_instruction"]}
            ]
            
        if "tools" in session_config:
            vertex_config["tools"] = self._transform_tools(session_config["tools"])
            
        if "safety_settings" in session_config:
            vertex_config["safetySettings"] = self._transform_safety_settings(
                session_config["safety_settings"]
            )
            
        return vertex_config

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform OpenAI-style tools to Vertex AI format.
        
        Args:
            tools: List of OpenAI-style tools
            
        Returns:
            List[Dict[str, Any]]: Vertex AI format tools
        """
        vertex_tools = []
        
        for tool in tools:
            if tool.get("type") == "function":
                vertex_tool = {
                    "functionDeclarations": [{
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"].get("parameters", {})
                    }]
                }
                vertex_tools.append(vertex_tool)
                
        return vertex_tools

    def _transform_safety_settings(
        self, safety_settings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transform OpenAI-style safety settings to Vertex AI format.
        
        Args:
            safety_settings: List of OpenAI-style safety settings
            
        Returns:
            List[Dict[str, Any]]: Vertex AI format safety settings
        """
        vertex_safety_settings = []
        
        # Map OpenAI safety categories to Vertex AI categories
        category_mapping = {
            "harassment": "HARM_CATEGORY_HARASSMENT",
            "hate": "HARM_CATEGORY_HATE_SPEECH",
            "sexual": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "violence": "HARM_CATEGORY_DANGEROUS_CONTENT"
        }
        
        for setting in safety_settings:
            category = setting.get("category", "").lower()
            threshold = setting.get("threshold", "BLOCK_MEDIUM_AND_ABOVE")
            
            if category in category_mapping:
                vertex_safety_settings.append({
                    "category": category_mapping[category],
                    "threshold": threshold
                })
                
        return vertex_safety_settings

    def _get_audio_mime_type(self) -> str:
        """
        Get the MIME type for audio data.
        
        Returns:
            str: Audio MIME type
        """
        return "audio/pcm"

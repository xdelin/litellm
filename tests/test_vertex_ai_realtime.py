"""
Test cases for Vertex AI Realtime API integration.

This module contains comprehensive tests for the Vertex AI Realtime (Live) API
integration through LiteLLM's unified realtime interface.
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
from litellm.llms.vertex_ai.realtime.handler import VertexAIRealtime


class TestVertexAIRealtimeConfig:
    """Test cases for VertexAIRealtimeConfig class."""

    def test_init(self):
        """Test initialization of VertexAIRealtimeConfig."""
        config = VertexAIRealtimeConfig()
        assert config.llm_provider == "vertex_ai"

    def test_validate_environment_with_project_header(self):
        """Test validate_environment with project header."""
        config = VertexAIRealtimeConfig()
        headers = {
            "x-goog-user-project": "test-project",
            "x-goog-vertex-location": "us-central1"
        }
        
        result = config.validate_environment(headers, "gemini-2.0-flash-live-preview-04-09")
        
        assert result["x-goog-user-project"] == "test-project"
        assert result["x-goog-vertex-location"] == "us-central1"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_missing_project(self):
        """Test validate_environment with missing project raises error."""
        config = VertexAIRealtimeConfig()
        headers = {}
        
        with pytest.raises(ValueError, match="Vertex AI project ID is required"):
            config.validate_environment(headers, "gemini-2.0-flash-live-preview-04-09")

    def test_get_complete_url_global_model(self):
        """Test get_complete_url for global model."""
        config = VertexAIRealtimeConfig()
        config.vertex_location = "global"
        
        url = config.get_complete_url(None, "gemini-2.0-flash-live-preview-04-09", None)
        
        assert url == "wss://aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

    def test_get_complete_url_regional_model(self):
        """Test get_complete_url for regional model."""
        config = VertexAIRealtimeConfig()
        config.vertex_location = "us-central1"
        
        url = config.get_complete_url(None, "gemini-2.0-flash-live-preview-04-09", None)
        
        assert url == "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

    def test_transform_realtime_request_session_update(self):
        """Test transform_realtime_request for session update."""
        config = VertexAIRealtimeConfig()
        message = json.dumps({
            "type": "session.update",
            "session": {
                "max_tokens": 1000,
                "temperature": 0.8,
                "system_instruction": "You are a helpful assistant."
            }
        })
        
        result = config.transform_realtime_request(message, "gemini-2.0-flash-live-preview-04-09")
        
        assert len(result) == 1
        setup_data = json.loads(result[0])
        assert "setup" in setup_data
        assert setup_data["setup"]["model"] == "models/gemini-2.0-flash-live-preview-04-09"
        assert setup_data["setup"]["generationConfig"]["maxOutputTokens"] == 1000
        assert setup_data["setup"]["generationConfig"]["temperature"] == 0.8

    def test_transform_realtime_request_text_input(self):
        """Test transform_realtime_request for text input."""
        config = VertexAIRealtimeConfig()
        message = "Hello, how are you?"
        
        result = config.transform_realtime_request(message, "gemini-2.0-flash-live-preview-04-09")
        
        assert len(result) == 1
        input_data = json.loads(result[0])
        assert input_data["input"]["text"] == "Hello, how are you?"

    def test_transform_realtime_request_audio_input(self):
        """Test transform_realtime_request for audio input."""
        config = VertexAIRealtimeConfig()
        message = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": "base64_audio_data"
        })
        
        result = config.transform_realtime_request(message, "gemini-2.0-flash-live-preview-04-09")
        
        assert len(result) == 1
        input_data = json.loads(result[0])
        assert "input" in input_data
        assert "audio" in input_data["input"]
        assert input_data["input"]["audio"]["mimeType"] == "audio/pcm"

    def test_transform_realtime_response_setup(self):
        """Test transform_realtime_response for setup response."""
        config = VertexAIRealtimeConfig()
        message = json.dumps({
            "setup": {
                "sessionId": "session-123",
                "createdAt": 1234567890
            }
        })
        
        result = config.transform_realtime_response(
            message, "gemini-2.0-flash-live-preview-04-09", Mock(), Mock()
        )
        
        assert result["type"] == "session.created"
        assert result["session"]["id"] == "session-123"
        assert result["session"]["model"] == "gemini-2.0-flash-live-preview-04-09"

    def test_transform_realtime_response_text_output(self):
        """Test transform_realtime_response for text output."""
        config = VertexAIRealtimeConfig()
        message = json.dumps({
            "output": {
                "text": "Hello! How can I help you?"
            }
        })
        
        result = config.transform_realtime_response(
            message, "gemini-2.0-flash-live-preview-04-09", Mock(), Mock()
        )
        
        assert result["type"] == "content.delta"
        assert result["delta"]["type"] == "text"
        assert result["delta"]["text"] == "Hello! How can I help you?"

    def test_transform_realtime_response_error(self):
        """Test transform_realtime_response for error response."""
        config = VertexAIRealtimeConfig()
        message = json.dumps({
            "error": {
                "message": "Invalid request",
                "code": "INVALID_REQUEST"
            }
        })
        
        result = config.transform_realtime_response(
            message, "gemini-2.0-flash-live-preview-04-09", Mock(), Mock()
        )
        
        assert result["type"] == "error"
        assert result["error"]["message"] == "Invalid request"
        assert result["error"]["code"] == "INVALID_REQUEST"

    def test_requires_session_configuration(self):
        """Test requires_session_configuration returns True."""
        config = VertexAIRealtimeConfig()
        assert config.requires_session_configuration() is True

    def test_session_configuration_request(self):
        """Test session_configuration_request returns valid config."""
        config = VertexAIRealtimeConfig()
        result = config.session_configuration_request("gemini-2.0-flash-live-preview-04-09")
        
        assert result is not None
        config_data = json.loads(result)
        assert "setup" in config_data
        assert config_data["setup"]["model"] == "models/gemini-2.0-flash-live-preview-04-09"
        assert "generationConfig" in config_data["setup"]
        assert "safetySettings" in config_data["setup"]


class TestVertexAIRealtime:
    """Test cases for VertexAIRealtime handler class."""

    def test_init(self):
        """Test initialization of VertexAIRealtime."""
        handler = VertexAIRealtime()
        assert handler.realtime_config is not None
        assert isinstance(handler.realtime_config, VertexAIRealtimeConfig)

    def test_get_realtime_config(self):
        """Test get_realtime_config returns the config."""
        handler = VertexAIRealtime()
        config = handler.get_realtime_config()
        assert config is not None
        assert isinstance(config, VertexAIRealtimeConfig)

    @pytest.mark.asyncio
    async def test_async_realtime_authentication_error(self):
        """Test async_realtime with authentication error."""
        handler = VertexAIRealtime()
        websocket = AsyncMock()
        websocket.close = AsyncMock()
        
        # Mock validate_environment to raise an error
        with patch.object(handler.realtime_config, 'validate_environment', side_effect=ValueError("Auth error")):
            await handler.async_realtime(
                model="gemini-2.0-flash-live-preview-04-09",
                websocket=websocket,
                logging_obj=Mock(),
                query_params={"vertex_project": "test-project"}
            )
            
            websocket.close.assert_called_once_with(code=400, reason="Authentication error: Auth error")

    @pytest.mark.asyncio
    async def test_async_realtime_access_token_error(self):
        """Test async_realtime with access token error."""
        handler = VertexAIRealtime()
        websocket = AsyncMock()
        websocket.close = AsyncMock()
        
        # Mock validate_environment to succeed
        with patch.object(handler.realtime_config, 'validate_environment', return_value={"x-goog-user-project": "test-project"}):
            # Mock _ensure_access_token_async to raise an error
            with patch.object(handler, '_ensure_access_token_async', side_effect=Exception("Token error")):
                await handler.async_realtime(
                    model="gemini-2.0-flash-live-preview-04-09",
                    websocket=websocket,
                    logging_obj=Mock(),
                    query_params={"vertex_project": "test-project"}
                )
                
                websocket.close.assert_called_once_with(code=401, reason="Authentication failed: Token error")

    @pytest.mark.asyncio
    async def test_async_realtime_success(self):
        """Test async_realtime with successful connection."""
        handler = VertexAIRealtime()
        websocket = AsyncMock()
        
        # Mock validate_environment to succeed
        with patch.object(handler.realtime_config, 'validate_environment', return_value={"x-goog-user-project": "test-project"}):
            # Mock _ensure_access_token_async to succeed
            with patch.object(handler, '_ensure_access_token_async', return_value=("access_token", "test-project")):
                # Mock websockets.connect
                with patch('websockets.connect') as mock_connect:
                    mock_backend_ws = AsyncMock()
                    mock_connect.return_value.__aenter__.return_value = mock_backend_ws
                    
                    # Mock RealTimeStreaming
                    with patch('litellm.litellm_core_utils.realtime_streaming.RealTimeStreaming') as mock_streaming:
                        mock_streaming_instance = AsyncMock()
                        mock_streaming.return_value = mock_streaming_instance
                        
                        await handler.async_realtime(
                            model="gemini-2.0-flash-live-preview-04-09",
                            websocket=websocket,
                            logging_obj=Mock(),
                            query_params={"vertex_project": "test-project"}
                        )
                        
                        # Verify websockets.connect was called with correct URL and headers
                        mock_connect.assert_called_once()
                        call_args = mock_connect.call_args
                        assert "wss://" in call_args[0][0]  # URL should start with wss://
                        assert "aiplatform.googleapis.com" in call_args[0][0]  # Should contain Vertex AI domain
                        
                        # Verify RealTimeStreaming was called
                        mock_streaming.assert_called_once()
                        mock_streaming_instance.bidirectional_forward.assert_called_once()


class TestIntegration:
    """Integration tests for Vertex AI Realtime API."""

    def test_provider_config_manager_integration(self):
        """Test that Vertex AI realtime config is properly registered."""
        from litellm.utils import ProviderConfigManager
        from litellm.types.utils import LlmProviders
        
        config = ProviderConfigManager.get_provider_realtime_config(
            model="gemini-2.0-flash-live-preview-04-09",
            provider=LlmProviders.VERTEX_AI
        )
        
        assert config is not None
        assert isinstance(config, VertexAIRealtimeConfig)

    def test_model_detection(self):
        """Test that Vertex AI models are properly detected."""
        from litellm import get_llm_provider
        
        model, provider, _, _ = get_llm_provider("vertex_ai/gemini-2.0-flash-live-preview-04-09")
        assert provider == "vertex_ai"
        assert model == "gemini-2.0-flash-live-preview-04-09"


if __name__ == "__main__":
    pytest.main([__file__])

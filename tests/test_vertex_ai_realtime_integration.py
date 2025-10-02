"""
Integration tests for Vertex AI Realtime API.

This module contains integration tests to verify that the Vertex AI Realtime API
integration works correctly with the LiteLLM proxy server.
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from litellm.realtime_api.main import _arealtime, _realtime_health_check


class TestVertexAIRealtimeIntegration:
    """Integration tests for Vertex AI Realtime API."""

    @pytest.mark.asyncio
    async def test_realtime_function_with_vertex_ai_model(self):
        """Test that the realtime function correctly routes Vertex AI models."""
        # Mock the necessary components
        websocket = AsyncMock()
        logging_obj = Mock()
        
        # Mock the provider config manager to return our config
        with patch('litellm.utils.ProviderConfigManager.get_provider_realtime_config') as mock_get_config:
            from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
            mock_get_config.return_value = VertexAIRealtimeConfig()
            
            # Mock the base LLM HTTP handler
            with patch('litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_realtime') as mock_async_realtime:
                await _arealtime(
                    model="vertex_ai/gemini-2.0-flash-live-preview-04-09",
                    websocket=websocket,
                    logging_obj=logging_obj,
                    headers={"x-goog-user-project": "test-project"}
                )
                
                # Verify that async_realtime was called
                mock_async_realtime.assert_called_once()

    @pytest.mark.asyncio
    async def test_realtime_health_check_with_vertex_ai(self):
        """Test that the health check function works with Vertex AI models."""
        # Mock the vertex AI realtime handler
        with patch('litellm.realtime_api.main.vertex_ai_realtime') as mock_vertex_ai:
            mock_config = Mock()
            mock_config.get_complete_url.return_value = "wss://test-aiplatform.googleapis.com/ws/test"
            mock_vertex_ai.get_realtime_config.return_value = mock_config
            
            # Mock websockets.connect
            with patch('websockets.connect') as mock_connect:
                mock_ws = AsyncMock()
                mock_connect.return_value.__aenter__.return_value = mock_ws
                
                result = await _realtime_health_check(
                    model="vertex_ai/gemini-2.0-flash-live-preview-04-09",
                    custom_llm_provider="vertex_ai",
                    api_key=None,
                    api_base=None
                )
                
                assert result is True
                mock_connect.assert_called_once()

    def test_model_provider_detection(self):
        """Test that Vertex AI models are correctly detected."""
        from litellm import get_llm_provider
        
        # Test with vertex_ai/ prefix
        model, provider, _, _ = get_llm_provider("vertex_ai/gemini-2.0-flash-live-preview-04-09")
        assert provider == "vertex_ai"
        assert model == "gemini-2.0-flash-live-preview-04-09"
        
        # Test with gemini-2.0-flash-live-preview-04-09 (should be detected as vertex_ai for live models)
        model, provider, _, _ = get_llm_provider("gemini-2.0-flash-live-preview-04-09")
        assert provider == "vertex_ai"  # Live models are detected as vertex_ai
        assert model == "gemini-2.0-flash-live-preview-04-09"

    def test_provider_config_manager_integration(self):
        """Test that the provider config manager correctly returns Vertex AI config."""
        from litellm.utils import ProviderConfigManager
        from litellm.types.utils import LlmProviders
        
        config = ProviderConfigManager.get_provider_realtime_config(
            model="gemini-2.0-flash-live-preview-04-09",
            provider=LlmProviders.VERTEX_AI
        )
        
        assert config is not None
        assert hasattr(config, 'llm_provider')
        assert config.llm_provider == "vertex_ai"

    @pytest.mark.asyncio
    async def test_vertex_ai_realtime_config_validation(self):
        """Test that the Vertex AI realtime config validates correctly."""
        from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
        
        config = VertexAIRealtimeConfig()
        
        # Test with valid headers
        headers = {
            "x-goog-user-project": "test-project",
            "x-goog-vertex-location": "us-central1"
        }
        
        result = config.validate_environment(headers, "gemini-2.0-flash-live-preview-04-09")
        assert result["x-goog-user-project"] == "test-project"
        assert result["x-goog-vertex-location"] == "us-central1"
        assert result["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_vertex_ai_realtime_url_generation(self):
        """Test that the Vertex AI realtime config generates correct URLs."""
        from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
        
        config = VertexAIRealtimeConfig()
        config.vertex_location = "us-central1"
        
        url = config.get_complete_url(None, "gemini-2.0-flash-live-preview-04-09", None)
        assert "wss://" in url
        assert "aiplatform.googleapis.com" in url
        assert "BidiGenerateContent" in url

    def test_vertex_ai_realtime_message_transformation(self):
        """Test that messages are correctly transformed for Vertex AI Live API."""
        from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
        
        config = VertexAIRealtimeConfig()
        
        # Test text input transformation
        result = config.transform_realtime_request("Hello, world!", "gemini-2.0-flash-live-preview-04-09")
        assert len(result) == 1
        
        message = json.loads(result[0])
        assert "input" in message
        assert message["input"]["text"] == "Hello, world!"
        
        # Test session update transformation
        session_message = json.dumps({
            "type": "session.update",
            "session": {
                "max_tokens": 1000,
                "temperature": 0.8
            }
        })
        
        result = config.transform_realtime_request(session_message, "gemini-2.0-flash-live-preview-04-09")
        assert len(result) == 1
        
        message = json.loads(result[0])
        assert "setup" in message
        assert message["setup"]["model"] == "models/gemini-2.0-flash-live-preview-04-09"

    def test_vertex_ai_realtime_response_transformation(self):
        """Test that responses are correctly transformed from Vertex AI Live API."""
        from litellm.llms.vertex_ai.realtime.transformation import VertexAIRealtimeConfig
        
        config = VertexAIRealtimeConfig()
        
        # Test text output transformation
        response = json.dumps({
            "output": {
                "text": "Hello! How can I help you?"
            }
        })
        
        result = config.transform_realtime_response(
            response, "gemini-2.0-flash-live-preview-04-09", Mock(), Mock()
        )
        
        assert result["type"] == "content.delta"
        assert result["delta"]["type"] == "text"
        assert result["delta"]["text"] == "Hello! How can I help you?"


if __name__ == "__main__":
    pytest.main([__file__])

"""
Test file for LumaImageToImageNode (Task #2)

This test file validates the LumaImageToImageNode implementation according to the challenge requirements:
- Required inputs: image, prompt
- Optional inputs: image_weight (0.02-1.0, default 0.5), model (default luma-v3), 
  aspect_ratio (default 1:1), seed, timeout
- Calls Luma's image-modify endpoint (POST /proxy/luma/generations/image)
- Authenticated with LUMA_API_KEY
- If LUMA_API_KEY is missing, surfaces a clear, non-crashing error
- Follows the official ComfyUI Luma Image to Image node pattern
"""

import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
from comfy_api_nodes.nodes_luma import LumaImageToImageNode


class TestLumaImageToImageNode:
    """Test cases for LumaImageToImageNode (Task #2)"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.node = LumaImageToImageNode()
        self.sample_image = torch.randn(1, 3, 512, 512)  # Batch, channels, height, width
        self.sample_prompt = "A beautiful sunset over mountains"
        
    def test_node_class_exists(self):
        """Test that the LumaImageToImageNode class exists"""
        assert LumaImageToImageNode is not None
        assert hasattr(LumaImageToImageNode, 'INPUT_TYPES')
        assert hasattr(LumaImageToImageNode, 'api_call')
        
    def test_input_types_structure(self):
        """Test that INPUT_TYPES has the correct structure"""
        input_types = self.node.INPUT_TYPES()
        
        # Check required inputs
        assert 'required' in input_types
        required = input_types['required']
        assert 'image' in required
        assert 'prompt' in required
        
        # Check optional inputs
        assert 'optional' in input_types
        optional = input_types['optional']
        assert 'image_weight' in optional
        assert 'model' in optional
        assert 'aspect_ratio' in optional
        assert 'seed' in optional
        assert 'timeout' in optional
        
        # Check hidden inputs
        assert 'hidden' in input_types
        hidden = input_types['hidden']
        assert 'auth_token' in hidden
        assert 'comfy_api_key' in hidden
        assert 'unique_id' in hidden
        
    def test_required_inputs(self):
        """Test that required inputs are properly defined"""
        input_types = self.node.INPUT_TYPES()
        required = input_types['required']
        
        # Check image input
        image_input = required['image']
        assert image_input[0] == 'IMAGE'  # IO.IMAGE
        
        # Check prompt input
        prompt_input = required['prompt']
        assert prompt_input[0] == 'STRING'  # IO.STRING
        assert 'multiline' in prompt_input[1]
        assert prompt_input[1]['multiline'] is True
        
    def test_optional_inputs(self):
        """Test that optional inputs have correct defaults and constraints"""
        input_types = self.node.INPUT_TYPES()
        optional = input_types['optional']
        
        # Check image_weight
        image_weight_input = optional['image_weight']
        assert image_weight_input[0] == 'FLOAT'  # IO.FLOAT
        config = image_weight_input[1]
        assert config['default'] == 0.5
        assert config['min'] == 0.02
        assert config['max'] == 1.0
        assert config['step'] == 0.01
        
        # Check model
        model_input = optional['model']
        assert isinstance(model_input[0], list)  # List of model options
        config = model_input[1]
        assert config['default'] == 'luma-v3'
        
        # Check timeout
        timeout_input = optional['timeout']
        assert timeout_input[0] == 'INT'  # IO.INT
        config = timeout_input[1]
        assert config['default'] == 120
        assert config['min'] == 30
        assert config['max'] == 600
        
    def test_missing_api_key_error(self):
        """Test that missing API key raises a clear error"""
        with pytest.raises(Exception) as exc_info:
            self.node.api_call(
                prompt=self.sample_prompt,
                image=self.sample_image,
                comfy_api_key=None  # Missing API key
            )
        
        error_message = str(exc_info.value)
        assert "LUMA_API_KEY is required" in error_message
        assert "environment variable" in error_message
        
    @patch('comfy_api_nodes.nodes_luma.upload_images_to_comfyapi')
    @patch('comfy_api_nodes.nodes_luma.SynchronousOperation')
    @patch('comfy_api_nodes.nodes_luma.PollingOperation')
    @patch('comfy_api_nodes.nodes_luma.requests.get')
    def test_api_call_with_valid_inputs(self, mock_requests_get, mock_polling, mock_sync, mock_upload):
        """Test successful API call with valid inputs"""
        # Mock upload_images_to_comfyapi
        mock_upload.return_value = ['https://example.com/image.png']
        
        # Mock SynchronousOperation
        mock_sync_instance = Mock()
        mock_sync_instance.execute.return_value = Mock(id='test-task-id')
        mock_sync.return_value = mock_sync_instance
        
        # Mock PollingOperation
        mock_polling_instance = Mock()
        mock_polling_instance.execute.return_value = Mock(assets=Mock(image='https://example.com/result.png'))
        mock_polling.return_value = mock_polling_instance
        
        # Mock requests.get
        mock_response = Mock()
        mock_response.content = b'fake_image_data'
        mock_requests_get.return_value = mock_response
        
        # Mock process_image_response
        with patch('comfy_api_nodes.nodes_luma.process_image_response') as mock_process:
            mock_process.return_value = torch.randn(1, 3, 512, 512)
            
            result = self.node.api_call(
                prompt=self.sample_prompt,
                image=self.sample_image,
                image_weight=0.5,
                model='luma-v3',
                aspect_ratio='1:1',
                seed=42,
                timeout=120,
                comfy_api_key='test-api-key'
            )
            
            # Verify upload was called
            mock_upload.assert_called_once()
            
            # Verify SynchronousOperation was called with correct parameters
            mock_sync.assert_called_once()
            call_args = mock_sync.call_args
            assert call_args[1]['request'].prompt == self.sample_prompt
            assert call_args[1]['request'].model == 'luma-v3'
            
            # Verify result is a tuple with image tensor
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            
    def test_input_validation(self):
        """Test input validation"""
        # Test empty prompt
        with pytest.raises(Exception):
            self.node.api_call(
                prompt="",  # Empty prompt
                image=self.sample_image,
                comfy_api_key='test-api-key'
            )
            
        # Test short prompt
        with pytest.raises(Exception):
            self.node.api_call(
                prompt="ab",  # Too short
                image=self.sample_image,
                comfy_api_key='test-api-key'
            )
            
    def test_image_weight_constraints(self):
        """Test image_weight parameter constraints"""
        input_types = self.node.INPUT_TYPES()
        optional = input_types['optional']
        image_weight_config = optional['image_weight'][1]
        
        # Test default value
        assert image_weight_config['default'] == 0.5
        
        # Test min/max constraints
        assert image_weight_config['min'] == 0.02
        assert image_weight_config['max'] == 1.0
        
    def test_timeout_parameter(self):
        """Test timeout parameter configuration"""
        input_types = self.node.INPUT_TYPES()
        optional = input_types['optional']
        timeout_config = optional['timeout'][1]
        
        # Test default and constraints
        assert timeout_config['default'] == 120
        assert timeout_config['min'] == 30
        assert timeout_config['max'] == 600
        assert timeout_config['step'] == 10
        
    def test_node_registration(self):
        """Test that the node is properly registered"""
        from comfy_api_nodes.nodes_luma import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        # Check that the node is in the mappings
        assert 'LumaImageToImageNode' in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS['LumaImageToImageNode'] == LumaImageToImageNode
        
        # Check display name
        assert 'LumaImageToImageNode' in NODE_DISPLAY_NAME_MAPPINGS
        assert NODE_DISPLAY_NAME_MAPPINGS['LumaImageToImageNode'] == 'Luma Image to Image (Task #2)'


if __name__ == '__main__':
    pytest.main([__file__]) 
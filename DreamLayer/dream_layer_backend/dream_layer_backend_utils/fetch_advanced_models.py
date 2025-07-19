"""
Advanced Model Fetcher Utility

This module provides utilities to fetch and manage advanced models like ControlNet models,
LoRA models, and other extensions from the ComfyUI directory structure.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def get_settings():
    """Load settings from file"""
    try:
        settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    return {}

def is_valid_directory(dir_path: Optional[str]) -> bool:
    """Check if directory path is valid (not None and doesn't start with '/path')"""
    if dir_path is None:
        return True  # None is valid, get_directories will handle it
    dir_path_lower = dir_path.lower()
    return not dir_path_lower.startswith('/path')

def get_controlnet_models() -> List[str]:
    """
    Get a list of available ControlNet models from the ComfyUI models directory.
    Reads path from settings.json if available, otherwise uses default path.
    
    Returns:
        List[str]: List of ControlNet model filenames
    """
    try:
        # Get settings and check for custom controlnet path
        settings = get_settings()
        controlnet_path = settings.get('controlNetModelsPath')
        
        if controlnet_path and is_valid_directory(controlnet_path):
            # Use custom path from settings
            controlnet_dir = os.path.abspath(controlnet_path)
            logger.info(f"Using custom ControlNet directory from settings: {controlnet_dir}")
        else:
            # Fallback to default path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            controlnet_dir = os.path.join(current_dir, 'ComfyUI', 'models', 'controlnet')
            logger.info(f"Using default ControlNet directory: {controlnet_dir}")
        
        if not os.path.exists(controlnet_dir):
            logger.warning(f"ControlNet directory not found: {controlnet_dir}")
            return []
        
        # List all files in the controlnet directory
        model_files = []
        for filename in os.listdir(controlnet_dir):
            # Include common ControlNet model extensions
            if filename.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                model_files.append(filename)
        
        logger.info(f"Found {len(model_files)} ControlNet models")
        return sorted(model_files)
        
    except Exception as e:
        logger.error(f"Error fetching ControlNet models: {str(e)}")
        return []

def get_lora_models() -> List[str]:
    """
    Get a list of available LoRA models from the ComfyUI models directory.
    Reads path from settings.json if available, otherwise uses default path.
    
    Returns:
        List[str]: List of LoRA model filenames
    """
    try:
        # Get settings and check for custom lora path
        settings = get_settings()
        lora_path = settings.get('loraEmbeddingsPath')
        
        if lora_path and is_valid_directory(lora_path):
            # Use custom path from settings
            lora_dir = os.path.abspath(lora_path)
            logger.info(f"Using custom LoRA directory from settings: {lora_dir}")
        else:
            # Fallback to default path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            lora_dir = os.path.join(current_dir, 'ComfyUI', 'models', 'loras')
            logger.info(f"Using default LoRA directory: {lora_dir}")
        
        if not os.path.exists(lora_dir):
            logger.warning(f"LoRA directory not found: {lora_dir}")
            return []
        
        # List all files in the lora directory
        model_files = []
        for filename in os.listdir(lora_dir):
            # Include common LoRA model extensions
            if filename.endswith(('.safetensors', '.ckpt', '.pt', '.pth')):
                model_files.append(filename)
        
        logger.info(f"Found {len(model_files)} LoRA models")
        return sorted(model_files)
        
    except Exception as e:
        logger.error(f"Error fetching LoRA models: {str(e)}")
        return []

def get_upscaler_models() -> List[str]:
    """
    Get a list of available upscaler models from the ComfyUI models directory.
    Reads path from settings.json if available, otherwise uses default path.
    
    Returns:
        List[str]: List of upscaler model filenames
    """
    try:
        # Get settings and check for custom upscaler path
        settings = get_settings()
        upscaler_path = settings.get('upscalerModelsPath')
        
        if upscaler_path and is_valid_directory(upscaler_path):
            # Use custom path from settings
            upscaler_dir = os.path.abspath(upscaler_path)
            logger.info(f"Using custom upscaler directory from settings: {upscaler_dir}")
        else:
            # Fallback to default path
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            upscaler_dir = os.path.join(current_dir, 'ComfyUI', 'models', 'upscale_models')
            logger.info(f"Using default upscaler directory: {upscaler_dir}")
        
        if not os.path.exists(upscaler_dir):
            logger.warning(f"Upscaler directory not found: {upscaler_dir}")
            return []
        
        # List all files in the upscaler directory
        model_files = []
        for filename in os.listdir(upscaler_dir):
            # Include common upscaler model extensions
            if filename.endswith(('.pth', '.pt', '.ckpt')):
                model_files.append(filename)
        
        logger.info(f"Found {len(model_files)} upscaler models")
        return sorted(model_files)
        
    except Exception as e:
        logger.error(f"Error fetching upscaler models: {str(e)}")
        return []

def get_all_advanced_models() -> Dict[str, List[str]]:
    """
    Get all available advanced models organized by type.
    
    Returns:
        Dict[str, List[str]]: Dictionary containing lists of models by type
    """
    return {
        'controlnet': get_controlnet_models(),
        'lora': get_lora_models(),
        'upscaler': get_upscaler_models()
    } 
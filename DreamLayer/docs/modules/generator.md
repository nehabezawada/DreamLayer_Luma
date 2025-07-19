# Generator Module

Documentation for the image generation modules in DreamLayer AI.

## Overview

The generator module handles all image generation operations, including text-to-image, image-to-image, and ControlNet generation through ComfyUI integration.

## Core Components

### Text-to-Image Server (`txt2img_server.py`)

The text-to-image server handles prompt-based image generation using Stable Diffusion models.

#### Key Functions

**`generate_image(prompt, negative_prompt, model, settings)`**
```python
def generate_image(prompt: str, negative_prompt: str, model: str, settings: dict):
    """
    Generate an image from text prompt using the specified model
    
    Parameters:
        prompt (str): Positive prompt describing the desired image
        negative_prompt (str): Negative prompt describing what to avoid
        model (str): Model filename to use for generation
        settings (dict): Generation settings (steps, cfg, etc.)
    
    Returns:
        dict: Generation result with image data and metadata
    """
```

**`load_workflow(workflow_type)`**
```python
def load_workflow(workflow_type: str):
    """
    Load a pre-configured workflow template
    
    Parameters:
        workflow_type (str): Type of workflow ('txt2img', 'img2img', etc.)
    
    Returns:
        dict: Workflow configuration
    """
```

### Image-to-Image Server (`img2img_server.py`)

The image-to-image server handles image transformation and editing operations.

#### Key Functions

**`transform_image(input_image, prompt, strength, model)`**
```python
def transform_image(input_image: str, prompt: str, strength: float, model: str):
    """
    Transform an input image using the specified prompt and model
    
    Parameters:
        input_image (str): Path to input image
        prompt (str): Transformation prompt
        strength (float): Denoising strength (0.0-1.0)
        model (str): Model to use for transformation
    
    Returns:
        dict: Transformation result with output image
    """
```

### ControlNet Integration (`controlnet.py`)

ControlNet provides structure-guided image generation using reference images.

#### Key Functions

**`apply_controlnet(control_image, prompt, control_type, strength)`**
```python
def apply_controlnet(control_image: str, prompt: str, control_type: str, strength: float):
    """
    Apply ControlNet guidance to image generation
    
    Parameters:
        control_image (str): Path to control image
        prompt (str): Generation prompt
        control_type (str): Type of control (canny, depth, pose, etc.)
        strength (float): Control strength (0.0-1.0)
    
    Returns:
        dict: Generation result with controlled output
    """
```

## Workflow Management

### Workflow Templates

DreamLayer includes several pre-configured workflow templates:

#### Text-to-Image Workflows

**`workflows/txt2img/core_generation_workflow.json`**
```json
{
  "prompt": {
    "text": "",
    "weight": 1.0
  },
  "negative_prompt": {
    "text": "",
    "weight": 1.0
  },
  "model": {
    "name": "sd15.safetensors",
    "strength": 1.0
  },
  "sampler": {
    "name": "euler",
    "steps": 20,
    "cfg": 7.0
  },
  "output": {
    "width": 512,
    "height": 512,
    "batch_size": 1
  }
}
```

**`workflows/txt2img/bfl_core_generation_workflow.json`**
```json
{
  "api_type": "bfl",
  "model": "flux-pro",
  "prompt": "",
  "negative_prompt": "",
  "output": {
    "width": 1024,
    "height": 1024,
    "aspect_ratio": "1:1"
  }
}
```

#### Image-to-Image Workflows

**`workflows/img2img/core_generation_workflow.json`**
```json
{
  "input_image": {
    "image": "",
    "strength": 0.75
  },
  "prompt": {
    "text": "",
    "weight": 1.0
  },
  "model": {
    "name": "sd15.safetensors"
  },
  "sampler": {
    "name": "euler",
    "steps": 20,
    "cfg": 7.0
  }
}
```

### Custom Workflow Creation

Create custom workflows by extending the base templates:

```python
def create_custom_workflow(base_workflow, customizations):
    """
    Create a custom workflow from base template
    
    Parameters:
        base_workflow (dict): Base workflow template
        customizations (dict): Custom parameters to override
    
    Returns:
        dict: Customized workflow
    """
    workflow = base_workflow.copy()
    
    # Apply customizations
    for key, value in customizations.items():
        if key in workflow:
            if isinstance(workflow[key], dict):
                workflow[key].update(value)
            else:
                workflow[key] = value
        else:
            workflow[key] = value
    
    return workflow
```

## Model Integration

### Local Models

DreamLayer supports various local model types:

#### Checkpoint Models
- **Stable Diffusion 1.5** - Base model for most operations
- **Stable Diffusion 2.1** - Higher quality, more VRAM required
- **Stable Diffusion XL** - Latest model, highest quality
- **Custom Models** - Any compatible .safetensors files

#### LoRA Models
- **Style LoRAs** - Artistic style transfer
- **Character LoRAs** - Specific character generation
- **Concept LoRAs** - Abstract concept representation

#### ControlNet Models
- **Canny** - Edge detection for structure guidance
- **Depth** - Depth map for 3D structure
- **Pose** - Human pose estimation
- **Segmentation** - Object segmentation masks

### Cloud APIs

#### OpenAI DALL-E
```python
def generate_dalle(prompt, model="dall-e-3", size="1024x1024"):
    """
    Generate image using OpenAI DALL-E API
    
    Parameters:
        prompt (str): Generation prompt
        model (str): DALL-E model version
        size (str): Output image size
    
    Returns:
        dict: API response with image URL
    """
```

#### Ideogram
```python
def generate_ideogram(prompt, aspect_ratio="1:1"):
    """
    Generate image using Ideogram API
    
    Parameters:
        prompt (str): Generation prompt
        aspect_ratio (str): Output aspect ratio
    
    Returns:
        dict: API response with image data
    """
```

#### FLUX
```python
def generate_flux(prompt, model="flux-pro"):
    """
    Generate image using FLUX API
    
    Parameters:
        prompt (str): Generation prompt
        model (str): FLUX model variant
    
    Returns:
        dict: API response with image data
    """
```

## Performance Optimization

### Memory Management

```python
def optimize_memory_usage(workflow):
    """
    Optimize workflow for memory efficiency
    
    Parameters:
        workflow (dict): Workflow to optimize
    
    Returns:
        dict: Optimized workflow
    """
    # Use CPU offloading for VAE
    if 'vae' in workflow:
        workflow['vae']['device'] = 'cpu'
    
    # Reduce batch size for large models
    if workflow.get('output', {}).get('batch_size', 1) > 1:
        workflow['output']['batch_size'] = 1
    
    return workflow
```

### Speed Optimization

```python
def optimize_for_speed(workflow):
    """
    Optimize workflow for faster generation
    
    Parameters:
        workflow (dict): Workflow to optimize
    
    Returns:
        dict: Optimized workflow
    """
    # Reduce sampling steps
    if 'sampler' in workflow:
        workflow['sampler']['steps'] = min(workflow['sampler']['steps'], 20)
    
    # Use faster sampler
    workflow['sampler']['name'] = 'euler'
    
    return workflow
```

## Error Handling

### Generation Errors

```python
def handle_generation_error(error, workflow):
    """
    Handle generation errors gracefully
    
    Parameters:
        error (Exception): The error that occurred
        workflow (dict): Workflow that caused the error
    
    Returns:
        dict: Error response with details
    """
    error_response = {
        "status": "error",
        "message": str(error),
        "workflow": workflow,
        "timestamp": time.time()
    }
    
    # Log error for debugging
    logging.error(f"Generation error: {error}")
    
    return error_response
```

### Model Loading Errors

```python
def handle_model_error(model_name, error):
    """
    Handle model loading errors
    
    Parameters:
        model_name (str): Name of the model that failed to load
        error (Exception): The loading error
    
    Returns:
        dict: Error response with fallback options
    """
    # Try to load fallback model
    fallback_model = "sd15.safetensors"
    
    return {
        "status": "error",
        "message": f"Failed to load model {model_name}: {error}",
        "fallback_model": fallback_model,
        "suggestion": f"Try using {fallback_model} instead"
    }
```

## Monitoring and Logging

### Generation Metrics

```python
def log_generation_metrics(workflow, result, generation_time):
    """
    Log generation metrics for monitoring
    
    Parameters:
        workflow (dict): Workflow used for generation
        result (dict): Generation result
        generation_time (float): Time taken for generation
    """
    metrics = {
        "timestamp": time.time(),
        "model": workflow.get("model", {}).get("name"),
        "steps": workflow.get("sampler", {}).get("steps"),
        "generation_time": generation_time,
        "success": result.get("status") == "success",
        "image_size": result.get("image_size"),
        "memory_used": result.get("memory_usage")
    }
    
    logging.info(f"Generation metrics: {metrics}")
```

## Best Practices

### Workflow Design

1. **Start Simple** - Begin with basic workflows and add complexity gradually
2. **Validate Inputs** - Always validate prompt and parameter inputs
3. **Handle Errors** - Implement proper error handling for all operations
4. **Optimize Resources** - Use appropriate model sizes and settings for your hardware

### Performance Tips

1. **Batch Processing** - Generate multiple images when possible
2. **Model Caching** - Keep frequently used models in memory
3. **Memory Management** - Clear GPU memory after large generations
4. **Parallel Processing** - Use multiple workers for concurrent generations

---

*For more details, see the [API Reference](../api_reference.md) and [Usage Guide](../usage.md).* 
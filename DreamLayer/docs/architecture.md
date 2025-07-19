# Architecture

High-level system design and component architecture of DreamLayer AI with detailed flow diagrams and technical specifications based on the actual codebase.

## System Overview

DreamLayer AI follows a three-tier architecture pattern, separating concerns between presentation, business logic, and data generation layers.

```mermaid
graph TB
    subgraph "Presentation Layer"
        A[React Frontend] --> B[TypeScript Components]
        B --> C[Modern UI/UX]
        C --> D[Real-time Updates]
    end
    
    subgraph "API Layer"
        E[Flask API Server] --> F[Request Handler]
        F --> G[Workflow Manager]
        G --> H[Model Manager]
        H --> I[Queue Manager]
    end
    
    subgraph "Generation Layer"
        J[ComfyUI Engine] --> K[Stable Diffusion]
        J --> L[ControlNet]
        J --> M[LoRA Models]
        J --> N[Upscalers]
    end
    
    subgraph "External APIs"
        O[OpenAI DALL-E]
        P[Ideogram API]
        Q[FLUX API]
    end
    
    A --> E
    E --> J
    E --> O
    E --> P
    E --> Q
```

## Component Architecture

### Frontend Layer (React/TypeScript)

The frontend is built with React and TypeScript, providing a modern, responsive interface.

```mermaid
graph LR
    subgraph "Frontend Components"
        A[App.tsx] --> B[Txt2ImgPage]
        A --> C[Img2ImgPage]
        A --> D[ConfigurationsPage]
        A --> E[ExtrasPage]
        
        B --> F[PromptInput]
        B --> G[ModelSelector]
        B --> H[SettingsPanel]
        
        C --> I[ImageUploader]
        C --> J[ControlNetPanel]
        
        D --> K[PathSettings]
        D --> L[ModelManagement]
        
        E --> M[Upscaling]
        E --> N[FaceRestoration]
    end
```

**Key Components:**
- **`App.tsx`** - Main application component with routing
- **`Txt2ImgPage.tsx`** - Text-to-image generation interface
- **`Img2ImgPage.tsx`** - Image-to-image generation interface
- **`ConfigurationsPage.tsx`** - Settings and configuration management
- **`ExtrasPage.tsx`** - Advanced features (upscaling, face restoration)

### API Layer (Flask)

The Flask API server handles all backend operations and coordinates between frontend and generation engines.

```mermaid
graph TB
    subgraph "Flask API Server"
        A[dream_layer.py] --> B[Model Management]
        A --> C[File Operations]
        A --> D[Settings Management]
        A --> E[Prompt Generation]
        
        B --> F[Local Models]
        B --> G[Cloud APIs]
        
        C --> H[Image Upload]
        C --> I[File Serving]
        C --> J[Directory Operations]
        
        D --> K[Path Configuration]
        D --> L[Environment Variables]
        
        E --> M[Random Prompts]
        E --> N[Prompt Templates]
    end
```

**Core API Endpoints:**
- **`/api/models`** - Get available checkpoint models
- **`/api/lora-models`** - Get available LoRA models
- **`/api/upscaler-models`** - Get available upscaler models
- **`/api/controlnet/models`** - Get available ControlNet models
- **`/api/fetch-prompt`** - Get random positive/negative prompts
- **`/api/upload-controlnet-image`** - Upload ControlNet images
- **`/api/settings/paths`** - Configure output and models directories

### Generation Layer (ComfyUI)

ComfyUI serves as the core generation engine, handling all local model inference.

```mermaid
graph LR
    subgraph "ComfyUI Engine"
        A[main.py] --> B[Node System]
        B --> C[Checkpoint Loader]
        B --> D[CLIP Text Encode]
        B --> E[KSampler]
        B --> F[VAE Decode]
        
        C --> G[Stable Diffusion Models]
        D --> H[Text Processing]
        E --> I[Sampling Algorithms]
        F --> J[Image Output]
    end
    
    subgraph "Custom Nodes"
        K[Face Restoration]
        L[ControlNet]
        M[LoRA Loader]
        N[Upscalers]
    end
    
    B --> K
    B --> L
    B --> M
    B --> N
```

**Key Components:**
- **`main.py`** - ComfyUI server entry point
- **Node System** - Modular processing nodes
- **Checkpoint Loader** - Model loading and management
- **KSampler** - Diffusion sampling algorithms
- **VAE Decode** - Image decoding and output

## Data Flow

### Text-to-Image Generation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as Flask API
    participant C as ComfyUI
    participant M as Models
    
    U->>F: Enter prompt & settings
    F->>A: POST /api/generate
    A->>A: Load workflow template
    A->>A: Validate parameters
    A->>C: POST /prompt (workflow)
    C->>M: Load checkpoint model
    C->>C: Process generation
    C->>A: Return image data
    A->>F: Return image URL
    F->>U: Display generated image
```

### Image-to-Image Generation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as Flask API
    participant C as ComfyUI
    
    U->>F: Upload input image
    F->>A: POST /api/upload-controlnet-image
    A->>A: Save image to served_images/
    A->>F: Return image filename
    F->>A: POST /api/generate (img2img)
    A->>A: Load img2img workflow
    A->>C: POST /prompt (workflow + image)
    C->>C: Process img2img generation
    C->>A: Return result image
    A->>F: Return image URL
    F->>U: Display result
```

### Cloud API Integration Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant A as Flask API
    participant O as OpenAI
    participant I as Ideogram
    participant FL as FLUX
    
    U->>F: Select cloud model
    F->>A: POST /api/generate (cloud)
    A->>A: Check API key
    A->>O: POST /v1/images/generations
    A->>I: POST /api/generation
    A->>FL: POST /api/generate
    O->>A: Return image URL
    I->>A: Return image URL
    FL->>A: Return image URL
    A->>F: Return image data
    F->>U: Display result
```

## File Structure

```
DreamLayer/
├── dream_layer_backend/           # Flask API server
│   ├── dream_layer.py            # Main API server
│   ├── txt2img_server.py         # Text-to-image server
│   ├── img2img_server.py         # Image-to-image server
│   ├── controlnet.py             # ControlNet integration
│   └── dream_layer_backend_utils/ # Utility modules
│       ├── api_key_injector.py   # API key management
│       ├── fetch_advanced_models.py # Model fetching
│       └── random_prompt_generator.py # Prompt generation
├── dream_layer_frontend/          # React frontend
│   ├── src/
│   │   ├── App.tsx               # Main app component
│   │   ├── components/           # UI components
│   │   ├── features/             # Feature pages
│   │   ├── services/             # API services
│   │   ├── stores/               # State management
│   │   └── types/                # TypeScript types
│   └── public/                   # Static assets
├── ComfyUI/                      # ComfyUI engine
│   ├── main.py                   # ComfyUI server
│   ├── comfy/                    # Core modules
│   ├── custom_nodes/             # Custom nodes
│   └── models/                   # Model storage
├── workflows/                    # Pre-configured workflows
│   ├── txt2img/                  # Text-to-image workflows
│   └── img2img/                  # Image-to-image workflows
└── Dream_Layer_Resources/        # Output and resources
    ├── output/                   # Generated images
    └── served_images/            # Uploaded images
```

## Technology Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **Shadcn/ui** - Component library

### Backend
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Requests** - HTTP client
- **JSON** - Data serialization

### Generation Engine
- **ComfyUI** - Node-based generation engine
- **PyTorch** - Deep learning framework
- **Transformers** - Model loading and inference
- **Diffusers** - Diffusion model utilities

### External Integrations
- **OpenAI API** - DALL-E 2/3 integration
- **Ideogram API** - Ideogram V3 integration
- **FLUX API** - FLUX Pro/Dev integration

## Security Architecture

### API Key Management
```mermaid
graph LR
    A[Environment Variables] --> B[API Key Injector]
    B --> C[Secure Storage]
    C --> D[Request Signing]
    D --> E[External APIs]
```

### CORS Configuration
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})
```

## Performance Considerations

### Memory Management
- **GPU Memory** - Automatic cleanup after generation
- **Model Loading** - Lazy loading of models
- **Image Caching** - Temporary storage of generated images

### Scalability
- **Multi-threading** - Concurrent request handling
- **Queue Management** - Generation job queuing
- **Resource Pooling** - Efficient resource utilization

### Optimization Strategies
- **Model Quantization** - Reduced memory usage
- **Batch Processing** - Multiple image generation
- **Caching** - Workflow and result caching

## Error Handling

### Error Flow
```mermaid
graph TD
    A[Request] --> B{Validation}
    B -->|Pass| C[Processing]
    B -->|Fail| D[400 Bad Request]
    C --> E{Generation}
    E -->|Success| F[Return Result]
    E -->|Fail| G[500 Internal Error]
    G --> H[Log Error]
    H --> I[Return Error Response]
```

### Error Types
- **Validation Errors** - Invalid input parameters
- **Model Errors** - Model loading/inference failures
- **API Errors** - External API failures
- **System Errors** - Resource/configuration issues

## Monitoring and Logging

### Log Structure
```
logs/
├── dream_layer.log              # Main application logs
├── comfyui.log                  # ComfyUI engine logs
└── access.log                   # Request access logs
```

### Metrics
- **Generation Time** - Time per image generation
- **Success Rate** - Percentage of successful generations
- **Resource Usage** - CPU, GPU, memory utilization
- **API Response Time** - Endpoint response times

---

*For implementation details, see the [API Reference](api_reference.md) and [Usage Guide](usage.md).* 
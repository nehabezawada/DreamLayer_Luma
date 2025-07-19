# Quick Start Guide

Get DreamLayer AI up and running in **60 seconds** with this lightning-fast setup guide.

## ⚡ Prerequisites Check

Before you start, ensure you have:

- **Python 3.8+** installed
- **Node.js 16+** installed  
- **8GB+ RAM** (16GB+ recommended)
- **NVIDIA GPU** with 6GB+ VRAM (or Apple Silicon)

> **Note:** DreamLayer works on CPU-only setups, but GPU acceleration is highly recommended for optimal performance.

## 🚀 60-Second Setup

### Step 1: Clone & Navigate
```bash
git clone https://github.com/DreamLayer-AI/DreamLayer.git
cd DreamLayer
```

### Step 2: Install Dependencies

**macOS/Linux:**
```bash
chmod +x install_mac_dependencies.sh
./install_mac_dependencies.sh
```

**Windows:**
```bash
install_windows_dependencies.bat
```

### Step 3: Start DreamLayer

**macOS/Linux:**
```bash
chmod +x start_dream_layer.sh
./start_dream_layer.sh
```

**Windows:**
```bash
start_dream_layer.bat
```

### Step 4: Open Your Browser

Navigate to **http://localhost:8080** and start creating!

## 🎯 First Image Generation

1. **Enter a prompt** in the text area
   ```
   A beautiful sunset over mountains, digital art, masterpiece
   ```

2. **Select a model** from the dropdown (SD 1.5 is recommended for beginners)

3. **Click "Generate"** and wait for your image

4. **Download or share** your creation

## 🔧 Quick Configuration

### Set Output Directory
1. Go to **Settings** → **Paths**
2. Set your preferred output directory
3. Click **Save**

### Add Models
1. Download Stable Diffusion models (.safetensors files)
2. Place them in your models directory
3. Restart DreamLayer to detect new models

## 🚨 Troubleshooting

### Common Issues

**"ComfyUI not found"**
- Ensure ComfyUI is properly installed in the `ComfyUI/` directory
- Check that all dependencies are installed

**"No models available"**
- Verify your models directory path in Settings
- Ensure models are in .safetensors format
- Check file permissions

**"GPU not detected"**
- Install NVIDIA drivers (if using NVIDIA GPU)
- For Apple Silicon, ensure PyTorch MPS is available

### Getting Help

- **Documentation:** [Installation Guide](installation.md)
- **Issues:** [GitHub Issues](https://github.com/DreamLayer-AI/DreamLayer/issues)
- **Discussions:** [GitHub Discussions](https://github.com/DreamLayer-AI/DreamLayer/discussions)

## 🎨 Next Steps

Now that you're up and running:

1. **Explore Advanced Features** - [Usage Guide](usage.md)
2. **Learn the Architecture** - [Architecture Overview](architecture.md)
3. **Check the API** - [API Reference](api_reference.md)
4. **Contribute** - [Contributing Guide](contributing.md)

---

*Need more help? Check out the [Installation Guide](installation.md) for detailed setup instructions.* 
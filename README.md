# ComfyUI Vector Engine Image Generator Node

This is a custom node for ComfyUI that integrates Vector Engine's Gemini API for advanced image generation and editing.

## Features

- 🖼️ Support up to 5 optional input images
- 🎨 Configurable aspect ratios (1:1, 2:3, 3:2, 4:3, 3:4, 16:9, 9:16)
- 📐 Multiple resolution options (1K, 2K, 4K)
- 🎲 Seed control for reproducible results
- 📝 Custom system and user prompts
- ℹ️ Detailed generation information output

## Installation

1. Navigate to your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone or copy this repository:
```bash
git clone <repository-url> ComfyUI-Vector-Engine
# or copy the folder manually
```

3. Restart ComfyUI

## Node Inputs

### Required Parameters

- **prompt** (STRING, multiline): User prompt describing what to generate
  - Default: "Generate a creative image based on the provided pictures."

- **system_prompt** (STRING, multiline): System instructions for the AI
  - Default: "You are an AI assistant skilled in generating images and editing pictures."

- **aspect_ratio** (DROPDOWN): Image aspect ratio
  - Options: 1:1, 2:3, 3:2, 4:3, 3:4, 16:9, 9:16
  - Default: 1:1

- **image_size** (DROPDOWN): Output resolution
  - Options: 1K, 2K, 4K
  - Default: 1K

- **seed** (INT): Random seed for reproducibility
  - Range: 0 to max int
  - Default: 0

### Optional Parameters

- **image_1** to **image_5** (IMAGE): Up to 5 optional input images
  - All images are optional, you can use 0 to 5 images

## Node Outputs

1. **image** (IMAGE): The generated image tensor
   - Can be connected to Preview Image, Save Image, or other image processing nodes

2. **info** (STRING): Generation information including:
   - Model name
   - Aspect ratio
   - Image size
   - Actual resolution
   - Seed value
   - Number of input images
   - Generation time
   - Success/failure status
   - Error message (if failed)

## Usage Examples

### Example 1: Generate from Text Only
```
1. Add "Vector Engine Image Generator" node
2. Set prompt: "A beautiful sunset over mountains"
3. Leave all image inputs empty
4. Connect output to "Preview Image"
5. Run the workflow
```

### Example 2: Edit/Combine Multiple Images
```
1. Load 2-5 images using "Load Image" nodes
2. Add "Vector Engine Image Generator" node
3. Connect images to image_1, image_2, etc.
4. Set prompt: "Combine these people in a group photo at the beach"
5. Connect output to "Save Image"
6. Run the workflow
```

### Example 3: Generate with Specific Aspect Ratio
```
1. Add "Vector Engine Image Generator" node
2. Set aspect_ratio: "16:9"
3. Set image_size: "2K"
4. Set prompt: "A cinematic landscape photo"
5. Run the workflow
```

## Configuration

The API key is currently hardcoded in the node. To change it:

1. Open `vector_engine_node.py`
2. Find the line: `self.api_key = "sk-..."`
3. Replace with your API key

For production use, consider using environment variables:
```python
import os
self.api_key = os.getenv("VECTOR_ENGINE_API_KEY", "default-key")
```

## Model Information

- **Current Model**: gemini-3-pro-image-preview
- **API Endpoint**: api.vectorengine.ai
- **Features**: Text-to-image and image-to-image generation

## Troubleshooting

### No image generated (black output)
- Check the info output for error messages
- Verify your API key is valid
- Check network connectivity
- Ensure input images are in correct format

### Image quality issues
- Try increasing image_size (1K → 2K → 4K)
- Adjust aspect_ratio to match desired output
- Refine your prompt for better results

### Slow generation
- Higher resolutions (4K) take longer
- Multiple input images increase processing time
- Check your network connection

## Dependencies

- torch
- numpy
- PIL (Pillow)
- Standard Python libraries (http.client, json, base64, time, io)

These should already be available in a standard ComfyUI installation.

## License

[Add your license information here]

## Support

For issues and questions, please open an issue on the repository.


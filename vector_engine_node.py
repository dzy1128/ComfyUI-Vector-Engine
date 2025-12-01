import http.client
import json
import base64
import time
import numpy as np
import torch
from PIL import Image
import io


class VectorEngineImageGenerator:
    """
    ComfyUI node for Vector Engine Image Generation API
    """
    
    def __init__(self):
        self.api_key = "sk-RZBLe4v8MolmD3fwOd6vdRTCaj7PohDhf4f44UNgWHEAA4zF"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate a creative image based on the provided pictures."
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are an AI assistant skilled in generating images and editing pictures."
                }),
                "aspect_ratio": (["1:1", "2:3", "3:2", "4:3", "3:4", "16:9", "9:16"], {
                    "default": "1:1"
                }),
                "image_size": (["1K", "2K", "4K"], {
                    "default": "1K"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate_image"
    CATEGORY = "VectorEngine"
    
    def tensor_to_base64(self, tensor, mime_type="image/png"):
        """
        Convert ComfyUI image tensor to base64 string
        tensor shape: [batch, height, width, channels] with values in [0, 1]
        """
        # Take first image if batch
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Convert to numpy and scale to 0-255
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(numpy_image)
        
        # Convert to bytes
        buffer = io.BytesIO()
        if "png" in mime_type:
            pil_image.save(buffer, format="PNG")
        else:
            pil_image.save(buffer, format="JPEG")
        
        # Encode to base64
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_str, mime_type
    
    def base64_to_tensor(self, base64_str):
        """
        Convert base64 string to ComfyUI image tensor
        """
        # Decode base64
        img_bytes = base64.b64decode(base64_str)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor [1, height, width, channels]
        tensor = torch.from_numpy(numpy_image)[None,]
        
        return tensor
    
    def generate_image(self, prompt, system_prompt, aspect_ratio, image_size, seed,
                      image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        Main function to generate image using Vector Engine API
        """
        start_time = time.time()
        
        try:
            # Prepare connection
            conn = http.client.HTTPSConnection("api.vectorengine.ai")
            
            # Build parts array
            parts = []
            
            # Add system prompt and user prompt as text
            full_prompt = f"{system_prompt}\n\nUser request: {prompt}"
            parts.append({
                "text": full_prompt
            })
            
            # Add images if provided
            images = [image_1, image_2, image_3, image_4, image_5]
            image_count = 0
            for idx, img in enumerate(images, 1):
                if img is not None:
                    image_count += 1
                    base64_data, mime_type = self.tensor_to_base64(img, "image/png")
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            
            # Build request payload
            payload = json.dumps({
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                        "imageSize": image_size
                    },
                    "seed": seed
                }
            })
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Make API request
            conn.request("POST", 
                        f"/v1beta/models/gemini-3-pro-image-preview:generateContent?key={self.api_key}",
                        payload, headers)
            
            res = conn.getresponse()
            data = res.read()
            
            # Parse response
            response_json = json.loads(data.decode("utf-8"))
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Extract image from response
            candidates = response_json.get("candidates", [])
            
            if not candidates:
                # No candidates, check for error
                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                info_text = self._format_info(
                    model_name="gemini-3-pro-image-preview",
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    generation_time=generation_time,
                    success=False,
                    error_message=error_msg,
                    input_images=image_count,
                    seed=seed
                )
                
                # Return a black image as placeholder
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            # Try to extract image from first candidate
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    img_base64 = None
                    mime_type = None
                    
                    # Check different possible field names
                    if "inline_data" in part:
                        img_base64 = part["inline_data"]["data"]
                        mime_type = part["inline_data"].get("mime_type", "image/jpeg")
                    elif "inlineData" in part:
                        img_base64 = part["inlineData"]["data"]
                        mime_type = part["inlineData"].get("mimeType", "image/jpeg")
                    elif "data" in part and not "text" in part:
                        img_base64 = part["data"]
                        mime_type = part.get("mimeType", part.get("mime_type", "image/jpeg"))
                    
                    if img_base64:
                        # Convert to tensor
                        output_tensor = self.base64_to_tensor(img_base64)
                        
                        # Get actual dimensions
                        _, height, width, _ = output_tensor.shape
                        
                        # Format info text
                        info_text = self._format_info(
                            model_name="gemini-3-pro-image-preview",
                            aspect_ratio=aspect_ratio,
                            image_size=image_size,
                            resolution=f"{width}x{height}",
                            generation_time=generation_time,
                            success=True,
                            input_images=image_count,
                            seed=seed
                        )
                        
                        return (output_tensor, info_text)
            
            # If we reach here, no image was found
            info_text = self._format_info(
                model_name="gemini-3-pro-image-preview",
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                generation_time=generation_time,
                success=False,
                error_message="No image found in API response",
                input_images=image_count,
                seed=seed
            )
            
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
            
        except Exception as e:
            generation_time = time.time() - start_time
            
            info_text = self._format_info(
                model_name="gemini-3-pro-image-preview",
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                generation_time=generation_time,
                success=False,
                error_message=str(e),
                input_images=sum(1 for img in images if img is not None),
                seed=seed
            )
            
            # Return black image as placeholder
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
    
    def _format_info(self, model_name, aspect_ratio, image_size, generation_time,
                     success, resolution=None, error_message=None, input_images=0, seed=0):
        """
        Format generation information as text
        """
        lines = [
            "=" * 60,
            "Vector Engine Image Generation Result",
            "=" * 60,
            f"Model Name: {model_name}",
            f"Aspect Ratio: {aspect_ratio}",
            f"Image Size: {image_size}",
        ]
        
        if resolution:
            lines.append(f"Resolution: {resolution}")
        
        lines.extend([
            f"Seed: {seed}",
            f"Input Images: {input_images}",
            f"Generation Time: {generation_time:.2f}s",
            f"Status: {'SUCCESS' if success else 'FAILED'}",
        ])
        
        if not success and error_message:
            lines.extend([
                "",
                "Error Message:",
                error_message
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "VectorEngineImageGenerator": VectorEngineImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VectorEngineImageGenerator": "Vector Engine Image Generator"
}


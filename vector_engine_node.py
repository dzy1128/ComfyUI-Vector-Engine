import http.client
import json
import base64
import time
import os
import numpy as np
import torch
from PIL import Image
import io
import urllib.request


class VectorEngineImageGenerator:
    """
    ComfyUI node for Vector Engine Image Generation API
    """
    
    # Model configuration: defines API endpoints and parameter formats for each model
    MODEL_CONFIG = {
        "gemini-3-pro-image-preview": {
            "api_type": "gemini",
            "endpoint": "/v1beta/models/{model}:generateContent",
            "supports_multi_image": True,
            "supports_system_prompt": True,
        },
        "gpt-image-1.5": {
            "api_type": "openai",
            "endpoint": "/v1/images/generations",
            "supports_multi_image": False,
            "supports_system_prompt": False,
        },
    }
    
    # Size mapping for OpenAI-style models (aspect_ratio + image_size -> pixel size)
    SIZE_MAPPING = {
        # 1K sizes
        ("1:1", "1K"): "1024x1024",
        ("2:3", "1K"): "1024x1536",
        ("3:2", "1K"): "1536x1024",
        ("4:3", "1K"): "1024x768",
        ("3:4", "1K"): "768x1024",
        ("16:9", "1K"): "1024x576",
        ("9:16", "1K"): "576x1024",
        # 2K sizes
        ("1:1", "2K"): "2048x2048",
        ("2:3", "2K"): "1536x2304",
        ("3:2", "2K"): "2304x1536",
        ("4:3", "2K"): "2048x1536",
        ("3:4", "2K"): "1536x2048",
        ("16:9", "2K"): "2048x1152",
        ("9:16", "2K"): "1152x2048",
        # 4K sizes
        ("1:1", "4K"): "4096x4096",
        ("2:3", "4K"): "2730x4096",
        ("3:2", "4K"): "4096x2730",
        ("4:3", "4K"): "4096x3072",
        ("3:4", "4K"): "3072x4096",
        ("16:9", "4K"): "4096x2304",
        ("9:16", "4K"): "2304x4096",
    }
    
    def __init__(self):
        # Read API key from environment variable
        self.api_key = os.getenv("VECTOR_ENGINE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VECTOR_ENGINE_API_KEY environment variable is not set. "
                "Please set it before using this node: "
                "export VECTOR_ENGINE_API_KEY='your-api-key-here'"
            )
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["gemini-3-pro-image-preview", "gpt-image-1.5"], {
                    "default": "gemini-3-pro-image-preview"
                }),
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
    
    def tensor_to_base64(self, tensor, mime_type="image/jpeg", max_size=2048, quality=85):
        """
        Convert ComfyUI image tensor to base64 string with optimization
        tensor shape: [batch, height, width, channels] with values in [0, 1]
        
        Args:
            tensor: Image tensor
            mime_type: Output mime type (default JPEG for better compression)
            max_size: Maximum dimension (width or height) in pixels
            quality: JPEG quality (1-100, default 85 for good balance)
        """
        # Take first image if batch
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        # Convert to numpy and scale to 0-255
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(numpy_image)
        
        # Convert to RGB first (more efficient to do before resize)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize if image is too large
        width, height = pil_image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Use BILINEAR for faster resizing (3-4x faster than LANCZOS)
            # Quality difference is minimal for photo content
            pil_image = pil_image.resize((new_width, new_height), Image.BILINEAR)
        
        # Save with compression
        buffer = io.BytesIO()
        # Remove optimize=True for faster encoding (file size increase is minimal ~2-5%)
        # Use progressive for better streaming (no speed impact)
        pil_image.save(buffer, format="JPEG", quality=quality, progressive=True)
        
        # Encode to base64
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_str, "image/jpeg"
    
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
    
    def generate_image(self, model, prompt, system_prompt, aspect_ratio, image_size, seed,
                      image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        Main function to generate image using Vector Engine API
        Routes to appropriate API based on model type
        """
        # Get model configuration
        model_config = self.MODEL_CONFIG.get(model, self.MODEL_CONFIG["gemini-3-pro-image-preview"])
        api_type = model_config["api_type"]
        
        print(f"[VectorEngine] Using {api_type} API for model: {model}")
        
        if api_type == "openai":
            return self._generate_image_openai(model, prompt, aspect_ratio, image_size, seed)
        else:
            return self._generate_image_gemini(model, prompt, system_prompt, aspect_ratio, image_size, seed,
                                               image_1, image_2, image_3, image_4, image_5)
    
    def _generate_image_openai(self, model, prompt, aspect_ratio, image_size, seed):
        """
        Generate image using OpenAI-style API (for gpt-image-1.5 and similar models)
        """
        try:
            # Prepare connection
            conn = http.client.HTTPSConnection("api.vectorengine.ai")
            
            # Get pixel size from mapping
            size = self.SIZE_MAPPING.get((aspect_ratio, image_size), "1024x1024")
            
            print(f"[VectorEngine] OpenAI API - Size: {size}")
            
            # Build request payload
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "size": size,
                "n": 1,
                "response_format": "url"
            })
            
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Calculate payload size
            payload_size_kb = len(payload) / 1024
            print(f"[VectorEngine] Sending request to API (payload size: {payload_size_kb:.2f}KB)...")
            
            # Record start time
            start_time = time.time()
            
            # Make API request
            conn.request("POST", "/v1/images/generations", payload, headers)
            
            res = conn.getresponse()
            data = res.read()
            
            # Record end time
            api_generation_time = time.time() - start_time
            
            print(f"[VectorEngine] API request completed in {api_generation_time:.2f}s")
            
            # Parse response
            response_json = json.loads(data.decode("utf-8"))
            
            # Check for error
            if "error" in response_json:
                error_msg = response_json["error"].get("message", str(response_json["error"]))
                info_text = self._format_info(
                    model_name=model,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    generation_time=api_generation_time,
                    success=False,
                    error_message=error_msg,
                    input_images=0,
                    seed=seed,
                    encode_time=0.0,
                    decode_time=0.0
                )
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            # Extract image from response
            data_list = response_json.get("data", [])
            
            if not data_list:
                info_text = self._format_info(
                    model_name=model,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    generation_time=api_generation_time,
                    success=False,
                    error_message="No image data in response",
                    input_images=0,
                    seed=seed,
                    encode_time=0.0,
                    decode_time=0.0
                )
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            # Get first image
            image_data = data_list[0]
            
            decode_start = time.time()
            
            # Handle URL response
            if "url" in image_data:
                image_url = image_data["url"]
                print(f"[VectorEngine] Downloading image from URL...")
                
                # Download image from URL
                req = urllib.request.Request(image_url)
                with urllib.request.urlopen(req) as response:
                    img_bytes = response.read()
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(img_bytes))
                
            # Handle base64 response
            elif "b64_json" in image_data:
                img_base64 = image_data["b64_json"]
                img_bytes = base64.b64decode(img_base64)
                pil_image = Image.open(io.BytesIO(img_bytes))
            else:
                info_text = self._format_info(
                    model_name=model,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    generation_time=api_generation_time,
                    success=False,
                    error_message="Unknown response format",
                    input_images=0,
                    seed=seed,
                    encode_time=0.0,
                    decode_time=0.0
                )
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to tensor
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(numpy_image)[None,]
            
            decode_time = time.time() - decode_start
            print(f"[VectorEngine] Image downloaded and decoded in {decode_time:.2f}s")
            
            # Get actual dimensions
            _, height, width, _ = output_tensor.shape
            
            # Format info text
            info_text = self._format_info(
                model_name=model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                resolution=f"{width}x{height}",
                generation_time=api_generation_time,
                success=True,
                input_images=0,
                seed=seed,
                encode_time=0.0,
                decode_time=decode_time
            )
            
            return (output_tensor, info_text)
            
        except Exception as e:
            try:
                error_generation_time = time.time() - start_time
            except:
                error_generation_time = 0.0
            
            info_text = self._format_info(
                model_name=model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                generation_time=error_generation_time,
                success=False,
                error_message=str(e),
                input_images=0,
                seed=seed,
                encode_time=0.0,
                decode_time=0.0
            )
            
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
    
    def _generate_image_gemini(self, model, prompt, system_prompt, aspect_ratio, image_size, seed,
                               image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        Generate image using Gemini-style API (for gemini models)
        """
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
            total_encode_time = 0.0
            
            print(f"[VectorEngine] Processing {sum(1 for img in images if img is not None)} input images...")
            
            for idx, img in enumerate(images, 1):
                if img is not None:
                    image_count += 1
                    encode_start = time.time()
                    # Use optimized compression: max 2048px, 85% quality JPEG
                    base64_data, mime_type = self.tensor_to_base64(img, max_size=2048, quality=85)
                    encode_time = time.time() - encode_start
                    total_encode_time += encode_time
                    
                    # Calculate compressed size
                    data_size_kb = len(base64_data) / 1024
                    print(f"[VectorEngine] Image {idx}: encoded in {encode_time:.2f}s, size: {data_size_kb:.1f}KB")
                    
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
                    }
                }
            })
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # Calculate payload size
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"[VectorEngine] Sending request to API (payload size: {payload_size_mb:.2f}MB)...")
            
            # Record start time right before API request
            start_time = time.time()
            
            # Make API request
            conn.request("POST", 
                        f"/v1beta/models/{model}:generateContent?key={self.api_key}",
                        payload, headers)
            
            res = conn.getresponse()
            data = res.read()
            
            # Record end time right after receiving response
            api_generation_time = time.time() - start_time
            
            print(f"[VectorEngine] API request completed in {api_generation_time:.2f}s")
            
            # Parse response
            response_json = json.loads(data.decode("utf-8"))
            
            # Extract image from response
            candidates = response_json.get("candidates", [])
            
            if not candidates:
                # No candidates, check for error
                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                info_text = self._format_info(
                    model_name=model,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    generation_time=api_generation_time,
                    success=False,
                    error_message=error_msg,
                    input_images=image_count,
                    seed=seed,
                    encode_time=total_encode_time,
                    decode_time=0.0
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
                        # Convert to tensor and record decode time
                        decode_start = time.time()
                        output_tensor = self.base64_to_tensor(img_base64)
                        decode_time = time.time() - decode_start
                        
                        print(f"[VectorEngine] Image decoded in {decode_time:.2f}s")
                        
                        # Get actual dimensions
                        _, height, width, _ = output_tensor.shape
                        
                        # Format info text
                        info_text = self._format_info(
                            model_name=model,
                            aspect_ratio=aspect_ratio,
                            image_size=image_size,
                            resolution=f"{width}x{height}",
                            generation_time=api_generation_time,
                            success=True,
                            input_images=image_count,
                            seed=seed,
                            encode_time=total_encode_time,
                            decode_time=decode_time
                        )
                        
                        return (output_tensor, info_text)
            
            # If we reach here, no image was found
            info_text = self._format_info(
                model_name=model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                generation_time=api_generation_time,
                success=False,
                error_message="No image found in API response",
                input_images=image_count,
                seed=seed,
                encode_time=total_encode_time,
                decode_time=0.0
            )
            
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
            
        except Exception as e:
            # If error occurred before API request, set generation time to 0
            try:
                error_generation_time = time.time() - start_time
            except:
                error_generation_time = 0.0
            
            # Get encode time if available
            try:
                error_encode_time = total_encode_time
            except:
                error_encode_time = 0.0
            
            info_text = self._format_info(
                model_name=model,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                generation_time=error_generation_time,
                success=False,
                error_message=str(e),
                input_images=sum(1 for img in [image_1, image_2, image_3, image_4, image_5] if img is not None),
                seed=seed,
                encode_time=error_encode_time,
                decode_time=0.0
            )
            
            # Return black image as placeholder
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
    
    def _format_info(self, model_name, aspect_ratio, image_size, generation_time,
                     success, resolution=None, error_message=None, input_images=0, seed=0,
                     encode_time=0.0, decode_time=0.0):
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
            "",
            "--- Timing Information ---",
            f"Image Encode Time: {encode_time:.3f}s",
            f"API Generation Time: {generation_time:.2f}s",
            f"Image Decode Time: {decode_time:.3f}s",
            f"Total Time: {(encode_time + generation_time + decode_time):.2f}s",
            "",
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


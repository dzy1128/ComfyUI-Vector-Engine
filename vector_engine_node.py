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


class VectorEngineBase:
    """
    Base class with shared utilities for Vector Engine nodes
    """
    
    def __init__(self):
        # Read API key from environment variable
        self.api_key = os.getenv("VECTOR_ENGINE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VECTOR_ENGINE_API_KEY environment variable is not set. "
                "Please set it before using this node: "
                "export VECTOR_ENGINE_API_KEY='your-api-key-here'"
            )
    
    def tensor_to_base64(self, tensor, mime_type="image/jpeg", max_size=2048, quality=85):
        """
        Convert ComfyUI image tensor to base64 string with optimization
        """
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(numpy_image)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        width, height = pil_image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            pil_image = pil_image.resize((new_width, new_height), Image.BILINEAR)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, progressive=True)
        
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_str, "image/jpeg"
    
    def base64_to_tensor(self, base64_str):
        """
        Convert base64 string to ComfyUI image tensor
        """
        img_bytes = base64.b64decode(base64_str)
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        numpy_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(numpy_image)[None,]
        
        return tensor
    
    def _tensor_to_png_bytes(self, tensor, max_size=2048):
        """
        Convert ComfyUI image tensor to PNG bytes for multipart upload
        """
        if len(tensor.shape) == 4:
            tensor = tensor[0]
        
        numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(numpy_image)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        width, height = pil_image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            pil_image = pil_image.resize((new_width, new_height), Image.BILINEAR)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        
        return buffer.getvalue()


class VectorEngineGemini(VectorEngineBase):
    """
    ComfyUI node for Gemini Image Generation via Vector Engine API
    Supports multi-image input and system prompts
    """
    
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
    
    def generate_image(self, prompt, system_prompt, aspect_ratio, image_size, seed,
                      image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        Generate image using Gemini API via Vector Engine
        """
        model = "gemini-3-pro-image-preview"
        
        try:
            conn = http.client.HTTPSConnection("api.vectorengine.ai")
            
            # Build parts array
            parts = []
            
            # Add system prompt and user prompt as text
            full_prompt = f"{system_prompt}\n\nUser request: {prompt}"
            parts.append({"text": full_prompt})
            
            # Add images if provided
            images = [image_1, image_2, image_3, image_4, image_5]
            image_count = 0
            total_encode_time = 0.0
            
            print(f"[VectorEngine Gemini] Processing {sum(1 for img in images if img is not None)} input images...")
            
            for idx, img in enumerate(images, 1):
                if img is not None:
                    image_count += 1
                    encode_start = time.time()
                    base64_data, mime_type = self.tensor_to_base64(img, max_size=2048, quality=85)
                    encode_time = time.time() - encode_start
                    total_encode_time += encode_time
                    
                    data_size_kb = len(base64_data) / 1024
                    print(f"[VectorEngine Gemini] Image {idx}: encoded in {encode_time:.2f}s, size: {data_size_kb:.1f}KB")
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            
            # Build request payload
            payload = json.dumps({
                "contents": [{"role": "user", "parts": parts}],
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
            
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"[VectorEngine Gemini] Sending request (payload: {payload_size_mb:.2f}MB)...")
            
            start_time = time.time()
            
            conn.request("POST", 
                        f"/v1beta/models/{model}:generateContent?key={self.api_key}",
                        payload, headers)
            
            res = conn.getresponse()
            data = res.read()
            api_generation_time = time.time() - start_time
            
            print(f"[VectorEngine Gemini] API request completed in {api_generation_time:.2f}s")
            
            response_json = json.loads(data.decode("utf-8"))
            candidates = response_json.get("candidates", [])
            
            if not candidates:
                error_msg = response_json.get("error", {}).get("message", "Unknown error")
                info_text = self._format_info(model, aspect_ratio, image_size, api_generation_time,
                                             False, None, error_msg, image_count, seed,
                                             total_encode_time, 0.0)
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            # Extract image from response
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    img_base64 = None
                    
                    if "inline_data" in part:
                        img_base64 = part["inline_data"]["data"]
                    elif "inlineData" in part:
                        img_base64 = part["inlineData"]["data"]
                    elif "data" in part and "text" not in part:
                        img_base64 = part["data"]
                    
                    if img_base64:
                        decode_start = time.time()
                        output_tensor = self.base64_to_tensor(img_base64)
                        decode_time = time.time() - decode_start
                        
                        print(f"[VectorEngine Gemini] Image decoded in {decode_time:.2f}s")
                        
                        _, height, width, _ = output_tensor.shape
                        
                        info_text = self._format_info(model, aspect_ratio, image_size, api_generation_time,
                                                     True, f"{width}x{height}", None, image_count, seed,
                                                     total_encode_time, decode_time)
                        
                        return (output_tensor, info_text)
            
            # No image found
            info_text = self._format_info(model, aspect_ratio, image_size, api_generation_time,
                                         False, None, "No image found in API response", image_count, seed,
                                         total_encode_time, 0.0)
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
            
        except Exception as e:
            try:
                error_time = time.time() - start_time
            except:
                error_time = 0.0
            try:
                error_encode_time = total_encode_time
            except:
                error_encode_time = 0.0
            
            info_text = self._format_info(model, aspect_ratio, image_size, error_time,
                                         False, None, str(e),
                                         sum(1 for img in [image_1, image_2, image_3, image_4, image_5] if img is not None),
                                         seed, error_encode_time, 0.0)
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
    
    def _format_info(self, model_name, aspect_ratio, image_size, generation_time,
                     success, resolution, error_message, input_images, seed,
                     encode_time, decode_time):
        lines = [
            "=" * 60,
            "Vector Engine Gemini - Generation Result",
            "=" * 60,
            f"Model: {model_name}",
            f"Aspect Ratio: {aspect_ratio}",
            f"Image Size: {image_size}",
        ]
        
        if resolution:
            lines.append(f"Resolution: {resolution}")
        
        lines.extend([
            f"Seed: {seed}",
            f"Input Images: {input_images}",
            "",
            "--- Timing ---",
            f"Encode: {encode_time:.3f}s | API: {generation_time:.2f}s | Decode: {decode_time:.3f}s",
            f"Total: {(encode_time + generation_time + decode_time):.2f}s",
            "",
            f"Status: {'SUCCESS' if success else 'FAILED'}",
        ])
        
        if not success and error_message:
            lines.extend(["", "Error:", error_message])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class VectorEngineGPT(VectorEngineBase):
    """
    ComfyUI node for GPT Image Generation via Vector Engine API
    Supports image generation and editing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful landscape with mountains and a lake."
                }),
                "size": (["1024x1024", "1536x1024", "1024x1536", "auto"], {
                    "default": "1024x1024"
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
    
    def generate_image(self, prompt, size, seed,
                      image_1=None, image_2=None, image_3=None, image_4=None, image_5=None):
        """
        Generate or edit image using GPT API via Vector Engine
        """
        model = "gpt-image-1.5"
        
        # Collect input images
        images = [image_1, image_2, image_3, image_4, image_5]
        input_images = [img for img in images if img is not None]
        
        try:
            conn = http.client.HTTPSConnection("api.vectorengine.ai")
            
            image_count = len(input_images)
            total_encode_time = 0.0
            
            # Determine endpoint based on whether images are provided
            if image_count > 0:
                # Use /v1/images/edits with multipart/form-data
                endpoint = "/v1/images/edits"
                print(f"[VectorEngine GPT] Edit mode - Size: {size}, Images: {image_count}")
                
                boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
                dataList = []
                
                # Add images
                for idx, img in enumerate(input_images):
                    encode_start = time.time()
                    img_bytes = self._tensor_to_png_bytes(img)
                    encode_time = time.time() - encode_start
                    total_encode_time += encode_time
                    
                    data_size_kb = len(img_bytes) / 1024
                    print(f"[VectorEngine GPT] Image {idx + 1}: encoded in {encode_time:.2f}s, size: {data_size_kb:.1f}KB")
                    
                    dataList.append(f'--{boundary}'.encode())
                    dataList.append(f'Content-Disposition: form-data; name=image; filename=image_{idx + 1}.png'.encode())
                    dataList.append(b'Content-Type: image/png')
                    dataList.append(b'')
                    dataList.append(img_bytes)
                
                # Add form fields
                for name, value in [("prompt", prompt), ("model", model), ("n", "1"), ("size", size)]:
                    dataList.append(f'--{boundary}'.encode())
                    dataList.append(f'Content-Disposition: form-data; name={name};'.encode())
                    dataList.append(b'Content-Type: text/plain')
                    dataList.append(b'')
                    dataList.append(value.encode() if isinstance(value, str) else value)
                
                dataList.append(f'--{boundary}--'.encode())
                dataList.append(b'')
                
                payload = b'\r\n'.join(dataList)
                
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': f'multipart/form-data; boundary={boundary}'
                }
            else:
                # Use /v1/images/generations with JSON
                endpoint = "/v1/images/generations"
                print(f"[VectorEngine GPT] Generation mode - Size: {size}")
                
                payload = json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                    "n": 1
                })
                
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            
            payload_size_kb = len(payload) / 1024
            print(f"[VectorEngine GPT] Sending request (payload: {payload_size_kb:.2f}KB)...")
            
            start_time = time.time()
            conn.request("POST", endpoint, payload, headers)
            
            res = conn.getresponse()
            data = res.read()
            api_generation_time = time.time() - start_time
            
            print(f"[VectorEngine GPT] API request completed in {api_generation_time:.2f}s")
            
            response_json = json.loads(data.decode("utf-8"))
            
            # Check for error
            if "error" in response_json:
                error_msg = response_json["error"].get("message", str(response_json["error"]))
                info_text = self._format_info(model, size, api_generation_time,
                                             False, None, error_msg, image_count, seed,
                                             total_encode_time, 0.0)
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            # Extract image from response
            data_list = response_json.get("data", [])
            
            if not data_list:
                info_text = self._format_info(model, size, api_generation_time,
                                             False, None, "No image data in response", image_count, seed,
                                             total_encode_time, 0.0)
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            image_data = data_list[0]
            decode_start = time.time()
            
            # Handle URL response
            if "url" in image_data:
                image_url = image_data["url"]
                print(f"[VectorEngine GPT] Downloading image from URL...")
                
                req = urllib.request.Request(image_url)
                with urllib.request.urlopen(req) as response:
                    img_bytes = response.read()
                
                pil_image = Image.open(io.BytesIO(img_bytes))
                
            # Handle base64 response
            elif "b64_json" in image_data:
                img_base64 = image_data["b64_json"]
                img_bytes = base64.b64decode(img_base64)
                pil_image = Image.open(io.BytesIO(img_bytes))
            else:
                info_text = self._format_info(model, size, api_generation_time,
                                             False, None, "Unknown response format", image_count, seed,
                                             total_encode_time, 0.0)
                black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                return (black_image, info_text)
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            numpy_image = np.array(pil_image).astype(np.float32) / 255.0
            output_tensor = torch.from_numpy(numpy_image)[None,]
            
            decode_time = time.time() - decode_start
            print(f"[VectorEngine GPT] Image downloaded and decoded in {decode_time:.2f}s")
            
            _, height, width, _ = output_tensor.shape
            
            info_text = self._format_info(model, size, api_generation_time,
                                         True, f"{width}x{height}", None, image_count, seed,
                                         total_encode_time, decode_time)
            
            return (output_tensor, info_text)
            
        except Exception as e:
            try:
                error_time = time.time() - start_time
            except:
                error_time = 0.0
            try:
                error_encode_time = total_encode_time
                error_image_count = image_count
            except:
                error_encode_time = 0.0
                error_image_count = 0
            
            info_text = self._format_info(model, size, error_time,
                                         False, None, str(e), error_image_count, seed,
                                         error_encode_time, 0.0)
            black_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (black_image, info_text)
    
    def _format_info(self, model_name, size, generation_time,
                     success, resolution, error_message, input_images, seed,
                     encode_time, decode_time):
        lines = [
            "=" * 60,
            "Vector Engine GPT - Generation Result",
            "=" * 60,
            f"Model: {model_name}",
            f"Size: {size}",
        ]
        
        if resolution:
            lines.append(f"Resolution: {resolution}")
        
        lines.extend([
            f"Seed: {seed}",
            f"Input Images: {input_images}",
            "",
            "--- Timing ---",
            f"Encode: {encode_time:.3f}s | API: {generation_time:.2f}s | Decode: {decode_time:.3f}s",
            f"Total: {(encode_time + generation_time + decode_time):.2f}s",
            "",
            f"Status: {'SUCCESS' if success else 'FAILED'}",
        ])
        
        if not success and error_message:
            lines.extend(["", "Error:", error_message])
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "VectorEngineGemini": VectorEngineGemini,
    "VectorEngineGPT": VectorEngineGPT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VectorEngineGemini": "Vector Engine Gemini",
    "VectorEngineGPT": "Vector Engine GPT",
}

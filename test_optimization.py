"""
Performance test script to compare optimization effects
测试脚本：对比优化前后的性能
"""

import base64
import time
from PIL import Image
import io

def test_compression(image_path, max_sizes=[None, 4096, 2048, 1536], qualities=[100, 95, 85, 75]):
    """
    Test different compression settings
    """
    print(f"\n{'='*70}")
    print(f"Testing image: {image_path}")
    print(f"{'='*70}\n")
    
    # Load original image
    original_img = Image.open(image_path)
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')
    
    orig_width, orig_height = original_img.size
    print(f"Original size: {orig_width}x{orig_height}")
    
    # Get original file size
    with open(image_path, 'rb') as f:
        original_file_size = len(f.read()) / 1024
    print(f"Original file size: {original_file_size:.1f} KB\n")
    
    results = []
    
    for max_size in max_sizes:
        for quality in qualities:
            # Prepare image
            img = original_img.copy()
            
            # Resize if needed
            if max_size is not None:
                width, height = img.size
                if max(width, height) > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int(height * (max_size / width))
                    else:
                        new_height = max_size
                        new_width = int(width * (max_size / height))
                    
                    resize_start = time.time()
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    resize_time = time.time() - resize_start
                else:
                    new_width, new_height = width, height
                    resize_time = 0
            else:
                new_width, new_height = img.size
                resize_time = 0
            
            # Compress and encode
            buffer = io.BytesIO()
            
            encode_start = time.time()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            encode_time = time.time() - encode_start
            
            img_bytes = buffer.getvalue()
            compressed_size_kb = len(img_bytes) / 1024
            
            # Base64 encode
            base64_start = time.time()
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            base64_time = time.time() - base64_start
            base64_size_kb = len(base64_str) / 1024
            
            # Total time
            total_time = resize_time + encode_time + base64_time
            
            # Compression ratio
            compression_ratio = (1 - compressed_size_kb / original_file_size) * 100 if original_file_size > 0 else 0
            
            result = {
                'max_size': max_size or 'Original',
                'quality': quality,
                'dimensions': f"{new_width}x{new_height}",
                'compressed_size_kb': compressed_size_kb,
                'base64_size_kb': base64_size_kb,
                'total_time': total_time,
                'compression_ratio': compression_ratio
            }
            results.append(result)
            
            print(f"Max Size: {str(max_size or 'Original'):>8} | Quality: {quality:>3}% | "
                  f"Dimensions: {new_width:>4}x{new_height:<4} | "
                  f"Size: {compressed_size_kb:>6.1f}KB (base64: {base64_size_kb:>6.1f}KB) | "
                  f"Time: {total_time:.3f}s | "
                  f"Compression: {compression_ratio:>5.1f}%")
    
    # Recommend best settings
    print(f"\n{'='*70}")
    print("Recommendations:")
    print(f"{'='*70}")
    
    # Find best balance (around 85% quality, reasonable size)
    balanced = [r for r in results if r['quality'] == 85 and r['max_size'] == 2048]
    if balanced:
        r = balanced[0]
        print(f"\n✅ Recommended (Balanced):")
        print(f"   max_size=2048, quality=85")
        print(f"   Size: {r['compressed_size_kb']:.1f}KB | Time: {r['total_time']:.3f}s | Compression: {r['compression_ratio']:.1f}%")
    
    # Find fastest
    fastest = min(results, key=lambda x: x['total_time'])
    print(f"\n⚡ Fastest:")
    print(f"   max_size={fastest['max_size']}, quality={fastest['quality']}")
    print(f"   Size: {fastest['compressed_size_kb']:.1f}KB | Time: {fastest['total_time']:.3f}s | Compression: {fastest['compression_ratio']:.1f}%")
    
    # Find best compression
    best_compression = max(results, key=lambda x: x['compression_ratio'])
    print(f"\n📦 Best Compression:")
    print(f"   max_size={best_compression['max_size']}, quality={best_compression['quality']}")
    print(f"   Size: {best_compression['compressed_size_kb']:.1f}KB | Time: {best_compression['total_time']:.3f}s | Compression: {best_compression['compression_ratio']:.1f}%")
    
    print()


if __name__ == "__main__":
    # Test with sample images
    test_images = [
        "./assets/girl.png",
        "./assets/boy.jpg",
    ]
    
    for img_path in test_images:
        try:
            test_compression(img_path)
        except Exception as e:
            print(f"Error testing {img_path}: {e}")
    
    print(f"\n{'='*70}")
    print("Test Complete!")
    print(f"{'='*70}")
    print("\nCurrent node settings: max_size=2048, quality=85")
    print("This provides the best balance of quality, speed, and file size.")


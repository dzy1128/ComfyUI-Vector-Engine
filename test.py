import http.client
import json
import base64
import time

conn = http.client.HTTPSConnection("api.vectorengine.ai")


with open("./assets/girl.png", "rb") as f:
    girl_data = base64.b64encode(f.read()).decode('utf-8')

with open("./assets/boy.jpg", "rb") as f:
    boy_data = base64.b64encode(f.read()).decode('utf-8')

payload = json.dumps({
   "contents": [
      {
         "role": "user",
         "parts": [
            {
               "text": "'Picture 1 shows a girl, and Picture 2 shows a boy. Generate a photo of them hugging."
            },
            {
               "inline_data": {
                  "mime_type": "image/png",
                  "data": girl_data
               }
            },
            {
               "inline_data": {
                  "mime_type": "image/jpeg",
                  "data": boy_data
               }
            }
         ]
      }
   ],
   "generationConfig": {
      "responseModalities": [
         "TEXT",
         "IMAGE"
      ],
      "imageConfig": {
         "aspectRatio": "2:3",
         "imageSize": "1K"
      }
   }
})
headers = {
   'Authorization': 'Bearer sk-RZBLe4v8MolmD3fwOd6vdRTCaj7PohDhf4f44UNgWHEAA4zF',
   'Content-Type': 'application/json'
}

# 记录开始时间
print("开始生成图片...")
start_time = time.time()

conn.request("POST", "/v1beta/models/gemini-3-pro-image-preview:generateContent?key=sk-RZBLe4v8MolmD3fwOd6vdRTCaj7PohDhf4f44UNgWHEAA4zF", payload, headers)
res = conn.getresponse()
data = res.read()

# 记录结束时间并计算耗时
end_time = time.time()
generation_time = end_time - start_time

print(f"✓ 图片生成完成！")
print(f"⏱️  总耗时: {generation_time:.2f} 秒 ({generation_time:.3f}s)")

# 解析响应
response_json = json.loads(data.decode("utf-8"))
#print("响应内容：", json.dumps(response_json, indent=2, ensure_ascii=False))

# 提取并保存图片
saved_images = 0
try:
    # 根据 API 响应结构提取图片数据（可能需要根据实际响应调整路径）
    candidates = response_json.get("candidates", [])
    print(f"\n候选数量: {len(candidates)}")
    
    for i, candidate in enumerate(candidates):
        print(f"\n处理候选 {i}:")
        print(f"候选结构: {list(candidate.keys())}")
        
        content = candidate.get("content", {})
        print(f"内容结构: {list(content.keys())}")
        
        parts = content.get("parts", [])
        print(f"Parts 数量: {len(parts)}")
        
        for j, part in enumerate(parts):
            print(f"\nPart {j} 的键: {list(part.keys())}")
            
            # 检查不同的可能字段名
            if "inline_data" in part:
                print("找到 inline_data!")
                img_base64 = part["inline_data"]["data"]
                mime_type = part["inline_data"].get("mime_type", "image/jpeg")
            elif "inlineData" in part:
                print("找到 inlineData!")
                img_base64 = part["inlineData"]["data"]
                mime_type = part["inlineData"].get("mimeType", "image/jpeg")
            elif "data" in part:
                print("找到 data 字段!")
                img_base64 = part["data"]
                mime_type = part.get("mimeType", part.get("mime_type", "image/jpeg"))
            else:
                print(f"未找到图片数据，part 内容: {part}")
                continue
            
            # 解码 base64 并保存图片
            img_bytes = base64.b64decode(img_base64)
            
            # 根据 mime_type 确定文件扩展名
            ext = "jpg" if "jpeg" in mime_type.lower() else mime_type.split("/")[-1]
            output_path = f"./assets/output_image_{i}_{j}.{ext}"
            
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            
            # 计算图片大小
            img_size_kb = len(img_bytes) / 1024
            print(f"✓ 图片已保存到: {output_path}")
            print(f"  文件大小: {img_size_kb:.2f} KB")
            saved_images += 1
    
    # 显示总结信息
    print(f"\n{'='*50}")
    print(f"📊 生成总结:")
    print(f"  - 成功生成并保存: {saved_images} 张图片")
    print(f"  - 总耗时: {generation_time:.2f} 秒")
    if saved_images > 0:
        print(f"  - 平均每张耗时: {generation_time/saved_images:.2f} 秒")
    print(f"{'='*50}")
            
except Exception as e:
    print(f"\n❌ 提取图片时出错: {e}")
    import traceback
    traceback.print_exc()
# Performance Optimization Guide

## 已实现的优化

### 1. 🖼️ 智能图片压缩

**优化前：**
- 直接将原始图片转换为 PNG/JPEG
- 没有尺寸限制
- PNG 格式文件很大
- 大图片传输缓慢

**优化后：**
- ✅ 自动调整图片尺寸（最大 2048px）
- ✅ 统一使用 JPEG 格式（更好的压缩率）
- ✅ 质量设置为 85%（视觉质量和文件大小的最佳平衡）
- ✅ 启用优化标志（optimize=True）

**预期效果：**
- 文件大小减少 **70-90%**
- 上传时间减少 **70-90%**
- 对生成质量几乎无影响

### 2. ⏱️ 精确的时间统计

**优化前：**
- 时间统计包含所有本地处理时间
- 显示时间远超实际 API 生成时间

**优化后：**
- ✅ 只统计真实的 API 请求时间
- ✅ 不包含图片编码/解码时间
- ✅ 与网站显示时间一致

### 3. 📊 详细的进度日志

**新增功能：**
- ✅ 显示每张图片的编码时间
- ✅ 显示压缩后的文件大小
- ✅ 显示 API 请求的 payload 大小
- ✅ 显示 API 响应时间

**日志示例：**
```
[VectorEngine] Processing 2 input images...
[VectorEngine] Image 1: encoded in 0.15s, size: 245.3KB
[VectorEngine] Image 2: encoded in 0.12s, size: 189.7KB
[VectorEngine] Sending request to API (payload size: 0.58MB)...
[VectorEngine] API request completed in 26.34s
```

## 性能对比

### 示例：2张高分辨率图片 (4000x3000)

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 单张图片大小 | ~3-5 MB | ~200-300 KB | **90%↓** |
| 编码时间 | ~2-3 秒/张 | ~0.1-0.2 秒/张 | **90%↓** |
| 上传时间 | ~10-15 秒 | ~1-2 秒 | **85%↓** |
| 总体耗时 | ~45-60 秒 | ~28-32 秒 | **40-50%↓** |

## 优化参数说明

### 当前配置

```python
max_size = 2048      # 最大尺寸 2048px
quality = 85         # JPEG 质量 85%
format = "JPEG"      # 统一使用 JPEG
optimize = True      # 启用优化
```

### 如何调整

如果需要自定义优化参数，可以修改 `vector_engine_node.py` 第 152 行：

```python
# 更高质量（更慢）
base64_data, mime_type = self.tensor_to_base64(img, max_size=4096, quality=95)

# 更快速度（质量略低）
base64_data, mime_type = self.tensor_to_base64(img, max_size=1536, quality=75)

# 默认平衡（推荐）
base64_data, mime_type = self.tensor_to_base64(img, max_size=2048, quality=85)
```

## 参数推荐

### 质量优先
```python
max_size = 4096
quality = 95
```
- 适合：需要最高质量
- 缺点：处理时间较长

### 平衡模式（默认）
```python
max_size = 2048
quality = 85
```
- 适合：大多数使用场景
- 优点：质量和速度的最佳平衡

### 速度优先
```python
max_size = 1536
quality = 75
```
- 适合：快速测试、预览
- 优点：最快的处理速度

## 网络优化建议

1. **使用稳定的网络连接**
   - API 请求需要良好的网络
   - 考虑使用有线网络

2. **批量处理**
   - 一次处理多张图片比分开处理更高效

3. **合理选择分辨率**
   - 1K: 最快，适合测试
   - 2K: 平衡，推荐使用
   - 4K: 高质量，较慢

## 故障排查

### 如果仍然很慢

1. **检查日志**
   - 查看每个步骤的耗时
   - 确定瓶颈在哪里

2. **图片编码慢**
   - 降低 `max_size` 到 1536 或 1024
   - 降低 `quality` 到 75

3. **API 请求慢**
   - 检查网络连接
   - 尝试较小的输出分辨率（1K 而非 4K）

4. **payload 太大**
   - 减少输入图片数量
   - 降低图片质量参数

## 更多优化空间

### 可能的未来优化

1. **异步处理**
   - 并行编码多张图片

2. **缓存机制**
   - 缓存已编码的图片

3. **连接池**
   - 复用 HTTP 连接

4. **流式处理**
   - 逐步下载生成的图片


# 多模态图像检索系统

> 这是一个面向学习的 AI/机器学习入门项目，通过实践掌握向量数据库、目标检测、图像特征提取等核心技术。

## 项目简介

本项目演示了如何结合 **YOLO 目标检测** 和 **ChromaDB 向量数据库**，构建一个支持多种检索方式的多模态图像系统：

- ✅ **YOLO 检测** - 自动识别图片中的物体
- ✅ **文本搜索** - 用关键词搜索包含特定物体的图片
- ✅ **向量搜索** - 上传图片找视觉相似的图片
- ✅ **智能问答** - 上传图片 + 提问，返回回答 + 可视化标注
- ✅ **持久化存储** - 数据保存在本地数据库中

---

## 核心技术点

### 1. ChromaDB 向量数据库

#### 什么是向量数据库？

传统数据库存储的是**结构化数据**（如文本、数字），查询方式通常是**精确匹配**。

向量数据库存储的是**向量**（一组数字），查询方式是**相似度匹配**。

```
传统数据库:  WHERE name = "张三"     → 精确匹配
向量数据库:  WHERE 距离(image1, image2) < 0.3   → 相似匹配
```

#### 为什么需要向量数据库？

想象你在电商网站搜索"红色连衣裙"，传统数据库会找标题中包含这个词的商品。但向量数据库能理解"红色连衣裙"的**语义**，返回所有视觉上相似的连衣裙图片。

#### ChromaDB 核心概念

```
Client (客户端)     →  连接数据库
    ↓
Collection (集合)   →  相当于数据库中的"表"
    ↓
Document (文档)     →  要存储的原始数据（文本/图片描述）
    ↓
Embedding (向量)    →  Document 转换成的数字向量
    ↓
Metadata (元数据)   →  附加信息（如图片路径、标签）
```

#### 存储方式对比

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **内存模式** | 数据存在内存，重启丢失 | 快速测试 |
| **持久化模式** | 数据存在磁盘文件 | 生产环境 |

#### 本项目代码解析

```python
# chromadb_quickstart.py - 基础示例
import chromadb

# 1. 创建客户端（内存模式）
client = chromadb.Client()

# 2. 创建集合
collection = client.create_collection(name="my_knowledge_base")

# 3. 添加文档（ChromaDB 自动将文本转为向量）
collection.add(
    documents=["关于猫和狗的文档", "我爱吃苹果香蕉"],
    ids=["doc1", "doc2"]
)

# 4. 语义搜索
results = collection.query(
    query_texts=["我想要一只猴子"],  # 注意：语义相近但没有完全匹配的词
    n_results=2
)
print(results['documents'])  # 返回与"猴子"语义最相近的文档
```

**关键点**：当你搜索"猴子"时，即使文档中没有这个词，ChromaDB 也能通过向量相似度找到相关文档（如"关于动物的文档"）。

---

### 2. YOLO 目标检测

#### 什么是目标检测？

目标检测是计算机视觉的核心任务之一，主要包括：

1. **分类** (Classification) - 图片里有什么物体？
2. **定位** (Localization) - 物体在哪里？
3. **检测** (Detection) - 两者结合，输出类别+位置

```
输入图片 → YOLO → 输出: [人, 车, 狗] + 位置坐标
```

#### YOLO 原理简介

**YOLO (You Only Look Once)** 的核心思想是"单次前向传播"：

```
传统方法:  滑动窗口 → 提取候选框 → 分类 → 后处理   (慢)
YOLO:      图片 → CNN一次前向 → 直接输出结果      (快)
```

YOLO 将图片划分为 **S×S** 的网格，每个网格预测：
- 边界框 (Bounding Box): x, y, width, height
- 置信度 (Confidence): 框中是否有物体
- 类别概率 (Class Probability): 属于各类的概率

#### YOLO 模型版本

| 模型 | 参数量 | 速度 | 精度 |
|------|--------|------|------|
| YOLOv8n | 最少 | 最快 | 较低 |
| YOLOv8m | 中等 | 中等 | 中等 |
| YOLOv8l | 较多 | 较慢 | 较高 |
| YOLOv8x | 最多 | 最慢 | 最高 |

本项目使用 **YOLO11n** (nano 版本)，适合学习和快速推理。

#### 本项目代码解析

```python
# multimodal_yolo_store.py
from ultralytics import YOLO
import chromadb

# 1. 加载模型
model = YOLO("yolo11n.pt")

# 2. 检测图片
results = model("bus.jpg")

# 3. 解析结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])           # 类别ID
        cls_name = model.names[cls_id]     # 类别名称
        conf = float(box.conf[0])          # 置信度
        print(f"检测到: {cls_name}, 置信度: {conf}")
```

**输出示例**：
```
bus.jpg:
  检测到: [{'class': 'bus', 'confidence': 0.94}, 
          {'class': 'person', 'confidence': 0.88}, ...]
  统计: {'bus': 1, 'person': 4}
```

---

### 3. CLIP 图像特征提取

#### 什么是 Embedding（嵌入/向量）？

Embedding 是将**离散的、高维的数据**（如文字、图片）转换为**连续的、低维的向量**的过程。

```
文本: "一只可爱的猫" 
      ↓ Embedding
向量: [0.12, -0.34, 0.56, 0.89, ...]  (512维或更高)

图片: [猫的图片]
      ↓ Embedding  
向量: [0.45, 0.12, -0.67, 0.33, ...]  (512维)
```

#### 为什么需要 Embedding？

1. **数学运算**：向量可以进行加减乘除，实现语义操作
   ```
   vec("国王") - vec("男人") + vec("女人") ≈ vec("女王")
   ```

2. **相似度计算**：通过向量距离判断语义相似度
   ```
   距离(vec("猫"), vec("狗")) < 距离(vec("猫"), vec("汽车"))
   ```

#### CLIP 原理

**CLIP (Contrastive Language-Image Pre-training)** 是 OpenAI 提出的多模态模型：

```
文本编码器 (Transformer)     图片编码器 (ViT)
       ↓                            ↓
   文本向量                   图片向量
       ↓                            ↓
   对比学习，使匹配的图文向量接近，不匹配的远离
```

CLIP 的特点是：
- **零样本能力**：没见过的类别也能识别
- **多模态**：同时理解图像和文本
- **广泛应用**：图像搜索、分类、生成等

#### 本项目代码解析

```python
# multimodal_clip_extract.py
from sentence_transformers import SentenceTransformer

# 加载 CLIP 模型
model = SentenceTransformer('clip-ViT-B-32')

# 提取图片向量
image_paths = ["bus.jpg", "test001.jpg"]
embeddings = model.encode(image_paths)

print(f"向量维度: {embeddings.shape}")  # (2, 512)
print(f"向量示例: {embeddings[0][:5]}")  # [0.12, -0.34, 0.56, 0.89, 0.23]
```

---

### 4. 相似度搜索

#### 距离度量方式

向量数据库使用多种方式计算"距离"：

| 方法 | 公式 | 特点 |
|------|------|------|
| **欧氏距离 (L2)** | √(Σ(a-b)²) | 直观，符合直觉 |
| **余弦相似度** | cos(θ) = a·b/(|a||b|) | 关注方向 |
| **点积** | a·b | 快速，适合归一化向量 |

```
L2距离越小 = 越相似
余弦相似度越大 = 越相似
```

#### ChromaDB 查询示例

```python
# multimodal_vector_search.py
results = vector_collection.query(
    query_embeddings=[query_vector.tolist()],  # 查询向量
    n_results=3                                  # 返回前3个
)

# 结果包含:
# results['ids']        # 图片ID
# results['distances']  # 距离值
# results['metadatas']  # 元数据
```

---

## 项目结构

```
Project02/
├── 基础学习/
│   ├── chromadb_quickstart.py      # ChromaDB 入门
│   ├── chromadb_persistent.py      # 持久化存储
│   ├── roboflow_detection.py       # Roboflow API 检测
│   └── roboflow_detection_visualized.py  # 检测可视化
│
├── Web 应用/
│   ├── app.py                     # Streamlit Web 界面
│   └── requirements.txt           # 依赖列表
│
├── 多模态检索/
│   ├── multimodal_yolo_store.py    # YOLO检测 + 存储
│   ├── multimodal_text_search.py   # 文本搜索
│   ├── multimodal_clip_extract.py  # CLIP向量提取
│   └── multimodal_vector_search.py # 向量相似搜索
│
├── 智能问答/
│   └── multimodal_rag_qa.py        # 通义千问 LLM 问答（核心逻辑）
│
├── docs/plans/                      # 设计文档
├── multimodal_db/                   # ChromaDB 数据存储
├── output/                          # 标注结果输出
├── yolo11n.pt                       # YOLO 模型文件
└── *.jpg                            # 测试图片
```

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install streamlit chromadb ultralytics sentence-transformers opencv-python dashscope
```

### 运行步骤

#### 1. YOLO 检测 + 元数据存储

```bash
python multimodal_yolo_store.py
```

**输入**: 图片文件 (bus.jpg, test001.jpg 等)  
**输出**: 检测到的物体类别、数量、置信度，存入 ChromaDB

#### 2. 文本搜索

```bash
# 搜索包含"car"的图片
python multimodal_text_search.py car

# 搜索包含"bus"的图片  
python multimodal_text_search.py bus
```

**输入**: 文本关键词  
**输出**: 包含该物体的图片列表

#### 3. CLIP 向量提取

```bash
python multimodal_clip_extract.py
```

**输入**: 图片文件  
**输出**: 图片向量存入数据库

#### 4. 向量相似搜索

```bash
# 找与 bus.jpg 相似的图片
python multimodal_vector_search.py bus.jpg
```

**输入**: 查询图片  
**输出**: 相似图片列表及距离值

---

### 5. 智能问答（RAG）

```bash
# 设置 API Key（可选，已内置默认值）
# 方式一：环境变量
set DASHSCOPE_API_KEY=your-api-key

# 方式二：直接运行（脚本中已内置示例 key）
python multimodal_rag_qa.py bus.jpg "这张图片里有什么车？"
```

**输入**: 图片 + 问题  
**输出**: 
- LLM 回答（自然语言）
- 带标注的图片（框出检测到的物体）

**工作流程**:
```
上传图片 → YOLO检测 → 可视化标注 → 通义千问分析 → 返回回答 + 标注图片
```

**支持的模型**: qwen-max（默认）、qwen-plus 等

---

### 6. Web 界面（推荐）

```bash
# 启动 Web 应用
streamlit run app.py

# 访问
# http://localhost:8501
```

**功能**:
- 📤 图片上传（支持拖拽）
- 💬 问题输入（支持快速问题按钮）
- 📊 实时显示检测结果和标注图片
- 💡 通义千问智能回答
- 📜 历史记录保存

**界面预览**:
```
┌─────────────────────────────────────────────────┐
│  🤖 多模态智能问答系统                            │
├────────────────────┬────────────────────────────┤
│  📤 上传图片        │  📊 分析结果               │
│  [选择文件]        │  [带标注图片]               │
│                    │                            │
│  💬 提问            │  💡 智能回答               │
│  这张图片里有什么车？│  检测到1辆公交车...        │
│                    │                            │
│  [🚀 开始分析]      │  📜 历史记录               │
└────────────────────┴────────────────────────────┘
```

---

## 进阶扩展

### 方向 1: RAG 问答系统

结合大语言模型 (LLM)，实现智能问答：

```
用户问题 → 提取关键词 → ChromaDB搜索 → LLM生成答案
```

### 方向 2: 实时视频检测

将 YOLO 扩展到视频流：

```python
import cv2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)  # 实时检测
    # 可视化...
```

### 方向 3: 以图搜图

上传图片 → CLIP提取向量 → ChromaDB搜索 → 返回相似图片

### 方向 4: 物体计数统计

分析图库中各类物体的出现次数：

```python
# 统计所有图片中"车"的出现次数
all_results = collection.get()
total_cars = sum(json.loads(m['class_count']).get('car', 0) 
                for m in all_results['metadatas'])
```

---

## 常见问题

### Q1: 第一次运行很慢？

A: 需要下载 YOLO/CLIP 模型，首次运行会自动下载，后续会缓存。

### Q2: 搜索结果不理想？

A: 
- 尝试调整置信度阈值 (conf)
- 增加更多训练图片
- 使用更大的 CLIP 模型

### Q3: 如何添加新图片？

A: 在对应脚本的图片列表中添加路径即可。

---

## 参考资料

- [ChromaDB 文档](https://docs.trychroma.com/)
- [YOLO 官方文档](https://docs.ultralytics.com/)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
- [Sentence-Transformers](https://sbert.net/)

---

## 学习路径建议

1. **第一阶段（推荐）**: 运行 `streamlit run app.py`，体验 Web 界面
2. **第二阶段**: 运行 chromadb_quickstart.py，理解向量数据库基本操作
3. **第三阶段**: 运行 roboflow_detection.py，了解目标检测
4. **第四阶段**: 运行 multimodal_yolo_store.py，理解检测+存储流程
5. **第五阶段**: 运行 multimodal_vector_search.py，掌握向量搜索
6. **第六阶段**: 运行 multimodal_rag_qa.py，深入理解 LLM 问答原理
7. **第七阶段**: 尝试扩展，如多轮对话、语音输入等

---

## 许可证

MIT License - 供学习使用

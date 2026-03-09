# 多模态图像检索系统实现计划

> **For Claude:** 使用 executing-plans skill 逐任务实现。

**目标:** 构建结合 YOLO 目标检测和 ChromaDB 向量数据库的多模态检索系统

**架构:** 分步实现 - 先实现 YOLO+元数据存储，再实现 CLIP 向量搜索

**技术栈:** YOLO, ChromaDB, CLIP, OpenCV

---

## Task 1: YOLO检测 + ChromaDB元数据存储

**Files:**
- Create: `demo07.py`

**Step 1: 编写 demo07.py**

```python
from ultralytics import YOLO
import chromadb
import os
import json

# 初始化 YOLO 模型 (使用本地 yolo11n.pt)
model = YOLO("yolo11n.pt")

# 初始化 ChromaDB
client = chromadb.PersistentClient(path="./multimodal_db")
collection = client.get_or_create_collection(name="images_with_metadata")

def detect_and_store(image_path, image_id):
    """检测图片物体并存储到ChromaDB"""
    # YOLO检测
    results = model(image_path)
    
    # 提取检测结果
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({
                "class": cls_name,
                "confidence": conf
            })
    
    # 统计物体类别
    classes = [d["class"] for d in detections]
    class_count = {}
    for c in classes:
        class_count[c] = class_count.get(c, 0) + 1
    
    # 存入ChromaDB (元数据方式)
    metadata = {
        "image_path": image_path,
        "classes": json.dumps(classes),
        "class_count": json.dumps(class_count),
        "total_objects": len(detections)
    }
    
    collection.upsert(
        ids=[image_id],
        metadatas=[metadata],
        documents=[f"Image with {', '.join(set(classes))}"]
    )
    
    return detections, class_count

# 测试：处理示例图片
if __name__ == "__main__":
    test_images = ["bus.jpg", "test_imag01.jpg"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            detections, counts = detect_and_store(img_path, img_path)
            print(f"\n{img_path}:")
            print(f"  检测到: {detections}")
            print(f"  统计: {counts}")
```

**Step 2: 运行测试**

Run: `python demo07.py`
Expected: 输出检测结果并存储到ChromaDB

**Step 3: 验证数据存储**

Run: `python -c "import chromadb; c=chromadb.PersistentClient(path='./multimodal_db'); col=c.get_collection('images_with_metadata'); print(col.get())"`
Expected: 看到存储的元数据

---

## Task 2: 文本搜索图片

**Files:**
- Create: `demo08.py`

**Step 1: 编写 demo08.py**

```python
import chromadb
from PIL import Image
import cv2

client = chromadb.PersistentClient(path="./multimodal_db")
collection = client.get_or_create_collection(name="images_with_metadata")

def search_by_text(query):
    """文本搜索：按物体类别搜索图片"""
    # 查询时匹配包含该类别的图片
    results = collection.query(
        query_texts=[query],
        n_results=10
    )
    return results

def show_results(results):
    """展示搜索结果"""
    if not results['ids'] or not results['ids'][0]:
        print("未找到匹配图片")
        return
    
    print(f"\n找到 {len(results['ids'][0])} 张图片:")
    
    for i, (img_id, meta, doc) in enumerate(zip(
        results['ids'][0],
        results['metadatas'][0],
        results['documents'][0]
    )):
        print(f"\n--- 结果 {i+1} ---")
        print(f"图片: {meta['image_path']}")
        print(f"类别: {meta['classes']}")
        print(f"物体数量: {meta['total_objects']}")
        
        # 显示图片
        img = cv2.imread(meta['image_path'])
        if img is not None:
            cv2.imshow(f"Result {i+1}: {img_id}", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 测试文本搜索
    query = "car"  # 搜索包含汽车的图片
    results = search_by_text(query)
    show_results(results)
```

**Step 2: 运行测试**

Run: `python demo08.py`
Expected: 搜索并显示包含指定物体的图片

---

## Task 3: CLIP向量提取 + 存入ChromaDB

**Files:**
- Create: `demo09.py`

**Step 1: 安装 CLIP**

Run: `pip install sentence-transformers`

**Step 2: 编写 demo09.py**

```python
from sentence_transformers import SentenceTransformer
import chromadb
import os
from PIL import Image
import torch

# 加载 CLIP 模型
model = SentenceTransformer('clip-ViT-B-32')

# 初始化向量数据库
client = chromadb.PersistentClient(path="./multimodal_db")
vector_collection = client.get_or_create_collection(name="image_vectors")

def extract_and_store_vectors(image_paths):
    """提取图片向量并存入ChromaDB"""
    # 批量提取向量
    embeddings = model.encode(image_paths)
    
    # 存入数据库
    for img_path, embedding in zip(image_paths, embeddings):
        vector_collection.upsert(
            ids=[img_path],
            embeddings=[embedding.tolist()],
            metadatas=[{"image_path": img_path}]
        )
    
    print(f"已存储 {len(image_paths)} 张图片的向量")

def extract_single_vector(image_path):
    """提取单张图片向量"""
    embedding = model.encode([image_path])
    return embedding[0]

if __name__ == "__main__":
    # 测试：处理现有图片
    test_images = ["bus.jpg", "test_imag01.jpg"]
    test_images = [f for f in test_images if os.path.exists(f)]
    
    if test_images:
        extract_and_store_vectors(test_images)
        print("向量存储完成")
```

**Step 3: 运行测试**

Run: `python demo09.py`
Expected: 提取并存储图片向量

---

## Task 4: 向量相似图片搜索

**Files:**
- Create: `demo10.py`

**Step 1: 编写 demo10.py**

```python
from sentence_transformers import SentenceTransformer
import chromadb
import cv2

# 加载 CLIP 模型
model = SentenceTransformer('clip-ViT-B-32')

# 初始化向量数据库
client = chromadb.PersistentClient(path="./multimodal_db")
vector_collection = client.get_or_create_collection(name="image_vectors")

def search_similar_images(query_image_path, top_k=3):
    """向量相似度搜索"""
    # 提取查询图片向量
    query_embedding = model.encode([query_image_path])
    
    # 向量搜索
    results = vector_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    return results

def show_similar_results(results, query_image_path):
    """展示相似图片结果"""
    if not results['ids'] or not results['ids'][0]:
        print("未找到相似图片")
        return
    
    print(f"\n找到 {len(results['ids'][0])} 张相似图片:")
    
    # 显示查询图片
    query_img = cv2.imread(query_image_path)
    cv2.imshow(f"Query: {query_image_path}", query_img)
    
    # 显示结果
    for i, (img_id, meta) in enumerate(zip(
        results['ids'][0],
        results['metadatas'][0]
    )):
        img = cv2.imread(meta['image_path'])
        if img is not None:
            cv2.imshow(f"Similar {i+1}: {img_id}", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 测试：用 bus.jpg 搜索相似图片
    results = search_similar_images("bus.jpg")
    show_similar_results(results, "bus.jpg")
```

**Step 2: 运行测试**

Run: `python demo10.py`
Expected: 显示与查询图片相似的图片

---

## 执行完成

全部完成后，你将拥有：
- `demo07.py` - YOLO检测 + 元数据存储
- `demo08.py` - 文本搜索图片
- `demo09.py` - CLIP向量提取和存储
- `demo10.py` - 向量相似度搜索

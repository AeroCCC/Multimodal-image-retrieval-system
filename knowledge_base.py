"""
知识库管理模块
功能：管理 ChromaDB 向量知识库，支持文本检索和向量检索
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from datetime import datetime

# 知识库配置
KNOWLEDGE_DB_PATH = "./knowledge_db"
COLLECTION_NAME = "image_knowledge"

# 全局变量
_clip_model = None
_client = None
_collection = None


def get_clip_model():
    """获取 CLIP 模型（单例模式）"""
    global _clip_model
    if _clip_model is None:
        print("加载 CLIP 模型...")
        _clip_model = SentenceTransformer("clip-ViT-B-32")
    return _clip_model


def get_client():
    """获取 ChromaDB 客户端"""
    global _client
    if _client is None:
        os.makedirs(KNOWLEDGE_DB_PATH, exist_ok=True)
        _client = chromadb.PersistentClient(path=KNOWLEDGE_DB_PATH)
    return _client


def get_collection():
    """获取知识库集合"""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"description": "多模态图像知识库"}
        )
    return _collection


def init_knowledge_base():
    """初始化知识库"""
    collection = get_collection()
    count = collection.count()
    print(f"知识库初始化完成，当前包含 {count} 条记录")
    return count


def get_image_embedding(image_path):
    """
    使用 CLIP 提取图片的特征向量

    参数:
        image_path: 图片路径

    返回:
        list: 512维向量
    """
    model = get_clip_model()
    embedding = model.encode(image_path)
    return embedding.tolist()


def get_text_embedding(text):
    """
    使用 CLIP 提取文本的特征向量

    参数:
        text: 文本

    返回:
        list: 512维向量
    """
    model = get_clip_model()
    embedding = model.encode(text)
    return embedding.tolist()


def add_to_knowledge_base(image_path, detections, metadata=None):
    """
    添加图片到知识库

    参数:
        image_path: 图片路径
        detections: YOLO 检测结果列表
        metadata: 额外元数据（如场景标签等）

    返回:
        str: 添加的图片 ID
    """
    collection = get_collection()

    # 生成唯一 ID
    image_id = f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # 提取向量
    embedding = get_image_embedding(image_path)

    # 构建文档内容
    classes = [d["class"] for d in detections]
    class_count = {}
    for cls in classes:
        class_count[cls] = class_count.get(cls, 0) + 1

    class_summary = ", ".join([f"{k}{v}个" for k, v in class_count.items()])
    document = f"图片包含: {class_summary}"

    # 构建元数据
    meta = {
        "image_path": image_path,
        "classes": ",".join(set(classes)),
        "class_count": str(class_count),
        "total_objects": len(detections),
        "timestamp": datetime.now().isoformat(),
    }

    # 添加额外元数据
    if metadata:
        meta.update(metadata)

    # 存入 ChromaDB
    collection.add(
        ids=[image_id], embeddings=[embedding], documents=[document], metadatas=[meta]
    )

    print(f"已添加图片到知识库: {image_id}")
    return image_id


def search_by_vector(image_path, top_k=3):
    """
    向量相似度搜索
    找到与给定图片最相似的历史图片

    参数:
        image_path: 查询图片路径
        top_k: 返回结果数量

    返回:
        dict: 检索结果
    """
    collection = get_collection()

    # 提取查询图片向量
    query_embedding = get_image_embedding(image_path)

    # 向量检索
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return results


def search_by_text(query_text, top_k=3):
    """
    文本语义搜索
    根据文本描述找到相关的历史图片

    参数:
        query_text: 查询文本
        top_k: 返回结果数量

    返回:
        dict: 检索结果
    """
    collection = get_collection()

    # 提取文本向量
    query_embedding = get_text_embedding(query_text)

    # 向量检索
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return results


def search_hybrid(image_path=None, query_text=None, top_k=3):
    """
    混合检索
    同时使用向量检索和文本检索，然后融合结果

    参数:
        image_path: 查询图片路径（可选）
        query_text: 查询文本（可选）
        top_k: 返回结果数量

    返回:
        dict: 融合后的检索结果
    """
    results = {"vector_results": None, "text_results": None, "combined": []}

    if image_path:
        results["vector_results"] = search_by_vector(image_path, top_k)

    if query_text:
        results["text_results"] = search_by_text(query_text, top_k)

    # 融合结果
    if results["vector_results"] and results["vector_results"]["ids"]:
        for i, (img_id, meta, dist) in enumerate(
            zip(
                results["vector_results"]["ids"][0],
                results["vector_results"]["metadatas"][0],
                results["vector_results"]["distances"][0],
            )
        ):
            results["combined"].append(
                {"id": img_id, "metadata": meta, "distance": dist, "source": "vector"}
            )

    if results["text_results"] and results["text_results"]["ids"]:
        for i, (img_id, meta, dist) in enumerate(
            zip(
                results["text_results"]["ids"][0],
                results["text_results"]["metadatas"][0],
                results["text_results"]["distances"][0],
            )
        ):
            # 避免重复
            existing_ids = [r["id"] for r in results["combined"]]
            if img_id not in existing_ids:
                results["combined"].append(
                    {"id": img_id, "metadata": meta, "distance": dist, "source": "text"}
                )

    return results


def get_all_images():
    """
    获取知识库中所有图片

    返回:
        dict: 所有图片的信息
    """
    collection = get_collection()
    return collection.get()


def get_knowledge_base_stats():
    """
    获取知识库统计信息

    返回:
        dict: 统计信息
    """
    collection = get_collection()
    all_data = collection.get()

    if not all_data["ids"]:
        return {"total_images": 0, "classes": {}}

    # 统计各类别
    all_classes = []
    for meta in all_data["metadatas"]:
        if "classes" in meta:
            all_classes.extend(meta["classes"].split(","))

    class_count = {}
    for cls in all_classes:
        class_count[cls] = class_count.get(cls, 0) + 1

    return {"total_images": len(all_data["ids"]), "classes": class_count}


def clear_knowledge_base():
    """清空知识库"""
    client = get_client()
    client.delete_collection(name=COLLECTION_NAME)
    global _collection
    _collection = None
    print("知识库已清空")


if __name__ == "__main__":
    # 测试
    init_knowledge_base()
    stats = get_knowledge_base_stats()
    print(f"统计: {stats}")

from sentence_transformers import SentenceTransformer
import chromadb
import os

# 加载 CLIP 模型
model = SentenceTransformer("clip-ViT-B-32")

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
            metadatas=[{"image_path": img_path}],
        )

    print(f"已存储 {len(image_paths)} 张图片的向量")


def extract_single_vector(image_path):
    """提取单张图片向量"""
    embedding = model.encode([image_path])
    return embedding[0]


if __name__ == "__main__":
    # 测试：处理现有图片
    test_images = ["bus.jpg", "test001.jpg"]
    test_images = [f for f in test_images if os.path.exists(f)]

    if test_images:
        extract_and_store_vectors(test_images)
        print("向量存储完成")

        # 验证
        result = vector_collection.get()
        print(f"向量库中共有 {len(result['ids'])} 张图片")

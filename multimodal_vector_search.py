from sentence_transformers import SentenceTransformer
import chromadb

# 加载 CLIP 模型
model = SentenceTransformer("clip-ViT-B-32")

# 初始化向量数据库
client = chromadb.PersistentClient(path="./multimodal_db")
vector_collection = client.get_or_create_collection(name="image_vectors")


def search_similar_images(query_image_path, top_k=3):
    """向量相似度搜索"""
    # 提取查询图片向量
    query_embedding = model.encode(query_image_path)

    # 确保是2D数组
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # 向量搜索
    results = vector_collection.query(
        query_embeddings=query_embedding.tolist(), n_results=top_k
    )

    return results


def show_similar_results(results, query_image_path):
    """展示相似图片结果"""
    if not results["ids"] or not results["ids"][0]:
        print("未找到相似图片")
        return

    print(f"\n查询图片: {query_image_path}")
    print(f"找到 {len(results['ids'][0])} 张相似图片:")

    for i, (img_id, meta, distance) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ):
        print(f"\n--- 相似 {i + 1}: {img_id} ---")
        print(f"路径: {meta['image_path']}")
        print(f"距离(L2): {distance:.4f}")


if __name__ == "__main__":
    import sys

    # 支持命令行参数：python multimodal_vector_search.py <图片路径>
    if len(sys.argv) > 1:
        query_image = sys.argv[1]
        results = search_similar_images(query_image)
        show_similar_results(results, query_image)
    else:
        # 默认测试
        # 测试：用 bus.jpg 搜索相似图片
        results = search_similar_images("bus.jpg")
        show_similar_results(results, "bus.jpg")

        print("\n" + "=" * 50)

        # 测试：用 test001 搜索
        results = search_similar_images("test001.jpg")
        show_similar_results(results, "test001.jpg")

        print("\n" + "=" * 50)

        # 测试：用 test002 搜索
        results = search_similar_images("test002.jpg")
        show_similar_results(results, "test002.jpg")

        print("\n" + "=" * 50)

        # 测试：用 test003 搜索
        results = search_similar_images("test003.jpg")
        show_similar_results(results, "test003.jpg")

        print("\n" + "=" * 50)

        # 测试：用 test004 搜索
        results = search_similar_images("test004.jpg")
        show_similar_results(results, "test004.jpg")

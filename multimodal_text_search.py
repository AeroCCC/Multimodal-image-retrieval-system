import chromadb
import cv2
import json

client = chromadb.PersistentClient(path="./multimodal_db")
collection = client.get_or_create_collection(name="images_with_metadata")


def search_by_text(query):
    """文本搜索：按物体类别搜索图片"""
    results = collection.query(query_texts=[query], n_results=10)
    return results


def show_results(results, show_image=False):
    """展示搜索结果"""
    if not results["ids"] or not results["ids"][0]:
        print("未找到匹配图片")
        return

    print(f"找到 {len(results['ids'][0])} 张图片:")

    for i, (img_id, meta, doc) in enumerate(
        zip(results["ids"][0], results["metadatas"][0], results["documents"][0])
    ):
        print(f"\n--- 结果 {i + 1} ---")
        print(f"图片: {meta['image_path']}")
        print(f"类别: {meta['classes']}")
        print(f"物体数量: {meta['total_objects']}")

        # 显示图片（仅在需要时）
        if show_image:
            try:
                img = cv2.imread(meta["image_path"])
                if img is not None:
                    cv2.imshow(f"Result {i + 1}: {img_id}", img)
            except Exception:
                pass

    if show_image:
        try:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    # 支持命令行参数搜索
    if len(sys.argv) > 1:
        query = sys.argv[1]
        results = search_by_text(query)
        show_results(results)
    else:
        # 默认测试
        queries = ["car", "bus", "person"]
        for query in queries:
            print(f"\n{'=' * 50}")
            print(f"搜索: {query}")
            print("=" * 50)
            results = search_by_text(query)
            show_results(results, show_image=False)

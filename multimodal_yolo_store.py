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
            detections.append({"class": cls_name, "confidence": conf})

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
        "total_objects": len(detections),
    }

    collection.upsert(
        ids=[image_id],
        metadatas=[metadata],
        documents=[f"Image with {', '.join(set(classes))}"],
    )

    return detections, class_count


# 测试：处理示例图片
if __name__ == "__main__":
    test_images = ["bus.jpg", "test001.jpg","test002.jpg","test003.jpg","test004.jpg"]

    for img_path in test_images:
        if os.path.exists(img_path):
            detections, counts = detect_and_store(img_path, img_path)
            print(f"\n{img_path}:")
            print(f"  检测到: {detections}")
            print(f"  统计: {counts}")

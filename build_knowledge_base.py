"""
构建知识库脚本
功能：批量将图片及其检测结果添加到 ChromaDB 知识库
"""

import os
import glob
import argparse
from ultralytics import YOLO
from knowledge_base import (
    init_knowledge_base,
    add_to_knowledge_base,
    get_knowledge_base_stats,
    clear_knowledge_base,
    get_collection,
)
from scene_classifier import classify_scene

# 配置
YOLO_MODEL_PATH = "yolo11n.pt"
DEFAULT_IMAGE_DIR = "./images"


def yolo_detect(image_path, model):
    """
    YOLO 目标检测

    参数:
        image_path: 图片路径
        model: YOLO 模型

    返回:
        list: 检测结果列表
    """
    results = model(image_path)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            detections.append({"class": cls_name, "confidence": conf})

    return detections


def build_from_directory(image_dir, scene_label=None, clear_existing=False):
    """
    从目录批量构建知识库

    参数:
        image_dir: 图片目录路径
        scene_label: 场景标签（可选）
        clear_existing: 是否清空现有知识库

    返回:
        int: 添加的图片数量
    """
    # 初始化 YOLO 模型
    print("加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL_PATH)

    # 初始化知识库
    init_knowledge_base()

    if clear_existing:
        print("清空现有知识库...")
        clear_knowledge_base()

    # 获取图片文件
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_paths:
        print(f"目录 {image_dir} 中没有找到图片")
        return 0

    print(f"找到 {len(image_paths)} 张图片")

    # 批量处理
    success_count = 0
    for i, img_path in enumerate(image_paths):
        print(f"\n[{i + 1}/{len(image_paths)}] 处理: {img_path}")

        try:
            # YOLO 检测
            detections = yolo_detect(img_path, model)
            print(f"  检测到 {len(detections)} 个物体")

            # 构建元数据
            metadata = {}
            if scene_label:
                metadata["scene"] = scene_label
            else:
                # 自动场景标签（零样本 CLIP）
                scene_info = classify_scene(img_path)
                metadata["scene"] = scene_info["scene"]
                metadata["scene_day_night"] = scene_info["day_night"]
                metadata["scene_indoor_outdoor"] = scene_info["indoor_outdoor"]
                metadata["scene_confidence"] = scene_info["confidence"]

            # 添加到知识库
            add_to_knowledge_base(img_path, detections, metadata)
            success_count += 1

        except Exception as e:
            print(f"  处理失败: {e}")

    # 打印统计
    print("\n" + "=" * 50)
    stats = get_knowledge_base_stats()
    print(f"知识库构建完成!")
    print(f"  总图片数: {stats['total_images']}")
    print(f"  物体类别: {stats['classes']}")
    print(f"  成功添加: {success_count}")
    print("=" * 50)

    return success_count


def build_from_file_list(file_list, metadata_list=None):
    """
    从文件列表构建知识库

    参数:
        file_list: 图片路径列表
        metadata_list: 元数据列表（可选）

    返回:
        int: 添加的图片数量
    """
    print("加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL_PATH)

    init_knowledge_base()

    success_count = 0
    for i, img_path in enumerate(file_list):
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue

        print(f"\n[{i + 1}/{len(file_list)}] 处理: {img_path}")

        try:
            detections = yolo_detect(img_path, model)
            print(f"  检测到 {len(detections)} 个物体")

            metadata = (
                metadata_list[i] if metadata_list and i < len(metadata_list) else None
            )
            if metadata is None:
                metadata = {}
            if "scene" not in metadata:
                scene_info = classify_scene(img_path)
                metadata["scene"] = scene_info["scene"]
                metadata["scene_day_night"] = scene_info["day_night"]
                metadata["scene_indoor_outdoor"] = scene_info["indoor_outdoor"]
                metadata["scene_confidence"] = scene_info["confidence"]
            add_to_knowledge_base(img_path, detections, metadata)
            success_count += 1

        except Exception as e:
            print(f"  处理失败: {e}")

    print(f"\n成功添加 {success_count} 张图片到知识库")
    return success_count


def interactive_build():
    """交互式构建知识库"""
    print("=" * 50)
    print("知识库构建工具")
    print("=" * 50)

    # 选择模式
    print("\n请选择模式:")
    print("1. 从目录批量导入")
    print("2. 从文件列表导入")
    print("3. 查看知识库统计")
    print("4. 清空知识库")

    choice = input("\n请输入选项 (1-4): ").strip()

    if choice == "1":
        image_dir = input("图片目录路径 (默认: ./images): ").strip()
        if not image_dir:
            image_dir = DEFAULT_IMAGE_DIR

        scene_label = input("场景标签 (可选，直接回车跳过): ").strip() or None

        confirm = (
            input(f"将从目录 '{image_dir}' 构建知识库，是否继续? (y/n): ")
            .strip()
            .lower()
        )
        if confirm == "y":
            build_from_directory(image_dir, scene_label)

    elif choice == "2":
        print("请输入图片路径（每行一个），完成后输入空行结束:")
        files = []
        while True:
            line = input().strip()
            if not line:
                break
            files.append(line)

        if files:
            build_from_file_list(files)

    elif choice == "3":
        init_knowledge_base()
        stats = get_knowledge_base_stats()
        print(f"\n知识库统计:")
        print(f"  总图片数: {stats['total_images']}")
        print(f"  物体类别统计: {stats['classes']}")

    elif choice == "4":
        confirm = input("确定要清空知识库吗? (y/n): ").strip().lower()
        if confirm == "y":
            clear_knowledge_base()
            print("知识库已清空")

    else:
        print("无效选项")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 命令行模式
        parser = argparse.ArgumentParser(description="构建知识库")
        parser.add_argument(
            "image_dir", nargs="?", default=DEFAULT_IMAGE_DIR, help="图片目录"
        )
        parser.add_argument("--scene", type=str, help="场景标签")
        parser.add_argument("--clear", action="store_true", help="清空现有知识库")

        args = parser.parse_args()

        build_from_directory(args.image_dir, args.scene, args.clear)
    else:
        # 交互模式
        interactive_build()

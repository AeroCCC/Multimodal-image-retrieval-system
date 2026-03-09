"""
多模态智能问答系统
功能：用户上传图片 + 提问 -> YOLO检测 + 可视化标注 -> 通义千问生成回答
"""

import os
import cv2
import json
import dashscope
from ultralytics import YOLO
from datetime import datetime

# 配置
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY", "sk-7b542bb841ba4994b086998bbc45afeb"
)
YOLO_MODEL_PATH = "yolo11n.pt"
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置 API Key
dashscope.api_key = DASHSCOPE_API_KEY

# 加载 YOLO 模型
print("加载 YOLO 模型...")
yolo_model = YOLO(YOLO_MODEL_PATH)


def detect_objects(image_path):
    """
    YOLO 目标检测
    返回: 检测结果列表
    """
    results = yolo_model(image_path)

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            conf = float(box.conf[0])

            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 1) for x in [x1, y1, x2, y2]],
                }
            )

    return detections


def visualize_detections(image_path, detections, output_path):
    """
    在图片上绘制检测框
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 颜色映射
    color_map = {
        "car": (0, 255, 0),  # 绿色
        "bus": (255, 0, 0),  # 蓝色
        "truck": (0, 165, 255),  # 橙色
        "person": (255, 0, 255),  # 紫色
        "motorcycle": (255, 255, 0),  # 青色
        "bicycle": (0, 255, 255),  # 黄色
    }

    for det in detections:
        cls_name = det["class"]
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]

        # 获取颜色
        color = color_map.get(cls_name, (0, 255, 0))

        # 绘制矩形框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 绘制标签
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # 标签背景
        cv2.rectangle(
            img,
            (int(x1), int(y1) - label_size[1] - 10),
            (int(x1) + label_size[0], int(y1)),
            color,
            -1,
        )

        # 标签文字
        cv2.putText(
            img,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    # 保存图片
    cv2.imwrite(output_path, img)
    return output_path


def format_detection_summary(detections):
    """
    将检测结果格式化为文本描述
    """
    if not detections:
        return "图片中未检测到任何物体"

    # 统计各类别数量
    class_count = {}
    for det in detections:
        cls = det["class"]
        class_count[cls] = class_count.get(cls, 0) + 1

    # 格式化
    summary = []
    for cls, count in sorted(class_count.items()):
        summary.append(f"{cls} {count}个")

    # 添加详细信息
    details = "\n".join(
        [
            f"- {d['class']}: 置信度 {d['confidence']:.2%}"
            for d in detections[:10]  # 最多显示10个
        ]
    )

    return f"检测到 {len(detections)} 个物体：{', '.join(summary)}\n\n详细信息：\n{details}"


def ask_qwen(question, detections, image_name):
    """
    调用通义千问 API 生成回答
    """
    detection_summary = format_detection_summary(detections)

    prompt = f"""你是一个图像分析助手。用户上传了一张图片「{image_name}」，并提出了问题。

图片检测结果：
{detection_summary}

用户问题：{question}

请根据图片检测结果，用友好的方式回答用户的问题。如果检测结果与问题相关，请结合实际情况回答。如果无法从检测结果中找到答案，请如实说明。
"""

    try:
        response = dashscope.Generation.call(
            model="qwen-max", prompt=prompt, result_format="message"
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"API 调用失败: {response.code} - {response.message}"

    except Exception as e:
        return f"调用出错: {str(e)}"


def process_image_question(image_path, question, api_key=None):
    """
    主处理流程：检测 -> 可视化 -> 问答

    参数:
        image_path: 图片路径
        question: 用户问题
        api_key: 通义千问 API Key（可选，默认使用环境变量）

    返回:
        dict: {
            "detections": 检测结果,
            "answer": LLM回答,
            "annotated_image_path": 带标注图片路径
        }
    """
    # 设置 API Key
    if api_key:
        dashscope.api_key = api_key

    # 检查文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    image_name = os.path.basename(image_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"result_{timestamp}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print(f"\n{'=' * 50}")
    print(f"图片: {image_name}")
    print(f"问题: {question}")
    print("=" * 50)

    # 1. YOLO 检测
    print("\n[1/3] YOLO 目标检测中...")
    detections = detect_objects(image_path)
    print(f"检测到 {len(detections)} 个物体")
    for det in detections:
        print(f"  - {det['class']}: {det['confidence']:.2%}")

    # 2. 可视化标注
    print("\n[2/3] 生成标注图片...")
    annotated_path = visualize_detections(image_path, detections, output_path)
    print(f"标注图片已保存: {annotated_path}")

    # 3. 调用通义千问
    print("\n[3/3] 通义千问分析中...")
    answer = ask_qwen(question, detections, image_name)
    print(f"\n回答:\n{answer}")

    return {
        "detections": detections,
        "answer": answer,
        "annotated_image_path": annotated_path,
        "detection_summary": format_detection_summary(detections),
    }


# 命令行交互模式
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("多模态智能问答系统 - 通义千问 + YOLO")
    print("=" * 60)
    print("\n使用方法:")
    print("  python multimodal_rag_qa.py <图片路径> <问题>")
    print("  或设置环境变量 DASHSCOPE_API_KEY 后运行")
    print("\n示例:")
    print('  python multimodal_rag_qa.py bus.jpg "这张图片里有什么车？"')
    print("=" * 60)

    if len(sys.argv) >= 3:
        image_path = sys.argv[1]
        question = sys.argv[2]

        # 检查 API Key
        if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your-api-key-here":
            print("\n错误: 请设置 DASHSCOPE_API_KEY 环境变量")
            print("或修改脚本中的 DASHSCOPE_API_KEY 默认值")
            sys.exit(1)

        result = process_image_question(image_path, question)

        print(f"\n{'=' * 50}")
        print("处理完成!")
        print(f"标注图片: {result['annotated_image_path']}")
        print("=" * 50)

    elif len(sys.argv) == 2:
        image_path = sys.argv[1]

        # 交互式问答
        if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your-api-key-here":
            print("\n错误: 请设置 DASHSCOPE_API_KEY 环境变量")
            print("或修改脚本中的 DASHSCOPE_API_KEY 默认值")
            sys.exit(1)

        # 先检测
        detections = detect_objects(image_path)
        print(f"\n检测到 {len(detections)} 个物体:")
        for det in detections[:10]:
            print(f"  - {det['class']}: {det['confidence']:.2%}")

        # 循环问答
        while True:
            question = input("\n请输入问题 (输入 q 退出): ")
            if question.lower() in ["q", "quit", "exit"]:
                break

            result = process_image_question(image_path, question)
            print(f"\n标注图片: {result['annotated_image_path']}")
    else:
        # 使用默认测试
        print("\n使用默认测试...")
        if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "your-api-key-here":
            print("错误: 请设置 DASHSCOPE_API_KEY 环境变量")
            sys.exit(1)

        test_image = "bus.jpg"
        if os.path.exists(test_image):
            result = process_image_question(test_image, "这张图片里有什么车？")
        else:
            print(f"测试图片 {test_image} 不存在")

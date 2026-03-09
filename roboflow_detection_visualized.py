from inference_sdk import InferenceHTTPClient
import cv2

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8QivlLrlv9942RhNqZu3"
)

image_path = "test_imag01.jpg"

result = client.run_workflow(
    workspace_name="demo01-7gzoy",
    workflow_id="find-cars",
    images={
        "image": "test_imag01.jpg"
    },
    use_cache=True
)

print(result)

img = cv2.imread(image_path)

for prediction in result['predictions']:
    x = int(prediction['x'])
    y = int(prediction['y'])
    w = int(prediction['width'])
    h = int(prediction['height'])

    # Roboflow 返回的是中心点坐标 (x,y)，需要转换成左上角坐标画框
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # 画框 (绿色)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 写字
    cv2.putText(img, prediction['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
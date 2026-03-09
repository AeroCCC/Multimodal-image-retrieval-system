from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8QivlLrlv9942RhNqZu3"
)

result = client.run_workflow(
    workspace_name="demo01-7gzoy",
    workflow_id="find-cars",
    images={
        "image": "test_imag01.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)

result.save("yolo_res01.jpg")

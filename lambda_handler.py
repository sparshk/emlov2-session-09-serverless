import json

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image

from typing import Dict

from utils import decode_base64_to_image, load_label_mapping, map_class_to_label


print("Loading scripted model...")
model = torch.jit.load("cifar-model.script.pt")
print("Scripted Model Loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()
predict_transforms = T.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
topk = 5

response_headers = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Credentials": True,
}

categories = load_label_mapping("index_to_name.json")

def inference(image: Image) -> Dict[str, int]:
    img_tensor = predict_transforms(image).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    print("Completed setting tensor to GPU...")
    with torch.no_grad():

        print("Inference call...")
        logits = model(img_tensor)
        print("Logits received...")
        preds = F.softmax(logits, dim=-1)

    return preds

def handle_request(event, context):
    print(f"Lambda function ARN: {context.invoked_function_arn}")
    print(f"Lambda funtion version: {context.function_version}")
    print(f"Lambda Request ID: {context.aws_request_id}")

    print(f"Got event", event)

    img_b64 = event["body"]

    try:
        image = decode_base64_to_image(img_b64)

        predictions = inference(image)

        probs, classes = torch.topk(predictions, topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()

        print(f"Lambda time remaining in MS: {context.get_remaining_time_in_millis()}")

        class_to_label = map_class_to_label(probs, categories, classes)

        return {
            "statusCode": 200,
            "headers": response_headers,
            "body": json.dumps(class_to_label),
        }

    except Exception as e:
        print(e)

        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({"message": "Failed to process image: {}".format(e)}),
        }
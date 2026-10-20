import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from Models.model import create_model


# Locate the Models directory
MODEL_DIR = Path(__file__).resolve().parent

# Define the paths to the saved files
MODEL_PATH = MODEL_DIR / "resnet18_neu_det.pth"
CLASS_MAPPING_PATH = MODEL_DIR / "class_mapping.json"


# Load the class mapping
with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as file:
    class_mapping = json.load(file)

# JSON keys are strings, so convert them to integers
class_mapping = {
    int(index): class_name
    for index, class_name in class_mapping.items()
}


# Create the model architecture
model = create_model(num_classes=len(class_mapping))

# Load the trained parameters
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)

# Put the model into evaluation mode
model.eval()


# Define the preprocessing pipeline
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_defect(image_path):
    """
    Predict the steel defect class in an image.

    Args:
        image_path: Path to the input image.

    Returns:
        Dictionary containing the predicted class and confidence.
    """

    # Convert the supplied path into a Path object
    image_path = Path(image_path)

    # Check that the image exists
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open the image and convert it to RGB
    with Image.open(image_path) as image:
        image = image.convert("RGB")

    # Apply the preprocessing pipeline
    image_tensor = inference_transform(image)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Generate predictions
    with torch.no_grad():
        outputs = model(image_tensor)

        # Convert logits into probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Select the highest-probability class
        confidence, predicted_index = probabilities.max(dim=1)

    # Convert the predicted index to its class name
    predicted_class = class_mapping[predicted_index.item()]

    # Convert confidence to a percentage
    confidence_score = confidence.item() * 100

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence_score, 2)
    }

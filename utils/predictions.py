"""predictions.py
This module contains a function to predict on a target image using a trained PyTorch model.
"""
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: Optional[transforms.Compose] = None,
    device: torch.device = device,
    show: bool = True,
) -> Tuple[str, float]:
    """Predicts on a target image with a target model and plots the result.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        class_names (List[str]): List of class names.
        image_path (str): Path to the image.
        image_size (Tuple[int, int], optional): Image resize. Defaults to (224, 224).
        transform (transforms.Compose, optional): Custom transform. Defaults to None.
        device (torch.device, optional): Device to use. Defaults to device.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        Tuple[str, float]: Predicted class name and probability.
    """
    img = Image.open(image_path).convert("RGB")

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(0).to(device)
        preds = model(transformed_image)
        probs = torch.softmax(preds, dim=1)
        pred_prob, pred_label = torch.max(probs, dim=1)
        pred_class = class_names[pred_label.item()]
        prob = pred_prob.item()

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {pred_class} | Prob: {prob:.3f}")
    plt.axis("off")
    if show:
        plt.show()
    plt.close()

    return pred_class, prob
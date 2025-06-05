import argparse
from pathlib import Path
import torch
from torch import nn, optim

from src import model as model_lib
from src import engine
from src.download_zip_dataset import download_dataset
from utils import save_load
from src import data_loader

def ensure_dataset_exists(dataset_path: Path, data_dir: Path):
    if not dataset_path.exists():
        print(f"[INFO] Downloading dataset to {dataset_path}...")
        download_dataset(
            url_dataset="https://hessenbox.tu-darmstadt.de/dl/fi8G8VfXgXPa8zEW8asqDL7D/project_capsule_dataset.dir",
            dataset_name="healthy_bleeding_capsule_dataset",
            data_dir=str(data_dir),
            delete_after=True
        )
    else:
        print(f"[INFO] Dataset already exists at {dataset_path}, skipping download.")

def get_model(model_name: str):
    models = {
        "vgg16": model_lib.VGG16,
        "vgg19": model_lib.VGG19,
        "resnet50": model_lib.ResNet50,
        "inceptionv3": model_lib.InceptionV3,
        "mobilenetv2": model_lib.MobileNetV2,
        "efficientnet": model_lib.EfficientNet,
        "vitam": model_lib.ViTAM,
    }
    try:
        return models[model_name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}")

def main(args):
    """Main training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(args.data_dir) / "healthy_bleeding_capsule_dataset"
    ensure_dataset_exists(dataset_path, Path(args.data_dir))

    train_transform, test_transform = data_loader.get_data_transforms(
        image_size=(224, 224),
        augment=args.augment,
        grayscale=False
    )

    train_loader, test_loader, class_names = data_loader.create_dataloaders(
        data_dir=str(dataset_path),
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=args.batch_size
    )

    model = get_model(args.model).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        logs_dir=args.logs_dir,
        save_dir=args.save_dir,
        model_name=args.model_name,
        save_best=True
    )
    print(f"[INFO] Training complete. Results saved to {args.logs_dir}/{args.model_name}_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a deep learning model for bleeding detection on capsule endoscopy images.")
    parser.add_argument("--model", type=str, required=True, help="Model name: vgg16, vgg19, resnet50, inceptionv3, mobilenetv2, efficientnet, vitam")
    parser.add_argument("--data_dir", type=str, default="data/raw/", help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--logs_dir", type=str, default="output/logs/", help="Directory to save logs")
    parser.add_argument("--save_dir", type=str, default="output/models/", help="Directory to save models")
    parser.add_argument("--model_name", type=str, default="model", help="Base name for saved model and logs")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation for training")
    args = parser.parse_args()
    main(args)
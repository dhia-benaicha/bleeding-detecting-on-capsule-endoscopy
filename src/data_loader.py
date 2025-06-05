"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

NUM_WORKERS = os.cpu_count()

def get_data_transforms(
  image_size=(224, 224), 
  augment=True, 
  grayscale=False, 
  center_crop=None, 
  additional_transforms=None
):
  """
  Create train and test transforms with optional grayscale, center crop, and extra transforms.
  Args:
    image_size: Tuple of (height, width) for resizing images.
    augment: Boolean indicating whether to apply data augmentation.
    grayscale: Boolean to convert images to grayscale.
    center_crop: Optional int or tuple for center cropping.
    additional_transforms: Optional list of extra transforms to append.
  Returns:
    train_transform: Transformations for training data.
    test_transform: Transformations for testing data.
  """
  base_transforms = []
  if grayscale:
    base_transforms.append(transforms.Grayscale(num_output_channels=3))
  if center_crop:
    base_transforms.append(transforms.CenterCrop(center_crop))
  base_transforms.append(transforms.Resize(image_size))

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])

  if augment:
    train_transform = transforms.Compose(
      base_transforms +
      [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
      ] +
      (additional_transforms if additional_transforms else []) +
      [
        transforms.ToTensor(),
        normalize
      ]
    )
  else:
    train_transform = transforms.Compose(
      base_transforms +
      (additional_transforms if additional_transforms else []) +
      [
        transforms.ToTensor(),
        normalize
      ]
    )

  test_transform = transforms.Compose(
    base_transforms +
    (additional_transforms if additional_transforms else []) +
    [
      transforms.ToTensor(),
      normalize
    ]
  )

  return train_transform, test_transform


def create_dataloaders(
  data_dir,
  train_transform,
  test_transform,
  batch_size=32,
  num_workers=NUM_WORKERS,
  shuffle_train=True,
):
  """
  Create DataLoaders for training and testing datasets from a single directory.
  Args:
    train_dir: Directory with images (should contain 'bleeding' and 'healthy' subfolders).
    train_transform: Transformations for training data.
    test_transform: Transformations for testing data.
    batch_size: Batch size for DataLoaders.
    num_workers: Number of workers for DataLoader.
    shuffle_train: Whether to shuffle training data.

  Returns:
    train_loader, test_loader, class_names
  """

  full_data = datasets.ImageFolder(data_dir, transform=None)
  class_names = full_data.classes

  # Split into train/test (80/20 split)
  total_size = len(full_data)
  test_size = int(0.2 * total_size)
  train_size = total_size - test_size
  train_subset, test_subset = random_split(full_data, [train_size, test_size])

  # Apply transforms
  train_subset.dataset.transform = train_transform
  test_subset.dataset.transform = test_transform

  train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=shuffle_train,
    num_workers=num_workers,
  )
  test_loader = DataLoader(
    test_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
  )
  return train_loader, test_loader, class_names

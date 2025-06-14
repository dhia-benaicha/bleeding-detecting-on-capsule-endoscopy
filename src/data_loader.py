import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset

class ImageDataLoader(Dataset):
  def __init__(
    self,
    data_dir,
    image_size=(224, 224),
    grayscale=False,
    center_crop=None,
    additional_transforms=None,
    batch_size=32,
    num_workers=None,
    pin_memory=False,
    shuffle_train=True,
    test_split=0.2,
    invert_classes=False,
  ):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers if num_workers is not None else os.cpu_count()
    self.pin_memory = pin_memory
    self.shuffle_train = shuffle_train
    self.test_split = test_split
    self.image_size = image_size
    self.grayscale = grayscale
    self.center_crop = center_crop
    self.additional_transforms = additional_transforms
    self.invert_classes = invert_classes

    self.train_transform = self.get_data_transforms(augment=False)
    self.test_transform = self.get_data_transforms(augment=False)

    self.train_loader, self.test_loader, self.class_names, self.class_to_idx = self.create_dataloaders()

  def get_data_transforms(self, augment=False):
    
    base_transforms = []

    if self.grayscale:
      base_transforms.append(transforms.Grayscale(num_output_channels=3))
    if self.center_crop:
      base_transforms.append(transforms.CenterCrop(self.center_crop))
    
    base_transforms.append(transforms.Resize(self.image_size))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

    aug_transforms = []
    if augment:
      aug_transforms = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
      ]

    transform = transforms.Compose(
      base_transforms +
      aug_transforms +
      (self.additional_transforms if self.additional_transforms else []) +
      [
        transforms.ToTensor(),
        normalize
      ]
    )
    return transform

  def create_dataloaders(self):
    full_data = datasets.ImageFolder(self.data_dir, transform=None)
    class_names = full_data.classes
    class_to_idx = full_data.class_to_idx.copy()

    # Invert class_to_idx if requested
    if self.invert_classes:
      # Reverse the order of class names and reassign indices
      class_names = list(reversed(class_names))
      full_data.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
      # Also update samples with new indices
      full_data.samples = [
        (path, full_data.class_to_idx[cls])
        for (path, _), cls in zip(full_data.samples, [os.path.basename(os.path.dirname(p)) for p, _ in full_data.samples])
      ]
      full_data.targets = [s[1] for s in full_data.samples]
      class_to_idx = full_data.class_to_idx.copy()

    total_size = len(full_data)
    test_size = int(self.test_split * total_size)
    train_size = total_size - test_size
    train_subset, test_subset = random_split(full_data, [train_size, test_size])

    train_subset.dataset.transform = self.train_transform
    test_subset.dataset.transform = self.test_transform

    train_loader = DataLoader(
      train_subset,
      batch_size=self.batch_size,
      shuffle=self.shuffle_train,
      num_workers=self.num_workers,
      pin_memory=self.pin_memory
    )
    test_loader = DataLoader(
      test_subset,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=self.pin_memory
    )
    return train_loader, test_loader, class_names, class_to_idx

  def apply_augmentation(self):
    self.train_transform = self.get_data_transforms(augment=True)
    self.train_loader.dataset.dataset.transform = self.train_transform

  def __len__(self):
    full_data = datasets.ImageFolder(self.data_dir, transform=None)
    return len(full_data)

  def __getitem__(self, idx):
    full_data = datasets.ImageFolder(self.data_dir, transform=self.test_transform)
    return full_data[idx]

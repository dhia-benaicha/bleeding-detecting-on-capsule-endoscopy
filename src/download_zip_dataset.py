"""Download and unzip a dataset from a given URL."""
import os
import requests
import zipfile
from pathlib import Path


def download_file(url: str, dest_path: Path):
    """Download a file from a URL to a specified destination path.
    Args:
        url (str): The URL of the file to download.
        dest_path (Path): The path where the downloaded file will be saved.
    Example usage:
        download_file("https://example.com/file.zip", Path("file.zip"))
    """
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)

def unzip_file(zip_path: Path, extract_to: Path):
    """Unzip a zip file to a specified directory.
    Args:
        zip_path (Path): The path to the zip file.
        extract_to (Path): The directory where the contents will be extracted.
    Example usage:
        unzip_file(Path("file.zip"), Path("extracted/"))
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def download_dataset(url_dataset: str, dataset_name: str, data_dir: str = "data/raw/",  delete_after: bool = True):
    """Download and unzip a dataset from a given URL.
    Args:
        url_dataset (str): The URL of the dataset zip file.
        dataset_name (str): The name of the dataset to be used as the directory name.
        data_dir (str): The base directory where the dataset will be stored.
        delete_after (bool): Whether to delete the zip file after extraction.
    Example usage:
        download_dataset(
            "https://example.com/dataset.zip",
            "my_dataset",
            data_dir="dataset/raw/",
            delete_after=True
        )
    """
    data_path = Path(data_dir)
    image_path = data_path / dataset_name
    zip_filename = f"{dataset_name}.zip"
    zip_path = data_path / zip_filename

    if image_path.is_dir():
        print(f"{image_path} directory exists.")
        return

    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name}...")
    download_file(url_dataset, zip_path)

    print(f"Unzipping {zip_filename}...")
    unzip_file(zip_path, image_path)

    if delete_after:
        print(f"Removing zip file: {zip_path}")
        os.remove(zip_path)

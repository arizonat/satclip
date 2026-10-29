import pandas as pd
import os
import torchvision.transforms as T
import rasterio

import numpy as np
import timm
import torch
import torchgeo.models
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights
from tqdm import tqdm
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform):
        self.paths = [os.path.join(root_dir, "images", fn) for fn in df["fn"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return self.transform(load_image(path)), idx

def load_image(image_path):
    """Load an image from disk and return it as a numpy array."""
    try:
        with rasterio.open(image_path) as f:
            data = f.read().astype(np.float32)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    return data

def load_encoder(encoder_model_name, satellite_name):
    """Load the specified encoder model."""
    if satellite_name == "landsat":
        if encoder_model_name == "moco_vit16":
            # Load the MoCo ViT16 model
            print('using pretrained moco vit16 landsat')
            weights = ViTSmall16_Weights.LANDSAT_TM_TOA_MOCO
            in_chans = weights.meta["in_chans"]
            visual = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=embed_dim)
            visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            visual.requires_grad_(False)
            visual.head.requires_grad_(True)
            visual.eval()
            return visual
    elif satellite_name == "sentinel2":
        if encoder_model_name == "moco_vit16":
            # Load the MoCo ViT16 model
            print('using pretrained moco vit16 sentinel2')
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            visual = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=embed_dim)
            visual.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            visual.requires_grad_(False)
            visual.head.requires_grad_(True)
            visual.eval()
            return visual
    else:
        raise ValueError(f"Unsupported encoder model or satellite: {encoder_model_name} {satellite_name}")

def get_transform(encoder_model_name, satellite_name):
    """Get the appropriate transform for the specified encoder model."""
    if encoder_model_name == "moco_vit16" and satellite_name == "landsat":
        # Define the transform for MoCo ViT16
        def transform(image):
            # B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
            # image = np.concatenate([image[:10], B10, image[10:]], axis=0)
            image = image[:7] # get the first 7 bands (B1-B7) for the MoCo ViT16 model
            image = torch.tensor(image)

            augmentation = T.Compose([
                T.CenterCrop((crop_size, crop_size)),
            ])
            return augmentation(image)
    elif encoder_model_name == "moco_vit16" and satellite_name == "sentinel2":
        # Define the transform for MoCo ViT16
        def transform(image):
            # B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
            # image = np.concatenate([image[:10], B10, image[10:]], axis=0)
            B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
            image = np.concatenate([image[:10], B10, image[10:]], axis=0)
            image = torch.tensor(image)
            
            augmentation = T.Compose([
                T.CenterCrop((crop_size, crop_size)),
            ])
            return augmentation(image)
    else:
        raise ValueError(f"Unsupported encoder model: {encoder_model_name}")
    return transform

if __name__ == "__main__":

    satellite = sys.argv[1] # landsat or sentinel2
    root_dir = sys.argv[2]

    num_workers = 1

    dataset_csv = "index.csv"  # Path to the CSV file containing the dataset
    encoder_model_name = "moco_vit16"
    crop_size = 224  # Input image size for the encoder model
    embed_dim = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(root_dir, dataset_csv))

    # Load the encoder and dataset

    encoder = load_encoder(encoder_model_name, satellite).to(device).eval()
    transform = get_transform(encoder_model_name, satellite)

    df = pd.read_csv(os.path.join(root_dir, dataset_csv))

    image_paths = []
    embeddings = []
    batch_size = 32

    loader = DataLoader(ImageDataset(df, root_dir, transform),
            batch_size=batch_size,
            shuffle=False,              # keep order so paths line up
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),)

    out_path = os.path.join(root_dir, f"{encoder_model_name}_{satellite}_embeddings.parquet")
    writer = None

    try:
        with torch.inference_mode():  # no autograd graph -> far less GPU memory
            for images, idxs in tqdm(loader, desc="Computing embeddings"):
                images = images.to(device, non_blocking=True)
                emb = encoder(images).float().cpu().numpy()   # (B, embed_dim)

                batch_paths = [loader.dataset.paths[i] for i in idxs.tolist()]
                table = pa.table({
                    "image_path": batch_paths,
                    "embedding": pa.FixedSizeListArray.from_arrays(
                        pa.array(emb.ravel(), type=pa.float32()), emb.shape[1]
                    ),
                })

                # Stream each batch to disk instead of holding everything in RAM
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    # # Iterate through the dataset and compute embeddings for each image
    # # Modify to use batch processing for efficiency
    # for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Computing embeddings"):

    #     image_path = os.path.join(root_dir, "images", row['fn'])
    #     image = transform(load_image(image_path)).unsqueeze(0).to(device)  # Add batch dimension
    #     embedding = encoder(image)

    #     embeddings.append(embedding.cpu().detach().numpy())
    #     image_paths.append(image_path)

    # # Save embeddings to disk as a parquet file
    # embeddings_array = np.vstack(embeddings)
    # df_embeddings = pd.DataFrame({'image_path': image_paths, 'embedding': embeddings_array.tolist()})
    # df_embeddings.to_parquet(os.path.join(root_dir, f"{encoder_model_name}_landsat_embeddings.parquet"), compression='snappy', index=False)

import argparse
import io
import os
import time
import warnings

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray  # rioxarray is required for the .rio methods in xarray despite what mypy, ruff, etc. says :)
import stackstac
from tqdm import tqdm
import dask

def set_up_parser() -> argparse.ArgumentParser:
    """
    Set up and return a command-line argument parser for the Landsat 4/5 patch downloader.

    The parser defines required and optional arguments for specifying the patch download range,
    Azure blob storage configuration, and the source GeoParquet file used to sample Landsat 4/5 items.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all necessary CLI options.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--low",
        default=0,
        type=int,
        required=True,
        help="Starting index",
    )

    parser.add_argument(
        "--batch_size",
        default=500,
        type=int,
        required=True,
        help="Number of patches per query",
    )

    parser.add_argument(
        "--high",
        default=100_000,
        type=int,
        required=True,
        help="Ending index",
    )

    parser.add_argument(
        "--output_fn",
        default="patch_locations.csv",
        type=str,
        required=True,
        help="Output filename",
    )

    parser.add_argument(
        "--img_output_dir",
        default="images",
        type=str,
        required=True,
        help="Directory to save downloaded image patches",
    )

    parser.add_argument(
        "--ls_parquet_fn",
        type=str,
        required=True,
        help="GeoParquet index file to sample from",
    )

    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        required=False,
        help="Number of parallel workers for downloading and processing patches",
    )

    return parser


def main(args):
    """
    Main processing function for downloading Landsat 4/5 image patches from a STAC catalog
    and uploading them to an Azure Blob container as Cloud-Optimized GeoTIFFs (COGs).

    The function selects valid image patches from the input GeoParquet file,
    extracts a 256x256 region from a Landsat 4/5 STAC item, filters based on NaN content,
    and embeds relevant metadata before uploading the patch to Azure Blob Storage.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, including input range, output file, Azure credentials,
        and STAC sampling source.
    """
    # Sanity checks: output file shouldn't already exist, input parquet must exist
    assert not os.path.exists(args.output_fn)
    assert os.path.exists(args.ls_parquet_fn)

    os.makedirs(args.img_output_dir, exist_ok=True)

    # Connect to Microsoft Planetary Computer STAC API
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1/",
        modifier=planetary_computer.sign_inplace,
    )

    collection = catalog.get_collection("landsat-c2-l2")

    # Load input patch candidates from parquet
    df = pd.read_parquet(args.ls_parquet_fn)
    num_rows = df.shape[0]

    # Initialize stats and result tracking
    num_retries = 0
    num_error_hits = 0
    num_empty_hits = 0
    num_samples = args.high - args.low
    # progress_bar = tqdm(num_samples)
    results = []

    # Begin sampling loop
    idx = args.low

    sample_idxs = np.random.choice(num_rows, size=num_samples, replace=False)

    batch_idxs = np.array_split(sample_idxs, np.ceil(num_samples / args.batch_size))

    for batch_idx in tqdm(batch_idxs):
        batch_ids = df.iloc[batch_idx]["id"].values

        for j in range(5):
            try:
                # Download metadata for batch of items
                # Does NOT maintain order of IDs!
                items = catalog.search(
                    collections=["landsat-c2-l2"],
                    ids=batch_ids,
                ).item_collection()
                
                # Download raster data for items
                stacks = []
                for item in items:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        stack = stackstac.stack(
                            item,
                            assets=[
                                "blue",
                                "green",
                                "red",
                                "nir08",
                                "swir16",
                                "lwir",
                                "swir22",
                                "qa_pixel",
                                "qa_radsat",
                                "cloud_qa",
                                "qa"
                            ],
                            epsg=32612,
                        )
                    stacks.append(stack)
                xs = [np.random.randint(0, width - 256) for width in [stack.shape[3] for stack in stacks]]
                ys = [np.random.randint(0, height - 256) for height in [stack.shape[2] for stack in stacks]]
                stacks_sampled = [stack[0, :, y : y + 256, x : x + 256] for stack, x, y in zip(stacks, xs, ys)]
                patches = dask.compute(*stacks_sampled, scheduler="threads", num_workers=args.num_workers)
                break

            except Exception as e:
                print(e)
                print("retrying batch", j)
                num_retries += 1
                time.sleep(2**(j+1))

        if items is None:
            print(f"failed to get items")
            num_error_hits += len(batch_ids)
            continue

        for i, patch in enumerate(patches):
            # Filter patches with more than 10% missing data (this should already be handled by parquet)
            item = items[i]
            x_coord = xs[i]
            y_coord = ys[i]
            num_channels = patch.shape[0]
            percent_empty = np.mean((np.isnan(patch.data)).sum(axis=0) == num_channels)
            percent_zero = np.mean((patch.data == 0).sum(axis=0) == num_channels)

            if percent_empty > 0.1 or percent_zero > 0.1:
                num_empty_hits += 1
                continue

            # Save valid patch to Azure Blob Storage as GeoTIFF with metadata
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                patch = patch.astype(np.uint16)

            # Extract STAC metadata for provenance and traceability
            metadata = {
                "id": item.id, 
                "datetime": item.datetime.isoformat(),
                "platform": item.properties.get("platform", ""),
                "proj:code": item.properties.get("proj:code", ""),
                "instruments": str(item.properties.get("instruments", "")),
                "sci:doi": item.properties.get("sci:doi", ""),
                "landsat:scene_id": item.properties.get("landsat:scene_id", ""),
                "cloud_cover": str(item.properties.get("eo:cloud_cover", "")),
                "landsat:correction": str(item.properties.get("landsat:correction", "")),
                "view:sun_elevation": str(item.properties.get("view:sun_elevation", "")),
                "view:sun_azimuth": str(item.properties.get("view:sun_azimuth", "")),
                "x_coord_sampled": str(x_coord),
                "y_coord_sampled": str(y_coord),
            }

            # Attach metadata to the patch for inclusion in the raster tags
            patch.attrs.update(metadata)

            containing_dir = f"{args.img_output_dir}/images/{item.datetime.year}"
            os.makedirs(containing_dir, exist_ok=True)

            # Write GeoTIFF to disk
            patch.rio.to_raster(
                f"{containing_dir}/patch_{item.id}.tif",
                driver="GTiff",
                dtype=np.uint16,
                compress="LZW",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                interleave="pixel",
            )

            # Store patch info for CSV log
            results.append(
                (
                    idx,
                    xs[i],
                    ys[i],
                    metadata["datetime"],
                    metadata["id"],
                    metadata["landsat:scene_id"],
                )
            )

            idx += 1
            # progress_bar.update(1)

        # progress_bar.close()

    # Save all patch locations and sample info to CSV
    df = pd.DataFrame(results, columns=["idx", "x", "y", "datetime", "id", "landsat:scene_id"])
    df.to_csv(args.output_fn)

    # Print final stats
    print("Summary:")
    print(f"range: [{args.low}, {args.high})")
    print(f"num hits: {len(results)}")
    print(f"num empty hits: {num_empty_hits}")
    print(f"num error hits: {num_error_hits}")
    print(f"num retries: {num_retries}")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)

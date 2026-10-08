import rasterio
import fiona
from tqdm import tqdm
import fiona.transform
import pandas as pd
import shapely.geometry
import glob
import pathlib
import sys
from pathlib import Path

def main(root_dir, satellite_name="sentinel2"):
    """
    satellite_name: str, either "sentinel2" or "landsat"
    """
    
    root_dir = Path(root_dir)
    urls = Path(f"{root_dir}").rglob("*.tif")

    lats = []
    lons = []
    ts = []
    ids = []
    fns = []

    for url in tqdm(urls):
        with rasterio.open(url) as src:
            geom = shapely.geometry.mapping(shapely.geometry.box(*src.bounds))
            warped_geom = fiona.transform.transform_geom(src.crs, "EPSG:4326", geom)
            shape = shapely.geometry.shape(warped_geom)
            x, y = shape.centroid.xy
            x = x[0]
            y = y[0]
            filename = pathlib.Path(url).relative_to(root_dir/"images")

            timestamp = src.tags().get("datetime")

            if satellite_name == "landsat":
                id = src.tags().get("id")
            elif satellite_name == "sentinel2":
                id = src.tags().get("granule_id")
            else:
                raise ValueError(f"Unknown satellite name: {satellite_name}")

            fns.append(filename)
            ids.append(id)
            ts.append(timestamp)
            lats.append(y)
            lons.append(x)

    df = pd.DataFrame({
        "fn": fns,
        "id": ids,
        "lat": lats,
        "lon": lons,
        "ts": ts,
    })
    df.to_csv(f"{root_dir}/index.csv", index=False)


if __name__ == '__main__':
    ROOT = sys.argv[1]
    SATELLITE_NAME = sys.argv[2]
    main(ROOT, satellite_name=SATELLITE_NAME)

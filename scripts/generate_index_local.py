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

def main(root_dir):
    
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
            granule_id = src.tags().get("granule_id")

            fns.append(filename)
            ids.append(granule_id)
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
    main(ROOT)

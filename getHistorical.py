import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
import ee
import requests
from PIL import Image


def load_local_points(csv_path: Path):
    """Read a CSV with id/lon/lat columns and emit Feature-like dicts."""
    required_columns = {'id', 'lon', 'lat'}
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open(newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row with 'id', 'lon', and 'lat'.")
        missing = required_columns - set(map(str.lower, reader.fieldnames))
        if missing:
            raise ValueError(f"CSV header missing required columns: {', '.join(sorted(missing))}")
        features = []
        for idx, row in enumerate(reader, start=1):
            try:
                lon = float(row.get('lon') or row.get('Lon') or row.get('LON'))
                lat = float(row.get('lat') or row.get('Lat') or row.get('LAT'))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid coordinates on row {idx}: {row}") from None
            point_id = (row.get('id') or row.get('ID') or f"point_{idx}").strip()
            features.append({
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]
                },
                'properties': {
                    'id': point_id
                }
            })
    if not features:
        raise ValueError("No point records found in CSV.")
    return features

def maskS2(image):
    """Mask Sentinel-2 surface reflectance pixels using the SCL band."""
    scl = image.select('SCL')
    return image.updateMask(
        scl.neq(3)  # cloud shadow
        .And(scl.neq(8))  # cloud
        .And(scl.neq(9))  # high cloud
        .And(scl.neq(10))  # cirrus
    )


def mask_landsat(image):
    """Mask Landsat Collection 2 pixels with QA_PIXEL bit flags."""
    qa = image.select('QA_PIXEL')
    shadow = 1 << 4
    cloud = 1 << 5
    cirrus = 1 << 9
    mask = (qa.bitwiseAnd(shadow).eq(0)
            .And(qa.bitwiseAnd(cloud).eq(0))
            .And(qa.bitwiseAnd(cirrus).eq(0)))
    return image.updateMask(mask)


# Sensor-specific configuration used to drive filtering/visualization.
DATASETS = {
    'sentinel': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands': ['B4', 'B3', 'B2'],
        'scale': 10,
        'cloud_property': 'CLOUDY_PIXEL_PERCENTAGE',
        #'mask_func': maskS2,
        'mask_func': None,
        'vis_min': 0,
        'vis_max': 3000,
    },
    'landsat8': {
        'collection': 'LANDSAT/LC08/C02/T1_L2',
        'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
        'scale': 30,
        'cloud_property': 'CLOUD_COVER',
        # 'mask_func': mask_landsat,
        'mask_func': None,
        'vis_min': 7000,
        'vis_max': 16000,
    },
    'landsat': {  # Alias for Landsat 8 to keep previous CLI stable
        'collection': 'LANDSAT/LC08/C02/T1_L2',
        'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
        'scale': 30,
        'cloud_property': 'CLOUD_COVER',
        #'mask_func': mask_landsat,
        'mask_func': None,
        'vis_min': 7000,
        'vis_max': 16000,
    },
    'landsat7': {
        'collection': 'LANDSAT/LE07/C02/T1_L2',
        'bands': ['SR_B3', 'SR_B2', 'SR_B1'],
        'scale': 30,
        'cloud_property': 'CLOUD_COVER',
        # 'mask_func': mask_landsat,
        'mask_func': None,
        'vis_min': 7000,
        'vis_max': 16000,
    },
    'landsat5': {
        'collection': 'LANDSAT/LT05/C02/T1_L2',
        'bands': ['SR_B3', 'SR_B2', 'SR_B1'],
        'scale': 30,
        'cloud_property': 'CLOUD_COVER',
        # 'mask_func': mask_landsat,
        'mask_func': None,
        'vis_min': 7000,
        'vis_max': 16000,
    },
}


def generate_periods(start_date: date, end_date: date, interval: str):
    """Build continuous (start, end) windows for the requested cadence."""
    if interval not in {'week', 'month', 'year'}:
        raise ValueError("Interval must be one of: week, month, year")

    if end_date < start_date:
        raise ValueError("End date must be on or after start date")

    periods = []
    current = start_date
    final = end_date

    while current <= final:
        if interval == 'week':
            next_start = current + timedelta(days=7)
        elif interval == 'month':
            if current.month == 12:
                next_start = date(current.year + 1, 1, 1)
            else:
                next_start = date(current.year, current.month + 1, 1)
        else:  # year
            next_start = date(current.year + 1, 1, 1)

        period_end = min(next_start, final + timedelta(days=1))
        periods.append((current, period_end))
        current = next_start

    return periods


def parse_args():
    """Define and parse the CLI surface for the downloader."""
    parser = argparse.ArgumentParser(description="Download square Sentinel/Landsat tiles from GEE")
    parser.add_argument('--auth', action='store_true', help='Authenticate with Earth Engine and exit')
    parser.add_argument('--project', default='dae-2026', help='Google Earth Engine project ID to initialize/authenticate')
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='sentinel', help='Imagery collection to use')
    parser.add_argument('--interval', choices=['week', 'month', 'year'], default='month', help='Grouping interval for composites')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD) inclusive')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD) inclusive')
    parser.add_argument('--output', default='output', help='Directory to store downloaded rasters')
    parser.add_argument('--input-csv', default='coordinates.csv', help='CSV containing id, lon, lat columns')
    parser.add_argument('--cloud-threshold', type=float, default=35.0, help='Max cloud percentage allowed per scene')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile dimension in pixels (square)')
    parser.add_argument('--pixel-scale', type=float, default=None, help='Override pixel scale (meters)')
    args = parser.parse_args()

    if not args.auth:
        if not args.start_date or not args.end_date:
            parser.error('--start-date and --end-date are required unless --auth is specified')

    return args


def main():
    args = parse_args()

    if args.auth:
        print("Starting Earth Engine authentication flow...")
        ee.Authenticate()
        print("Authentication completed. You can now run downloads.")
        return

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)

    dataset_cfg = DATASETS[args.dataset]
    pixel_scale = args.pixel_scale or dataset_cfg['scale']
    tile_size = args.tile_size
    half_tile_meters = (tile_size * pixel_scale) / 2

    ee.Initialize(project=args.project)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    points_csv = Path(args.input_csv)
    features = load_local_points(points_csv)
    periods = generate_periods(start_date, end_date, args.interval)

    print(f"Found {len(features)} points")
    print(f"Downloading to: {output_dir.absolute()}")
    print(f"Dataset: {args.dataset} | Interval: {args.interval}")
    print("Starting downloads...")

    download_count = 0

    for f in features:
        # Build a square AOI around each input point once, reuse for every period.
        geom = ee.Geometry(f['geometry']).centroid()
        region_geom = geom.buffer(half_tile_meters).bounds()
        region_geojson = region_geom.getInfo()
        point_id = f['properties']['id']

        # Iterate over the requested temporal slices (week/month/year).
        for period_start, period_end in periods:
            start = ee.Date.fromYMD(period_start.year, period_start.month, period_start.day)
            end = ee.Date.fromYMD(period_end.year, period_end.month, period_end.day)
            label_start = period_start.strftime('%Y%m%d')
            label_end = (period_end - timedelta(days=1)).strftime('%Y%m%d')
            period_label = f"{label_start}_{label_end}"

            collection = (ee.ImageCollection(dataset_cfg['collection'])
                          .filterBounds(geom)
                          .filterDate(start, end)
                          .filter(ee.Filter.lte(dataset_cfg['cloud_property'], args.cloud_threshold)))

            mask_fn = dataset_cfg.get('mask_func')
            if mask_fn is not None:
                collection = collection.map(mask_fn)

            if collection.size().getInfo() == 0:
                print(f"No imagery for {point_id} {period_label} under cloud threshold {args.cloud_threshold}%")
                continue

            # Pick the clearest scene via the collection-specific cloud property.
            best_scene = ee.Image(collection.sort(dataset_cfg['cloud_property']).first())
            vis = best_scene.visualize(
                bands=dataset_cfg['bands'],
                min=dataset_cfg['vis_min'],
                max=dataset_cfg['vis_max']
            )

            filename = f"{point_id}_{args.dataset}_{period_label}.tif"
            filepath = output_dir / filename

            if filepath.exists():
                print(f"Skipping {filename} (already exists)")
                continue

            try:
                url = vis.clip(region_geom).reproject(
                    ee.Projection('EPSG:3857').atScale(pixel_scale)
                ).getDownloadURL({
                    'dimensions': f"{tile_size}x{tile_size}",
                    'region': region_geojson,
                    'format': 'GEO_TIFF',
                    'crs': 'EPSG:3857',
                    'maxPixels': 1e8
                })

                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)

                png_path = filepath.with_suffix('.png')
                try:
                    with Image.open(filepath) as img:
                        img.save(png_path, format='PNG')
                    filepath.unlink()
                    print(f"Downloaded {download_count + 1}: {png_path.name}")
                except Exception as convert_err:
                    print(f"Saved original TIFF due to conversion error: {convert_err}")
                    print(f"Downloaded {download_count + 1}: {filename}")

                download_count += 1

            except Exception as err:
                print(f"Error downloading {filename}: {err}")

    print(f"\nTotal files downloaded: {download_count}")
    print(f"Saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

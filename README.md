# Google Earth Engine Tile Downloader

Python utility for downloading square Sentinel-2 or Landsat tiles around a list of coordinates using Google Earth Engine. The script can authenticate, fetch the clearest scene per interval, and exports PNG snapshots ready for downstream tooling.

## Requirements
- Python 3.10+
- Google Earth Engine Python API (`earthengine-api`)
- `requests`
- `Pillow`

Install dependencies with:

```
pip install earthengine-api requests pillow
```

## Authentication
Run the script once with the `--auth` flag to link your Google account and store local credentials:

```
python gee.py --auth --project YOUR_PROJECT_ID
```

A browser window opens for consent; afterward the credentials are cached and subsequent runs can initialize directly.

## Coordinate CSV format
`coordinates.csv` must contain at least the columns:

```
id,lon,lat
p001,10.123,45.456
p002,10.200,45.500
```

Additional columns are ignored.

## Download usage
Key arguments exposed by `gee.py`:

| Flag | Description |
| --- | --- |
| `--project` | Earth Engine project to initialize/authenticate |
| `--dataset` | `sentinel`, `landsat`/`landsat8` (2013+), `landsat7` (1999-2021), `landsat5` (1984-2013) |
| `--interval` | `week`, `month`, or `year` grouping |
| `--start-date`, `--end-date` | Inclusive date range (YYYY-MM-DD) |
| `--input-csv` | Path to the coordinate CSV |
| `--output` | Folder for GeoTIFF/PNG exports |
| `--cloud-threshold` | Maximum global cloud percentage per scene |
| `--tile-size` | Width/height in pixels (square) |
| `--pixel-scale` | Optional meter-per-pixel override |

Example command:

```
python getHistorical.py \
  --project dae-2026 \
  --dataset sentinel \
  --interval month \
  --start-date 2019-01-01 \
  --end-date 2020-12-31 \
  --input-csv coordinates.csv \
  --output output_tiles
```

The script builds square regions around each coordinate, picks the clearest scene per interval, downloads GeoTIFFs, and converts them to PNGs. Existing files are skipped to avoid re-downloading.


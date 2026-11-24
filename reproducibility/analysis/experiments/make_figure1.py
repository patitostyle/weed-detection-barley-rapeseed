"""Publication location map from the original field vectors and Natural Earth.

The field shapes are measured source geometries, not traced from a screenshot.
Only the regional locator uses the generalized Natural Earth country dataset.
Writes local figure assets; does not modify the author's manuscript or raw data.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

if os.name == "nt":
    native = Path(sys.prefix) / "Library" / "bin"
    if native.is_dir():
        os.environ["PATH"] = str(native) + os.pathsep + os.environ.get("PATH", "")

from osgeo import ogr, osr
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
MAPDATA = HERE / "cartography"
SOURCE = Path("C:/datos tfm/areas")
COUNTRIES = MAPDATA / "ne_110m_admin_0_countries.geojson"
COUNTRY_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
FIELDS = [("cebada_divi", "Barley 1", "Source training and validation", "#78AED0", 11.83),
          ("cebada2", "Barley 2", "Retrospective evaluation", "#E9BB64", 15.93),
          ("colza", "Rapeseed", "Cross-crop transfer and adaptation", "#74BBA4", 8.39)]
INK = "#243746"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(epsg):
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(epsg)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return sr


def patch_from_geometry(geometry, transform=lambda xy: xy, **kwargs):
    """Create a compound Matplotlib path while preserving polygon rings."""
    vertices = []
    codes = []

    def add(g):
        name = g.GetGeometryName()
        if name == "POLYGON":
            for j in range(g.GetGeometryCount()):
                points = np.array(g.GetGeometryRef(j).GetPoints())[:, :2]
                if not np.array_equal(points[0], points[-1]):
                    points = np.vstack([points, points[0]])
                points = transform(points)
                vertices.extend(points.tolist())
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2)
                             + [MplPath.CLOSEPOLY])
        elif name in ("MULTIPOLYGON", "GEOMETRYCOLLECTION"):
            for j in range(g.GetGeometryCount()):
                add(g.GetGeometryRef(j))
        else:
            raise ValueError("Unexpected geometry: " + name)
    add(geometry)
    return PathPatch(MplPath(vertices, codes), **kwargs)


def main():
    ogr.UseExceptions()
    source_sr = reference(25830)
    geographic_sr = reference(4326)
    to_geographic = osr.CoordinateTransformation(source_sr, geographic_sr)
    to_utm = osr.CoordinateTransformation(geographic_sr, source_sr)
    rows = []
    geojson_features = []
    for filename, label, role, color, reported_area in FIELDS:
        path = SOURCE / (filename + ".shp")
        if path.exists():
            ds = ogr.Open(str(path), 0)
            layer = ds.GetLayer(0)
            sr = layer.GetSpatialRef()
            sr.AutoIdentifyEPSG()
            assert sr.GetAuthorityCode(None) == "25830"
            assert layer.GetFeatureCount() == 1
            feature = layer.GetNextFeature()
            geom = feature.GetGeometryRef().Clone()
            original_hashes = {suffix: sha256(path.with_suffix(suffix))
                               for suffix in [".shp", ".shx", ".dbf", ".prj"]}
        else:
            archived = json.loads((MAPDATA / "study_fields_utm.json").read_text())
            assert archived["crs"] == "EPSG:25830"
            field = next(f for f in archived["fields"] if f["source_basename"] == filename)
            geom = ogr.CreateGeometryFromJson(json.dumps(field["geometry"]))
            original_hashes = field["source_files_sha256"]
        assert geom.IsValid()
        area = geom.GetArea() / 10000
        assert round(area, 2) == reported_area, (label, area)
        geographic = geom.Clone()
        geographic.Transform(to_geographic)
        geojson_features.append({"type": "Feature", "properties": {
            "field": label, "role": role, "area_ha": area,
            "source_geometry_crs": "EPSG:25830"},
            "geometry": json.loads(geographic.ExportToJson())})
        rows.append({"field": label, "role": role, "color": color,
                     "geometry": geom, "area_ha": area, "source": path,
                     "source_files_sha256": original_hashes})

    xmin = min(r["geometry"].GetEnvelope()[0] for r in rows)
    xmax = max(r["geometry"].GetEnvelope()[1] for r in rows)
    ymin = min(r["geometry"].GetEnvelope()[2] for r in rows)
    ymax = max(r["geometry"].GetEnvelope()[3] for r in rows)
    combined = rows[0]["geometry"].Clone()
    for row in rows[1:]:
        combined = combined.Union(row["geometry"])
    centre = combined.Centroid()
    centre_x, centre_y = centre.GetX(), centre.GetY()
    site_lon, site_lat, _ = to_geographic.TransformPoint(centre_x, centre_y)

    countries = json.loads(COUNTRIES.read_text(encoding="utf-8"))
    assert countries["type"] == "FeatureCollection"
    assert len(countries["features"]) > 150
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.labelcolor": INK,
                         "text.color": INK, "axes.edgecolor": "#8A9AA6",
                         "xtick.color": INK, "ytick.color": INK})
    fig = plt.figure(figsize=(8.0, 6.7), facecolor="white")
    local = fig.add_axes([0.465, 0.13, 0.515, 0.80])
    locator = fig.add_axes([0.075, 0.515, 0.30, 0.39])
    legend = fig.add_axes([0.075, 0.135, 0.325, 0.315])
    legend.set(xlim=(0, 1), ylim=(0, 1))
    legend.axis("off")

    # Regional locator in geographic coordinates. The cosine adjustment is for
    # display only; all metric distance and area calculations use EPSG:25830.
    locator.set_facecolor("#EFF6FA")
    for feature in countries["features"]:
        name = feature["properties"].get("ADMIN", "")
        g = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
        a, b, c, d = g.GetEnvelope()
        if b < -11 or a > 5 or d < 35 or c > 45:
            continue
        locator.add_patch(patch_from_geometry(
            g, facecolor="#DDE7EF" if name == "Spain" else "#F4F2ED",
            edgecolor="#9AAAB5", linewidth=0.65, zorder=2))
    locator.set(xlim=(-10.6, 4.8), ylim=(35.5, 44.8))
    locator.set_aspect(1 / np.cos(np.deg2rad(40)), adjustable="box")
    locator.set_title("(a) Regional setting", fontsize=11, fontweight="bold", loc="left", pad=12)
    locator.set_xticks([-10, -5, 0])
    locator.set_yticks([36, 40, 44])
    locator.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}°W" if x < 0 else f"{x:g}°"))
    locator.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}°N"))
    locator.tick_params(labelsize=8.5, length=3)
    locator.grid(color="#C7D4DD", linewidth=0.45, alpha=0.6, zorder=1)
    locator.text(-3.8, 38.6, "Spain", fontsize=10, fontweight="bold", ha="center", zorder=3)
    locator.text(-8.1, 39.2, "Portugal", fontsize=8.5, rotation=90, ha="center", zorder=3)
    locator.text(1.3, 44.15, "France", fontsize=8.5, ha="center", zorder=3)
    locator.scatter([site_lon], [site_lat], marker="*", s=105, color="#A33B42",
                    edgecolor="white", linewidth=0.6, zorder=5)
    locator.annotate("Valdeavero", xy=(site_lon, site_lat), xytext=(-0.4, 42.1),
                     fontsize=9, color="#8E3037", ha="center", zorder=6,
                     arrowprops={"arrowstyle": "-", "color": "#8E3037", "lw": 0.7})

    legend.text(0, 0.98, "Study fields", fontsize=11, fontweight="bold", va="top")
    for row, y in zip(rows, [0.77, 0.49, 0.21]):
        legend.add_patch(Rectangle((0, y - 0.015), 0.07, 0.07,
                                   facecolor=row["color"], edgecolor="#536673", linewidth=0.6))
        legend.text(0.11, y + 0.035, row["field"], fontsize=10.5,
                    fontweight="bold", va="center")
        legend.text(0.11, y - 0.045, row["role"], fontsize=8.9, va="center")

    local.set_facecolor("#FCFDFE")
    for row in rows:
        local.add_patch(patch_from_geometry(row["geometry"], transform=lambda xy: xy / 1000,
                                           facecolor=row["color"], edgecolor="#465D6B",
                                           linewidth=0.85, zorder=3))
        position = row["geometry"].PointOnSurface()
        local.text(position.GetX() / 1000, position.GetY() / 1000,
                   row["field"] + "\n" + f"{row['area_ha']:.2f} ha",
                   ha="center", va="center", fontsize=10, color="#192E3D",
                   linespacing=1.35, zorder=5,
                   bbox={"boxstyle": "round,pad=0.30", "facecolor": "white",
                         "edgecolor": "none", "alpha": 0.88})
    local.set(xlim=((xmin - 80) / 1000, (xmax + 80) / 1000),
              ylim=((ymin - 100) / 1000, (ymax + 100) / 1000))
    local.set_aspect("equal", adjustable="box")
    local.set_title("(b) Study field boundaries", fontsize=11,
                    fontweight="bold", loc="left", pad=12)
    local.set_xlabel("UTM easting (km)", fontsize=10, labelpad=7)
    local.set_ylabel("UTM northing (km)", fontsize=10, labelpad=7)
    local.xaxis.set_major_locator(MultipleLocator(0.2))
    local.yaxis.set_major_locator(MultipleLocator(0.2))
    local.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    local.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    local.tick_params(labelsize=8.5, length=3)
    local.grid(color="#D8E0E6", linewidth=0.5, zorder=1)

    # True north, computed in the same projected CRS, rather than assumed from
    # the image orientation. The small convergence is retained in the arrow.
    nx, ny, _ = to_utm.TransformPoint(site_lon, site_lat + 0.001)
    direction = np.array([nx - centre_x, ny - centre_y])
    direction /= np.linalg.norm(direction)
    arrow_base = np.array([(xmax + 15) / 1000, (ymax - 140) / 1000])
    arrow_tip = arrow_base + direction * 0.105
    local.annotate("", xy=arrow_tip, xytext=arrow_base,
                   arrowprops={"arrowstyle": "-|>", "color": INK,
                               "lw": 1.4, "mutation_scale": 14}, zorder=7)
    local.text(arrow_tip[0], arrow_tip[1] + 0.026, "N", fontsize=11,
               fontweight="bold", ha="center", va="center", zorder=7)

    # All positions and lengths below are kilometres in the projected map.
    bar_x = (xmin - 35) / 1000
    bar_y = (ymin - 38) / 1000
    for j in range(2):
        local.add_patch(Rectangle((bar_x + j * 0.1, bar_y), 0.1, 0.014,
                                   facecolor=INK if j == 0 else "white",
                                   edgecolor=INK, linewidth=0.7, zorder=6))
    for offset, label in [(0, "0"), (0.1, "100"), (0.2, "200 m")]:
        local.text(bar_x + offset, bar_y - 0.016, label,
                   fontsize=8.4, ha="center", va="top", zorder=7)

    OUT.mkdir(exist_ok=True)
    MAPDATA.mkdir(exist_ok=True)
    fig.savefig(OUT / "figure1.png", dpi=400, facecolor="white")
    fig.savefig(OUT / "figure1.svg", facecolor="white")
    fig.savefig(OUT / "figure1_preview.png", dpi=150, facecolor="white")
    plt.close(fig)
    (MAPDATA / "study_fields.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": geojson_features}, indent=2), encoding="utf-8")
    metadata = {
        "field_map_crs": "EPSG:25830",
        "field_map_crs_name": "ETRS89 / UTM zone 30N",
        "locator_crs": "EPSG:4326",
        "site_marker": "centroid of the union of the three original field geometries",
        "site_marker_longitude_latitude": [site_lon, site_lat],
        "scale_bar_metres": 200,
        "north_arrow": "true-north direction derived by geographic-to-projected transformation",
        "country_boundaries_source": "Natural Earth, Admin 0 countries, 1:110 million",
        "country_boundaries_url": COUNTRY_URL,
        "country_boundaries_sha256": sha256(COUNTRIES),
        "raster_dpi": 400,
        "embedded_footer": "removed at author request; CRS and Natural Earth attribution remain in the LaTeX caption",
        "figsize_inches": [8.0, 6.7],
        "fields": [{k: str(v) if isinstance(v, Path) else v
                    for k, v in row.items() if k != "geometry"} for row in rows],
        "code_sha256": sha256(Path(__file__)),
        "original_sources_modified": False,
    }
    (OUT / "figure1_data.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    print("FIGURE 1 COMPLETE: original polygons and manuscript areas verified.", flush=True)


if __name__ == "__main__":
    main()

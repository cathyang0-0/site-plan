"""Tests for the infrastructure stage — geometry handling + export behavior.
Suite convention: no network — pure conversion logic plus cache-hit fetch."""
import sys
from pathlib import Path

import ezdxf
import pytest
from shapely.geometry import Polygon, MultiPolygon, LineString, Point

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import overture_cache
from app.pipeline.infrastructure import (
    fetch_infrastructure, infrastructure_to_pixels, ATTRIBUTION,
)
from app.export.dxf import export_dxf, DEFAULT_LINE_WEIGHT_MM


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(overture_cache, "CACHE_DIR", tmp_path / "cache")


BBOX = dict(west=-1.0, south=-1.0, east=1.0, north=1.0)


class TestFetchAndConvert:
    def test_cache_hit_skips_network(self):
        overture_cache.put("infrastructure-v2", (-1.0, -1.0, 1.0, 1.0),
                           {"polygons": ["sentinel"], "lines": []})
        assert fetch_infrastructure(-1.0, -1.0, 1.0, 1.0)["polygons"] == ["sentinel"]

    def test_uses_v2_cache_key(self, monkeypatch):
        # The v1 cache key held UNFILTERED data (power lines included); the
        # fetch must look up the bumped key so stale v1 entries never serve.
        seen = {}
        def fake_get(kind, bbox):
            seen["kind"] = kind
            return {"polygons": [], "lines": []}
        monkeypatch.setattr(overture_cache, "get", fake_get)
        fetch_infrastructure(-9.0, -9.0, 9.0, 9.0)
        assert seen["kind"] == "infrastructure-v2"

    def test_subtype_filter(self):
        from app.pipeline.infrastructure import subtype_included, EXCLUDED_SUBTYPES
        for sub in EXCLUDED_SUBTYPES:
            assert not subtype_included(sub)
        for sub in ("pier", "bridge", "barrier", "pedestrian", "transit",
                    "water", "transportation", None):
            assert subtype_included(sub)

    def test_pixels_split_types_and_clip(self):
        data = {
            "polygons": [Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])],
            "lines": [LineString([(-2, 0), (2, 0)])],   # straddles the bbox
        }
        px = infrastructure_to_pixels(data, **BBOX, img_w=200, img_h=200)
        assert len(px["polygons"]) == 1 and len(px["lines"]) == 1
        assert px["lines"][0].bounds[0] >= 0        # clipped to frame
        assert px["lines"][0].bounds[2] <= 200

    def test_attribution_is_odbl(self):
        assert "ODbL" in ATTRIBUTION


def _export(tmp_path, infrastructure, buildings=(), land=None, contours=None):
    out = tmp_path / "infra.dxf"
    export_dxf(
        output_path=out, buildings=list(buildings), roads=[],
        tree_placements=[], tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]],
        land_types=([{"label": "vegetation", "polygons": land,
                      "style": {"hatch_type": "acad", "hatch_pattern": "AR-SAND"}}]
                    if land else []),
        style={}, scale_m_per_px=1.0, origin_px=(0, 0),
        contours=contours, infrastructure=infrastructure,
    )
    return ezdxf.readfile(str(out))


class TestExportBehavior:
    def test_layer_between_trees_and_roofs_with_030_weight(self, tmp_path):
        doc = _export(tmp_path, {"polygons": [Polygon([(0, 0), (10, 0), (10, 10)])],
                                 "lines": []})
        names = [l.dxf.name for l in doc.layers]
        assert names.index("TREES") < names.index("INFRASTRUCTURE") < names.index("ROOFS")
        layer = doc.layers.get("INFRASTRUCTURE")
        assert layer.dxf.lineweight == 30            # 0.30 mm
        assert DEFAULT_LINE_WEIGHT_MM["ROADS"] < 0.30 < DEFAULT_LINE_WEIGHT_MM["ROOFS"]

    def test_clipped_by_buildings_roof_wins(self, tmp_path):
        # Pier deck under a building: the overlap is cut out of the deck.
        deck = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        bldg = Polygon([(40, -5), (60, -5), (60, 25), (40, 25)])
        doc = _export(tmp_path, {"polygons": [deck], "lines": []}, buildings=[bldg])
        infra = [e for e in doc.modelspace().query("LWPOLYLINE")
                 if e.dxf.layer == "INFRASTRUCTURE"]
        assert len(infra) == 2                       # split into two pieces
        xs = [x for e in infra for x, y in e.get_points("xy")]
        assert not any(40 < x < 60 for x in xs)      # nothing under the roof

    def test_hatches_clipped_at_infra_boundary(self, tmp_path):
        # Land hatch must hole out where the pier deck sits.
        deck = Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
        land = MultiPolygon([Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])])
        doc = _export(tmp_path, {"polygons": [deck], "lines": []}, land=land)
        h = [h for h in doc.modelspace().query("HATCH")
             if h.dxf.layer == "LANDTYPE_1"][0]
        assert len([pp for pp in h.paths if hasattr(pp, "vertices")]) == 2  # ring + hole

    def test_contours_clipped_under_infra(self, tmp_path):
        deck = Polygon([(40, 0), (60, 0), (60, 30), (40, 30)])
        doc = _export(tmp_path, {"polygons": [deck], "lines": []},
                      contours=[{"points": [(0, 15), (100, 15)], "level": 3.0}])
        segs = [e for e in doc.modelspace().query("LWPOLYLINE")
                if e.dxf.layer == "CONTOURS"]
        assert len(segs) == 2                        # broken at the deck

    def test_roads_clipped_at_infra_boundary(self, tmp_path):
        # A road corridor crossing a pier deck is cut where the deck covers it
        # (user spec: the deck's surface wins, same as under buildings).
        import ezdxf as _e
        from shapely.geometry import LineString as LS, Polygon as P
        deck = P([(40, -20), (60, -20), (60, 20), (40, 20)])
        out = tmp_path / "roadclip.dxf"
        export_dxf(
            output_path=out, buildings=[],
            roads=[{"line": LS([(0, 0), (100, 0)]), "width_px": 10}],
            tree_placements=[], tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]],
            land_types=[], style={}, scale_m_per_px=1.0, origin_px=(0, 0),
            infrastructure={"polygons": [deck], "lines": []},
        )
        doc = _e.readfile(str(out))
        road_pts = [(x, y) for e in doc.modelspace().query("LWPOLYLINE")
                    if e.dxf.layer == "ROADS" for x, y in e.get_points("xy")]
        assert road_pts  # road still drawn outside the deck
        assert not any(41 < x < 59 for x, y in road_pts)  # nothing under it

    def test_lines_drawn_open(self, tmp_path):
        doc = _export(tmp_path, {"polygons": [],
                                 "lines": [LineString([(0, 0), (50, 5), (100, 0)])]})
        infra = [e for e in doc.modelspace().query("LWPOLYLINE")
                 if e.dxf.layer == "INFRASTRUCTURE"]
        assert len(infra) == 1 and not infra[0].closed

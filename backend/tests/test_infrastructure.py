"""Tests for the infrastructure stage — per-class treatment (user-specified
after reviewing rendered samples of every class), geometry handling, export.
Suite convention: no network — pure logic plus cache-key checks."""
import sys
from pathlib import Path

import ezdxf
import pytest
from shapely.geometry import Polygon, LineString

sys.path.insert(0, str(Path(__file__).parent.parent))

from siteplan_backend.pipeline import overture_cache
from siteplan_backend.pipeline.infrastructure import (
    fetch_infrastructure, infrastructure_to_pixels, classify,
    subtype_included, EXCLUDED_SUBTYPES, WIDTH_M, GROUP_STYLE, ATTRIBUTION,
)
from siteplan_backend.export.dxf import export_dxf, _nearest_dxf_lineweight


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(overture_cache, "CACHE_DIR", tmp_path / "cache")


class TestClassify:
    def test_service_subtypes_dropped(self):
        for sub in EXCLUDED_SUBTYPES:
            assert classify(sub, "power_line", "LineString") is None

    def test_pier_closed_shapes_only(self):
        # OSM maps a pier deck AND its edge lines; the lines double the
        # deck's own outline — structure keeps closed shapes only (user spec).
        assert classify("pier", "pier", "Polygon") == ("structure", None)
        assert classify("pier", "pier", "LineString") is None

    def test_thin_groups_are_light_gray(self):
        for g in ("micro", "lift", "wall", "kerb", "fence"):
            assert GROUP_STYLE[g]["color"] == "#b4b4b4"
        for g in ("structure", "bridge", "airfield", "breakwater"):
            assert GROUP_STYLE[g]["color"] == "#000000"

    def test_bridge_strip_over_mapped_deck_dropped(self):
        # Same bridge mapped twice: deck polygon + centerline. The buffered
        # strip must be pruned; the mapped deck stays.
        deck = Polygon([(-0.001, -0.00005), (0.001, -0.00005),
                        (0.001, 0.00005), (-0.001, 0.00005)])
        data = {"features": [
            {"subtype": "bridge", "class": "bridge", "geom": deck},
            {"subtype": "bridge", "class": "bridge",
             "geom": LineString([(-0.001, 0.0), (0.001, 0.0)])},
        ]}
        px = infrastructure_to_pixels(data, west=-0.01, south=-0.01,
                                      east=0.01, north=0.01,
                                      img_w=2000, img_h=2000)
        assert len(px["groups"]["bridge"]["polys"]) == 1   # deck only

    def test_bridge_over_structure_dropped(self):
        pier = Polygon([(-0.002, -0.001), (0.002, -0.001),
                        (0.002, 0.001), (-0.002, 0.001)])
        data = {"features": [
            {"subtype": "pier", "class": "pier", "geom": pier},
            {"subtype": "bridge", "class": "bridge",
             "geom": LineString([(-0.001, 0.0), (0.001, 0.0)])},  # on the pier
        ]}
        px = infrastructure_to_pixels(data, west=-0.01, south=-0.01,
                                      east=0.01, north=0.01,
                                      img_w=2000, img_h=2000)
        assert "bridge" not in px["groups"] or not px["groups"]["bridge"]["polys"]
        assert len(px["groups"]["structure"]["polys"]) == 1

    def test_bridge_polygon_kept_line_offset(self):
        assert classify("bridge", "bridge", "Polygon") == ("bridge", None)
        assert classify("bridge", "bridge", "LineString") == ("bridge", WIDTH_M["bridge"])

    def test_breakwater_offset(self):
        assert classify("water", "breakwater", "LineString") == \
            ("breakwater", WIDTH_M["breakwater"])

    def test_barrier_split_by_class(self):
        assert classify("barrier", "kerb", "LineString") == ("kerb", None)
        assert classify("barrier", "wall", "LineString") == ("wall", WIDTH_M["wall"])
        assert classify("barrier", "retaining_wall", "LineString") == \
            ("wall", WIDTH_M["wall"])
        assert classify("barrier", "fence", "LineString") == ("fence", WIDTH_M["fence"])

    def test_airport_pavement_only(self):
        assert classify("airport", "runway", "LineString") == \
            ("airfield", WIDTH_M["runway"])
        assert classify("airport", "taxiway", "LineString") == \
            ("airfield", WIDTH_M["taxiway"])
        assert classify("airport", "apron", "Polygon") == ("airfield", None)
        assert classify("airport", "municipal_airport", "Polygon") is None  # zoning

    def test_lifts_thin_lines(self):
        assert classify("aerialway", "chair_lift", "LineString") == ("lift", None)
        assert classify("aerialway", "gondola", "LineString") == ("lift", None)

    def test_pedestrian_micro(self):
        assert classify("pedestrian", "toilets", "Polygon") == ("micro", None)

    def test_weights_ordering(self):
        # User spec: structure 0.30 > bridge/airfield (road weight) >
        # breakwater > micro > lift > wall > kerb > fence.
        w = {k: v["weight_mm"] for k, v in GROUP_STYLE.items()}
        assert w["structure"] > w["bridge"] == w["airfield"] > w["breakwater"] \
            > w["micro"] > w["lift"] > w["wall"] > w["kerb"] > w["fence"]


class TestFetchAndConvert:
    def test_cache_hit_and_v3_key(self, monkeypatch):
        seen = {}
        def fake_get(kind, bbox):
            seen["kind"] = kind
            return {"features": []}
        monkeypatch.setattr(overture_cache, "get", fake_get)
        assert fetch_infrastructure(-1, -1, 1, 1) == {"features": []}
        assert seen["kind"] == "infrastructure-v3"

    def test_offset_classes_become_strips(self):
        # A wall centerline must come out as a thin POLYGON strip (two edges),
        # a kerb as a bare line, and both scaled correctly: bbox 2°x2° at
        # equator over a 2000px image -> ~111.32 m/px... use a small bbox.
        data = {"features": [
            {"subtype": "barrier", "class": "wall",
             "geom": LineString([(-0.005, 0.0), (0.005, 0.0)])},
            {"subtype": "barrier", "class": "kerb",
             "geom": LineString([(-0.005, 0.001), (0.005, 0.001)])},
        ]}
        px = infrastructure_to_pixels(data, west=-0.01, south=-0.01,
                                      east=0.01, north=0.01,
                                      img_w=2000, img_h=2000)
        groups = px["groups"]
        assert len(groups["wall"]["polys"]) == 1     # strip
        assert groups["wall"]["lines"] == []
        assert len(groups["kerb"]["lines"]) == 1     # stroke
        # strip thickness ~ WALL width: bbox is ~2226 m wide -> 1.113 m/px,
        # so 0.4 m -> ~0.36 px thick
        strip = groups["wall"]["polys"][0]
        thickness_px = strip.bounds[3] - strip.bounds[1]
        assert 0.2 < thickness_px < 0.6

    def test_attribution_is_odbl(self):
        assert "ODbL" in ATTRIBUTION


def _export(tmp_path, groups, buildings=()):
    out = tmp_path / "infra.dxf"
    export_dxf(
        output_path=out, buildings=list(buildings), roads=[],
        tree_placements=[], tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]],
        land_types=[], style={}, scale_m_per_px=1.0, origin_px=(0, 0),
        infrastructure={"groups": groups},
    )
    return ezdxf.readfile(str(out))


def _grp(name, polys=(), lines=()):
    return {**GROUP_STYLE[name], "polys": list(polys), "lines": list(lines)}


class TestExportBehavior:
    def test_per_class_layers_carry_the_weights(self, tmp_path):
        # Weights live at LAYER level (Rhino discards entity-level lineweight
        # overrides on import — user-observed), one layer per class group.
        doc = _export(tmp_path, {
            "structure": _grp("structure", polys=[Polygon([(0, 0), (10, 0), (10, 10)])]),
            "wall": _grp("wall", polys=[Polygon([(20, 0), (30, 0), (30, 1)])]),
            "kerb": _grp("kerb", lines=[LineString([(40, 0), (50, 0)])]),
            "fence": _grp("fence", polys=[Polygon([(60, 0), (70, 0), (70, 1)])]),
        })
        lw = {l.dxf.name: l.dxf.lineweight for l in doc.layers}
        assert lw["INFRA_STRUCTURE"] == _nearest_dxf_lineweight(0.30)
        assert lw["INFRA_WALL"] == _nearest_dxf_lineweight(0.08)
        assert lw["INFRA_KERB"] == _nearest_dxf_lineweight(0.05)
        assert lw["INFRA_FENCE"] == _nearest_dxf_lineweight(0.04)
        # entities sit on their class layer, weight BYLAYER
        for e in doc.modelspace().query("LWPOLYLINE"):
            if e.dxf.layer.startswith("INFRA_"):
                assert e.dxf.lineweight == -1   # BYLAYER
        placed = {e.dxf.layer for e in doc.modelspace().query("LWPOLYLINE")}
        assert {"INFRA_STRUCTURE", "INFRA_WALL", "INFRA_KERB",
                "INFRA_FENCE"} <= placed

    def test_infra_layers_between_trees_and_roofs_thin_to_thick(self, tmp_path):
        doc = _export(tmp_path, {"structure": _grp(
            "structure", polys=[Polygon([(0, 0), (10, 0), (10, 10)])])})
        names = [l.dxf.name for l in doc.layers]
        infra = [n for n in names if n.startswith("INFRA_")]
        assert infra  # registered
        assert all(names.index("TREES") < names.index(n) < names.index("ROOFS")
                   for n in infra)
        weights = [doc.layers.get(n).dxf.lineweight for n in infra]
        assert weights == sorted(weights)   # thin -> thick in table order

    def test_micro_overlapping_building_dropped(self, tmp_path):
        bldg = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        overlapping = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        standalone = Polygon([(40, 40), (50, 40), (50, 50), (40, 50)])
        doc = _export(tmp_path,
                      {"micro": _grp("micro", polys=[overlapping, standalone])},
                      buildings=[bldg])
        infra = [e for e in doc.modelspace().query("LWPOLYLINE")
                 if e.dxf.layer == "INFRA_MICRO"]
        assert len(infra) == 1                       # only the standalone one
        xs = [x for x, y in infra[0].get_points("xy")]
        assert min(xs) >= 40

    def test_clip_groups_mask_roads_walls_do_not(self, tmp_path):
        # A structure deck cuts the road corridor; a wall strip must NOT.
        deck = Polygon([(40, -20), (60, -20), (60, 20), (40, 20)])
        wall = Polygon([(70, -20), (70.4, -20), (70.4, 20), (70, 20)])
        out = tmp_path / "clip.dxf"
        export_dxf(
            output_path=out, buildings=[],
            roads=[{"line": LineString([(0, 0), (100, 0)]), "width_px": 10}],
            tree_placements=[], tree_block_curves=[[[(0, 0), (1, 0), (1, 1)]]],
            land_types=[], style={}, scale_m_per_px=1.0, origin_px=(0, 0),
            infrastructure={"groups": {"structure": _grp("structure", polys=[deck]),
                                       "wall": _grp("wall", polys=[wall])}},
        )
        doc = ezdxf.readfile(str(out))
        road_pts = [(x, y) for e in doc.modelspace().query("LWPOLYLINE")
                    if e.dxf.layer == "ROADS" for x, y in e.get_points("xy")]
        assert not any(41 < x < 59 for x, y in road_pts)  # deck clips road
        assert any(x > 65 for x, y in road_pts)           # road continues past wall

    def test_infra_still_clipped_by_buildings(self, tmp_path):
        deck = Polygon([(0, 0), (100, 0), (100, 20), (0, 20)])
        bldg = Polygon([(40, -5), (60, -5), (60, 25), (40, 25)])
        doc = _export(tmp_path, {"structure": _grp("structure", polys=[deck])},
                      buildings=[bldg])
        infra = [e for e in doc.modelspace().query("LWPOLYLINE")
                 if e.dxf.layer == "INFRA_STRUCTURE"]
        assert len(infra) == 2
        xs = [x for e in infra for x, y in e.get_points("xy")]
        assert not any(41 < x < 59 for x in xs)

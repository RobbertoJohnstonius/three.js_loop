#!/usr/bin/env python3
"""
threejs_loop.py — Autonomous Three.js game asset generator.

4-agent pipeline per iteration:
  Playtester → geometry validation (Node) + 4-angle render (Puppeteer) + image analysis
  Vision     → OpenRouter multimodal model reviews the 2×2 angle grid
  Critic     → LLM identifies quality issues, merges vision + geometry context (JSON)
  Planner    → LLM proposes one specific improvement (JSON)
  Coder A/B  → Two competing implementations; winner evaluated by playtester

Architectural patterns adapted from auto_game(gamedev)/gamedev/game_loop.py:
  - A/B candidate system  (generate two variants, evaluate both, commit winner)
  - Episodic memory       (.jsonl — one line per iteration for pattern analysis)
  - steering.json         (human override: force_plan, focus_hints, forbidden improvements)
  - rules.json            (permanent lessons extracted from failures, injected into coder)
  - status.txt            (human-readable summary overwritten each iteration)
  - Failure categorisation (SYNTAX / GEOMETRY / RENDER / GENERATION)
  - Trust score           (impl_rate × 40 + score_trend × 30 + stability × 30)

Usage:
  python threejs_loop.py rock "A rough boulder for a fantasy RPG"
  python threejs_loop.py --resume
  python threejs_loop.py --score-only --resume
"""

import argparse
import hashlib
import json
import logging
import math as _math
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from llm_router import call_llm, call_llm_coder, call_llm_vision
from target_adapter import SCREENSHOTS_DIR, ThreeJSAdapter

# ── Paths ─────────────────────────────────────────────────────────────────────

LOOP_DIR       = Path(__file__).parent.resolve()
CANDIDATES_DIR = LOOP_DIR / "candidates"
STATE_PATH     = LOOP_DIR / "loop_state.json"
EPISODIC_PATH  = LOOP_DIR / "episodic.jsonl"
STATUS_PATH    = LOOP_DIR / "status.txt"
RULES_PATH     = LOOP_DIR / "rules.json"
STEERING_PATH  = LOOP_DIR / "steering.json"
BRIEF_PATH     = LOOP_DIR / "brief.json"
REFERENCES_DIR = LOOP_DIR / "references"
DIST_DIR       = LOOP_DIR / "dist"
GRAVEYARD_DIR  = CANDIDATES_DIR / "graveyard"
GRAVEYARD_PATH = LOOP_DIR / "graveyard.jsonl"

CANDIDATES_DIR.mkdir(exist_ok=True)
REFERENCES_DIR.mkdir(exist_ok=True)
DIST_DIR.mkdir(exist_ok=True)
GRAVEYARD_DIR.mkdir(exist_ok=True)

# ── Quality thresholds ────────────────────────────────────────────────────────

POLY_IDEAL_MIN = 200
POLY_IDEAL_MAX = 2000
POLY_MIN       = 50
POLY_MAX       = 8000

# Calibrated from benchmark suite: single-colour MeshStandardMaterial with the
# #1a1a2e background rig produces diversity 14–24 for proper geometry; flat/no-normal
# assets score 9–13. Threshold of 12 reliably separates them.
COLOR_DIVERSITY_THRESHOLD    = 12
COVERAGE_UNIFORMITY_THRESHOLD = 0.70

# Raised from 0.85 after scoring formula update: poly_ok (50–199 tris) + full visual now
# reaches 0.90, so 0.85 would pass too easily. 0.92 requires poly_ideal (≥200 tris) plus
# uniformity + diversity — matching the 300–800 tri target in brief.json.
PASS_THRESHOLD = 0.92
MAX_ITERATIONS = 30
REVERT_AFTER_N_REGRESSIONS = 3

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    handlers=[
        logging.FileHandler("threejs_loop.log"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ── Failure categories (adapted from auto_game) ───────────────────────────────

FAIL_SYNTAX   = "SYNTAX"       # JS parse/validate_js_basic failed
FAIL_GEOMETRY = "GEOMETRY"     # geometry validator returned ok=False
FAIL_RENDER   = "RENDER"       # Puppeteer render crashed
FAIL_GEN      = "GENERATION"   # LLM output too short or missing export

# ── Permanent rule strings (hoisted so _maybe_extract_rule can reference them) ─

RULE_SYNTAX   = "always call geometry.computeVertexNormals() after modifying vertex positions, and verify brace balance before completing the file"
RULE_GEOMETRY = "createAsset() must add geometry to a Mesh and the Mesh to the returned Object3D; never return an empty Group"

# ── Scoring constants ─────────────────────────────────────────────────────────

MAX_SCORE_DELTA = 4.0   # normalisation divisor in compute_trust score trend

# ── Rubric thresholds ─────────────────────────────────────────────────────────

SPIKE_RATIO_MAX      = 3.0   # max/mean vertex-centroid distance; above = spike artefacts
DARK_PATCH_MAX       = 0.18  # fraction of near-black (max_channel < 20) non-bg pixels;
                             # only scored when vision also confirms normal_issues=True.
                             # 0.18 chosen because directional light on a sphere naturally
                             # puts ~15% of pixels in shadow — only flag egregious cases.
SHARED_EDGE_MIN      = 0.97  # mesh connectivity: 1.0 = fully connected, 0.0 = fully torn.
                             # 0.97 catches UV-seam cracks (0.963) while allowing minor boundary
                             # edges on open meshes. Closed boulders must be 1.0.
                             # Sequential per-index LCG on non-indexed geometry scores 0.0.
                             # Position-based noise scores 1.0. Below 0.85 = disconnected mesh.

# ── Rubric tier labels ────────────────────────────────────────────────────────

TIER_PRODUCTION = "PRODUCTION_READY"
TIER_POLISH     = "NEEDS_POLISH"
TIER_FAILED     = "FAILED"

# ── Permanent rule strings for rubric-pattern learning ────────────────────────

RULE_SPIKE = (
    "When displacing vertices, normalize the direction and clamp magnitude to ≤30% of the "
    "mesh's average vertex radius (i.e. `disp = rand() * 0.09` for a unit-radius mesh) "
    "to prevent spike artefacts (spike_ratio must stay below 3.0)"
)
RULE_INVERTED_NORMALS = (
    "After any vertex position change — including clamping, offsetting, or flat-bottom ops — "
    "always call geometry.computeVertexNormals() immediately; never modify positions after "
    "computing normals or the normals will be wrong"
)
RULE_MATERIAL_TYPE = (
    "Every mesh must use MeshStandardMaterial or MeshPhysicalMaterial — "
    "never MeshBasicMaterial, MeshLambertMaterial, or MeshPhongMaterial"
)
RULE_CONNECTED_MESH = (
    "IcosahedronGeometry and PolyhedronGeometry are non-indexed BufferGeometries. "
    "Call mergeVertices() immediately after creation to convert to indexed geometry before any "
    "displacement. This gives each unique vertex one index so sequential LCG is consistent "
    "(shared_edge_fraction ≥ 0.95, mesh stays connected) and computeVertexNormals() produces "
    "smooth shading. Pattern: "
    "import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'; "
    "let geo = new THREE.IcosahedronGeometry(r, d); geo = mergeVertices(geo); "
    "// then displace vertices, then geo.computeVertexNormals()"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(path: Path, default):
    """Read and parse a JSON file. Returns default if missing or invalid."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _build_rules_block(rules: list) -> str:
    if not rules:
        return "(none yet)"
    hard, soft = [], []
    for r in rules:
        if isinstance(r, dict):
            sev = r.get("severity", "")
            tag = f" [{sev.upper()}]" if sev else ""
            hard.append(f"- {r['rule']}{tag}")
        else:
            soft.append(f"- {r}")
    parts = []
    if hard:
        parts.append("HARD CONSTRAINTS (override all others):\n" + "\n".join(hard))
    if soft:
        parts.append("GENERAL RULES (asset-specific rules below override generic ones):\n" + "\n".join(soft))
    return "\n\n".join(parts)

# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict | None:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception as e:
            log.warning(f"Could not load loop_state.json: {e}")
    return None


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def fresh_state(asset_name: str, asset_description: str) -> dict:
    return {
        "asset_name":             asset_name,
        "asset_description":      asset_description,
        "started_at":             datetime.now().isoformat(),
        "iteration":              0,
        "current_version":        None,
        "best_version":           None,
        "best_score":             0.0,
        "consecutive_regressions": 0,
        "done":                   False,
        "trust_score":            50.0,
        "history":                [],
        "stagnation_count":       0,
    }

# ── Rules (permanent lessons, never rolled) ───────────────────────────────────

def load_rules() -> list[str]:
    return _read_json(RULES_PATH, [])


def append_rule(lesson: str) -> None:
    rules = load_rules()
    if lesson not in rules:
        rules.append(lesson)
        RULES_PATH.write_text(json.dumps(rules, indent=2))
        log.info(f"[rules] new rule: {lesson!r}")

# ── Steering (human override) ─────────────────────────────────────────────────

def load_steering() -> dict:
    """
    Load steering.json if it exists. Human can edit this live to influence the loop.
    Fields:
      force_plan: {title, instruction} | null   — skip Planner, use this plan directly
      focus_hints: ["..."]                       — injected at top of Planner prompt
      forbidden_improvements: ["..."]            — Planner must avoid these
    """
    return _read_json(STEERING_PATH, {"force_plan": None, "focus_hints": [], "forbidden_improvements": []})


def load_brief() -> dict:
    """Load brief.json — art style and technical constraints for the target game."""
    return _read_json(BRIEF_PATH, {})


_BRIEF_CODER_SKIP = {
    # Loop-internal metadata — not useful coder instructions
    "reference_spec", "pixel_targets", "expected_color_hsv",
    "expected_silhouette_hw_ratio", "expected_cluster_density_max",
    "expected_proportions", "expected_arrangement", "expected_mesh_types",
    "expected_material_properties", "min_reference_similarity_to_pass",
    "expected_mesh_colors",
}

def _brief_block(brief: dict) -> str:
    if not brief:
        return ""
    lines = ["ART BRIEF (target game context):"]
    for k, v in brief.items():
        if k in _BRIEF_CODER_SKIP or k.startswith("_"):
            continue
        if isinstance(v, list):
            lines.append(f"  {k}:")
            for item in v:
                lines.append(f"    - {item}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)

def _camera_basis_block(brief: dict) -> str:
    cam = brief.get("camera_config")
    if not cam:
        return ""
    pos = cam.get("camera_position", [])
    tgt = cam.get("camera_target", [])
    basis = cam.get("camera_basis", {})
    rules = cam.get("derived_rules", [])
    lines = [
        "CAMERA BASIS (world-space → image mapping for this asset):",
        f"  Camera: position={pos}  target={tgt}",
    ]
    for axis, label in basis.items():
        lines.append(f"  {axis}: {label}")
    if rules:
        lines.append("  Derived axis rules:")
        for r in rules:
            lines.append(f"    - {r}")
    return "\n".join(lines)


def _check_arrangement(geo: dict, brief: dict) -> None:
    """Cluster primary mesh positions to check arrangement. Mutates geo in-place."""
    exp = brief.get("expected_arrangement")
    if not exp or not geo.get("ok"):
        return
    # Support any primary mesh name, not just 'body'
    primary_name = exp.get("primary_mesh_name", "body")
    named = geo.get("named_mesh_positions", {})
    body_positions = named.get(primary_name) or geo.get("body_mesh_positions", [])
    body_count = len(body_positions)
    expected_count = exp.get("body_count", 0)
    expected_rows = exp.get("row_count", 2)
    expected_per_row = exp.get("bodies_per_row", 3)
    row_axis = exp.get("row_axis", "y")

    geo["arrangement_body_count"] = body_count

    if body_count == 0:
        geo["arrangement_ok"] = False
        geo["arrangement_row_count"] = 0
        return

    coords = [p.get(row_axis, 0.0) for p in body_positions]

    if expected_rows == 2 and body_count >= 2:
        median = sorted(coords)[body_count // 2]
        n_low = sum(1 for c in coords if c < median)
        n_high = sum(1 for c in coords if c >= median)
        det_rows = 2 if n_low > 0 and n_high > 0 else 1
        geo["arrangement_row_count"] = det_rows
        geo["arrangement_ok"] = (
            body_count == expected_count
            and det_rows == expected_rows
            and abs(n_low - expected_per_row) <= 1
            and abs(n_high - expected_per_row) <= 1
        )
    else:
        geo["arrangement_row_count"] = 1 if body_count > 0 else 0
        geo["arrangement_ok"] = body_count == expected_count

    # Cluster density: mean pairwise Euclidean distance between primary mesh centroids
    if body_count >= 2:
        capped = body_positions[:50]      # O(n²) capped at 50 objects
        total_dist, pair_count = 0.0, 0
        for i in range(len(capped)):
            for j in range(i + 1, len(capped)):
                pi, pj = capped[i], capped[j]
                dx = pi.get("x", 0.0) - pj.get("x", 0.0)
                dy = pi.get("y", 0.0) - pj.get("y", 0.0)
                dz = pi.get("z", 0.0) - pj.get("z", 0.0)
                total_dist += _math.sqrt(dx*dx + dy*dy + dz*dz)
                pair_count += 1
        geo["cluster_density_mean"] = round(total_dist / pair_count, 4)
    else:
        geo["cluster_density_mean"] = 0.0


def _sync_to_dist(asset_name: str, version: str, geometry: dict | None = None) -> None:
    """Copy winning candidate to dist/ so viewer stays current."""
    src = candidate_path(asset_name, version)
    if src.exists():
        dst = DIST_DIR / f"{asset_name}.mjs"
        shutil.copy2(src, dst)
        log.info(f"[dist] synced {src.name} → {dst.name}")

    if geometry:
        collision_meshes = geometry.get("collision_meshes")
        if collision_meshes:
            col_dst = DIST_DIR / f"{asset_name}_collision.json"
            col_dst.write_text(json.dumps({"asset": asset_name, "meshes": collision_meshes}, indent=2))
            log.info(f"[dist] wrote {col_dst.name} ({len(collision_meshes)} mesh(es))")


def _auto_calibrate_color(metrics: dict) -> None:
    """On first accepted iteration, write observed hue stats to brief.json if unset."""
    brief = load_brief()
    if brief.get("expected_color_hsv") and not brief["expected_color_hsv"].get("_auto_calibrated"):
        return
    pixel_m = (metrics.get("screenshot_analysis") or {}).get("pixel_metrics") or {}
    dom_hue = pixel_m.get("dominant_hue")
    mean_sat = pixel_m.get("mean_saturation")
    hue_var = pixel_m.get("hue_variance", 25)
    if dom_hue is None:
        return
    tol = round(max(20.0, min(60.0, float(hue_var) * 0.9)), 1)
    brief["expected_color_hsv"] = {
        "hue_center": round(dom_hue, 1),
        "hue_tolerance": tol,
        "saturation_min": round(max(0.0, float(mean_sat or 0.3) * 0.55), 2),
        "_auto_calibrated": True,
    }
    BRIEF_PATH.write_text(json.dumps(brief, indent=2))
    log.info(f"[auto-calibrate] set expected_color_hsv: hue={dom_hue:.0f}° ±{tol}°")


def _auto_append_focus_hint(plan: dict) -> None:
    """Append the successful planner instruction to steering.json focus_hints."""
    instruction = plan.get("instruction", "")
    if not instruction or len(instruction) < 20:
        return
    hint = f"PREVIOUSLY SUCCESSFUL ({plan.get('improvement_type', '?')}): {instruction[:130]}"
    steering = load_steering()
    hints = steering.get("focus_hints", [])
    other = [h for h in hints if not h.startswith("PREVIOUSLY SUCCESSFUL")]
    prev = [h for h in hints if h.startswith("PREVIOUSLY SUCCESSFUL")][-1:]
    steering["focus_hints"] = other + prev + [hint]
    STEERING_PATH.write_text(json.dumps(steering, indent=2))
    log.info(f"[auto-steering] appended hint: {hint[:70]!r}")


def _inject_stagnation_recovery(asset_path: Path) -> None:
    """Inject a geometry-strategy-swap force_plan when score has plateaued."""
    code = asset_path.read_text() if asset_path.exists() else ""
    uses_icosahedron = "IcosahedronGeometry" in code
    uses_custom_buf = "BufferGeometry" in code and "Float32Array" in code

    if uses_icosahedron and not uses_custom_buf:
        approach = "switch from IcosahedronGeometry to custom BufferGeometry"
        instruction = (
            "The current IcosahedronGeometry approach has stagnated — score not improving. "
            "Rebuild the main shape using a custom BufferGeometry: construct vertex positions "
            "explicitly with Float32Array, set faces with setIndex(). "
            "Keep the same visual intent but use a completely different geometry construction."
        )
    elif uses_custom_buf:
        approach = "switch from custom BufferGeometry to Three.js built-in primitives"
        instruction = (
            "The custom BufferGeometry approach has stagnated. Rebuild using Three.js built-in "
            "primitives (SphereGeometry, CylinderGeometry, BoxGeometry) combined as a Group. "
            "Keep the same visual intent but use a completely different construction strategy."
        )
    else:
        approach = "change geometry construction strategy"
        instruction = (
            "The current geometry approach has stagnated. Try a fundamentally different construction: "
            "if using primitives, switch to custom BufferGeometry; if using custom geometry, "
            "switch to combining built-in primitives."
        )

    steering = load_steering()
    steering["force_plan"] = {
        "improvement_type": "shape",
        "title": f"Stagnation recovery: {approach}",
        "rationale": "Score unchanged for 4+ iterations — geometry strategy change required",
        "instruction": instruction,
        "_stagnation_injected": True,
    }
    STEERING_PATH.write_text(json.dumps(steering, indent=2))
    log.info(f"[stagnation] injected force_plan: {approach}")


# ── Reference image system ────────────────────────────────────────────────────

def find_reference(asset_name: str) -> Path | None:
    """Return the reference image path for asset_name if one exists in references/."""
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = REFERENCES_DIR / f"{asset_name}.{ext}"
        if p.exists():
            return p
    return None


def run_reference_comparison(
    asset_name: str,
    ref_path: Path,
    render_description: str,
    stitched_path: str | None = None,
    geo_count: int | None = None,
) -> dict:
    """
    Compare the current render against a reference image.
    When stitched_path is provided: single dual-image vision call — IMAGE A is the
    reference photo, IMAGE B is the 4-angle render grid (direct visual comparison).
    Fallback when stitched_path is absent: describe reference → text LLM compare (legacy).

    Returns {reference_similarity, what_matches, what_to_improve,
             reference_count_match, reference_critical_gap} or {} on failure.
    """
    if stitched_path and Path(stitched_path).exists():
        # ── Direct dual-image comparison (Item 6) ────────────────────────────
        geo_count_note = ""
        if geo_count:
            geo_count_note = (
                f"\nIMPORTANT: The geometry validator has confirmed {geo_count} mesh(es) of the "
                "primary type exist in the 3D model. If you cannot see all of them in IMAGE B, "
                "some are occluded by overlapping neighbours — they are NOT missing. "
                f"Set reference_count_match=true if {geo_count} equals the reference count.\n"
            )
        dual_prompt = (
            f"You are comparing a reference photo (IMAGE A) to a 3D model render (IMAGE B) "
            f"of a game asset called \"{asset_name}\".\n"
            "IMAGE A is the reference (ground truth). "
            "IMAGE B shows the model from four angles: top-left=quarter, top-right=front, "
            f"bottom-left=right, bottom-right=top-down.\n{geo_count_note}\n"
            "Compare them carefully and return ONLY valid JSON:\n"
            "{\n"
            '  "reference_similarity": <int 0-10>,\n'
            '  "what_matches": "<features that look correct>",\n'
            '  "what_to_improve": "<specific differences — count, shape, connectors, arrangement, colour>",\n'
            '  "reference_count_match": <bool>,\n'
            '  "reference_critical_gap": "<one critical structural difference or null>"\n'
            "}\n\n"
            "reference_similarity: 10=identical, 7=clearly resembles reference, "
            "4=vaguely similar, 0=unrelated."
        )
        log.info("[reference] dual-image direct comparison")
        raw_vision = call_llm_vision(
            dual_prompt, str(ref_path), json_mode=True, image_path_b=stitched_path
        )
        result = extract_json(raw_vision) if raw_vision else {}
        if result:
            log.info(
                f"[reference] similarity={result.get('reference_similarity', '?')}/10  "
                f"count_match={result.get('reference_count_match', '?')}  "
                f"gap={result.get('reference_critical_gap', 'none')!r:.60}"
            )
            return result
        log.warning("[reference] dual-image parse failed, falling back to legacy mode")

    # ── Legacy: describe reference → text LLM compare ────────────────────────
    describe_ref_prompt = (
        f'Look at this reference image for a 3D game asset called "{asset_name}". '
        "In 3-4 sentences describe the key visual characteristics: shape, proportions, "
        "distinctive features, and colour/material."
    )
    log.info("[reference] describing reference image (legacy mode)")
    ref_description = call_llm_vision(describe_ref_prompt, str(ref_path), json_mode=False)
    if not ref_description:
        return {}

    log.info(f"[reference] reference: {ref_description[:120]!r}")

    compare_prompt = f"""Compare a 3D render description against a reference image description.

Reference "{asset_name}" description:
\"\"\"{ref_description}\"\"\"

Current render description:
\"\"\"{render_description}\"\"\"

Rate visual similarity and identify what the render is missing.
Return ONLY valid JSON:
{{
  "reference_similarity": 6,
  "what_matches": "features that look correct",
  "what_to_improve": "specific differences from reference",
  "reference_count_match": null,
  "reference_critical_gap": null
}}

reference_similarity: 10=identical, 7=clearly resembles reference, 4=vaguely similar, 0=unrelated."""

    raw = call_llm(compare_prompt, json_mode=True, system=CRITIC_SYSTEM)
    result = extract_json(raw)
    if result:
        log.info(f"[reference] similarity={result.get('reference_similarity', '?')}/10")
    return result

# ── Reference spec cache ─────────────────────────────────────────────────────

_REF_PASS_PROMPTS = {
    "count_and_identity": (
        "You are looking at a reference image for a 3D game asset.\n"
        "Count exactly how many distinct objects of the primary type are present "
        "(e.g. balloons, trees, rocks). Do not count background objects.\n"
        "Answer ONLY with valid JSON:\n"
        '{"object_type": "...", "object_count": <int>, '
        '"confidence": "high|medium|low", "notes": "one sentence"}'
    ),
    "shape_and_proportions": (
        "Describe the shape and proportions of the objects in this reference image.\n"
        "Answer ONLY with valid JSON:\n"
        '{"overall_shape": "...", "height_to_width_ratio": "e.g. roughly 1:1", '
        '"silhouette_description": "...", "dominant_axis": "vertical|horizontal|roughly_equal"}'
    ),
    "attachment_and_connection": (
        "How are the objects connected or supported? Look carefully for strings, sticks, "
        "wires, stems, or other connectors between and below the objects.\n"
        "Answer ONLY with valid JSON:\n"
        '{"connector_type": "rigid_stick|flexible_string|stem|none|other", '
        '"connector_description": "...", "all_at_same_height": <bool>, '
        '"attachment_point": "top|bottom|side|none"}'
    ),
    "spatial_arrangement": (
        "Describe the spatial arrangement of the objects in this reference image.\n"
        "Answer ONLY with valid JSON:\n"
        '{"arrangement_type": "cluster|ring|grid|scattered|single", '
        '"density": "tightly_packed|moderately_spaced|spread_out", '
        '"height_variation": "all_same|slight_variation|large_variation", '
        '"depth_variation": "flat_layer|some_depth|full_3d_cluster"}'
    ),
    "surface_and_material": (
        "Describe the surface finish and material of the objects in this reference image.\n"
        "Answer ONLY with valid JSON:\n"
        '{"surface_finish": "matte|satin|glossy|very_glossy", '
        '"texture_type": "smooth|rough|bumpy|faceted", '
        '"has_texture_map": <bool>, '
        '"color_description": "brief list of colors"}'
    ),
}


def _ref_image_checksum(ref_path: Path) -> str:
    """First-64KB SHA-256 hex of the reference image — fast change detection."""
    h = hashlib.sha256()
    h.update(ref_path.read_bytes()[:65536])
    return h.hexdigest()[:16]


def build_reference_spec(asset_name: str, ref_path: Path) -> dict:
    """
    Run five targeted vision passes against the reference image and cache results
    in brief.json["reference_spec"] keyed by image checksum. Re-runs only when
    the reference image changes. Also writes derived numeric brief fields:
      expected_cluster_density_max, expected_silhouette_hw_ratio.
    """
    brief = load_brief()
    current_checksum = _ref_image_checksum(ref_path)
    existing = brief.get("reference_spec", {})
    if existing.get("_checksum") == current_checksum and existing.get("count_and_identity"):
        log.info(f"[ref-spec] cache hit checksum={current_checksum} — skipping vision passes")
        return existing

    log.info(f"[ref-spec] building reference spec for {asset_name} (ref={ref_path.name})")
    spec: dict = {"_checksum": current_checksum, "_asset": asset_name}
    for pass_name, pass_prompt in _REF_PASS_PROMPTS.items():
        log.info(f"[ref-spec] pass: {pass_name}")
        raw = call_llm_vision(pass_prompt, str(ref_path), json_mode=True, timeout=45)
        parsed = extract_json(raw) if raw else None
        spec[pass_name] = parsed if parsed else {"_raw": (raw or "")[:300]}
        if parsed:
            log.info(f"[ref-spec]   → {str(parsed)[:120]}")
        else:
            log.warning(f"[ref-spec]   → parse failed")

    # Derive numeric thresholds for brief.json (only write if not already set manually)
    density_str = (spec.get("spatial_arrangement") or {}).get("density", "")
    density_map = {"tightly_packed": 0.45, "moderately_spaced": 0.80, "spread_out": 1.50}
    if density_str in density_map and "expected_cluster_density_max" not in brief:
        brief["expected_cluster_density_max"] = density_map[density_str]
        log.info(f"[ref-spec] set expected_cluster_density_max={density_map[density_str]} ({density_str})")

    dominant_axis = (spec.get("shape_and_proportions") or {}).get("dominant_axis", "")
    hw_map = {
        "vertical":      {"min": 1.3, "max": 2.5},
        "horizontal":    {"min": 0.3, "max": 0.75},
        "roughly_equal": {"min": 0.8, "max": 1.25},
    }
    if dominant_axis in hw_map and "expected_silhouette_hw_ratio" not in brief:
        brief["expected_silhouette_hw_ratio"] = hw_map[dominant_axis]
        log.info(f"[ref-spec] set expected_silhouette_hw_ratio={hw_map[dominant_axis]} ({dominant_axis})")

    brief["reference_spec"] = spec
    BRIEF_PATH.write_text(json.dumps(brief, indent=2))
    log.info(f"[ref-spec] spec written to brief.json")
    return spec

# ── Version helpers ───────────────────────────────────────────────────────────

def increment_version(v: str) -> str:
    return f"v{int(v.lstrip('v')) + 1}"


def candidate_path(asset_name: str, version: str) -> Path:
    return CANDIDATES_DIR / f"{asset_name}_{version}.mjs"

# ── Scoring ───────────────────────────────────────────────────────────────────

def calculate_score(metrics: dict) -> float:
    """
    Deterministic composite score 0.0–1.0.
      0.10  geometry loads without error
      0.20  triangle count in ideal range (0.10 if only acceptable, 0.02 if any)
      0.15  vertex normals present
      0.15  UV coordinates present
      0.10  renders without crash
      0.05  any pixel coverage > 2%  (basic visibility)
      0.10  uniformity ≥ 0.70 OR vision confirms recognisable shape (Phase 6)
      0.10  colour diversity > COLOR_DIVERSITY_THRESHOLD (real shading variation)
      0.05  good coverage > 10% OR reference_similarity ≥ 7 (Phase 5)
    Max: 1.00
    """
    geo      = metrics.get("geometry") or {}
    shot     = metrics.get("screenshot") or {}
    analysis = metrics.get("screenshot_analysis") or {}
    visual   = metrics.get("visual_analysis") or {}

    if not geo.get("ok"):
        return 0.0

    score = 0.10

    tris = geo.get("triangle_count", 0)
    if POLY_IDEAL_MIN <= tris <= POLY_IDEAL_MAX:
        score += 0.20
    elif POLY_MIN <= tris <= POLY_MAX:
        score += 0.10
    elif tris > 0:
        score += 0.02

    if geo.get("has_normals"):
        score += 0.15
    if geo.get("has_uvs"):
        score += 0.15

    if shot.get("ok"):
        score += 0.10

        coverage     = analysis.get("coverage") or 0.0
        color_div    = analysis.get("color_diversity") or 0.0
        uniformity   = analysis.get("coverage_uniformity") or 0.0
        recognisable = visual.get("recognisable")
        ref_sim      = visual.get("reference_similarity")

        if coverage > 0.02:
            score += 0.05

        # Prefer vision confirmation; fall back to pixel uniformity when vision absent
        if recognisable is True or (recognisable is None and uniformity >= COVERAGE_UNIFORMITY_THRESHOLD):
            score += 0.10

        if color_div > COLOR_DIVERSITY_THRESHOLD:
            score += 0.10

        # Reference similarity: progressive scoring (Phase 6)
        if ref_sim is not None and ref_sim >= 5:
            score += 0.05  # partial match
        if (ref_sim is not None and ref_sim >= 7) or (ref_sim is None and coverage > 0.10):
            score += 0.05  # strong match or good coverage fallback

    # Arrangement correctness (Phase 6): bonus when correct, penalty when wrong
    brief_for_arr = load_brief()
    exp_arr = brief_for_arr.get("expected_arrangement")
    arr_ok = geo.get("arrangement_ok")
    if exp_arr is not None and arr_ok is not None:
        if arr_ok:
            score += 0.05
        else:
            score -= 0.10

    # ── Penalties (after main score, before cap) ───────────────────────────
    spike = geo.get("spike_ratio", 0.0)
    if spike > SPIKE_RATIO_MAX:
        score -= 0.05

    # Disconnected mesh: sequential per-index noise on non-indexed geometry tears faces apart.
    # shared_edge_fraction < 0.85 → reduce score by 0.40 (fatal — visually broken)
    shared_edge = geo.get("shared_edge_fraction", 1.0)
    if shared_edge < SHARED_EDGE_MIN:
        score -= 0.40

    # Inverted normals penalty: only when pixel dark-patches AND vision both confirm
    dark_frac = analysis.get("dark_patch_fraction", 0.0)
    if dark_frac > DARK_PATCH_MAX and (metrics.get("visual_analysis") or {}).get("normal_issues") is True:
        score -= 0.05

    return round(max(0.0, min(score, 1.0)), 4)


def geo_only_score(geo: dict) -> float:
    """Fast geometry-only score for A/B comparison before rendering."""
    if not geo.get("ok"):
        return 0.0
    score = 0.10
    tris = geo.get("triangle_count", 0)
    if POLY_IDEAL_MIN <= tris <= POLY_IDEAL_MAX:
        score += 0.40
    elif POLY_MIN <= tris <= POLY_MAX:
        score += 0.20
    if geo.get("has_normals"):
        score += 0.25
    if geo.get("has_uvs"):
        score += 0.25
    return round(min(score, 1.0), 4)

# ── Production quality rubric ─────────────────────────────────────────────────

def _rgb_to_hue(r: float, g: float, b: float) -> float:
    """Convert linear RGB [0,1] to hue in degrees [0, 360)."""
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d < 0.01:
        return 0.0
    if mx == r:
        h = (60.0 * ((g - b) / d)) % 360.0
    elif mx == g:
        h = (60.0 * ((b - r) / d) + 120.0) % 360.0
    else:
        h = (60.0 * ((r - g) / d) + 240.0) % 360.0
    return h


def get_rubric(metrics: dict, asset_name: str) -> dict:
    """
    Structured production-quality rubric (100 pts across 5 categories).

    Returns:
      tier             — PRODUCTION_READY | NEEDS_POLISH | FAILED
      score_100        — 0–100 weighted composite
      categories       — {name: {score, max, status, failures}}
      critical_failures — blockers that force FAILED tier
      polish_items     — non-blocking improvement suggestions
      remediation      — single most important fix instruction for the coder
    """
    geo      = metrics.get("geometry") or {}
    shot     = metrics.get("screenshot") or {}
    analysis = metrics.get("screenshot_analysis") or {}
    visual   = metrics.get("visual_analysis") or {}

    critical_failures: list[str] = []
    polish_items:      list[str] = []

    # ── F: Reference Hard Constraints (no pts — critical_failures only) ───────
    _brief_f = load_brief()
    _ref_spec = _brief_f.get("reference_spec", {})
    _ci = _ref_spec.get("count_and_identity", {})
    _sa = _ref_spec.get("spatial_arrangement", {})
    _ac = _ref_spec.get("attachment_and_connection", {})
    _primary_name = _brief_f.get("expected_arrangement", {}).get("primary_mesh_name", "object")
    _named = geo.get("named_mesh_positions", {})
    _actual_count = len(_named.get(_primary_name, []))

    _ref_count = _ci.get("object_count")
    if _ref_count is not None and _actual_count > 0 and _actual_count != _ref_count:
        _confidence = _ci.get("confidence", "high")
        _msg = (
            f"wrong object count: {_actual_count} '{_primary_name}' meshes found, "
            f"reference shows {_ref_count}. "
            f"Create exactly {_ref_count} meshes named '{_primary_name}'."
        )
        if _confidence in ("high", "medium"):
            critical_failures.append(_msg)
        else:
            polish_items.append(_msg + " (low-confidence count — verify manually)")

    _ref_height_var = _sa.get("height_variation")
    if _ref_height_var == "all_same" and _actual_count > 1:
        _positions = _named.get(_primary_name, [])
        _ys = [p.get("y", 0.0) for p in _positions]
        if _ys and (max(_ys) - min(_ys)) > 0.55:
            critical_failures.append(
                f"height variation {max(_ys)-min(_ys):.2f} is too large — reference shows "
                "objects at roughly the same height (≤0.55 spread allowed)."
            )

    _ref_connector = _ac.get("connector_type")
    if _ref_connector == "rigid_stick" and len(_named.get("stick", [])) == 0:
        polish_items.append(
            "reference shows rigid sticks — name connector meshes 'stick' "
            "rather than 'string'."
        )

    # ── A: Geometry Integrity (35 pts) ────────────────────────────────────────
    # 5 geo loads + 15 normals + 10 uvs + 5 clean displacement = 35
    gi_score    = 0
    gi_failures = []

    if geo.get("ok"):
        gi_score += 5
        if geo.get("has_normals"):
            gi_score += 15
        else:
            nm = geo.get("meshes_missing_normals", 0)
            gi_failures.append(f"{nm} mesh(es) missing vertex normals — call geometry.computeVertexNormals()")
        if geo.get("has_uvs"):
            gi_score += 10
        else:
            um = geo.get("meshes_missing_uvs", 0)
            gi_failures.append(f"{um} mesh(es) missing UV coords — add spherical or box UV projection")
        spike = geo.get("spike_ratio", 0.0)
        shared_edge = geo.get("shared_edge_fraction", 1.0)
        if shared_edge < SHARED_EDGE_MIN:
            gi_failures.append(
                f"disconnected mesh: shared_edge_fraction={shared_edge:.3f} < {SHARED_EDGE_MIN} — "
                "sequential per-index noise tears non-indexed geometry; use position-based hash instead"
            )
            critical_failures.append(
                f"mesh torn apart (shared_edge_fraction={shared_edge:.3f}) — "
                "use posNoise(x,y,z) hash, not sequential LCG by vertex index"
            )
        elif spike <= SPIKE_RATIO_MAX:
            gi_score += 5
        else:
            gi_failures.append(
                f"spike artefact: spike_ratio={spike:.2f} > {SPIKE_RATIO_MAX} "
                f"(worst mesh: {geo.get('spike_worst_mesh', '?')}) — "
                "clamp vertex displacement to ≤30% of average mesh radius"
            )
            critical_failures.append(
                f"spike artefacts (spike_ratio={spike:.2f}) — vertex displacement is too large"
            )
    else:
        gi_failures.append(f"geometry load failed: {geo.get('error', 'unknown')}")
        critical_failures.append(
            f"geometry load error — createAsset() threw: {geo.get('error', 'unknown')}. "
            "Fix this specific JS error; the asset must return a valid THREE.Object3D with at least one Mesh."
        )

    gi_status = "FAIL" if not geo.get("ok") else ("WARN" if gi_failures else "PASS")

    # ── B: Render Quality (25 pts) ────────────────────────────────────────────
    # 10 renders + 5 coverage + 5 uniformity + 5 diversity = 25
    rq_score    = 0
    rq_failures = []

    coverage        = analysis.get("coverage") or 0.0
    uniformity      = analysis.get("coverage_uniformity") or 0.0
    color_div       = analysis.get("color_diversity") or 0.0
    dark_frac       = analysis.get("dark_patch_fraction") or 0.0
    normal_issues   = visual.get("normal_issues")
    recognisable    = visual.get("recognisable")
    ref_sim         = visual.get("reference_similarity")

    if shot.get("ok"):
        rq_score += 10
        if coverage > 0.02:
            rq_score += 5
        else:
            rq_failures.append(f"coverage {coverage:.1%} < 2% — asset near-invisible, check scale/origin")
            critical_failures.append("render coverage < 2% — asset cannot be evaluated")
        if uniformity >= COVERAGE_UNIFORMITY_THRESHOLD or recognisable is True:
            rq_score += 5
        else:
            rq_failures.append(
                f"coverage uniformity {uniformity:.2f} < {COVERAGE_UNIFORMITY_THRESHOLD}"
                + (" and vision says not recognisable" if recognisable is False else "")
            )
        if color_div > COLOR_DIVERSITY_THRESHOLD:
            rq_score += 5
        else:
            rq_failures.append(
                f"colour diversity {color_div:.1f} ≤ {COLOR_DIVERSITY_THRESHOLD} — "
                "flat shading; add more geometry detail or vertex displacement"
            )
        # Dark-patch check — WARN only unless vision also confirms normal_issues
        if dark_frac > DARK_PATCH_MAX:
            if normal_issues is True:
                rq_failures.append(
                    f"inverted normals confirmed: dark_patch_fraction={dark_frac:.3f} "
                    "and vision normal_issues=True — call computeVertexNormals() after every position change"
                )
                critical_failures.append(
                    f"inverted normals (dark_patch_fraction={dark_frac:.3f}, vision confirmed) "
                    "— computeVertexNormals() missing or called too early"
                )
            else:
                polish_items.append(
                    f"dark_patch_fraction={dark_frac:.3f} > {DARK_PATCH_MAX} "
                    "(may be dark material or shadow — check normals if shading looks wrong)"
                )
    else:
        rq_failures.append(
            f"render failed: {shot.get('render_error') or shot.get('error', 'Puppeteer crash')}"
        )
        critical_failures.append(
            f"render crash — asset threw: {shot.get('render_error') or shot.get('error', 'unknown error')}. "
            "Fix this specific JS runtime error."
        )

    # Dominant hue validation (Item 3)
    pixel_m = analysis.get("pixel_metrics") or {}
    dom_hue = pixel_m.get("dominant_hue")
    mean_sat = pixel_m.get("mean_saturation")
    exp_hsv = load_brief().get("expected_color_hsv", {})
    if dom_hue is not None and exp_hsv:
        hue_ctr = exp_hsv.get("hue_center")
        hue_tol = exp_hsv.get("hue_tolerance", 35)
        sat_min = exp_hsv.get("saturation_min", 0.0)
        if hue_ctr is not None:
            diff = abs(dom_hue - hue_ctr)
            if diff > 180:
                diff = 360 - diff
            if diff > hue_tol:
                rq_failures.append(
                    f"dominant hue {dom_hue:.0f}° is {diff:.0f}° from expected {hue_ctr}° "
                    f"(tolerance ±{hue_tol}°) — vertex colors wrong hue for this asset type"
                )
        if mean_sat is not None and sat_min > 0 and mean_sat < sat_min:
            rq_failures.append(
                f"mean saturation {mean_sat:.2f} < {sat_min} — "
                "vertex colors too grey; increase palette saturation"
            )

    rq_status = (
        "FAIL" if not shot.get("ok") or coverage <= 0.02
        else "WARN" if rq_failures
        else "PASS"
    )

    # ── C: Material Compliance (15 pts) ──────────────────────────────────────
    # 5 correct type + 5 roughness explicit + 5 metalness explicit = 15
    mc_score    = 0
    mc_failures = []

    if geo.get("ok"):
        if geo.get("material_compliant", True):
            mc_score += 5
        else:
            mc_failures.append(
                f"non-standard material: {geo.get('materials', [])} — "
                "use MeshStandardMaterial or MeshPhysicalMaterial"
            )
            critical_failures.append(
                "non-PBR material type — PBR lighting requires MeshStandardMaterial/MeshPhysicalMaterial"
            )
        if geo.get("roughness_explicit", False):
            mc_score += 5
        else:
            mc_failures.append(
                "roughness at default (1.0) — set explicitly: stone 0.75-0.95, metal 0.10-0.35, crystal 0.0-0.15"
            )
        if geo.get("metalness_explicit", False) or not geo.get("uncustomized_material", False):
            # Give points if metalness was intentionally left at 0.0 (correct for stone/wood)
            # but roughness was set. Full credit when at least roughness was configured.
            if geo.get("roughness_explicit", False):
                mc_score += 5
            else:
                polish_items.append(
                    "metalness at THREE.js default 0.0 — confirm intentional for this material type"
                )
        else:
            polish_items.append(
                "both roughness and metalness at THREE.js defaults (1.0 / 0.0) — configure material properties"
            )

        # Material property range check (Item 1)
        exp_mat = load_brief().get("expected_material_properties", {})
        r_min = exp_mat.get("roughness_min")
        r_max = exp_mat.get("roughness_max")
        r_vals = geo.get("roughness_values", [])
        if r_vals and (r_min is not None or r_max is not None):
            bad = [r for r in r_vals if
                   (r_min is not None and r < r_min) or (r_max is not None and r > r_max)]
            if bad:
                rng = f"{r_min or '?'}–{r_max or '?'}"
                mc_failures.append(
                    f"roughness values {[round(r,3) for r in bad]} outside expected range {rng} "
                    f"— adjust material roughness for this asset type"
                )
        m_min = exp_mat.get("metalness_min")
        m_max = exp_mat.get("metalness_max")
        m_vals = geo.get("metalness_values", [])
        if m_vals and (m_min is not None or m_max is not None):
            bad_m = [m for m in m_vals if
                     (m_min is not None and m < m_min) or (m_max is not None and m > m_max)]
            if bad_m:
                rng_m = f"{m_min or '?'}–{m_max or '?'}"
                mc_failures.append(
                    f"metalness values {[round(m,3) for m in bad_m]} outside expected range {rng_m} "
                    f"— adjust material metalness for this asset type"
                )

    mc_status = "FAIL" if not geo.get("material_compliant", True) else ("WARN" if mc_failures else "PASS")

    # ── D: Geometric Quality (15 pts) ────────────────────────────────────────
    # 10 polygon budget + 5 scale = 15
    gq_score    = 0
    gq_failures = []

    tris      = geo.get("triangle_count", 0)
    scale_max = geo.get("scale_max", 0.0)

    if POLY_IDEAL_MIN <= tris <= POLY_IDEAL_MAX:
        gq_score += 10
    elif POLY_MIN <= tris <= POLY_MAX:
        gq_score += 5
        gq_failures.append(
            f"triangle count {tris} is acceptable but outside ideal range "
            f"{POLY_IDEAL_MIN}–{POLY_IDEAL_MAX}"
        )
    elif tris > 0:
        gq_failures.append(
            f"triangle count {tris} outside acceptable range {POLY_MIN}–{POLY_MAX}"
        )

    if 0.3 <= scale_max <= 6.0:
        gq_score += 5
    elif scale_max > 0:
        gq_failures.append(
            f"scale_max={scale_max:.2f} outside expected 0.3–6.0 range "
            "(auto-scaled in renderer but wrong design scale)"
        )
        polish_items.append(f"adjust asset to ~1–2 units tall (current scale_max={scale_max:.2f})")

    # Proportional ratio check (Item 4)
    brief_chk = load_brief()

    # Asset-specific hard minimum (brief.json poly_min_hard) — critical blocker
    poly_min_hard = brief_chk.get("poly_min_hard")
    if poly_min_hard is not None and tris < poly_min_hard:
        critical_failures.append(
            f"triangle count {tris} below required minimum {poly_min_hard} — "
            f"increase subdivisions on body, grip, and other meshes (target: {poly_min_hard}+)"
        )
    bbox = geo.get("bounding_box", {})
    bx, by = bbox.get("x", 0.0), bbox.get("y", 0.0)
    exp_props = brief_chk.get("expected_proportions", {})
    hw_range = exp_props.get("height_to_width_ratio")
    if hw_range and bx > 0.05:
        actual_hw = by / bx
        if not (hw_range[0] <= actual_hw <= hw_range[1]):
            gq_failures.append(
                f"height/width ratio {actual_hw:.2f} outside expected {hw_range[0]:.2f}–{hw_range[1]:.2f} "
                "— check asset orientation and proportions"
            )

    # Arrangement check (Phase 3)
    brief_chk = load_brief()
    exp_arr = brief_chk.get("expected_arrangement")
    arr_ok = geo.get("arrangement_ok")
    if exp_arr is not None:
        if arr_ok is False:
            exp_desc = (
                f"expected {exp_arr.get('body_count', '?')} body mesh(es) in "
                f"{exp_arr.get('row_count', '?')} row(s) of {exp_arr.get('bodies_per_row', '?')}, "
                f"found {geo.get('arrangement_body_count', 0)} body mesh(es) in "
                f"{geo.get('arrangement_row_count', 0)} row(s)"
            )
            gq_failures.append(f"incorrect arrangement: {exp_desc}")
            critical_failures.append(
                f"wrong mesh arrangement — {exp_desc}. "
                f"Fix: {exp_arr.get('description', 'check expected_arrangement in brief.json')}"
            )
        elif arr_ok is True:
            gq_score += 5  # bonus 5 pts for correct arrangement (Category D max goes 15→20)

    # Mesh type check (Phase 5 reduced)
    exp_types = brief_chk.get("expected_mesh_types", {})
    if exp_types and geo.get("ok"):
        actual_names = set(geo.get("mesh_names", []))
        for expected_name in exp_types:
            if expected_name not in actual_names:
                polish_items.append(
                    f"mesh '{expected_name}' not found in geometry — "
                    f"expected type '{exp_types[expected_name]}'; "
                    "add this mesh or rename an existing one for correct Realistic material assignment"
                )

    # Per-mesh colour accuracy check
    exp_mesh_colors = brief_chk.get("expected_mesh_colors", {})
    mesh_color_means = geo.get("mesh_color_means", {})
    if exp_mesh_colors and mesh_color_means:
        for mesh_nm, color_spec in exp_mesh_colors.items():
            if mesh_nm not in mesh_color_means:
                continue
            mc = mesh_color_means[mesh_nm]
            hue = _rgb_to_hue(mc["r"], mc["g"], mc["b"])
            hue_min = color_spec.get("hue_min")
            hue_max = color_spec.get("hue_max")
            dom_not = color_spec.get("dominant_not")  # [lo, hi] forbidden hue range
            label = color_spec.get("label", "correct hue")
            if hue_min is not None and hue_max is not None:
                # Handle wrap-around (e.g. hue_min=340, hue_max=20 for red)
                if hue_min <= hue_max:
                    in_range = hue_min <= hue <= hue_max
                else:
                    in_range = hue >= hue_min or hue <= hue_max
                if not in_range:
                    gq_failures.append(
                        f"mesh '{mesh_nm}' has wrong colour: hue={hue:.0f}° but expected "
                        f"{hue_min}°–{hue_max}° ({label}) — "
                        f"set '{mesh_nm}' vertex colours to {label} palette"
                    )
            if dom_not is not None and len(dom_not) == 2:
                lo, hi = dom_not[0], dom_not[1]
                if lo <= hue <= hi:
                    gq_failures.append(
                        f"mesh '{mesh_nm}' is in forbidden hue range {lo}°–{hi}°: "
                        f"hue={hue:.0f}° — expected {label}, not this hue"
                    )

    # Cluster density check (vision enhancement Item 4)
    density_max = brief_chk.get("expected_cluster_density_max")
    measured_density = geo.get("cluster_density_mean")
    if density_max is not None and measured_density is not None and measured_density > density_max:
        ref_density_str = (
            (brief_chk.get("reference_spec") or {})
            .get("spatial_arrangement", {})
            .get("density", "tightly_packed")
        )
        primary_nm = brief_chk.get("expected_arrangement", {}).get("primary_mesh_name", "object")
        gq_failures.append(
            f"cluster too spread out: cluster_density_mean={measured_density:.3f} > "
            f"max {density_max:.3f} — pack '{primary_nm}'s closer together. "
            f"Reference shows '{ref_density_str}' density."
        )

    # Silhouette H:W ratio check (vision enhancement Item 7)
    analysis_for_gq = metrics.get("screenshot_analysis") or {}
    pixel_metrics = analysis_for_gq.get("pixel_metrics") or {}
    exp_sil = brief_chk.get("expected_silhouette_hw_ratio")
    measured_hw = pixel_metrics.get("silhouette_hw_ratio")
    if exp_sil and measured_hw is not None:
        hw_min = exp_sil.get("min", 0.0)
        hw_max = exp_sil.get("max", 99.0)
        if not (hw_min <= measured_hw <= hw_max):
            gq_failures.append(
                f"silhouette H:W ratio {measured_hw:.2f} outside expected "
                f"{hw_min:.2f}–{hw_max:.2f} — "
                "asset proportions do not match reference shape."
            )

    # Pixel shape metrics (Phase 2)
    pixel_targets = brief_chk.get("pixel_targets") or {}
    arc = pixel_metrics.get("arc_depth_ratio")
    tgt_arc = pixel_targets.get("arc_depth_ratio_min")
    if arc is not None and tgt_arc is not None and arc < tgt_arc:
        gq_failures.append(
            f"arc_depth_ratio={arc:.3f} < {tgt_arc} — curvature too shallow; increase yDip parameter"
        )
    asym = pixel_metrics.get("coverage_asymmetry")
    tgt_asym = pixel_targets.get("coverage_asymmetry_max")
    if asym is not None and tgt_asym is not None and asym > tgt_asym:
        polish_items.append(
            f"coverage_asymmetry={asym:.3f} > {tgt_asym} — asset may be skewed or off-centre"
        )

    # Specular highlight check — low highlight_fraction means material looks matte
    hl = pixel_metrics.get("highlight_fraction")
    tgt_hl = pixel_targets.get("highlight_fraction_min")
    if hl is not None and tgt_hl is not None and hl < tgt_hl:
        polish_items.append(
            f"highlight_fraction={hl:.4f} < {tgt_hl} — surface looks matte; "
            "reduce roughness (target ≤0.25 for shiny materials like balloons)"
        )

    # Item 1 — Reference pixel similarity (hue histogram intersection)
    rps = pixel_metrics.get("ref_pixel_sim")
    tgt_rps = brief_chk.get("expected_ref_pixel_sim_min")
    if rps is not None and tgt_rps is not None and rps < tgt_rps:
        polish_items.append(
            f"ref_pixel_sim={rps:.3f} < {tgt_rps:.2f} — colour distribution does not match "
            "the reference photo; check hue palette matches reference colours"
        )

    # Item 2 — Palette colour count
    phc = pixel_metrics.get("palette_distinct_hues")
    tgt_phc = brief_chk.get("expected_palette_hues_min")
    if phc is not None and tgt_phc is not None and phc < tgt_phc:
        gq_failures.append(
            f"palette_distinct_hues={phc} < {tgt_phc} — too few distinct colours; "
            "each object should have a unique colour from a diverse palette"
        )

    # Item 3 — View consistency across angles
    vc = pixel_metrics.get("view_consistency_score")
    if vc is not None:
        if vc < 0.40:
            gq_failures.append(
                f"view_consistency_score={vc:.3f} — large coverage variation across render angles; "
                "asset may have missing geometry or cracks visible from some directions"
            )
        elif vc < 0.65:
            polish_items.append(
                f"view_consistency_score={vc:.3f} — moderate coverage variation across angles; "
                "check that the asset looks similar from all directions"
            )

    gq_status = "FAIL" if tris == 0 else ("WARN" if gq_failures else "PASS")

    # ── E: Visual / Code Quality (10 pts) ────────────────────────────────────
    # 5 recognisable + 5 normals-ok from vision = 10
    vq_score    = 0
    vq_failures = []

    if recognisable is True:
        vq_score += 5
    elif recognisable is False:
        vq_failures.append(f"vision says asset not recognisable as {asset_name!r}")
    # None = vision not run yet — no penalty, no points

    _dark_frac = analysis.get("dark_patch_fraction", 0.0)
    if normal_issues is False:
        vq_score += 5
    elif normal_issues is True:
        # Only escalate to a rubric failure when geometry evidence agrees.
        # Vision often flags shadow darkening on low-poly faceted rocks as "normal issues";
        # this is a false positive when dark_patch_fraction is within threshold AND
        # geometry has valid normals — in that case downgrade to a polish hint.
        if _dark_frac > DARK_PATCH_MAX or not geo.get("has_normals"):
            vq_failures.append("vision confirms normal artefacts (dark/inside-out faces)")
        else:
            polish_items.append(
                "vision flagged possible normal variation — geometry normals are valid and "
                "dark_patch_fraction is within threshold, likely directional-light shadow on "
                "low-poly faces rather than inverted normals; no code change needed"
            )
    # None = vision not run — no penalty

    # Feature checklist — visual fidelity check against brief.json requirements
    feature_check = visual.get("feature_check", {})
    feat_specs = brief_chk.get("feature_checklist", {})
    for feat_id in brief_chk.get("feature_checklist_required", []):
        if feature_check.get(feat_id) is False:
            desc = feat_specs.get(feat_id, feat_id)
            critical_failures.append(f"required visual feature absent: {feat_id!r} — {desc}")
        elif feature_check.get(feat_id) is None and feature_check:
            # feature_check ran but this key is missing → treat as absent
            desc = feat_specs.get(feat_id, feat_id)
            polish_items.append(f"feature check inconclusive: {feat_id!r} — {desc}")

    # Code complexity guard (Item 2)
    code_lines = metrics.get("code_line_count", 0)
    if code_lines > 500:
        vq_failures.append(
            f"code too long: {code_lines} lines — LLM rewrote the whole file; "
            "make MINIMAL targeted changes (add/modify 5–30 lines, not 100+)"
        )
    elif code_lines > 250:
        polish_items.append(
            f"code length {code_lines} lines — prefer shorter focused implementations; "
            "avoid rewriting unrelated sections"
        )

    vq_status = "FAIL" if vq_failures else ("PASS" if vq_score == 10 else "WARN")

    # ── Aggregate ─────────────────────────────────────────────────────────────

    score_100 = max(0, min(100, gi_score + rq_score + mc_score + gq_score + vq_score))

    if critical_failures or score_100 < 60:
        tier = TIER_FAILED
    elif score_100 >= 85:
        tier = TIER_PRODUCTION
    else:
        tier = TIER_POLISH

    # Reference similarity gate — demote to POLISH when shape doesn't match reference photo
    if ref_sim is not None and ref_sim < 6:
        what_to_improve = visual.get("what_to_improve") or "shape, proportions, and distinctive features"
        ref_item = (
            f"reference_similarity={ref_sim}/10 < 6 — shape must more closely match the reference photo. "
            f"Specific gaps: {what_to_improve}"
        )
        # Insert after any vertex-color requirement (which is a mandatory brief constraint)
        insert_pos = 1 if polish_items and "vertex color" in polish_items[0] else 0
        polish_items.insert(insert_pos, ref_item)
        if tier == TIER_PRODUCTION:
            tier = TIER_POLISH

    # Brief-driven texture requirements — demote to POLISH so loop keeps iterating
    brief = load_brief()
    if (
        brief.get("texture_style") == "vertex-color"
        and geo.get("ok")
        and not geo.get("has_vertex_colors")
        and tier == TIER_PRODUCTION
    ):
        polish_items.insert(0,
            "vertex color texture required by brief (texture_style='vertex-color') — "
            "compute per-vertex RGB from vertex 3D position using hash-based FBM noise, "
            "map noise value to an earthy palette (dark-shadow → mid-rock → sandy-highlight), "
            "then: geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); "
            "material.vertexColors = true"
        )
        tier = TIER_POLISH  # prevent loop from stopping until vertex colours are present

    # Remediation: first critical blocker → first category failure → first polish item
    all_issues = (
        critical_failures
        + gi_failures + rq_failures + mc_failures + gq_failures + vq_failures
        + polish_items
    )
    remediation = all_issues[0] if all_issues else (
        "all categories pass — polish visual fidelity and match reference/description"
    )

    return {
        "tier":             tier,
        "score_100":        score_100,
        "categories": {
            "geometry_integrity":  {"score": gi_score, "max": 35, "status": gi_status, "failures": gi_failures},
            "render_quality":      {"score": rq_score, "max": 25, "status": rq_status, "failures": rq_failures},
            "material_compliance": {"score": mc_score, "max": 15, "status": mc_status, "failures": mc_failures},
            "geometric_quality":   {"score": gq_score, "max": 15, "status": gq_status, "failures": gq_failures},
            "visual_quality":      {"score": vq_score, "max": 10, "status": vq_status, "failures": vq_failures},
        },
        "critical_failures": critical_failures,
        "polish_items":      polish_items,
        "remediation":       remediation,
    }

# ── Trust score (adapted from auto_game) ──────────────────────────────────────

def compute_trust(history: list) -> float:
    """
    0–100 composite trust:
      impl_rate     × 40   (fraction of iterations that produced a valid candidate)
      score_trend   × 30   (avg last-3 vs prior-3, normalised)
      stability     × 30   (1 - reversion_rate over last 20)
    """
    if len(history) < 3:
        return 50.0

    recent = history[-20:]
    impl_rate = sum(1 for h in recent if h.get("outcome") == "accepted") / len(recent)

    scores = [h.get("score", 0) for h in history if "score" in h]
    if len(scores) >= 6:
        trend = (sum(scores[-3:]) / 3 - sum(scores[-6:-3]) / 3) / MAX_SCORE_DELTA
        trend = max(-1.0, min(1.0, trend))
    else:
        trend = 0.0

    reversion_rate = sum(1 for h in recent if h.get("outcome") == "reverted") / len(recent)
    stability = 1.0 - reversion_rate

    trust = impl_rate * 40 + ((trend + 1) / 2) * 30 + stability * 30
    return round(min(100.0, max(0.0, trust)), 1)

# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_geo(geo: dict) -> str:
    if not geo or not geo.get("ok"):
        return f"FAILED: {(geo or {}).get('error', 'no data')}"
    bb = geo.get("bounding_box", {})
    return (
        f"tri={geo.get('triangle_count', 0)}  "
        f"vert={geo.get('vertex_count', 0)}  "
        f"normals={'yes' if geo.get('has_normals') else 'NO'}  "
        f"uvs={'yes' if geo.get('has_uvs') else 'NO'}  "
        f"mats={geo.get('materials', [])}  "
        f"connectivity={geo.get('shared_edge_fraction', '?')}  "
        f"bbox={bb.get('x', 0):.2f}×{bb.get('y', 0):.2f}×{bb.get('z', 0):.2f}"
    )


def fmt_screenshot(screenshot: dict, analysis: dict | None) -> str:
    if not screenshot:
        return "not run"
    if not screenshot.get("ok"):
        return f"FAILED: {screenshot.get('error') or screenshot.get('render_error', 'unknown')}"
    if analysis:
        unif = analysis.get('coverage_uniformity')
        return (
            f"ok  cov={analysis.get('coverage', 0):.1%}  "
            f"div={analysis.get('color_diversity', 0):.1f}  "
            f"uniformity={unif:.2f}" if unif is not None else ""
        ).rstrip()
    return "ok"


def history_summary(history: list, max_entries: int = 3) -> str:
    if not history:
        return "  (none)"
    lines = []
    for h in history[-max_entries:]:
        lines.append(
            f"  {h.get('version', '?')} score={h.get('score', 0):.3f} "
            f"[{h.get('outcome', '?')}] "
            f"plan={h.get('plan', {}).get('title', '?')!r}"
        )
    return "\n".join(lines)

# ── JSON + JS extraction ──────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    log.warning(f"Could not extract JSON from LLM response ({len(text)} chars)")
    return {}


def extract_js(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:javascript|js)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    return text

# ── Validate generated JS ─────────────────────────────────────────────────────

def validate_js_basic(js: str, forbidden_patterns: list[str] | None = None) -> tuple[bool, str]:
    if not js or len(js) < 50:
        return False, FAIL_GEN
    if "export function createAsset" not in js:
        return False, FAIL_SYNTAX
    if "THREE" not in js or "import" not in js.lower():
        return False, FAIL_SYNTAX
    if js.count("{") < js.count("}") - 3 or js.count("{") > js.count("}") + 8:
        return False, FAIL_SYNTAX
    line_count = len(js.splitlines())
    if line_count > 500:
        log.warning(f"[validate_js] code very long: {line_count} lines — over-engineering risk")
    if forbidden_patterns:
        for pattern in forbidden_patterns:
            if pattern in js:
                log.warning(f"[validate_js] forbidden pattern found: {pattern!r}")
                return False, FAIL_SYNTAX
    return True, "ok"


def _banned_js_patterns(steering: dict) -> list[str]:
    """Extract literal JS code patterns to block from the forbidden_improvements list."""
    patterns = []
    for f in steering.get("forbidden_improvements", []):
        if "group.rotation" in f:
            patterns.append("group.rotation")
    return patterns

# ── Episodic memory ───────────────────────────────────────────────────────────

def write_episodic(
    iteration: int,
    asset_name: str,
    version: str,
    metrics: dict,
    score: float,
    plan: dict,
    outcome: str,
) -> None:
    """Append one line to episodic.jsonl — one record per iteration."""
    geo = metrics.get("geometry") or {}
    analysis = metrics.get("screenshot_analysis") or {}
    js_loc = 0
    p = candidate_path(asset_name, version)
    if p.exists():
        js_loc = len(p.read_text().splitlines())

    record = [
        iteration,
        datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        asset_name,
        version,
        plan.get("improvement_type", "?"),
        plan.get("title", "?"),
        geo.get("ok", False),
        geo.get("triangle_count", 0),
        geo.get("has_normals", False),
        geo.get("has_uvs", False),
        (metrics.get("screenshot") or {}).get("ok", False),
        analysis.get("coverage"),
        score,
        outcome,
        js_loc,
    ]
    with EPISODIC_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

# ── Episodic feedback ────────────────────────────────────────────────────────

def read_episodic_summary(n: int = 10) -> str:
    """
    Read the last n episodic records and summarise which improvement_types
    led to score increases vs. regressions. Returns a short text block for the planner.
    """
    if not EPISODIC_PATH.exists():
        return ""
    lines = EPISODIC_PATH.read_text().strip().splitlines()
    recent = []
    for line in lines[-n:]:
        try:
            r = json.loads(line)
            # record = [iter, ts, name, version, imp_type, title, geo_ok, tris, nrm, uv, sc_ok, cov, score, outcome, loc]
            if len(r) >= 14:
                recent.append({"type": r[4], "title": r[5], "score": r[12], "outcome": r[13]})
        except Exception:
            pass

    if not recent:
        return ""

    from collections import defaultdict
    by_type: dict = defaultdict(list)
    for rec in recent:
        by_type[rec["type"]].append(rec)

    lines_out = ["EPISODIC MEMORY (last iterations):"]
    for imp_type, recs in by_type.items():
        accepted = [r for r in recs if r["outcome"] == "accepted"]
        regressed = [r for r in recs if r["outcome"] in ("regression", "reverted")]
        avg_score = sum(r["score"] for r in recs) / len(recs)
        lines_out.append(
            f"  {imp_type}: {len(accepted)} improved / {len(regressed)} regressed  avg_score={avg_score:.3f}"
        )
        for r in recs[-2:]:
            lines_out.append(f"    [{r['outcome']}] {r['title']!r} score={r['score']:.3f}")

    return "\n".join(lines_out)

# ── Status.txt (human-readable) ───────────────────────────────────────────────

def write_status(
    state: dict, metrics: dict, score: float, critique: dict, rubric: dict | None = None
) -> None:
    geo      = metrics.get("geometry") or {}
    analysis = metrics.get("screenshot_analysis") or {}
    version  = state.get("current_version", "?")
    best     = state.get("best_version", "?")
    iteration = state.get("iteration", 0)
    trust     = state.get("trust_score", 50)
    regress   = state.get("consecutive_regressions", 0)

    tier_line = ""
    if rubric:
        tier      = rubric.get("tier", "?")
        s100      = rubric.get("score_100", "?")
        blockers  = rubric.get("critical_failures", [])
        tier_line = f"Rubric: {tier}  ({s100}/100)"
        if blockers:
            tier_line += f"  BLOCKERS: {blockers[0]}"

    lines = [
        f"=== threejs_loop — {state['asset_name']} ===",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Iteration {iteration} / {MAX_ITERATIONS}  |  Best: {best}  |  Best score: {state.get('best_score', 0):.3f}",
        f"Current: {version}  |  Score: {score:.3f}  |  Trust: {trust:.0f}/100",
    ]
    if tier_line:
        lines.append(tier_line)
    lines += [
        "",
        f"Geometry: tri={geo.get('triangle_count', '?')}  "
        f"normals={'yes' if geo.get('has_normals') else 'NO'}  "
        f"uvs={'yes' if geo.get('has_uvs') else 'NO'}  "
        f"spike={geo.get('spike_ratio', '?')}  "
        f"connectivity={geo.get('shared_edge_fraction', '?')}",
        f"Screenshot: cov={analysis.get('coverage') or 0:.1%}  "
        f"div={analysis.get('color_diversity') or 0:.1f}  "
        f"dark={analysis.get('dark_patch_fraction') or 0:.3f}",
        "",
        f"Last critique: {critique.get('quality_summary', '(none)')}",
        f"Top priority: {critique.get('top_priority', '(none)')}",
        "",
    ]
    if regress > 0:
        lines.append(f"REGRESSIONS: {regress}/{REVERT_AFTER_N_REGRESSIONS} consecutive")
        if regress >= REVERT_AFTER_N_REGRESSIONS - 1:
            lines.append(f"WARNING: next regression will revert to {best}")
    else:
        lines.append("Status: ITERATING")

    STATUS_PATH.write_text("\n".join(lines))

# ── Agent: initial generation ─────────────────────────────────────────────────

def generate_initial_asset(asset_name: str, asset_description: str) -> str:
    rules_block = _build_rules_block(load_rules())
    brief_block = _brief_block(load_brief())

    prompt = f"""You are a Three.js expert generating a game-ready 3D asset.

Asset name: {asset_name}
Description: {asset_description}

{brief_block}

PERMANENT RULES (must follow):
{rules_block}

Write a complete ES module with these exact requirements:
1. First line: import * as THREE from 'three';
2. Export exactly one named function: export function createAsset() {{ ... }}
3. createAsset() must return a THREE.Object3D (Mesh or Group)
4. Target 300–800 triangles (game-optimised — no unnecessary subdivision)
5. Call geometry.computeVertexNormals() after any vertex position changes
6. Include UV coordinates on every mesh
7. Use THREE.MeshStandardMaterial (roughness + metalness values)
8. Asset centred near origin, approximately 1–2 units tall

Write ONLY the JavaScript. No markdown fences, no explanation."""

    log.info(f"[initial-gen] generating {asset_name} from scratch")
    return extract_js(call_llm_coder(prompt))

# ── Agent: Visual analysis (OpenRouter vision) ────────────────────────────────

VISUAL_SYSTEM = (
    "You are a 3D game asset visual quality expert. "
    "Return only valid JSON with no prose or markdown."
)

def run_feature_checklist(render_path: str, checklist: dict) -> dict:
    """
    Ask the vision model yes/no questions about specific visual features.
    Returns {feature_id: bool} for each question. Used to gate PRODUCTION_READY tier.
    """
    if not checklist:
        return {}
    prompt = (
        "Look at this 3D game asset render (four angles: quarter, front, right side, top-down).\n"
        "For each feature below, answer true if clearly visible, false if absent or unclear.\n"
        "Return ONLY valid JSON with boolean values (true/false), no prose:\n{\n"
        + "\n".join(f'  "{k}": <true|false>  // {v}' for k, v in checklist.items())
        + "\n}"
    )
    raw = call_llm_vision(prompt, render_path, json_mode=True)
    result = extract_json(raw) if raw else {}
    # Normalise any string booleans
    for k, v in list(result.items()):
        if isinstance(v, str):
            result[k] = v.lower() in ("true", "yes", "1")
    log.info(f"[feature-check] {result}")
    return result


def run_visual_analysis(asset_name: str, stitched_path: str, geo_count: int | None = None) -> dict:
    """
    Describe-then-structure pattern:
      Step 1 — vision model describes what it sees (prose). Vision models are
               reliable at description but unreliable at complex JSON schemas.
      Step 2 — text model (call_llm) converts the prose into structured JSON.

    Grid layout: [TOP-LEFT: quarter] [TOP-RIGHT: front]
                 [BOTTOM-LEFT: right] [BOTTOM-RIGHT: top-down]
    Returns visual_critique dict, or {} if vision is unavailable.
    """
    describe_prompt = f"""Look at these four renders of a 3D game asset called "{asset_name}".

The image shows the same object from four angles:
  TOP-LEFT: 3/4 quarter view  |  TOP-RIGHT: front
  BOTTOM-LEFT: right side     |  BOTTOM-RIGHT: top-down

In 4-6 sentences describe:
1. The shape — does it resemble a {asset_name}?
2. Any dark holes, black patches, or see-through faces? (suggests flipped normals)
3. Is the shading physically plausible, or flat/overlit?
4. Any texture stretching, seams, or UV distortion visible?
5. Does any angle look broken or very different from the others?"""

    log.info("[vision] describing 4-angle render grid")
    description = call_llm_vision(describe_prompt, stitched_path, json_mode=False)
    if not description:
        return {}

    log.info(f"[vision] description: {description[:160]!r}")

    structure_prompt = f"""A visual reviewer described this 3D "{asset_name}" asset:

\"\"\"{description}\"\"\"

Extract a structured JSON critique from this description.
Return ONLY valid JSON:
{{
  "shading_quality": "good",
  "normal_issues": false,
  "uv_artifacts": false,
  "recognisable": true,
  "broken_angles": [],
  "visual_issues": [{{"severity": "minor", "description": "example issue"}}],
  "visual_strengths": ["example strength"],
  "visual_summary": "one sentence summary",
  "top_visual_priority": "most important visual fix"
}}"""

    raw = call_llm(structure_prompt, json_mode=True, system=CRITIC_SYSTEM)
    result = extract_json(raw)
    if result:
        result["_vision_description"] = description[:400]
        log.info(
            f"[vision] shading={result.get('shading_quality', '?')}  "
            f"normals_ok={not result.get('normal_issues', True)}  "
            f"recognisable={result.get('recognisable', '?')}"
        )

        # Reference comparison: dual-image direct compare (Item 6) with legacy fallback
        ref_path = find_reference(asset_name)
        if ref_path:
            ref_cmp = run_reference_comparison(
                asset_name, ref_path, description,
                stitched_path=stitched_path, geo_count=geo_count,
            )
            result["reference_similarity"]   = ref_cmp.get("reference_similarity")
            result["reference_notes"]        = ref_cmp.get("what_to_improve")
            result["reference_count_match"]  = ref_cmp.get("reference_count_match")
            result["reference_critical_gap"] = ref_cmp.get("reference_critical_gap")

        # Feature checklist — per-feature visual fidelity check
        checklist = load_brief().get("feature_checklist", {})
        if checklist:
            result["feature_check"] = run_feature_checklist(stitched_path, checklist)

    return result

# ── Agent: Critic ─────────────────────────────────────────────────────────────

def _format_reference_spec_block(reference_spec: dict) -> str:
    """Format reference_spec as a structured text block for critic/planner prompts."""
    if not reference_spec:
        return ""
    ci = reference_spec.get("count_and_identity", {})
    sa = reference_spec.get("spatial_arrangement", {})
    ac = reference_spec.get("attachment_and_connection", {})
    sm = reference_spec.get("surface_and_material", {})
    sh = reference_spec.get("shape_and_proportions", {})
    lines = ["REFERENCE SPEC (ground truth from reference image — do NOT hallucinate):"]
    if ci.get("object_count"):
        lines.append(f"  Object count: {ci['object_count']} {ci.get('object_type', 'objects')}")
    if sh.get("overall_shape"):
        lines.append(f"  Shape: {sh['overall_shape']}")
    if sh.get("silhouette_description"):
        lines.append(f"  Silhouette: {sh['silhouette_description']}")
    if ac.get("connector_type"):
        lines.append(f"  Connectors: {ac['connector_type']} — {ac.get('connector_description', '')}")
    if ac.get("all_at_same_height") is not None:
        lines.append(f"  All at same height: {ac['all_at_same_height']}")
    if sa.get("arrangement_type"):
        lines.append(f"  Arrangement: {sa['arrangement_type']}, density={sa.get('density', '?')}")
    if sa.get("height_variation"):
        lines.append(f"  Height variation: {sa['height_variation']}")
    if sm.get("surface_finish"):
        lines.append(f"  Surface: {sm['surface_finish']} / {sm.get('texture_type', '?')}")
    if sm.get("has_texture_map") is not None:
        lines.append(
            f"  Texture map: {'YES' if sm['has_texture_map'] else 'NO — vertex colors only'}"
        )
    return "\n".join(lines) + "\n"


CRITIC_SYSTEM = (
    "You are a technical 3D asset quality critic. "
    "Return only valid JSON with no prose or markdown."
)

def run_critic(
    asset_name: str,
    metrics: dict,
    visual: dict,
    history: list,
    iteration: int,
    rubric: dict | None = None,
) -> dict:
    """LLM critic evaluates asset quality. Returns critique dict."""
    geo      = metrics.get("geometry") or {}
    screenshot = metrics.get("screenshot") or {}
    analysis = metrics.get("screenshot_analysis") or {}

    asset_path = Path(metrics.get("asset_path", ""))
    code_preview = ""
    if asset_path.exists():
        code_preview = "\n".join(asset_path.read_text().splitlines()[:60])

    # Merge visual observations into the prompt
    visual_block = ""
    if visual:
        visual_block = f"""
VISUAL ANALYSIS (from 4-angle render grid):
{json.dumps(visual, indent=2)}
"""

    # Rubric block: surface only failing categories so the critic focuses its effort
    rubric_block = ""
    if rubric:
        tier       = rubric.get("tier", "?")
        score_100  = rubric.get("score_100", "?")
        failing    = [
            f"  [{cat}] {data['status']} ({data['score']}/{data['max']}) — "
            + (data["failures"][0] if data["failures"] else "no specific failure")
            for cat, data in rubric.get("categories", {}).items()
            if data.get("status") in ("FAIL", "WARN")
        ]
        if failing or rubric.get("critical_failures"):
            rubric_block = (
                f"\nPRODUCTION RUBRIC (tier={tier}, score={score_100}/100):\n"
                + "\n".join(failing)
                + ("\nBLOCKERS: " + "; ".join(rubric["critical_failures"][:2]) if rubric.get("critical_failures") else "")
                + "\n"
            )

    brief = load_brief()
    style_notes = ""
    rs = brief.get("render_style", "")
    if rs == "low-poly":
        _ms = brief.get("material_style", "")
        _exp_mat = brief.get("expected_material_properties", {})
        _r_min = _exp_mat.get("roughness_min")
        _r_max = _exp_mat.get("roughness_max")
        _mat_line = (
            f"- Material: check roughness {_r_min}–{_r_max}, metalness 0.0–0.05"
            if _r_min is not None and _r_max is not None
            else "- Material: check roughness 0.75–0.95, metalness 0.0–0.05 (unless metal/crystal)"
        )
        if _ms:
            _mat_line += f"\n- Brief material style: {_ms}"
        style_notes = f"""
STYLE TARGET — low-poly stylised:
- Evaluate for FACETED appearance: flat faces, visible polygon edges, no smooth normals
- Penalise over-subdivision: more than 800 triangles for a simple prop is too dense
- Penalise over-smoothing: if all faces look rounded/organic, the low-poly character is lost
- Reward: readable silhouette, bold proportions, hand-crafted feel
{_mat_line}
"""
    elif rs == "pbr":
        style_notes = """
STYLE TARGET — PBR realistic:
- Evaluate for physically plausible materials (roughness, metalness set correctly)
- High detail is appropriate; penalise visibly low polygon count
- Reward: natural surface variation, correct material response to lighting
"""

    # Phase 2: quantitative pixel metrics
    pixel_metrics_block = ""
    pixel_metrics = (metrics.get("screenshot_analysis") or {}).get("pixel_metrics") or {}
    if pixel_metrics:
        brief_pm = load_brief()
        pixel_targets = brief_pm.get("pixel_targets") or {}
        pm_lines = ["PIXEL SHAPE MEASUREMENTS (quantitative):"]
        arc = pixel_metrics.get("arc_depth_ratio")
        if arc is not None:
            tgt = pixel_targets.get("arc_depth_ratio_min", 0.12)
            status = "OK" if arc >= tgt else f"LOW (target ≥{tgt}) — increase yDip"
            pm_lines.append(f"  arc_depth_ratio = {arc:.3f}  {status}")
        asym = pixel_metrics.get("coverage_asymmetry")
        if asym is not None:
            tgt = pixel_targets.get("coverage_asymmetry_max", 0.30)
            status = "OK" if asym <= tgt else f"HIGH (target ≤{tgt}) — asset may be skewed"
            pm_lines.append(f"  coverage_asymmetry = {asym:.3f}  {status}")
        czb = pixel_metrics.get("color_zone_boundary")
        if czb is not None:
            tgt_min = pixel_targets.get("color_zone_boundary_min", 0.05)
            tgt_max = pixel_targets.get("color_zone_boundary_max", 0.30)
            if tgt_min <= czb <= tgt_max:
                status = "OK"
            elif czb < tgt_min:
                status = f"TOO NARROW (target {tgt_min:.2f}–{tgt_max:.2f}) — green zone missing"
            else:
                status = f"TOO WIDE (target {tgt_min:.2f}–{tgt_max:.2f}) — green zone too large"
            pm_lines.append(f"  color_zone_boundary = {czb:.3f}  {status}")
        hl = pixel_metrics.get("highlight_fraction")
        if hl is not None:
            tgt = pixel_targets.get("highlight_fraction_min")
            if tgt is not None:
                status = "OK" if hl >= tgt else f"LOW (target ≥{tgt}) — surface looks matte; reduce roughness"
            else:
                status = "(no target set — informational)"
            pm_lines.append(f"  highlight_fraction = {hl:.4f}  {status}")
        rps = pixel_metrics.get("ref_pixel_sim")
        if rps is not None:
            tgt = (load_brief().get("expected_ref_pixel_sim_min") or 0)
            status = "OK" if rps >= tgt else f"LOW (target ≥{tgt:.2f}) — colours differ from reference"
            pm_lines.append(f"  ref_pixel_sim      = {rps:.3f}  {status}")
        phc = pixel_metrics.get("palette_distinct_hues")
        if phc is not None:
            tgt = load_brief().get("expected_palette_hues_min")
            status = ("OK" if tgt is None or phc >= tgt
                      else f"LOW (target ≥{tgt}) — add more distinct colours to the palette")
            pm_lines.append(f"  palette_hues       = {phc}      {status}")
        vc = pixel_metrics.get("view_consistency_score")
        if vc is not None:
            ac = pixel_metrics.get("angle_coverages", {})
            ac_str = "  ".join(f"{k}:{v:.2f}" for k, v in ac.items())
            status = "OK" if vc >= 0.65 else ("WARN" if vc >= 0.40 else "FAIL — geometry missing from some angles")
            pm_lines.append(f"  view_consistency   = {vc:.3f}  {status}  [{ac_str}]")
        pixel_metrics_block = "\n".join(pm_lines) + "\n"

    # Phase 1: camera basis
    cam_block = _camera_basis_block(load_brief())

    # Reference spec block (Item 5)
    ref_spec_block = _format_reference_spec_block(load_brief().get("reference_spec", {}))
    if visual.get("reference_critical_gap"):
        ref_spec_block += f"\nREFERENCE CRITICAL GAP: {visual['reference_critical_gap']}\n"

    # Suppress normals false positive when dark_patch_fraction is within threshold.
    # Vision frequently flags directional-light shadow on low-poly faces as "normal issues".
    _dark_frac_crit = analysis.get("dark_patch_fraction", 0.0)
    _has_normals_crit = geo.get("has_normals", True)
    normals_fp_note = ""
    if (
        visual.get("normal_issues") is True
        and _dark_frac_crit <= DARK_PATCH_MAX
        and _has_normals_crit
    ):
        normals_fp_note = (
            f"\nNOTE: normal_issues=True with dark_patch_fraction={_dark_frac_crit:.3f} ≤ {DARK_PATCH_MAX} "
            "is a KNOWN FALSE POSITIVE — directional-light shadow on low-poly faces, NOT inverted normals. "
            "Do NOT flag this as a normals issue and do NOT set top_priority to anything normals-related. "
            "Geometry validator confirms normals are present and valid.\n"
        )

    prompt = f"""Evaluate this Three.js game asset and produce a JSON critique.

ASSET: {asset_name}  (iteration {iteration})
{cam_block}
{style_notes}{rubric_block}{ref_spec_block}{pixel_metrics_block}
GEOMETRY METRICS:
{json.dumps(geo, indent=2)}

SCREENSHOT ANALYSIS (pixel statistics over 4-angle grid):
{json.dumps(analysis, indent=2)}
Render succeeded: {screenshot.get('ok', False)}
Console errors: {screenshot.get('console_errors', [])}
{visual_block}{normals_fp_note}
ASSET CODE (first 60 lines):
{code_preview}

PREVIOUS ITERATIONS:
{history_summary(history)}

Evaluate against:
- Polygon count: ideal 200–800 triangles for a low-poly game prop
- Normals: required for correct lighting (flat or smooth depending on style)
- UVs: required for texturing
- Material: MeshStandardMaterial with explicit roughness and metalness values
- Scale: bounding box ~1–3 units for a small prop
- Visual: clearly recognisable as a {asset_name}, no broken angles
- Roughness/metalness: physically appropriate for the asset type

Return ONLY this JSON:
{{
  "issues": [
    {{"severity": "critical|major|minor", "description": "..."}}
  ],
  "strengths": ["..."],
  "quality_summary": "one sentence",
  "top_priority": "the single most important improvement",
  "material_ok": true
}}"""

    log.info("[critic] evaluating asset")
    raw = call_llm(prompt, json_mode=True, system=CRITIC_SYSTEM)
    result = extract_json(raw)

    if not result:
        result = {
            "issues": [{"severity": "major", "description": "critic parse failed"}],
            "strengths": [],
            "quality_summary": "evaluation failed",
            "top_priority": "fix geometry or material issues",
        }

    # Merge visual issues in (prepend, since visual is ground-truth)
    if visual and visual.get("visual_issues"):
        result["issues"] = visual["visual_issues"] + result.get("issues", [])
    if visual and visual.get("top_visual_priority") and not result.get("top_priority"):
        result["top_priority"] = visual["top_visual_priority"]

    return result

# ── Agent: Planner ────────────────────────────────────────────────────────────

PLANNER_SYSTEM = (
    "You are a Three.js asset improvement planner. "
    "Return only valid JSON with no prose or markdown."
)

def _score_bottleneck(metrics: dict, score: float) -> str:
    """Derive the single lowest-scoring component to guide the planner."""
    geo = metrics.get("geometry") or {}
    analysis = metrics.get("screenshot_analysis") or {}
    if not geo.get("ok"):
        return "fix geometry export so createAsset() returns a valid THREE.Object3D with at least one Mesh"
    shared_edge = geo.get("shared_edge_fraction", 1.0)
    if shared_edge < SHARED_EDGE_MIN:
        return (
            f"PREREQUISITE — mesh connectivity broken (shared_edge_fraction={shared_edge:.3f} < {SHARED_EDGE_MIN}): "
            "call geo.deleteAttribute('uv') before mergeVertices() on IcosahedronGeometry, "
            "or use position-based hash (not sequential vertex index) for noise. "
            "Fix this BEFORE any other improvement."
        )
    tris = geo.get("triangle_count", 0)
    if tris < POLY_IDEAL_MIN:
        return f"increase triangle count from {tris} to at least {POLY_IDEAL_MIN} (currently in {'acceptable' if tris >= POLY_MIN else 'very low'} range, losing 0.10 poly bonus)"
    if not geo.get("has_normals"):
        return "add vertex normals — call geometry.computeVertexNormals() after any position changes"
    if not geo.get("has_uvs"):
        return "add UV coordinates — use spherical/box UV projection or a parametric geometry with built-in UVs"
    unif = analysis.get("coverage_uniformity") or 0.0
    if unif < COVERAGE_UNIFORMITY_THRESHOLD:
        return f"improve shape consistency across angles — coverage uniformity is {unif:.2f} (need ≥{COVERAGE_UNIFORMITY_THRESHOLD}), suggesting broken or very thin geometry from some views"
    div = analysis.get("color_diversity") or 0.0
    if div <= COLOR_DIVERSITY_THRESHOLD:
        return f"increase surface shading variation — color diversity is {div:.0f} (need >{COLOR_DIVERSITY_THRESHOLD}), try more vertex displacement, varied roughness, or higher-detail geometry so lighting creates visible gradients"
    brief = load_brief()
    if brief.get("texture_style") == "vertex-color" and not geo.get("has_vertex_colors"):
        return ("add vertex color variation — the brief requires vertex-color texture style. "
                "Compute per-vertex colors from the vertex's 3D position using hash noise, "
                "map noise value to an earthy palette (dark-shadow → mid-rock → sandy-highlight), "
                "set geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)) "
                "and material.vertexColors = true")
    return "overall quality is good — focus on visual recognisability and shape accuracy"


def run_planner(
    asset_name: str,
    critique: dict,
    current_code: str,
    history: list,
    metrics: dict | None = None,
    score: float = 0.0,
    rubric: dict | None = None,
) -> dict:
    """LLM proposes one specific, implementable improvement. Returns plan dict."""
    steering = load_steering()

    # Steering: force_plan overrides planner entirely
    if steering.get("force_plan"):
        fp = steering["force_plan"]
        log.info(f"[planner] steering force_plan: {fp.get('title', '?')!r}")
        return fp

    past_plans = [h.get("plan", {}).get("title", "?") for h in history[-6:] if h.get("plan")]
    forbidden = steering.get("forbidden_improvements", [])
    focus_hints = steering.get("focus_hints", [])

    # Score-aware context
    gap = PASS_THRESHOLD - score
    status_line = (
        f"Current score: {score:.3f}  |  Threshold: {PASS_THRESHOLD}  |  Gap: {gap:+.3f}"
        if gap > 0
        else f"Current score: {score:.3f}  |  ABOVE THRESHOLD — optimising for quality"
    )
    # Rubric critical failures take absolute priority over score-based bottleneck
    if rubric and rubric.get("critical_failures"):
        bottleneck = f"[RUBRIC BLOCKER] {rubric['remediation']}"
    elif rubric and rubric.get("tier") in (TIER_POLISH, TIER_FAILED):
        bottleneck = rubric.get("remediation") or (_score_bottleneck(metrics or {}, score) if metrics else "see critic findings")
    else:
        bottleneck = _score_bottleneck(metrics or {}, score) if metrics else "see critic findings"

    # Stagnation detection: same score 4+ recent iterations
    recent_scores = [h.get("score", 0) for h in history[-4:] if "score" in h]
    stagnation_note = ""
    if len(recent_scores) >= 4 and max(recent_scores) - min(recent_scores) < 0.01:
        recent_types = [h.get("plan", {}).get("improvement_type", "") for h in history[-4:] if h.get("plan")]
        stagnation_note = (
            f"\nSTAGNATION DETECTED: score has not changed in the last 4 iterations "
            f"(all ≈{recent_scores[-1]:.3f}). Recent attempts: {recent_types}. "
            "You MUST choose a fundamentally different approach — change geometry type, material strategy, or structural organisation."
        )

    focus_block = ""
    if focus_hints:
        focus_block = "FOCUS HINTS (from steering.json):\n" + "\n".join(f"  - {h}" for h in focus_hints) + "\n\n"

    forbidden_block = ""
    if forbidden:
        forbidden_block = "FORBIDDEN (must not attempt):\n" + "\n".join(f"  - {f}" for f in forbidden) + "\n\n"

    cam_block = _camera_basis_block(load_brief())
    episodic_block   = read_episodic_summary()
    graveyard_block  = read_graveyard_summary()
    planner_ref_spec = _format_reference_spec_block(load_brief().get("reference_spec", {}))

    prompt = f"""{focus_block}{cam_block}
{planner_ref_spec}Propose ONE specific improvement for this Three.js {asset_name} asset.

SCORING STATUS:
{status_line}
Primary bottleneck: {bottleneck}{stagnation_note}

CRITIC'S FINDINGS:
{json.dumps(critique, indent=2)}

CURRENT CODE (first 40 lines):
{chr(10).join(current_code.splitlines()[:40])}

IMPROVEMENTS ALREADY TRIED (avoid repeating):
{', '.join(past_plans) if past_plans else 'none yet'}

{episodic_block}
{graveyard_block}
{forbidden_block}{"MANDATORY: Address the primary bottleneck above. This is a required brief constraint — critic suggestions are secondary and must not override it." if rubric and rubric.get("tier") == TIER_POLISH and rubric.get("remediation", "").startswith("vertex color") else "Address the primary bottleneck above unless the critic identifies something more urgent."}
The improvement must be implementable in a single coherent code change.

Return ONLY this JSON:
{{
  "improvement_type": "polygon_count|normals|uvs|material|shape|detail|scale|texture|other",
  "title": "short title, e.g. 'Add box UV mapping'",
  "rationale": "why this addresses the bottleneck",
  "instruction": "precise technical instruction for the coder (2–5 sentences)"
}}"""

    log.info("[planner] proposing improvement")
    raw = call_llm(prompt, json_mode=True, system=PLANNER_SYSTEM)
    result = extract_json(raw)
    if not result:
        result = {
            "improvement_type": "other",
            "title": "general quality improvement",
            "rationale": "planner parse failed",
            "instruction": (
                "Ensure computeVertexNormals() is called after vertex modifications, "
                "add UV projection, keep triangle count between 200 and 1000."
            ),
        }

    # Phase 1: validate plan doesn't violate forbidden improvements
    if result and forbidden:
        instruction_lower = result.get("instruction", "").lower()
        title_lower = result.get("title", "").lower()
        combined = instruction_lower + " " + title_lower
        violations = []
        axis_checks = [
            ("bow in z", "bow"),
            ("fan in y", "fan"),
            ("group.rotation", "rotation"),
            ("vertical layout", "vertical"),
        ]
        for kw, label in axis_checks:
            if kw in combined:
                matching_forbidden = [f for f in forbidden if label in f.lower() or kw in f.lower()]
                if matching_forbidden:
                    violations.append(f"'{kw}' violates: {matching_forbidden[0][:80]}")

        if violations:
            log.warning(f"[planner] plan violates forbidden improvements: {violations}")
            violation_msg = "; ".join(violations)
            retry_prompt = prompt + f"\n\nREJECTED PLAN: The plan above violated these forbidden improvements: {violation_msg}. Produce a NEW plan that does not use these approaches."
            raw2 = call_llm(retry_prompt, json_mode=True, system=PLANNER_SYSTEM)
            result2 = extract_json(raw2)
            if result2:
                result = result2
                log.info(f"[planner] re-planned after violation: {result.get('title', '?')!r}")

    return result

# ── Agent: Coder (single variant) ────────────────────────────────────────────

def run_coder(
    asset_name: str,
    current_code: str,
    plan: dict,
    variant: str = "A",
    rules: list[str] | None = None,
    brief: dict | None = None,
    ref_path=None,
) -> str:
    """Generate one improved JS asset. variant A = primary, B = alternative approach.

    ref_path: optional Path (or str) to a reference PNG. When provided, the image is
    sent alongside the prompt so the coder can see the design target directly.
    Falls back to text-only if image encoding fails or Ollama is used.
    """
    if rules is None:
        rules = load_rules()
    if brief is None:
        brief = load_brief()
    rules_block = _build_rules_block(rules)
    brief_block = _brief_block(brief)

    alt_constraint = ""
    if variant == "B":
        alt_constraint = (
            "\nCONSTRAINT: Use a DIFFERENT Three.js technique than Candidate A. "
            "If A would use IcosahedronGeometry, B should use a custom BufferGeometry or different primitive. "
            "If A would use spherical UV projection, B should use box UV mapping. Think creatively."
        )

    ref_note = ""
    image_arg = None
    if ref_path is not None:
        from pathlib import Path as _Path
        _rp = _Path(ref_path)
        if _rp.exists():
            image_arg = str(_rp)
            ref_note = (
                "\nREFERENCE IMAGE: The image above is the reference design target. "
                "Your code must produce geometry and colours that closely match this reference. "
                "Study the proportions, colours per region, and key features."
            )

    prompt = f"""Modify this Three.js {asset_name} asset according to the improvement instruction.

IMPROVEMENT [{variant}]: {plan.get('title', 'general improvement')}
INSTRUCTION: {plan.get('instruction', 'improve quality')}
{alt_constraint}
{brief_block}{ref_note}

PERMANENT RULES (must follow):
{rules_block}

CURRENT CODE:
```javascript
{current_code}
```

Requirements:
1. Keep: import * as THREE from 'three';
2. Keep: export function createAsset() {{ ... }}
3. createAsset() must return a THREE.Object3D
4. Apply ONLY the specified improvement — no unrelated changes
5. Make MINIMAL targeted changes — add or modify 5–30 lines, do not rewrite unrelated sections
6. Output the COMPLETE file, nothing else

Write ONLY the JavaScript. No markdown, no explanation."""

    log.info(
        f"[coder-{variant}] applying: {plan.get('title', '?')!r}"
        + (" [+ref_image]" if image_arg else "")
    )
    return extract_js(call_llm_coder(prompt, image_path=image_arg))

# ── A/B candidate evaluation ──────────────────────────────────────────────────

def evaluate_ab_candidates(
    asset_name: str,
    current_code: str,
    plan: dict,
    adapter: ThreeJSAdapter,
    state: dict,
) -> tuple[str | None, str | None, dict, str]:
    """
    Generate Candidates A and B, geometry-validate both, render the better one.
    Returns (winning_version_str, winning_js, metrics_for_winner, fail_category_or_ok).

    Strategy (adapted from auto_game A/B/R):
      1. Generate A and B via LLM (both fast, parallel conceptually)
      2. Geometry-validate both (Node, ~1s each)
      3. If one fails geometry, render only the other
      4. If both fail geometry, return (None, None, {}, FAIL_GEOMETRY)
      5. If geo scores differ by >0.15: render only the better
      6. If scores close: render both, pick higher full score
      7. Winner must beat current best_score to be 'accepted'
    """
    current_version = state.get("current_version", "v1")
    next_version    = increment_version(current_version)

    path_a = candidate_path(asset_name, next_version + "_a")
    path_b = candidate_path(asset_name, next_version + "_b")

    # Generate both in parallel (independent HTTP I/O calls)
    rules = load_rules()
    brief = load_brief()
    _ref_path = REFERENCES_DIR / f"{asset_name}.png"
    ref_img = _ref_path if _ref_path.exists() else None
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(run_coder, asset_name, current_code, plan, "A", rules, brief, ref_img)
        fut_b = ex.submit(run_coder, asset_name, current_code, plan, "B", rules, brief, ref_img)
        js_a = fut_a.result()
        js_b = fut_b.result()

    steering_for_val = load_steering()
    banned = _banned_js_patterns(steering_for_val)
    ok_a, reason_a = validate_js_basic(js_a, banned)
    ok_b, reason_b = validate_js_basic(js_b, banned)

    if not ok_a and not ok_b:
        log.warning(f"[A/B] both failed basic validation: A={reason_a}  B={reason_b}")
        if any(r in (reason_a, reason_b) for r in (FAIL_SYNTAX, FAIL_GEN)):
            _maybe_extract_rule(js_a or js_b, FAIL_SYNTAX)
        return None, None, {}, FAIL_SYNTAX

    _brief_patch = load_brief()
    if ok_a:
        js_a = _patch_candidate_js(js_a, _brief_patch)
        path_a.write_text(js_a)
    if ok_b:
        js_b = _patch_candidate_js(js_b, _brief_patch)
        path_b.write_text(js_b)

    # Geometry-validate both passing candidates
    geo_a = adapter.validate_geometry(path_a) if ok_a else {"ok": False}
    geo_b = adapter.validate_geometry(path_b) if ok_b else {"ok": False}

    brief_for_arr = load_brief()
    _check_arrangement(geo_a, brief_for_arr)
    _check_arrangement(geo_b, brief_for_arr)

    score_geo_a = geo_only_score(geo_a)
    score_geo_b = geo_only_score(geo_b)

    log.info(f"[A/B geo] A={score_geo_a:.3f}  B={score_geo_b:.3f}")

    # Decide which to render
    if not geo_a.get("ok") and not geo_b.get("ok"):
        log.warning("[A/B] both failed geometry validation")
        _maybe_extract_rule("", FAIL_GEOMETRY)
        _bury_candidates(asset_name, next_version, [path_a, path_b], geo_a, geo_b)
        return None, None, {}, FAIL_GEOMETRY

    # Determine render candidates
    if not geo_b.get("ok") or (geo_a.get("ok") and score_geo_a >= score_geo_b + 0.15):
        render_candidates = [("A", path_a, js_a, geo_a)]
    elif not geo_a.get("ok") or (geo_b.get("ok") and score_geo_b >= score_geo_a + 0.15):
        render_candidates = [("B", path_b, js_b, geo_b)]
    else:
        render_candidates = [("A", path_a, js_a, geo_a), ("B", path_b, js_b, geo_b)]

    best_variant = None
    best_js = None
    best_metrics: dict = {}
    best_score = -1.0

    for (v_label, v_path, v_js, v_geo) in render_candidates:
        v_tag = f"{asset_name}_{current_version}_{v_label}"
        metrics = adapter.render_and_analyze(v_path, v_tag, v_geo)

        s = calculate_score(metrics)
        log.info(f"[A/B render] Candidate {v_label}: score={s:.3f}")

        if s > best_score:
            best_score = s
            best_variant = v_label
            best_js = v_js
            best_metrics = metrics

    if best_variant is None:
        return None, None, {}, FAIL_RENDER

    return best_variant, best_js, best_metrics, "ok"

# ── Post-coder candidate patching (steering-driven) ───────────────────────────

def _patch_candidate_js(js: str, brief: dict) -> str:
    """Apply mandatory fixes from steering hints that coders reliably ignore.

    Reads steering.json focus_hints for PATCH: directives and applies regex
    replacements. Currently hard-wired for croc_pistol guard/spines until a
    generic directive format is added.
    """
    import re

    # Guard: ensure height=0.28 at y=-0.04 whenever a guardGeo BoxGeometry exists
    if "BoxGeometry" in js and "guardGeo" in js:
        # Fix any BoxGeometry assigned to guardGeo — force correct dimensions
        js = re.sub(
            r"(let guardGeo\s*=\s*new THREE\.BoxGeometry\()[^)]+\)",
            r"\g<1>0.52, 0.28, 0.44, 4, 3, 2)",
            js,
        )
        # Fix guard orange colour (ensure R >= 0.90)
        js = re.sub(
            r"(// ── Guard[^\n]*\n(?:.*\n)*?guardGeo\s*=\s*applyFaceColors\(guardGeo,\s*t\s*=>\s*\[)\s*[\d.]+\s*\+\s*t\s*\*\s*[\d.]+,\s*[\n ]*[\d.]+\s*\+\s*t\s*\*\s*[\d.]+,\s*[\n ]*[\d.]+\s*\+\s*t\s*\*\s*[\d.]+,",
            r"\g<1>\n        0.92 + t * 0.05,\n        0.42 + t * 0.07,\n        0.12 + t * 0.03,",
            js,
        )
        # Fix guard y position
        def fix_guard_pos(m):
            inner = m.group(1)
            # Replace y value (second arg) with -0.04
            parts = inner.split(",")
            if len(parts) >= 2:
                parts[1] = " -0.04"
            return "guardMesh.position.set(" + ",".join(parts) + ")"
        js = re.sub(r"guardMesh\.position\.set\(([^)]+)\)", fix_guard_pos, js)

    # Guard emissive: add emissive orange so guard is visible under low ambient
    if "guardMesh" in js and "emissiveIntensity: 0.9" not in js:
        js = re.sub(
            r"(guardMesh\s*=\s*new THREE\.Mesh\(guardGeo,\s*new THREE\.MeshStandardMaterial\(\{[^}]+)\}",
            lambda m: m.group(0).rstrip("}").rstrip()
                + ", emissive: new THREE.Color(0.38, 0.10, 0.01), emissiveIntensity: 0.9 }",
            js,
            count=1,
        )

    # Spines: add 5 gold ConeGeometry spines with emissive if none present
    if "ConeGeometry" not in js and "railMesh" in js:
        spine_block = (
            "\n\n    // ── Gold ornamental spines — 5 pyramid cones on rail ──────────────────\n"
            "    const _spineXs = [-0.55, -0.28, 0, 0.28, 0.55];\n"
            "    for (const _sx of _spineXs) {\n"
            "        let _spGeo = new THREE.ConeGeometry(0.06, 0.11, 4, 1);\n"
            "        posNoise(_spGeo, 0.006);\n"
            "        _spGeo = applyFaceColors(_spGeo, t => [0.95 + t * 0.05, 0.75 + t * 0.10, 0.15 + t * 0.05]);\n"
            "        const _spMesh = new THREE.Mesh(_spGeo, new THREE.MeshStandardMaterial({\n"
            "            color: 0xffffff, vertexColors: true, roughness: 0.10, metalness: 0.92,\n"
            "            emissive: new THREE.Color(0.30, 0.22, 0.0), emissiveIntensity: 0.7\n"
            "        }));\n"
            "        _spMesh.name = 'rail';\n"
            "        _spMesh.position.set(_sx, 0.665, 0);\n"
            "        group.add(_spMesh);\n"
            "    }"
        )
        # Insert after group.add(railMesh)
        js = re.sub(
            r"(group\.add\(railMesh\);)",
            r"\1" + spine_block,
            js,
            count=1,
        )

    # Remove duplicate orangeGuard/secondary guard meshes added by coders
    js = re.sub(
        r"\s*// ── Orange Guard Element[^\n]*\n.*?group\.add\(orangeGuardMesh\);",
        "",
        js,
        flags=re.DOTALL,
    )

    return js


# ── Rule extraction from failures ─────────────────────────────────────────────

def _maybe_extract_rule(js: str, fail_category: str) -> None:
    """Extract a permanent lesson when the same failure type recurs."""
    rules = load_rules()
    if fail_category == FAIL_SYNTAX and RULE_SYNTAX not in rules:
        append_rule(RULE_SYNTAX)
    elif fail_category == FAIL_GEOMETRY and RULE_GEOMETRY not in rules:
        append_rule(RULE_GEOMETRY)


def _extract_rubric_rules(rubric: dict) -> None:
    """
    Detect rubric failure patterns and append permanent learning rules.
    Spike and material-type rules fire immediately (clear code errors).
    Inverted-normals rule only fires when vision confirms (prevents false positives
    from dark materials triggering a misleading rule).
    """
    critical = rubric.get("critical_failures", [])
    cats     = rubric.get("categories", {})

    gi_failures = cats.get("geometry_integrity", {}).get("failures", [])
    rq_failures = cats.get("render_quality", {}).get("failures", [])
    mc_failures = cats.get("material_compliance", {}).get("failures", [])

    if any("spike artefact" in f for f in gi_failures):
        append_rule(RULE_SPIKE)

    if any("inverted normals confirmed" in f for f in rq_failures + critical):
        append_rule(RULE_INVERTED_NORMALS)

    if any("non-standard material" in f or "non-PBR material" in f for f in mc_failures + critical):
        append_rule(RULE_MATERIAL_TYPE)

# ── Graveyard ─────────────────────────────────────────────────────────────────

def _bury_candidates(asset_name: str, version: str, paths: list[Path], *geo_results) -> None:
    """Move both-failed A/B files to candidates/graveyard/ and log the errors."""
    GRAVEYARD_DIR.mkdir(exist_ok=True)
    for p in paths:
        if p.exists():
            p.rename(GRAVEYARD_DIR / p.name)
    record = {
        "ts":         datetime.now().isoformat(),
        "asset_name": asset_name,
        "version":    version,
        "errors":     [g.get("error") for g in geo_results if isinstance(g, dict)],
    }
    with GRAVEYARD_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    log.info(f"[graveyard] buried {[p.name for p in paths if not p.exists()]}")


def read_graveyard_summary(n: int = 5) -> str:
    """Return a short summary of recent both-failed iterations for the planner."""
    if not GRAVEYARD_PATH.exists():
        return ""
    lines = GRAVEYARD_PATH.read_text().strip().splitlines()
    if not lines:
        return ""
    recent = []
    for line in lines[-n:]:
        try:
            recent.append(json.loads(line))
        except Exception:
            pass
    if not recent:
        return ""
    errors = [e for r in recent for e in (r.get("errors") or []) if e]
    if not errors:
        return ""
    return (
        f"GRAVEYARD (last {len(recent)} both-failed iterations — avoid repeating):\n"
        + "\n".join(f"  - {e}" for e in errors[:6])
    )

def _check_failure_patterns(asset_name: str, outcome: str) -> None:
    """If same improvement_type regresses 3 consecutive times, write a rule."""
    if not EPISODIC_PATH.exists():
        return
    lines = EPISODIC_PATH.read_text().strip().splitlines()
    recent = []
    for line in lines[-6:]:
        try:
            r = json.loads(line)
            if len(r) >= 14 and r[2] == asset_name:
                recent.append({"type": r[4], "outcome": r[13]})
        except Exception:
            pass
    if len(recent) < 3:
        return
    last3 = recent[-3:]
    bad_outcomes = {FAIL_GEOMETRY, FAIL_RENDER, FAIL_SYNTAX, "regression", "reverted"}
    if all(r["outcome"] in bad_outcomes for r in last3):
        imp_type = last3[-1]["type"]
        if all(r["type"] == imp_type for r in last3):
            rule = (
                f"Attempting '{imp_type}' improvements for {asset_name} has regressed or failed "
                "3 consecutive times — avoid this approach and try a fundamentally different type"
            )
            append_rule(rule)
            log.info(f"[pattern-rule] wrote negative rule for repeated {imp_type!r} failures")


# ── On-pass export ─────────────────────────────────────────────────────────────

def export_passed_asset(state: dict, metrics: dict, score: float, adapter: ThreeJSAdapter) -> None:
    """Copy the winning asset to dist/ with thumbnail and metadata."""
    asset_name   = state["asset_name"]
    best_version = state["best_version"]
    src_path     = candidate_path(asset_name, best_version)
    if not src_path.exists():
        log.warning(f"[export] source {src_path.name} missing — skipping")
        return

    DIST_DIR.mkdir(exist_ok=True)

    dst_mjs = DIST_DIR / f"{asset_name}.mjs"
    shutil.copy2(src_path, dst_mjs)
    log.info(f"[export] {dst_mjs.name}")

    # Thumbnail — single quarter-angle render
    thumb_path = DIST_DIR / f"{asset_name}_thumb.png"
    adapter.render_single(src_path, thumb_path)

    # Per-asset metadata
    geo = metrics.get("geometry") or {}
    meta = {
        "asset_name":     asset_name,
        "version":        best_version,
        "score":          score,
        "timestamp":      datetime.now().isoformat(),
        "description":    state.get("asset_description", ""),
        "triangle_count": geo.get("triangle_count"),
        "has_normals":    geo.get("has_normals"),
        "has_uvs":        geo.get("has_uvs"),
    }
    (DIST_DIR / f"{asset_name}_meta.json").write_text(json.dumps(meta, indent=2))

    # Rolling manifest (one entry per asset_name, newest wins)
    manifest_path = DIST_DIR / "manifest.json"
    manifest = _read_json(manifest_path, [])
    manifest = [m for m in manifest if m.get("asset_name") != asset_name]
    manifest.append(meta)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"[export] manifest updated ({len(manifest)} assets)")

# ── Main loop ─────────────────────────────────────────────────────────────────

MATERIAL_RULES = [
    "Stone, rock, wood, dirt: set roughness 0.75–0.95 and metalness 0.0–0.05 explicitly",
    "Metal (ore, armour, coins): roughness 0.10–0.35 and metalness 0.70–1.0",
    "Crystal, glass: roughness 0.0–0.15 and metalness 0.0; add emissive colour for glow",
    "Always set roughness and metalness explicitly on every MeshStandardMaterial — never leave as default (0.5/0.5)",
]


def run_loop(state: dict, max_iter: int) -> None:
    adapter = ThreeJSAdapter()
    asset_name = state["asset_name"]
    asset_desc = state["asset_description"]

    # Seed material rules into rules.json on first run
    for rule in MATERIAL_RULES:
        append_rule(rule)

    log.info(f"=== threejs_loop START  asset={asset_name}  max_iter={max_iter} ===")

    # Generate initial asset if no current version
    if not state["current_version"]:
        v1_path = candidate_path(asset_name, "v1")
        if v1_path.exists():
            log.info(f"[init] using existing {v1_path.name}")
        else:
            js = generate_initial_asset(asset_name, asset_desc)
            ok, reason = validate_js_basic(js)
            if not ok:
                log.warning(f"[init] first attempt invalid ({reason}), retrying")
                js = generate_initial_asset(asset_name, asset_desc)
            v1_path.write_text(js)
            log.info(f"[init] wrote {v1_path.name} ({len(js)} chars)")
        state["current_version"] = "v1"
        save_state(state)

    # Build reference spec once (cached by image checksum in brief.json)
    ref_path_for_spec = find_reference(asset_name)
    if ref_path_for_spec:
        build_reference_spec(asset_name, ref_path_for_spec)

    start_iter = state["iteration"]

    for _ in range(max_iter):
        state["iteration"] += 1
        iteration = state["iteration"]
        version   = state["current_version"]
        asset_path = candidate_path(asset_name, version)

        if not asset_path.exists():
            log.error(f"[iter {iteration}] {asset_path} missing — aborting")
            break

        log.info(f"\n{'='*60}")
        log.info(f"[iter {iteration}] version={version}  trust={state.get('trust_score', 50):.0f}")

        # ── Playtester ──────────────────────────────────────────────────────
        t0 = time.time()
        metrics = adapter.run(asset_path, f"{asset_name}_{version}")
        _check_arrangement(metrics.get("geometry") or {}, load_brief())
        metrics["code_line_count"] = len(asset_path.read_text().splitlines())

        # Vision runs before scoring so recognisable-bonus and reference-bonus apply
        stitched = metrics.get("stitched_path")
        visual = {}
        if stitched and Path(stitched).exists():
            _geo_count = (metrics.get("geometry") or {}).get("arrangement_body_count") or None
            visual = run_visual_analysis(asset_name, stitched, geo_count=_geo_count)
            metrics["visual_analysis"] = visual

        elapsed = time.time() - t0
        score  = calculate_score(metrics)
        rubric = get_rubric(metrics, asset_name)
        _extract_rubric_rules(rubric)

        log.info(f"[playtester] {elapsed:.1f}s  score={score:.3f}  rubric={rubric['tier']} ({rubric['score_100']}/100)")
        log.info(f"  geometry:   {fmt_geo(metrics.get('geometry') or {})}")
        log.info(f"  screenshot: {fmt_screenshot(metrics.get('screenshot'), metrics.get('screenshot_analysis'))}")
        if rubric["critical_failures"]:
            log.info(f"  [BLOCKERS] {'; '.join(rubric['critical_failures'][:2])}")

        # ── Stagnation detection (Item 5) ───────────────────────────────────
        prev_scores = [h.get("score", 0) for h in state["history"][-5:] if "score" in h]
        is_stagnant = (
            len(prev_scores) >= 4
            and max(prev_scores) - min(prev_scores) < 0.015
            and score < PASS_THRESHOLD
            and rubric["tier"] in (TIER_POLISH, TIER_FAILED)
        )
        if is_stagnant:
            state["stagnation_count"] = state.get("stagnation_count", 0) + 1
            log.info(f"[stagnation] count={state['stagnation_count']} score={score:.3f} (not improving)")
        else:
            state["stagnation_count"] = 0

        if state.get("stagnation_count", 0) == 4:
            _inject_stagnation_recovery(asset_path)
        elif state.get("stagnation_count", 0) == 8:
            log.warning(f"[stagnation] {asset_name} stuck for 8 iters — human review recommended")
            steering_s = load_steering()
            if steering_s.get("force_plan", {}).get("_stagnation_injected"):
                steering_s["force_plan"] = None
                STEERING_PATH.write_text(json.dumps(steering_s, indent=2))

        # ── Accept / revert ─────────────────────────────────────────────────
        # ref_sim < 3 means completely wrong shape (e.g. 0/10 vs reference).
        # Block acceptance even if automated score is higher — prevents a bad
        # shape from becoming new best and spawning further divergent candidates.
        ref_sim_ok = (visual.get("reference_similarity") or 10) >= 3
        if not ref_sim_ok:
            log.info(
                f"[ref_sim block] ref_sim={visual.get('reference_similarity')}/10 < 3 — "
                f"shape regression, not updating best"
            )

        if state["best_version"] is None or iteration == start_iter + 1:
            state["best_version"] = version
            state["best_score"]   = score
            state["consecutive_regressions"] = 0
        elif score >= state["best_score"] and ref_sim_ok:
            state["best_version"] = version
            state["best_score"]   = score
            state["consecutive_regressions"] = 0
            log.info(f"[accept] new best: {version} score={score:.3f}")
            if state["iteration"] <= 3:  # only auto-calibrate early, from clean starts
                _auto_calibrate_color(metrics)
            # Clear manual force_plan once it leads to an accepted improvement
            _steer = load_steering()
            if _steer.get("force_plan") and not _steer["force_plan"].get("_stagnation_injected"):
                _steer["force_plan"] = None
                STEERING_PATH.write_text(json.dumps(_steer, indent=2))
                log.info("[steer] cleared manual force_plan after acceptance")
        else:
            state["consecutive_regressions"] += 1
            reason = (f"score {score:.3f} < best {state['best_score']:.3f}" if ref_sim_ok else
                      f"ref_sim={visual.get('reference_similarity')}/10 < 3 blocked acceptance")
            log.info(
                f"[regression {state['consecutive_regressions']}/{REVERT_AFTER_N_REGRESSIONS}] "
                f"{reason}"
            )
            if state["consecutive_regressions"] >= REVERT_AFTER_N_REGRESSIONS:
                log.info(f"[revert] → {state['best_version']}")
                state["current_version"] = state["best_version"]
                state["consecutive_regressions"] = 0
                state["iteration"] -= 1   # don't count revert as productive iteration
                save_state(state)
                continue

        # ── Done? — requires score, rubric tier, and optional ref_sim floor ────
        if score >= PASS_THRESHOLD and rubric["tier"] == TIER_PRODUCTION:
            _min_ref_sim = load_brief().get("min_reference_similarity_to_pass")
            _cur_ref_sim = (visual or {}).get("reference_similarity")
            # Treat None as 0 — no vision result is conservative, not a pass
            _eff_ref_sim = _cur_ref_sim if _cur_ref_sim is not None else 0
            if _cur_ref_sim is None and _min_ref_sim:
                log.warning(
                    "[ref_sim gate] reference comparison returned None — treating as 0, gate active"
                )
            # Adaptive degradation: if ref_sim has been the same for 4+ consecutive
            # iterations, lower the gate by 1 (the vision model has hit its ceiling).
            if _min_ref_sim and _eff_ref_sim < _min_ref_sim:
                recent_ref_sims = [
                    h.get("ref_sim")
                    for h in state.get("history", [])[-4:]
                    if h.get("ref_sim") is not None
                ]
                if (
                    len(recent_ref_sims) >= 4
                    and len(set(recent_ref_sims)) == 1
                    and recent_ref_sims[0] == _eff_ref_sim
                ):
                    _min_ref_sim -= 1
                    log.warning(
                        f"[ref_sim gate] ref_sim={_eff_ref_sim}/10 stuck for 4 iterations — "
                        f"lowering gate to {_min_ref_sim} (vision model ceiling reached)"
                    )
            if _min_ref_sim and _eff_ref_sim < _min_ref_sim:
                log.info(
                    f"[ref_sim gate] ref_sim={_cur_ref_sim}/10 (eff={_eff_ref_sim}) < {_min_ref_sim} — "
                    f"score passes but reference match not good enough yet — continuing"
                )
            else:
                log.info(f"\n{'='*60}")
                log.info(
                    f"DONE — {asset_name} passed "
                    f"(score={score:.3f} ≥ {PASS_THRESHOLD}, rubric={rubric['score_100']}/100)"
                )
                log.info(f"Best: {candidate_path(asset_name, state['best_version'])}")
                state["done"] = True
                save_state(state)
                export_passed_asset(state, metrics, score, adapter)
                return

        if iteration >= start_iter + max_iter:
            log.info(f"[limit] reached max_iter={max_iter}")
            break

        # ── Critic ──────────────────────────────────────────────────────────
        critique = run_critic(asset_name, metrics, visual, state["history"], iteration, rubric=rubric)
        log.info(f"[critic] {critique.get('quality_summary', '?')!r}")
        log.info(f"  top_priority: {critique.get('top_priority', '?')!r}")

        # ── Planner ─────────────────────────────────────────────────────────
        current_code = asset_path.read_text()
        plan = run_planner(
            asset_name, critique, current_code, state["history"], metrics, score, rubric=rubric
        )
        log.info(f"[planner] {plan.get('improvement_type', '?')} → {plan.get('title', '?')!r}")

        # ── Coder A/B ───────────────────────────────────────────────────────
        winner_variant, winner_js, winner_metrics, ab_status = evaluate_ab_candidates(
            asset_name, current_code, plan, adapter, state
        )

        if ab_status != "ok" or winner_js is None:
            log.warning(f"[A/B] both candidates failed ({ab_status})")
            outcome = ab_status
            entry = {
                "version":       version,
                "score":         score,
                "ref_sim":       (visual or {}).get("reference_similarity"),
                "critique":      critique,
                "plan":          plan,
                "outcome":       outcome,
                "fail_category": ab_status,
                "rubric_tier":   rubric["tier"],
                "rubric_score":  rubric["score_100"],
            }
            state["history"].append(entry)
            write_episodic(iteration, asset_name, version, metrics, score, plan, outcome)
            _check_failure_patterns(asset_name, outcome)
            state["trust_score"] = compute_trust(state["history"])
            write_status(state, metrics, score, critique, rubric=rubric)
            save_state(state)
            continue

        # ── Write winning candidate ─────────────────────────────────────────
        new_version = increment_version(version)
        new_path = candidate_path(asset_name, new_version)
        new_path.write_text(winner_js)
        log.info(f"[A/B] winner=Candidate {winner_variant}  wrote {new_path.name} ({len(winner_js)} chars)")

        winner_score  = calculate_score(winner_metrics)
        winner_rubric = get_rubric(winner_metrics, asset_name)
        # ref_sim_ok was set above from the current iteration's visual analysis.
        # If the current version has critically wrong shape, don't accept the winner either.
        outcome = "accepted" if winner_score >= state["best_score"] and ref_sim_ok else "regression"

        if outcome == "accepted" and winner_score > state.get("best_score", 0) + 0.04:
            _auto_append_focus_hint(plan)

        entry = {
            "version":       version,
            "score":         score,
            "ref_sim":       (visual or {}).get("reference_similarity"),
            "critique":      critique,
            "plan":          plan,
            "winner_variant": winner_variant,
            "outcome":       outcome,
            "rubric_tier":   rubric["tier"],
            "rubric_score":  rubric["score_100"],
        }
        state["history"].append(entry)
        state["current_version"] = new_version
        if outcome == "accepted":
            _sync_to_dist(asset_name, new_version, geometry=winner_metrics.get("geometry"))
        state["trust_score"] = compute_trust(state["history"])

        write_episodic(iteration, asset_name, version, metrics, score, plan, outcome)
        _check_failure_patterns(asset_name, outcome)
        write_status(state, winner_metrics, winner_score, critique, rubric=winner_rubric)
        save_state(state)

    best = state.get("best_version", "v1")
    log.info(f"\n=== threejs_loop END ===  best={best}  score={state.get('best_score', 0):.3f}")
    log.info(f"Best asset: {candidate_path(asset_name, best)}")
    _git_backup(asset_name, best, state.get("best_score", 0))

def _git_backup(asset_name: str, best_version: str, score: float) -> None:
    """Commit and push latest assets to the configured git remote (best-effort)."""
    import subprocess
    repo = Path(__file__).parent
    if not (repo / ".git").exists():
        return
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
        msg = f"auto: {asset_name} {best_version} score={score:.3f}"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo, capture_output=True, text=True
        )
        if "nothing to commit" in result.stdout:
            log.info("[git] nothing new to commit")
            return
        push = subprocess.run(
            ["git", "push"], cwd=repo, capture_output=True, text=True, timeout=30
        )
        if push.returncode == 0:
            log.info(f"[git] pushed: {msg}")
        else:
            log.warning(f"[git] push failed: {push.stderr.strip()}")
    except Exception as exc:
        log.warning(f"[git] backup error: {exc}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Three.js game asset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("asset_name", nargs="?", help="Asset type, e.g. 'rock'")
    parser.add_argument("asset_description", nargs="?", default="",
                        help="Description for initial generation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing loop_state.json")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS,
                        help=f"Max iterations this run (default {MAX_ITERATIONS})")
    parser.add_argument("--score-only", action="store_true",
                        help="Run playtester on current version and print score, then exit")

    args = parser.parse_args()

    if args.resume or (not args.asset_name and STATE_PATH.exists()):
        state = load_state()
        if not state:
            log.error("--resume: no loop_state.json found")
            sys.exit(1)
        log.info(
            f"Resuming: asset={state['asset_name']}  iter={state['iteration']}  "
            f"best_score={state.get('best_score', 0):.3f}  "
            f"trust={state.get('trust_score', 50):.0f}"
        )
    elif args.asset_name:
        state = fresh_state(args.asset_name, args.asset_description)
        save_state(state)
    else:
        parser.print_help()
        sys.exit(1)

    if args.score_only:
        adapter = ThreeJSAdapter()
        version = state.get("current_version") or state.get("best_version", "v1")
        path    = candidate_path(state["asset_name"], version)
        metrics = adapter.run(path, f"{state['asset_name']}_{version}")
        score   = calculate_score(metrics)
        rubric  = get_rubric(metrics, state["asset_name"])
        print(json.dumps({
            "version":             version,
            "score":               score,
            "rubric":              rubric,
            "geometry":            metrics.get("geometry"),
            "screenshot_analysis": metrics.get("screenshot_analysis"),
        }, indent=2))
        return

    run_loop(state, args.max_iter)


if __name__ == "__main__":
    main()

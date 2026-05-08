"""
reference_analyst.py — Multi-pass reference image analysis for threejs_loop.

Produces extended_brief.json: a detailed per-part geometry + color spec derived
from a reference image. Runs once per asset (cached by ref-image SHA256).

Passes:
  A. PIL silhouette + color regions (no LLM, ~0.1s)
  B. VLM global part decomposition — one Groq call (~4s)
  C. VLM per-part zoom analysis — parallel Groq calls, ≤6 workers (~8s for 12 parts)
  D. Aggregation → extended_brief.json
"""

import base64
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

log = logging.getLogger(__name__)

LOOP_DIR = Path(__file__).parent.resolve()
EXTENDED_BRIEF_PATH = LOOP_DIR / "extended_brief.json"

GROQ_URL   = "https://api.groq.com/openai/v1"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Max parts to zoom-analyse in Stage C. Sorted by bbox area (largest first).
# Keeps Stage C latency bounded while still covering all major parts.
MAX_PARTS_TO_ZOOM = 10


# ── Utilities ─────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _encode_pil(img, max_size: int = 900) -> str:
    """Resize PIL Image if needed, return base64 PNG string."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        from PIL import Image
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _groq_vision(prompt: str, b64: str, timeout: int = 35, max_tokens: int = 2000) -> str | None:
    """Single Groq vision call. Returns response text or None on failure."""
    import requests
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    try:
        r = requests.post(
            f"{GROQ_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ]}],
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"[ref-analyst] Groq call failed: {e}")
        return None


def _parse_json_from(text: str) -> list | dict | None:
    """Extract first JSON array or object from raw VLM text."""
    if not text:
        return None
    # Try array first, then object
    for start_char, end_char in (("[", "]"), ("{", "}")):
        s = text.find(start_char)
        e = text.rfind(end_char)
        if s != -1 and e > s:
            try:
                return json.loads(text[s:e + 1])
            except Exception:
                pass
    return None


# ── Stage A: PIL silhouette + dominant color regions ─────────────────────────

def extract_silhouette(ref_path: Path) -> dict:
    """
    Extract object mask, bounding box, and dominant color palette
    from the reference image. Works with alpha-transparent PNGs
    (checkered background) or plain background.
    No LLM required.
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(ref_path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        alpha = arr[:, :, 3]

        if alpha.min() < 200:
            # PNG with real transparency — use alpha channel directly
            mask = alpha > 64
        else:
            # No useful alpha (all 255) — detect background from corners
            rgb = arr[:, :, :3].astype(np.float32)
            corners = np.concatenate([
                rgb[:30, :30].reshape(-1, 3),
                rgb[:30, -30:].reshape(-1, 3),
                rgb[-30:, :30].reshape(-1, 3),
                rgb[-30:, -30:].reshape(-1, 3),
            ])
            bg = corners.mean(axis=0)
            bg_std = float(corners.std())
            tol = max(35.0, bg_std * 2.5)
            mask = ~np.all(np.abs(rgb - bg) < tol, axis=2)

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if not len(rows) or not len(cols):
            return {"ok": False, "error": "no object pixels found"}

        h_img, w_img = int(arr.shape[0]), int(arr.shape[1])
        bx, by = int(cols[0]), int(rows[0])
        bw, bh = int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)

        # Dominant palette: k-means on hue of non-bg pixels
        obj_rgb = arr[mask][:, :3].astype(np.float32) / 255.0
        palette = _extract_palette(obj_rgb, k=8)

        return {
            "ok": True,
            "image_wh": [w_img, h_img],
            "object_bbox_px": [bx, by, bw, bh],
            "aspect_ratio": round(bw / max(bh, 1), 3),
            "fill_fraction": round(float(mask[by:by+bh, bx:bx+bw].mean()), 3),
            "dominant_palette_rgb": palette,
        }
    except Exception as e:
        log.warning(f"[ref-analyst] extract_silhouette failed: {e}")
        return {"ok": False, "error": str(e)}


def _extract_palette(obj_rgb_f32, k: int = 8) -> list[list[int]]:
    """Simple k-means hue clustering → top-k representative colors (RGB 0-255)."""
    try:
        import numpy as np
        import colorsys
        # Convert to HSV, bucket by hue into k bins
        hues, sats, vals = [], [], []
        for px in obj_rgb_f32[::6]:   # sample every 6th pixel for speed
            h, s, v = colorsys.rgb_to_hsv(float(px[0]), float(px[1]), float(px[2]))
            if v > 0.08 and s > 0.05:
                hues.append(h)
                sats.append(s)
                vals.append(v)
        if len(hues) < 20:
            return []
        hues_arr = np.array(hues)
        hist, edges = np.histogram(hues_arr, bins=k, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2
        # For each hue bin: pick the median sat and val
        palette = []
        for i in np.argsort(-hist)[:k]:
            if hist[i] < 10:
                continue
            hbin_mask = (hues_arr >= edges[i]) & (hues_arr < edges[i + 1])
            hue_c = centers[i]
            sat_c = float(np.median([sats[j] for j, m in enumerate(hbin_mask) if m]))
            val_c = float(np.median([vals[j] for j, m in enumerate(hbin_mask) if m]))
            r, g, b = colorsys.hsv_to_rgb(hue_c, sat_c, val_c)
            palette.append([int(r * 255), int(g * 255), int(b * 255)])
        return palette[:k]
    except Exception:
        return []


# ── Stage B: VLM global part decomposition ───────────────────────────────────

_DECOMPOSE_PROMPT = """\
Look at this image of a stylised game asset carefully.

List EVERY distinct named component you can see. Include main parts AND small details
(buttons, screws, loops, cords, engravings, rails, sights, triggers, guards, orbs).

For EACH component output a JSON object:
{
  "name": "snake_case_identifier",
  "bbox_px": [x, y, width, height],
  "dominant_color": "plain english color name",
  "dominant_rgb": [R, G, B],
  "geometry_hint": "box|cylinder|cone|torus|tube|flat_strip|custom",
  "description": "one sentence"
}

Output ONLY a JSON ARRAY of these objects. No explanation text."""


def decompose_parts(ref_path: Path) -> list[dict]:
    """VLM pass B: identify all named parts in the reference image."""
    try:
        from PIL import Image
        img = Image.open(ref_path).convert("RGB")
        b64 = _encode_pil(img, max_size=1024)
        text = _groq_vision(_DECOMPOSE_PROMPT, b64, timeout=45, max_tokens=2500)
        result = _parse_json_from(text or "")
        if isinstance(result, list):
            parts = []
            for p in result:
                if not isinstance(p, dict) or "name" not in p:
                    continue
                # Validate bbox
                bbox = p.get("bbox_px", [])
                if len(bbox) == 4:
                    w_img, h_img = img.size
                    x, y, bw, bh = bbox
                    p["bbox_px"] = [
                        max(0, min(int(x), w_img)),
                        max(0, min(int(y), h_img)),
                        max(1, min(int(bw), w_img)),
                        max(1, min(int(bh), h_img)),
                    ]
                parts.append(p)
            log.info(f"[ref-analyst] Stage B: {len(parts)} parts identified")
            return parts
        log.warning(f"[ref-analyst] Stage B parse failed — raw: {str(text)[:200]}")
        return []
    except Exception as e:
        log.warning(f"[ref-analyst] decompose_parts failed: {e}")
        return []


# ── Stage C: VLM per-part zoom analysis ──────────────────────────────────────

def _part_detail_prompt(part_name: str) -> str:
    return f"""\
This cropped image shows the '{part_name}' component of a stylised game weapon.

Output a JSON object describing it:
{{
  "geometry_primitive": "box|cylinder|cone|torus|tube|sphere|flat_strip|custom",
  "dominant_rgb": [R, G, B],
  "secondary_rgb": [R, G, B] or null,
  "color_gradient": "none|horizontal|vertical|radial",
  "has_detail_texture": true or false,
  "texture_description": "one phrase e.g. scrollwork engraving, crocodile scales, rope braid" or null,
  "roughness_estimate": "very_glossy|glossy|satin|matte",
  "is_metallic": true or false,
  "emissive_hint": true or false,
  "visible_sub_features": ["list", "of", "notable", "details"]
}}

Output ONLY the JSON object."""


def analyze_part(ref_path: Path, part: dict, obj_bbox: list) -> dict:
    """Zoom into part bbox, run a targeted VLM analysis. Returns detail dict."""
    try:
        import numpy as np
        from PIL import Image
        img = Image.open(ref_path).convert("RGB")
        arr = np.array(img)
        bbox = part.get("bbox_px", [])
        if len(bbox) < 4:
            return {}
        x, y, bw, bh = [int(v) for v in bbox]
        pad = 15
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(arr.shape[1], x + bw + pad)
        y1 = min(arr.shape[0], y + bh + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return {}
        crop = Image.fromarray(arr[y0:y1, x0:x1])
        b64 = _encode_pil(crop, max_size=512)
        text = _groq_vision(_part_detail_prompt(part["name"]), b64, timeout=25, max_tokens=600)
        result = _parse_json_from(text or "")
        if isinstance(result, dict):
            return result
        return {}
    except Exception as e:
        log.warning(f"[ref-analyst] analyze_part({part.get('name')}) failed: {e}")
        return {}


# ── Stage D: World-space projection + aggregation ────────────────────────────

def _px_to_world(bbox_px: list, obj_bbox: list, world_size: float = 2.0) -> dict:
    """
    Map a pixel bounding box (in reference image space) to approximate
    world-space centre and dimensions. Assumes the object occupies a
    world_size × world_size bounding box centred at origin.
    Y axis is inverted (image-down = world-down = -Y).
    """
    if len(bbox_px) < 4 or len(obj_bbox) < 4:
        return {}
    ox, oy, ow, oh = [float(v) for v in obj_bbox]
    x, y, bw, bh = [float(v) for v in bbox_px]
    if ow < 1 or oh < 1:
        return {}
    # Fractional position of part centre relative to object centre
    cx_frac = (x + bw / 2 - ox) / ow - 0.5   # -0.5 (left) to +0.5 (right)
    # Image Y increases downward; world Y increases upward
    cy_frac = 0.5 - (y + bh / 2 - oy) / oh   # +0.5 (top) to -0.5 (bottom)
    # Preserve object aspect in world space
    aspect = ow / max(oh, 1)
    wx = round(cx_frac * world_size, 3)
    wy = round(cy_frac * world_size / aspect, 3)
    ww = round((bw / ow) * world_size, 3)
    wh = round((bh / oh) * world_size / aspect, 3)
    return {"x": wx, "y": wy, "w": ww, "h": wh, "depth": 0.44}


def _checklist_question(part: dict) -> str:
    """Generate a yes/no VLM feature-check question for a part."""
    name = part["name"].replace("_", " ")
    color = part.get("dominant_color", "")
    desc = part.get("description", "")
    detail = part.get("detail", {})
    texture = detail.get("texture_description") if detail else None
    q = f"Is there a {color} {name}" if color else f"Is there a {name}"
    if texture:
        q += f" featuring {texture}"
    if desc:
        short_desc = desc.split(".")[0]
        q += f"? ({short_desc})"
    else:
        q += "?"
    return q


# ── Public entry point ────────────────────────────────────────────────────────

def build_extended_brief(asset_name: str, ref_path: Path, brief: dict) -> dict:
    """
    Run all analysis passes and produce extended_brief.json.
    Cached by ref-image SHA256 — only re-runs when the reference image changes.

    Returns the extended_brief dict (also written to extended_brief.json).
    Falls back gracefully to brief.geometry_spec if VLM passes fail.
    """
    ref_sha = _sha256(ref_path)

    # Cache check
    if EXTENDED_BRIEF_PATH.exists():
        try:
            cached = json.loads(EXTENDED_BRIEF_PATH.read_text())
            if cached.get("_ref_sha") == ref_sha and cached.get("_asset") == asset_name:
                log.info(f"[ref-analyst] cache hit (sha={ref_sha}, {cached.get('part_count',0)} parts)")
                return cached
        except Exception:
            pass

    log.info(f"[ref-analyst] building extended brief for '{asset_name}' ...")

    # ── Stage A: silhouette ────────────────────────────────────────────────
    sil = extract_silhouette(ref_path)
    obj_bbox = sil.get("object_bbox_px", [0, 0, 1000, 600])
    obj_area = float(obj_bbox[2]) * float(obj_bbox[3])
    log.info(f"[ref-analyst] Stage A: bbox={obj_bbox}, aspect={sil.get('aspect_ratio')}")

    # ── Stage B: part decomposition ────────────────────────────────────────
    parts = decompose_parts(ref_path)
    if not parts:
        log.warning("[ref-analyst] Stage B: no parts — falling back to brief geometry_spec")
        for pname, spec in brief.get("geometry_spec", {}).items():
            if isinstance(spec, dict):
                parts.append({
                    "name": pname,
                    "bbox_px": obj_bbox,   # placeholder
                    "dominant_color": "unknown",
                    "dominant_rgb": [128, 128, 128],
                    "geometry_hint": "box",
                    "description": spec.get("note", pname),
                })
    log.info(f"[ref-analyst] Stage B: {len(parts)} parts")

    # ── Stage C: per-part zoom (top N by bbox area) ────────────────────────
    def _part_area(p: dict) -> float:
        bbox = p.get("bbox_px", [])
        return float(bbox[2]) * float(bbox[3]) if len(bbox) >= 4 else 0.0

    parts_sorted = sorted(parts, key=_part_area, reverse=True)
    parts_to_zoom = parts_sorted[:MAX_PARTS_TO_ZOOM]

    log.info(f"[ref-analyst] Stage C: zooming {len(parts_to_zoom)}/{len(parts)} parts")
    with ThreadPoolExecutor(max_workers=4) as exe:
        futures = {exe.submit(analyze_part, ref_path, p, obj_bbox): p for p in parts_to_zoom}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                p["detail"] = fut.result()
            except Exception as e:
                log.warning(f"[ref-analyst] Stage C future failed ({p['name']}): {e}")
                p["detail"] = {}

    # Ensure all parts have a detail key
    for p in parts:
        p.setdefault("detail", {})

    # ── Stage D: aggregation ───────────────────────────────────────────────
    geometry_spec_generated: dict = {}
    feature_checklist_generated: dict = {}

    for p in parts:
        name = p["name"]
        detail = p.get("detail", {})
        wc = _px_to_world(p.get("bbox_px", []), obj_bbox)

        # Merge dominant_rgb: prefer detail (zoomed crop is more accurate)
        rgb_raw = detail.get("dominant_rgb") or p.get("dominant_rgb") or [128, 128, 128]
        rgb_01 = [round(max(0.0, min(1.0, v / 255.0)), 3) for v in rgb_raw[:3]]

        sec_raw = detail.get("secondary_rgb")
        sec_01 = [round(max(0.0, min(1.0, v / 255.0)), 3) for v in sec_raw[:3]] if sec_raw else None

        roughness_map = {
            "very_glossy": 0.08, "glossy": 0.15, "satin": 0.28, "matte": 0.45
        }
        roughness_est = roughness_map.get(detail.get("roughness_estimate", "satin"), 0.28)
        metalness_est = 0.75 if detail.get("is_metallic") else 0.10

        geometry_spec_generated[name] = {
            "primitive":        detail.get("geometry_primitive", p.get("geometry_hint", "box")),
            "world_pos":        [wc.get("x", 0.0), wc.get("y", 0.0), 0.0],
            "world_size":       [wc.get("w", 0.3), wc.get("h", 0.3), wc.get("depth", 0.44)],
            "dominant_rgb_01":  rgb_01,
            "secondary_rgb_01": sec_01,
            "color_gradient":   detail.get("color_gradient", "none"),
            "roughness":        roughness_est,
            "metalness":        metalness_est,
            "emissive":         bool(detail.get("emissive_hint", False)),
            "has_texture":      bool(detail.get("has_detail_texture", False)),
            "texture_desc":     detail.get("texture_description"),
            "sub_features":     detail.get("visible_sub_features", []),
        }

        feature_checklist_generated[name] = _checklist_question(p)

    result = {
        "_ref_sha":     ref_sha,
        "_asset":       asset_name,
        "_generated_at": __import__("datetime").datetime.now().isoformat(),
        "silhouette":   sil,
        "parts":        parts,
        "part_count":   len(parts),
        "geometry_spec_generated":   geometry_spec_generated,
        "feature_checklist_generated": feature_checklist_generated,
    }

    EXTENDED_BRIEF_PATH.write_text(json.dumps(result, indent=2))
    log.info(
        f"[ref-analyst] extended_brief.json written "
        f"({len(parts)} parts, {len(parts_to_zoom)} zoomed)"
    )
    return result


def load_extended_brief() -> dict:
    """Load cached extended_brief.json, or return empty dict."""
    try:
        if EXTENDED_BRIEF_PATH.exists():
            return json.loads(EXTENDED_BRIEF_PATH.read_text())
    except Exception:
        pass
    return {}


def format_geometry_table(extended_brief: dict, max_parts: int = 15) -> str:
    """
    Compact one-line-per-part geometry table for injection into coder prompts.
    Stays under ~400 tokens even for complex assets.
    """
    spec = extended_brief.get("geometry_spec_generated", {})
    if not spec:
        return ""
    lines = ["## REFERENCE-MEASURED PART SPECS (from reference image analysis)"]
    lines.append("Follow these positions, sizes, and colors precisely.\n")
    lines.append(f"{'Part':<20} {'Prim':<12} {'Pos (x,y)':<16} {'Size (w,h)':<16} {'Color (rgb01)':<20} {'R':<6} {'M':<6}")
    lines.append("-" * 100)
    for name, s in list(spec.items())[:max_parts]:
        pos = s.get("world_pos", [0, 0, 0])
        sz = s.get("world_size", [0, 0, 0])
        rgb = s.get("dominant_rgb_01", [0.5, 0.5, 0.5])
        prim = s.get("primitive", "box")[:12]
        r_est = s.get("roughness", 0.25)
        m_est = s.get("metalness", 0.5)
        tex = f" [{s['texture_desc'][:20]}]" if s.get("texture_desc") else ""
        lines.append(
            f"{name:<20} {prim:<12} "
            f"({pos[0]:+.2f},{pos[1]:+.2f})    "
            f"({sz[0]:.2f},{sz[1]:.2f})      "
            f"({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})   "
            f"{r_est:.2f}   {m_est:.2f}"
            f"{tex}"
        )
    return "\n".join(lines) + "\n"

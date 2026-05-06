# threejs_loop — Master Improvement Plan (Final v2)
_Revised after self-critique of v1. This is the implementation guide._
_2026-05-06_

---

## Guiding principles

1. **Viewer first.** The user cannot evaluate any improvement without seeing the result. Build the viewer before touching anything else.
2. **Lighting before viewer.** The viewer should show the correct lighting from day one. Fix the rig first (10 min), then build the viewer.
3. **Fix score formula before adding bonuses.** The scoring total must stay ≤ 1.00 — new bonuses replace existing weight, they don't add to it.
4. **Low-poly style only.** The target is a fantasy RPG low-poly game. PBR env maps and cartoon outline passes are out of scope for now; implement low-poly `flatShading` only.
5. **Describe-then-compare for reference images.** The current vision model doesn't support multi-image prompts. Use the existing describe-then-structure pattern: describe reference separately, describe render separately, synthesise similarity with a text LLM.
6. **No feature flags or backwards compat.** Change the code directly.

---

## What is already done (do not redo)

- Scoring formula restructured (coverage, uniformity, diversity) ✓
- COLOR_DIVERSITY_THRESHOLD = 12 ✓
- COVERAGE_UNIFORMITY_THRESHOLD = 0.70 ✓
- PASS_THRESHOLD = 0.92 ✓
- Parallel A/B LLM generation (ThreadPoolExecutor) ✓
- `_run_node`, `_build_rules_block`, `_read_json` helpers ✓
- `_score_bottleneck`, `read_episodic_summary` ✓
- `brief.json` created, injected into coder + planner ✓
- `run_benchmarks.py` calibrated and passing ✓

---

## Implementation phases

---

### PHASE 1 — Lighting fix (10 min)
**File:** `headless/render_scene.mjs`

Reduce ambient, increase sun intensity, enable shadow casting from asset to floor grid. This makes shading gradients stronger and color_diversity naturally higher — improving the metric without changing the formula.

**Change:**
```js
// Before
const sun = new THREE.DirectionalLight(0xfff5e0, 1.6);
scene.add(new THREE.AmbientLight(0x8899bb, 0.55));

// After
const sun = new THREE.DirectionalLight(0xfff5e0, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(10, 10),
  new THREE.ShadowMaterial({ opacity: 0.18 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);
scene.add(new THREE.AmbientLight(0x8899bb, 0.20));
```

Also apply to viewer (Phase 2) — use identical constants so viewer matches renders.

**Verify:** re-run `python run_benchmarks.py`. Color_diversity values for rendered assets should increase. Failure modes should still score 0.000.

---

### PHASE 2 — Interactive viewer
**New files:** `viewer/index.html`, `serve_viewer.py`

#### viewer/index.html
Full-screen Three.js scene with OrbitControls. Features:
- **importmap** maps 'three' and 'three/addons/' to node_modules (same as render pipeline)
- Asset loaded from `?asset=<relative_path>` query param; falls back to reading `loop_state.json` best version
- Same auto-normalise+centre code as render_scene.mjs
- Same lighting rig as render_scene.mjs (import constants, don't hardcode separately)
- Grid floor (matching render_scene.mjs)
- OrbitControls from `three/addons/controls/OrbitControls.js`
- **Stats overlay** (bottom-left): reads `metrics/<version>_<latest>.json` for score, tris, normals, uvs, iteration — refreshes every 3 seconds
- **Auto-swap**: polls `loop_state.json` every 3 seconds; if `current_version` changes, reloads the asset (no full page reload — re-imports the module)
- Background: `#1a1a2e` matching render pipeline

#### serve_viewer.py
```python
#!/usr/bin/env python3
import http.server, socketserver, webbrowser, sys
from pathlib import Path

PORT = 8765
LOOP_DIR = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(LOOP_DIR), **kw)
    def log_message(self, *a): pass  # suppress request noise

version = sys.argv[1] if len(sys.argv) > 1 else None
url = f"http://localhost:{PORT}/viewer/?asset=candidates/rock_v1.mjs" if version else f"http://localhost:{PORT}/viewer/"

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Viewer: {url}")
    webbrowser.open(url)
    httpd.serve_forever()
```

**Race condition guard:** viewer catches `SyntaxError` / import failures (file being written) and retries on the next 3-second poll tick.

**Path traversal:** Python's `SimpleHTTPRequestHandler` with `directory=` constrains to LOOP_DIR.

---

### PHASE 3 — Low-poly style rendering
**File:** `headless/render_scene.mjs`, `viewer/index.html`, `brief.json`

Add `"render_style": "low-poly"` to `brief.json`. Read it in both render_scene.mjs and viewer/index.html.

**In render_scene.mjs:**
After `mod.createAsset()` is called and before rendering, if style=low-poly:
```js
asset.traverse(child => {
  if (child.isMesh && child.material) {
    const mats = Array.isArray(child.material) ? child.material : [child.material];
    mats.forEach(m => { m.flatShading = true; m.needsUpdate = true; });
  }
});
```

Pass the style as a `--style low-poly` flag to render_scene.mjs. Default is no override (respect the material's own flatShading setting).

**In critic prompt:** append to system prompt when style=low-poly:
```
Style target: low-poly stylised. Evaluate for:
- Faceted appearance (flatShading or low subdivision count)
- Readable silhouette at small game-camera distance
- No unnecessary smoothness or subdivision
- Bold, slightly stylised proportions rather than photorealistic accuracy
Penalise: over-subdivision, imperceptible surface detail, overly smooth normals.
```

**In coder prompts (via brief.json style_notes):** already injected via `_brief_block`. Add: "For low-poly style: keep detail count ≤ IcosahedronGeometry(1, 2). Prefer faceted over smooth. Do NOT call `computeVertexNormals()` — let Three.js use face normals for the faceted look, unless vertex displacement makes it necessary."

**Note:** For PBR and cartoon styles — defer. Not needed for current game target.

---

### PHASE 4 — Material quality rules
**Files:** `threejs_loop.py` (rules + critic prompt), `headless/render_scene.mjs` (emissive)

#### 4.1 Roughness/metalness in rules.json
At session start (in `run_loop`), ensure these rules exist in rules.json:
```python
MATERIAL_RULES = [
    "Stone, rock, wood, dirt: roughness 0.75–0.95, metalness 0.0–0.05",
    "Metal (ore, armour, coin): roughness 0.1–0.35, metalness 0.7–1.0",
    "Crystal, glass: roughness 0.0–0.15, metalness 0.0, add emissive glow",
    "Always set roughness and metalness explicitly — never leave as default",
]
```
Call `append_rule(r)` for each at session start if not present.

#### 4.2 Emissive support in brief.json
Add `"emissive": false` field to brief.json. Users set it to true for glowing/crystal assets.
Inject in coder prompt: "This asset requires emissive: add `emissive: new THREE.Color(0x...)` and `emissiveIntensity: 0.4` to the material."

#### 4.3 Critic checks roughness/metalness
Add to critic prompt:
```
Also check: does the material have explicitly set roughness and metalness values?
Are they appropriate for this asset type? Flagging wrong values as a 'major' issue.
```

---

### PHASE 5 — Reference image system
**Files:** `threejs_loop.py`, `llm_router.py`, `threejs_loop.py` (calculate_score)

#### 5.1 Reference image detection
```python
REFERENCES_DIR = LOOP_DIR / "references"

def find_reference(asset_name: str) -> Path | None:
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = REFERENCES_DIR / f"{asset_name}.{ext}"
        if p.exists():
            return p
    return None
```

#### 5.2 Reference comparison — describe-then-compare pattern
Two vision calls (model supports only one image per call):

**Call 1:** describe the reference image
```
"Describe this reference image for a 3D game asset called '{name}'.
In 3–4 sentences: shape, proportions, surface texture style, colors, overall character."
```

**Call 2:** describe the render (this is already done in `run_visual_analysis`)
The existing `_vision_description` from the main visual analysis can be reused.

**Call 3 (text LLM):** synthesise similarity
```
Reference description: "{ref_desc}"
Render description: "{render_desc}"

Rate the visual similarity of the render to the reference for a game asset called '{name}'.
Return JSON: {"similarity": 0-10, "match_notes": "what matches", "gap_notes": "what differs most"}
```

Add `reference_similarity` to the visual dict. Store in metrics.

#### 5.3 Reference score contribution
Replace 0.05 of the existing coverage>10% bonus:
```python
# In calculate_score():
ref_sim = (metrics.get("visual_analysis") or {}).get("reference_similarity", 0)
if ref_sim >= 7:
    score += 0.05   # replaces coverage>10% bonus when reference available
elif coverage > 0.10:
    score += 0.05   # original coverage bonus if no reference
```
Total score stays ≤ 1.00. Reference similarity acts as a smarter version of coverage when available.

**Note on open question — multi-image models:** If switching to `google/gemini-2.0-flash-001` on OpenRouter proves reliable, consolidate to a single two-image comparison call. Test first; don't break the fallback path.

---

### PHASE 6 — Recognisable bonus
**File:** `threejs_loop.py`

#### 6.1 Pass visual dict into calculate_score
Current signature: `calculate_score(metrics: dict) -> float`
The `metrics` dict already has a `stitched_path` key. Add a `visual_analysis` key in `run_loop` before scoring:

```python
metrics["visual_analysis"] = visual   # visual from run_visual_analysis()
score = calculate_score(metrics)
```

Also update evaluate_ab_candidates: run visual analysis on the winning candidate after selecting it.

#### 6.2 Recognisable in calculate_score
Replace 0.05 of coverage_uniformity bonus:
```python
visual_analysis = metrics.get("visual_analysis") or {}
recognisable = visual_analysis.get("recognisable", None)
if recognisable is True:
    score += 0.05          # replaces half of uniformity
elif uniformity >= COVERAGE_UNIFORMITY_THRESHOLD:
    score += 0.05          # original uniformity half-bonus
```
**Important:** total weight is unchanged at 1.00.

---

### PHASE 7 — Vertex color support
**Files:** `headless/validate_geometry.mjs`, `target_adapter.py`, `threejs_loop.py`

#### 7.1 Detect vertex colors in validator
Add to `validate_geometry.mjs`:
```js
const hasVertexColors = !!geo.attributes.color;
// include in output JSON
```

#### 7.2 Vertex color diversity metric
In `_analyze_image()` in target_adapter.py, the color_diversity metric already captures overall pixel RGB variation — vertex-colored assets naturally produce higher scores. No formula change needed.
The key addition: make `has_vertex_colors` visible in geometry output so the critic can reference it.

#### 7.3 Coder prompt addition
In `_brief_block()` output for organic low-poly assets, add:
```
Vertex color technique (optional but encouraged for organic assets):
  const colors = new Float32Array(positions.count * 3);
  for (let i = 0; i < positions.count; i++) {
    const y = positions.getY(i);
    colors[i*3] = 0.45 + Math.random()*0.1;   // R: earth tone
    colors[i*3+1] = 0.32 + y * 0.05;           // G: vary with height
    colors[i*3+2] = 0.22;                       // B: fixed
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  material.vertexColors = true;
```

---

### PHASE 8 — On-pass export
**File:** `threejs_loop.py`

When an asset passes (score ≥ PASS_THRESHOLD), in addition to logging:
1. Copy best `.mjs` to `dist/<asset_name>.mjs`
2. Generate a thumbnail: run render_scene.mjs in single mode (1024×1024), save to `dist/<asset_name>_thumb.png`
3. Write `dist/<asset_name>_meta.json`: score, tris, normals, uvs, version, timestamp, description
4. Append a completion entry to a `dist/manifest.json` (index of all passed assets)

```python
def export_passed_asset(state: dict, metrics: dict, score: float) -> None:
    dist = LOOP_DIR / "dist"
    dist.mkdir(exist_ok=True)
    name = state["asset_name"]
    best_path = candidate_path(name, state["best_version"])
    # ... copy, thumbnail render, meta write
```

---

### PHASE 9 — Architecture cleanup
**Files:** `target_adapter.py`, `threejs_loop.py`

#### 9.1 Graveyard for geometry-failing candidates
When both A and B fail geometry validation (not basic JS validation — JS-failing files are not written):
```python
graveyard = CANDIDATES_DIR / "graveyard"
graveyard.mkdir(exist_ok=True)
for p in (path_a, path_b):
    if p.exists():
        p.rename(graveyard / p.name)
# append to graveyard.jsonl
```

Add `read_graveyard_summary(n=5) -> str` — inject recent failures into planner prompt alongside episodic memory.

#### 9.2 Extract render_and_analyze() from evaluate_ab_candidates
The 15-line render+stitch+analyze block in `evaluate_ab_candidates` duplicates `adapter.run()` logic. Extract to `adapter.render_and_analyze(asset_path, tag, ts) -> dict` and call it from both places.

---

## Score formula — final state after all phases

```
0.10  geo loads
0.20  poly ideal (200–2000) / 0.10 ok / 0.02 any
0.15  normals
0.15  uvs
0.10  renders
0.05  coverage > 2%
0.10  uniformity ≥ 0.70   OR   recognisable = True  (Phase 6)
0.10  diversity > 12
0.05  coverage > 10%       OR   reference_sim ≥ 7   (Phase 5)
─────
1.00 max (capped)
```

The OR logic means the bonuses are replaced when better signals are available, never stacked past 1.00.

---

## Revised implementation order

| # | Phase | Task | Est. | Complexity |
|---|---|---|---|---|
| 1 | P1 | Lighting fix (render_scene.mjs) | 10m | Trivial |
| 2 | P2 | viewer/index.html + serve_viewer.py | 90m | Medium |
| 3 | P3 | Low-poly flatShading style | 30m | Easy |
| 4 | P4.1–4.3 | Material rules + critic + emissive brief | 30m | Easy |
| 5 | P5 | Reference image — detection + describe-then-compare | 2h | Medium |
| 6 | P6 | Recognisable bonus wired into score | 30m | Easy |
| 7 | P7 | Vertex color validator + coder prompt | 45m | Easy |
| 8 | P8 | On-pass export to dist/ | 45m | Medium |
| 9 | P9 | Graveyard + render_and_analyze | 45m | Easy |

**Total estimate: ~7 hours of implementation.**

---

## Benchmark checkpoints

Run `python run_benchmarks.py` after:
- Phase 1 (verify color_diversity rises, failures still 0.000)
- Phase 3 (verify low-poly rendering doesn't break scoring)
- Phase 6 (verify score total still ≤ 1.00)
- Phase 8 (full regression check)

---

## What this plan does NOT include

- PBR HDR environment maps (not needed for current game style)
- Cartoon outline pass / EffectComposer (not the target style)
- Auto-generation of reference images via DALL-E/SD (deferred — user provides reference)
- CLIP embedding similarity (requires separate API, high cost)
- Multi-asset batch generation
- Rig for evaluating at game camera distance/FOV

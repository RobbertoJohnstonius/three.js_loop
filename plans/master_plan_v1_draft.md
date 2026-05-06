# threejs_loop — Master Improvement Plan (Draft v1)
_Covering: viewer, style system, lighting, materials, reference images, scoring, loop architecture_
_Drafted: 2026-05-06_

---

## Background

The benchmark suite and design review revealed two categories of problems:

**Scoring is blind to appearance.** A plain sphere scores 1.000. A displaced rock scores 1.000. The loop optimises geometry metrics and pixel statistics — it has no concept of style, recognisability, or visual intent. It never compares the render to any target image or description beyond the asset name.

**Output is invisible.** There is no interactive viewer. The only output is 4-angle 512×512 PNG screenshots stitched into a grid. The user cannot orbit, inspect normals, or see how the asset responds to different lighting without re-running the pipeline manually.

**Material quality is shallow.** Every asset uses a single-color `MeshStandardMaterial` with no texture maps, no vertex colors, and the lighting rig produces weak shading gradients. Color diversity never exceeds 24 in benchmarks. The system can't express style (low-poly vs PBR vs cartoon) at either the generation or evaluation level.

---

## Phase 1 — Interactive Viewer

**Why first:** Every subsequent phase produces visual output. The viewer makes improvements immediately tangible. Without it, the user must trust the PNG grid.

### 1.1 viewer/index.html
A full-screen Three.js scene served from LOOP_DIR:
- Same lighting rig as render_scene.mjs (three-point: sun + fill + ambient)
- OrbitControls for rotate/zoom/pan
- Same auto-normalise+centre code as the renderer
- Grid floor for spatial reference
- Asset loaded from `?asset=<path>` query param (default: reads loop_state.json best version)
- Poll loop_state.json every 2 seconds — auto-swaps to new best without page reload
- Stats overlay: score, triangle count, normals, UVs, iteration count

### 1.2 serve_viewer.py
A 3-line Python HTTP server:
```bash
python serve_viewer.py          # opens at http://localhost:8765
python serve_viewer.py rock_v3  # loads specific version
```
Serves LOOP_DIR as the web root (same as render_scene.mjs does internally).

### 1.3 Style-matched viewer
When brief.json specifies a style, the viewer applies the same rendering variant as the render pipeline (flat shading, toon shader, HDR env map). What you see in the viewer matches what the pipeline scores.

---

## Phase 2 — Style System

**Why second:** Style affects rendering, critique, scoring, and coder prompts. Everything downstream depends on it.

### 2.1 brief.json style parameter
Add `"render_style"` to brief.json:
```json
{
  "render_style": "low-poly",    // "low-poly" | "pbr" | "cartoon"
  "style_notes": "Faceted appearance, no smooth normals, saturated earth tones"
}
```

### 2.2 Style-specific render variants in render_scene.mjs
Three scene HTML templates, selected by `--style` flag:

**low-poly:**
- `flatShading: true` on material
- Stronger directional light, less fill
- Disable `computeVertexNormals()` in renderer (preserve facets)

**pbr:**
- `PMREMGenerator` with an HDR env map (use Three.js RoomEnvironment as fallback)
- `renderer.toneMapping = THREE.ACESFilmicToneMapping`
- `renderer.toneMappingExposure = 1.0`

**cartoon:**
- Outline pass (EffectComposer + OutlinePass) or MeshToonMaterial swap
- Solid background, no grid

### 2.3 Style-aware critic prompt
The critic receives the style from brief.json and evaluates against it:
- **low-poly**: "Is the geometry faceted? Does it look hand-crafted? Is it over-smooth?"
- **pbr**: "Do the materials respond physically to lighting? Are roughness/metalness values physically plausible?"
- **cartoon**: "Are the forms bold and readable at small size? Are colors saturated?"

### 2.4 Style injected into coder prompts
The brief block (already injected) includes style_notes. Add explicit technique constraints per style:
- **low-poly**: `flatShading: true`, avoid `computeVertexNormals`, use IcosahedronGeometry with low detail
- **pbr**: use high-detail geometry, add normal variation, multiple material zones
- **cartoon**: `MeshToonMaterial`, thick outlines, solid fill colors

---

## Phase 3 — Lighting and Material Quality

**Why third:** Better lighting makes the visual quality metrics more meaningful, and better materials make assets more useful in the actual game.

### 3.1 Lighting rig improvements in render_scene.mjs
Current: `sun=1.6, fill=0.35, ambient=0.55` — ambient is too high, washes out shadows.
Proposed:
```js
sun:     DirectionalLight(0xfff5e0, 2.2)   // stronger key
fill:    DirectionalLight(0x4466cc, 0.25)   // dimmer, cooler
ambient: AmbientLight(0x8899bb, 0.20)       // much lower — let shadows breathe
```
Also: enable shadow casting from asset onto the grid floor.
Effect: color_diversity should naturally rise to 20–30+ for well-lit geometry, making the metric more discriminating.

### 3.2 Vertex color support
Add to coder system prompt and brief.json technique notes:
```js
// Vertex color painting (organic assets — no UV atlas needed)
const colors = new Float32Array(pos.count * 3);
// per-vertex RGB variation based on position/noise
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
material.vertexColors = true;
```
Add `has_vertex_colors` to geometry validator output.
Add vertex color diversity to pixel analysis as a separate metric.

### 3.3 Roughness/metalness validation
Add to rules.json at init time (if not already present):
- "Stone/rock/wood assets: roughness 0.7–0.95, metalness 0.0–0.05"
- "Metal assets: roughness 0.1–0.4, metalness 0.7–1.0"
- "Crystal assets: roughness 0.0–0.2, metalness 0.0, use emissive for glow"

Add to critic prompt: "Check roughness and metalness values against the asset type. Flag physically implausible values."

### 3.4 Emissive support
Add to generate_initial_asset and run_coder prompts:
"If the asset description includes 'glowing', 'crystal', 'rune', or 'magic', add `emissive: new THREE.Color(0x...)` and `emissiveIntensity: 0.3–0.8` to the material."

---

## Phase 4 — Reference Image System

**Why fourth:** The reference image is the most powerful tool for closing the loop between "correct metrics" and "looks right". Everything before this phase improves the proxy metrics; this phase connects to ground truth.

### 4.1 Reference image detection
`references/<asset_name>.png` (or .jpg, .webp) — user drops a file here.
Sources (any of these work):
- A photo of a real-world object (rock, tree trunk, barrel)
- Concept art or an illustration from the target game style
- An AI-generated image from DALL-E or Stable Diffusion using the asset description
- A frame from a reference game (e.g. Zelda: Wind Waker for cartoon, Dark Souls for PBR)

If no reference file exists, the system skips this step silently.

### 4.2 Reference comparison in vision pipeline
Add a second vision call in `run_visual_analysis()`:
```
"Here are two images:
  LEFT: a reference image of a {asset_name}
  RIGHT: a render of the current 3D asset
Rate visual similarity 0–10 and list the 3 most important differences."
```
Extract: `{"similarity": 7, "differences": ["too round", "missing surface texture", "wrong color"]}`

Use a model that supports multi-image input (GPT-4o, gemini-flash-1.5, or llava-next).

### 4.3 Reference similarity in score
Add to `calculate_score()`:
- similarity ≥ 7: +0.05
- similarity ≥ 9: +0.08 (replaces the 0.05)
This caps at 0.08 so it can't single-handedly pass an asset.

---

## Phase 5 — Score Improvements (Remaining)

### 5.1 Wire recognisable flag into score
`run_visual_analysis()` already returns `recognisable: bool`.
Add to `calculate_score()`: if `visual.get('recognisable') == True`, +0.05.
This requires passing the visual dict into calculate_score — currently it doesn't receive it.
The visual result is available in metrics via a new `visual_analysis` key.

### 5.2 Style conformance score
If brief.json has `render_style`, the critic assigns a 0–5 style conformance rating.
Map to: ≥4 = +0.05, ≥5 = +0.08.

### 5.3 Re-run benchmarks after each phase
After each phase, run `python run_benchmarks.py` and verify:
- Failure modes still score 0.000
- Score staircase is preserved
- No unintended regressions

---

## Phase 6 — Architecture Cleanup

### 6.1 Graveyard for failed candidates
When both A/B candidates fail geometry validation, move them to `candidates/graveyard/` instead of leaving them in `candidates/`. Append entry to `graveyard.jsonl` with: plan, fail_category, js_preview (first 20 lines).

The planner receives recent graveyard entries to avoid repeating the same failures.

### 6.2 Extract render_and_analyze() from evaluate_ab_candidates
The render+stitch+analyze block inside `evaluate_ab_candidates()` in threejs_loop.py is 15 lines that duplicate what `adapter.run()` already does in a slightly different way. Extract to `adapter.render_and_analyze(path, tag, ts) -> dict` and use it in both places.

### 6.3 A3 from improvement_plan.md — recognisable bonus
Already captured in Phase 5.1.

---

## Implementation sequence

| # | Phase | Task | Files | Est. | Priority |
|---|---|---|---|---|---|
| 1 | P1 | viewer/index.html + serve_viewer.py | viewer/index.html, serve_viewer.py | 1h | NOW |
| 2 | P3.1 | Lighting rig (reduce ambient, stronger sun) | render_scene.mjs | 10m | NOW |
| 3 | P2.1–2.2 | Style parameter + render variants | brief.json, render_scene.mjs | 2h | HIGH |
| 4 | P2.3 | Style-aware critic prompt | threejs_loop.py | 30m | HIGH |
| 5 | P3.2 | Vertex color support | validate_geometry.mjs, threejs_loop.py | 1h | MED |
| 6 | P3.3 | Roughness/metalness rules | threejs_loop.py | 20m | MED |
| 7 | P4.1–4.2 | Reference image detection + vision comparison | threejs_loop.py, llm_router.py | 2h | MED |
| 8 | P4.3 + P5.1 | Reference + recognisable scoring | threejs_loop.py | 30m | MED |
| 9 | P3.4 | Emissive support in prompts | threejs_loop.py | 20m | LOW |
| 10 | P5.2 | Style conformance score | threejs_loop.py | 30m | LOW |
| 11 | P6.1 | Graveyard | threejs_loop.py | 45m | LOW |
| 12 | P6.2 | render_and_analyze() extraction | target_adapter.py, threejs_loop.py | 30m | LOW |

---

## Open questions

1. **Reference image source**: Should the loop auto-generate a reference image using DALL-E/SD at session start if none exists? Or always require the user to provide one?
2. **Cartoon style**: MeshToonMaterial requires no PBR workflow — should the coder be instructed to swap material type, or use post-processing?
3. **Score total**: After adding reference (+0.08) + recognisable (+0.05) + style_conformance (+0.08), max score becomes 1.21 → needs renormalisation or a cap adjustment
4. **Multi-image vision**: The current vision model (llama-3.2-11b via OpenRouter) may not support two images in one prompt. Need to test or switch to gpt-4o/gemini for reference comparison.
5. **Viewer auto-open**: Should serve_viewer.py open the browser automatically on start?

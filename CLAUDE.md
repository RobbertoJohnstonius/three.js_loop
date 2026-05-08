# threejs_loop — Claude context

## What this project does
Autonomous AI loop that generates, scores, and iterates Three.js game assets (`.mjs` files).
Pipeline: Playtester → Vision (Groq) → Critic → Planner → Coder A/B → eval → repeat.
Auto-backs up to git@github.com:RobbertoJohnstonius/three.js_loop.git at end of each run.

## Running the loop
```bash
source .env   # sets OPENROUTER_API_KEY and GROQ_API_KEY
python3 threejs_loop.py boab_tree "description"   # new asset
python3 threejs_loop.py --resume                  # continue
```
Viewer: `python serve_viewer.py --port 8766` → http://localhost:8766/viewer/

## API keys (env vars only — never hardcode)
- `OPENROUTER_API_KEY` — primary coder + critic (qwen-2.5-72b, llama-3.3-70b)
- `GROQ_API_KEY` — primary vision (llama-4-scout-17b, ~1s vs 24s on OpenRouter)
- Keys live in `.env` (gitignored). Never hardcode a fallback value.

## Critical geometry rules (hard-won fixes)

### 1. UV seam crack — THE most important rule
`IcosahedronGeometry` ships with 9 vertex pairs at the same 3D position but u=0 vs u=1.
`mergeVertices()` keeps them separate because UV differs → LCG gives them different
displacements → physical seam crack appears.

**Fix: always delete UV BEFORE mergeVertices():**
```javascript
let geo = new THREE.IcosahedronGeometry(r, d);
geo.deleteAttribute('uv');          // ← CRITICAL: removes seam duplicates
geo = mergeVertices(geo);           // now 92 unique verts (detail=2), connectivity=1.0
// ... displace vertices ...
geo.computeVertexNormals();
// ... set UVs manually after ...
geo.setAttribute('uv', uvAttr);
```
Without this: connectivity=0.963, visible seam crack. With it: connectivity=1.0, no crack.

### 2. Sequential LCG on non-indexed geometry
`IcosahedronGeometry` is non-indexed (540 vertex copies for detail=2×3 before merge).
LCG displacement by buffer index gives each copy of the same position a different value
→ faces tear apart (shared_edge_fraction → 0). `mergeVertices()` fixes this.

### 3. Displacement amplitude limit
Keep displacement ≤ 40% of face edge length to prevent face folding (visible splits).
- detail=2 (r=0.75): edge≈0.39u → max safe disp ≈ 0.08u
- detail=3 (r=0.75): edge≈0.26u → max safe disp ≈ 0.10u
Exceeding causes inward face folding → MeshStandardMaterial backface-culls → crack.

### 4. yFactor formula — taper at poles to avoid spike
Wrong (old): `0.5 + 0.5 * (y/len + 1) * 0.5` → gives 1.0 at top (maximum!) → spike
Right: `0.3 + 0.7 * (1 - yNorm * yNorm)` → 1.0 at equator, 0.3 at poles

### 5. Per-face flat vertex colors (low-poly style)
After mergeVertices() + displacement, call `geo.toNonIndexed()` then
`geo.computeVertexNormals()`. In a non-indexed geometry each vertex belongs to exactly
one face → computeVertexNormals gives true flat (per-face) normals.
Compute color from the **face centroid** (average of 3 vertex positions) and assign the
same RGB to all 3 vertices. This gives discrete facet colors with no cross-face bleed.
Connectivity scoring is position-based so toNonIndexed() does NOT reduce shared_edge_fraction.

```javascript
geo = geo.toNonIndexed();
geo.computeVertexNormals();
const fpos = geo.attributes.position;
const nFaces = Math.floor(fpos.count / 3);
const colors = new Float32Array(fpos.count * 3);
for (let f = 0; f < nFaces; f++) {
  const i0=f*3, i1=f*3+1, i2=f*3+2;
  const cx=(fpos.getX(i0)+fpos.getX(i1)+fpos.getX(i2))/3;
  const cy=(fpos.getY(i0)+fpos.getY(i1)+fpos.getY(i2))/3;
  const cz=(fpos.getZ(i0)+fpos.getZ(i1)+fpos.getZ(i2))/3;
  const t = hash(cx, cy, cz); // must return [0,1]
  const [r,g,b] = palette(t);
  for (const idx of [i0,i1,i2]) { colors[idx*3]=r; colors[idx*3+1]=g; colors[idx*3+2]=b; }
}
geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
```

### 6. Vertex colors + material base color multiply
When `vertexColors: true`, always set `color: 0xffffff` on the material.
Using `color: 0x777777` halves effective brightness — under the headless renderer's 0.20
ambient, this pushes shadow pixels below 20/255 (dark_patch threshold) → false normal penalty.

### 7. Hash function must return [0, 1]
`Math.sin(x*a + y*b + z*c)` returns [-1, 1]. Sum of three such terms is still [-3, 3].
Always normalize: `((h1 + h2 + h3) / 3 + 1) * 0.5` → [0, 1] before palette lerp.
Failing to do this makes r/g/b channels go negative → GPU clamps to 0 → blue/black patches.

### 8. Custom capped tube geometry for curved tubes (TubeGeometry open-end fix)
TubeGeometry with closed=false has boundary edges at both ends → shared_edge_fraction < 0.97 → FAILED.
Fix: build a custom BufferGeometry with cap-center vertices and triangle fan caps.

**CRITICAL winding rule** — side quads MUST use `idxs.push(a, d, c, a, b, d)`:
- a=ring_i[j], b=ring_i[(j+1)%R], c=ring_{i+1}[j], d=ring_{i+1}[(j+1)%R]
- The WRONG winding `push(a,c,d, a,d,b)` makes both cap-fan and side-quad emit the
  SAME directed edge at end rings → connectivity check never finds the reverse → sharedFraction=0.926

Cap fans: `idxs.push(startCapIdx, (j+1)%R, j)` and `idxs.push(endCapIdx, lr+j, lr+(j+1)%R)`

```javascript
function buildCappedTube(p0, p1, p2, tubeSeg, radSeg, bodyR) {
  const curve = new THREE.QuadraticBezierCurve3(p0, p1, p2);
  const frames = curve.computeFrenetFrames(tubeSeg, false);
  const verts = [], idxs = [];
  for (let i = 0; i <= tubeSeg; i++) {
    const t = i / tubeSeg;
    const c = curve.getPoint(t);
    const N = frames.normals[i], B = frames.binormals[i];
    const r = bodyR * (0.12 + 0.88 * Math.sin(Math.PI * t));
    for (let j = 0; j < radSeg; j++) {
      const ang = (j / radSeg) * 2 * Math.PI;
      const co = Math.cos(ang), si = Math.sin(ang);
      verts.push(c.x+r*(co*N.x+si*B.x), c.y+r*(co*N.y+si*B.y), c.z+r*(co*N.z+si*B.z));
    }
  }
  const startCapIdx = (tubeSeg+1)*radSeg, endCapIdx = startCapIdx+1;
  const cp0 = curve.getPoint(0), cp1 = curve.getPoint(1);
  verts.push(cp0.x, cp0.y, cp0.z, cp1.x, cp1.y, cp1.z);
  for (let i = 0; i < tubeSeg; i++) for (let j = 0; j < radSeg; j++) {
    const a=i*radSeg+j, b=i*radSeg+(j+1)%radSeg, c=(i+1)*radSeg+j, d=(i+1)*radSeg+(j+1)%radSeg;
    idxs.push(a,d,c, a,b,d);  // ← CORRECT winding (outward normals + matched directed edges)
  }
  for (let j=0;j<radSeg;j++) idxs.push(startCapIdx,(j+1)%radSeg,j);
  const lr=tubeSeg*radSeg;
  for (let j=0;j<radSeg;j++) idxs.push(endCapIdx,lr+j,lr+(j+1)%radSeg);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(verts), 3));
  geo.setIndex(idxs);
  return geo;  // pass directly to applyFaceColors — no mergeVertices or deleteAttribute
}
```

## Scoring thresholds
- Pass: score ≥ 0.92
- SHARED_EDGE_MIN = 0.97 (below → -0.40 penalty; seam crack = 0.963 → fails correctly)
- DARK_PATCH_MAX = 0.18 (directional light creates ~10% shadow pixels naturally)
- Headless ambient: only 0.20 intensity — much weaker than viewer. Dark vertex colors
  combined with a non-white material base color easily breach the dark_patch threshold.

## Loop steering: vertex color requirement
`brief.json` `texture_style: vertex-color` → rubric demotes to TIER_POLISH (not PRODUCTION)
until vertex colors are present. The remediation is `polish_items.insert(0, ...)` so it
wins `all_issues[0]` over any vision false-positive. Planner prompt becomes MANDATORY
(overrides critic) when the rubric is TIER_POLISH and remediation starts with "vertex color".

## Vision false-positive normals
Vision often flags low-poly shadow darkening as "inverted normals". Gate on
`dark_patch_fraction > DARK_PATCH_MAX` before escalating to a rubric blocker.
If fraction is within threshold, downgrade to a polish_item (no score impact).

## Viewer modes
Four style buttons: **Flat / Smooth / Wire / Realistic**.
- `swapAsset()` builds two geometry versions per mesh:
  - `_flatGeo` — asset as-is (flat normals for non-indexed, per-face vertex colors)
  - `_smoothGeo` — clone with position-key-averaged normals (`buildSmoothGeo()`)
- `applyStyle()` swaps `child.geometry` per mode — no flatShading flag needed.
- Realistic: `buildRealisticMaterial(assetName)` — world-space FBM GLSL shader,
  asset-type-aware palette (granite/bark/metal/stone), procedural normal bump,
  IBL from `getSkyEnv()`. Ground gets `buildGroundMaterial()`.

## Arrangement topology validation (Phase 3)
`validate_geometry.mjs` now emits `mesh_names` and `body_mesh_positions` (world-space bbox centres
for all meshes named `'body'`). Python `_check_arrangement()` clusters those Y-positions and
adds `arrangement_ok`, `arrangement_body_count`, `arrangement_row_count` to the geometry dict.

`brief.json` `expected_arrangement` drives the check:
```json
{ "body_count": 6, "row_count": 2, "bodies_per_row": 3, "row_axis": "y" }
```
Wrong arrangement → rubric `critical_failures` → `TIER_FAILED` → loop cannot exit until fixed.
Correct arrangement → +5 pts in Category D + +0.05 score bonus.
Wrong arrangement → −0.10 score penalty.

`_check_arrangement(geo, brief)` is called: after `adapter.run()` in the main loop, and after
`validate_geometry()` for both A/B candidates inside `evaluate_ab_candidates()`.

## Camera-aware geometry context (Phase 1)
`brief.json` `camera_config` block carries camera position, target, world→image axis mappings,
and derived axis rules (e.g. "bow in -Y", "fan in Z"). `_camera_basis_block(brief)` formats
this into a text block injected into the Critic and Planner prompts.

Planner post-validation: after LLM returns a plan, if the instruction matches a forbidden-axis
keyword (bow in Z, fan in Y, group.rotation, vertical layout) that corresponds to a
`forbidden_improvements` entry, the planner re-invokes the LLM with an explicit rejection message.

`validate_js_basic(js, forbidden_patterns)` now accepts a list of literal JS code substrings
to block. `_banned_js_patterns(steering)` extracts `group.rotation` from forbidden_improvements.

## Pixel shape metrics (Phase 2)
`target_adapter._measure_shape_metrics()` runs on the top-left quadrant (quarter-view angle):
- `arc_depth_ratio`: top-silhouette vertical variation / horizontal object span. Target ≥ 0.12.
- `coverage_asymmetry`: |left coverage − right coverage|. Target ≤ 0.30.
- `color_zone_boundary`: fraction of object width where green channel dominates (banana assets).

Returned as `pixel_metrics` inside `screenshot_analysis`. Fed to Critic prompt as quantitative
constraints with target comparison. Checked in rubric Category D via `brief.json` `pixel_targets`.

## Loop improvement v3 systems (9 items, 2026-05-07)

### Material roughness range check (Item 1)
`brief.json` `expected_material_properties: { roughness_min, roughness_max }`. `get_rubric()` Category C cross-references `geo.roughness_values` against this range. Out-of-range roughness → rubric failure message directing the coder to fix material properties.

### Code complexity guard (Item 2)
`metrics["code_line_count"]` injected after `adapter.run()`. Category E: >250 lines → polish item; >500 lines → vq_failure. `run_coder()` prompt now says "Make MINIMAL targeted changes — 5–30 lines, not 100+."

### Dominant hue validation (Item 3)
`_measure_shape_metrics()` computes `dominant_hue`, `hue_variance`, `mean_saturation` via HSV conversion of sampled non-background pixels. `brief.json` `expected_color_hsv: { hue_center, hue_tolerance, saturation_min }`. Category B checks hue distance and saturation. `_auto_calibrate_color()` writes the first accepted iteration's observed hue to brief.json if no range is set yet.

### Proportional ratio check (Item 4)
`brief.json` `expected_proportions: { height_to_width_ratio: [min, max] }`. Category D checks `bbox.y / bbox.x` against this range.

### Stagnation detection + structured recovery (Item 5)
`state["stagnation_count"]` tracks iterations where score is stuck (±0.015 over last 4, below threshold). At count=4: `_inject_stagnation_recovery()` writes a geometry-strategy-swap `force_plan` to `steering.json` (IcosahedronGeometry → custom BufferGeometry or vice versa). At count=8: clears the injected plan and logs a human-review warning.

### Auto-steering from accepted improvements (Item 6)
When an iteration is accepted with score gain >0.04, `_auto_append_focus_hint()` prepends the successful planner instruction to `steering.json` focus_hints as `"PREVIOUSLY SUCCESSFUL: ..."`. Keeps at most 2 such entries.

### Prerequisite-aware bottleneck (Item 7)
`_score_bottleneck()` now checks `shared_edge_fraction < SHARED_EDGE_MIN` first, before any other metric. Returns a `PREREQUISITE —` prefixed message that blocks the planner from proposing polish changes while the mesh is torn.

### Failure pattern memory (Item 8)
`_check_failure_patterns(asset_name, outcome)` reads last 6 episodic records. If the same `improvement_type` fails/regresses 3 consecutive times for the same asset, `append_rule()` writes a negative pattern rule to `rules.json`. Called after both `write_episodic()` sites in `run_loop()`.

### Early render skip (Item 9)
In `ThreeJSAdapter.run()`, after geometry validation: if `shared_edge_fraction < 0.85`, skip the Puppeteer render entirely. Returns `{ ok: False, render_skipped: True }` immediately, saving ~15s per badly-torn candidate.

## Auto-dist sync (Phase 6)
`_sync_to_dist(asset_name, version)` copies winning candidate to `dist/<asset_name>.mjs` after
every accepted iteration (not just on final pass). Eliminates the manual `cp` step.

## Reference similarity (Phase 6)
Progressive scoring: +0.05 at `ref_sim >= 5`, +0.05 more at `ref_sim >= 7` (previously a flat
+0.05 at ≥7 only). Max contribution from reference: 0.10 instead of 0.05.

## Mesh type registry (Phase 5 reduced)
`brief.json` `expected_mesh_types` maps mesh names to material type strings:
```json
{ "body": "banana", "tip": "tip", "crown": "crown" }
```
`get_rubric()` cross-references `geo.mesh_names` against this map — missing names become
polish_items flagging the need for a Realistic material case in the viewer.

## Reference images
Place `references/<asset_name>.png` to enable reference comparison scoring.
Current assets with references:
- `references/rock.png` — granite boulder
- `references/boab_tree.png` — Australian boab (bottle-shaped trunk, spreading canopy)
- `references/banana_bunch.png` — hand-picked bunch, 6 bananas in 2 rows of 3
- `references/balloon_bunch.png` — 9 party balloons on rigid white sticks, tight cluster
- `references/croc_pistol.png` — decorative teal/jade pistol with gold rail, orange guard, croc-scale grip

## Vision provider
Groq (llama-4-scout-17b) is primary (~1s). OpenRouter (qwen2.5-vl-72b) is fallback (~24s).
Groq key has no llama-3.2-vision models; llama-4-scout IS multimodal.

## Vision critique pipeline v1 (2026-05-07)
Seven coordinated improvements so the loop "sees" the reference photo accurately instead of
inflating scores via generic prose descriptions.

### Item 1 — Multi-image vision (llm_router.py)
`call_llm_vision(prompt, image_path, ..., image_path_b=None)` now accepts a second image.
Both Groq (llama-4-scout) and OpenRouter paths encode and send two `image_url` blocks.

### Item 2 — 5-pass structured reference spec (threejs_loop.py)
`build_reference_spec(asset_name, ref_path)` runs five targeted vision passes and caches
the result in `brief.json["reference_spec"]` (keyed by SHA-256 of reference file):
- `count_and_identity`: exact object count + confidence
- `shape_and_proportions`: overall_shape, dominant_axis
- `attachment_and_connection`: connector_type (rigid_stick|flexible_string|stem|none|other)
- `spatial_arrangement`: arrangement_type, density (tightly_packed|moderately_spaced|spread_out)
- `surface_and_material`: surface_finish, has_texture_map

Also writes derived brief.json fields:
- `expected_cluster_density_max` (from density label)
- `expected_silhouette_hw_ratio` (from dominant_axis)

Called at loop start (before `start_iter`) if a reference image exists.

### Item 3 — Count hard constraints in rubric (Section F)
`get_rubric()` Section F loads `reference_spec` and compares mesh count vs `ref_count`.
High/medium confidence mismatch → `critical_failures`; low confidence → `polish_items`.
Height variation check: if ref says `all_at_same_height=True` and Y spread > 0.10 → critical.
Connector type check: if ref says `rigid_stick` and no `stick` meshes present → polish.

### Item 4 — Cluster density metric (_check_arrangement)
After row-count logic, `_check_arrangement` computes mean pairwise Euclidean distance between
`body_mesh_positions` centroids (O(n²), capped at 50). Stored as `geo["cluster_density_mean"]`.
Rubric Category D checks against `expected_cluster_density_max` from brief.json.

### Item 5 — Reference spec injection into critic/planner
`_format_reference_spec_block(reference_spec)` formats the spec as a human-readable block
injected into both `run_critic` and `run_planner` prompts. Also propagates
`visual["reference_critical_gap"]` into the critic block when present.

### Item 6 — Dual-image direct comparison (run_reference_comparison)
`run_reference_comparison(asset_name, ref_path, render_description, stitched_path=None)`:
- When `stitched_path` is provided: single dual-image vision call — IMAGE A = reference photo,
  IMAGE B = 4-angle render grid. Returns structured JSON including `reference_count_match`
  and `reference_critical_gap` fields.
- Falls back to legacy describe→text-LLM compare when stitched_path is absent or parse fails.
`run_visual_analysis` now passes `stitched_path` and propagates both new fields to the result.

### Item 7 — Silhouette H:W metrics (target_adapter.py)
`ThreeJSAdapter.compute_silhouette_metrics(image_path)` uses PIL + NumPy to threshold on
`BACKGROUND_COLOR_RGB=(26,26,46)` and compute `silhouette_hw_ratio` and
`silhouette_fill_fraction` from the front-angle ("f") render.
Wired in Stage 3c of `ThreeJSAdapter.run`, merged into `screenshot_analysis["pixel_metrics"]`.
Rubric Category D checks `silhouette_hw_ratio` against `expected_silhouette_hw_ratio` range
from brief.json.

## Git auto-backup
`_git_backup()` runs at end of every loop run — commits changed files and pushes.
Remote: git@github.com:RobbertoJohnstonius/three.js_loop.git

## Reference analyst pipeline (Phase 7 — 2026-05-08)
`reference_analyst.py` runs a 6-pass analysis of the reference image and writes `extended_brief.json`.
Called at loop startup (cached by ref-image SHA256 — only re-runs when the image changes).

### Passes
- **Stage A** — PIL silhouette: object bbox, aspect ratio, dominant palette. No LLM.
- **Stage B** — VLM global part decomposition: lists every named part with bbox_px, dominant_rgb, geometry_hint.
- **Stage C** — VLM per-part zoom: crop + targeted analysis per part (geometry_primitive, color gradient, texture description, roughness/metalness estimates). Top 10 parts by bbox area. 4 workers.
- **Stage E-1** — PIL accurate color: for each part, PIL-median of non-grey crop pixels → accurate `dominant_rgb`, adds `gradient_direction` (horizontal/vertical/none).
- **Stage E-2** — Feature inventory VLM: single pass identifying text labels, decorative elements, connecting elements, engraving zones, emissive areas.
- **Stage F** — Texture crops: PIL-crop each part bbox → 256×256 PNG → `dist/textures/<asset>/<part>.png`.

All Groq calls use `_groq_vision_with_retry()`: exponential backoff (2s, 4s, 8s) on 429 rate-limit responses.

### extended_brief.json fields
- `geometry_spec_generated` — per-part: primitive, world_pos, world_size, dominant_rgb_01, color_gradient, roughness, metalness, has_texture, texture_desc, sub_features
- `feature_checklist_generated` — auto-generated yes/no VLM questions per part (merged with manual brief.json checklist; manual takes precedence)
- `feature_inventory` — {text_labels, decorative_elements, connecting_elements, engraving_zones, emissive_areas}
- `texture_crops` — {part_name: absolute_path} for all saved 256×256 crops

### Integration hooks
- **Coder prompt**: `format_geometry_table(eb)` injects per-part position/size/color table. Texture crops table lists available `dist/textures/<asset>/<part>.png` paths with `has_texture` flag and feature inventory decorative/engraving notes.
- **Critic/planner**: feature_checklist_generated merged into feature checklist for VLM verification.
- **Rubric Category G**: color_zones coverage < 3% → polish_item; missing_parts → polish_item; hw_ratio_delta > 0.25 → polish_item.

## Layer 1 programmatic metrics (Phase 7)
New methods in `ThreeJSAdapter`, wired into Stage 3e of `run()`:

- **3e-i** `_check_color_zones(front_path, eb)` — for each part with saturated expected color, measures fraction of non-bg render pixels within ±40° hue. Returns `{part_name: coverage_fraction}`.
- **3e-ii** `_check_part_presence(geo, eb)` — compares `geo.mesh_names` vs `geometry_spec_generated` keys. Returns `{part_presence: {name: bool}, missing_parts: [name]}`.
- **3e-iii** `_compare_proportions_to_ref(front_path, eb)` — PIL silhouette of render vs reference `object_bbox_px` aspect ratio. Returns `{render_hw_ratio, ref_hw_ratio, hw_ratio_delta}`.

All results merged into `screenshot_analysis["pixel_metrics"]`.

## Layer 2 targeted part critique (Phase 7)
`run_targeted_part_critique(stitched_path, ref_path, visual, max_parts=3)` fires from `run_visual_analysis()` after feature checklist.

Identifies failing parts: missing from `mesh_names` OR color_zone coverage < 5%. For each (up to 3), runs a dual-image VLM call (render=IMAGE A, reference=IMAGE B) focused on that specific part. Returns `[{part, finding, action}]`.

Results stored as `visual["targeted_part_critique"]` and injected into the planner prompt as a `TARGETED PART CRITIQUE` block — highest-priority signal, listed before critic findings.

## Texture pipeline rules (Phase 7)
Two new entries in `rules.json`:
- **CanvasTexture**: for `has_texture=True` parts (scrollwork, engraving, scales). `canvas.getContext('2d')` draw pattern → `new THREE.CanvasTexture(canvas)`. Set `mat.vertexColors = false` — cannot combine with vertexColors.
- **TextureLoader**: for parts with crops in `dist/textures/<asset>/<part>.png`. Served at `/dist/textures/...` by render_scene.mjs built-in HTTP server (works in both headless and viewer). Set `mat.vertexColors = false`.

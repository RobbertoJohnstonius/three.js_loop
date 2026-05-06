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

## Reference images
Place `references/<asset_name>.png` to enable reference comparison scoring.
Current assets with references:
- `references/rock.png` — granite boulder
- `references/boab_tree.png` — Australian boab (bottle-shaped trunk, spreading canopy)

## Vision provider
Groq (llama-4-scout-17b) is primary (~1s). OpenRouter (qwen2.5-vl-72b) is fallback (~24s).
Groq key has no llama-3.2-vision models; llama-4-scout IS multimodal.

## Git auto-backup
`_git_backup()` runs at end of every loop run — commits changed files and pushes.
Remote: git@github.com:RobbertoJohnstonius/three.js_loop.git

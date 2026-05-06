# threejs_loop — Claude context

## What this project does
Autonomous AI loop that generates, scores, and iterates Three.js game assets (`.mjs` files).
Pipeline: Playtester → Vision (Groq) → Critic → Planner → Coder A/B → eval → repeat.

## Running the loop
```bash
GROQ_API_KEY="..." OPENROUTER_API_KEY="..." python3 threejs_loop.py rock "description"
GROQ_API_KEY="..." OPENROUTER_API_KEY="..." python3 threejs_loop.py --resume
```
Viewer: `python serve_viewer.py` → http://localhost:8765/viewer/

## API keys (env vars only — never hardcode)
- `OPENROUTER_API_KEY` — primary coder + critic (qwen-2.5-72b, llama-3.3-70b)
- `GROQ_API_KEY` — primary vision (llama-4-scout-17b, ~1s vs 24s on OpenRouter)

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

## Scoring thresholds
- Pass: score ≥ 0.92
- SHARED_EDGE_MIN = 0.97 (below → -0.40 penalty; seam crack = 0.963 → fails correctly)
- DARK_PATCH_MAX = 0.18 (directional light creates ~10% shadow pixels naturally)

## Viewer/headless consistency
`brief.json` `render_style: low-poly` → headless renderer applies `flatShading: true`.
Viewer now reads `brief.json` and applies the same flat shading.
Without this, smooth normals in viewer show dramatic splits that the scorer never sees.

## Reference images
Place `references/<asset_name>.png` to enable reference comparison scoring.
Current: `references/rock.png` (granite boulder photo).

## Vision provider
Groq (llama-4-scout-17b) is primary (~1s). OpenRouter (qwen2.5-vl-72b) is fallback (~24s).
Groq key has no llama-3.2-vision models; llama-4-scout IS multimodal.

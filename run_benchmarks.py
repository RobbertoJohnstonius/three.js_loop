#!/usr/bin/env python3
"""
run_benchmarks.py — Graduated benchmark suite for the threejs_loop scoring pipeline.

Creates test assets at eight quality levels and runs each through the full
validate_geometry → render_multi_angle → stitch → score pipeline.

Run from threejs_loop directory:
  python run_benchmarks.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from target_adapter import ThreeJSAdapter
from threejs_loop import calculate_score, PASS_THRESHOLD, POLY_IDEAL_MIN, POLY_IDEAL_MAX, POLY_MIN, POLY_MAX

BENCH_DIR = Path(__file__).parent / "tests" / "bench_assets"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

# ── Test asset definitions ─────────────────────────────────────────────────────

ASSETS = []   # list of (name, expected_outcome, js_source)


def asset(name, note, js):
    ASSETS.append((name, note, js))


# ── L0: Hard failures — should score 0.0 ──────────────────────────────────────

asset("L0a_syntax_error", "JS syntax error", """\
import * as THREE from 'three';

export function createAsset( {   // missing ) — syntax error
  return new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial()
  );
}
""")

asset("L0b_empty_group", "createAsset returns empty Group", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Group();   // no Mesh children — validator rejects
}
""")

asset("L0c_wrong_export", "exports buildModel not createAsset", """\
import * as THREE from 'three';

export function buildModel() {
  return new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial()
  );
}
""")

# ── L1: Loads, renders, very poor geometry ────────────────────────────────────

asset("L1a_manual_flat", "2 tris, no normals, no UVs", """\
import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.BufferGeometry();
  const verts = new Float32Array([
    -0.5, -0.5, 0,   0.5, -0.5, 0,   0.5, 0.5, 0,
    -0.5, -0.5, 0,   0.5,  0.5, 0,  -0.5, 0.5, 0,
  ]);
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.8 }));
}
""")

asset("L1b_manual_normals_only", "40 tris, normals only, no UVs", """\
import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.IcosahedronGeometry(1, 0);   // 20 faces = 40 tris
  geo.computeVertexNormals();
  // Intentionally omit UVs
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.85 }));
}
""")

# ── L2: Low poly, built-in normals + UVs (below POLY_MIN=50) ──────────────────

asset("L2a_box_12tris", "BoxGeometry — 12 tris, normals+uvs built-in", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.8, metalness: 0.1 })
  );
}
""")

asset("L2b_tetrahedron_8tris", "TetrahedronGeometry — 4 tris", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.TetrahedronGeometry(1, 0),
    new THREE.MeshStandardMaterial({ color: 0x776655, roughness: 0.9 })
  );
}
""")

# ── L3: Acceptable poly range (50–2000), normals + UVs ───────────────────────

asset("L3a_sphere_80tris", "SphereGeometry 8×6 — ~80 tris", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.85, metalness: 0.05 })
  );
}
""")

asset("L3b_cylinder_120tris", "CylinderGeometry 16 seg — ~120 tris", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.CylinderGeometry(0.4, 0.6, 1.5, 16, 3),
    new THREE.MeshStandardMaterial({ color: 0x775533, roughness: 0.9 })
  );
}
""")

# ── L4: Ideal poly range (200–2000), normals + UVs ───────────────────────────

asset("L4a_sphere_336tris", "SphereGeometry 16×12 — ~336 tris", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.85, metalness: 0.05 })
  );
}
""")

asset("L4b_torus_600tris", "TorusGeometry 16×32 — ~600 tris", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.TorusGeometry(0.7, 0.3, 16, 32),
    new THREE.MeshStandardMaterial({ color: 0x996655, roughness: 0.7, metalness: 0.2 })
  );
}
""")

asset("L4c_icosa_detail2", "IcosahedronGeometry detail=2, normals+UVs via projection", """\
import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.IcosahedronGeometry(1, 2);
  geo.computeVertexNormals();
  const pos = geo.attributes.position;
  const uvs = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const len = Math.sqrt(x*x + y*y + z*z);
    uvs[i*2]   = 0.5 + Math.atan2(z/len, x/len) / (2 * Math.PI);
    uvs[i*2+1] = 0.5 - Math.asin(Math.max(-1, Math.min(1, y/len))) / Math.PI;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  return new THREE.Mesh(geo,
    new THREE.MeshStandardMaterial({ color: 0x776655, roughness: 0.8, metalness: 0.1 })
  );
}
""")

# ── L5: High quality — ideal poly, displacement, normals, UVs ─────────────────

asset("L5a_icosa_detail3_displaced", "IcosahedronGeometry detail=3 + displacement — 1280 tris", """\
import * as THREE from 'three';

export function createAsset() {
  const geo = new THREE.IcosahedronGeometry(1, 3);
  const pos = geo.attributes.position;

  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const noise = Math.sin(x * 4.1) * 0.07 + Math.cos(y * 3.7) * 0.06 + Math.sin(z * 5.3) * 0.05;
    const scale = 1 + noise;
    pos.setXYZ(i, x * scale, y * scale, z * scale);
  }
  geo.computeVertexNormals();

  const uvs = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const len = Math.sqrt(x*x + y*y + z*z);
    uvs[i*2]   = 0.5 + Math.atan2(z/len, x/len) / (2 * Math.PI);
    uvs[i*2+1] = 0.5 - Math.asin(Math.max(-1, Math.min(1, y/len))) / Math.PI;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

  return new THREE.Mesh(geo,
    new THREE.MeshStandardMaterial({ color: 0x887766, roughness: 0.85, metalness: 0.05 })
  );
}
""")

asset("L5b_multi_mesh_group", "Group with 3 meshes — rock-like cluster", """\
import * as THREE from 'three';

export function createAsset() {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0x776655, roughness: 0.9, metalness: 0.0 });

  const configs = [
    [0.8, 3, [0, 0, 0]],
    [0.5, 2, [0.6, 0.1, 0.3]],
    [0.4, 2, [-0.5, 0.15, -0.3]],
  ];

  for (const [radius, detail, pos] of configs) {
    const geo = new THREE.IcosahedronGeometry(radius, detail);
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
      const noise = Math.sin(x * 5) * 0.08 + Math.cos(y * 4) * 0.06;
      positions.setXYZ(i, x * (1 + noise), y * (1 + noise), z * (1 + noise));
    }
    geo.computeVertexNormals();
    const uvs = new Float32Array(positions.count * 2);
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
      const len = Math.sqrt(x*x + y*y + z*z);
      uvs[i*2]   = 0.5 + Math.atan2(z/len, x/len) / (2 * Math.PI);
      uvs[i*2+1] = 0.5 - Math.asin(Math.max(-1, Math.min(1, y/len))) / Math.PI;
    }
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(...pos);
    group.add(mesh);
  }
  return group;
}
""")

# ── L6: Over-dense — above ideal range, tests poly penalty ceiling ────────────

asset("L6a_sphere_5ktris", "SphereGeometry 50×50 — ~5000 tris (acceptable, not ideal)", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1, 50, 50),
    new THREE.MeshStandardMaterial({ color: 0x886644, roughness: 0.85, metalness: 0.05 })
  );
}
""")

asset("L6b_torusknot_extreme", "TorusKnotGeometry 200×32 — ~12700 tris (above POLY_MAX)", """\
import * as THREE from 'three';

export function createAsset() {
  return new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.7, 0.25, 200, 32),
    new THREE.MeshStandardMaterial({ color: 0x997766, roughness: 0.7, metalness: 0.1 })
  );
}
""")

# ── L7: Existing asset baseline ───────────────────────────────────────────────

_rock_candidates = [
    Path(__file__).parent / "candidates" / "rock_v1.mjs",
    Path(__file__).parent / "rock_v1.mjs",
]
for _p in _rock_candidates:
    if _p.exists():
        asset("L7_rock_v1_baseline", "existing starter rock asset", _p.read_text())
        break


# ── Runner ─────────────────────────────────────────────────────────────────────

def score_breakdown(metrics: dict) -> str:
    """One-line explanation of score components."""
    geo = metrics.get("geometry") or {}
    sc = metrics.get("screenshot") or {}
    an = metrics.get("screenshot_analysis") or {}

    if not geo.get("ok"):
        return f"geo-fail: {geo.get('error', '?')[:55]}"

    parts = []
    tris = geo.get("triangle_count", 0)
    if POLY_IDEAL_MIN <= tris <= POLY_IDEAL_MAX:
        parts.append("poly+ideal")
    elif POLY_MIN <= tris <= POLY_MAX:
        parts.append("poly+ok")
    else:
        parts.append("poly+poor")

    if geo.get("has_normals"):  parts.append("nrm+")
    else:                       parts.append("nrm-")
    if geo.get("has_uvs"):      parts.append("uv+")
    else:                       parts.append("uv-")

    if sc.get("ok"):
        cov = an.get("coverage") or 0
        div = an.get("color_diversity") or 0
        parts.append(f"rend+ cov={cov:.0%} div={div:.0f}")
    else:
        parts.append("rend-")

    return "  ".join(parts)


def run_benchmarks():
    adapter = ThreeJSAdapter()
    results = []

    W = 76
    print(f"\n{'='*W}")
    print(" threejs_loop — Graduated Benchmark Suite")
    print(f" PASS_THRESHOLD={PASS_THRESHOLD}  poly_ideal={POLY_IDEAL_MIN}–{POLY_IDEAL_MAX}  poly_ok={POLY_MIN}–{POLY_MAX}")
    print(f"{'='*W}")

    col = f"{'Asset':<32} {'Tris':>6}  {'Nrm':>3} {'UV':>3} {'Rnd':>3}  {'Score':>6}  {'':>5}"
    print(col)
    print("-"*W)

    for (name, note, js) in ASSETS:
        asset_path = BENCH_DIR / f"{name}.mjs"
        asset_path.write_text(js)

        t0 = time.time()
        metrics = adapter.run(asset_path, f"bench_{name}")
        elapsed = time.time() - t0

        score = calculate_score(metrics)
        geo   = metrics.get("geometry") or {}
        sc    = metrics.get("screenshot") or {}

        tris_v  = geo.get("triangle_count", "-") if geo.get("ok") else "-"
        nrm_v   = ("y" if geo.get("has_normals") else "N") if geo.get("ok") else "-"
        uv_v    = ("y" if geo.get("has_uvs") else "N") if geo.get("ok") else "-"
        rnd_v   = ("y" if sc.get("ok") else "N") if sc else "-"
        pass_v  = "PASS" if score >= PASS_THRESHOLD else "."

        print(f"{name:<32} {str(tris_v):>6}  {nrm_v:>3} {uv_v:>3} {rnd_v:>3}  {score:>6.3f}  {pass_v:<5}  ({elapsed:.0f}s)")

        results.append({
            "name": name,
            "note": note,
            "score": score,
            "tris": tris_v,
            "has_normals": geo.get("has_normals"),
            "has_uvs": geo.get("has_uvs"),
            "geo_ok": geo.get("ok", False),
            "rendered": sc.get("ok", False),
            "coverage": (metrics.get("screenshot_analysis") or {}).get("coverage"),
            "diversity": (metrics.get("screenshot_analysis") or {}).get("color_diversity"),
            "elapsed": round(elapsed, 1),
            "breakdown": score_breakdown(metrics),
        })

    print(f"{'='*W}")

    # ── Summary ────────────────────────────────────────────────────────────────
    passing = [r for r in results if r["score"] >= PASS_THRESHOLD]
    failing = [r for r in results if r["score"] < PASS_THRESHOLD]

    print(f"\nRESULTS  {len(passing)} pass / {len(failing)} fail  (threshold={PASS_THRESHOLD})\n")

    # Group by level prefix
    for r in results:
        mark = "PASS" if r["score"] >= PASS_THRESHOLD else "fail"
        print(f"  {mark}  {r['name']:<34}  score={r['score']:.3f}  [{r['breakdown']}]")

    # ── Diagnostic insights ────────────────────────────────────────────────────
    print("\n── Diagnostic insights ──────────────────────────────────────────────")

    geo_fail = [r for r in results if not r["geo_ok"]]
    geo_ok   = [r for r in results if r["geo_ok"]]
    print(f"  Geometry failures (score=0.0):          {len(geo_fail)} assets  {[r['name'] for r in geo_fail]}")

    render_fail = [r for r in geo_ok if not r["rendered"]]
    print(f"  Geometry OK but render failed:          {len(render_fail)} assets  {[r['name'] for r in render_fail]}")

    low_poly_pass = [r for r in passing if isinstance(r["tris"], int) and r["tris"] < POLY_MIN]
    print(f"  Pass with tris < POLY_MIN ({POLY_MIN}):       {len(low_poly_pass)} assets  {[r['name'] for r in low_poly_pass]}")

    plain_prim_pass = [r for r in passing
                       if r["geo_ok"] and r["has_normals"] and r["has_uvs"]
                       and isinstance(r["tris"], int) and r["tris"] < 200]
    print(f"  Plain primitive (<200 tris) passes:     {len(plain_prim_pass)} assets  {[r['name'] for r in plain_prim_pass]}")

    overdense = [r for r in geo_ok if isinstance(r["tris"], int) and r["tris"] > POLY_MAX]
    print(f"  Over-dense (tris > POLY_MAX={POLY_MAX}):   {len(overdense)} assets  {[r['name'] for r in overdense]}")

    no_uv_pass = [r for r in passing if not r.get("has_uvs")]
    print(f"  Pass without UVs:                       {len(no_uv_pass)} assets  {[r['name'] for r in no_uv_pass]}")

    scores_ok = sorted([r["score"] for r in geo_ok])
    if scores_ok:
        print(f"  Score range (geo-ok assets):  {min(scores_ok):.3f} – {max(scores_ok):.3f}"
              f"  (median {scores_ok[len(scores_ok)//2]:.3f})")

    baseline = next((r for r in results if "rock_v1" in r["name"]), None)
    if baseline:
        print(f"\n  Existing rock_v1 baseline score: {baseline['score']:.3f}  "
              f"({'passes' if baseline['score'] >= PASS_THRESHOLD else 'does not pass'} threshold)")

    print()


if __name__ == "__main__":
    run_benchmarks()

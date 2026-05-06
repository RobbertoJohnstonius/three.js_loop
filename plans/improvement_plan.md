# threejs_loop — Improvement Plan
_Drafted 2026-05-06 from benchmark suite + simplify audit_

---

## Context

The benchmark suite (`run_benchmarks.py`) confirms the pipeline is structurally sound but exposed three scoring weaknesses and several loop-level gaps. The plan below addresses them in priority order.

---

## Part A — Scoring fixes (high urgency, implement first)

### A1. Lower color_diversity threshold from 25 → 12  ★ CRITICAL

**Finding:** Every rendered asset scores diversity 9–24. The +0.10 bonus is never earned. Effective score ceiling is 0.900 despite the formula going to 1.000.

**Root cause:** The threshold was calibrated against a brighter background or higher-contrast scenes. With #1a1a2e background and single-colour `MeshStandardMaterial`, real shading variation produces diversity 14–24 — just below the threshold.

**Fix:** Lower `COLOR_DIVERSITY_THRESHOLD` from 25 to 12 in `threejs_loop.py`. With the current rig:
- Flat plane, no normals: div≈13 → just above 12 (borderline, acceptable)
- Rock/sphere/torus with proper normals: div≈14–24 → reliably earns the bonus

**After fix:** rock_v1 (div=14) earns +0.10, pushing it from 0.800 → 0.900 (already passes). The loop retains a real gradient to optimise toward.

**Critique:** Lowering to 12 is conservative enough that pathologically flat assets (div<12) still don't earn it. A threshold of 10 would be too permissive. 12 is right.

---

### A2. Separate the "visible" and "coverage" bonuses — add uniformity penalty  ★ HIGH

**Finding:** Four different geometries (sphere, torus, icosahedron, box) all score identically at 0.900. There is no score signal above 0.900, so the loop will stagnate once any asset crosses 200 tris with normals+UVs.

**Fix:** Replace the flat 0.15 "visible" bonus with a graduated uniformity component:
```
0.05  any pixel coverage > 2% (was 0.15 — keep basic visibility check small)
0.10  coverage uniformity ≥ 0.70 (4-quadrant std check already computed by analyzer)
0.10  color_diversity > 12 (A1 fix)
0.05  coverage > 10%
```
This redistributes the same max weight (0.30) but creates real gradient above 0.85 that differentiates a well-shaped asset from a blob. Coverage uniformity already computed in `analyze_screenshot` — just not in the score formula.

**Critique:** This changes the score formula mid-run. Must clear `loop_state.json` and rebaseline.

---

### A3. Add a recognition bonus via vision model  ★ MEDIUM

**Finding:** A plain sphere and a rock-shaped icosahedron score identically. The loop has no incentive to make assets look like rocks specifically.

**Fix:** Add an optional +0.05 "recognisable" bonus: if the vision model confirms `recognisable: true` for the target asset name, award it. This is already in the vision JSON schema. Wire it into `calculate_score`.

**Critique:** Adds LLM dependency to scoring (currently deterministic). Cap at 0.05 so it can't swing pass/fail alone.

---

## Part B — Loop-level improvements (medium urgency)

### B1. Episodic feedback into Planner  ★ HIGH

**Finding:** `episodic.jsonl` is written every iteration but never read back. The planner proposes improvements in a vacuum.

**Fix:** Before calling `run_planner`, read the last 10 episodic records. Compute: which improvement types led to score increases vs. regressions. Inject a "what worked / what regressed" summary into the planner prompt.

**Implementation:** Add `read_episodic_summary(n=10) -> dict` function that aggregates outcomes by `improvement_type`.

---

### B2. Stagnation detection + escape  ★ HIGH

**Finding:** If three consecutive iterations produce score=0.900 (all pass geometry+poly+normals+UVs but diversity is capped), the loop will accept each one as "not better" and stagnate.

**Fix:** Detect plateau: if score hasn't changed by >0.01 in the last 4 iterations:
1. Force the planner to focus on a different improvement_type than the last 3 attempts.
2. Inject into planner prompt: "The score has plateaued at X. Try a fundamentally different approach."

---

### B3. Score-aware planner context  ★ MEDIUM

**Finding:** The planner doesn't know the current score or how close to threshold the asset is.

**Fix:** Add `score`, `score_gap` (PASS_THRESHOLD - score), and `bottleneck` (the scoring component contributing least) to the planner prompt. Derive bottleneck from the metrics:
- If tris < 200: bottleneck = "polygon_count"
- If not has_normals: bottleneck = "normals"
- If coverage_uniformity < 0.70: bottleneck = "shape_quality"
- If color_diversity < 12: bottleneck = "material_variation"

---

### B4. Graveyard for failed candidates  ★ MEDIUM

**Finding:** When an A/B pair fails geometry validation, the JS files are written to `candidates/` and left there, cluttering it.

**Fix:** Move failed candidates to `candidates/graveyard/` after failure. Keep a `graveyard.jsonl` with the failure reason and plan that produced them. The planner can inspect recent graveyard entries to avoid repeating them.

---

### B5. Asset design brief injection  ★ MEDIUM

**Finding:** The loop generates geometry with no target art style, game context, or polygon budget rationale.

**Fix:** Add `brief.json` (sibling of `steering.json`) that the Coder and Planner always receive:
```json
{
  "style": "low-poly stylised game asset",
  "target_game": "fantasy RPG",
  "poly_target": "300–600 triangles",
  "material_style": "MeshStandardMaterial, roughness 0.7–0.9, no metalness",
  "size_reference": "fits in 2×2×2 unit bounding box"
}
```
This prevents the loop from optimising toward arbitrary blobs.

---

## Part C — Code quality (lower urgency, already partially done)

### C1. Merge duplicate A/B rendering logic  (already in scope from simplify)

The render+stitch+analyze block inside `evaluate_ab_candidates` duplicates logic from `adapter.run()`. Extract `adapter.render_and_analyze(path, tag, ts)` method.

### C2. `candidate_path` A/B suffix  (done in simplify pass)

### C3. Pass rules to coder  (done in simplify pass)

---

## Implementation sequence

| Priority | Task | File(s) | Est. |
|---|---|---|---|
| 1 | A1: lower color_diversity threshold 25→12 | threejs_loop.py | 5 min |
| 2 | A2: graduated coverage/uniformity scoring | threejs_loop.py | 30 min |
| 3 | B3: score-aware planner context | threejs_loop.py | 20 min |
| 4 | B1: episodic feedback into planner | threejs_loop.py | 40 min |
| 5 | B2: stagnation detection | threejs_loop.py | 20 min |
| 6 | A3: recognition bonus | threejs_loop.py | 20 min |
| 7 | B5: brief.json | threejs_loop.py + brief.json | 20 min |
| 8 | B4: graveyard | threejs_loop.py | 30 min |
| 9 | C1: extract render_and_analyze | target_adapter.py | 30 min |

---

## Scoring formula (after A1 + A2)

```
0.10  geometry loads (geo.ok)
0.20  triangle count ideal (200–2000) / 0.10 acceptable / 0.02 any
0.15  vertex normals present
0.15  UV coordinates present
0.10  renders without crash
0.05  any pixel coverage > 2%     (was 0.15 — basic visibility)
0.10  coverage uniformity ≥ 0.70  (NEW — rewards consistent shape across angles)
0.10  color_diversity > 12        (was >25, never earned)
0.05  coverage > 10%
─────
1.00 max
```

**Verify against benchmarks after change:**
- L0 failures → still 0.000
- L1a flat/no normals → ~0.35 (worse than before — good)
- L2a plain box → ~0.720 (similar, still below threshold)
- L3a sphere 80 tris → ~0.80
- L4a sphere 336 tris → ~0.90 (passes)
- rock_v1 → ~0.80 (same, still needs more tris)

---

## Self-critique of this plan

**What's good:**
- A1 is minimal risk, high impact, unambiguously correct — do first.
- A2 redistributes existing weight rather than adding new weight — total stays 1.00.
- B3 requires no new LLM calls — just adds existing data to existing prompt.
- Sequence is ordered correctly: fix scoring before changing loop behaviour that depends on it.

**What's risky:**
- A2 changes the scoring formula — must re-run benchmarks to validate, and clear loop_state.json.
- A3 adds non-determinism to scoring. Keep it optional (skip if no vision result).
- B1 episodic summary is only useful after many iterations — won't help the first few runs.
- B5 brief.json constrains the coder more tightly — could make it harder to find creative solutions.

**What's missing:**
- No plan for improving the lighting rig (the auto-normalise+centre code in render_scene.mjs is good but camera angles may not show all asset types well).
- No plan for multi-asset generation (currently always one asset at a time).
- No plan for the vision model's `recognisable` score being gamed by a sphere named "rock".

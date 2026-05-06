# Master Plan v1 — Self-Critique
_2026-05-06_

---

## What the draft gets right

- **Sequencing**: viewer first is correct — all subsequent phases produce visual output and the user needs a way to see it immediately. Nothing else matters until there's a feedback loop the user can actually use.
- **Style system before reference images**: Style determines what "good" looks like. Reference images are only useful once the rendering matches the target style. Correct order.
- **Lighting before vertex colors**: Fixing the lighting rig makes the existing diversity metric more useful before adding new metrics. Right order.
- **Phase 6 last**: Architecture cleanup has no user-visible benefit. Deferring is correct.
- **Benchmark gates**: Running `run_benchmarks.py` after each phase is the right discipline.

---

## Problems with the draft

### P1: Viewer is under-specified

The draft says "poll loop_state.json every 2 seconds" but doesn't specify:
- What happens if the asset file is being written (partial write) — could crash the viewer
- Where loop_state.json lives relative to the viewer's web root
- Whether OrbitControls comes from node_modules/three/examples/jsm (it does — already installed)

The viewer also needs a way to show the **current score overlay** — not just the 3D model. The stats overlay (score, tris, iteration) requires reading metrics JSON, not just loop_state.json.

Also: `serve_viewer.py` serving LOOP_DIR creates a path traversal risk (Python's http.server doesn't sandbox). Should use `--directory` flag or a proper constraint.

### P2: Style system is too ambitious for one phase

Implementing three full rendering variants (low-poly flat shading, PBR with HDR, cartoon with outline pass) is 4–6 hours of work, not 2h. The outline pass for cartoon style requires EffectComposer which is a full dependency addition.

Better: implement low-poly first (trivial — just `flatShading: true`), defer PBR HDR and cartoon to separate tasks. The current target game is "fantasy RPG low-poly" — PBR and cartoon are hypothetical.

### P3: Score total exceeds 1.0 after Phase 5

The draft adds reference (+0.08) + recognisable (+0.05) + style_conformance (+0.08) on top of the existing 1.00 formula. Max becomes 1.21. The draft notes this as an open question but doesn't propose a solution.

This is a showstopper if unresolved before implementation. Options:
1. Renormalise: divide total weights by 1.21 → messy, breaks the clean per-component interpretation
2. Cap at 1.0 (already done with `min(score, 1.0)`) → reference/recognisable bonuses are additive overflows that push lower-scoring assets over threshold. This is actually fine — they function as tie-breakers.
3. Replace existing components: recognisable replaces 0.05 of the coverage>10% bonus. Reference replaces 0.05 of coverage_uniformity. Total stays 1.00.

Option 3 is cleanest. Recognisability IS a form of coverage quality; reference similarity IS a form of uniformity. Swap them rather than add.

### P4: Reference image vision — model capability gap

The draft assumes the vision model can compare two images in one prompt. `meta-llama/llama-3.2-11b-vision-instruct` on OpenRouter does NOT support multi-image prompts — only one image per call. The draft doesn't address this.

Fix: send two calls (one describing the reference, one describing the render), then use a text LLM to synthesise a similarity score from both descriptions. This is the "describe-then-structure" pattern already used in the main vision pipeline — consistent and reliable.

Alternative: switch to `google/gemini-2.0-flash-001` which does support multi-image. But this requires testing the model on OpenRouter first.

### P5: Graveyard implementation detail missing

The draft says "move to candidates/graveyard/" but the A/B evaluation writes files with `path_a.write_text(js_a)` only when `ok_a` is True (basic JS validation passed). Files that fail basic validation are never written. Files that fail geometry validation ARE written (they need to be written for the geometry validator to run them). Only geometry-failing candidates need to be moved.

The draft's description conflates these two failure modes.

### P6: No plan for what to do when the loop passes

The draft is entirely about quality improvement, but says nothing about what happens AFTER an asset passes (score ≥ 0.92). Currently the loop just logs "DONE" and exits. The user then has a .mjs file. 

Missing: export to game project, export to GLTF, thumbnail generation for the asset library, notification to the user.

### P7: Emissive support is too vague

"If the description includes 'glowing', 'crystal'…" — this is a heuristic in a prompt string, not a feature. If the asset type is "glowing_crystal" the description match works, but "rune_stone" might not trigger it. This belongs in brief.json as a field: `"emissive": true` or as part of style_notes, not hidden in a prompt heuristic.

### P8: Vertex color support needs UV interaction clarification

If an asset uses `vertexColors: true`, the UV attribute is still needed for other purposes (normal mapping etc), but the color comes from vertex attributes, not a UV-mapped texture. The scoring currently checks `has_uvs` — vertex-colored assets that deliberately have no UV-mapped texture would be penalised. Need to clarify: does `vertexColors: true` satisfy the UV requirement? (No — it doesn't. Both can coexist. The scoring should stay as-is.)

---

## Revised priorities and cuts

### Cut (from v1):
- Cartoon outline pass / EffectComposer (too complex, not the target style)
- Style conformance score in scoring (circular — would need a reliable style classifier)
- Auto-generate reference image at session start (scope creep; user providing reference is sufficient)

### Add:
- Viewer stats overlay with score + metrics JSON read
- Explicit handling of partial-write race condition in viewer (read then validate)
- On-pass export: copy best .mjs to a `dist/` folder and generate a thumbnail
- brief.json `"emissive": bool` field (explicit, not a heuristic)
- Multi-image note: use describe-then-compare pattern for reference, not single-prompt comparison

### Reorder:
- Move lighting fix (P3.1) to happen immediately after viewer (it costs 10 minutes and makes the viewer immediately look better)
- Split style system: implement low-poly only first (flatShading, 20 minutes), defer PBR/cartoon
- Vertex colors (P3.2) is medium complexity — move after reference images, not before

---

## Critique of sequencing

The original sequence has a hidden dependency: improving the **lighting rig** (P3.1) should happen BEFORE implementing the **viewer** (P1), because the viewer should show the improved lighting from day one. Otherwise users get used to the wrong lighting and are surprised when renders differ from the viewer.

Revised order: lighting fix → viewer → style (low-poly only) → roughness rules → reference image → recognisable bonus → vertex colors → remaining.

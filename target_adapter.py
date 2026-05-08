"""
target_adapter.py — ThreeJSAdapter for threejs_loop.

Three-stage playtester:
  1. validate_geometry()    — Node.js, checks mesh structure (instant, no browser)
  2. render_multi_angle()   — Puppeteer, 4-angle headless renders
  3. stitch + analyze()     — PIL: 2×2 grid assembly + pixel statistics

run() chains all three and returns a unified metrics dict.
"""

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

LOOP_DIR = Path(__file__).parent.resolve()
HEADLESS_DIR = LOOP_DIR / "headless"
SCREENSHOTS_DIR = LOOP_DIR / "screenshots"
METRICS_DIR = LOOP_DIR / "metrics"

SCREENSHOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

BACKGROUND_COLOR_RGB = (26, 26, 46)

# Camera angle names produced by render_scene.mjs --multi
ANGLE_NAMES = ("q", "f", "r", "t")   # quarter, front, right, top


class ThreeJSAdapter:
    name = "threejs"

    # ── Private: Node subprocess runner ──────────────────────────────────────

    def _run_node(self, script: Path, args: list[str], timeout: int) -> dict:
        """Run a Node script, parse its JSON stdout. Returns result dict."""
        try:
            proc = subprocess.run(
                ["node", str(script)] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(LOOP_DIR),
            )
            out = proc.stdout.strip()
            if out:
                return json.loads(out)
            err = proc.stderr.strip()
            return {"ok": False, "error": err or "no output"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timed out ({timeout}s)"}
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"bad JSON: {e}"}
        except FileNotFoundError:
            return {"ok": False, "error": "node not found — is Node.js installed?"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Stage 1: geometry validation ─────────────────────────────────────────

    def validate_geometry(self, asset_path: Path, timeout: int = 30) -> dict:
        """Run the Node geometry validator. Returns metrics dict."""
        return self._run_node(
            HEADLESS_DIR / "validate_geometry.mjs",
            [str(asset_path.resolve())],
            timeout,
        )

    # ── Stage 2: multi-angle headless render ──────────────────────────────────

    def render_single(self, asset_path: Path, output_path: Path, timeout: int = 60) -> dict:
        """Single-angle render (quarter view). Returns render result dict."""
        return self._run_node(
            HEADLESS_DIR / "render_scene.mjs",
            [str(asset_path.resolve()), str(output_path)],
            timeout,
        )

    def render_multi_angle(
        self,
        asset_path: Path,
        output_base: Path,
        timeout: int = 90,
    ) -> dict:
        """
        Run render_scene.mjs --multi. Returns render result dict:
          {ok, screenshots: [{name, path}], console_errors, render_error}
        output_base should NOT include .png extension.
        """
        return self._run_node(
            HEADLESS_DIR / "render_scene.mjs",
            [str(asset_path.resolve()), str(output_base), "--multi"],
            timeout,
        )

    # ── Stage 3a: stitch screenshots into grid ────────────────────────────────

    def stitch_screenshots(
        self,
        screenshots: list[dict],
        output_path: Path,
    ) -> Path | None:
        """
        Combine angle screenshots into a grid PNG.
        screenshots: list of {name, path} dicts from render_multi_angle.
        Returns path of stitched image, or None on failure.
        """
        try:
            from PIL import Image

            imgs = []
            for s in screenshots:
                p = Path(s["path"])
                if p.exists():
                    imgs.append(Image.open(p).convert("RGB"))

            if not imgs:
                return None

            w, h = imgs[0].size
            cols = math.ceil(math.sqrt(len(imgs)))
            rows = math.ceil(len(imgs) / cols)
            grid = Image.new("RGB", (w * cols, h * rows), color=BACKGROUND_COLOR_RGB)
            for i, img in enumerate(imgs):
                grid.paste(img, ((i % cols) * w, (i // cols) * h))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            grid.save(str(output_path))
            return output_path

        except ImportError:
            return None
        except Exception:
            return None

    # ── Stage 3b: pixel statistics ────────────────────────────────────────────

    def analyze_screenshot(self, screenshot_path: Path, ref_path: Path | None = None) -> dict:
        """
        Programmatic analysis of a PNG. No LLM required.
        Background colour: BACKGROUND_COLOR_RGB.

        For the stitched grid, metrics are computed across the whole image
        so coverage/diversity reflect the average across all four angles.
        """
        try:
            from PIL import Image
            img = Image.open(screenshot_path).convert("RGB")
            return self._analyze_image(img, ref_path=ref_path)
        except ImportError:
            return {
                "render_ok": True,
                "coverage": None,
                "color_diversity": None,
                "brightness": None,
                "error": "Pillow not installed — run: pip install Pillow numpy",
            }
        except Exception as e:
            return {"render_ok": False, "error": str(e)}

    def _analyze_image(self, img, ref_path: Path | None = None) -> dict:
        """Pixel statistics on an in-memory PIL Image."""
        import numpy as np
        w, h = img.size
        arr = np.array(img, dtype=np.float32)

        bg = np.array(list(BACKGROUND_COLOR_RGB), dtype=np.float32)
        bg_mask = np.all(np.abs(arr - bg) < 28, axis=2)
        non_bg_count = int((~bg_mask).sum())
        coverage = non_bg_count / (w * h)

        non_bg_arr = arr[~bg_mask] if non_bg_count > 200 else None
        color_std = float(np.std(non_bg_arr)) if non_bg_arr is not None else 0.0
        avg_brightness = float(np.mean(arr)) / 255.0

        quadrant_coverages = []
        for qr in range(2):
            for qc in range(2):
                r0, r1 = qr * (h // 2), (qr + 1) * (h // 2)
                c0, c1 = qc * (w // 2), (qc + 1) * (w // 2)
                quad = ~bg_mask[r0:r1, c0:c1]
                quadrant_coverages.append(float(quad.sum()) / quad.size)
        coverage_std = float(np.std(quadrant_coverages))

        # Dark-patch detection: near-black non-background pixels suggest inverted normals.
        # Threshold 20 (max channel) is deliberately conservative — only catches effectively-
        # zero brightness. Paired with vision normal_issues flag to prevent false positives
        # from legitimately dark materials or deep shadow areas.
        if non_bg_arr is not None:
            dark_patch_fraction = float((non_bg_arr.max(axis=1) < 20).mean())
        else:
            dark_patch_fraction = 0.0

        pixel_metrics = self._measure_shape_metrics(arr, bg_mask, w, h)

        # Item 1 — reference pixel similarity (histogram intersection, pose-invariant)
        if ref_path is not None:
            rps = self._compute_ref_pixel_sim(ref_path, arr, bg_mask)
            if rps is not None:
                pixel_metrics["ref_pixel_sim"] = rps

        # Item 2 — palette colour count (distinct hue clusters in the render)
        phc = self._count_palette_hues(arr, bg_mask)
        if phc is not None:
            pixel_metrics["palette_distinct_hues"] = phc

        return {
            "render_ok":            coverage > 0.02,
            "coverage":             round(coverage, 4),
            "color_diversity":      round(color_std, 2),
            "brightness":           round(avg_brightness, 4),
            "non_bg_pixels":        non_bg_count,
            "coverage_uniformity":  round(1.0 - min(coverage_std * 4, 1.0), 3),
            "dark_patch_fraction":  round(dark_patch_fraction, 4),
            "resolution":           f"{w}x{h}",
            "pixel_metrics":        pixel_metrics,
        }

    def _measure_shape_metrics(self, arr, bg_mask, w, h):
        import numpy as np
        qh, qw = h // 2, w // 2
        quad_bg = bg_mask[:qh, :qw]
        non_bg = ~quad_bg

        if non_bg.sum() < 100:
            return {}

        rows_with_content = np.where(non_bg.any(axis=1))[0]
        cols_with_content = np.where(non_bg.any(axis=0))[0]

        if len(rows_with_content) == 0 or len(cols_with_content) == 0:
            return {}

        horiz_span = int(cols_with_content[-1]) - int(cols_with_content[0])
        if horiz_span < 10:
            return {}

        sample_cols = np.linspace(cols_with_content[0], cols_with_content[-1], 12, dtype=int)
        top_edges = []
        for col in sample_cols:
            col_non_bg = non_bg[:, col]
            idxs = np.where(col_non_bg)[0]
            if len(idxs) > 0:
                top_edges.append(int(idxs[0]))

        arc_depth_ratio = 0.0
        if len(top_edges) >= 3:
            top_variation = max(top_edges) - min(top_edges)
            arc_depth_ratio = round(top_variation / horiz_span, 4)

        left_cov = float((~bg_mask[:, :w // 2]).sum()) / (bg_mask[:, :w // 2].size)
        right_cov = float((~bg_mask[:, w // 2:]).sum()) / (bg_mask[:, w // 2:].size)
        coverage_asymmetry = round(abs(left_cov - right_cov), 4)

        quad_arr = arr[:qh, :qw].astype(np.float32)
        color_zone_boundary = None
        strip_green_cols = []
        for col in range(qw):
            pixels = quad_arr[:, col][~quad_bg[:, col]]
            if len(pixels) < 3:
                continue
            r, g, b = pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean()
            total = r + g + b + 1e-9
            if g / total > 0.42:
                strip_green_cols.append(col)

        if strip_green_cols and horiz_span > 0:
            rightmost_green = max(strip_green_cols)
            green_fraction = (rightmost_green - int(cols_with_content[0])) / horiz_span
            color_zone_boundary = round(max(0.0, min(1.0, green_fraction)), 4)

        result = {
            "arc_depth_ratio": arc_depth_ratio,
            "coverage_asymmetry": coverage_asymmetry,
        }
        if color_zone_boundary is not None:
            result["color_zone_boundary"] = color_zone_boundary

        # Specular highlight detection — fraction of object pixels above 215/255 luminance.
        # Glossy surfaces (roughness ≤ 0.25) produce concentrated bright spots; matte surfaces
        # spread specular energy so widely no individual pixel reaches this threshold.
        try:
            obj_pixels = arr[~bg_mask].astype(np.float32)
            if len(obj_pixels) > 200:
                lum = 0.299 * obj_pixels[:, 0] + 0.587 * obj_pixels[:, 1] + 0.114 * obj_pixels[:, 2]
                result["highlight_fraction"] = round(float((lum > 215.0).sum()) / len(lum), 4)
        except Exception:
            pass

        # HSV color statistics for hue validation
        try:
            import colorsys
            non_bg_pixels = arr[~bg_mask] if (~bg_mask).sum() > 200 else None
            if non_bg_pixels is not None:
                hues, sats = [], []
                for px in non_bg_pixels[::5]:  # sample every 5th pixel
                    r, g, b = float(px[0]) / 255, float(px[1]) / 255, float(px[2]) / 255
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    if v > 0.12:
                        hues.append(h * 360)
                        sats.append(s)
                if hues:
                    import numpy as np
                    result["dominant_hue"] = round(float(np.median(hues)), 1)
                    result["hue_variance"] = round(float(np.std(hues)), 1)
                    result["mean_saturation"] = round(float(np.mean(sats)), 3)
        except Exception:
            pass

        return result

    # ── Silhouette aspect-ratio metrics (Item 7) ──────────────────────────────

    def compute_silhouette_metrics(self, image_path: Path) -> dict:
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(image_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            bg = np.array([26.0, 26.0, 46.0])
            bg_mask = np.all(np.abs(arr - bg) < 28, axis=2)
            non_bg = ~bg_mask
            if non_bg.sum() < 200:
                return {}
            rows = np.where(non_bg.any(axis=1))[0]
            cols = np.where(non_bg.any(axis=0))[0]
            if not len(rows) or not len(cols):
                return {}
            sil_h = int(rows[-1]) - int(rows[0]) + 1
            sil_w = int(cols[-1]) - int(cols[0]) + 1
            if sil_w < 10 or sil_h < 10:
                return {}
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
            fill = float(non_bg[r0:r1, c0:c1].sum()) / (sil_h * sil_w)
            return {
                "silhouette_hw_ratio": round(sil_h / sil_w, 3),
                "silhouette_fill_fraction": round(fill, 3),
            }
        except Exception as e:
            logging.warning(f"compute_silhouette_metrics failed: {e}")
            return {}

    # ── Item 1: Reference pixel similarity ───────────────────────────────────

    def _compute_ref_pixel_sim(self, ref_path: Path, arr, bg_mask) -> float | None:
        """
        Hue-histogram intersection between render and reference image.
        Pose-invariant: compares colour distributions, not pixel positions.
        Returns [0, 1] — 1.0 means identical hue distribution.
        """
        try:
            from PIL import Image
            import numpy as np

            ref_img = Image.open(ref_path).convert("RGB")
            ref_arr = np.array(ref_img, dtype=np.float32)

            # Detect reference background from corners (photo may have any background)
            corners = np.concatenate([
                ref_arr[:20, :20].reshape(-1, 3),   ref_arr[:20, -20:].reshape(-1, 3),
                ref_arr[-20:, :20].reshape(-1, 3),  ref_arr[-20:, -20:].reshape(-1, 3),
            ])
            ref_bg = corners.mean(axis=0)
            ref_bg_mask = np.all(np.abs(ref_arr - ref_bg) < 40, axis=2)

            def _hue_hist(image_arr, mask, n_bins=36):
                px = image_arr[~mask].astype(np.float32) / 255.0
                if len(px) < 100:
                    return None
                R, G, B = px[:, 0], px[:, 1], px[:, 2]
                maxc = np.maximum(np.maximum(R, G), B)
                minc = np.minimum(np.minimum(R, G), B)
                delta = maxc - minc
                # Only saturated, mid-brightness pixels carry meaningful hue
                valid = (delta > 0.08) & (maxc > 0.08) & (maxc < 0.97)
                if valid.sum() < 50:
                    return None
                R, G, B, maxc, delta = R[valid], G[valid], B[valid], maxc[valid], delta[valid]
                hue = np.zeros(valid.sum())
                mr = maxc == R; mg = maxc == G; mb = maxc == B
                hue[mr] = ((G[mr] - B[mr]) / delta[mr]) % 6
                hue[mg] = (B[mg] - R[mg]) / delta[mg] + 2
                hue[mb] = (R[mb] - G[mb]) / delta[mb] + 4
                hue = hue / 6.0  # normalise to [0, 1]
                hist, _ = np.histogram(hue, bins=n_bins, range=(0.0, 1.0))
                return hist.astype(np.float64)

            h_render = _hue_hist(arr, bg_mask)
            h_ref    = _hue_hist(ref_arr, ref_bg_mask)
            if h_render is None or h_ref is None:
                return None

            h_render /= h_render.sum() + 1e-9
            h_ref    /= h_ref.sum()    + 1e-9
            intersection = float(np.minimum(h_render, h_ref).sum())
            return round(intersection, 4)

        except Exception as e:
            logging.debug(f"ref_pixel_sim failed: {e}")
            return None

    # ── Item 2: Palette colour count ─────────────────────────────────────────

    def _count_palette_hues(self, arr, bg_mask) -> int | None:
        """
        Count distinct hue clusters visible in the render.
        Each cluster = a meaningfully different colour (separated by ≥20°).
        Returns None if there are not enough coloured pixels to measure.
        """
        try:
            import numpy as np
            px = arr[~bg_mask].astype(np.float32) / 255.0
            if len(px) < 200:
                return None
            R, G, B = px[:, 0], px[:, 1], px[:, 2]
            maxc = np.maximum(np.maximum(R, G), B)
            minc = np.minimum(np.minimum(R, G), B)
            delta = maxc - minc
            # Require saturation > 0.08 and mid-brightness — filters grey/white/black
            valid = (delta > 0.08) & (maxc > 0.08) & (maxc < 0.97)
            if valid.sum() < 100:
                return 0
            R, G, B, maxc, delta = R[valid], G[valid], B[valid], maxc[valid], delta[valid]
            hue = np.zeros(valid.sum())
            mr = maxc == R; mg = maxc == G; mb = maxc == B
            hue[mr] = ((G[mr] - B[mr]) / delta[mr]) % 6
            hue[mg] = (B[mg] - R[mg]) / delta[mg] + 2
            hue[mb] = (R[mb] - G[mb]) / delta[mb] + 4
            hue_deg = hue * 60.0  # [0, 360]

            # 36-bin histogram (10° per bin); each bin = one potential colour
            hist, _ = np.histogram(hue_deg, bins=36, range=(0.0, 360.0))
            frac = hist / (hist.sum() + 1e-9)

            # Peak finding: bins with >3% share, not adjacent to a bigger peak (20° gap)
            peaks = []
            for i in np.argsort(-frac):
                if frac[i] < 0.03:
                    break
                if not any(abs(i - p) <= 1 or abs(i - p) >= 35 for p in peaks):
                    peaks.append(int(i))
            return len(peaks)
        except Exception:
            return None

    # ── Item 3: View consistency ──────────────────────────────────────────────

    def _compute_view_consistency(self, angle_paths: dict) -> dict:
        """
        Check that key pixel metrics are consistent across all rendered angles.
        A large coverage drop in one angle signals a back-of-object crack or missing geometry.
        """
        try:
            from PIL import Image
            import numpy as np
            coverages, dark_fracs = [], []
            angle_coverages = {}
            # Exclude top-down ('t') — overhead views inherently show more coverage
            # than side views; the difference is expected, not a defect.
            side_angles = {k: v for k, v in angle_paths.items() if k != 't'}
            for name, path in side_angles.items():
                p = Path(path)
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                m = self._analyze_image(img)
                cov = m.get("coverage")
                dpf = m.get("dark_patch_fraction")
                if cov is not None:
                    coverages.append(cov)
                    angle_coverages[name] = round(cov, 3)
                if dpf is not None:
                    dark_fracs.append(dpf)

            if len(coverages) < 2:
                return {}

            cov_arr = np.array(coverages)
            cov_mean  = float(cov_arr.mean())
            cov_range = float(cov_arr.max() - cov_arr.min())
            # Normalise spread by mean — relative variation [0, 1]
            rel_spread = cov_range / (cov_mean + 1e-6)
            score = round(max(0.0, 1.0 - min(rel_spread, 1.0)), 3)

            result: dict = {
                "view_consistency_score": score,
                "angle_coverages": angle_coverages,
            }
            if dark_fracs:
                result["max_angle_dark_patch"] = round(max(dark_fracs), 4)
            return result
        except Exception as e:
            logging.debug(f"view_consistency failed: {e}")
            return {}

    # ── Full playtester run ───────────────────────────────────────────────────

    def run(self, asset_path: Path, version_tag: str) -> dict:
        """
        Chain: validate_geometry → render_multi_angle → stitch → analyze.
        Returns a unified metrics dict. Skips render if geometry fails.
        """
        asset_path = Path(asset_path).resolve()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        result: dict = {
            "version": version_tag,
            "asset_path": str(asset_path),
            "timestamp": ts,
            "geometry": None,
            "screenshot": None,
            "screenshot_analysis": None,
            "screenshot_path": None,
            "stitched_path": None,
            "angle_paths": None,
        }

        # Stage 1: geometry validation
        result["geometry"] = self.validate_geometry(asset_path)
        if not result["geometry"].get("ok"):
            self._save_metrics(result, version_tag, ts)
            return result

        # Early render skip for severely torn geometry (Item 9)
        geo_ok = result["geometry"].get("ok", False)
        shared = result["geometry"].get("shared_edge_fraction", 1.0) if geo_ok else 1.0
        if geo_ok and shared < 0.85:
            result["screenshot"] = {
                "ok": False,
                "error": f"render skipped — mesh too torn (shared_edge_fraction={shared:.3f} < 0.85)",
                "render_skipped": True,
            }
            self._save_metrics(result, version_tag, ts)
            return result

        # Derive reference image path for pixel similarity (Item 1)
        import re as _re
        _stem = Path(asset_path).stem
        _asset_name = _re.sub(r'_v\d+.*$', '', _stem)
        _ref_path = LOOP_DIR / 'references' / f'{_asset_name}.png'
        ref_path: Path | None = _ref_path if _ref_path.exists() else None

        # Stage 2: multi-angle render
        output_base = SCREENSHOTS_DIR / f"{version_tag}_{ts}"
        render = self.render_multi_angle(asset_path, output_base)
        result["screenshot"] = render

        if not render.get("ok"):
            self._save_metrics(result, version_tag, ts)
            return result

        screenshots = render.get("screenshots", [])
        result["angle_paths"] = {s["name"]: s["path"] for s in screenshots}

        # Stage 3a: stitch into grid
        stitched_path = SCREENSHOTS_DIR / f"{version_tag}_{ts}_grid.png"
        stitched = self.stitch_screenshots(screenshots, stitched_path)
        if stitched and stitched.exists():
            result["stitched_path"] = str(stitched)
            result["screenshot_path"] = str(stitched)  # backward-compat alias

        # Stage 3b: pixel analysis on the grid (pass ref_path for Item 1)
        if stitched and stitched.exists():
            result["screenshot_analysis"] = self.analyze_screenshot(stitched, ref_path=ref_path)
        elif screenshots:
            first = Path(screenshots[0]["path"])
            if first.exists():
                result["screenshot_analysis"] = self.analyze_screenshot(first, ref_path=ref_path)

        # Stage 3c: silhouette metrics from front angle (Item 7)
        front_path = (result.get("angle_paths") or {}).get("f")
        if front_path and Path(front_path).exists():
            sil = self.compute_silhouette_metrics(Path(front_path))
            if sil and result.get("screenshot_analysis") is not None:
                result["screenshot_analysis"].setdefault("pixel_metrics", {}).update(sil)

        # Stage 3d: view consistency across all angles (Item 3)
        if result.get("angle_paths"):
            vc = self._compute_view_consistency(result["angle_paths"])
            if vc and result.get("screenshot_analysis") is not None:
                result["screenshot_analysis"].setdefault("pixel_metrics", {}).update(vc)

        # Stage 3e: extended_brief metrics (color zones + part presence + proportions)
        front_path_e = (result.get("angle_paths") or {}).get("f")
        if front_path_e and Path(front_path_e).exists():
            try:
                from reference_analyst import load_extended_brief
                eb = load_extended_brief()
                if eb.get("geometry_spec_generated"):
                    # 3e-i: color zones
                    cz = self._check_color_zones(Path(front_path_e), eb)
                    if cz and result.get("screenshot_analysis") is not None:
                        result["screenshot_analysis"].setdefault("pixel_metrics", {}).update({"color_zones": cz})
                    # 3e-ii: part presence check (Layer 1)
                    geo_for_parts = result.get("geometry") or {}
                    pp = self._check_part_presence(geo_for_parts, eb)
                    if pp and result.get("screenshot_analysis") is not None:
                        result["screenshot_analysis"].setdefault("pixel_metrics", {}).update(pp)
                    # 3e-iii: proportions vs reference (Layer 1)
                    pr = self._compare_proportions_to_ref(Path(front_path_e), eb)
                    if pr and result.get("screenshot_analysis") is not None:
                        result["screenshot_analysis"].setdefault("pixel_metrics", {}).update(pr)
            except Exception as _cze:
                import logging as _lg
                _lg.debug(f"Stage 3e extended_brief metrics failed: {_cze}")

        self._save_metrics(result, version_tag, ts)
        return result

    def render_and_analyze(
        self,
        asset_path: Path,
        version_tag: str,
        geo: dict,
        ts: str | None = None,
    ) -> dict:
        """
        Run render → stitch → analyze without repeating geometry validation.
        Use when geo is already known (A/B candidate evaluation).
        """
        if ts is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        result: dict = {
            "asset_path": str(asset_path),
            "geometry": geo,
            "screenshot": None,
            "screenshot_analysis": None,
            "stitched_path": None,
            "angle_paths": None,
        }

        output_base = SCREENSHOTS_DIR / f"{version_tag}_{ts}"
        render = self.render_multi_angle(asset_path, output_base)
        result["screenshot"] = render

        if render.get("ok"):
            screenshots = render.get("screenshots", [])
            result["angle_paths"] = {s["name"]: s["path"] for s in screenshots}
            stitched_path = SCREENSHOTS_DIR / f"{version_tag}_{ts}_grid.png"
            stitched = self.stitch_screenshots(screenshots, stitched_path)
            if stitched and stitched.exists():
                result["stitched_path"] = str(stitched)
                result["screenshot_analysis"] = self.analyze_screenshot(stitched)

        return result

    def _check_color_zones(self, front_render_path: Path, extended_brief: dict) -> dict:
        """
        Phase 7 — Programmatic per-part color coverage check.
        For each expected part in extended_brief, measures what fraction of
        non-background render pixels fall within ±40 hue-degrees of the expected
        dominant color. Returns a dict of {part_name: coverage_fraction}.
        No LLM required — fast PIL/numpy computation.
        """
        try:
            import colorsys
            import numpy as np
            from PIL import Image

            img = Image.open(front_render_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            bg = np.array(list(BACKGROUND_COLOR_RGB), dtype=np.float32)
            bg_mask = np.all(np.abs(arr - bg) < 28, axis=2)
            obj_pixels = arr[~bg_mask] / 255.0
            if len(obj_pixels) < 100:
                return {}

            # Precompute hue for every object pixel
            hues = np.empty(len(obj_pixels))
            for i, px in enumerate(obj_pixels):
                h, s, v = colorsys.rgb_to_hsv(float(px[0]), float(px[1]), float(px[2]))
                hues[i] = h * 360.0  # [0, 360]

            spec = extended_brief.get("geometry_spec_generated", {})
            result: dict = {}
            for part_name, part_spec in spec.items():
                rgb_01 = part_spec.get("dominant_rgb_01")
                if not rgb_01 or len(rgb_01) < 3:
                    continue
                target_h, target_s, target_v = colorsys.rgb_to_hsv(
                    rgb_01[0], rgb_01[1], rgb_01[2]
                )
                target_hue_deg = target_h * 360.0

                # Skip near-grey colors (low saturation) — unreliable hue comparison
                if target_s < 0.15:
                    continue

                # Angular distance in hue space (wraps at 360)
                hue_dist = np.abs(hues - target_hue_deg)
                hue_dist = np.minimum(hue_dist, 360.0 - hue_dist)
                coverage = float((hue_dist <= 40.0).sum()) / max(len(hues), 1)
                result[part_name] = round(coverage, 4)

            return result
        except Exception as e:
            logging.debug(f"_check_color_zones failed: {e}")
            return {}

    def _check_part_presence(self, geo: dict, extended_brief: dict) -> dict:
        """
        Layer 1: compare geo mesh_names against parts expected by extended_brief.
        Returns {part_name: True/False} presence map and list of missing part names.
        """
        try:
            spec = extended_brief.get("geometry_spec_generated", {})
            if not spec:
                return {}
            mesh_names = set(geo.get("mesh_names", []))
            presence: dict[str, bool] = {}
            missing: list[str] = []
            for part_name in spec:
                found = part_name in mesh_names
                presence[part_name] = found
                if not found:
                    missing.append(part_name)
            return {"part_presence": presence, "missing_parts": missing}
        except Exception as e:
            import logging
            logging.debug(f"_check_part_presence failed: {e}")
            return {}

    def _compare_proportions_to_ref(self, front_render_path: Path, extended_brief: dict) -> dict:
        """
        Layer 1: measure rendered object H:W ratio from the front render and
        compare to the reference image aspect ratio from extended_brief.
        Returns {render_hw_ratio, ref_hw_ratio, hw_ratio_delta}.
        """
        try:
            import numpy as np
            from PIL import Image

            sil = extended_brief.get("silhouette", {})
            ref_bbox = sil.get("object_bbox_px", [])
            if len(ref_bbox) < 4 or ref_bbox[2] < 1:
                return {}
            ref_hw = round(ref_bbox[3] / max(ref_bbox[2], 1), 3)

            img = Image.open(front_render_path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            bg = np.array(list(BACKGROUND_COLOR_RGB), dtype=np.float32)
            obj_mask = ~np.all(np.abs(arr - bg) < 28, axis=2)
            rows = np.where(obj_mask.any(axis=1))[0]
            cols = np.where(obj_mask.any(axis=0))[0]
            if not len(rows) or not len(cols):
                return {}
            render_h = int(rows[-1] - rows[0] + 1)
            render_w = int(cols[-1] - cols[0] + 1)
            render_hw = round(render_h / max(render_w, 1), 3)

            return {
                "render_hw_ratio": render_hw,
                "ref_hw_ratio": ref_hw,
                "hw_ratio_delta": round(abs(render_hw - ref_hw), 3),
            }
        except Exception as e:
            import logging
            logging.debug(f"_compare_proportions_to_ref failed: {e}")
            return {}

    def _save_metrics(self, result: dict, version_tag: str, ts: str) -> None:
        path = METRICS_DIR / f"{version_tag}_{ts}.json"
        path.write_text(json.dumps(result, indent=2))

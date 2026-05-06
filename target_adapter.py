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

    def analyze_screenshot(self, screenshot_path: Path) -> dict:
        """
        Programmatic analysis of a PNG. No LLM required.
        Background colour: BACKGROUND_COLOR_RGB.

        For the stitched grid, metrics are computed across the whole image
        so coverage/diversity reflect the average across all four angles.
        """
        try:
            from PIL import Image
            img = Image.open(screenshot_path).convert("RGB")
            return self._analyze_image(img)
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

    def _analyze_image(self, img) -> dict:
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

        return {
            "render_ok":            coverage > 0.02,
            "coverage":             round(coverage, 4),
            "color_diversity":      round(color_std, 2),
            "brightness":           round(avg_brightness, 4),
            "non_bg_pixels":        non_bg_count,
            "coverage_uniformity":  round(1.0 - min(coverage_std * 4, 1.0), 3),
            "dark_patch_fraction":  round(dark_patch_fraction, 4),
            "resolution":           f"{w}x{h}",
        }

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

        # Stage 3b: pixel analysis on the grid
        if stitched and stitched.exists():
            result["screenshot_analysis"] = self.analyze_screenshot(stitched)
        elif screenshots:
            first = Path(screenshots[0]["path"])
            if first.exists():
                result["screenshot_analysis"] = self.analyze_screenshot(first)

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

    def _save_metrics(self, result: dict, version_tag: str, ts: str) -> None:
        path = METRICS_DIR / f"{version_tag}_{ts}.json"
        path.write_text(json.dumps(result, indent=2))

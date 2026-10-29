"""Validate actual stage outputs before completion or reuse, without loading ML models."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

from scripts.configuration.forensics import AVAILABLE_METHODS, JPEG_EXTENSIONS, JPEG_ONLY_METHODS
from scripts.prompting import compile_description, compile_rankings, default_profiles, validate_response
from scripts.services.synthetic_inference_client import probabilities
from scripts.utilities.analysis_items import enumerate_analysis_items, granular_directory, safe_component


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _vector(value, length):
    return isinstance(value, list) and len(value) == length and all(_finite(n) for n in value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON property: {key}")
        result[key] = value
    return result


def read_document(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid number: {value}")))
        _require(isinstance(data, dict), "Expected a JSON object")
        return data
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()  # WebP/JPEG can have a readable header and truncated pixels.
            return image.size
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Missing or unreadable image {path}: {exc}") from exc


class Artifacts:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()

    def path(self, relative) -> Path:
        _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Invalid output path")
        path = (self.folder / relative).resolve()
        _require(path.is_relative_to(self.folder), f"Output points outside this analysis: {relative}")
        return path

    def document(self, relative):
        return read_document(self.path(relative))

    def image(self, relative):
        return validate_image(self.path(relative))

    def crop(self, folder, entry):
        name = entry.get("name")
        _require(isinstance(name, str) and bool(name.strip()), "An extracted item has no name")
        if entry.get("image_path"):
            path = self.path(entry["image_path"])
        else:
            candidates = [self.path(f"{folder}/{safe_component(name)}{extension}")
                          for extension in (".webp", ".png", ".jpg", ".jpeg")]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        return validate_image(path)

    def metadata(self):
        data = self.document("metadata.json")
        _require(isinstance(data.get("extracts"), list), "metadata.json has no extraction list")
        for dimension in ("image_width", "image_height"):
            _require(_finite(data.get(dimension)) and data[dimension] > 0, f"Invalid {dimension} in metadata.json")
        names = set()
        for item in data["extracts"]:
            _require(isinstance(item, dict), "Invalid extraction entry")
            name = item.get("name")
            _require(isinstance(name, str) and name and name not in names, "Missing or duplicate extraction identity")
            names.add(name)
            self.box(item.get("bbox"), name)
        return data

    @staticmethod
    def box(box, name):
        _require(_vector(box, 4) and box[0] < box[2] and box[1] < box[3], f"Invalid bounding box for {name}")

    def parts(self, metadata, options):
        for parent in metadata["extracts"]:
            relative = parent.get("parts_metadata_path")
            if relative is None:
                relative = (granular_directory(self.folder, parent["name"], parent.get("group") or "objects")
                            / "segment_metadata.json").relative_to(self.folder).as_posix()
            path = self.path(relative)
            data = read_document(path)
            _require(data.get("parent_name") == parent["name"], f"Mismatched parent in {relative}")
            if "skip_reason" in data:
                reason = data["skip_reason"]
                _require(reason in {"Object too small", "No visible parts detected"}, f"Invalid skipped result: {relative}")
                if reason == "Object too small":
                    size = self.crop("segments", parent)
                    minimum = options.get("min_item_size")
                    _require(min(size) < (12 if minimum is None else minimum), f"Incorrect size skip: {relative}")
                continue
            _require(isinstance(data.get("extracts"), list), f"Incomplete part metadata: {relative}")
            if data.get("source_image"):
                self.image(data["source_image"])
            names = set()
            for child in data["extracts"]:
                _require(isinstance(child, dict), f"Invalid part in {relative}")
                name = child.get("name")
                _require(isinstance(name, str) and name and name not in names, f"Missing or duplicate part identity: {relative}")
                names.add(name)
                self.box(child.get("bbox"), name)
                self.crop(path.parent.relative_to(self.folder).as_posix(), child)

    def structured(self, key, metadata, options):
        description = key == "image_description"
        full = key == "full_image_rankings"
        filename = "descriptions.v2.json" if description else f"{'full_image' if full else 'per_item'}_rankings.v2.json"
        if full:
            expected = {"full_image": None}
        else:
            items = enumerate_analysis_items(self.folder, metadata)
            minimum = options.get("min_item_size" if description else "min_region_size")
            minimum = 12 if minimum is None else minimum
            expected = {item["name"]: item for item in items
                        if item.get("bbox") and min(item["bbox"][2] - item["bbox"][0],
                                                    item["bbox"][3] - item["bbox"][1]) >= minimum}
            if not items and not self.path(filename).exists():
                return  # The runner legitimately has no objects or parts to process.
        data = self.document(filename)
        _require(data.get("schema_version") == 2, f"Invalid schema version: {filename}")
        profile = options.get("prompt_profile") or default_profiles()[key]
        descriptor = "questions" if description else "criteria"
        prefix = "question" if description else "criterion"
        expected_descriptors = [{**entry, "id": f"{prefix}_{i}"} for i, entry in enumerate(profile[descriptor], 1)]
        _require(data.get(descriptor) == expected_descriptors, f"{filename} does not match the saved analysis questions/criteria")
        _require(data.get("style") == profile["style"] if description else data.get("scope") == ("full_image" if full else "per_item"),
                 f"Invalid result scope/style: {filename}")
        compiled = compile_description(profile) if description else compile_rankings(profile)
        rows = data.get("items")
        _require(isinstance(rows, list), f"Missing item results: {filename}")
        seen = set()
        for row in rows:
            _require(isinstance(row, dict), f"Invalid item result: {filename}")
            name = row.get("name")
            _require(isinstance(name, str) and name in expected and name not in seen, f"Unexpected or duplicate item {name}: {filename}")
            seen.add(name)
            field = "answers" if description else "ratings"
            validate_response(compiled, {field: row.get(field)})
            if expected[name] is not None:
                _require(row.get("bbox") == expected[name]["bbox"], f"Outdated item coordinates for {name}: {filename}")
                _require(row.get("image_path") == expected[name].get("image_path")
                         and row.get("parent_name") == expected[name].get("parent_name"),
                         f"Outdated item identity for {name}: {filename}")
                if row.get("image_path"):
                    self.image(row["image_path"])
        missing = set(expected) - seen
        _require(not missing, f"{filename} is missing {len(missing)} item result(s): {', '.join(sorted(missing)[:3])}")


def validate_forgery_entry(folder, name, entry):
    _require(isinstance(entry, dict) and "error" not in entry, f"Forensic check {name} failed: {entry}")
    _require(_finite(entry.get("score")) and 0 <= entry["score"] <= 1, f"Invalid forensic score: {name}")
    assets = Artifacts(folder)
    for key in ("heatmap_file", "mask_file"):
        if key in entry:
            _require(isinstance(entry[key], str) and Path(entry[key]).name == entry[key], f"Invalid forensic asset: {name}")
            assets.image(f"forgery_detection/{name}/{entry[key]}")


def validate_stage_outputs(key: str, folder: Path, options) -> None:
    assets = Artifacts(folder)
    metadata = assets.metadata()
    extracts = metadata["extracts"]
    if key == "extract_objects":
        for entry in extracts:
            assets.crop("segments", entry)
    elif key == "bbox_composition":
        for entry in extracts:
            composition = entry.get("bbox_composition")
            _require(isinstance(composition, dict), f"Missing bbox composition: {entry['name']}")
            for side in ("left", "right", "top", "bottom"):
                line = composition.get(side, {})
                _require(isinstance(line, dict) and _vector(line.get("start"), 2) and _vector(line.get("end"), 2)
                         and _finite(line.get("distance_px")) and line["distance_px"] >= 0,
                         f"Incomplete bbox composition: {entry['name']} / {side}")
    elif key == "depth_map":
        assets.image("depth/depth.webp")
        for entry in extracts:
            position = entry.get("bbox_position")
            _require(_vector(position, 3) and 0 <= position[2] <= 1, f"Missing/invalid depth: {entry['name']}")
    elif key in {"negative_space", "deepgaze_msdb", "original_export"}:
        assets.image({"negative_space": "negative_space/negative_space.webp",
                      "deepgaze_msdb": "saliency_deepgaze_msdb/saliency_deepgaze_msdb.webp",
                      "original_export": "original/original.webp"}[key])
    elif key == "composition_lines":
        assets.image("composition/composition_lines.webp")
        lines = assets.document("composition/composition_lines.json")
        for name, line in lines.items():
            _require(isinstance(line, dict) and _vector(line.get("start"), 2) and _vector(line.get("end"), 2),
                     f"Invalid composition line: {name}")
    elif key == "analyze_segments":
        assets.parts(metadata, options)
    elif key == "pose_extraction":
        names = {entry["name"] for entry in extracts if (entry.get("group") or "").lower() == "people"}
        if not names and not assets.path("pose/pose.json").exists():
            return
        rows = assets.document("pose/pose.json").get("items")
        _require(isinstance(rows, list), "Missing pose results")
        seen = set()
        for row in rows:
            _require(isinstance(row, dict) and isinstance(row.get("name"), str), "Invalid pose item")
            name = row["name"]
            _require(name in names and name not in seen, f"Unexpected or duplicate pose: {name}")
            seen.add(name)
            _require(isinstance(row.get("points"), list) and isinstance(row.get("lines"), list), f"Incomplete pose: {name}")
            for point in row["points"]:
                _require(isinstance(point, dict) and _finite(point.get("x")) and _finite(point.get("y")), f"Invalid pose point: {name}")
            for line in row["lines"]:
                _require(isinstance(line, dict) and _vector(line.get("start"), 2) and _vector(line.get("end"), 2), f"Invalid pose line: {name}")
            assets.image(f"pose/{name}_pose.webp")
        _require(seen == names, "Pose results are missing one or more people")
    elif key in {"image_description", "full_image_rankings", "per_item_rankings"}:
        assets.structured(key, metadata, options)
    elif key == "forgery_detection":
        data = assets.document("forgery_detection.json")
        jpeg = Path(metadata.get("source_image_path", "")).suffix.lower() in JPEG_EXTENSIONS
        for name in options.get("methods") or AVAILABLE_METHODS:
            if name in JPEG_ONLY_METHODS and not jpeg:
                continue
            _require(name in data, f"Missing forensic result: {name}")
            validate_forgery_entry(folder, name, data[name])
    elif key == "synthetic_analysis":
        data = assets.document("synthetic_analysis.json")
        _, probability = probabilities({"ok": True, "probs": data.get("raw_probs")})
        _require(_finite(data.get("score")) and 0 <= data["score"] <= 10
                 and math.isclose(data["score"], 10 * probability, abs_tol=0.051),
                 "Synthetic score is missing or inconsistent with its probabilities")
    else:
        raise ValueError(f"No output validator for stage: {key}")

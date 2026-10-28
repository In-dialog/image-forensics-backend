"""Versioned semantic configuration. Engine keys and JSON schemas are never settings."""
from __future__ import annotations

import copy
import math

STAGE_KINDS = {"object_detect": "detection", "analyze_segments": "detection", "image_description": "description",
               "full_image_rankings": "ranking", "per_item_rankings": "ranking"}
GROUPS = ("people", "objects", "texts")


def _score(label, positive, low_label, high_label, *, minimum=0, maximum=10, negative=""):
    return dict(label=label, type="score", positive=positive, negative=negative,
                min=minimum, max=maximum, low_label=low_label, high_label=high_label)


def _contextual_description() -> dict:
    return dict(kind="description", style="bullets",
                positive="Describe the target's literal features and visible relations before supported function, action or contextual meaning. Distinguish observations from interpretations and the target from nearby content.",
                negative="Avoid invented context, unseen text, personal identity, beliefs and creator intent. Do not convert a vague resemblance into a definite logo, nationality, religion or political affiliation.",
                questions=[
                    dict(label="Observations", instruction="Describe the target's literal form, orientation, material or graphic carrier, colors and resolved parts. Describe what is visible without introducing a cause, intended use or symbolic identity."),
                    dict(label="Signs and readable text", instruction="Describe separate pictures, emblems, garment graphics, marks and legible writing. State the visual form and carrier of each. Transcribe only readable characters and describe unreadable motifs without inventing their text or identity."),
                    dict(label="Role and relationships", instruction="Describe visible contact, attachment, pose, placement and effects involving the target, including connections beyond its box. Then state any supported use or action as an interpretation; leave unsupported action or intent undetermined."),
                    dict(label="Supported associations", instruction="What plausible functional or semiotic interpretations are supported by the observed features and relationships? Tie each reading to evidence and distinguish it from literal observation. When use or meaning is ambiguous, mention relevant alternatives without choosing an unsupported purpose."),
                    dict(label="Ambiguities", instruction="Which visible details are too small, obscured or context-dependent to identify or interpret reliably? Name the uncertainty and avoid resolving it by guessing."),
                ])


def _contextual_rankings() -> dict:
    return dict(kind="ranking",
                positive="Assess the target's visible role through its features, use, contacts, attachments, effects and placement. Include relationships outside the target box. Distinguish a literal observation from a supported interpretation of function, action or contextual meaning.",
                negative="Do not transfer the scene's overall meaning to every item or invent intended use. Do not attribute private beliefs, identity or affiliation to depicted people.",
                criteria=[
                    _score("Contextual importance", "Rate how much the target helps explain the depicted activity, interaction, composition or tone through its visible use, contact, placement or effects. Consider connected evidence outside the box. A target can be important without political meaning or readable marks.", "Incidental, with little visible contribution", "Central to the depicted activity or meaning", negative="Do not equate mere proximity or physical size with an active role."),
                    _score("Political role in the scene", "Rate political or advocacy relevance supported by the target's visible role, use, interactions, placement or marks in this scene. An unmarked target can have political relevance; when political context is unclear, leave that interpretation undetermined.", "No supported political role", "Clear, central role in political framing", negative="Leave uncertain use or intent undetermined rather than treating it as a visible fact."),
                    _score("Cultural or symbolic role", "Rate cultural, religious or symbolic meaning supported by the target's visible form or its role in this scene. Name the literal cue or relationship and distinguish a contextual reading from a definite identification.", "No supported cultural or symbolic role", "Clear, central cultural or symbolic role", negative="Do not attribute identity or beliefs to a wearer, holder or nearby person."),
                    dict(label="Depicted role", type="choice",
                         positive="Choose the most specific observed role: sign display or worn/attached design when clear, otherwise active use/handling or background/incidental presence. Use null when the role cannot be determined.",
                         negative="Do not substitute a person's beliefs or an imagined intended use for a visible role.",
                         choices=["Used/handled", "Displayed as sign", "Worn/attached design", "Background/incidental"]),
                    dict(label="Explicit call to action", type="boolean",
                         positive="Does the target convey an explicit request to act through readable wording or a directly depicted invitation involving it? Use false for assessable absence and null for ambiguous or unreadable evidence.",
                         negative="Expressive poses or dramatic composition alone do not establish a request."),
                ])


def default_profiles() -> dict:
    detection = dict(kind="detection",
                     positive="Find clear targets across the image, including smaller details and image edges.",
                     negative="Skip ambiguous clutter and guesses. Never fill a quota.", groups={
                         "people": dict(positive="Locate each person in the main scene, including clear partial bodies. Use a neutral person label with useful visible clothing or position attributes.",
                                        negative="Exclude figures on pictures, prints and statues inside the scene. Do not identify people."),
                         "objects": dict(positive="Find physical objects, furniture, structures, garments, small containers and loose items. Include distinct pictures, statues, emblems and printed graphics. Use generic object names with visible distinguishing attributes.",
                                         negative="Exclude whole people and diffuse backgrounds. Avoid symbolic identities and inferred condition."),
                         "texts": dict(positive="Locate readable writing on signs, packaging, books, screens, clothing and other visible surfaces. Transcribe the characters exactly, once per distinct text region.",
                                       negative="Skip unreadable writing. Do not guess words, slogans or names.")})
    parts = copy.deepcopy(detection)
    parts.update(positive="Locate clear parts, attachments and visual details inside this item crop, including bounded graphics or writing on its surface. Use generic labels with useful visible distinguishing attributes.",
                 negative="Exclude the whole parent item and unrelated background. Skip ambiguous details; do not fill the count limit or guess identities.")
    parts["groups"]["people"].update(positive="Find visible garments, accessories, hair, facial or body details, and distinct graphics or patches on clothing. Label a print as a print with visible shape or color.",
                                     negative="Exclude the complete person. Do not assign identity, affiliation or beliefs.")
    parts["groups"]["objects"].update(positive="Find distinct components, attachments, contained items, decorations, pictures, emblems and printed marks. Use visible type and attributes to distinguish each target.",
                                      negative="Exclude the whole parent object, broad surfaces and texture alone. Keep symbolic identities and inferred condition out of labels.")
    description = _contextual_description()
    full = dict(kind="ranking", positive="Assess the whole image using observable features and their locations. Tie interpretations of framing to those observations.",
                negative="Do not infer private beliefs, creator intent or audience reactions.", criteria=[
                    _score("Political visual framing", "Rate visible political or advocacy framing from readable words or the arrangement of symbols. Describe literal features without assigning unprinted names.", "No visible political or advocacy cue", "Explicit, central political or advocacy framing"),
                    _score("Commercial promotion", "Rate explicit promotion through readable offers, prices, branding or directed product presentation. Ordinary objects alone provide no promotional cue.", "No visible promotion", "Explicit, central selling or promotional appeal"),
                    dict(label="Explicit call to action", type="boolean",
                         positive="Is an explicit request to act visibly present? Use true for a readable request or unambiguous depicted invitation, false for assessable absence, and null for a potentially present request that is unreadable or ambiguous.",
                         negative="Expressive poses alone do not establish a request."),
                    _score("Us-versus-them framing", "Rate explicit division through readable us-versus-them wording or a clearly oppositional arrangement. Different symbols alone do not demonstrate hostility.", "No visible divisive framing", "Explicit, strong us-versus-them framing"),
                    _score("Depicted emotional intensity", "Rate the intensity depicted through visible expressions, gestures, posture or contrast. Describe those cues rather than claiming people's actual feelings or predicting audience reactions.", "Little depicted emotional intensity", "Strong depicted emotional intensity"),
                    _score("Visual emphasis", "Rate visual emphasis from contrast, scale, placement and focal structure. Report those features without predicting audience effects.", "Low visual emphasis", "Strong, concentrated visual emphasis")])
    per_item = _contextual_rankings()
    return {"object_detect": detection, "analyze_segments": parts, "image_description": description,
            "full_image_rankings": full, "per_item_rankings": per_item}


def prompt_presets() -> dict:
    defaults = default_profiles()
    result = {stage: [dict(name="Default analysis", profile=profile)] for stage, profile in defaults.items()}
    for stage in ("full_image_rankings", "per_item_rankings"):
        base = defaults[stage]
        for name, criteria in (
            ("Emotional tone", [_score("Valence", "Assess pleasant versus aversive visual tone.", "Aversive", "Pleasant", minimum=-5, maximum=5),
                                _score("Intensity", "Assess visible emotional intensity separately from positive or negative tone.", "Quiet", "Intense")]),
            ("Scene classification", [dict(label="Scene setting", type="choice", positive="Choose the setting supported by visible evidence.", negative="Do not guess when the scene is unclear.", choices=["Indoor", "Outdoor", "Mixed"])]),
            ("Visible evidence", [dict(label="Readable text present", type="boolean", positive="Determine whether readable text is visibly present.", negative="Do not count marks that cannot be read.")]),
            ("Depicted stance", [dict(label="Depicted stance", type="choice",
                                      positive="Classify the depicted stance only when wording or a clear visual relation supports it. Use null when unsupported; a symbol's presence alone is insufficient.",
                                      negative="Do not assign the depicted stance to a person's private beliefs.",
                                      choices=["Support", "Critique", "Neutral depiction", "Mixed"])]),
            ("Political and symbolic evidence", [
                _score("Political visual cues", "Rate political or advocacy cues in the image or marked item. Identify visible wording, forms and their carrier; showing a sign does not establish endorsement.", "No visible political cue", "Clear, central political cues"),
                _score("Cultural or religious motifs", "Rate visible cultural or religious motifs and their prominence. Describe the motif's form and carrier, leaving an uncertain association undetermined.", "No visible motif", "Clear, central motifs", negative="Do not attribute identity or beliefs to the wearer or nearby people."),
                dict(label="Depicted stance", type="choice",
                     positive="Classify the depicted stance only when wording or a clear visual relation supports it. Use null when unsupported; a symbol's presence alone is insufficient.",
                     negative="Do not assign the depicted stance to a person's private beliefs.",
                     choices=["Support", "Critique", "Neutral depiction", "Mixed"]),
                dict(label="Explicit call to action", type="boolean",
                     positive="Is an explicit request to act visible in readable wording or a directly depicted invitation? Use false for clear absence and null for ambiguous or unreadable evidence.",
                     negative="Expressive poses or dramatic composition alone do not establish a request."),
            ]),
        ):
            profile = copy.deepcopy(base)
            profile["criteria"] = criteria
            if stage == "per_item_rankings" and name == "Political and symbolic evidence":
                profile = copy.deepcopy(defaults[stage])
            result[stage].append(dict(name=name, profile=profile))
    for style, name in (("detailed", "Detailed inventory"), ("bullets", "Visible details checklist")):
        profile = copy.deepcopy(defaults["image_description"])
        profile["style"] = style
        profile["questions"] = [dict(label="Visible details", instruction="Describe the subject's distinct visible features, components, materials, color and condition.")]
        result["image_description"].append(dict(name=name, profile=profile))
    inventory = copy.deepcopy(defaults["image_description"])
    result["image_description"].append(dict(name="Semiotic inventory", profile=inventory))
    for stage in ("object_detect", "analyze_segments"):
        profile = copy.deepcopy(defaults[stage])
        profile["positive"] += " Give particular attention to small discarded items, pictures, statues, emblems and graphics carried by garments or other objects, wherever they are visibly present."
        result[stage].append(dict(name="Objects and visual signifiers", profile=profile))
    return result


def _object(value, keys, path):
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a structured analysis object. Reload the analysis interface and edit its focus fields.")
    extra, missing = set(value) - set(keys), set(keys) - set(value)
    if extra or missing:
        raise ValueError(f"{path} has unsupported fields {sorted(extra)} or missing fields {sorted(missing)}.")


def _text(value, path, *, limit=12000, required=False):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"{path} must be {'nonempty ' if required else ''}text of at most {limit} characters.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{path} contains unsupported control characters.")
    return value.strip()


def _list(value, path, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{path} must contain {minimum} to {maximum} entries.")


def _unique_labels(items, path):
    labels = [item["label"].casefold() for item in items]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path} labels must be unique.")


def validate_profile(stage: str, raw: object) -> dict:
    if stage not in STAGE_KINDS:
        raise ValueError(f"Unknown prompt stage: {stage}")
    kind = STAGE_KINDS[stage]
    fields = {"kind", "positive", "negative"} | ({"groups"} if kind == "detection" else {"style", "questions"} if kind == "description" else {"criteria"})
    _object(raw, fields, stage)
    if raw["kind"] != kind:
        raise ValueError(f"{stage}.kind must be {kind}.")
    result = dict(kind=kind, positive=_text(raw["positive"], f"{stage}.positive"), negative=_text(raw["negative"], f"{stage}.negative"))
    if kind == "detection":
        _object(raw["groups"], GROUPS, f"{stage}.groups")
        result["groups"] = {}
        for group in GROUPS:
            value = raw["groups"][group]
            _object(value, ("positive", "negative"), f"{stage}.{group}")
            result["groups"][group] = {key: _text(value[key], f"{stage}.{group}.{key}") for key in ("positive", "negative")}
    elif kind == "description":
        if raw["style"] not in ("concise", "detailed", "bullets"):
            raise ValueError("Description style must be concise, detailed or bullets.")
        result["style"] = raw["style"]
        _list(raw["questions"], "questions", 1, 12)
        result["questions"] = []
        for index, item in enumerate(raw["questions"]):
            path = f"questions[{index}]"
            _object(item, ("label", "instruction"), path)
            result["questions"].append(dict(label=_text(item["label"], path + ".label", limit=120, required=True),
                                            instruction=_text(item["instruction"], path + ".instruction", limit=5000, required=True)))
        _unique_labels(result["questions"], "Question")
    else:
        _list(raw["criteria"], "criteria", 1, 16)
        result["criteria"] = []
        for index, item in enumerate(raw["criteria"]):
            path = f"criteria[{index}]"
            if not isinstance(item, dict) or item.get("type") not in ("score", "choice", "boolean"):
                raise ValueError(f"{path}.type must be score, choice or boolean.")
            extra = {"min", "max", "low_label", "high_label"} if item["type"] == "score" else {"choices"} if item["type"] == "choice" else set()
            _object(item, {"label", "type", "positive", "negative"} | extra, path)
            criterion = dict(label=_text(item["label"], path + ".label", limit=120, required=True), type=item["type"],
                             positive=_text(item["positive"], path + ".positive", limit=5000), negative=_text(item["negative"], path + ".negative", limit=5000))
            if item["type"] == "score":
                for key in ("min", "max"):
                    value = item[key]
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not -1e6 <= value <= 1e6 or not math.isfinite(value):
                        raise ValueError(f"{path}.{key} must be a finite number between -1000000 and 1000000.")
                    criterion[key] = value
                if criterion["min"] >= criterion["max"]:
                    raise ValueError(f"{path} minimum must be less than maximum.")
                for key in ("low_label", "high_label"):
                    criterion[key] = _text(item[key], path + "." + key, limit=200)
            elif item["type"] == "choice":
                _list(item["choices"], path + ".choices", 2, 20)
                criterion["choices"] = [_text(value, path + ".choices", limit=120, required=True) for value in item["choices"]]
                if len({value.casefold() for value in criterion["choices"]}) != len(criterion["choices"]):
                    raise ValueError(f"{path} choices must be unique.")
            result["criteria"].append(criterion)
        _unique_labels(result["criteria"], "Criterion")
    return result


def validate_profiles(raw: object, current: dict | None = None) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("prompts must be an object of structured analysis profiles.")
    result = copy.deepcopy(current or {})
    for stage, profile in raw.items():
        result[stage] = validate_profile(stage, profile)
    return result

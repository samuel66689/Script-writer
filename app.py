"""
Flow AI Video Script & Prompt Generator
========================================
Turns a few choices into a full Flow-AI-ready production package: title,
character reference sheet, a global visual style bible, and scene-by-scene
dialogue + image/motion prompts.

Key design decisions in this version:

  1. Reliability - every Gemini call walks a fallback chain of models with
     retries, reported live to the user, instead of dying on first error.
  2. Longer videos are generated in multiple smaller batched calls (not one
     giant call) so quality doesn't degrade over 20+ scenes, while still
     feeling like a single "Generate" click to the user.
  3. A "story beat" plan (hook -> setup -> escalation -> climax -> twist) is
     computed in code and handed to the model per batch, so pacing holds up
     across the whole video instead of drifting.
  4. Consistency is enforced by CODE - a global style bible and per-character
     reference prompts are appended to every scene's image prompt
     deterministically, rather than hoping the model remembers them.
  5. The API key and this session's generated projects are saved to the
     VIEWER'S OWN BROWSER (localStorage) - never to a server - so a page
     refresh doesn't force re-entering the key or losing past scripts.
"""

from __future__ import annotations

import html
import json
import time
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import errors, types

try:
    from streamlit_local_storage import LocalStorage
    _LOCAL_STORAGE_AVAILABLE = True
except ImportError:
    _LOCAL_STORAGE_AVAILABLE = False


# =====================================================================
# 1. CONSTANTS & CONFIG
# =====================================================================

MODEL_CHOICES: List[str] = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

PRIMARY_RETRY_DELAYS = [2, 4, 8]
FALLBACK_RETRY_DELAYS = [3]
MAX_SCENES_PER_BATCH = 10
MAX_HISTORY_PROJECTS = 15

CATEGORIES: Dict[str, List[str]] = {
    "အသီးခေါင်း AI (Talking Fruit)": [
        "ဟာသ",
        "အချစ်",
        "ပညာပေး",
        "ကလဲ့စား",
        "ဈေးနှုန်းငွေကြေး သရော်စာ",
        "ဘဝရှင်သန်ရေး ဟာသ",
    ],
    "တရားတော် ပုံပြ (Studio Ghibli)": [
        "၅၅၀ ဇာတ်တော်",
        "ဒဿဇာတ်တော်",
        "ဗုဒ္ဓဝင်ဖြစ်ရပ်",
        "စိတ်ခွန်အားဖြည့် ဓမ္မပုံပြင်",
    ],
    "ကလေးများအတွက် 3D Animation": [
        "တိရစ္ဆာန်ပုံပြင်",
        "စာရိတ္တပညာပေး",
        "အိပ်ရာဝင်ပုံပြင်",
        "ဟာသ စွန့်စားခန်း",
    ],
}

SATIRE_GENRES = {"ဈေးနှုန်းငွေကြေး သရော်စာ", "ဘဝရှင်သန်ရေး ဟာသ"}

STYLE_VISUAL_RULES: Dict[str, str] = {
    "အသီးခေါင်း AI (Talking Fruit)": "Expressive, anthropomorphic fruits with clear facial emotion.",
    "တရားတော် ပုံပြ (Studio Ghibli)": "Studio Ghibli style, serene nature, traditional aesthetic, soft warm light.",
    "ကလေးများအတွက် 3D Animation": "Pixar-like Disney 3D animation style, cute and rounded.",
}

# Each duration has a fixed target scene count (not a range) so batching and
# story-beat math is exact, plus a hand-tuned 5-beat pacing plan.
# beats = list of (start_scene, end_scene, BEAT_NAME, beat_description)
DURATIONS: Dict[str, Dict[str, Any]] = {
    "၁ မိနစ်ဝန်းကျင် (Scene ၆-၇ ခု)": {
        "target_scenes": 7,
        "total_seconds": 60,
        "beats": [
            (1, 1, "HOOK", "Open with a surprising, attention-grabbing moment."),
            (2, 3, "SETUP", "Introduce the characters and the situation/conflict."),
            (4, 5, "ESCALATION", "The problem gets worse, funnier, or more dramatic."),
            (6, 6, "CLIMAX", "The peak moment of conflict or comedy."),
            (7, 7, "PUNCHLINE/TWIST", "A satisfying, funny, or surprising resolution."),
        ],
    },
    "၂ မိနစ်ဝန်းကျင် (Scene ၁၀-၁၅ ခု)": {
        "target_scenes": 13,
        "total_seconds": 120,
        "beats": [
            (1, 1, "HOOK", "Open with a surprising, attention-grabbing moment."),
            (2, 4, "SETUP", "Introduce the characters and the situation/conflict."),
            (5, 9, "ESCALATION", "The problem gets worse, funnier, or more dramatic, in stages."),
            (10, 11, "CLIMAX", "The peak moment of conflict or comedy."),
            (12, 13, "PUNCHLINE/TWIST", "A satisfying, funny, or surprising resolution."),
        ],
    },
    "၃ မိနစ်ဝန်းကျင် (Scene ၂၀-၂၅ ခု)": {
        "target_scenes": 23,
        "total_seconds": 180,
        "beats": [
            (1, 1, "HOOK", "Open with a surprising, attention-grabbing moment."),
            (2, 6, "SETUP", "Introduce the characters and the situation/conflict, with a bit more texture."),
            (7, 15, "ESCALATION", "The problem gets worse, funnier, or more dramatic, in multiple stages."),
            (16, 19, "CLIMAX", "The peak moment of conflict or comedy."),
            (20, 23, "PUNCHLINE/TWIST", "A satisfying, funny, or surprising resolution and wrap-up."),
        ],
    },
}

SERIES_OPTIONS = [
    "တစ်ပိုင်းတည်း အပြီး (Standalone Episode)",
    "အပိုင်းဆက် Series - အပိုင်း (၁) အစပျိုး",
]

ASPECT_RATIOS: Dict[str, Dict[str, str]] = {
    "📱 9:16 (Shorts / Reels / TikTok)": {
        "code": "9:16",
        "prompt_text": "Framing: vertical 9:16 mobile aspect ratio, portrait framing.",
    },
    "🖥️ 16:9 (YouTube / Landscape)": {
        "code": "16:9",
        "prompt_text": "Framing: widescreen 16:9 cinematic aspect ratio, landscape framing.",
    },
}

INTENSITY_OPTIONS = ["ချိုသာစွာ", "ပုံမှန်", "စူးရှစွာ"]
INTENSITY_RULES: Dict[str, str] = {
    "ချိုသာစွာ": "Keep the satire light, gentle and family-friendly - playful teasing rather than harsh criticism.",
    "ပုံမှန်": "Keep the satire balanced - clearly witty and critical, but not mean-spirited.",
    "စူးရှစွာ": "Make the satire sharp and biting, using irony and exaggeration, while staying humorous rather than hateful.",
}

SATIRE_SAFETY_RULE = (
    "This is social/economic satire, not a political attack. Do NOT name, depict, imitate, or reference "
    "any real living politician, government official, specific political party, or specific real "
    "organization - no real names, uniforms, flags, or logos tied to real institutions. Comment on the "
    "general situation only (rising prices, long queues, power outages, red tape, unemployment, etc.) "
    "through the fruit characters, never on a named real person. Never write content that incites hatred "
    "or violence against any group."
)

CHARACTER_DETAIL_RULE = (
    "For each character, flow_ai_ref_prompt must be ONE dense English sentence with at least 5 concrete, "
    "specific visual details: exact colors, body/fruit shape, facial features, any accessory or clothing, "
    "and a signature expression or pose. Never rely on vague adjectives like 'cute' or 'happy' alone - pair "
    "every adjective with something a camera could actually see."
)

STYLE_BIBLE_RULE = (
    "Also write a 'style_bible': 2-3 sentences in English describing the ONE global visual identity for "
    "the entire video - name 2-3 specific colors in the palette, the lighting mood, and the art "
    "medium/texture. Keep it generic enough to apply to every scene's background, since it will be "
    "appended to every image prompt."
)

MOTION_PROMPT_RULE = (
    "img_to_video_prompt_en must explicitly state three things and never leave any of them implicit: "
    "(1) the camera movement (e.g. slow push-in, handheld pan, static locked shot, orbit), "
    "(2) exactly what the subject physically does during the clip, and (3) the pacing (slow/medium/fast)."
)

# ---- JSON schemas for Gemini's structured-output mode ----
_CHARACTER_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "character_name": {"type": "STRING"},
        "visual_description": {"type": "STRING"},
        "flow_ai_ref_prompt": {"type": "STRING"},
    },
    "required": ["character_name", "visual_description", "flow_ai_ref_prompt"],
}
_SCENE_ITEM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scene_number": {"type": "INTEGER"},
        "dialogue_myanmar": {"type": "STRING"},
        "image_prompt_en": {"type": "STRING"},
        "img_to_video_prompt_en": {"type": "STRING"},
    },
    "required": ["scene_number", "dialogue_myanmar", "image_prompt_en", "img_to_video_prompt_en"],
}
FIRST_BATCH_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "logline": {"type": "STRING"},
        "style_bible": {"type": "STRING"},
        "character_sheet": {"type": "ARRAY", "items": _CHARACTER_ITEM_SCHEMA},
        "scenes": {"type": "ARRAY", "items": _SCENE_ITEM_SCHEMA},
    },
    "required": ["title", "logline", "style_bible", "character_sheet", "scenes"],
}
FIRST_BATCH_CONTINUATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "logline": {"type": "STRING"},
        "scenes": {"type": "ARRAY", "items": _SCENE_ITEM_SCHEMA},
    },
    "required": ["title", "logline", "scenes"],
}
SCENES_ONLY_SCHEMA = {
    "type": "OBJECT",
    "properties": {"scenes": {"type": "ARRAY", "items": _SCENE_ITEM_SCHEMA}},
    "required": ["scenes"],
}
SINGLE_SCENE_SCHEMA = _SCENE_ITEM_SCHEMA


# =====================================================================
# 2. PAGE CONFIG + DARK CINEMATIC THEME
# =====================================================================

st.set_page_config(
    page_title="Flow AI Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Myanmar:wght@400;500;700&display=swap');
:root {
    --bg: #0F1115; --bg-soft: #171A21; --card: #1B1F27; --border: #2A2F3A;
    --text: #F5F1E8; --text-muted: #9CA3AF; --accent: #F2B134; --accent-soft: rgba(242,177,52,0.12);
}
html, body, [class*="css"] { font-family: 'Noto Sans Myanmar', sans-serif; }
.stApp { background-color: var(--bg); color: var(--text); }
.app-header { border-bottom: 1px solid var(--border); padding-bottom: 14px; margin-bottom: 18px; }
.app-title { color: var(--accent); font-size: 26px; font-weight: 700; margin-bottom: 2px; }
.app-subtitle { color: var(--text-muted); font-size: 14px; }
.section-label { color: var(--accent); font-weight: 700; font-size: 14px; text-transform: uppercase;
    letter-spacing: 0.04em; margin: 18px 0 6px 0; }
.scene-card { background-color: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--accent);
    border-radius: 10px; padding: 16px; margin-bottom: 16px; }
.badge { background-color: var(--accent-soft); color: var(--accent); font-size: 12px; font-weight: 700;
    padding: 3px 10px; border-radius: 12px; display: inline-block; margin-bottom: 10px; border: 1px solid var(--accent); }
.dialogue-box { background-color: var(--bg-soft); border-left: 3px solid var(--accent); padding: 10px 12px;
    border-radius: 6px; margin-bottom: 10px; color: var(--text); }
.hint-text { color: var(--text-muted); font-size: 12px; }
.style-bible-box { background-color: var(--bg-soft); border: 1px dashed var(--accent); border-radius: 8px;
    padding: 10px 12px; margin-bottom: 14px; color: var(--text-muted); font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3. JSON PARSING
# =====================================================================

def parse_json_response(raw_text: Optional[str]) -> Dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise ValueError("AI ဘက်က အဖြေ အလွတ်ပြန်လာပါသည်။")
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("AI ပြန်လာတဲ့ ပုံစံ မမှန်ပါ။")
    return data


# =====================================================================
# 4. PROMPT BUILDERS
# =====================================================================

def build_system_prompt(style_key: str, satire_intensity: Optional[str], hook_required: bool, mode: str) -> str:
    parts = [
        "You are an expert AI scriptwriter and prompt engineer specialized for Flow AI "
        "(image-to-video) production pipelines, creating short entertainment videos for a Myanmar audience.",
        "",
        f"VISUAL STYLE: {STYLE_VISUAL_RULES.get(style_key, '')}",
        "",
        "LANGUAGE: Write dialogue_myanmar in natural, spoken Myanmar. Write image_prompt_en and "
        "img_to_video_prompt_en in detailed, professional English optimized for an AI image/video generator.",
        "",
        MOTION_PROMPT_RULE,
    ]

    if mode in ("first_batch", "first_batch_continuation"):
        parts += ["", CHARACTER_DETAIL_RULE if mode == "first_batch" else ""]
        if mode == "first_batch":
            parts += ["", STYLE_BIBLE_RULE]

    if satire_intensity:
        parts += ["", f"TONE: {INTENSITY_RULES.get(satire_intensity, '')}", "", f"SAFETY: {SATIRE_SAFETY_RULE}"]

    if hook_required and mode in ("first_batch", "first_batch_continuation"):
        parts += ["", "HOOK: This is for short-form vertical platforms (TikTok/Reels/Shorts). Scene 1 must open "
                       "with a strong visual or verbal hook within the first 3 seconds to stop viewers scrolling."]

    if mode == "first_batch_continuation":
        parts += ["", "CONTINUITY: This is the NEXT part of an ongoing series. Reuse the existing characters "
                       "(provided in the user message) exactly as they are - do not redesign them. Write a new "
                       "logline that moves the story forward without repeating earlier events."]
    elif mode == "later_batch":
        parts += ["", "CONTINUITY: You are writing the middle/end portion of a script whose title, characters "
                       "and style bible are already fixed (given in the user message). Stay consistent with them "
                       "and continue the plot forward from the last scene given - do not restart or repeat it."]
    elif mode == "single_scene":
        parts += ["", "You are rewriting ONE specific scene inside an existing script. Keep it consistent with "
                       "the surrounding scenes, characters, style bible and tone given in the context. Return "
                       "only that one scene, with the same scene_number."]

    return "\n".join(p for p in parts if p != "")


def beats_overlapping(beats: List[Tuple[int, int, str, str]], start: int, end: int) -> List[str]:
    lines = []
    for b_start, b_end, name, desc in beats:
        os_, oe_ = max(start, b_start), min(end, b_end)
        if os_ <= oe_:
            rng = f"Scene {os_}" if os_ == oe_ else f"Scenes {os_}-{oe_}"
            lines.append(f"{rng}: {name} - {desc}")
    return lines


def build_batch_instruction(
    batch_start: int, batch_end: int, beats: List[Tuple[int, int, str, str]],
    style: str, genre: str, duration_meta: Dict[str, Any],
    idea: str = "", trending_topic: str = "",
    context: Optional[Dict[str, Any]] = None,
    continuation_idea: str = "", is_first: bool = True, is_continuation: bool = False,
) -> str:
    target = duration_meta["target_scenes"]
    seconds_per_scene = round(duration_meta["total_seconds"] / target)
    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"Total video target: {target} scenes ({duration_meta['total_seconds']}s).",
        f"This request covers ONLY scenes {batch_start} to {batch_end}. Number them exactly "
        f"{batch_start} to {batch_end} in order, with no gaps or repeats.",
        f"Each scene is about {seconds_per_scene}s - keep dialogue_myanmar speakable within that time.",
    ]
    beat_lines = beats_overlapping(beats, batch_start, batch_end)
    if beat_lines:
        lines.append("STORY BEATS for these scenes:")
        lines.extend(beat_lines)

    if context:
        lines.append(f"Established title: {context['title']}")
        lines.append(f"Logline so far: {context['logline']}")
        char_names = ", ".join(c.get("character_name", "") for c in context.get("character_sheet", []))
        lines.append(f"Existing characters (keep consistent, do not redesign): {char_names or 'none'}")
        lines.append(f"Global style bible (keep consistent): {context.get('style_bible', '')}")
        if context.get("last_scene_dialogue"):
            lines.append(f"Previous scene's dialogue, for continuity only: {context['last_scene_dialogue']}")

    if is_first and not is_continuation:
        if trending_topic.strip():
            lines.append(f"Real-world situation to satirize generically (never name real people): {trending_topic.strip()}")
        lines.append(f"Topic/Idea: {idea.strip() if idea.strip() else 'Creative natural storyline'}")

    if is_first and is_continuation:
        lines.append("This is the NEXT part of the same series. Write a short new logline continuing the "
                      "story - do not repeat earlier events.")
        if continuation_idea.strip():
            lines.append(f"Direction for this part: {continuation_idea.strip()}")

    return "\n".join(lines)


def build_single_scene_instruction(
    data: Dict[str, Any], scene_number: int, style: str, genre: str, duration_meta: Dict[str, Any],
) -> str:
    scenes = data.get("scenes", [])
    target = next((s for s in scenes if s.get("scene_number") == scene_number), {})
    prev_s = next((s for s in scenes if s.get("scene_number") == scene_number - 1), None)
    next_s = next((s for s in scenes if s.get("scene_number") == scene_number + 1), None)
    char_lines = "\n".join(
        f"- {c.get('character_name')}: {c.get('visual_description')}" for c in data.get("character_sheet", [])
    )
    beat = next((b for b in duration_meta.get("beats", []) if b[0] <= scene_number <= b[1]), None)

    lines = [
        f"Style: {style}", f"Genre: {genre}",
        f"Story title: {data.get('title', '')}", f"Logline: {data.get('logline', '')}",
        f"Global style bible: {data.get('style_bible', '')}",
        f"Characters:\n{char_lines}",
    ]
    if beat:
        lines.append(f"This scene's story beat: {beat[2]} - {beat[3]}")
    lines.append(f"Rewrite ONLY scene number {scene_number}. Current dialogue to improve or replace: "
                 f"{target.get('dialogue_myanmar', '')}")
    if prev_s:
        lines.append(f"Previous scene ({prev_s['scene_number']}) dialogue for context: {prev_s.get('dialogue_myanmar', '')}")
    if next_s:
        lines.append(f"Next scene ({next_s['scene_number']}) dialogue for context: {next_s.get('dialogue_myanmar', '')}")
    lines.append("Return an improved version of this single scene, keeping the same scene_number.")
    return "\n".join(lines)


# =====================================================================
# 5. CONSISTENCY POST-PROCESSING
# =====================================================================

def build_character_clause(character_sheet: List[Dict[str, Any]]) -> str:
    parts = [f"{c.get('character_name', '')} - {c.get('flow_ai_ref_prompt', '')}"
             for c in character_sheet if c.get("flow_ai_ref_prompt")]
    return "; ".join(parts)


def apply_consistency(
    data: Dict[str, Any], style_bible: str, character_clause: str, aspect_text: str,
    scene_number: Optional[int] = None,
) -> Dict[str, Any]:
    extras = []
    if style_bible:
        extras.append(f"Global style: {style_bible}")
    if character_clause:
        extras.append(f"Character consistency: {character_clause}")
    if aspect_text:
        extras.append(aspect_text)
    suffix = (" | " + " | ".join(extras)) if extras else ""

    for scene in data.get("scenes", []):
        if scene_number is not None and scene.get("scene_number") != scene_number:
            continue
        base_prompt = scene.get("image_prompt_en", "").rstrip()
        if suffix and not base_prompt.endswith(suffix):
            scene["image_prompt_en"] = base_prompt + suffix
    return data


def plan_batches(target_scenes: int, max_per_batch: int = MAX_SCENES_PER_BATCH) -> List[Tuple[int, int]]:
    batches, start = [], 1
    while start <= target_scenes:
        end = min(start + max_per_batch - 1, target_scenes)
        batches.append((start, end))
        start = end + 1
    return batches


# =====================================================================
# 6. GEMINI CALLS WITH FALLBACK + RETRY
# =====================================================================

def _call_gemini(client: "genai.Client", model_name: str, schema: Dict[str, Any],
                  system_prompt: str, user_instruction: str):
    return client.models.generate_content(
        model=model_name,
        contents=user_instruction,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )


def generate_with_fallback(
    status, api_key: str, model_order: List[str], schema: Dict[str, Any],
    system_prompt: str, user_instruction: str, prefix: str = "",
) -> Tuple[str, Dict[str, Any], List[Tuple[str, Any]]]:
    client = genai.Client(api_key=api_key)
    attempt_log: List[Tuple[str, Any]] = []
    last_error: Optional[BaseException] = None

    for idx, model_name in enumerate(model_order):
        delays = PRIMARY_RETRY_DELAYS if idx == 0 else FALLBACK_RETRY_DELAYS
        max_attempts = len(delays) + 1

        for attempt in range(1, max_attempts + 1):
            status.update(label=f"{prefix}🎬 {model_name} ဖြင့် ရေးနေသည်… (ကြိုးစားမှု {attempt}/{max_attempts})")
            try:
                response = _call_gemini(client, model_name, schema, system_prompt, user_instruction)
                data = parse_json_response(response.text)
                status.update(label=f"{prefix}✅ {model_name} ဖြင့် အောင်မြင်ပါသည်", state="complete")
                return model_name, data, attempt_log
            except errors.APIError as e:
                code = getattr(e, "code", None)
                attempt_log.append((model_name, code))
                last_error = e
                if code in (400, 401, 403):
                    status.update(label="❌ API Key (သို့) တောင်းဆိုမှု ပြဿနာ", state="error")
                    raise
                if code in (404, 429):
                    break
                if code and code >= 500 and attempt < max_attempts:
                    time.sleep(delays[attempt - 1])
                    continue
                break
            except ValueError as e:
                attempt_log.append((model_name, "json_error"))
                last_error = e
                if attempt < max_attempts:
                    continue
                break

    status.update(label=f"{prefix}❌ Model အားလုံး ကြိုးစားပြီးပါပြီ၊ မအောင်မြင်ပါ", state="error")
    if last_error:
        raise last_error
    raise RuntimeError("Model တစ်ခုမှ အလုပ်မလုပ်ပါ။")


def friendly_api_error(e: BaseException) -> str:
    code = getattr(e, "code", None)
    msg = str(getattr(e, "message", "") or e)
    low = msg.lower()
    if code == 404:
        return ("⚠️ Model အားလုံး ရှာမတွေ့ပါ။ MODEL_CHOICES ထဲက model နာမည်တွေကို "
                "https://ai.google.dev/gemini-api/docs/models မှာ စစ်ပြီး အသစ်ပြောင်းပါ။")
    if code in (401, 403) or (code == 400 and "api key" in low):
        return "🔑 API Key မှားနေပါသည် (သို့) ခွင့်ပြုချက်မရှိပါ။ Google AI Studio မှ Key ကို ပြန်ကူးပြီး ထည့်ကြည့်ပါ။"
    if code == 429:
        return "⏳ Free quota (သို့) တောင်းဆိုမှု အကန့်အသတ် ပြည့်နေပါသည်။ ခဏစောင့်ပြီး ပြန်စမ်းပါ။"
    if code and code >= 500:
        return "🌐 Google server အားလုံး အလုပ်များနေပါသည်။ ၁-၂ မိနစ်နေမှ ပြန်စမ်းကြည့်ပါ။"
    return f"Error ({code}): {msg}"


# =====================================================================
# 7. MULTI-BATCH SCRIPT ORCHESTRATION
# =====================================================================

def generate_full_script(
    status, api_key: str, model_order: List[str], style_key: str, genre: str,
    duration_meta: Dict[str, Any], satire_intensity: Optional[str], hook_required: bool,
    idea: str = "", trending_topic: str = "",
    continuation_context: Optional[Dict[str, Any]] = None, continuation_idea: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    beats = duration_meta["beats"]
    batches = plan_batches(duration_meta["target_scenes"])
    is_continuation = continuation_context is not None
    models_used: List[str] = []

    # ---- batch 1: title / logline / (characters + style bible if new) / scenes ----
    b_start, b_end = batches[0]
    mode = "first_batch_continuation" if is_continuation else "first_batch"
    schema = FIRST_BATCH_CONTINUATION_SCHEMA if is_continuation else FIRST_BATCH_SCHEMA
    system_prompt = build_system_prompt(style_key, satire_intensity, hook_required, mode)
    user_instruction = build_batch_instruction(
        b_start, b_end, beats, style_key, genre, duration_meta, idea, trending_topic,
        context=None, continuation_idea=continuation_idea, is_first=True, is_continuation=is_continuation,
    )
    prefix = f"[Batch 1/{len(batches)}] " if len(batches) > 1 else ""
    model_used, result, _ = generate_with_fallback(status, api_key, model_order, schema, system_prompt,
                                                     user_instruction, prefix=prefix)
    models_used.append(model_used)

    if is_continuation:
        title = result.get("title") or continuation_context["title"]
        logline = result.get("logline", "")
        character_sheet = continuation_context["character_sheet"]
        style_bible = continuation_context["style_bible"]
    else:
        title = result.get("title", "")
        logline = result.get("logline", "")
        character_sheet = result.get("character_sheet", [])
        style_bible = result.get("style_bible", "")

    scenes = result.get("scenes", [])
    for i, s in enumerate(scenes):
        s["scene_number"] = b_start + i  # force-correct numbering, don't trust the model's count

    # ---- later batches: scenes only, with accumulated context ----
    for i, (b_start, b_end) in enumerate(batches[1:], start=2):
        ctx = {
            "title": title, "logline": logline, "character_sheet": character_sheet,
            "style_bible": style_bible, "last_scene_dialogue": scenes[-1].get("dialogue_myanmar", ""),
        }
        system_prompt = build_system_prompt(style_key, satire_intensity, hook_required=False, mode="later_batch")
        user_instruction = build_batch_instruction(
            b_start, b_end, beats, style_key, genre, duration_meta,
            context=ctx, is_first=False, is_continuation=is_continuation,
        )
        model_used, result2, _ = generate_with_fallback(
            status, api_key, model_order, SCENES_ONLY_SCHEMA, system_prompt, user_instruction,
            prefix=f"[Batch {i}/{len(batches)}] ",
        )
        models_used.append(model_used)
        new_scenes = result2.get("scenes", [])
        for j, s in enumerate(new_scenes):
            s["scene_number"] = b_start + j
        scenes.extend(new_scenes)

    data = {"title": title, "logline": logline, "character_sheet": character_sheet,
            "style_bible": style_bible, "scenes": scenes}
    return models_used, data


# =====================================================================
# 8. BROWSER-SIDE PERSISTENCE (API key + project history)
# =====================================================================
# Saved to the VIEWER'S OWN BROWSER via localStorage - never sent to any
# server. Each visitor keeps only their own key and their own history.

local_storage = LocalStorage() if _LOCAL_STORAGE_AVAILABLE else None


def _ls_get_all() -> Dict[str, Any]:
    if not local_storage:
        return {}
    try:
        return local_storage.getAll(key="ls_get_all") or {}
    except Exception:
        return {}


def _ls_set(key: str, value: str, widget_key: str) -> None:
    if not local_storage:
        return
    try:
        local_storage.setItem(key, value, key=widget_key)
    except Exception:
        pass


def save_projects() -> None:
    trimmed = st.session_state.projects[-MAX_HISTORY_PROJECTS:]
    _ls_set("flow_ai_projects", json.dumps(trimmed, ensure_ascii=False), "ls_set_projects")


if "projects" not in st.session_state:
    st.session_state.projects: List[Dict[str, Any]] = []
if "active_project_idx" not in st.session_state:
    st.session_state.active_project_idx = 0
if "scene_undo" not in st.session_state:
    st.session_state.scene_undo: Dict[str, Dict[str, Any]] = {}
if "_ls_attempts" not in st.session_state:
    st.session_state._ls_attempts = 0
if "bootstrapped" not in st.session_state:
    st.session_state.bootstrapped = False

# The localStorage component needs a browser round-trip before real data
# comes back, so give it a couple of reruns before treating an empty result
# as "genuinely nothing saved yet" rather than "not loaded yet".
if not st.session_state.bootstrapped:
    saved = _ls_get_all()
    st.session_state._ls_attempts += 1
    if saved or st.session_state._ls_attempts >= 3:
        if "api_key_field" not in st.session_state:
            st.session_state["api_key_field"] = saved.get("flow_ai_api_key", "") or ""
        try:
            st.session_state.projects = json.loads(saved.get("flow_ai_projects") or "[]")
        except Exception:
            st.session_state.projects = []
        st.session_state.bootstrapped = True


def current_project() -> Optional[Dict[str, Any]]:
    if not st.session_state.projects:
        return None
    idx = min(st.session_state.active_project_idx, len(st.session_state.projects) - 1)
    return st.session_state.projects[idx]


def current_part(project: Dict[str, Any]) -> Dict[str, Any]:
    idx = min(project.get("active_part_idx", 0), len(project["parts"]) - 1)
    return project["parts"][idx]


# =====================================================================
# 9. HEADER + API KEY
# =====================================================================

st.markdown(
    '<div class="app-header">'
    '<div class="app-title">🎬 Flow AI Video Script & Prompt Generator</div>'
    '<div class="app-subtitle">Flow AI အတွက် Script၊ Character Sheet၊ Style Bible နှင့် Scene-by-Scene Prompts များ ထုတ်ပေးသည့်စနစ်</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">🔑 API Key</div>', unsafe_allow_html=True)
key_col, link_col = st.columns([3, 1])
with key_col:
    api_key_input = st.text_input(
        "Google AI Studio API Key", type="password", placeholder="AIzaSy...",
        label_visibility="collapsed", key="api_key_field",
    )
with link_col:
    st.link_button("🔗 Key ယူရန်", "https://aistudio.google.com/apikey", width="stretch")

if api_key_input.strip() and st.session_state.get("_last_saved_key") != api_key_input.strip():
    _ls_set("flow_ai_api_key", api_key_input.strip(), "ls_set_api_key")
    st.session_state["_last_saved_key"] = api_key_input.strip()

clear_col, note_col = st.columns([1, 3])
with clear_col:
    if st.button("🗑️ Key ဖျက်မည်", width="stretch"):
        _ls_set("flow_ai_api_key", "", "ls_clear_api_key")
        st.session_state["api_key_field"] = ""
        st.session_state["_last_saved_key"] = ""
        st.rerun()
with note_col:
    st.caption("🔒 Key ကို ဒီ browser ထဲမှာပဲ (device တစ်ခုတည်း) မှတ်ထားမှာဖြစ်ပြီး Server ဆီ ဘယ်တော့မှ မပို့ပါဘူး။")

if not _LOCAL_STORAGE_AVAILABLE:
    st.warning("⚠️ `streamlit-local-storage` package ကို install မလုပ်ရသေးလို့ Key/Project များကို browser "
               "ထဲ မသိမ်းနိုင်သေးပါ။ requirements.txt ထဲ `streamlit-local-storage` ထည့်ပါ။")

with st.expander("🔧 Advanced: Model Settings"):
    selected_model = st.selectbox("Gemini Model (ဦးစားပေး)", MODEL_CHOICES, index=0)
    custom_model = st.text_input("Custom model name (မလိုရင် ဗလာထားပါ)", placeholder="gemini-3.8-flash")

# ---- Project history (saved in-browser) ----
if st.session_state.projects:
    with st.expander(f"🕘 ရှေးက Project များ ({len(st.session_state.projects)})"):
        for i in range(len(st.session_state.projects) - 1, -1, -1):
            proj = st.session_state.projects[i]
            proj_title = proj["parts"][0]["data"].get("title", "Untitled") if proj["parts"] else "Untitled"
            row1, row2, row3 = st.columns([3, 1, 1])
            with row1:
                marker = "▶️ " if i == st.session_state.active_project_idx else ""
                st.markdown(f"{marker}**{proj_title}**  \n<span class='hint-text'>{proj.get('created_at', '')}</span>",
                            unsafe_allow_html=True)
            with row2:
                if st.button("ဖွင့်ရန်", key=f"open_proj_{proj['id']}", width="stretch"):
                    st.session_state.active_project_idx = i
                    st.rerun()
            with row3:
                if st.button("ဖျက်ရန်", key=f"del_proj_{proj['id']}", width="stretch"):
                    st.session_state.projects.pop(i)
                    st.session_state.active_project_idx = max(0, min(st.session_state.active_project_idx, len(st.session_state.projects) - 1))
                    save_projects()
                    st.rerun()


# =====================================================================
# 10. GENERATION FORM
# =====================================================================

st.markdown('<div class="section-label">၁။ ဗီဒီယို ပုံစံ (Style)</div>', unsafe_allow_html=True)
style_keys = list(CATEGORIES.keys())
selected_style = st.pills("Style", style_keys, default=style_keys[0], label_visibility="collapsed") or style_keys[0]

st.markdown('<div class="section-label">၂။ ဇာတ်လမ်း အမျိုးအစား (Genre)</div>', unsafe_allow_html=True)
genre_options = CATEGORIES[selected_style]
selected_genre = st.pills("Genre", genre_options, default=genre_options[0],
                           key=f"genre_{selected_style}", label_visibility="collapsed") or genre_options[0]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="section-label">၃။ ကြာချိန်</div>', unsafe_allow_html=True)
    duration_keys = list(DURATIONS.keys())
    video_duration = st.pills("Duration", duration_keys, default=duration_keys[0], label_visibility="collapsed") or duration_keys[0]
with col_b:
    st.markdown('<div class="section-label">၄။ Format</div>', unsafe_allow_html=True)
    aspect_keys = list(ASPECT_RATIOS.keys())
    aspect_label = st.pills("Aspect", aspect_keys, default=aspect_keys[0], label_visibility="collapsed") or aspect_keys[0]

st.markdown('<div class="section-label">၅။ ဇာတ်လမ်း ဖွဲ့စည်းပုံ</div>', unsafe_allow_html=True)
series_type = st.pills("Series", SERIES_OPTIONS, default=SERIES_OPTIONS[0], label_visibility="collapsed") or SERIES_OPTIONS[0]

is_satire = selected_genre in SATIRE_GENRES
trending_topic, satire_intensity = "", None
if is_satire:
    st.markdown('<div class="section-label">🗞️ ယနေ့/လက်ရှိ အခြေအနေ (Optional)</div>', unsafe_allow_html=True)
    trending_topic = st.text_area(
        "Trending topic",
        placeholder="ဥပမာ - ဈေးကွက်ထဲ ဆီစျေးမြင့်တက်နေမှု၊ လျှပ်စစ်မီးပြတ်တောက်မှု၊ ဘတ်စ်ကားတန်းစီနေရမှု...",
        label_visibility="collapsed", height=80,
    )
    st.caption("⚠️ ပုဂ္ဂိုလ်ရေး နာမည် မထည့်ဘဲ 'အခြေအနေ' ကိုသာ ဖော်ပြပါ - AI ကလည်း အမည်ဖော်တာမျိုး ရေးမည်မဟုတ်ပါ။")
    satire_intensity = st.select_slider("သရော်အား", options=INTENSITY_OPTIONS, value=INTENSITY_OPTIONS[1])

st.markdown('<div class="section-label">၆။ ထည့်သွင်းလိုသော အကြောင်းအရာ</div>', unsafe_allow_html=True)
custom_idea = st.text_area(
    "Idea", placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်...",
    label_visibility="collapsed",
)

generate_clicked = st.button("🚀 Script & Prompts ဖန်တီးမည်", type="primary", width="stretch")

if generate_clicked:
    api_key = (api_key_input or "").strip()
    if not api_key:
        st.error("⚠️ ကျေးဇူးပြု၍ အပေါ်ဆုံးက API Key ကို ဦးစွာထည့်သွင်းပေးပါ။")
    else:
        first_choice = custom_model.strip() if custom_model.strip() else selected_model
        model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]
        duration_meta = DURATIONS[video_duration]
        aspect_meta = ASPECT_RATIOS[aspect_label]
        hook_required = aspect_meta["code"] == "9:16"
        n_batches = len(plan_batches(duration_meta["target_scenes"]))

        with st.status(
            f"Gemini model ကို ချိတ်ဆက်နေသည်… ({n_batches} batch လိုအပ်ပါသည်)" if n_batches > 1 else "Gemini model ကို ချိတ်ဆက်နေသည်…",
            expanded=True,
        ) as status:
            try:
                models_used, data = generate_full_script(
                    status, api_key, model_order, selected_style, selected_genre, duration_meta,
                    satire_intensity if is_satire else None, hook_required,
                    idea=custom_idea, trending_topic=trending_topic,
                )
                character_clause = build_character_clause(data["character_sheet"])
                data = apply_consistency(data, data["style_bible"], character_clause, aspect_meta["prompt_text"])

                new_part = {
                    "data": data, "models": models_used, "style": selected_style, "genre": selected_genre,
                    "duration_label": video_duration, "aspect_label": aspect_label,
                    "character_clause": character_clause, "is_satire": is_satire,
                    "satire_intensity": satire_intensity,
                }
                new_project = {
                    "id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "parts": [new_part], "active_part_idx": 0,
                }
                st.session_state.projects.append(new_project)
                if len(st.session_state.projects) > MAX_HISTORY_PROJECTS:
                    st.session_state.projects = st.session_state.projects[-MAX_HISTORY_PROJECTS:]
                st.session_state.active_project_idx = len(st.session_state.projects) - 1
                save_projects()

            except errors.APIError as e:
                st.error(friendly_api_error(e))
                with st.expander("Technical details"):
                    st.code(str(e), language="text")
            except ValueError:
                st.error("⚠️ AI ပြန်လာတဲ့ အဖြေကို ဖတ်မရပါ။ ထပ်နှိပ်ကြည့်ပါ။")
            except Exception as e:
                st.error(f"မမျှော်လင့်သော Error: {e}")


# =====================================================================
# 11. RESULTS DISPLAY
# =====================================================================

project = current_project()
if project:
    st.markdown("---")

    parts = project["parts"]
    if len(parts) > 1:
        part_labels = [f"Part {i + 1}" for i in range(len(parts))]
        chosen = st.pills("Part ရွေးရန်", part_labels, default=part_labels[project.get("active_part_idx", 0)],
                           key=f"part_pills_{project['id']}")
        if chosen:
            project["active_part_idx"] = part_labels.index(chosen)

    part = current_part(project)
    data = part["data"]
    duration_meta = DURATIONS[part["duration_label"]]
    aspect_meta = ASPECT_RATIOS[part["aspect_label"]]

    st.subheader(f"📌 {data.get('title', 'Video Script')}")
    st.markdown(f"**ဇာတ်လမ်း အကျဉ်း:** {data.get('logline', '')}")
    st.markdown(f'<div class="style-bible-box">🎨 <b>Style Bible:</b> {html.escape(data.get("style_bible", ""))}</div>',
                unsafe_allow_html=True)
    st.caption(f"Model: {', '.join(dict.fromkeys(part.get('models', [])))} · Style: {part['style']} · "
               f"Genre: {part['genre']} · {part['aspect_label']}")

    tab_chars, tab_scenes, tab_export = st.tabs(["👤 ဇာတ်ကောင်များ", "🎬 Scenes", "💾 Export"])

    with tab_chars:
        for char in data.get("character_sheet", []):
            with st.expander(f"✨ {char.get('character_name', 'Character')}", expanded=True):
                st.markdown(f"**ရုပ်သွင်:** {char.get('visual_description', '')}")
                st.markdown("**Flow AI Master Reference Prompt:**")
                st.code(char.get("flow_ai_ref_prompt", ""), language="text")

    with tab_scenes:
        for scene in data.get("scenes", []):
            sc_num = scene.get("scene_number", 1)
            undo_key = f"{project['id']}_{part.get('id', project.get('active_part_idx', 0))}_{sc_num}"
            dialogue = html.escape(str(scene.get("dialogue_myanmar", ""))).replace("\n", "<br>")
            st.markdown(
                f'<div class="scene-card"><span class="badge">Scene {sc_num}</span>'
                f'<div class="dialogue-box">🗣️ <b>စကားပြော:</b> {dialogue}</div></div>',
                unsafe_allow_html=True,
            )
            col_img, col_vid = st.columns(2)
            with col_img:
                st.markdown("🖼️ **Image Prompt:**")
                st.code(scene.get("image_prompt_en", ""), language="text")
            with col_vid:
                st.markdown("🎥 **Motion Prompt:**")
                st.code(scene.get("img_to_video_prompt_en", ""), language="text")

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(f"🔄 Scene {sc_num} ပြန်ဆွဲမည်", key=f"regen_{undo_key}", width="stretch"):
                    api_key = (api_key_input or "").strip()
                    if not api_key:
                        st.error("⚠️ API Key ထည့်ပါဦး။")
                    else:
                        first_choice = custom_model.strip() if custom_model.strip() else selected_model
                        model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]
                        system_prompt = build_system_prompt(
                            part["style"], part.get("satire_intensity") if part.get("is_satire") else None,
                            hook_required=False, mode="single_scene",
                        )
                        user_instruction = build_single_scene_instruction(
                            data, sc_num, part["style"], part["genre"], duration_meta,
                        )
                        with st.status(f"Scene {sc_num} ပြန်ဆွဲနေသည်…", expanded=True) as status:
                            try:
                                _model, new_scene, _ = generate_with_fallback(
                                    status, api_key, model_order, SINGLE_SCENE_SCHEMA, system_prompt, user_instruction,
                                )
                                for i, s in enumerate(data["scenes"]):
                                    if s.get("scene_number") == sc_num:
                                        st.session_state.scene_undo[undo_key] = dict(s)
                                        data["scenes"][i] = new_scene
                                        break
                                data = apply_consistency(
                                    data, data["style_bible"], part["character_clause"], aspect_meta["prompt_text"],
                                    scene_number=sc_num,
                                )
                                part["data"] = data
                                save_projects()
                                st.rerun()
                            except errors.APIError as e:
                                st.error(friendly_api_error(e))
                            except Exception as e:
                                st.error(f"မအောင်မြင်ပါ: {e}")
            with btn_col2:
                if undo_key in st.session_state.scene_undo:
                    if st.button(f"↩️ Scene {sc_num} ယခင်ဗားရှင်း ပြန်ယူမည်", key=f"undo_{undo_key}", width="stretch"):
                        for i, s in enumerate(data["scenes"]):
                            if s.get("scene_number") == sc_num:
                                data["scenes"][i] = st.session_state.scene_undo.pop(undo_key)
                                break
                        part["data"] = data
                        save_projects()
                        st.rerun()

    with tab_export:
        st.markdown("**📋 Prompt အားလုံး Copy ရန်:**")
        combined_prompts = "\n\n".join(
            f"Scene {s.get('scene_number')} - IMAGE:\n{s.get('image_prompt_en', '')}\n\n"
            f"Scene {s.get('scene_number')} - MOTION:\n{s.get('img_to_video_prompt_en', '')}"
            for s in data.get("scenes", [])
        )
        st.code(combined_prompts, language="text")

        def build_export_text(d: Dict[str, Any]) -> str:
            out = f"Title: {d.get('title')}\nLogline: {d.get('logline')}\nStyle Bible: {d.get('style_bible')}\n\n--- CHARACTER SHEET ---\n"
            for c in d.get("character_sheet", []):
                out += f"Name: {c.get('character_name')}\nDescription: {c.get('visual_description')}\nRef Prompt: {c.get('flow_ai_ref_prompt')}\n\n"
            out += "--- SCENES ---\n"
            for s in d.get("scenes", []):
                out += (f"Scene {s.get('scene_number')}:\nDialogue: {s.get('dialogue_myanmar')}\n"
                        f"Image Prompt: {s.get('image_prompt_en')}\nMotion Prompt: {s.get('img_to_video_prompt_en')}\n\n")
            return out

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("📄 ဒီ Part Text Download", data=build_export_text(data),
                                file_name=f"flow_ai_part{project.get('active_part_idx', 0) + 1}.txt",
                                mime="text/plain", width="stretch")
        with dl2:
            all_text = "\n\n===== NEXT PART =====\n\n".join(build_export_text(p["data"]) for p in parts)
            st.download_button("📄 Part အားလုံး Download", data=all_text, file_name="flow_ai_full_series.txt",
                                mime="text/plain", width="stretch")
        with dl3:
            st.download_button("💾 Project JSON Save", data=json.dumps(project, ensure_ascii=False, indent=2),
                                file_name="flow_ai_project.json", mime="application/json", width="stretch")

        st.markdown("---")
        st.markdown("**➕ ဆက်တိုက် Part ထပ်ရေးမည်:**")
        next_idea = st.text_area("ဒီအပိုင်းအတွက် ဇာတ်လမ်းလမ်းညွှန် (Optional)", key=f"next_idea_{project['id']}")
        if st.button("➕ နောက် Part ထပ်ဖန်တီးမည်", width="stretch"):
            api_key = (api_key_input or "").strip()
            if not api_key:
                st.error("⚠️ API Key ထည့်ပါဦး။")
            else:
                first_choice = custom_model.strip() if custom_model.strip() else selected_model
                model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]
                cont_context = {"title": data["title"], "character_sheet": data["character_sheet"],
                                 "style_bible": data["style_bible"], "logline": data["logline"]}
                n_batches = len(plan_batches(duration_meta["target_scenes"]))
                with st.status(f"နောက် Part ရေးနေသည်… ({n_batches} batch)" if n_batches > 1 else "နောက် Part ရေးနေသည်…",
                               expanded=True) as status:
                    try:
                        models_used, new_data = generate_full_script(
                            status, api_key, model_order, part["style"], part["genre"], duration_meta,
                            part.get("satire_intensity") if part.get("is_satire") else None,
                            aspect_meta["code"] == "9:16",
                            continuation_context=cont_context, continuation_idea=next_idea,
                        )
                        character_clause = build_character_clause(new_data["character_sheet"])
                        new_data = apply_consistency(new_data, new_data["style_bible"], character_clause, aspect_meta["prompt_text"])

                        new_part = dict(part)
                        new_part["data"] = new_data
                        new_part["models"] = models_used
                        project["parts"].append(new_part)
                        project["active_part_idx"] = len(project["parts"]) - 1
                        save_projects()
                        st.rerun()
                    except errors.APIError as e:
                        st.error(friendly_api_error(e))
                    except Exception as e:
                        st.error(f"မအောင်မြင်ပါ: {e}")


"""
Flow AI Video Script & Prompt Generator
========================================
A Streamlit app that turns a few dropdown choices into a full Flow-AI-ready
video production package: title, character reference sheet, and
scene-by-scene dialogue + image/motion prompts.

Design goals for this version:
  1. Reliability first - Gemini model availability changes often, so every
     generation call walks a fallback chain of models with retries and
     reports live status to the user instead of dying on the first error.
  2. Consistency is enforced by CODE, not by hoping the model remembers -
     character reference prompts and aspect-ratio framing are appended to
     every scene's image prompt deterministically after generation.
  3. Series support - a script can be continued into further parts, each
     one aware of the previous part's title, characters and story so far.
  4. Nothing is lost - the whole project (all parts) can be saved to a
     JSON file and re-loaded later, since Streamlit Cloud has no
     persistent storage between sessions.
"""

from __future__ import annotations

import html
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import errors, types

# =====================================================================
# 1. CONSTANTS & CONFIG
# =====================================================================

# Fallback order for Gemini models. The first entry is tried first; if it
# 404s, is overloaded, or hits quota, the app automatically moves down the
# list. Google renames/retires models fairly often - if everything starts
# 404ing, check https://ai.google.dev/gemini-api/docs/models and update
# this list.
MODEL_CHOICES: List[str] = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# The first model in the chain gets extra retries (it's usually the newest
# / most popular, so it's the one most likely to be temporarily overloaded).
# Fallback models are tried once each so a full outage doesn't take forever.
PRIMARY_RETRY_DELAYS = [2, 4, 8]     # seconds between attempts on the 1st model
FALLBACK_RETRY_DELAYS = [3]          # seconds between attempts on later models

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

# Genres that get the extra satire safety + tone handling.
SATIRE_GENRES = {"ဈေးနှုန်းငွေကြေး သရော်စာ", "ဘဝရှင်သန်ရေး ဟာသ"}

STYLE_VISUAL_RULES: Dict[str, str] = {
    "အသီးခေါင်း AI (Talking Fruit)": "Expressive, anthropomorphic fruits with clear facial emotion.",
    "တရားတော် ပုံပြ (Studio Ghibli)": "Studio Ghibli style, serene nature, traditional aesthetic, soft warm light.",
    "ကလေးများအတွက် 3D Animation": "Pixar-like Disney 3D animation style, cute and rounded.",
}

# label -> (scene count description shown to the model, avg scene count for
# pacing math, total seconds implied by the label)
DURATIONS: Dict[str, Dict[str, Any]] = {
    "၁ မိနစ်ဝန်းကျင် (Scene ၄ မှ ၅ ခု)": {"scenes": "4 to 5", "avg_scenes": 4.5, "total_seconds": 60},
    "၂ မိနစ်ဝန်းကျင် (Scene ၇ မှ ၈ ခု)": {"scenes": "7 to 8", "avg_scenes": 7.5, "total_seconds": 120},
    "၃ မိနစ်ဝန်းကျင် (Scene ၁၀ မှ ၁၂ ခု)": {"scenes": "10 to 12", "avg_scenes": 11, "total_seconds": 180},
}

SERIES_OPTIONS = [
    "တစ်ပိုင်းတည်း အပြီး (Standalone Episode)",
    "အပိုင်းဆက် Series - အပိုင်း (၁) အစပျိုး",
]

ASPECT_RATIOS: Dict[str, Dict[str, str]] = {
    "📱 9:16 (Shorts / Reels / TikTok)": {
        "code": "9:16",
        "prompt_text": "vertical 9:16 mobile aspect ratio, portrait framing",
    },
    "🖥️ 16:9 (YouTube / Landscape)": {
        "code": "16:9",
        "prompt_text": "widescreen 16:9 cinematic aspect ratio, landscape framing",
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

# JSON schemas passed to Gemini's structured-output mode. Using a real
# schema (instead of asking nicely for JSON in the prompt) is far more
# reliable than regex-stripping markdown fences after the fact.
FULL_SCRIPT_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "logline": {"type": "STRING"},
        "character_sheet": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "character_name": {"type": "STRING"},
                    "visual_description": {"type": "STRING"},
                    "flow_ai_ref_prompt": {"type": "STRING"},
                },
                "required": ["character_name", "visual_description", "flow_ai_ref_prompt"],
            },
        },
        "scenes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "scene_number": {"type": "INTEGER"},
                    "dialogue_myanmar": {"type": "STRING"},
                    "image_prompt_en": {"type": "STRING"},
                    "img_to_video_prompt_en": {"type": "STRING"},
                },
                "required": ["scene_number", "dialogue_myanmar", "image_prompt_en", "img_to_video_prompt_en"],
            },
        },
    },
    "required": ["title", "logline", "character_sheet", "scenes"],
}

SINGLE_SCENE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "scene_number": {"type": "INTEGER"},
        "dialogue_myanmar": {"type": "STRING"},
        "image_prompt_en": {"type": "STRING"},
        "img_to_video_prompt_en": {"type": "STRING"},
    },
    "required": ["scene_number", "dialogue_myanmar", "image_prompt_en", "img_to_video_prompt_en"],
}


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
    --bg: #0F1115;
    --bg-soft: #171A21;
    --card: #1B1F27;
    --border: #2A2F3A;
    --text: #F5F1E8;
    --text-muted: #9CA3AF;
    --accent: #F2B134;
    --accent-soft: rgba(242, 177, 52, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Noto Sans Myanmar', sans-serif;
}
.stApp { background-color: var(--bg); color: var(--text); }

.app-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 14px;
    margin-bottom: 18px;
}
.app-title {
    color: var(--accent);
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 2px;
}
.app-subtitle { color: var(--text-muted); font-size: 14px; }

.section-label {
    color: var(--accent);
    font-weight: 700;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 18px 0 6px 0;
}

.scene-card {
    background-color: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}
.badge {
    background-color: var(--accent-soft);
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 12px;
    display: inline-block;
    margin-bottom: 10px;
    border: 1px solid var(--accent);
}
.dialogue-box {
    background-color: var(--bg-soft);
    border-left: 3px solid var(--accent);
    padding: 10px 12px;
    border-radius: 6px;
    margin-bottom: 10px;
    color: var(--text);
}
.hint-text { color: var(--text-muted); font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3. HELPER: JSON PARSING
# =====================================================================

def parse_json_response(raw_text: Optional[str]) -> Dict[str, Any]:
    """Safely turn Gemini's text output into a dict, even if it wraps the
    JSON in markdown fences or adds stray text around it."""
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
# 4. HELPER: PROMPT BUILDERS
# =====================================================================

def build_system_prompt(
    style_key: str,
    satire_intensity: Optional[str] = None,
    hook_required: bool = False,
    mode: str = "full",
) -> str:
    parts = [
        "You are an expert AI scriptwriter and prompt engineer specialized for Flow AI "
        "(image-to-video) production pipelines, creating short entertainment videos for a "
        "Myanmar audience.",
        "",
        f"VISUAL STYLE: {STYLE_VISUAL_RULES.get(style_key, '')}",
        "",
        "LANGUAGE: Write all dialogue/voiceover (dialogue_myanmar) in natural, spoken Myanmar. "
        "Write all image and video-motion prompts (image_prompt_en, img_to_video_prompt_en) in "
        "detailed, professional English optimized for an AI image/video generator.",
    ]

    if satire_intensity:
        parts.append("")
        parts.append(f"TONE: {INTENSITY_RULES.get(satire_intensity, '')}")
        parts.append("")
        parts.append(f"SAFETY: {SATIRE_SAFETY_RULE}")

    if hook_required:
        parts.append("")
        parts.append(
            "HOOK: This is for short-form vertical platforms (TikTok/Reels/Shorts). The very "
            "first scene must open with a strong visual or verbal hook within the first 3 "
            "seconds - a surprising image, a bold question, or a dramatic action - to stop "
            "viewers from scrolling past."
        )

    if mode == "continuation":
        parts.append("")
        parts.append(
            "CONTINUITY: This is a continuation of an earlier part of the same series. Reuse "
            "the same characters (same names and appearances) unless the story naturally "
            "introduces a new one. Move the plot forward - do not repeat earlier events."
        )
    elif mode == "single_scene":
        parts.append("")
        parts.append(
            "You are rewriting ONE specific scene inside an existing script. Keep it "
            "consistent with the surrounding scenes, characters and tone given in the context. "
            "Return only that one scene, with the same scene_number."
        )

    return "\n".join(parts)


def build_user_instruction(
    style: str,
    genre: str,
    duration_meta: Dict[str, Any],
    series_type: str,
    idea: str,
    trending_topic: str = "",
) -> str:
    seconds_per_scene = round(duration_meta["total_seconds"] / duration_meta["avg_scenes"])
    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"Create exactly {duration_meta['scenes']} scenes, numbered sequentially starting at 1.",
        f"Each scene covers about {seconds_per_scene} seconds of video - keep dialogue_myanmar "
        f"short enough to be spoken naturally within that time.",
        f"Series Structure: {series_type}",
    ]
    if trending_topic.strip():
        lines.append(
            f"Real-world situation to satirize (use it only as a generic backdrop, do not name "
            f"real people): {trending_topic.strip()}"
        )
    lines.append(f"Topic/Idea: {idea.strip() if idea.strip() else 'Creative natural storyline'}")
    return "\n".join(lines)


def build_continuation_instruction(
    previous_data: Dict[str, Any],
    style: str,
    genre: str,
    duration_meta: Dict[str, Any],
    idea: str,
) -> str:
    seconds_per_scene = round(duration_meta["total_seconds"] / duration_meta["avg_scenes"])
    char_names = ", ".join(c.get("character_name", "") for c in previous_data.get("character_sheet", []))
    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"This is the NEXT part of an ongoing series titled '{previous_data.get('title', '')}'.",
        f"Previous logline: {previous_data.get('logline', '')}",
        f"Existing characters to reuse: {char_names or 'none yet'}",
        f"Create exactly {duration_meta['scenes']} new scenes for this part, numbered "
        f"sequentially starting at 1.",
        f"Each scene covers about {seconds_per_scene} seconds of video.",
    ]
    if idea.strip():
        lines.append(f"Direction for this part: {idea.strip()}")
    return "\n".join(lines)


def build_single_scene_instruction(
    data: Dict[str, Any], scene_number: int, style: str, genre: str
) -> str:
    scenes = data.get("scenes", [])
    target = next((s for s in scenes if s.get("scene_number") == scene_number), {})
    prev_s = next((s for s in scenes if s.get("scene_number") == scene_number - 1), None)
    next_s = next((s for s in scenes if s.get("scene_number") == scene_number + 1), None)
    char_lines = "\n".join(
        f"- {c.get('character_name')}: {c.get('visual_description')}"
        for c in data.get("character_sheet", [])
    )

    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"Story title: {data.get('title', '')}",
        f"Logline: {data.get('logline', '')}",
        f"Characters:\n{char_lines}",
        f"Rewrite ONLY scene number {scene_number}. Current dialogue to improve or replace: "
        f"{target.get('dialogue_myanmar', '')}",
    ]
    if prev_s:
        lines.append(f"Previous scene ({prev_s['scene_number']}) dialogue for context: {prev_s.get('dialogue_myanmar', '')}")
    if next_s:
        lines.append(f"Next scene ({next_s['scene_number']}) dialogue for context: {next_s.get('dialogue_myanmar', '')}")
    lines.append("Return an improved version of this single scene, keeping the same scene_number.")
    return "\n".join(lines)


# =====================================================================
# 5. HELPER: POST-PROCESSING (character + aspect-ratio consistency)
# =====================================================================

def build_character_clause(data: Dict[str, Any]) -> str:
    chars = data.get("character_sheet", [])
    parts = [
        f"{c.get('character_name', '')} - {c.get('flow_ai_ref_prompt', '')}"
        for c in chars
        if c.get("flow_ai_ref_prompt")
    ]
    return "; ".join(parts)


def apply_consistency(
    data: Dict[str, Any], character_clause: str, aspect_text: str, scene_number: Optional[int] = None
) -> Dict[str, Any]:
    """Deterministically append character-consistency and aspect-ratio notes to
    image prompts, instead of relying on the model to remember them. If
    scene_number is given, only that scene is touched (used for single-scene
    regeneration)."""
    extras = []
    if character_clause:
        extras.append(f"Character consistency: {character_clause}")
    if aspect_text:
        extras.append(aspect_text)
    suffix = (" | " + " | ".join(extras)) if extras else ""

    scenes = data.get("scenes", [])
    for scene in scenes:
        if scene_number is not None and scene.get("scene_number") != scene_number:
            continue
        base_prompt = scene.get("image_prompt_en", "").rstrip()
        if suffix and not base_prompt.endswith(suffix):
            scene["image_prompt_en"] = base_prompt + suffix
    return data


# =====================================================================
# 6. HELPER: GEMINI CALLS WITH FALLBACK + RETRY
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
    status,
    api_key: str,
    model_order: List[str],
    schema: Dict[str, Any],
    system_prompt: str,
    user_instruction: str,
) -> Tuple[str, Dict[str, Any], List[Tuple[str, Any]]]:
    """Try each model in model_order, retrying transient failures, and
    return (model_used, parsed_data, attempt_log). Raises the last error if
    every model fails, or immediately for auth/bad-request errors since
    retrying won't help those."""
    client = genai.Client(api_key=api_key)
    attempt_log: List[Tuple[str, Any]] = []
    last_error: Optional[BaseException] = None

    for idx, model_name in enumerate(model_order):
        delays = PRIMARY_RETRY_DELAYS if idx == 0 else FALLBACK_RETRY_DELAYS
        max_attempts = len(delays) + 1

        for attempt in range(1, max_attempts + 1):
            status.update(label=f"🎬 {model_name} ဖြင့် ရေးနေသည်… (ကြိုးစားမှု {attempt}/{max_attempts})")
            try:
                response = _call_gemini(client, model_name, schema, system_prompt, user_instruction)
                data = parse_json_response(response.text)
                status.update(label=f"✅ {model_name} ဖြင့် အောင်မြင်ပါသည်", state="complete")
                return model_name, data, attempt_log

            except errors.APIError as e:
                code = getattr(e, "code", None)
                attempt_log.append((model_name, code))
                last_error = e

                if code in (400, 401, 403):
                    # API key / request itself is invalid - retrying or
                    # switching models will not fix this.
                    status.update(label="❌ API Key (သို့) တောင်းဆိုမှု ပြဿနာ", state="error")
                    raise
                if code == 404:
                    break  # this model doesn't exist for this key - next model
                if code == 429:
                    break  # quota/rate-limited on this model - next model
                if code and code >= 500 and attempt < max_attempts:
                    time.sleep(delays[attempt - 1])
                    continue
                break  # exhausted retries or unknown error - next model

            except ValueError as e:
                # Model returned text that could not be parsed as JSON.
                attempt_log.append((model_name, "json_error"))
                last_error = e
                if attempt < max_attempts:
                    continue
                break

    status.update(label="❌ Model အားလုံး ကြိုးစားပြီးပါပြီ၊ မအောင်မြင်ပါ", state="error")
    if last_error:
        raise last_error
    raise RuntimeError("Model တစ်ခုမှ အလုပ်မလုပ်ပါ။")


def friendly_api_error(e: BaseException) -> str:
    code = getattr(e, "code", None)
    msg = str(getattr(e, "message", "") or e)
    low = msg.lower()

    if code == 404:
        return (
            "⚠️ Model အားလုံး ရှာမတွေ့ပါ / အသုံးပြုခွင့်မရှိတော့ပါ။ ဖိုင်ထိပ်က MODEL_CHOICES "
            "ထဲက model နာမည်တွေကို https://ai.google.dev/gemini-api/docs/models မှာ စစ်ပြီး "
            "အသစ်ပြောင်းပါ။"
        )
    if code in (401, 403) or (code == 400 and "api key" in low):
        return "🔑 API Key မှားနေပါသည် (သို့) ခွင့်ပြုချက်မရှိပါ။ Google AI Studio မှ Key ကို ပြန်ကူးပြီး ထည့်ကြည့်ပါ။"
    if code == 429:
        return "⏳ Free quota (သို့) တောင်းဆိုမှု အကန့်အသတ် ပြည့်နေပါသည်။ ခဏစောင့်ပြီး ပြန်စမ်းပါ။"
    if code and code >= 500:
        return "🌐 Google server အားလုံး အလုပ်များနေပါသည်။ ၁-၂ မိနစ်နေမှ ပြန်စမ်းကြည့်ပါ။"
    return f"Error ({code}): {msg}"


# =====================================================================
# 7. SESSION STATE
# =====================================================================

if "parts" not in st.session_state:
    st.session_state.parts: List[Dict[str, Any]] = []
if "active_part" not in st.session_state:
    st.session_state.active_part = 0


def current_part() -> Optional[Dict[str, Any]]:
    if not st.session_state.parts:
        return None
    idx = min(st.session_state.active_part, len(st.session_state.parts) - 1)
    return st.session_state.parts[idx]


# =====================================================================
# 8. HEADER
# =====================================================================

st.markdown(
    '<div class="app-header">'
    '<div class="app-title">🎬 Flow AI Video Script & Prompt Generator</div>'
    '<div class="app-subtitle">Flow AI အတွက် Script၊ Character Sheet နှင့် Scene-by-Scene Prompts များ ထုတ်ပေးသည့်စနစ်</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---- API key (kept in main body, not the sidebar, so it's visible on mobile) ----
st.markdown('<div class="section-label">🔑 API Key</div>', unsafe_allow_html=True)
key_col, link_col = st.columns([3, 1])
with key_col:
    api_key_input = st.text_input(
        "Google AI Studio API Key",
        type="password",
        placeholder="AIzaSy...",
        label_visibility="collapsed",
    )
with link_col:
    st.link_button("🔗 Key ယူရန်", "https://aistudio.google.com/apikey", width="stretch")
st.caption("🔒 Key ကို လက်ရှိ session အတွင်းသာ သုံးပြီး မည်သည့်နေရာမျှ မသိမ်းထားပါ။")

with st.expander("🔧 Advanced: Model Settings"):
    selected_model = st.selectbox("Gemini Model (ဦးစားပေး)", MODEL_CHOICES, index=0)
    custom_model = st.text_input("Custom model name (မလိုရင် ဗလာထားပါ)", placeholder="gemini-3.8-flash")

with st.expander("📂 လက်ရှိစီမံကိန်း ဆက်လုပ်ရန် (Project ဖိုင် Upload)"):
    uploaded_project = st.file_uploader("သိမ်းထားတဲ့ Project JSON ဖိုင်", type=["json"])
    if uploaded_project is not None and st.button("📥 Project ကို Load လုပ်မည်"):
        try:
            loaded = json.load(uploaded_project)
            if isinstance(loaded, list):
                st.session_state.parts = loaded
                st.session_state.active_part = len(loaded) - 1
                st.success(f"Part {len(loaded)} ခု Load လုပ်ပြီးပါပြီ။")
                st.rerun()
            else:
                st.error("ဒီဖိုင်က Project JSON ပုံစံ မှန်ကန်ပါဘူး။")
        except Exception as e:
            st.error(f"ဖိုင်ဖတ်လို့ မရပါ: {e}")


# =====================================================================
# 9. GENERATION FORM
# =====================================================================

st.markdown('<div class="section-label">၁။ ဗီဒီယို ပုံစံ (Style)</div>', unsafe_allow_html=True)
selected_style = st.pills("Style", list(CATEGORIES.keys()), default=list(CATEGORIES.keys())[0],
                           label_visibility="collapsed")
selected_style = selected_style or list(CATEGORIES.keys())[0]

st.markdown('<div class="section-label">၂။ ဇာတ်လမ်း အမျိုးအစား (Genre)</div>', unsafe_allow_html=True)
genre_options = CATEGORIES[selected_style]
selected_genre = st.pills(
    "Genre", genre_options, default=genre_options[0],
    key=f"genre_{selected_style}", label_visibility="collapsed",
)
selected_genre = selected_genre or genre_options[0]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="section-label">၃။ ကြာချိန်</div>', unsafe_allow_html=True)
    video_duration = st.pills("Duration", list(DURATIONS.keys()), default=list(DURATIONS.keys())[0],
                               label_visibility="collapsed")
    video_duration = video_duration or list(DURATIONS.keys())[0]
with col_b:
    st.markdown('<div class="section-label">၄။ Format</div>', unsafe_allow_html=True)
    aspect_label = st.pills("Aspect", list(ASPECT_RATIOS.keys()), default=list(ASPECT_RATIOS.keys())[0],
                             label_visibility="collapsed")
    aspect_label = aspect_label or list(ASPECT_RATIOS.keys())[0]

st.markdown('<div class="section-label">၅။ ဇာတ်လမ်း ဖွဲ့စည်းပုံ</div>', unsafe_allow_html=True)
series_type = st.pills("Series", SERIES_OPTIONS, default=SERIES_OPTIONS[0], label_visibility="collapsed")
series_type = series_type or SERIES_OPTIONS[0]

is_satire = selected_genre in SATIRE_GENRES
trending_topic = ""
satire_intensity = None
if is_satire:
    st.markdown('<div class="section-label">🗞️ ယနေ့/လက်ရှိ အခြေအနေ (Optional)</div>', unsafe_allow_html=True)
    trending_topic = st.text_area(
        "Trending topic",
        placeholder="ဥပမာ - ဈေးကွက်ထဲ ဆီစျေးမြင့်တက်နေမှု၊ လျှပ်စစ်မီးပြတ်တောက်မှု၊ ဘတ်စ်ကားတန်းစီနေရမှု...",
        label_visibility="collapsed",
        height=80,
    )
    st.caption("⚠️ ကျေးဇူးပြု၍ ပုဂ္ဂိုလ်ရေး နာမည် မထည့်ဘဲ 'အခြေအနေ' ကိုသာ ဖော်ပြပါ - AI ကလည်း အမည်ဖော်တာမျိုး ရေးမည်မဟုတ်ပါ။")
    satire_intensity = st.select_slider("သရော်အား", options=INTENSITY_OPTIONS, value=INTENSITY_OPTIONS[1])

st.markdown('<div class="section-label">၆။ ထည့်သွင်းလိုသော အကြောင်းအရာ</div>', unsafe_allow_html=True)
custom_idea = st.text_area(
    "Idea",
    placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်...",
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

        system_prompt = build_system_prompt(
            style_key=selected_style,
            satire_intensity=satire_intensity if is_satire else None,
            hook_required=hook_required,
            mode="full",
        )
        user_instruction = build_user_instruction(
            style=selected_style, genre=selected_genre, duration_meta=duration_meta,
            series_type=series_type, idea=custom_idea, trending_topic=trending_topic,
        )

        with st.status("Gemini model ကို ချိတ်ဆက်နေသည်…", expanded=True) as status:
            try:
                used_model, data, _log = generate_with_fallback(
                    status, api_key, model_order, FULL_SCRIPT_SCHEMA, system_prompt, user_instruction,
                )
                character_clause = build_character_clause(data)
                data = apply_consistency(data, character_clause, aspect_meta["prompt_text"])

                new_part = {
                    "data": data,
                    "model": used_model,
                    "style": selected_style,
                    "genre": selected_genre,
                    "duration_label": video_duration,
                    "aspect_label": aspect_label,
                    "character_clause": character_clause,
                    "is_satire": is_satire,
                    "satire_intensity": satire_intensity,
                }
                st.session_state.parts = [new_part]
                st.session_state.active_part = 0

            except errors.APIError as e:
                st.error(friendly_api_error(e))
                with st.expander("Technical details"):
                    st.code(str(e), language="text")
            except ValueError:
                st.error("⚠️ AI ပြန်လာတဲ့ အဖြေကို ဖတ်မရပါ။ ထပ်နှိပ်ကြည့်ပါ။")
            except Exception as e:
                st.error(f"မမျှော်လင့်သော Error: {e}")


# =====================================================================
# 10. RESULTS DISPLAY
# =====================================================================

parts = st.session_state.parts
if parts:
    st.markdown("---")

    if len(parts) > 1:
        part_labels = [f"Part {i + 1}" for i in range(len(parts))]
        chosen = st.pills("Part ရွေးရန်", part_labels, default=part_labels[st.session_state.active_part])
        if chosen:
            st.session_state.active_part = part_labels.index(chosen)

    part = current_part()
    data = part["data"]
    aspect_meta = ASPECT_RATIOS[part["aspect_label"]]

    st.subheader(f"📌 {data.get('title', 'Video Script')}")
    st.markdown(f"**ဇာတ်လမ်း အကျဉ်း:** {data.get('logline', '')}")
    st.caption(f"Model: {part['model']} · Style: {part['style']} · Genre: {part['genre']} · {part['aspect_label']}")

    tab_chars, tab_scenes, tab_export = st.tabs(["👤 ဇာတ်ကောင်များ", "🎬 Scenes", "💾 Export"])

    # ---- Characters tab ----
    with tab_chars:
        for char in data.get("character_sheet", []):
            with st.expander(f"✨ {char.get('character_name', 'Character')}", expanded=True):
                st.markdown(f"**ရုပ်သွင်:** {char.get('visual_description', '')}")
                st.markdown("**Flow AI Master Reference Prompt:**")
                st.code(char.get("flow_ai_ref_prompt", ""), language="text")

    # ---- Scenes tab ----
    with tab_scenes:
        for scene in data.get("scenes", []):
            sc_num = scene.get("scene_number", 1)
            dialogue = html.escape(str(scene.get("dialogue_myanmar", ""))).replace("\n", "<br>")
            st.markdown(
                f'<div class="scene-card">'
                f'<span class="badge">Scene {sc_num}</span>'
                f'<div class="dialogue-box">🗣️ <b>စကားပြော:</b> {dialogue}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col_img, col_vid = st.columns(2)
            with col_img:
                st.markdown("🖼️ **Image Prompt:**")
                st.code(scene.get("image_prompt_en", ""), language="text")
            with col_vid:
                st.markdown("🎥 **Motion Prompt:**")
                st.code(scene.get("img_to_video_prompt_en", ""), language="text")

            if st.button(f"🔄 Scene {sc_num} ပြန်ဆွဲမည်", key=f"regen_{st.session_state.active_part}_{sc_num}"):
                api_key = (api_key_input or "").strip()
                if not api_key:
                    st.error("⚠️ API Key ထည့်ပါဦး။")
                else:
                    first_choice = custom_model.strip() if custom_model.strip() else selected_model
                    model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]

                    system_prompt = build_system_prompt(
                        style_key=part["style"],
                        satire_intensity=part.get("satire_intensity") if part.get("is_satire") else None,
                        mode="single_scene",
                    )
                    user_instruction = build_single_scene_instruction(
                        data, sc_num, part["style"], part["genre"],
                    )

                    with st.status(f"Scene {sc_num} ပြန်ဆွဲနေသည်…", expanded=True) as status:
                        try:
                            _model, new_scene, _log = generate_with_fallback(
                                status, api_key, model_order, SINGLE_SCENE_SCHEMA,
                                system_prompt, user_instruction,
                            )
                            for i, s in enumerate(data["scenes"]):
                                if s.get("scene_number") == sc_num:
                                    data["scenes"][i] = new_scene
                                    break
                            data = apply_consistency(
                                data, part["character_clause"], aspect_meta["prompt_text"],
                                scene_number=sc_num,
                            )
                            part["data"] = data
                            st.rerun()
                        except errors.APIError as e:
                            st.error(friendly_api_error(e))
                        except Exception as e:
                            st.error(f"မအောင်မြင်ပါ: {e}")

    # ---- Export tab ----
    with tab_export:
        st.markdown("**📋 Prompt အားလုံး Copy ရန်:**")
        combined_prompts = "\n\n".join(
            f"Scene {s.get('scene_number')} - IMAGE:\n{s.get('image_prompt_en', '')}\n\n"
            f"Scene {s.get('scene_number')} - MOTION:\n{s.get('img_to_video_prompt_en', '')}"
            for s in data.get("scenes", [])
        )
        st.code(combined_prompts, language="text")

        def build_export_text(d: Dict[str, Any]) -> str:
            out = f"Title: {d.get('title')}\nLogline: {d.get('logline')}\n\n--- CHARACTER SHEET ---\n"
            for c in d.get("character_sheet", []):
                out += f"Name: {c.get('character_name')}\n"
                out += f"Description: {c.get('visual_description')}\n"
                out += f"Ref Prompt: {c.get('flow_ai_ref_prompt')}\n\n"
            out += "--- SCENES ---\n"
            for s in d.get("scenes", []):
                out += f"Scene {s.get('scene_number')}:\n"
                out += f"Dialogue: {s.get('dialogue_myanmar')}\n"
                out += f"Image Prompt: {s.get('image_prompt_en')}\n"
                out += f"Motion Prompt: {s.get('img_to_video_prompt_en')}\n\n"
            return out

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                "📄 ဒီ Part Text Download", data=build_export_text(data),
                file_name=f"flow_ai_part{st.session_state.active_part + 1}.txt",
                mime="text/plain", width="stretch",
            )
        with dl_col2:
            all_text = "\n\n===== NEXT PART =====\n\n".join(
                build_export_text(p["data"]) for p in parts
            )
            st.download_button(
                "📄 Part အားလုံး Download", data=all_text,
                file_name="flow_ai_full_series.txt", mime="text/plain",
                width="stretch",
            )
        with dl_col3:
            st.download_button(
                "💾 Project JSON Save", data=json.dumps(parts, ensure_ascii=False, indent=2),
                file_name="flow_ai_project.json", mime="application/json",
                width="stretch",
            )

        st.markdown("---")
        st.markdown("**➕ ဆက်တိုက် Part ထပ်ရေးမည်:**")
        next_idea = st.text_area(
            "ဒီအပိုင်းအတွက် ဇာတ်လမ်းလမ်းညွှန် (Optional)",
            key=f"next_idea_{st.session_state.active_part}",
        )
        if st.button("➕ နောက် Part ထပ်ဖန်တီးမည်", width="stretch"):
            api_key = (api_key_input or "").strip()
            if not api_key:
                st.error("⚠️ API Key ထည့်ပါဦး။")
            else:
                first_choice = custom_model.strip() if custom_model.strip() else selected_model
                model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]
                duration_meta = DURATIONS[part["duration_label"]]

                system_prompt = build_system_prompt(
                    style_key=part["style"],
                    satire_intensity=part.get("satire_intensity") if part.get("is_satire") else None,
                    hook_required=aspect_meta["code"] == "9:16",
                    mode="continuation",
                )
                user_instruction = build_continuation_instruction(
                    data, part["style"], part["genre"], duration_meta, next_idea,
                )

                with st.status("နောက် Part ရေးနေသည်…", expanded=True) as status:
                    try:
                        used_model, new_data, _log = generate_with_fallback(
                            status, api_key, model_order, FULL_SCRIPT_SCHEMA,
                            system_prompt, user_instruction,
                        )
                        character_clause = build_character_clause(new_data)
                        new_data = apply_consistency(new_data, character_clause, aspect_meta["prompt_text"])

                        new_part = dict(part)
                        new_part["data"] = new_data
                        new_part["model"] = used_model
                        new_part["character_clause"] = character_clause

                        st.session_state.parts.append(new_part)
                        st.session_state.active_part = len(st.session_state.parts) - 1
                        st.rerun()
                    except errors.APIError as e:
                        st.error(friendly_api_error(e))
                    except Exception as e:
                        st.error(f"မအောင်မြင်ပါ: {e}")
                        

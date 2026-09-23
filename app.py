"""
Flow AI Video Script & Prompt Generator
========================================
Turns a few choices into a Flow-AI-ready video production package: title,
character reference sheet, a reusable visual "style bible", and
scene-by-scene dialogue + image/motion prompts.

Key design decisions in this version:

1. RELIABILITY - every Gemini call walks a fallback chain of models with
   retries, and reports live status instead of dying on the first error.
2. QUALITY AT LENGTH - longer scripts (10-25 scenes) are generated in
   batches of ~7-8 scenes instead of one giant call, and each batch is
   told exactly which part of the story arc (hook / setup / rising action /
   climax / resolution) it is responsible for. This keeps later scenes as
   sharp as the opening instead of trailing off into generic filler.
3. CONSISTENCY BY CODE - a character reference clause, a one-paragraph
   "style bible" (palette/lighting/medium), and the aspect-ratio framing
   are appended to every scene's image prompt deterministically after
   generation, instead of hoping the model remembers them 20 scenes later.
4. NOTHING IS LOST ON REFRESH - the API key (opt-in), the current project,
   and a short history of past generations are mirrored into the viewer's
   own browser localStorage, so a page refresh doesn't send the user back
   to a blank form. This is per-browser, client-side only; nothing is
   written to any server or shared between users.
"""

from __future__ import annotations

import html
import json
import math
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from google import genai
from google.genai import errors, types

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

MAX_SCENES_PER_BATCH = 8  # keeps each single call short enough to stay sharp

CATEGORIES: Dict[str, List[str]] = {
    "အသီးခေါင်း AI (Talking Fruit)": [
        "ဟာသ",
        "အချစ်",
        "မိသားစု ဒရမ်မာ",
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

MELODRAMA_GENRE = "မိသားစု ဒရမ်မာ"

# Story archetypes modeled on top-performing Myanmar TikTok talking-fruit
# melodramas (family guilt, regret, karma, betrayal, hidden kindness).
MELODRAMA_ARCHETYPES: Dict[str, str] = {
    "😭 နောင်တ": (
        "A child (or spouse) rejects, mocks or abandons their poor or aging parents "
        "(or devoted partner), chasing pride or wealth - then falls hard and realizes "
        "the truth only when it is TOO LATE. Ends in kneeling regret and tears."
    ),
    "⚖️ ဝဋ်ကြွေး": (
        "Someone commits a cruel or unjust act (cheating, cruelty to in-laws, greed) "
        "and karma returns it multiplied. The wrongdoer faces a mirror of their own "
        "cruelty; moral justice lands like a thunderclap."
    ),
    "🎭 နှစ်မျက်နှာ": (
        "A trusted character (friend, relative, spouse) secretly works against the "
        "protagonist - sweet face in public, poison in private. Build dramatic irony "
        "where the audience suspects before the victim does, then a public unmasking."
    ),
    "💔 အထင်လွဲ": (
        "A stepmother, mother-in-law or daughter-in-law is assumed cruel by everyone, "
        "but she secretly loves and sacrifices for the family. The twist reveals her "
        "hidden kindness and shames those who judged her."
    ),
    "👻 ဝိညာဉ်": (
        "A ghost or supernatural presence tied to guilt or injustice haunts the living "
        "until a wrong is righted. Spooky atmosphere in service of a moral reckoning, "
        "not jump-scare horror."
    ),
}

MELODRAMA_RULE = (
    "MELODRAMA MODE - Burmese family melodrama mini-movie in the spirit of top Myanmar "
    "TikTok drama channels. Forget comedy: the engine of this story is strong moral emotion - "
    "family guilt, parental love, betrayal, regret, karmic justice. The audience should feel "
    "like crying or be deeply moved, never laughing.\n"
    "HOOK (scene 1): open on an extreme emotional close-up already in motion - a crying mother "
    "caressing her child's face, a shocked face, trembling hands, a shattered photo frame on "
    "the floor. NO exposition, NO greetings, NO setup dialogue. Pure feeling in the first seconds, "
    "paired with one punchy Burmese subtitle-style line.\n"
    "CHARACTERS: 3-5 expressive anthropomorphic fruits with clear family roles (mother, "
    "daughter-in-law, stepmother, son, elder, rival). At least one character carries a hidden "
    "motive or secret that pays off in the twist.\n"
    "DIALOGUE: natural spoken Myanmar, emotional but never cringe; short lines an actor could "
    "cry through. Let silence and close-ups do half the work.\n"
    "TITLE: give the story a short, evocative Burmese moral phrase as its title (in the spirit of "
    "'ဝဋ်ကြွေး', 'အချိန်လွန်နောင်တ', 'မိထွေးမေတ္တာ'). Never a generic or English title."
)

MELODRAMA_VISUAL_RULE = (
    "MELODRAMA VISUAL DNA - match the look of top Myanmar TikTok talking-fruit dramas:\n"
    "CHARACTERS: every character is an anthropomorphic fruit - a realistic, highly expressive "
    "fruit head (coconut, durian, jackfruit, eggplant, orange, strawberry, mango, mangosteen, "
    "lemon, pumpkin...) on a human-like body wearing Myanmar traditional dress (longyi, aingyi, "
    "taikpon where fitting). Faces must be built for extreme close-ups: able to cry, tremble, "
    "glare and break down convincingly.\n"
    "CINEMATOGRAPHY (Burmese TV-drama grammar): carry emotional beats with extreme close-ups and "
    "slow push-ins (a tear rolling down, trembling hands, a shocked stare); stage confrontations "
    "as wide theatrical shots (family gathered in a wooden house, a wedding hall full of guests); "
    "keep meaningful objects in frame (a shattered photo frame, an empty chair, a worn family photo).\n"
    "LIGHTING & COLOR: warm golden interiors for tender family moments; cold blue-grey moonlight or "
    "rain for sorrow and showdowns. Rich, saturated cinematic color like a Burmese movie poster.\n"
    "SETTINGS: authentic Myanmar life - bamboo hut, wooden stilt house, village lane, market, "
    "wedding hall, pagoda compound. Never generic Western interiors.\n"
    "STYLE BIBLE: the style_bible paragraph MUST lock in this Burmese melodrama cinema look "
    "(palette, lighting mood, lens feel, film texture) so every scene reads as frames from the "
    "same mini-movie."
)

# (lower_bound_fraction, upper_bound_fraction, act description)
MELODRAMA_ACTS: List[Tuple[float, float, str]] = [    (0.00, 0.10, "EMOTIONAL HOOK - an extreme emotional close-up already in motion (tears, shock, trembling hands). No exposition, no greetings - pure feeling."),
    (0.10, 0.25, "SETUP - who this family is, shown through action and meaningful objects (a worn photo, an empty chair), never explained."),
    (0.25, 0.65, "ESCALATION - confrontations and accusations tighten step by step; raise the emotional stakes with every scene."),
    (0.65, 0.85, "TWIST / REVEAL - one revelation that reframes everything the viewer believed so far."),
    (0.85, 1.01, "MORAL PAYOFF - a tearful reckoning: kneeling apology, forgiveness, or too-late regret. Close on one quotable moral line."),
]

STYLE_VISUAL_RULES: Dict[str, str] = {
    "အသီးခေါင်း AI (Talking Fruit)": "Expressive, anthropomorphic fruits with clear facial emotion.",
    "တရားတော် ပုံပြ (Studio Ghibli)": "Studio Ghibli style, serene nature, traditional aesthetic, soft warm light.",
    "ကလေးများအတွက် 3D Animation": "Pixar-like Disney 3D animation style, cute and rounded.",
}

DURATIONS: Dict[str, Dict[str, Any]] = {
    "၁ မိနစ်ဝန်းကျင် (Scene ၆ မှ ၇ ခု)": {"scenes": "6 to 7", "avg_scenes": 6.5, "total_seconds": 60},
    "၂ မိနစ်ဝန်းကျင် (Scene ၁၀ မှ ၁၅ ခု)": {"scenes": "10 to 15", "avg_scenes": 12.5, "total_seconds": 120},
    "၃ မိနစ်ဝန်းကျင် (Scene ၂၀ မှ ၂၅ ခု)": {"scenes": "20 to 25", "avg_scenes": 22.5, "total_seconds": 180},
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

# (lower_bound_fraction, upper_bound_fraction, act description)
STORY_ACTS: List[Tuple[float, float, str]] = [
    (0.00, 0.10, "HOOK - open with something surprising, funny, or dramatic already in motion. No slow intro."),
    (0.10, 0.25, "SETUP - quickly establish who the characters are and their situation, shown through action, not exposition."),
    (0.25, 0.65, "RISING ACTION - escalate the problem or joke, add complications, raise the stakes step by step."),
    (0.65, 0.85, "CLIMAX - the peak conflict, biggest joke, or key turning point of the story."),
    (0.85, 1.01, "RESOLUTION - a satisfying, funny, or ironic ending that pays off the setup."),
]

CHARACTER_DETAIL_RULE = (
    "CHARACTER SHEET: for every character's flow_ai_ref_prompt, include at least five concrete visual "
    "anchors in one reusable English description: exact color(s), a distinguishing shape/texture, a "
    "signature accessory or marking, a typical facial expression or personality cue, and rough "
    "proportions. Vague phrases like 'a happy fruit character' are not acceptable."
)

MOTION_DETAIL_RULE = (
    "VIDEO MOTION: for every img_to_video_prompt_en, always specify (1) a precise camera movement "
    "(push-in, pull-out, pan left/right, tilt, orbit, handheld shake, or static), (2) exactly what the "
    "subject physically does during the shot, and (3) the pacing (slow/measured or quick/snappy). Avoid "
    "vague prompts like 'camera moves around'."
)

STYLE_BIBLE_RULE = (
    "STYLE BIBLE: also write a short one-paragraph 'style_bible' in English describing visual choices to "
    "reuse in EVERY scene - color palette, lighting mood, camera lens feel, and art medium texture. Keep "
    "it general and reusable, not scene-specific; it will be appended to every image prompt automatically."
)

# JSON schemas for Gemini structured output (uppercase type names match the
# API's Type enum and are more reliable than plain lowercase JSON-Schema).
FULL_SCRIPT_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "logline": {"type": "STRING"},
        "style_bible": {"type": "STRING"},
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
    "required": ["title", "logline", "style_bible", "character_sheet", "scenes"],
}

BATCH_SCENES_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
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
    "required": ["scenes"],
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
    page_title="AI Script Studio — Flow AI Prompt Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Noto+Sans+Myanmar:wght@400;500;600;700&display=swap');

:root {
    --bg: #0B0D12;
    --bg2: #0E1118;
    --surface: #131722;
    --card: #161B28;
    --card2: #1A2030;
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text: #F4F1EA;
    --muted: #9AA3B2;
    --faint: #6B7280;
    --gold: #F2B134;
    --gold2: #E8590C;
    --grad: linear-gradient(135deg, #F6C453 0%, #F2B134 45%, #E8590C 100%);
    --green: #34D399;
    --red: #F87171;
    --blue: #60A5FA;
    --violet: #A78BFA;
    --radius: 14px;
}

html, body, [class*="css"] { font-family: 'Noto Sans Myanmar', 'Sora', sans-serif; }
.stApp {
    background:
        radial-gradient(1100px 420px at 12% -8%, rgba(242, 177, 52, 0.07), transparent 60%),
        radial-gradient(900px 380px at 95% 0%, rgba(232, 89, 12, 0.05), transparent 60%),
        var(--bg);
    color: var(--text);
}

/* ---------- scrollbars ---------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #2A3040; border-radius: 8px; border: 2px solid var(--bg); }
::-webkit-scrollbar-thumb:hover { background: #3A4256; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stMarkdown { color: var(--text); }
.side-brand { display: flex; align-items: center; gap: 10px; padding: 4px 2px 2px 2px; }
.side-brand .logo {
    width: 38px; height: 38px; border-radius: 11px; background: var(--grad);
    display: flex; align-items: center; justify-content: center; font-size: 20px;
    box-shadow: 0 4px 18px rgba(242, 177, 52, 0.35);
}
.side-brand .name { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 16px; letter-spacing: 0.01em; }
.side-brand .sub { font-size: 11px; color: var(--muted); }
.side-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--gold); margin: 20px 0 8px 0;
}
.side-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 14px; margin-bottom: 6px;
}
.key-status { display: flex; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 10px; }
.key-status .dot { width: 9px; height: 9px; border-radius: 50%; }
.key-status.ok .dot { background: var(--green); box-shadow: 0 0 10px rgba(52, 211, 153, 0.8); }
.key-status.off .dot { background: var(--faint); }
.key-status.ok { color: var(--green); } .key-status.off { color: var(--muted); }
.hist-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; margin-bottom: 8px;
}
.hist-card .t { font-weight: 600; font-size: 13.5px; }
.hist-card .m { font-size: 11.5px; color: var(--muted); }

/* ---------- hero ---------- */
.hero {
    background: linear-gradient(135deg, rgba(242,177,52,0.10) 0%, rgba(232,89,12,0.06) 55%, transparent 100%),
                var(--surface);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 28px 30px; margin-bottom: 22px; position: relative; overflow: hidden;
}
.hero::after {
    content: "🎬"; position: absolute; right: 26px; top: 50%; transform: translateY(-50%) rotate(-8deg);
    font-size: 92px; opacity: 0.10; pointer-events: none;
}
.hero-kicker {
    display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.18em;
    color: var(--gold); border: 1px solid rgba(242,177,52,0.4); background: rgba(242,177,52,0.08);
    padding: 5px 12px; border-radius: 20px; margin-bottom: 12px;
}
.hero-title { font-family: 'Sora', sans-serif; font-size: 34px; font-weight: 800; line-height: 1.15; margin-bottom: 6px; }
.hero-title .grad {
    background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--muted); font-size: 14.5px; max-width: 640px; }
.stat-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
.stat-chip {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 14px; font-size: 13px; color: var(--text);
    display: flex; align-items: center; gap: 7px;
}
.stat-chip b { color: var(--gold); }

/* ---------- generic cards & steps ---------- */
.panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 18px; padding: 24px 26px; margin-bottom: 22px;
}
.panel-head { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.panel-head .step-num {
    width: 34px; height: 34px; border-radius: 10px; background: var(--grad); color: #1A1206;
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 16px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(242, 177, 52, 0.3);
}
.panel-title { font-size: 18px; font-weight: 700; }
.panel-sub { color: var(--muted); font-size: 13px; margin: 2px 0 14px 46px; }
.field-label {
    font-size: 12.5px; font-weight: 700; color: var(--muted);
    letter-spacing: 0.06em; text-transform: uppercase; margin: 16px 0 8px 0;
}
.field-label:first-child { margin-top: 4px; }
.field-label .n {
    display: inline-flex; width: 20px; height: 20px; border-radius: 6px; align-items: center; justify-content: center;
    background: rgba(242,177,52,0.14); color: var(--gold); font-size: 11px; margin-right: 7px;
}

/* ---------- empty state ---------- */
.empty-state {
    text-align: center; padding: 46px 20px; border: 1.5px dashed var(--border-strong);
    border-radius: 18px; background: rgba(255,255,255,0.015); margin-bottom: 22px;
}
.empty-state .big { font-size: 54px; margin-bottom: 10px; }
.empty-state h3 { font-size: 19px; margin-bottom: 6px; }
.empty-state p { color: var(--muted); font-size: 13.5px; max-width: 520px; margin: 0 auto 18px auto; }
.steps3 { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-top: 6px; }
.step3 {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; width: 200px; text-align: left;
}
.step3 .e { font-size: 22px; } .step3 .t { font-weight: 700; font-size: 13.5px; margin: 6px 0 2px 0; }
.step3 .d { font-size: 12px; color: var(--muted); }

/* ---------- buttons ---------- */
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
    border-radius: 11px !important; font-weight: 600 !important;
    border: 1px solid var(--border-strong) !important;
    background: var(--card2) !important; color: var(--text) !important;
    transition: transform 0.08s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover {
    border-color: rgba(242,177,52,0.55) !important; box-shadow: 0 4px 16px rgba(242,177,52,0.12) !important;
}
div[data-testid="stButton"] > button:active, div[data-testid="stDownloadButton"] > button:active { transform: scale(0.98); }
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--grad) !important; border: none !important; color: #1A1206 !important;
    font-weight: 700 !important; font-size: 16px !important; padding: 12px 20px !important;
    box-shadow: 0 6px 24px rgba(232, 89, 12, 0.35) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 8px 30px rgba(232, 89, 12, 0.5) !important; filter: brightness(1.05);
}
div[data-testid="stLinkButton"] > a {
    border-radius: 11px !important; border: 1px solid var(--border-strong) !important;
    background: var(--card2) !important; color: var(--text) !important; font-weight: 600 !important;
}
.icon-btn div[data-testid="stButton"] > button { padding: 6px 10px !important; font-size: 13px !important; }

/* ---------- pills ---------- */
div[data-testid="stPills"] button {
    border-radius: 10px !important; border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.03) !important; color: var(--muted) !important;
    font-weight: 500 !important; transition: all 0.12s ease !important;
}
div[data-testid="stPills"] button:hover { border-color: var(--border-strong) !important; color: var(--text) !important; }
div[data-testid="stPills"] button[aria-pressed="true"] {
    background: rgba(242,177,52,0.14) !important; border-color: var(--gold) !important;
    color: var(--gold) !important; font-weight: 700 !important;
    box-shadow: 0 2px 12px rgba(242,177,52,0.15) !important;
}

/* ---------- inputs ---------- */
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    border-radius: 11px !important; color: var(--text) !important;
}
div[data-testid="stTextInput"] input:focus, div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(242,177,52,0.15) !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(242,177,52,0.15) !important;
}
input::placeholder, textarea::placeholder { color: var(--faint) !important; }

/* ---------- tabs ---------- */
div[data-testid="stTabs"] div[role="tablist"] { gap: 6px; border-bottom: 1px solid var(--border); }
div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 10px 10px 0 0 !important; color: var(--muted) !important;
    font-weight: 600 !important; padding: 10px 18px !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important;
    background: rgba(242,177,52,0.06) !important;
}

/* ---------- expanders ---------- */
div[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 12px !important;
    background: var(--surface) !important; overflow: hidden; }
div[data-testid="stExpander"] summary { font-weight: 600 !important; }
div[data-testid="stExpander"] summary:hover { color: var(--gold) !important; }

/* ---------- code blocks ---------- */
div[data-testid="stCode"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
div[data-testid="stCode"] pre { background: #0D1017 !important; }

/* ---------- status / alerts ---------- */
div[data-testid="stStatus"] { border: 1px solid var(--border) !important; border-radius: 12px !important;
    background: var(--surface) !important; }
div[data-testid="stAlert"] { border-radius: 12px !important; }

/* ---------- scene cards ---------- */
.scene-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px 20px; margin-bottom: 16px;
    transition: border-color 0.15s ease;
}
.scene-card:hover { border-color: var(--border-strong); }
.scene-top { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap; }
.scene-num {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 13px; color: #1A1206;
    background: var(--grad); border-radius: 8px; padding: 4px 12px; letter-spacing: 0.04em;
}
.act-chip { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; padding: 4px 11px;
    border-radius: 20px; border: 1px solid; }
.dialogue-box {
    background: linear-gradient(135deg, rgba(242,177,52,0.09), rgba(242,177,52,0.03));
    border: 1px solid rgba(242,177,52,0.25); border-left: 3px solid var(--gold);
    padding: 12px 14px; border-radius: 10px; margin-bottom: 14px; font-size: 14.5px; line-height: 1.65;
}
.prompt-panel-label { font-size: 12px; font-weight: 700; letter-spacing: 0.08em; color: var(--muted);
    margin-bottom: 6px; text-transform: uppercase; }
.scene-actions { display: flex; gap: 8px; margin-top: 12px; }

/* ---------- character cards ---------- */
.char-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 18px 20px; height: 100%;
}
.char-card .cname { font-size: 16px; font-weight: 700; margin-bottom: 8px; }
.char-card .cname .e { font-size: 22px; margin-right: 8px; }
.char-card .cdesc { font-size: 13.5px; color: var(--muted); line-height: 1.6; margin-bottom: 12px; }
.char-card .clabel { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: var(--gold);
    text-transform: uppercase; margin-bottom: 6px; }

/* ---------- project header ---------- */
.proj-head { margin-bottom: 4px; }
.proj-title { font-family: 'Sora', sans-serif; font-size: 24px; font-weight: 800; margin-bottom: 6px; }
.proj-logline { color: var(--muted); font-size: 14px; line-height: 1.6; margin-bottom: 14px; max-width: 760px; }

/* ---------- misc ---------- */
.hint-text { color: var(--muted); font-size: 12.5px; }
.divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
.footer { text-align: center; color: var(--faint); font-size: 12px; padding: 26px 0 10px 0; }
.footer b { color: var(--muted); }
.stCaption { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3. OPTIONAL BROWSER-LOCALSTORAGE LAYER (best-effort, never fatal)
# =====================================================================
# Requires `streamlit-local-storage` in requirements.txt. If it's missing
# or fails to load for any reason, every helper below silently no-ops so
# the rest of the app still works - it just won't remember anything
# across a refresh.

def _get_local_store():
    try:
        from streamlit_local_storage import LocalStorage
        return LocalStorage()
    except Exception:
        return None


_LOCAL_STORE = _get_local_store()


def ls_get(key: str) -> Optional[str]:
    """Read one value from this browser's localStorage (best-effort)."""
    if _LOCAL_STORE is None:
        return None
    try:
        return _LOCAL_STORE.getItem(key)
    except Exception:
        return None


def ls_set(key: str, value: str, widget_key: str) -> None:
    if _LOCAL_STORE is None:
        return
    try:
        _LOCAL_STORE.setItem(key, value, key=widget_key)
    except Exception:
        pass


def ls_delete(key: str, widget_key: str) -> None:
    if _LOCAL_STORE is None:
        return
    try:
        _LOCAL_STORE.deleteItem(key, key=widget_key)
    except Exception:
        pass


# =====================================================================
# 4. HELPER: JSON PARSING
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
# 5. HELPER: BATCH PLANNING + NARRATIVE ARC
# =====================================================================

def plan_batches(duration_meta: Dict[str, Any], max_per_batch: int = MAX_SCENES_PER_BATCH) -> List[Dict[str, int]]:
    """Split the target scene count into batches of at most `max_per_batch`
    scenes, so each single Gemini call stays short enough to keep every
    scene sharp instead of trailing off in a huge single response."""
    total_scenes = round(duration_meta["avg_scenes"])
    num_batches = max(1, math.ceil(total_scenes / max_per_batch))
    base, remainder = divmod(total_scenes, num_batches)
    sizes = [base + (1 if i < remainder else 0) for i in range(num_batches)]

    batches = []
    offset = 0
    for size in sizes:
        batches.append({"start": offset + 1, "count": size, "total": total_scenes})
        offset += size
    return batches


def act_for_position(fraction: float, melodrama: bool = False) -> str:
    acts = MELODRAMA_ACTS if melodrama else STORY_ACTS
    for lo, hi, desc in acts:
        if lo <= fraction < hi:
            return desc
    return acts[-1][2]


def narrative_arc_instruction(
    batch: Dict[str, int], series_type: str, melodrama: bool = False
) -> str:
    start, count, total = batch["start"], batch["count"], batch["total"]
    end = start + count - 1
    acts_seen: List[str] = []
    for frac in (
        (start - 1) / total,
        (start - 1 + count / 2) / total,
        min(end / total, 0.999),
    ):
        act = act_for_position(frac, melodrama)
        if act not in acts_seen:
            acts_seen.append(act)

    note = f"NARRATIVE ARC: this batch is scenes {start}-{end} of {total} total. " + " Then move into: ".join(acts_seen)

    is_final_batch = end >= total
    if not is_final_batch:
        note += " End the LAST scene of this batch on a small hook or unresolved beat so it flows naturally into the next scenes."
    elif "အစပျိုး" in series_type:
        if melodrama:
            note += (" Since this is Part 1 of a series, end the FINAL scene on a devastating cliffhanger "
                     "that sets up Part 2 - a secret overheard, a door slammed, a phone ringing, a face "
                     "turning pale. Do NOT resolve the story.")
        else:
            note += " Since this is Part 1 of a series, end the FINAL scene on a cliffhanger that sets up Part 2 rather than fully resolving the story."
    elif melodrama:
        note += " End the FINAL scene with the tearful moral payoff described above - never a joke."
    else:
        note += " End the FINAL scene with a satisfying, funny punchline or resolution."
    return note


# =====================================================================
# 6. HELPER: PROMPT BUILDERS
# =====================================================================

def build_system_prompt(
    style_key: str,
    batch: Dict[str, int],
    series_type: str,
    satire_intensity: Optional[str] = None,
    hook_required: bool = False,
    mode: str = "full",
    melodrama_archetype: Optional[str] = None,
) -> str:
    is_melodrama = bool(melodrama_archetype)
    if is_melodrama:
        intro = (
            "You are an expert AI scriptwriter and prompt engineer specialized for Flow AI "
            "(image-to-video) production pipelines, creating Burmese family melodrama mini-movies "
            "for a Myanmar audience - tearful, moral, unforgettable, in the spirit of the top "
            "Myanmar TikTok talking-fruit drama channels."
        )
    else:
        intro = (
            "You are an expert AI scriptwriter and prompt engineer specialized for Flow AI "
            "(image-to-video) production pipelines, creating short entertainment videos for a "
            "Myanmar audience that should feel as engaging as top international 'talking fruit' "
            "or mascot-comedy channels."
        )
    parts = [
        intro,
        "",
        f"VISUAL STYLE: {STYLE_VISUAL_RULES.get(style_key, '')}",
        "",
        "LANGUAGE: Write all dialogue/voiceover (dialogue_myanmar) in natural, spoken Myanmar. "
        "Write all image and video-motion prompts (image_prompt_en, img_to_video_prompt_en) in "
        "detailed, professional English optimized for an AI image/video generator.",
        "",
        CHARACTER_DETAIL_RULE,
        "",
        MOTION_DETAIL_RULE,
    ]

    if is_melodrama:
        parts.append("")
        parts.append(MELODRAMA_RULE)
        parts.append("")
        parts.append(MELODRAMA_VISUAL_RULE)
        parts.append("")
        parts.append(
            f"STORY ARCHETYPE for this script: {melodrama_archetype} - "
            f"{MELODRAMA_ARCHETYPES.get(melodrama_archetype, '')}"
        )

    if mode in ("full", "continuation"):
        parts.append("")
        parts.append(STYLE_BIBLE_RULE)

    if mode != "single_scene":
        parts.append("")
        parts.append(narrative_arc_instruction(batch, series_type, melodrama=is_melodrama))

    if satire_intensity:
        parts.append("")
        parts.append(f"TONE: {INTENSITY_RULES.get(satire_intensity, '')}")
        parts.append("")
        parts.append(f"SAFETY: {SATIRE_SAFETY_RULE}")

    if hook_required and batch["start"] == 1 and not is_melodrama:
        parts.append("")
        parts.append(
            "HOOK: This is for short-form vertical platforms (TikTok/Reels/Shorts). The very "
            "first scene must open with a strong visual or verbal hook within the first 3 "
            "seconds - a surprising image, a bold question, or a dramatic action."
        )

    if mode == "continuation":
        parts.append("")
        parts.append(
            "CONTINUITY: This is a continuation of an earlier part of the same series. Reuse "
            "the same characters (same names and appearances) unless the story naturally "
            "introduces a new one. Move the plot forward - do not repeat earlier events."
        )
    elif mode == "batch_continue":
        parts.append("")
        parts.append(
            "You are continuing an already-started script. Do not repeat the title, logline, "
            "style bible or character sheet - only return the new scenes for this batch, "
            "picking up exactly where the story left off."
        )
    elif mode == "single_scene":
        parts.append("")
        parts.append(
            "You are rewriting ONE specific scene inside an existing script. Keep it "
            "consistent with the surrounding scenes, characters and tone given in the context. "
            "Return only that one scene, with the same scene_number."
        )

    return "\n".join(parts)


def dialogue_pacing_line(duration_meta: Dict[str, Any]) -> str:
    seconds_per_scene = duration_meta["total_seconds"] / duration_meta["avg_scenes"]
    max_words = max(4, round(seconds_per_scene * 2.2))
    return (
        f"Each scene covers about {round(seconds_per_scene)} seconds of video - keep "
        f"dialogue_myanmar to roughly {max_words} Myanmar words or fewer so it can be spoken "
        f"naturally in that time."
    )


def build_user_instruction(
    style: str,
    genre: str,
    duration_meta: Dict[str, Any],
    series_type: str,
    idea: str,
    batch: Dict[str, int],
    trending_topic: str = "",
    melodrama_archetype: Optional[str] = None,
) -> str:
    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"Create exactly {batch['count']} scenes, numbered sequentially starting at {batch['start']}.",
        dialogue_pacing_line(duration_meta),
        f"Series Structure: {series_type}",
    ]
    if melodrama_archetype:
        lines.append(
            f"Melodrama archetype (follow it faithfully): {melodrama_archetype} - "
            f"{MELODRAMA_ARCHETYPES.get(melodrama_archetype, '')}"
        )
    if trending_topic.strip():
        lines.append(
            f"Real-world situation to satirize (use only as a generic backdrop, do not name "
            f"real people): {trending_topic.strip()}"
        )
    lines.append(f"Topic/Idea: {idea.strip() if idea.strip() else 'Creative natural storyline'}")
    return "\n".join(lines)


def build_batch_continue_instruction(
    data: Dict[str, Any], style: str, genre: str, duration_meta: Dict[str, Any], batch: Dict[str, int],
    melodrama_archetype: Optional[str] = None,
) -> str:
    prior_scenes = data.get("scenes", [])
    last_two = prior_scenes[-2:] if len(prior_scenes) >= 2 else prior_scenes
    context = "\n".join(
        f"Scene {s.get('scene_number')}: {s.get('dialogue_myanmar', '')}" for s in last_two
    )
    char_names = ", ".join(c.get("character_name", "") for c in data.get("character_sheet", []))
    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"Story title: {data.get('title', '')}",
        f"Logline: {data.get('logline', '')}",
        f"Characters to keep consistent: {char_names}",
        f"Most recent scenes so far, for continuity:\n{context}",
        f"Now write exactly {batch['count']} NEW scenes, numbered sequentially starting at {batch['start']}.",
        dialogue_pacing_line(duration_meta),
    ]
    if melodrama_archetype:
        lines.append(
            f"Keep following the melodrama archetype: {melodrama_archetype}. "
            f"Do not turn it into comedy."
        )
    return "\n".join(lines)


def build_continuation_first_batch_instruction(
    previous_data: Dict[str, Any], style: str, genre: str, duration_meta: Dict[str, Any],
    batch: Dict[str, int], idea: str, melodrama_archetype: Optional[str] = None,
) -> str:
    char_names = ", ".join(c.get("character_name", "") for c in previous_data.get("character_sheet", []))
    lines = [
        f"Style: {style}",
        f"Genre: {genre}",
        f"This is the NEXT part of an ongoing series titled '{previous_data.get('title', '')}'.",
        f"Previous logline: {previous_data.get('logline', '')}",
        f"Existing characters to reuse: {char_names or 'none yet'}",
        f"Create exactly {batch['count']} new scenes for this part, numbered sequentially "
        f"starting at {batch['start']}.",
        dialogue_pacing_line(duration_meta),
    ]
    if melodrama_archetype:
        lines.append(
            f"Keep following the melodrama archetype: {melodrama_archetype}. "
            f"Do not turn it into comedy."
        )
    if idea.strip():
        lines.append(f"Direction for this part: {idea.strip()}")
    return "\n".join(lines)


def build_single_scene_instruction(data: Dict[str, Any], scene_number: int, style: str, genre: str) -> str:
    scenes = data.get("scenes", [])
    target = next((s for s in scenes if s.get("scene_number") == scene_number), {})
    prev_s = next((s for s in scenes if s.get("scene_number") == scene_number - 1), None)
    next_s = next((s for s in scenes if s.get("scene_number") == scene_number + 1), None)
    char_lines = "\n".join(
        f"- {c.get('character_name')}: {c.get('visual_description')}" for c in data.get("character_sheet", [])
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
# 7. HELPER: POST-PROCESSING (consistency enforced by code, not hope)
# =====================================================================

def build_character_clause(data: Dict[str, Any]) -> str:
    chars = data.get("character_sheet", [])
    parts = [
        f"{c.get('character_name', '')} - {c.get('flow_ai_ref_prompt', '')}"
        for c in chars if c.get("flow_ai_ref_prompt")
    ]
    return "; ".join(parts)


def apply_consistency(
    data: Dict[str, Any], character_clause: str, style_bible: str, aspect_text: str,
    scene_number: Optional[int] = None,
) -> Dict[str, Any]:
    extras = []
    if character_clause:
        extras.append(f"Character consistency: {character_clause}")
    if style_bible:
        extras.append(f"Style bible: {style_bible}")
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


# =====================================================================
# 8. HELPER: GEMINI CALLS WITH FALLBACK + RETRY
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


def _short_error(e: BaseException, limit: int = 160) -> str:
    msg = str(getattr(e, "message", "") or e).replace("\n", " ").strip()
    return (msg[:limit] + "…") if len(msg) > limit else msg


def generate_with_fallback(
    status, api_key: str, model_order: List[str], schema: Dict[str, Any],
    system_prompt: str, user_instruction: str,
) -> Tuple[str, Dict[str, Any]]:
    client = genai.Client(api_key=api_key)
    last_error: Optional[BaseException] = None
    attempt_log: List[str] = []

    def _fail(e: BaseException, note: str) -> None:
        # Attach the full attempt history to the exception so UI handlers can show it.
        e._attempt_log = list(attempt_log)  # type: ignore[attr-defined]
        status.update(label=note, state="error")

    for idx, model_name in enumerate(model_order):
        delays = PRIMARY_RETRY_DELAYS if idx == 0 else FALLBACK_RETRY_DELAYS
        max_attempts = len(delays) + 1

        for attempt in range(1, max_attempts + 1):
            status.update(label=f"🎬 {model_name} ဖြင့် ရေးနေသည်… (ကြိုးစားမှု {attempt}/{max_attempts})")
            try:
                response = _call_gemini(client, model_name, schema, system_prompt, user_instruction)
                data = parse_json_response(response.text)
                status.update(label=f"✅ {model_name} ဖြင့် အောင်မြင်ပါသည်", state="complete")
                return model_name, data

            except errors.APIError as e:
                code = getattr(e, "code", None)
                last_error = e
                attempt_log.append(f"{model_name} try {attempt}: APIError {code} - {_short_error(e)}")
                if code in (400, 401, 403):
                    _fail(e, "❌ API Key (သို့) တောင်းဆိုမှု ပြဿနာ")
                    raise
                if code and code >= 500 and attempt < max_attempts:
                    wait = delays[attempt - 1]
                    status.update(label=f"⚠️ {model_name}: Google server error {code} — {wait} စက္ကန့်စောင့်ပြီး ပြန်ကြိုးစားမည်…")
                    time.sleep(wait)
                    continue
                # This model's attempts are exhausted (or non-retriable code) -> move on.
                if idx < len(model_order) - 1:
                    reason = {404: "model ရှာမတွေ့ပါ", 429: "quota ပြည့်နေသည်"}.get(code, f"error {code}")
                    status.update(label=f"⚠️ {model_name}: {reason} — နောက် model ({model_order[idx + 1]}) ပြောင်းနေသည်…")
                break

            except ValueError as e:
                last_error = e
                attempt_log.append(f"{model_name} try {attempt}: parse error - {_short_error(e)}")
                if attempt < max_attempts:
                    status.update(label=f"⚠️ {model_name}: အဖြေဖတ်မရ — ပြန်ကြိုးစားနေသည်… ({attempt}/{max_attempts})")
                    continue
                break

    if last_error:
        _fail(last_error, "❌ Model အားလုံး ကြိုးစားပြီးပါပြီ၊ မအောင်မြင်ပါ")
        raise last_error
    err = RuntimeError("Model တစ်ခုမှ အလုပ်မလုပ်ပါ။")
    _fail(err, "❌ Model အားလုံး ကြိုးစားပြီးပါပြီ၊ မအောင်မြင်ပါ")
    raise err


def friendly_api_error(e: BaseException) -> str:
    code = getattr(e, "code", None)
    msg = str(getattr(e, "message", "") or e)
    low = msg.lower()

    if code == 404:
        return ("⚠️ Model အားလုံး ရှာမတွေ့ပါ။ MODEL_CHOICES ထဲက model နာမည်တွေကို "
                "https://ai.google.dev/gemini-api/docs/models မှာ စစ်ပြီး အသစ်ပြောင်းပါ။")
    if code in (401, 403) or (code == 400 and "api key" in low):
        return "🔑 API Key မှားနေပါသည်။ Google AI Studio မှ Key ကို ပြန်ကူးပြီး ထည့်ကြည့်ပါ။"
    if code == 429:
        return "⏳ Quota ပြည့်နေပါသည်။ ခဏစောင့်ပြီး ပြန်စမ်းပါ။"
    if code and code >= 500:
        return "🌐 Google server အားလုံး အလုပ်များနေပါသည်။ ၁-၂ မိနစ်နေမှ ပြန်စမ်းကြည့်ပါ။"
    return f"Error ({code}): {msg}"


def show_attempt_log(e: BaseException) -> None:
    """Renders the per-attempt failure history attached by generate_with_fallback."""
    log = getattr(e, "_attempt_log", [])
    if log:
        st.markdown("**📝 ကြိုးစားမှု မှတ်တမ်း (ဘာကြောင့် fail လဲဆိုတာ):**")
        st.code("\n".join(log), language="text")


def generate_full_script(
    status, api_key: str, model_order: List[str], style: str, genre: str,
    duration_meta: Dict[str, Any], series_type: str, idea: str, trending_topic: str,
    satire_intensity: Optional[str], hook_required: bool,
    melodrama_archetype: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Runs the (possibly multi-batch) generation for a brand-new script and
    returns (model_used, data)."""
    batches = plan_batches(duration_meta)
    used_models: List[str] = []

    # --- first batch: title, logline, style bible, characters + its scenes ---
    sys0 = build_system_prompt(style, batches[0], series_type, satire_intensity, hook_required,
                               mode="full", melodrama_archetype=melodrama_archetype)
    instr0 = build_user_instruction(style, genre, duration_meta, series_type, idea, batches[0],
                                    trending_topic, melodrama_archetype=melodrama_archetype)
    model_used, data = generate_with_fallback(status, api_key, model_order, FULL_SCRIPT_SCHEMA, sys0, instr0)
    used_models.append(model_used)

    # --- remaining batches: scenes only, continuing the same story ---
    for batch in batches[1:]:
        sys_b = build_system_prompt(style, batch, series_type, satire_intensity, hook_required,
                                    mode="batch_continue", melodrama_archetype=melodrama_archetype)
        instr_b = build_batch_continue_instruction(data, style, genre, duration_meta, batch,
                                                   melodrama_archetype=melodrama_archetype)
        model_b, batch_data = generate_with_fallback(status, api_key, model_order, BATCH_SCENES_SCHEMA, sys_b, instr_b)
        used_models.append(model_b)
        data.setdefault("scenes", []).extend(batch_data.get("scenes", []))

    label = used_models[0] if len(set(used_models)) == 1 else " → ".join(dict.fromkeys(used_models))
    return label, data


def generate_continuation(
    status, api_key: str, model_order: List[str], previous_data: Dict[str, Any],
    style: str, genre: str, duration_meta: Dict[str, Any], series_type: str,
    idea: str, satire_intensity: Optional[str], hook_required: bool,
    melodrama_archetype: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    batches = plan_batches(duration_meta)
    used_models: List[str] = []

    sys0 = build_system_prompt(style, batches[0], series_type, satire_intensity, hook_required,
                               mode="continuation", melodrama_archetype=melodrama_archetype)
    instr0 = build_continuation_first_batch_instruction(previous_data, style, genre, duration_meta,
                                                        batches[0], idea,
                                                        melodrama_archetype=melodrama_archetype)
    model_used, data = generate_with_fallback(status, api_key, model_order, FULL_SCRIPT_SCHEMA, sys0, instr0)
    used_models.append(model_used)

    for batch in batches[1:]:
        sys_b = build_system_prompt(style, batch, series_type, satire_intensity, hook_required,
                                    mode="batch_continue", melodrama_archetype=melodrama_archetype)
        instr_b = build_batch_continue_instruction(data, style, genre, duration_meta, batch,
                                                   melodrama_archetype=melodrama_archetype)
        model_b, batch_data = generate_with_fallback(status, api_key, model_order, BATCH_SCENES_SCHEMA, sys_b, instr_b)
        used_models.append(model_b)
        data.setdefault("scenes", []).extend(batch_data.get("scenes", []))

    label = used_models[0] if len(set(used_models)) == 1 else " → ".join(dict.fromkeys(used_models))
    return label, data


# =====================================================================
# 9. SESSION STATE + LOCALSTORAGE RESTORE
# =====================================================================

if "parts" not in st.session_state:
    st.session_state.parts: List[Dict[str, Any]] = []
if "active_part" not in st.session_state:
    st.session_state.active_part = 0
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []

# Fresh page loads run the script once before the LocalStorage frontend has
# posted this browser's stored values back, so the first restore pass sees
# nothing. Do one bounded rerun so the second pass picks them up. The flag
# guarantees this can never loop, even if the component misbehaves.
if _LOCAL_STORE is not None and not st.session_state.get("_ls_boot_rerun_done"):
    st.session_state["_ls_boot_rerun_done"] = True
    st.rerun()

# Restore the API key, current project and history from this browser's
# localStorage. These keep retrying every rerun while still empty, and stop
# naturally once populated (either restored, or the user generates something
# new).
if not st.session_state.get("api_key_field"):
    _restored_key = ls_get("flow_ai_api_key")
    if _restored_key:
        st.session_state["api_key_field"] = _restored_key

if not st.session_state.parts:
    _restored_project = ls_get("flow_ai_project")
    if _restored_project:
        try:
            loaded = json.loads(_restored_project)
            if loaded:
                st.session_state.parts = loaded
                st.session_state.active_part = len(loaded) - 1
        except Exception:
            pass

if not st.session_state.history:
    _restored_history = ls_get("flow_ai_history")
    if _restored_history:
        try:
            st.session_state.history = json.loads(_restored_history)
        except Exception:
            pass


def current_part() -> Optional[Dict[str, Any]]:
    if not st.session_state.parts:
        return None
    idx = min(st.session_state.active_part, len(st.session_state.parts) - 1)
    return st.session_state.parts[idx]


def persist_project() -> None:
    ls_set("flow_ai_project", json.dumps(st.session_state.parts, ensure_ascii=False), "set_project")


def persist_history() -> None:
    ls_set("flow_ai_history", json.dumps(st.session_state.history, ensure_ascii=False), "set_history")


def archive_current_project_to_history() -> None:
    if not st.session_state.parts:
        return
    last_data = st.session_state.parts[-1]["data"]
    entry = {
        "title": last_data.get("title", "Untitled"),
        "style": st.session_state.parts[-1]["style"],
        "genre": st.session_state.parts[-1]["genre"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "parts": st.session_state.parts,
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]
    persist_history()


# =====================================================================
# 10. SIDEBAR (brand + API key + model + history) & HERO HEADER
# =====================================================================

with st.sidebar:
    st.markdown(
        '<div class="side-brand"><div class="logo">🎬</div>'
        '<div><div class="name">Script Studio</div>'
        '<div class="sub">AI Video Script Factory</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">🔑 API Key</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    has_key = bool((st.session_state.get("api_key_field") or "").strip())
    st.markdown(
        f'<div class="key-status {"ok" if has_key else "off"}">'
        f'<span class="dot"></span>{"ချိတ်ဆက်ထားပြီး ✓" if has_key else "Key မထည့်ရသေးပါ"}</div>',
        unsafe_allow_html=True,
    )
    api_key_input = st.text_input(
        "Google AI Studio API Key", type="password", placeholder="AIzaSy...",
        key="api_key_field", label_visibility="collapsed",
    )
    remember_key = st.checkbox("🔒 ဒီ browser မှာပဲ မှတ်ထားမည်", value=True,
                               help="Key ကို ဒီ browser ရဲ့ local storage ထဲမှာပဲ သိမ်းထားပြီး server ကို ဘယ်တော့မှ ပို့မည်မဟုတ်ပါ။")
    if remember_key and api_key_input and api_key_input != st.session_state.get("_last_saved_key"):
        ls_set("flow_ai_api_key", api_key_input, "set_api_key")
        st.session_state["_last_saved_key"] = api_key_input
    elif not remember_key and st.session_state.get("_last_saved_key"):
        ls_delete("flow_ai_api_key", "clear_api_key")
        st.session_state["_last_saved_key"] = ""
    st.link_button("🔗 Google AI Studio မှာ Key ယူရန်", "https://aistudio.google.com/apikey", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-label">🤖 AI Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-card">', unsafe_allow_html=True)
    selected_model = st.selectbox("Gemini Model (ဦးစားပေး)", MODEL_CHOICES, index=0, label_visibility="collapsed")
    custom_model = st.text_input("Custom model", placeholder="မလိုရင် ဗလာထားပါ — ဥပမာ gemini-3.8-flash",
                                 label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown(f'<div class="side-label">🕘 History ({len(st.session_state.history)})</div>', unsafe_allow_html=True)
        for i, entry in enumerate(st.session_state.history):
            st.markdown(
                f'<div class="hist-card"><div class="t">{html.escape(entry.get("title", ""))}</div>'
                f'<div class="m">{entry.get("style", "")} · {entry.get("genre", "")} · {entry.get("timestamp", "")}</div></div>',
                unsafe_allow_html=True,
            )
            hcol1, hcol2 = st.columns([1, 1])
            with hcol1:
                if st.button("🔁 ဖွင့်မည်", key=f"restore_hist_{i}", width="stretch"):
                    archive_current_project_to_history()
                    st.session_state.parts = entry["parts"]
                    st.session_state.active_part = len(entry["parts"]) - 1
                    persist_project()
                    st.rerun()
            with hcol2:
                if st.button("🗑️", key=f"delete_hist_{i}", width="stretch"):
                    st.session_state.history.pop(i)
                    persist_history()
                    st.rerun()

    st.markdown(
        '<div class="hint-text" style="margin-top:18px;">💡 Tip: Scene တိုင်းရဲ့ prompt တွေမှာ '
        'copy ခလုတ်ပါပြီးသားမို့ Flow AI ထဲ တိုက်ရိုက်ကူးထည့်ရုံပါပဲ။</div>',
        unsafe_allow_html=True,
    )


# ---- Hero header (main area) ----
parts_preview = st.session_state.parts
if parts_preview:
    _pp = current_part()
    _pd = _pp["data"]
    _n_scenes = len(_pd.get("scenes", []))
    stats_html = (
        '<div class="stat-row">'
        f'<div class="stat-chip">🎬 <b>{_n_scenes}</b> scenes</div>'
        f'<div class="stat-chip">📦 <b>{len(parts_preview)}</b> parts</div>'
        f'<div class="stat-chip">🤖 {_pp.get("model", "")}</div>'
        f'<div class="stat-chip">🖥️ {_pp.get("aspect_label", "")}</div>'
        "</div>"
    )
else:
    stats_html = ""

st.markdown(
    '<div class="hero">'
    '<div class="hero-kicker">🎬 AI VIDEO STUDIO</div>'
    '<div class="hero-title">Script <span class="grad">Studio</span></div>'
    '<div class="hero-sub">TikTok / YouTube အတွက် ဇာတ်လမ်း၊ ဇာတ်ကောင်စာရင်းနဲ့ Flow AI image + motion prompts တွေကို '
    "တစ်နေရာတည်းမှာ professional-grade ထုတ်ပေးတဲ့စနစ်။</div>"
    f"{stats_html}</div>",
    unsafe_allow_html=True,
)


# =====================================================================
# 11. CREATE PANEL — ဇာတ်လမ်းအသစ် ဖန်တီးရန်
# =====================================================================

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="panel-head"><div class="step-num">✨</div>'
    '<div class="panel-title">ဇာတ်လမ်းအသစ် ဖန်တီးရန်</div></div>'
    '<div class="panel-sub">အောက်က အချက်တွေရွေးလိုက်ရုံနဲ့ AI က ဇာတ်လမ်း + ဇာတ်ကောင် + Flow AI prompts အပြည့်အစုံ ရေးပေးပါလိမ့်မယ်။</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="field-label"><span class="n">၁</span>ဗီဒီယို ပုံစံ (Style)</div>', unsafe_allow_html=True)
selected_style = st.pills("Style", list(CATEGORIES.keys()), default=list(CATEGORIES.keys())[0],
                           label_visibility="collapsed")
selected_style = selected_style or list(CATEGORIES.keys())[0]

st.markdown('<div class="field-label"><span class="n">၂</span>ဇာတ်လမ်း အမျိုးအစား (Genre)</div>', unsafe_allow_html=True)
genre_options = CATEGORIES[selected_style]
selected_genre = st.pills("Genre", genre_options, default=genre_options[0],
                           key=f"genre_{selected_style}", label_visibility="collapsed")
selected_genre = selected_genre or genre_options[0]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="field-label"><span class="n">၃</span>ကြာချိန်</div>', unsafe_allow_html=True)
    video_duration = st.pills("Duration", list(DURATIONS.keys()), default=list(DURATIONS.keys())[0],
                               label_visibility="collapsed")
    video_duration = video_duration or list(DURATIONS.keys())[0]
with col_b:
    st.markdown('<div class="field-label"><span class="n">၄</span>Format</div>', unsafe_allow_html=True)
    aspect_label = st.pills("Aspect", list(ASPECT_RATIOS.keys()), default=list(ASPECT_RATIOS.keys())[0],
                             label_visibility="collapsed")
    aspect_label = aspect_label or list(ASPECT_RATIOS.keys())[0]

st.markdown('<div class="field-label"><span class="n">၅</span>ဇာတ်လမ်း ဖွဲ့စည်းပုံ</div>', unsafe_allow_html=True)
series_type = st.pills("Series", SERIES_OPTIONS, default=SERIES_OPTIONS[0], label_visibility="collapsed")
series_type = series_type or SERIES_OPTIONS[0]

is_satire = selected_genre in SATIRE_GENRES
is_melodrama = selected_genre == MELODRAMA_GENRE
trending_topic = ""
satire_intensity = None
melodrama_archetype = None
if is_melodrama:
    st.markdown('<div class="field-label">🎭 ဒရမ်မာ ပုံစံ (Archetype)</div>', unsafe_allow_html=True)
    melodrama_archetype = st.pills(
        "Melodrama archetype", list(MELODRAMA_ARCHETYPES.keys()),
        default=list(MELODRAMA_ARCHETYPES.keys())[0],
        key="melodrama_archetype", label_visibility="collapsed")
    melodrama_archetype = melodrama_archetype or list(MELODRAMA_ARCHETYPES.keys())[0]
    st.caption("💡 ပေါက်တဲ့ ဒရမ်မာ video တွေရဲ့ ဇာတ်လမ်းပုံစံတွေပါ — တစ်ခုရွေးလိုက်ရင် AI က အဲဒီအတိုင်း မျက်ရည်ကျစရာ ဇာတ်လမ်း ရေးပေးမယ်။")
if is_satire:
    st.markdown('<div class="field-label">🗞️ ယနေ့/လက်ရှိ အခြေအနေ <span class="hint-text">(optional)</span></div>', unsafe_allow_html=True)
    trending_topic = st.text_area(
        "Trending topic",
        placeholder="ဥပမာ - ဈေးကွက်ထဲ ဆီစျေးမြင့်တက်နေမှု၊ လျှပ်စစ်မီးပြတ်တောက်မှု၊ ဘတ်စ်ကားတန်းစီနေရမှု...",
        label_visibility="collapsed", height=80,
    )
    st.caption("⚠️ ပုဂ္ဂိုလ်ရေး နာမည် မထည့်ဘဲ 'အခြေအနေ' ကိုသာ ဖော်ပြပါ - AI ကလည်း အမည်ဖော်တာမျိုး ရေးမည်မဟုတ်ပါ။")
    satire_intensity = st.select_slider("သရော်အား", options=INTENSITY_OPTIONS, value=INTENSITY_OPTIONS[1])

st.markdown('<div class="field-label"><span class="n">၆</span>ထည့်သွင်းလိုသော အကြောင်းအရာ <span class="hint-text">(optional)</span></div>', unsafe_allow_html=True)
custom_idea = st.text_area(
    "Idea", placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်...",
    label_visibility="collapsed",
)

_est = DURATIONS[video_duration].get("avg_scenes", 0)
st.caption(f"📊 ခန့်မှန်း scene <b>{_est:g}</b> ခု ထုတ်ပေးမည် · ဇာတ်ကောင် စာရင်း + Flow AI prompts အပြည့်အစုံ ပါဝင်မည်။")
generate_clicked = st.button("🚀 Script & Prompts ဖန်တီးမည်", type="primary", width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

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

        with st.status("Gemini model ကို ချိတ်ဆက်နေသည်…", expanded=True) as status:
            try:
                used_model, data = generate_full_script(
                    status, api_key, model_order, selected_style, selected_genre,
                    duration_meta, series_type, custom_idea, trending_topic,
                    satire_intensity if is_satire else None, hook_required,
                    melodrama_archetype=melodrama_archetype,
                )
                character_clause = build_character_clause(data)
                style_bible = data.get("style_bible", "")
                data = apply_consistency(data, character_clause, style_bible, aspect_meta["prompt_text"])

                archive_current_project_to_history()

                new_part = {
                    "data": data, "model": used_model, "style": selected_style, "genre": selected_genre,
                    "duration_label": video_duration, "aspect_label": aspect_label,
                    "series_type": series_type,
                    "character_clause": character_clause, "style_bible": style_bible,
                    "is_satire": is_satire, "satire_intensity": satire_intensity,
                    "melodrama_archetype": melodrama_archetype,
                }
                st.session_state.parts = [new_part]
                st.session_state.active_part = 0
                persist_project()
                st.toast("✅ Script ဖန်တီးပြီးပါပြီ! အောက်မှာ ရလဒ်တွေ ကြည့်နိုင်ပါပြီ။")

            except errors.APIError as e:
                st.error(friendly_api_error(e))
                with st.expander("Technical details"):
                    st.code(str(e), language="text")
                    show_attempt_log(e)
            except ValueError as e:
                st.error("⚠️ AI ပြန်လာတဲ့ အဖြေကို ဖတ်မရပါ။ ထပ်နှိပ်ကြည့်ပါ။")
                with st.expander("Technical details"):
                    show_attempt_log(e)
            except Exception as e:
                st.error(f"မမျှော်လင့်သော Error: {e}")


# =====================================================================
# 12. RESULTS
# =====================================================================

def _act_chip(fraction: float, melodrama: bool = False) -> str:
    """Scene card အတွက် act label + အရောင် chip HTML ထုတ်ပေးသည်။"""
    name = act_for_position(fraction, melodrama).split(" - ")[0].strip()
    color = "#F2B134"
    if "HOOK" in name:
        color = "#F2B134"
    elif "SETUP" in name:
        color = "#60A5FA"
    elif "RISING" in name:
        color = "#A78BFA"
    elif "CLIMAX" in name or "TWIST" in name or "REVEAL" in name:
        color = "#F87171"
    elif "RESOLUTION" in name or "PAYOFF" in name or "MORAL" in name:
        color = "#34D399"
    return (f'<span class="act-chip" style="color:{color};border-color:{color}66;'
            f'background:{color}14;">{html.escape(name)}</span>')


parts = st.session_state.parts
if not parts:
    st.markdown(
        '<div class="empty-state"><div class="big">🎬</div>'
        "<h3>ဇာတ်လမ်းတစ်ပုဒ် ဖန်တီးဖို့ အသင့်ဖြစ်နေပါပြီ</h3>"
        "<p>အပေါ်က အဆင့်တွေအတိုင်း ရွေးချယ်ပြီး <b>Script & Prompts ဖန်တီးမည်</b> ကို နှိပ်လိုက်ရုံပါပဲ — "
        "AI က ဇာတ်လမ်း၊ ဇာတ်ကောင်စာရင်းနဲ့ Flow AI prompts တွေ အလိုအလျောက် ရေးပေးပါလိမ့်မယ်။</p>"
        '<div class="steps3">'
        '<div class="step3"><div class="e">🔑</div><div class="t">၁။ Key ထည့်ပါ</div>'
        '<div class="d">ဘယ်ဘက် sidebar မှာ Google AI Studio API Key ထည့်ပါ</div></div>'
        '<div class="step3"><div class="e">✨</div><div class="t">၂။ ရွေးပြီး ဖန်တီးပါ</div>'
        '<div class="d">Style · Genre · ကြာချိန် ရွေးပြီး ခလုတ်နှိပ်ပါ</div></div>'
        '<div class="step3"><div class="e">🎥</div><div class="t">၃။ Copy ကူးပြီး သုံးပါ</div>'
        '<div class="d">Prompt တွေကို copy ခလုတ်နဲ့ ကူးပြီး Flow AI မှာ ဆက်လုပ်ပါ</div></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
else:
    if len(parts) > 1:
        part_labels = [f"Part {i + 1}" for i in range(len(parts))]
        st.markdown('<div class="field-label">အပိုင်း ရွေးရန်</div>', unsafe_allow_html=True)
        chosen = st.pills("Part ရွေးရန်", part_labels, default=part_labels[st.session_state.active_part],
                          label_visibility="collapsed")
        if chosen:
            st.session_state.active_part = part_labels.index(chosen)

    part = current_part()
    data = part["data"]
    aspect_meta = ASPECT_RATIOS[part["aspect_label"]]
    is_melo_part = part.get("genre") == MELODRAMA_GENRE
    n_scenes = len(data.get("scenes", []))

    st.markdown(
        '<div class="proj-head">'
        f'<div class="proj-title">📌 {html.escape(str(data.get("title", "Video Script")))}</div>'
        f'<div class="proj-logline">{html.escape(str(data.get("logline", "")))}</div>'
        "</div>"
        '<div class="stat-row">'
        f'<div class="stat-chip">🤖 {html.escape(str(part.get("model", "")))}</div>'
        f'<div class="stat-chip">🎨 {html.escape(str(part.get("style", "")))}</div>'
        f'<div class="stat-chip">🎭 {html.escape(str(part.get("genre", "")))}</div>'
        f'<div class="stat-chip">🖥️ {html.escape(str(part.get("aspect_label", "")))}</div>'
        f'<div class="stat-chip">🎬 <b>{n_scenes}</b>&nbsp;scenes</div>'
        "</div><div style='height:16px'></div>",
        unsafe_allow_html=True,
    )

    tab_scenes, tab_chars, tab_export = st.tabs(
        [f"🎬 Scenes ({n_scenes})",
         f"👤 ဇာတ်ကောင်များ ({len(data.get('character_sheet', []))})",
         "💾 Export & ဆက်ရန်"]
    )

    # ---- Characters tab ----
    _FRUIT_EMOJI = ["🍎", "🍊", "🍋", "🍇", "🍉", "🥭", "🍍", "🥥", "🍌", "🍑", "🍒", "🥝"]
    with tab_chars:
        if data.get("style_bible"):
            with st.expander("🎨 Style Bible — scene အားလုံးမှာ အလိုအလျောက် ပါဝင်ပြီးသား"):
                st.code(data["style_bible"], language="text")
        chars = data.get("character_sheet", [])
        if not chars:
            st.caption("ဇာတ်ကောင် စာရင်း မရှိပါ။")
        else:
            for _ci in range(0, len(chars), 2):
                _cols = st.columns(2)
                for _k, _col in enumerate(_cols):
                    if _ci + _k >= len(chars):
                        break
                    char = chars[_ci + _k]
                    _emoji = _FRUIT_EMOJI[(_ci + _k) % len(_FRUIT_EMOJI)]
                    with _col:
                        st.markdown(
                            '<div class="char-card">'
                            f'<div class="cname"><span class="e">{_emoji}</span>'
                            f'{html.escape(str(char.get("character_name", "Character")))}</div>'
                            f'<div class="cdesc">{html.escape(str(char.get("visual_description", "")))}</div>'
                            '<div class="clabel">Flow AI Master Reference Prompt</div>'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                        st.code(char.get("flow_ai_ref_prompt", ""), language="text")

    # ---- Scenes tab ----
    with tab_scenes:
        scenes = data.get("scenes", [])
        for scene in scenes:
            sc_num = scene.get("scene_number", 1)
            _frac = (sc_num - 1) / max(n_scenes, 1)
            dialogue = html.escape(str(scene.get("dialogue_myanmar", ""))).replace("\n", "<br>")
            st.markdown(
                '<div class="scene-card"><div class="scene-top">'
                f'<span class="scene-num">SCENE {sc_num}</span>'
                f"{_act_chip(_frac, is_melo_part)}"
                "</div>"
                f'<div class="dialogue-box">🗣️ {dialogue}</div></div>',
                unsafe_allow_html=True,
            )

            col_img, col_vid = st.columns(2)
            with col_img:
                st.markdown('<div class="prompt-panel-label">🖼️ Image Prompt</div>', unsafe_allow_html=True)
                st.code(scene.get("image_prompt_en", ""), language="text")
            with col_vid:
                st.markdown('<div class="prompt-panel-label">🎥 Motion Prompt</div>', unsafe_allow_html=True)
                st.code(scene.get("img_to_video_prompt_en", ""), language="text")

            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                regen_clicked = st.button("🔄 ပြန်ဆွဲမည်", key=f"regen_{st.session_state.active_part}_{sc_num}",
                                          width="stretch")
            with btn_col2:
                backup = part.get("scene_backup", {}).get(str(sc_num))
                undo_clicked = False
                if backup:
                    undo_clicked = st.button("↩️ ရှေ့ဟောင်းပြန်ယူမည်", key=f"undo_{st.session_state.active_part}_{sc_num}",
                                             width="stretch")

            if undo_clicked:
                for i, s in enumerate(data["scenes"]):
                    if s.get("scene_number") == sc_num:
                        data["scenes"][i] = backup
                        break
                part.setdefault("scene_backup", {}).pop(str(sc_num), None)
                part["data"] = data
                persist_project()
                st.rerun()

            if regen_clicked:
                api_key = (api_key_input or "").strip()
                if not api_key:
                    st.error("⚠️ API Key ထည့်ပါဦး။")
                else:
                    first_choice = custom_model.strip() if custom_model.strip() else selected_model
                    model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]
                    dummy_batch = {"start": sc_num, "count": 1, "total": len(data.get("scenes", []))}

                    system_prompt = build_system_prompt(
                        part["style"], dummy_batch, part.get("series_type", SERIES_OPTIONS[0]),
                        part.get("satire_intensity") if part.get("is_satire") else None,
                        mode="single_scene",
                    )
                    user_instruction = build_single_scene_instruction(data, sc_num, part["style"], part["genre"])

                    with st.status(f"Scene {sc_num} ပြန်ဆွဲနေသည်…", expanded=True) as status:
                        try:
                            _model, new_scene = generate_with_fallback(
                                status, api_key, model_order, SINGLE_SCENE_SCHEMA, system_prompt, user_instruction,
                            )
                            old_scene = None
                            for i, s in enumerate(data["scenes"]):
                                if s.get("scene_number") == sc_num:
                                    old_scene = s
                                    data["scenes"][i] = new_scene
                                    break
                            if old_scene:
                                part.setdefault("scene_backup", {})[str(sc_num)] = old_scene
                            data = apply_consistency(
                                data, part["character_clause"], part.get("style_bible", ""),
                                aspect_meta["prompt_text"], scene_number=sc_num,
                            )
                            part["data"] = data
                            persist_project()
                            st.rerun()
                        except errors.APIError as e:
                            st.error(friendly_api_error(e))
                            show_attempt_log(e)
                        except Exception as e:
                            st.error(f"မအောင်မြင်ပါ: {e}")

    # ---- Export & ဆက်ရန် tab ----
    with tab_export:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="step-num">💾</div>'
            '<div class="panel-title">Download & Copy</div></div>'
            '<div class="panel-sub">Script အပြည့်အစုံကို file အဖြစ် သိမ်းမလား၊ prompt တွေချည်း copy ကူးမလား — ကြိုက်တာရွေးပါ။</div>',
            unsafe_allow_html=True,
        )

        def build_export_text(d: Dict[str, Any]) -> str:
            out = f"Title: {d.get('title')}\nLogline: {d.get('logline')}\n\n"
            if d.get("style_bible"):
                out += f"Style Bible: {d.get('style_bible')}\n\n"
            out += "--- CHARACTER SHEET ---\n"
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
            st.download_button("📄 ဒီ Part (Text)", data=build_export_text(data),
                               file_name=f"script_studio_part{st.session_state.active_part + 1}.txt",
                               mime="text/plain", width="stretch")
        with dl_col2:
            all_text = "\n\n===== NEXT PART =====\n\n".join(build_export_text(p["data"]) for p in parts)
            st.download_button("📄 Part အားလုံး (Text)", data=all_text,
                               file_name="script_studio_full_series.txt", mime="text/plain", width="stretch")
        with dl_col3:
            st.download_button("💾 Project (JSON)", data=json.dumps(parts, ensure_ascii=False, indent=2),
                               file_name="script_studio_project.json", mime="application/json", width="stretch")

        st.markdown('<div class="field-label" style="margin-top:18px;">📋 Prompt အားလုံး တစ်စုတည်း Copy ကူးရန်</div>',
                    unsafe_allow_html=True)
        combined_prompts = "\n\n".join(
            f"Scene {s.get('scene_number')} - IMAGE:\n{s.get('image_prompt_en', '')}\n\n"
            f"Scene {s.get('scene_number')} - MOTION:\n{s.get('img_to_video_prompt_en', '')}"
            for s in data.get("scenes", [])
        )
        st.code(combined_prompts, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-head"><div class="step-num">➕</div>'
            '<div class="panel-title">နောက် Part ဆက်ရေးရန်</div></div>'
            '<div class="panel-sub">ဇာတ်ကောင်၊ style နဲ့ ဇာတ်လမ်းဆက်စပ်မှု အကုန် မှတ်ထားပြီးသား အတိုင်း နောက် အပိုင်း ထပ်ရေးပေးပါမယ်။</div>',
            unsafe_allow_html=True,
        )
        next_idea = st.text_area("ဒီအပိုင်းအတွက် ဇာတ်လမ်းလမ်းညွှန် (optional)",
                                 placeholder="ဥပမာ — ဒီအပိုင်းမှာ ဗီလိန်ရဲ့ လျှို့ဝှက်ချက်ပေါ်ပါစေ...",
                                 key=f"next_idea_{st.session_state.active_part}")
        if st.button("➕ နောက် Part ထပ်ဖန်တီးမည်", width="stretch", type="primary"):
            api_key = (api_key_input or "").strip()
            if not api_key:
                st.error("⚠️ API Key ထည့်ပါဦး။")
            else:
                first_choice = custom_model.strip() if custom_model.strip() else selected_model
                model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]
                duration_meta = DURATIONS[part["duration_label"]]
                hook_required = aspect_meta["code"] == "9:16"

                with st.status("နောက် Part ရေးနေသည်…", expanded=True) as status:
                    try:
                        used_model, new_data = generate_continuation(
                            status, api_key, model_order, data, part["style"], part["genre"],
                            duration_meta, part.get("series_type", SERIES_OPTIONS[0]), next_idea,
                            part.get("satire_intensity") if part.get("is_satire") else None, hook_required,
                            melodrama_archetype=part.get("melodrama_archetype"),
                        )
                        character_clause = build_character_clause(new_data)
                        style_bible = new_data.get("style_bible", "")
                        new_data = apply_consistency(new_data, character_clause, style_bible, aspect_meta["prompt_text"])

                        new_part = dict(part)
                        new_part["data"] = new_data
                        new_part["model"] = used_model
                        new_part["character_clause"] = character_clause
                        new_part["style_bible"] = style_bible
                        new_part.pop("scene_backup", None)

                        st.session_state.parts.append(new_part)
                        st.session_state.active_part = len(st.session_state.parts) - 1
                        persist_project()
                        st.toast(f"✅ Part {len(st.session_state.parts)} ဖန်တီးပြီးပါပြီ!")
                        st.rerun()
                    except errors.APIError as e:
                        st.error(friendly_api_error(e))
                        show_attempt_log(e)
                    except Exception as e:
                        st.error(f"မအောင်မြင်ပါ: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer">🎬 <b>Script Studio</b> — Flow AI video creators အတွက် ဖန်တီးထားသည် · '
    "API key ကို browser ထဲမှာပဲ သိမ်းသည်</div>",
    unsafe_allow_html=True,
)

import html
import json
import re
import time

import streamlit as st
from google import genai
from google.genai import errors, types

# =====================================================================
# MODEL SETTINGS
# gemini-2.5-flash ကို အသစ်သုံးသူတွေအတွက် Google က ပိတ်လိုက်ပြီ (404)
# Model အသစ်တွေ မကြာခဏ ထွက်/ပျက်လို့ ဒီစာရင်းတစ်နေရာတည်းမှာပဲ ပြင်ပါ
# ဒီစာရင်းထဲက ပထမဆုံးကို default အဖြစ်သုံးပြီး 404 တက်ရင် နောက်တစ်ခုကို အလိုအလျောက် စမ်းပါမယ်
# =====================================================================
MODEL_CHOICES = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

st.set_page_config(
    page_title="Flow AI Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    color: #1E40AF;
    font-size: 26px;
    font-weight: bold;
    margin-bottom: 4px;
}
.sub-title {
    color: #4B5563;
    font-size: 14px;
    margin-bottom: 20px;
}
.scene-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 14px;
}
.badge {
    background-color: #2563EB;
    color: white;
    font-size: 12px;
    font-weight: bold;
    padding: 3px 10px;
    border-radius: 12px;
    display: inline-block;
    margin-bottom: 8px;
}
.dialogue-box {
    background-color: #EFF6FF;
    border-left: 4px solid #3B82F6;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 10px;
    color: #1E293B;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def parse_json_response(raw_text):
    """AI ပြန်လာတဲ့ စာကို JSON အဖြစ် ဘေးကင်းစွာ ပြောင်းပေးသည်။"""
    if not raw_text or not raw_text.strip():
        raise ValueError("AI ဘက်က အဖြေ အလွတ်ပြန်လာပါသည်။")

    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # စာအပိုတွေ ပါလာရင် { ... } အပိုင်းကိုပဲ ဆွဲထုတ်ကြည့်မယ်
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("AI ပြန်လာတဲ့ ပုံစံ မမှန်ပါ။")
    return data


def generate_script(api_key, model_order, system_prompt, user_instruction):
    """Model တစ်ခုချင်းစီကို အစဉ်လိုက်စမ်းပြီး အောင်မြင်တဲ့ (model, data) ကို ပြန်ပေးသည်။"""
    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in model_order:
        for attempt in range(2):
            try:
                # မှတ်ချက် - Gemini 3 မိသားစုမှာ temperature / top_p / top_k
                # မသုံးတော့ပါ (default နဲ့ပဲ သုံးရမယ်) — ထို့ကြောင့် မထည့်တော့ပါ
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_instruction,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                    ),
                )
                return model_name, parse_json_response(response.text)

            except errors.APIError as e:
                last_error = e
                code = getattr(e, "code", None)

                if code == 404:
                    break  # ဒီ model မရတော့ဘူး → နောက် model ကို ကူးမယ်
                if code in (500, 503, 504):
                    if attempt == 0:
                        time.sleep(3)  # server ရှုပ်နေရင် ၃ စက္ကန့်စောင့်ပြီး တစ်ခါပြန်စမ်း
                        continue
                    break
                raise  # key မှား၊ quota ပြည့် စတာတွေကို ချက်ချင်းပြသမယ်

            except ValueError as e:
                # JSON ပျက်နေရင် တစ်ခါပြန်စမ်းမယ်
                last_error = e
                if attempt == 0:
                    continue
                raise

    if last_error:
        raise last_error
    raise RuntimeError("Model တစ်ခုမှ အလုပ်မလုပ်ပါ။")


def friendly_api_error(e):
    """API error ကို နားလည်လွယ်တဲ့ မြန်မာစာ message ပြောင်းပေးသည်။"""
    code = getattr(e, "code", None)
    msg = str(getattr(e, "message", "") or e)
    low = msg.lower()

    if code == 404:
        return ("⚠️ Model အားလုံး ရှာမတွေ့ပါ / အသုံးပြုခွင့်မရှိတော့ပါ။ "
                "ဖိုင်ထိပ်က MODEL_CHOICES ထဲက model နာမည်တွေကို "
                "https://ai.google.dev/gemini-api/docs/models မှာ စစ်ပြီး အသစ်ပြောင်းပါ။")
    if code in (401, 403) or (code == 400 and "api key" in low):
        return ("🔑 API Key မှားနေပါသည် (သို့) ခွင့်ပြုချက်မရှိပါ။ "
                "Google AI Studio မှ Key ကို ပြန်ကူးပြီး ထည့်ကြည့်ပါ။")
    if code == 429:
        return ("⏳ Free quota (သို့) တောင်းဆိုမှု အကန့်အသတ် ပြည့်နေပါသည်။ "
                "ခဏစောင့်ပြီး ပြန်စမ်းပါ၊ မဖြစ်သေးရင် ဒီနေ့ quota ကုန်နေနိုင်ပါသည်။")
    if code and code >= 500:
        return "🌐 Google server ဘက်က အလုပ်များနေပါသည်။ ခဏနေမှ ပြန်စမ်းကြည့်ပါ။"
    return f"Error ({code}): {msg}"


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ API Settings")
    api_key_input = st.text_input(
        "Google AI Studio API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Google AI Studio မှ ရရှိသော Gemini API Key ကို ထည့်ပါ",
    )
    st.caption("🔒 Key ကို လုံခြုံစွာ လက်ရှိ session အတွင်းသာ အသုံးပြုပါသည်။")

    selected_model = st.selectbox(
        "Gemini Model",
        MODEL_CHOICES,
        index=0,
        help="ရွေးထားတဲ့ model မရရင် ကျန်တဲ့ model တွေကို အလိုအလျောက် ဆက်စမ်းပါမယ်",
    )
    with st.expander("🔧 Advanced"):
        custom_model = st.text_input(
            "Custom model name (မလိုရင် ဗလာထားပါ)",
            placeholder="gemini-3.8-flash",
        )

    st.markdown("---")
    st.caption("🎯 Flow AI Platform အတွက် အထူးထုတ်လုပ်ထားပါသည်။")

# =====================================================================
# MAIN FORM
# =====================================================================
st.markdown('<div class="main-title">🎬 Flow AI Video Script & Prompt Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Flow AI အတွက် Script၊ Character Sheet နှင့် Scene-by-Scene Prompts များ ထုတ်ပေးသည့်စနစ်</div>', unsafe_allow_html=True)

categories = {
    "အသီးခေါင်း AI (Talking Fruit)": ["ဟာသ", "အချစ်", "ပညာပေး", "ကလဲ့စား"],
    "တရားတော် ပုံပြ (Studio Ghibli)": ["၅၅၀ ဇာတ်တော်", "ဒဿဇာတ်တော်", "ဗုဒ္ဓဝင်ဖြစ်ရပ်", "စိတ်ခွန်အားဖြည့် ဓမ္မပုံပြင်"],
    "ကလေးများအတွက် 3D Animation": ["တိရစ္ဆာန်ပုံပြင်", "စာရိတ္တပညာပေး", "အိပ်ရာဝင်ပုံပြင်", "ဟာသ စွန့်စားခန်း"],
}

# မြန်မာစာ label → AI နားလည်လွယ်တဲ့ အင်္ဂလိပ်လို တိကျတဲ့ scene အရေအတွက်
durations = {
    "၁ မိနစ်ဝန်းကျင် (Scene ၄ မှ ၅ ခု)": "about 1 minute, exactly 4 to 5 scenes",
    "၂ မိနစ်ဝန်းကျင် (Scene ၇ မှ ၈ ခု)": "about 2 minutes, exactly 7 to 8 scenes",
    "၃ မိနစ်ဝန်းကျင် (Scene ၁၀ မှ ၁၂ ခု)": "about 3 minutes, exactly 10 to 12 scenes",
}

col1, col2 = st.columns(2)
with col1:
    selected_style = st.selectbox("၁။ ဗီဒီယို ပုံစံ (Style):", list(categories.keys()))
with col2:
    selected_genre = st.selectbox("၂။ ဇာတ်လမ်း အမျိုးအစား (Genre):", categories[selected_style])

col3, col4 = st.columns(2)
with col3:
    video_duration = st.selectbox("၃။ ဗီဒီယို ကြာချိန် (Duration):", list(durations.keys()))
with col4:
    series_type = st.selectbox(
        "၄။ ဇာတ်လမ်း ဖွဲ့စည်းပုံ:",
        [
            "တစ်ပိုင်းတည်း အပြီး (Standalone Episode)",
            "အပိုင်းဆက် Series - အပိုင်း (၁) အစပျိုး",
        ],
    )

custom_idea = st.text_area(
    "၅။ ထည့်သွင်းလိုသော အကြောင်းအရာ (စိတ်ကြိုက်):",
    placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်...",
)

SYSTEM_PROMPT = """
You are an expert AI scriptwriter and Prompt Engineer specialized exclusively for Flow AI video workflows.
Return your output ONLY as a valid raw JSON object without markdown formatting or code blocks.
The JSON must follow this exact schema:
{
    "title": "Video title in Myanmar",
    "logline": "Short summary in Myanmar",
    "character_sheet": [
        {
            "character_name": "Character name",
            "visual_description": "Appearance details",
            "flow_ai_ref_prompt": "Master visual reference prompt in English"
        }
    ],
    "scenes": [
        {
            "scene_number": 1,
            "dialogue_myanmar": "မြန်မာလို စကားပြော သို့မဟုတ် voiceover",
            "image_prompt_en": "Flow AI high quality image prompt in English",
            "img_to_video_prompt_en": "Flow AI camera motion and movement prompt in English"
        }
    ]
}
Rules:
1. If 'အသီးခေါင်း AI': Expressive, anthropomorphic fruits with clear facial emotion.
2. If 'တရားတော် ပုံပြ': Studio Ghibli style, serene nature, traditional aesthetic, soft warm light.
3. If 'ကလေးများအတွက် 3D': Pixar-like Disney 3D animation style, cute and rounded.
4. Myanmar for dialogue, professional detailed English for image and video motion prompts.
5. Follow the scene count in the Duration line exactly. scene_number must be sequential integers starting from 1.
6. If the Series Structure is Part 1 of a series, end the last scene with a hook that sets up Part 2.
"""

# =====================================================================
# GENERATE
# =====================================================================
if st.button("🚀 Script & Prompts ဖန်တီးမည်", type="primary", width="stretch"):
    api_key = (api_key_input or "").strip()

    if not api_key:
        st.error("⚠️ ကျေးဇူးပြု၍ ဘယ်ဘက် Sidebar တွင် Google AI Studio API Key ကို ဦးစွာထည့်သွင်းပေးပါ။")
    else:
        # ရွေးထားတဲ့ model ကို ရှေ့ဆုံးထား၊ ကျန်တာတွေကို fallback အဖြစ်နောက်မှာထား
        first_choice = custom_model.strip() if custom_model and custom_model.strip() else selected_model
        model_order = [first_choice] + [m for m in MODEL_CHOICES if m != first_choice]

        user_instruction = f"""
Create a video production script with:
Style: {selected_style}
Genre: {selected_genre}
Duration: {durations[video_duration]}
Series Structure: {series_type}
Topic/Idea: {custom_idea.strip() if custom_idea and custom_idea.strip() else "Creative natural storyline"}
"""

        with st.spinner("AI က Flow AI အတွက် အတိကျဆုံး Prompt များနှင့် ဇာတ်ညွှန်းကို စဉ်းစားဖန်တီးနေပါသည်..."):
            try:
                used_model, result = generate_script(api_key, model_order, SYSTEM_PROMPT, user_instruction)
                st.session_state["script_result"] = result
                st.session_state["used_model"] = used_model
                st.success(f"အောင်မြင်စွာ ထုတ်လုပ်ပြီးပါပြီ! (Model: {used_model})")

            except errors.APIError as e:
                st.error(friendly_api_error(e))
                with st.expander("Technical details"):
                    st.code(str(e), language="text")

            except ValueError:
                st.error("⚠️ AI ပြန်လာတဲ့ အဖြေကို ဖတ်မရပါ (JSON ပျက်နေပါသည်)။ ခဏနေပြီး ထပ်နှိပ်ကြည့်ပါ။")

            except Exception as e:
                st.error(f"မမျှော်လင့်သော Error ဖြစ်ပေါ်ခဲ့ပါသည်: {e}")

# =====================================================================
# RESULT DISPLAY
# =====================================================================
if "script_result" in st.session_state:
    data = st.session_state["script_result"]

    st.markdown("---")
    st.subheader(f"📌 {data.get('title', 'Video Script')}")
    st.markdown(f"**ဇာတ်လမ်း အကျဉ်း:** {data.get('logline', '')}")
    if st.session_state.get("used_model"):
        st.caption(f"Model: {st.session_state['used_model']}")

    st.markdown("### 👤 Character Consistency Sheet (Flow AI)")
    for char in data.get("character_sheet", []):
        with st.expander(f"✨ ဇာတ်ကောင်: {char.get('character_name', 'Character')}", expanded=True):
            st.markdown(f"**ရုပ်သွင်:** {char.get('visual_description', '')}")
            st.markdown("**Flow AI Master Reference Prompt:**")
            st.code(char.get("flow_ai_ref_prompt", ""), language="text")

    st.markdown("### 🎬 Scene-by-Scene Production (အခန်းခွဲများ)")
    for scene in data.get("scenes", []):
        sc_num = scene.get("scene_number", 1)
        # HTML ထဲထည့်ခင် escape လုပ်ထားမှ AI စာထဲက < > တွေကြောင့် layout မပျက်မှာ
        dialogue = html.escape(str(scene.get("dialogue_myanmar", ""))).replace("\n", "<br>")
        st.markdown(
            f'<div class="scene-card">'
            f'<span class="badge">Scene {sc_num}</span>'
            f'<div class="dialogue-box">🗣️ <b>စကားပြော (Dialogue):</b> {dialogue}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        col_img, col_vid = st.columns(2)
        with col_img:
            st.markdown("🖼️ **Flow AI Image Prompt:**")
            st.code(scene.get("image_prompt_en", ""), language="text")
        with col_vid:
            st.markdown("🎥 **Video Motion Prompt:**")
            st.code(scene.get("img_to_video_prompt_en", ""), language="text")

    st.markdown("### 💾 Export Script")
    export_txt = f"Title: {data.get('title')}\nLogline: {data.get('logline')}\n\n"
    export_txt += "--- CHARACTER SHEET ---\n"
    for c in data.get("character_sheet", []):
        export_txt += f"Name: {c.get('character_name')}\n"
        export_txt += f"Description: {c.get('visual_description')}\n"
        export_txt += f"Ref Prompt: {c.get('flow_ai_ref_prompt')}\n\n"

    export_txt += "--- SCENES ---\n"
    for s in data.get("scenes", []):
        export_txt += f"Scene {s.get('scene_number')}:\n"
        export_txt += f"Dialogue: {s.get('dialogue_myanmar')}\n"
        export_txt += f"Image Prompt: {s.get('image_prompt_en')}\n"
        export_txt += f"Motion Prompt: {s.get('img_to_video_prompt_en')}\n\n"

    st.download_button(
        label="📄 Text (.txt) ဖြင့် Download ရယူမည်",
        data=export_txt,
        file_name="flow_ai_script.txt",
        mime="text/plain",
        width="stretch",
    )


import json
import re
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Flow AI Script Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
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

with st.sidebar:
    st.markdown("### ⚙️ API Settings")
    api_key_input = st.text_input(
        "Google AI Studio API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Google AI Studio မှ ရရှိသော Gemini API Key ကို ထည့်ပါ"
    )
    st.caption("🔒 Key ကို လုံခြုံစွာ လက်ရှိ session အတွင်းသာ အသုံးပြုပါသည်။")
    st.markdown("---")
    st.caption("🎯 Flow AI Platform အတွက် အထူးထုတ်လုပ်ထားပါသည်။")

st.markdown('<div class="main-title">🎬 Flow AI Video Script & Prompt Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Flow AI အတွက် Script၊ Character Sheet နှင့် Scene-by-Scene Prompts များ ထုတ်ပေးသည့်စနစ်</div>', unsafe_allow_html=True)

categories = {
    "အသီးခေါင်း AI (Talking Fruit)": ["ဟာသ", "အချစ်", "ပညာပေး", "ကလဲ့စား"],
    "တရားတော် ပုံပြ (Studio Ghibli)": ["၅၅၀ ဇာတ်တော်", "ဒဿဇာတ်တော်", "ဗုဒ္ဓဝင်ဖြစ်ရပ်", "စိတ်ခွန်အားဖြည့် ဓမ္မပုံပြင်"],
    "ကလေးများအတွက် 3D Animation": ["တိရစ္ဆာန်ပုံပြင်", "စာရိတ္တပညာပေး", "အိပ်ရာဝင်ပုံပြင်", "ဟာသ စွန့်စားခန်း"]
}

col1, col2 = st.columns(2)
with col1:
    selected_style = st.selectbox("၁။ ဗီဒီယို ပုံစံ (Style):", list(categories.keys()))
with col2:
    selected_genre = st.selectbox("၂။ ဇာတ်လမ်း အမျိုးအစား (Genre):", categories[selected_style])

col3, col4 = st.columns(2)
with col3:
    video_duration = st.selectbox(
        "၃။ ဗီဒီယို ကြာချိန် (Duration):",
        [
            "၁ မိနစ်ဝန်းကျင် (Scene ၄ မှ ၅ ခု)",
            "၂ မိနစ်ဝန်းကျင် (Scene ၇ မှ ၈ ခု)",
            "၃ မိနစ်ဝန်းကျင် (Scene ၁၀ မှ ၁၂ ခု)"
        ]
    )
with col4:
    series_type = st.selectbox(
        "၄။ ဇာတ်လမ်း ဖွဲ့စည်းပုံ:",
        [
            "တစ်ပိုင်းတည်း အပြီး (Standalone Episode)",
            "အပိုင်းဆက် Series - အပိုင်း (၁) အစပျိုး"
        ]
    )

custom_idea = st.text_area(
    "၅။ ထည့်သွင်းလိုသော အကြောင်းအရာ (စိတ်ကြိုက်):",
    placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်..."
)

if st.button("🚀 Script & Prompts ဖန်တီးမည်", type="primary", use_container_width=True):
    if not api_key_input or not api_key_input.strip():
        st.error("⚠️ ကျေးဇူးပြု၍ ဘယ်ဘက် Sidebar တွင် Google AI Studio API Key ကို ဦးစွာထည့်သွင်းပေးပါ။")
    else:
        with st.spinner("AI က Flow AI အတွက် အတိကျဆုံး Prompt များနှင့် ဇာတ်ညွှန်းကို စဉ်းစားဖန်တီးနေပါသည်..."):
            try:
                client = genai.Client(api_key=api_key_input.strip())

                system_prompt = """
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
"""

                user_instruction = f"""
Create a video production script with:
Style: {selected_style}
Genre: {selected_genre}
Duration: {video_duration}
Series Structure: {series_type}
Topic/Idea: {custom_idea if custom_idea else "Creative natural storyline"}
"""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_instruction,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.7,
                    )
                )

                raw_text = response.text.strip()
                cleaned_text = re.sub(r"^```json\s*", "", raw_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
                st.session_state["script_result"] = json.loads(cleaned_text)
                st.success("အောင်မြင်စွာ ထုတ်လုပ်ပြီးပါပြီ!")

            except Exception as e:
                st.error(f"Error ဖြစ်ပေါ်ခဲ့ပါသည်: {str(e)}")

if "script_result" in st.session_state:
    data = st.session_state["script_result"]

    st.markdown("---")
    st.subheader(f"📌 {data.get('title', 'Video Script')}")
    st.markdown(f"**ဇာတ်လမ်း အကျဉ်း:** {data.get('logline', '')}")

    st.markdown("### 👤 Character Consistency Sheet (Flow AI)")
    for char in data.get("character_sheet", []):
        with st.expander(f"✨ ဇာတ်ကောင်: {char.get('character_name', 'Character')}", expanded=True):
            st.markdown(f"**ရုပ်သွင်:** {char.get('visual_description', '')}")
            st.markdown("**Flow AI Master Reference Prompt:**")
            st.code(char.get("flow_ai_ref_prompt", ""), language="text")

    st.markdown("### 🎬 Scene-by-Scene Production (အခန်းခွဲများ)")
    for scene in data.get("scenes", []):
        sc_num = scene.get("scene_number", 1)
        st.markdown(f"""
        <div class="scene-card">
            <span class="badge">Scene {sc_num}</span>
            <div class="dialogue-box">🗣️ <b>စကားပြော (Dialogue):</b> {scene.get('dialogue_myanmar', '')}</div>
        </div>
        """, unsafe_allow_html=True)

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
        use_container_width=True
    )

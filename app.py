import json
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from streamlit_javascript import st_javascript
-------------------------------------------------------------
App Configuration & Theme
-------------------------------------------------------------
st.set_page_config(
page_title="Flow AI Studio Script & Prompt Generator",
page_icon="🎬",
layout="wide",
initial_sidebar_state="expanded"
)
Custom Styling (Modern Professional Blue Theme & Mobile Responsive)
st.markdown("""
<style>
:root {
--primary-blue: #1A56DB;
--light-blue: #EBF5FF;
--dark-blue: #1E429F;
}
.main-header {
font-size: 1.85rem;
font-weight: 700;
color: #1E3A8A;
margin-bottom: 0.25rem;
}
.sub-header {
font-size: 0.95rem;
color: #4B5563;
margin-bottom: 1.5rem;
}
.card-box {
background-color: #F8FAFC;
border: 1px solid #E2E8F0;
border-radius: 10px;
padding: 16px;
margin-bottom: 14px;
}
.scene-badge {
background-color: #1A56DB;
color: white;
padding: 3px 10px;
border-radius: 9999px;
font-size: 0.8rem;
font-weight: 600;
display: inline-block;
margin-bottom: 8px;
}
.dialogue-text {
font-size: 1.05rem;
line-height: 1.6;
color: #0F172A;
background-color: #FFFFFF;
padding: 10px;
border-left: 4px solid #2563EB;
border-radius: 4px;
margin: 8px 0;
}
/* Mobile optimization */
@media (max-width: 768px) {
.main-header { font-size: 1.4rem; }
.card-box { padding: 12px; }
}
</style>
""", unsafe_allow_html=True)
-------------------------------------------------------------
Structured Output Schema (Pydantic) for Flow AI
-------------------------------------------------------------
class SceneItem(BaseModel):
scene_number: int = Field(description="Scene number in sequential order")
dialogue_myanmar: str = Field(description="မြန်မာဘာသာဖြင့် စကားပြော Dialogue သို့မဟုတ် Voiceover စာသား")
image_prompt_en: str = Field(description="Detailed Flow AI Image prompt in English, including subject, lighting, angle, and style")
img_to_video_prompt_en: str = Field(description="Flow AI Image-to-Video camera movement and motion prompt in English")
class CharacterProfile(BaseModel):
character_name: str = Field(description="Name of the character")
visual_description: str = Field(description="Detailed traits, attire, facial features for consistency")
flow_ai_ref_prompt: str = Field(description="Master reference prompt for Flow AI to maintain consistent face and attire")
class VideoScriptPackage(BaseModel):
title: str = Field(description="Title of the video in Myanmar")
logline: str = Field(description="Short summary of the story in Myanmar")
character_sheet: list[CharacterProfile] = Field(description="List of characters and their consistency prompts")
scenes: list[SceneItem] = Field(description="List of scenes matching the requested length")
-------------------------------------------------------------
Persistent Local Storage for API Key
-------------------------------------------------------------
stored_key = st_javascript("""localStorage.getItem('gemini_api_key') || '';""")
with st.sidebar:
st.markdown("### ⚙️ Settings")
# Initialize input with stored key if available
default_key_val = stored_key if stored_key and stored_key != "null" else ""
user_api_key = st.text_input(
"Google AI Studio API Key",
value=default_key_val,
type="password",
placeholder="AIzaSy...",
help="သင့် Key ကို သင့် Browser Local Storage တွင်သာ သိမ်းဆည်းထားမည်ဖြစ်သည်။"
)
col_save, col_clear = st.columns(2)
with col_save:
if st.button("💾 Key သိမ်းရန်", use_container_width=True):
if user_api_key.strip():
st_javascript(f"""localStorage.setItem('gemini_api_key', '{user_api_key.strip()}');""")
st.success("Key ကို မှတ်သားပြီးပါပြီ!")
st.rerun()
else:
st.warning("Key ထည့်သွင်းပါ")
with col_clear:
if st.button("🗑️ ဖျက်မည်", use_container_width=True):
st_javascript("""localStorage.removeItem('gemini_api_key');""")
st.info("Key ကို ဖျက်လိုက်ပါပြီ။")
st.rerun()
st.markdown("---")
st.caption("🎯 Flow AI ပလက်ဖောင်းအတွက် Image နှင့် Image-to-Video Prompts များကို အထူးပြုပြင်ဆင်ပေးပါသည်။")
-------------------------------------------------------------
Main Application Interface
-------------------------------------------------------------
st.markdown('<div class="main-header">🎬 AI Video Script & Prompt Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Flow AI အတွက် Script၊ Character Sheet နှင့် Scene-by-Scene Prompts များ ထုတ်ပေးသည့်စနစ်</div>', unsafe_allow_html=True)
Selection Menus
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
["၁ မိနစ်ဝန်းကျင် (Scene ၄ မှ ၅ ခု)", "၂ မိနစ်ဝန်းကျင် (Scene ၇ မှ ၈ ခု)", "၃ မိနစ်ဝန်းကျင် (Scene ၁၀ မှ ၁၂ ခု)"]
)
with col4:
series_type = st.selectbox(
"၄။ ဇာတ်လမ်း ဖွဲ့စည်းပုံ:",
["တစ်ပိုင်းတည်း အပြီး (Standalone Episode)", "အပိုင်းဆက် Series - အပိုင်း (၁) အစပျိုး"]
)
custom_idea = st.text_area(
"၅။ ထည့်သွင်းလိုသော အကြောင်းအရာ သို့မဟုတ် ဇာတ်ကွက် (စိတ်ကြိုက်):",
placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်..."
)
-------------------------------------------------------------
Generation Logic
-------------------------------------------------------------
if st.button("🚀 Script & Prompts ဖန်တီးမည်", type="primary", use_container_width=True):
active_key = user_api_key.strip() if user_api_key else (stored_key if stored_key and stored_key != "null" else None)
if not active_key:
st.error("⚠️ ဘယ်ဘက် Sidebar တွင် Google AI Studio API Key အရင် ထည့်သွင်းပေးပါ။")
else:
with st.spinner("AI က Flow AI အတွက် အတိကျဆုံး Prompt များနှင့် ဇာတ်ညွှန်းကို ဖန်တီးနေပါသည်..."):
try:
client = genai.Client(api_key=active_key)
system_instruction = """
You are a world-class AI scriptwriter and Prompt Engineer specialized exclusively in Flow AI (Flux image generation and Flow AI image-to-video motion workflows).
Rules:
1. Style rules:
- If 'အသီးခေါင်း AI', produce photorealistic human expressions on fruit bodies, expressive eyes, comedic or dramatic tone.
- If 'တရားတော် ပုံပြ', produce serene, aesthetic Studio Ghibli anime style, warm lighting, pastel tones, tranquil nature.
- If 'ကလေးများအတွက် 3D', produce Pixar/Disney style 3D animation, bright vivid colors, cute rounded character design.
2. Consistency: Provide a rock-solid Character Consistency Sheet with master prompts so the user gets identical characters across all scenes in Flow AI.
3. Language:
- Myanmar language for all titles, loglines, and spoken Dialogues/Voiceovers.
- English for all Flow AI Image Prompts and Image-to-Video prompts.
4. Match the requested duration and scene count faithfully.
"""
user_prompt = f"""
Create a full video production script package with the following specifications:
- Style: {selected_style}
- Genre: {selected_genre}
- Target Duration: {video_duration}
- Format Structure: {series_type}
- Custom Notes: {custom_idea if custom_idea else "Creative standard"}
Ensure every scene includes:
- dialogue_myanmar (Natural, emotive Myanmar spoken dialogue)
- image_prompt_en (High quality prompt tailored for Flow AI Image generation)
- img_to_video_prompt_en (Smooth camera motion, e.g., slow zoom in, pan right, subtle facial motion)
"""
response = client.models.generate_content(
model="gemini-2.5-flash",
contents=user_prompt,
config=types.GenerateContentConfig(
system_instruction=system_instruction,
response_mime_type="application/json",
response_schema=VideoScriptPackage,
temperature=0.7,
)
)
data = json.loads(response.text)
st.session_state["script_result"] = data
st.success("အောင်မြင်စွာ ဖန်တီးပြီးပါပြီ!")
except Exception as e:
st.error(f"အမှားဖြစ်ပေါ်ခဲ့သည်: {str(e)}")
-------------------------------------------------------------
Render Output Results
-------------------------------------------------------------
if "script_result" in st.session_state:
data = st.session_state["script_result"]
st.markdown("---")
st.subheader(f"📌 {data.get('title', 'Video Script')}")
st.markdown(f"ဇာတ်လမ်း အကျဉ်း: {data.get('logline', '')}")
# 1. Character Consistency Sheet
st.markdown("### 👤 Character Consistency Sheet (Flow AI)")
st.caption("Flow AI တွင် ဇာတ်ကောင် မပြောင်းလဲစေရန် အောက်ပါ Reference Prompt ကို အသုံးပြုပါ။")
for char in data.get("character_sheet", []):
with st.expander(f"✨ ဇာတ်ကောင်: {char.get('character_name')}", expanded=True):
st.markdown(f"ရုပ်သွင် အသေးစိတ်: {char.get('visual_description')}")
st.markdown("Flow AI Master Reference Prompt:")
st.code(char.get('flow_ai_ref_prompt'), language="text")
# 2. Scene by Scene Section
st.markdown("### 🎬 Scene-by-Scene Production (အခန်းခွဲများ)")
for scene in data.get("scenes", []):
sc_num = scene.get('scene_number')
st.markdown(f"""
<div class="card-box">
<span class="scene-badge">Scene {sc_num}</span>
<div class="dialogue-text">🗣️ <b>စကားပြော (Dialogue):</b> {scene.get('dialogue_myanmar')}</div>
</div>
""", unsafe_allow_html=True)
c_img, c_vid = st.columns(2)
with c_img:
st.markdown("🖼️ Flow AI Image Prompt (English):")
st.code(scene.get('image_prompt_en'), language="text")
with c_vid:
st.markdown("🎥 Image-to-Video Motion Prompt (English):")
st.code(scene.get('img_to_video_prompt_en'), language="text")
# 3. Export Buttons
st.markdown("### 💾 Export Script")
col_dl_txt, col_dl_json = st.columns(2)
# Generate Formatted Plaintext
txt_content = f"Title: {data.get('title')}\nLogline: {data.get('logline')}\n\n"
txt_content += "=== CHARACTER SHEET =\n"
for c in data.get("character_sheet", []):
txt_content += f"- {c.get('character_name')}: {c.get('visual_description')}\n"
txt_content += f"  Prompt: {c.get('flow_ai_ref_prompt')}\n\n"
txt_content += "= SCENES ===\n"
for s in data.get("scenes", []):
txt_content += f"Scene {s.get('scene_number')}:\n"
txt_content += f"Dialogue: {s.get('dialogue_myanmar')}\n"
txt_content += f"Image Prompt: {s.get('image_prompt_en')}\n"
txt_content += f"Video Motion Prompt: {s.get('img_to_video_prompt_en')}\n"
txt_content += "-" * 40 + "\n"
with col_dl_txt:
st.download_button(
label="📄 Script ကို Text (.txt) ဖြင့် သိမ်းမည်",
data=txt_content,
file_name=f"script_{data.get('title', 'video')}.txt",
mime="text/plain",
use_container_width=True
)
with col_dl_json:
st.download_button(
label="📦 Backup Data (.json) ဖြင့် သိမ်းမည်",
data=json.dumps(data, ensure_ascii=False, indent=2),
file_name=f"script_{data.get('title', 'video')}.json",
mime="application/json",
use_container_width=True
)

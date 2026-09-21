import json
from typing import List
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
st.set_page_config(
page_title="Flow AI Video Script Generator",
page_icon="🎬",
layout="wide",
initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.main-header {
font-size: 1.8rem;
font-weight: 700;
color: #1E3A8A;
margin-bottom: 0.2rem;
}
.sub-header {
font-size: 0.95rem;
color: #4B5563;
margin-bottom: 1.2rem;
}
.card-box {
background-color: #F8FAFC;
border: 1px solid #E2E8F0;
border-radius: 8px;
padding: 14px;
margin-bottom: 12px;
}
.scene-badge {
background-color: #1A56DB;
color: white;
padding: 2px 8px;
border-radius: 12px;
font-size: 0.8rem;
font-weight: 600;
display: inline-block;
margin-bottom: 6px;
}
.dialogue-text {
font-size: 1rem;
color: #0F172A;
background-color: #FFFFFF;
padding: 10px;
border-left: 4px solid #2563EB;
border-radius: 4px;
margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)
class SceneItem(BaseModel):
scene_number: int = Field(description="Scene sequence number")
dialogue_myanmar: str = Field(description="မြန်မာဘာသာဖြင့် စကားပြော သို့မဟုတ် Voiceover စာသား")
image_prompt_en: str = Field(description="Detailed Flow AI image generation prompt in English")
img_to_video_prompt_en: str = Field(description="Flow AI Image-to-Video camera movement and motion prompt in English")
class CharacterProfile(BaseModel):
character_name: str = Field(description="Name of the character")
visual_description: str = Field(description="Visual traits and appearance")
flow_ai_ref_prompt: str = Field(description="Master reference prompt for Flow AI to keep consistent character")
class VideoScriptPackage(BaseModel):
title: str = Field(description="Video title in Myanmar")
logline: str = Field(description="Short summary in Myanmar")
character_sheet: List[CharacterProfile] = Field(description="List of characters")
scenes: List[SceneItem] = Field(description="List of generated scenes")
with st.sidebar:
st.markdown("### ⚙️ Settings")
user_api_key = st.text_input(
"Google AI Studio API Key",
type="password",
placeholder="AIzaSy...",
help="Google AI Studio မှ ရရှိသော Gemini API Key ကို ထည့်ပါ"
)
st.caption("🔒 Key ကို လုံခြုံစွာဖြင့် လက်ရှိ Session အတွင်းသာ အသုံးပြုပါမည်။")
st.markdown("---")
st.caption("🎯 Flow AI Video Platform အတွက် အထူးထုတ်လုပ်ထားပါသည်။")
st.markdown('<div class="main-header">🎬 Flow AI Script & Prompt Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Flow AI အတွက် Script၊ Character Sheet နှင့် Scene-by-Scene Prompts များ ထုတ်ပေးသည့်စနစ်</div>', unsafe_allow_html=True)
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
"၅။ ထည့်သွင်းလိုသော အကြောင်းအရာ (စိတ်ကြိုက်):",
placeholder="ဥပမာ- ရန်ဖြစ်နေသော ငှက်ပျောသီးနှင့် သရက်သီး၊ ဒါမှမဟုတ် သီလပေးပုံပြင်..."
)
if st.button("🚀 Script & Prompts ဖန်တီးမည်", type="primary", use_container_width=True):
if not user_api_key or not user_api_key.strip():
st.error("⚠️ ဘယ်ဘက် Sidebar တွင် Google AI Studio API Key ကို အရင်ဆုံး ထည့်သွင်းပေးပါ။")
else:
with st.spinner("AI က Flow AI အတွက် အတိကျဆုံး Prompt များနှင့် ဇာတ်ညွှန်းကို ဖန်တီးနေပါသည်..."):
try:
client = genai.Client(api_key=user_api_key.strip())
system_instruction = """
You are a world-class AI scriptwriter and Prompt Engineer specialized exclusively in Flow AI workflows (Flux image generation and Flow AI video motion).
Rules:
1. If 'အသီးခေါင်း AI', produce expressive fruit characters with humorous or dramatic emotions.
2. If 'တရားတော် ပုံပြ', produce serene, aesthetic Studio Ghibli style, warm lighting, tranquil nature.
3. If 'ကလေးများအတွက် 3D', produce Pixar/Disney style 3D animation, cute rounded character design.
4. Maintain character consistency with a strong reference prompt.
5. Myanmar language for dialogue/voiceovers, and high quality English for Image & Motion Prompts.
"""
user_prompt = f"""
Create a full video production script package with the following:
- Style: {selected_style}
- Genre: {selected_genre}
- Target Duration: {video_duration}
- Structure: {series_type}
- Custom Idea: {custom_idea if custom_idea else "Creative standard"}
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
st.session_state["script_result"] = json.loads(response.text)
st.success("အောင်မြင်စွာ ဖန်တီးပြီးပါပြီ!")
except Exception as e:
st.error(f"Error ဖြစ်ပေါ်ခဲ့သည်: {str(e)}")
if "script_result" in st.session_state:
data = st.session_state["script_result"]
st.markdown("---")
st.subheader(f"📌 {data.get('title', 'Video Script')}")
st.markdown(f"ဇာတ်လမ်း အကျဉ်း: {data.get('logline', '')}")
st.markdown("### 👤 Character Consistency Sheet (Flow AI)")
for char in data.get("character_sheet", []):
with st.expander(f"✨ ဇာတ်ကောင်: {char.get('character_name')}", expanded=True):
st.markdown(f"ရုပ်သွင်: {char.get('visual_description')}")
st.markdown("Flow AI Master Reference Prompt:")
st.code(char.get('flow_ai_ref_prompt'), language="text")
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
st.markdown("🖼️ Flow AI Image Prompt:")
st.code(scene.get('image_prompt_en'), language="text")
with c_vid:
st.markdown("🎥 Video Motion Prompt:")
st.code(scene.get('img_to_video_prompt_en'), language="text")
st.markdown("### 💾 Export Script")
txt_content = f"Title: {data.get('title')}\nLogline: {data.get('logline')}\n\n"
for s in data.get("scenes", []):
txt_content += f"Scene {s.get('scene_number')}:\n"
txt_content += f"Dialogue: {s.get('dialogue_myanmar')}\n"
txt_content += f"Image Prompt: {s.get('image_prompt_en')}\n"
txt_content += f"Motion Prompt: {s.get('img_to_video_prompt_en')}\n\n"
st.download_button(
label="📄 Text (.txt) ဖြင့် ဒေါင်းလုဒ်ဆွဲမည်",
data=txt_content,
file_name="script_flow_ai.txt",
mime="text/plain",
use_container_width=True
)

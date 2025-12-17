import streamlit as st
import base64
import os
import time
import random
from pathlib import Path
# Make sure this import works in your environment
from Utils.GenerateAnswer import multimodal_search

# --- 1. CONFIG & ASSETS ---
st.set_page_config(
    page_title="Spider-Verse Chat",
    page_icon="🕷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CACHING
@st.cache_data
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# LOAD ASSETS

BASE_DIR = Path(__file__).resolve().parent
EMB_PATH = BASE_DIR / "Clip_Embeddings" / "panel_clip_embeddings.pt"
LOGO_PATH = BASE_DIR / "Utils" / "logo.png"
BG_PATH = BASE_DIR / "Utils" / "comic_bg.jpg"

logo_base64 = get_base64_image(LOGO_PATH)
bg_base64 = get_base64_image(BG_PATH)

# --- 2. THE "COMIC ENGINE" CSS ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Roboto+Condensed:wght@400;700&display=swap');

    :root {{
        --red: #E62429;
        --dark-red: #8B0000;
        --blue: #2099FF;
        --green: #43A047;
        --purple: #9C27B0;
        --yellow: #FFEB3B;
        --border: 3px solid black;
        --shadow: 5px 5px 0px black;
    }}

    .stApp {{
        background-color: #ffffff;
        {'background-image: url("data:image/jpeg;base64,' + bg_base64 + '");' if bg_base64 else ''}
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--dark-red);
        border-right: var(--border);
    }}
    section[data-testid="stSidebar"] h1 {{
        color: white !important;
        font-family: 'Bangers', cursive;
        text-shadow: 2px 2px 0px black;
        font-size: 2rem;
        margin-bottom: 20px;
    }}
    section[data-testid="stSidebar"] .stButton button {{
        width: 100%;
        background-color: #F0F0F0 !important;
        color: black !important;
        border: 2px solid black !important;
        border-radius: 0px !important;
        font-family: 'Bangers', cursive !important;
        font-size: 1.2rem !important;
        text-transform: uppercase;
        margin-bottom: 8px;
        box-shadow: 3px 3px 0px rgba(0,0,0,0.5) !important;
    }}
    section[data-testid="stSidebar"] .stButton button:first-child {{
        background-color: var(--blue) !important;
        color: white !important;
        border: 2px solid white !important;
    }}

    /* Header Styling */
    .comic-header {{
        background: linear-gradient(90deg, var(--red), var(--dark-red));
        padding: 15px;
        border: var(--border);
        box-shadow: var(--shadow);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .comic-header h1 {{
        color: white;
        font-family: 'Bangers', cursive;
        font-size: 3rem;
        margin: 0;
        text-shadow: 3px 3px 0px black;
        letter-spacing: 1px;
    }}

    /* Chat Styling */
    div[data-testid="stChatMessage"][class*="user"] .stChatMessageContent {{
        background-color: var(--blue);
        color: white; /* Changed from red to white for readability */
        border: var(--border);
        border-radius: 0px;
        box-shadow: 4px 4px 0px black;
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: bold;
    }}

    div[data-testid="stChatMessage"][class*="assistant"] .stChatMessageContent {{
        background-color: white;
        border: 3px solid var(--green);
        border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px;
        box-shadow: -4px 4px 0px var(--purple);
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: bold;
        padding: 20px;
    }}

    /* Input Styling */
    .stChatInput textarea {{
        border-radius: 0px !important;
        font-family: 'Roboto Condensed', sans-serif;
        font-weight: bold;
    }}
    div[data-testid="stChatInput"] button {{
        color: var(--red);
    }}
    
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}

</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    if logo_base64:
        st.markdown(f'<img src="data:image/png;base64,{logo_base64}" style="width:100%; margin-bottom:20px; border:3px solid black; box-shadow: 3px 3px 0px black;">', unsafe_allow_html=True)
    
    st.markdown("<h1>DAILY BUGLE DATA</h1>", unsafe_allow_html=True)
    if st.button("➕ NEW MISSION"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("<p style='color: white; font-family: Roboto Condensed; text-align: center;'>SYSTEM STATUS: ONLINE</p>", unsafe_allow_html=True)

# --- 4. MAIN CONTENT ---
st.markdown("""
<div class="comic-header">
    <h1>SPIDER-MAN VS MYSTERIO COMIC CHAT</h1>
</div>
""", unsafe_allow_html=True)

# --- 5. CHAT LOGIC ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UPDATED DISPLAY LOOP ---
# This loop now checks if a message has text ("content") or an image ("image") or both
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🕷️" if message["role"] == "assistant" else "👤"):
        # Display Text
        if "content" in message and message["content"]:
            st.markdown(message["content"])
        
        # Display Image (If present in the message)
        if "image" in message and message["image"]:
            # Check if file exists to avoid ugly errors
            if os.path.exists(message["image"]):
                st.image(message["image"], caption="Archived Comic Panel", use_container_width=True)
            else:
                st.error(f"Error loading panel: {message['image']}")

# --- INPUT HANDLING ---
if prompt := st.chat_input("Ask about the comic..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Assistant Response (Logic for Image Retrieval)
    with st.chat_message("assistant", avatar="🕷️"):
        status_text = st.empty()
        status_text.markdown("*Accessing Multiverse Archives...*")
        
        try:
            # --- CALL YOUR FUNCTION ---
            mode, results = multimodal_search(prompt) 
            
            status_text.empty()

            if results and len(results) > 0:
                first_result = results[0]

                # --- CASE 1: RESPONSE IS TEXT (Image -> Text) ---
                if 'db_text' in first_result:
                    response_text = first_result['db_text']
                    
                    # Update UI
                    st.markdown(f"**Analysis:** {response_text}")
                    
                    # Save to History
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                else:
                    # Base path for your panels
                    base_path = Path(__file__).resolve().parent.parent / "panels"
                    
                    # Clean filename logic
                    filename = first_result['image_path'].replace("\\", "/").split("/")[-1]
                    full_image_path = os.path.join(base_path, filename)
                    
                    # Update UI
                    st.markdown(f"Found a matching panel for: **{prompt}**")
                    st.image(full_image_path, caption=f"Source: {filename}", use_container_width=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Found a matching panel for: **{prompt}**",
                        "image": full_image_path
                    })
            else:
                response = "No matching panels found in the archives."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Force navigation to Chatbot UI
st.switch_page("pages/Chatbot.py")

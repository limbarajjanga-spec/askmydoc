# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# On Streamlit Cloud — reads from secrets.toml
# On local machine — reads from .env
try:
    import streamlit as st
    ANTHROPIC_API_KEY = (
        st.secrets.get("ANTHROPIC_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )
except Exception:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is missing.")
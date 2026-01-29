# dashboard.py (Updated with Bottom Chat Input) - Part 1
# Imports, Configuration, and Authentication Check

import os
import streamlit as st
import openai
import html
from datetime import datetime
import time
import json
import streamlit.components.v1 as components
# ADD THESE IMPORTS FOR DOWNLOAD FUNCTIONALITY
import io
import base64
# PDF imports instead of DOCX
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from html import escape
import unicodedata
import textwrap

# -- Import your backend utility functions
from assistant import run_assistant, handle_file_upload, validate_file_exists, _handle_function_call
from core_assistant import wait_for_idle_thread, validate_thread_exists

# -- Import improved session management (cost-optimized, resilient)
from session_manager import (
    validate_thread_exists_cached,
    invalidate_thread_cache,
    get_optimized_context,
    create_checkpoint_summary,
    get_silent_instructions,
    is_complex_query,
    get_service_health,
    get_session_metrics,
    should_checkpoint,
    get_circuit_breaker,
)

# -- Import authentication module
from auth_popup import (
    show_auth_popup, logout_user, get_user_info, 
    check_access_level, sync_with_cloud_db,
    validate_session, log_user_action, get_access_restrictions
)

# --- CONFIG ---
CHECKPOINT_FREQ = 8
SUMMARY_WINDOW = 16
CONTEXT_WINDOW = 2
MAX_WAIT = 420  # 7 minutes
POLLING_INTERVAL = 1.0
RETRY_ATTEMPTS = 3

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="sAİmone - MedAffairs Assistant", 
    page_icon="🧬", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# AUTHENTICATION & SESSION (Azure Easy Auth + Disclaimers)
# ==========================================================
if not validate_session():
    if show_auth_popup():
        st.rerun()  # user just completed sign-in/disclaimers; reload once
    else:
        st.stop()

# User info (safe to use anywhere below)
user_info = get_user_info() or {}
user_name = user_info.get("name") or user_info.get("email") or "User"
tenant_id = user_info.get("tenant_id", "?")
object_id = user_info.get("object_id", "?")

# One-time session start log (idempotent)
if "session_started" not in st.session_state:
    log_user_action("session_start", {
        "name": user_name,
        "email": user_info.get("email"),
        "tenant_id": tenant_id,
        "object_id": object_id,
        "module": "dashboard.py"
    })
    st.session_state["session_started"] = True

# Optional: sync profile to your cloud DB (stub in auth_popup.py)
sync_with_cloud_db()

# Access control switches (optional, used later to hide features)
ACCESS = get_access_restrictions()

# Sidebar session panel removed - logout available at bottom of page


# --- API KEYS ---
def get_api_keys():
    try:
        return {
            'openai_api_key': st.secrets["OPENAI_API_KEY"],
            'assistant_id': st.secrets["ASSISTANT_ID"],
            'tavily_api_key': st.secrets.get("TAVILY_API_KEY", "")
        }
    except Exception:
        openai_key = os.getenv("OPENAI_API_KEY")
        assistant_id = os.getenv("ASSISTANT_ID")
        if not openai_key or not assistant_id:
            st.error("🔑 Missing API keys! Set OPENAI_API_KEY and ASSISTANT_ID in secrets or environment variables.")
            st.stop()
        return {
            'openai_api_key': openai_key,
            'assistant_id': assistant_id,
            'tavily_api_key': os.getenv("TAVILY_API_KEY", "")
        }

api_keys = get_api_keys()
openai.api_key = api_keys['openai_api_key']
ASSISTANT_ID = api_keys['assistant_id']

st.markdown("""
<style>
/* -----------------------------------------------------------
   FONTS & THEME TOKENS
----------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');

:root{
  --navy-dark:#0d1b2a; --navy-medium:#1b263b; --navy-light:#415a77; --navy-lighter:#778da9;
  --black-primary:#000; --black-secondary:#1a1a1a; --black-tertiary:#2d2d2d;
  --white-primary:#fff; --white-secondary:#f0f0f0;
  --light-blue:#87ceeb; --accent-blue:#4a90e2; --accent-cyan:#00bcd4;
  --text-primary:#fff; --text-secondary:#b0c4de; --text-muted:#778da9;
  --shadow-soft:rgba(0,0,0,.3); --shadow-strong:rgba(0,0,0,.5);
}

/* Base app */
.stApp{
  background:linear-gradient(135deg,var(--black-primary) 0%,var(--navy-dark) 100%);
  font-family:'Source Sans Pro',sans-serif; color:var(--text-primary);
}

/* -----------------------------------------------------------
   FIXED BOTTOM INPUT
----------------------------------------------------------- */
.fixed-bottom-input{
  position:fixed; bottom:0; left:21rem; right:0;
  background:linear-gradient(180deg,var(--navy-dark) 0%,var(--black-primary) 100%);
  border-top:3px solid var(--navy-light); padding:1rem 2rem; z-index:999;
  box-shadow:0 -4px 20px var(--shadow-strong);
}
@media (max-width:768px){ .fixed-bottom-input{ left:0; } }

.main-content-with-bottom-input{ padding-bottom:200px; } /* keep content above footer input */

.bottom-chat-textarea textarea{
  background:var(--black-secondary)!important; border:2px solid var(--navy-light)!important;
  border-radius:12px!important; color:var(--text-primary)!important;
  font-family:'Source Sans Pro',sans-serif!important; font-size:1rem!important; line-height:1.5!important;
  min-height:60px!important; max-height:200px!important; resize:vertical!important;
}
.bottom-chat-textarea textarea:focus{
  border-color:var(--light-blue)!important; box-shadow:0 0 0 3px rgba(135,206,235,.3)!important;
}

/* Bottom SEND: inherits global button look, just size/transform tweaks */
.bottom-send-button .stButton > button{
  height:60px!important; width:100%!important; padding:.75rem 2rem!important;
  text-transform:uppercase!important; letter-spacing:.5px!important; font-weight:600!important;
}

/* -----------------------------------------------------------
   HEADER
----------------------------------------------------------- */
.main-header{
  background:linear-gradient(135deg,var(--navy-dark) 0%,var(--navy-medium) 50%,var(--black-secondary) 100%);
  padding:3rem 2rem; margin:-1rem -1rem 2rem -1rem; text-align:center; border-radius:0 0 25px 25px;
  box-shadow:0 8px 32px var(--shadow-strong); position:relative; overflow:hidden; border:1px solid var(--navy-light);
}
.main-header::before{
  content:''; position:absolute; inset:0;
  background:url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="g" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23g)"/></svg>');
  opacity:.2;
}
.main-title{
  color:var(--white-primary); font-family:'Playfair Display',serif; font-size:3.5rem; font-weight:700;
  margin-bottom:.5rem; text-shadow:2px 2px 4px var(--shadow-strong); letter-spacing:.5px; position:relative; z-index:1;
}
.main-subtitle{ color:var(--light-blue); font-size:1.4rem; font-weight:300; font-style:italic; position:relative; z-index:1; }

/* -----------------------------------------------------------
   TABLES & MARKDOWN
----------------------------------------------------------- */
.stMarkdown table{
  background:var(--navy-medium)!important; border-radius:8px; overflow:hidden;
  box-shadow:0 4px 16px var(--shadow-soft); border:1px solid var(--navy-light);
}
.stMarkdown th,.stMarkdown td{
  background:var(--black-secondary)!important; color:var(--text-primary)!important;
  border:1px solid var(--navy-light)!important; padding:12px!important; font-family:'Source Sans Pro',sans-serif!important;
}
.stMarkdown th{ background:var(--navy-light)!important; font-weight:600!important; color:var(--white-primary)!important; }
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3{ color:var(--text-primary); font-family:'Playfair Display',serif; font-weight:600; }
.stMarkdown h3{ border-bottom:2px solid var(--navy-light); padding-bottom:.5rem; margin-bottom:1rem; }
.stMarkdown ul li::marker,.stMarkdown ol li::marker{ color:var(--light-blue)!important; }
.stMarkdown ul li,.stMarkdown ol li{ color:var(--text-primary)!important; }
hr.msg-divider{ border:none; border-top:2px solid var(--navy-light); margin:2em 0 1.5em 0; opacity:.6; }

/* -----------------------------------------------------------
   GLOBAL BUTTONS (main area + downloads + link-buttons)
   -> unified look & states
----------------------------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stLinkButton > button{
  background:linear-gradient(135deg,var(--accent-blue) 0%,var(--accent-cyan) 100%)!important;
  color:var(--white-primary)!important; border:none!important; border-radius:12px!important;
  padding:.7rem 1.2rem!important; font-weight:600!important; letter-spacing:.3px!important;
  box-shadow:0 4px 16px var(--shadow-soft)!important; transition:all .25s ease!important;
  text-transform:none!important; font-size:1rem!important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > button:hover{
  transform:translateY(-2px)!important; box-shadow:0 8px 24px var(--shadow-strong)!important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled,
.stLinkButton > button:disabled{
  opacity:.55!important; cursor:not-allowed!important; filter:grayscale(.15);
}

/* Icon-only mini actions (wrap your st.button in <div class="icon-button">...</div>) */
.icon-button .stButton > button{
  width:36px!important; height:36px!important; padding:0!important; border-radius:10px!important;
}

/* Danger variant (wrap the buttons in <div class="danger">...</div>) */
.danger .stButton > button,
.danger .stDownloadButton > button,
.danger .stLinkButton > button{
  background:linear-gradient(135deg,#dc3545,#c82333)!important; border:2px solid #b21f2d!important;
}

/* -----------------------------------------------------------
   SIDEBAR
----------------------------------------------------------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--black-primary) 0%,var(--navy-dark) 100%);
  border-right:3px solid var(--navy-light); box-shadow:4px 0 16px var(--shadow-soft);
}
[data-testid="stSidebar"] > div:first-child{ padding-top:2rem; }

.sidebar-section{
  margin-bottom:1.5rem; padding:1.2rem; background:var(--black-secondary);
  border-radius:12px; border:1px solid var(--navy-light); box-shadow:0 4px 12px var(--shadow-soft); position:relative;
}
.sidebar-section::before{
  content:''; position:absolute; top:0; left:0; width:4px; height:100%;
  background:linear-gradient(180deg,var(--light-blue) 0%,var(--accent-blue) 100%); border-radius:12px 0 0 12px;
}
.sidebar-title{ color:var(--light-blue); font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:600; margin-bottom:.8rem; padding-left:.5rem; }

/* Sidebar buttons inherit unified styles, with subtle container tweaks */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] .stLinkButton > button{
  width:100%!important; min-height:42px!important; border-radius:8px!important;
  border:2px solid var(--navy-light)!important;
  background:linear-gradient(135deg,var(--black-tertiary) 0%,var(--navy-dark) 100%)!important;
  color:var(--white-primary)!important; font-size:.95rem!important; font-weight:600!important;
  box-shadow:0 4px 12px var(--shadow-soft),
             inset 0 1px 2px rgba(255,255,255,.1),
             inset 0 -1px 2px rgba(0,0,0,.3)!important;
  position:relative!important; overflow:hidden!important; text-transform:none!important;
}
[data-testid="stSidebar"] .stButton > button::before,
[data-testid="stSidebar"] .stDownloadButton > button::before,
[data-testid="stSidebar"] .stLinkButton > button::before{
  content:''; position:absolute; inset:0; left:-100%;
  background:linear-gradient(90deg,transparent,rgba(135,206,235,.2),transparent); transition:left .5s ease;
}
[data-testid="stSidebar"] .stButton > button:hover::before,
[data-testid="stSidebar"] .stDownloadButton > button:hover::before,
[data-testid="stSidebar"] .stLinkButton > button:hover::before{ left:100%; }
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover,
[data-testid="stSidebar"] .stLinkButton > button:hover{
  background:linear-gradient(135deg,var(--navy-light) 0%,var(--navy-medium) 100%)!important;
  border-color:var(--light-blue)!important; transform:translateY(-2px)!important;
  box-shadow:0 6px 20px var(--shadow-strong),
             inset 0 1px 3px rgba(255,255,255,.15),
             inset 0 -1px 3px rgba(0,0,0,.2)!important;
}

/* Sidebar readability */
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stFileUploader label,
[data-testid="stSidebar"] .stTextArea label{ color:var(--light-blue)!important; font-weight:500!important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] div[data-testid="stText"]{ color:var(--text-primary)!important; }
[data-testid="stSidebar"] .stFileUploader > div{ color:var(--text-primary)!important; background:var(--black-tertiary)!important; border-color:var(--navy-light)!important; }
[data-testid="stSidebar"] .stFileUploader small{ color:var(--text-secondary)!important; }
[data-testid="stSidebar"] .sidebar-section p,
[data-testid="stSidebar"] .sidebar-section div,
[data-testid="stSidebar"] .sidebar-section span{ color:var(--text-primary)!important; }

/* Sidebar scrollbars */
[data-testid="stSidebar"]::-webkit-scrollbar{ width:6px; }
[data-testid="stSidebar"]::-webkit-scrollbar-track{ background:var(--black-secondary); border-radius:3px; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb{ background:var(--navy-light); border-radius:3px; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover{ background:var(--light-blue); }

/* -----------------------------------------------------------
   FORMS / INPUTS (main area)
----------------------------------------------------------- */
.stTextInput > div > div > input{
  background:var(--black-secondary)!important; border:2px solid var(--navy-light)!important;
  border-radius:8px!important; color:var(--text-primary)!important; padding:.75rem!important;
}
.stTextInput > div > div > input:focus{
  border-color:var(--light-blue)!important; box-shadow:0 0 0 2px rgba(135,206,235,.3)!important;
}
.stTextArea > div > div > textarea{
  background:var(--black-secondary)!important; border:2px solid var(--navy-light)!important;
  border-radius:12px!important; color:var(--text-primary)!important; padding:1rem!important;
  font-size:1rem!important; line-height:1.5!important;
}
.stTextArea > div > div > textarea:focus{
  border-color:var(--light-blue)!important; box-shadow:0 0 0 2px rgba(135,206,235,.3)!important;
}
.stFileUploader > div{
  background:var(--black-secondary)!important; border:2px dashed var(--navy-light)!important;
  border-radius:12px!important; padding:1.5rem!important; transition:all .2s ease!important; color:var(--text-primary)!important;
}
.stFileUploader > div:hover{ border-color:var(--light-blue)!important; background:var(--navy-medium)!important; }

/* -----------------------------------------------------------
   FEEDBACK, MISC
----------------------------------------------------------- */
.stSuccess{ background:rgba(74,144,226,.15)!important; border:1px solid var(--accent-blue)!important; border-radius:8px!important; color:var(--text-primary)!important; }
.stError{ background:rgba(220,53,69,.15)!important; border:1px solid #dc3545!important; border-radius:8px!important; color:var(--text-primary)!important; }
.stWarning{ background:rgba(255,193,7,.15)!important; border:1px solid #ffc107!important; border-radius:8px!important; color:var(--text-primary)!important; }
.stInfo{ background:rgba(135,206,235,.15)!important; border:1px solid var(--light-blue)!important; border-radius:8px!important; color:var(--text-primary)!important; }

.stSpinner > div{ border-color:var(--light-blue) transparent transparent transparent!important; }
.stSpinner{ color:var(--light-blue)!important; }
.stColumn{ background:transparent; padding:.5rem; }

.chat-message{ padding:1rem; margin:.5rem 0; border-radius:12px; box-shadow:0 2px 8px var(--shadow-soft); }
.stCaption,[data-testid="stCaption"]{ color:var(--text-secondary)!important; font-style:italic; }

/* Footer */
.footer-text{
  color:var(--text-secondary); text-align:center; font-style:italic; padding:1rem;
  border-top:1px solid var(--navy-light); margin-top:2rem; background:var(--black-secondary);
  border-radius:12px; font-size:.9rem;
}

/* Auth banners / access badges */
.user-info-banner{
  background:linear-gradient(135deg,var(--navy-medium) 0%,var(--black-secondary) 100%);
  padding:1rem; border-radius:8px; border:1px solid var(--navy-light); margin-bottom:1rem; text-align:center;
}
.logout-section{ background:var(--black-tertiary); padding:1rem; border-radius:8px; border:1px solid #dc3545; margin-top:1rem; text-align:center; }
.access-level-full{ background:linear-gradient(90deg,#28a745,#20c997); color:#fff; padding:.2rem .6rem; border-radius:12px; font-size:.8rem; font-weight:600; }
.access-level-limited{ background:linear-gradient(90deg,#ffc107,#fd7e14); color:#fff; padding:.2rem .6rem; border-radius:12px; font-size:.8rem; font-weight:600; }
.access-level-view{ background:linear-gradient(90deg,#6c757d,#495057); color:#fff; padding:.2rem .6rem; border-radius:12px; font-size:.8rem; font-weight:600; }
.access-level-admin{ background:linear-gradient(90deg,#dc3545,#c82333); color:#fff; padding:.2rem .6rem; border-radius:12px; font-size:.8rem; font-weight:600; }

/* -----------------------------------------------------------
   RESPONSIVE
----------------------------------------------------------- */
@media (max-width:768px){
  .main-title{ font-size:2.5rem; }
  .main-header{ padding:2rem 1rem; }
  .sidebar-section{ padding:1rem; }
}

/* ---------- STRICT UNIFORM BUTTONS ---------- */
:root{
  --btn-height: 44px;
  --btn-font-size: 15px;
  --btn-padding-x: 1rem;
  --btn-radius: 12px;
}

/* Make ALL button-like widgets identical */
.stButton > button,
.stDownloadButton > button,
.stLinkButton > button{
  height: var(--btn-height) !important;
  min-height: var(--btn-height) !important;
  width: 100% !important;                 /* fills column/sidebar width */
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: .5rem !important;

  padding: 0 var(--btn-padding-x) !important;  /* vertical padding controlled by height */
  border-radius: var(--btn-radius) !important;
  font-size: var(--btn-font-size) !important;
  line-height: calc(var(--btn-height) - 4px) !important; /* avoids odd text baselines */
  white-space: nowrap !important;           /* no multi-line (prevents random taller buttons) */
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  text-transform: none !important;          /* keep case consistent */
  letter-spacing: .3px !important;
}

/* Keep sidebar using exactly the same sizing — only width differs (already 100%) */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] .stLinkButton > button{
  height: var(--btn-height) !important;
  min-height: var(--btn-height) !important;
  font-size: var(--btn-font-size) !important;
  padding: 0 var(--btn-padding-x) !important;
  border-radius: var(--btn-radius) !important;
}

/* Icon-only variant (wrap your st.button with <div class="icon-button">...</div>) */
.icon-button .stButton > button{
  width: 36px !important;
  min-width: 36px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 !important;
  border-radius: 10px !important;
  font-size: 16px !important;   /* emoji/icons look crisper slightly larger */
  line-height: 34px !important;
}

/* Optional: keep the bottom SEND a tad taller.
   Remove this if you want it identical to all others. */
.bottom-send-button .stButton > button{
  height: 60px !important;
  min-height: 60px !important;
  font-size: 16px !important;
  letter-spacing: .5px !important;
}
/* 1) make the wrapper a block so it doesn't mess with layout */
.qa-tip { display: block; }

/* 2) pass hover from the wrapper to the actual button (restores shading) */
.qa-tip:hover .stButton > button,
.qa-tip:focus-within .stButton > button {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px var(--shadow-strong) !important;
}

/* if you use the sidebar shimmer stripe (::before), animate it on wrapper hover too */
[data-testid="stSidebar"] .qa-tip:hover .stButton > button::before {
  left: 100% !important;
}

/* optional: match your sidebar hover background when wrapper is hovered */
[data-testid="stSidebar"] .qa-tip:hover .stButton > button {
  background: linear-gradient(135deg, var(--navy-light) 0%, var(--navy-medium) 100%) !important;
  border-color: var(--light-blue) !important;
}

</style>
""", unsafe_allow_html=True)

# dashboard.py - Part 2
# Download Helper Functions

# --- DOWNLOAD HELPER FUNCTIONS ---
# --- DOWNLOAD HELPER FUNCTIONS ---
# --- DOWNLOAD HELPER FUNCTIONS ---
def generate_pdf_download():
    """PDF generator: exact session order, robust links, bullets/numbered lists, tables with repeating headers.
    Renders inline citation tokens as superscripts and *sanitizes/balances tags* to avoid paraparser errors."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, ListFlowable, ListItem, KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.pdfgen import canvas
        import re, io
        from html import escape
        from datetime import datetime

        # ---------- Pagination ----------
        class NumberedCanvas(canvas.Canvas):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._saved_page_states = []
            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()
            def save(self):
                total = len(self._saved_page_states)
                for state in self._saved_page_states:
                    self.__dict__.update(state)
                    self.setFont("Helvetica", 9)
                    self.setFillColor(colors.grey)
                    self.drawRightString(A4[0] - 0.75*inch, 0.75*inch, f"Page {self._pageNumber} of {total}")
                    super().showPage()
                super().save()

        # ---------- Text & link utils ----------
        def ensure_http(u: str) -> str:
            u = (u or '').strip()
            if not u:
                return u
            if u.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
                return u
            return 'https://' + u

        # Render tokens like 【24:12web_search】 or ■24:12†web_search■ as superscript [24:12]
        def prettify_inline_citations(text: str) -> str:
            if not text:
                return ""
            t = str(text)

            def _label_from_payload(payload: str) -> str:
                m = re.search(r'(\d+(?::\d+)?)', payload)
                if m:
                    return m.group(1)
                payload = re.sub(r'\s+', '', payload)
                return payload[:10] + ('…' if len(payload) > 10 else '')

            def _repl_fw(m):
                raw = m.group(1)
                label = _label_from_payload(raw)
                return f'<super><font color="#777777">[{label}]</font></super>'
            t = re.sub(r'【([^】]{1,200})】', _repl_fw, t)

            def _repl_blk(m):
                raw = m.group(1)
                label = _label_from_payload(raw)
                return f'<super><font color="#777777">[{label}]</font></super>'
            t = re.sub(r'■\s*\d+:\d+\s*†[^■\s]+■', lambda m: _repl_blk(re.match(r'■(.*)■', m.group(0))), t)
            t = re.sub(r'■([^■]{1,120})■', _repl_blk, t)

            t = re.sub(r'\s+([.,;:!?])', r'\1', t)
            t = re.sub(r' {2,}', ' ', t)
            t = re.sub(r'\n{3,}', '\n\n', t)
            return t

        def extract_and_format_links(text: str) -> str:
            if not text:
                return ""
            # Mask any pre-existing <link>…</link> so we never double-wrap
            kl_map = {}
            def _mask(m):
                key = f'§§KLINK§{len(kl_map)}§§'
                kl_map[key] = m.group(0)
                return key
            t = re.sub(r'<link\b[^>]*>.*?</link>', _mask, text, flags=re.I|re.S)

            # Markdown [label](url)
            def repl_md(m):
                label = re.sub(r'[<>]', '', m.group(1)).strip()
                url = ensure_http(m.group(2).strip())
                return f'§LINK§{url}§TEXT§{label}§END§'
            t = re.sub(r'\[([^\[\]]+)\]\(([^\(\)]+)\)', repl_md, t)

            # Raw URLs
            def repl_raw(m):
                url = m.group(0)
                ctx = t[max(0, m.start()-12):m.start()]
                if '§LINK§' in ctx or 'href="' in ctx:
                    return url
                disp = url if len(url) <= 50 else (url[:47] + '...')
                return f'§LINK§{url}§TEXT§{disp}§END§'
            t = re.sub(r'(https?://[^\s<>"\'\)\]]+)', repl_raw, t)

            # Placeholders -> <link>
            def repl_ph(m):
                safe_url = escape(ensure_http(m.group(1)), quote=True)
                return f'<link href="{safe_url}" color="blue">{escape(m.group(2))}</link>'
            t = re.sub(r'§LINK§([^§]+)§TEXT§([^§]+)§END§', repl_ph, t)

            # Unmask originals
            if kl_map:
                t = re.sub(r'§§KLINK§\d+§§', lambda mm: kl_map.get(mm.group(0), mm.group(0)), t)
            return t

        # NEW: escape all '<' and '>' outside a safe whitelist, normalize <br/>, and balance tags
        _SAFE_TAG_RE = re.compile(r'(</?(?:link|b|i|u|br/?|super|font)\b[^>]*>)', re.I)
        def sanitize_whitelist_and_balance(text: str) -> str:
            if not text:
                return ""
            parts = re.split(_SAFE_TAG_RE, text)
            out = []
            for p in parts:
                if not p:
                    continue
                if _SAFE_TAG_RE.fullmatch(p):
                    # normalize <br> -> <br/>
                    if re.match(r'<br\s*>', p, re.I):
                        out.append('<br/>')
                    else:
                        out.append(p)
                else:
                    # escape everything else
                    p = html.unescape(p)
                    p = p.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    out.append(p)
            t = ''.join(out)

            # balance tags (append missing closers at the end)
            for tag in ('b','i','u','super','font','link'):
                open_re  = re.compile(rf'<{tag}\b[^>]*>', re.I)
                close_re = re.compile(rf'</{tag}\s*>', re.I)
                opens  = len(open_re.findall(t))
                closes = len(close_re.findall(t))
                if opens > closes:
                    t += ('</' + tag + '>') * (opens - closes)
            return t

        def clean_text_with_links(text: str) -> str:
            if not text:
                return ""
            t = re.sub(r'<br\s*/?>', ' ', str(text), flags=re.I)
            t = prettify_inline_citations(t)   # render citation tokens as superscripts
            t = extract_and_format_links(t)    # then convert markdown/raw URLs to <link>…
            # strip markdown header sigils; headings handled separately
            t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)

            # bold
            def repl_b(m):
                inner = m.group(1)
                if '<link' in inner:
                    return f'**{inner}**'
                return f'<b>{inner}</b>'
            t = re.sub(r'\*\*([^\*]+)\*\*', repl_b, t)

            # italic
            def repl_i(m):
                inner = m.group(1)
                if '<link' in inner:
                    return f'*{inner}*'
                return f'<i>{inner}</i>'
            t = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', repl_i, t)

            # final tidy + harden for paraparser
            t = re.sub(r'\n{3,}', '\n\n', t)
            t = re.sub(r'\s+([.,;:!?])', r'\1', t)
            t = sanitize_whitelist_and_balance(t)
            return t.strip()

        # ---------- Block parser (headings, lists, tables, paragraphs) ----------
        bullet_rx_unordered = re.compile(r'^\s*(?:[-•*·])\s+')
        bullet_rx_ordered = re.compile(r'^\s*(\d+)[\.\)]\s+')

        def parse_blocks_preserving_lists(text: str):
            """Yield ('h1'|'h2'|'h3'|'hr'|'table'|'olist'|'ulist'|'para', payload)."""
            lines = text.splitlines()
            i = 0
            HDR = re.compile(r'^\s*\|.*\|\s*$')
            SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$')
            while i < len(lines):
                ln = lines[i].rstrip()

                if not ln.strip():
                    i += 1; continue

                if ln.startswith('### '):
                    yield ('h3', ln[4:].strip()); i += 1; continue
                if ln.startswith('## '):
                    yield ('h2', ln[3:].strip()); i += 1; continue
                if ln.startswith('# '):
                    yield ('h1', ln[2:].strip()); i += 1; continue

                if re.match(r'^\s*-{3,}\s*$', ln):
                    yield ('hr', None); i += 1; continue

                if i+1 < len(lines) and HDR.match(lines[i]) and SEP.match(lines[i+1]):
                    j = i+2
                    while j < len(lines) and '|' in lines[j]:
                        j += 1
                    yield ('table', '\n'.join(lines[i:j])); i = j; continue

                if bullet_rx_unordered.match(ln) or bullet_rx_ordered.match(ln):
                    ordered = bool(bullet_rx_ordered.match(ln))
                    start = int(bullet_rx_ordered.match(ln).group(1)) if ordered else 1
                    items, j = [], i
                    while j < len(lines):
                        cur = lines[j]
                        if ordered and bullet_rx_ordered.match(cur):
                            items.append(bullet_rx_ordered.sub('', cur, count=1).strip()); j += 1
                        elif (not ordered) and bullet_rx_unordered.match(cur):
                            items.append(bullet_rx_unordered.sub('', cur, count=1).strip()); j += 1
                        else:
                            break
                    yield ('olist' if ordered else 'ulist', (start, items)); i = j; continue

                j, buff = i, []
                while j < len(lines):
                    cur = lines[j]
                    if (not cur.strip()
                        or cur.startswith('#')
                        or bullet_rx_unordered.match(cur) or bullet_rx_ordered.match(cur)
                        or (j+1 < len(lines) and HDR.match(cur) and SEP.match(lines[j+1]))
                        or re.match(r'^\s*-{3,}\s*$', cur)):
                        break
                    buff.append(cur); j += 1
                yield ('para', '\n'.join(buff).strip()); i = j if j > i else i+1

        # ---------- Tables ----------
        def create_table(md_text, styles):
            lines = md_text.strip().split('\n')
            rows = []
            for line in lines:
                if all(c in '|-=: ' for c in line):
                    continue
                cells = [c.strip() for c in line.split('|')]
                if cells and cells[0] == '': cells = cells[1:]
                if cells and cells[-1] == '': cells = cells[:-1]
                if cells: rows.append(cells)
            if not rows:
                return None
            n = max(len(r) for r in rows)
            rows = [r + ['']*(n - len(r)) for r in rows]

            total_w = 6.5 * inch
            min_w = 0.9 * inch
            maxlen = [0]*n
            for r in rows:
                for i, c in enumerate(r):
                    maxlen[i] = max(maxlen[i], len(c))
            total_len = sum(maxlen) or n
            col_w = [max((L/total_len)*total_w, min_w) for L in maxlen]
            s = sum(col_w)
            if s > total_w:
                scale = total_w / s
                col_w = [w*scale for w in col_w]

            wrapped = []
            for r_i, r in enumerate(rows):
                wr = []
                for cell in r:
                    txt = clean_text_with_links(cell)
                    if r_i == 0:
                        wr.append(Paragraph(f"<b>{txt}</b>", styles['TableHeader']))
                    else:
                        wr.append(Paragraph(txt, styles['TableCell']))
                wrapped.append(wr)

            t = Table(wrapped, colWidths=col_w, repeatRows=1, splitByRow=1)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1b263b')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,0), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('LINEBELOW', (0,0), (-1,0), 2, colors.HexColor('#1b263b')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            return t

        # ---------- Lists ----------
        def build_list(start, items, ordered, styles):
            if not items: return None
            paras = [ListItem(Paragraph(clean_text_with_links(x), styles['BulletText']), leftIndent=20)
                     for x in items if x.strip()]
            return ListFlowable(paras, bulletType='1', start=start) if ordered else ListFlowable(paras, bulletType='bullet')

        # ---------- PDF scaffold ----------
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=0.75*inch, leftMargin=0.75*inch,
            topMargin=1.0*inch, bottomMargin=1.0*inch
        )
        styles = getSampleStyleSheet()
        story = []

        styles.add(ParagraphStyle('CoverTitle', fontSize=32, leading=36,
                                  textColor=colors.HexColor('#0d1b2a'),
                                  alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle('SectionTitle', fontSize=18, leading=22,
                                  textColor=colors.HexColor('#1b263b'),
                                  spaceAfter=16, spaceBefore=24, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle('H2', fontSize=14, leading=18,
                                  textColor=colors.HexColor('#415a77'),
                                  spaceAfter=10, spaceBefore=14, fontName='Helvetica-Bold',
                                  keepWithNext=True))
        styles.add(ParagraphStyle('H3', fontSize=12, leading=16,
                                  textColor=colors.HexColor('#415a77'),
                                  spaceAfter=8, spaceBefore=10, fontName='Helvetica-Bold',
                                  keepWithNext=True))
        styles.add(ParagraphStyle('MessageText', fontSize=10, leading=15,
                                  textColor=colors.black, alignment=TA_LEFT, spaceAfter=10))
        styles.add(ParagraphStyle('BulletText', fontSize=10, leading=15,
                                  textColor=colors.black, alignment=TA_LEFT, spaceAfter=4, leftIndent=0))
        styles.add(ParagraphStyle('TableCell', fontSize=8, leading=12,
                                  alignment=TA_LEFT, wordWrap='LTR', splitLongWords=True))
        styles.add(ParagraphStyle('TableHeader', fontSize=9, leading=12,
                                  alignment=TA_LEFT, textColor=colors.whitesmoke))

        # ---------- Cover ----------
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("sAÏmone", styles['CoverTitle']))
        story.append(Paragraph("Medical Affairs Session Report", styles['SectionTitle']))
        story.append(Spacer(1, 1*inch))

        user_name = 'Unknown'
        user_org = 'Unknown'
        if 'user_info' in globals() and user_info:
            user_name = user_info.get('name', 'Unknown')
            user_org = user_info.get('client', 'Unknown')

        try:
            history = list(st.session_state.get('history', []))
        except Exception:
            history = []

        info_data = [
            ['Report Generated', datetime.now().strftime("%B %d, %Y at %H:%M")],
            ['User', user_name],
            ['Organization', user_org],
            ['Total Messages', str(len(history))]
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('ALIGN', (0,0), (0,-1), 'RIGHT'),
            ('ALIGN', (1,0), (1,-1), 'LEFT'),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.grey),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(PageBreak())

        # ---------- Conversation ----------
        story.append(Paragraph("Conversation History", styles['SectionTitle']))
        story.append(Spacer(1, 0.25*inch))

        all_links, link_seen = [], set()
        def collect_links_from(text: str):
            if not text: return
            # remove citation tokens when *parsing links* only
            clean = re.sub(r'【[^】]{1,200}】', '', text)
            clean = re.sub(r'■[^■]{1,120}■', '', clean)
            for label, url in re.findall(r'\[([^\[\]]+)\]\(([^\(\)]+)\)', clean):
                url = ensure_http(url)
                if url not in link_seen:
                    link_seen.add(url); all_links.append((label.strip() or url, url))
            for url in re.findall(r'(https?://[^\s<>"\'\)\]]+)', clean):
                if url not in link_seen:
                    link_seen.add(url)
                    dom = re.search(r'https?://([^/]+)', url)
                    all_links.append((dom.group(1) if dom else url, url))

        num_section_rx = re.compile(r'^\s*\d+\.\s+\S.{0,118}$')

        for i, msg in enumerate(history, 1):
            role = (msg.get('role') or '').lower()
            is_user = (role == 'user')
            header = f"Message {i}: {'User' if is_user else 'sAÏmone Response'}"
            color = colors.HexColor('#4a90e2') if is_user else colors.HexColor('#00bcd4')

            header_table = Table([[header]], colWidths=[6.5*inch])
            header_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), color),
                ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 12),
                ('LEFTPADDING', (0,0), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(header_table)
            story.append(Spacer(1, 0.1*inch))

            content = msg.get('content', '') or ''
            m = re.match(r'\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.*)', content, flags=re.S)
            if m:
                ts, email, content = m.group(1), m.group(2), m.group(3)
                story.append(Paragraph(sanitize_whitelist_and_balance(
                    f"<i>{escape(ts)} • {escape(email)}</i>"),
                    ParagraphStyle('Meta', fontSize=9, textColor=colors.grey)))
                story.append(Spacer(1, 0.05*inch))

            blocks = list(parse_blocks_preserving_lists(content))
            b = 0
            while b < len(blocks):
                kind, payload = blocks[b]

                def render_block_to_flowables(kind, payload):
                    fl = []
                    if kind == 'h1':
                        txt = clean_text_with_links(payload).replace('\n', ' ').strip()
                        fl.append(Paragraph(sanitize_whitelist_and_balance(txt), styles['SectionTitle']))
                    elif kind == 'h2':
                        txt = clean_text_with_links(payload).replace('\n', ' ').strip()
                        fl.append(Paragraph(sanitize_whitelist_and_balance(txt), styles['H2']))
                    elif kind == 'h3':
                        txt = clean_text_with_links(payload).replace('\n', ' ').strip()
                        fl.append(Paragraph(sanitize_whitelist_and_balance(txt), styles['H3']))
                    elif kind == 'hr':
                        fl.append(Spacer(1, 0.08*inch))
                    elif kind == 'table':
                        fl.append(Spacer(1, 0.04*inch))
                        t = create_table(payload, styles)
                        if t: fl.append(t); fl.append(Spacer(1, 0.1*inch))
                    elif kind in ('olist', 'ulist'):
                        start, items = payload
                        lst = build_list(start, items, ordered=(kind=='olist'), styles=styles)
                        if lst: fl.append(lst); fl.append(Spacer(1, 0.06*inch))
                    else:
                        txt = clean_text_with_links(payload)
                        if txt:
                            fl.append(Paragraph(sanitize_whitelist_and_balance(txt), styles['MessageText']))
                            fl.append(Spacer(1, 0.04*inch))
                    return fl

                if (kind in ('h2', 'h3')) or (kind == 'para' and num_section_rx.match((payload or '').strip())):
                    use_style = 'H2' if kind == 'para' else ('H2' if kind=='h2' else 'H3')
                    combined = (payload or '').strip()
                    consume_next = 0
                    if b + 1 < len(blocks) and blocks[b+1][0] == 'para':
                        nxt = (blocks[b+1][1] or '').strip()
                        if 1 <= len(nxt) <= 60 and not re.search(r'[.:!?]$', nxt):
                            combined = combined + ' ' + nxt
                            consume_next = 1
                    head_flow = [
                        Paragraph(sanitize_whitelist_and_balance(
                            clean_text_with_links(combined).replace('\n', ' ').strip()), styles[use_style]),
                        Spacer(1, 0.06*inch)
                    ]
                    if b + 1 + consume_next < len(blocks):
                        next_kind, next_payload = blocks[b+1+consume_next]
                        body_flow = render_block_to_flowables(next_kind, next_payload)
                        story.append(KeepTogether(head_flow + body_flow))
                        b += 2 + consume_next
                        continue
                    else:
                        story.append(KeepTogether(head_flow))
                        b += 1 + consume_next
                        continue

                for fl in render_block_to_flowables(kind, payload):
                    story.append(fl)
                b += 1

            story.append(Spacer(1, 0.2*inch))
            sep = Table([['']], colWidths=[6.5*inch])
            sep.setStyle(TableStyle([('LINEBELOW', (0,0), (-1,0), 1, colors.HexColor('#e0e0e0'))]))
            story.append(sep)
            story.append(Spacer(1, 0.2*inch))

            collect_links_from(content)

        # ---------- References ----------
        if all_links:
            story.append(PageBreak())
            story.append(Paragraph("References", styles['SectionTitle']))
            story.append(Spacer(1, 0.2*inch))
            ref_data = [['#', 'Description', 'Full URL']]
            for idx, (label, url) in enumerate(all_links, 1):
                display = label[:35] + '...' if len(label) > 35 else label
                url_para = Paragraph(sanitize_whitelist_and_balance(
                    f'<link href="{url}" color="blue">{url}</link>'),
                    ParagraphStyle('URLStyle', fontSize=7, textColor=colors.blue, leading=9))
                desc_para = Paragraph(escape(display),
                                      ParagraphStyle('DescStyle', fontSize=8, textColor=colors.black, leading=10))
                ref_data.append([str(idx), desc_para, url_para])

            ref_table = Table(ref_data, colWidths=[0.4*inch, 1.8*inch, 4.3*inch], repeatRows=1)
            ref_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1b263b')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 9),
                ('ALIGN', (0,0), (0,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('LINEBELOW', (0,0), (-1,0), 2, colors.HexColor('#1b263b')),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            story.append(ref_table)

        # ---------- Build ----------
        try:
            doc.build(story, canvasmaker=NumberedCanvas)
        except Exception as e:
            fallback = [Paragraph(
                sanitize_whitelist_and_balance(
                    f"Error generating formatted PDF. Content sanitised. Details: {escape(str(e))}"),
                ParagraphStyle('Fallback', fontSize=10, textColor=colors.black)
            )]
            SimpleDocTemplate(buffer, pagesize=A4,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=1.0*inch, bottomMargin=1.0*inch).build(fallback)

        pdf_data = buffer.getvalue()
        buffer.close()

        user_name_clean = 'anonymous'
        if 'user_info' in globals() and user_info:
            user_name_clean = user_info.get('name', 'anonymous').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sAÏmone_Report_{user_name_clean}_{timestamp}.pdf"
        return pdf_data, filename

    except Exception as e:
        import traceback
        print("Error generating PDF:", e)
        print(traceback.format_exc())
        if 'st' in globals():
            st.error(f"Error generating PDF: {str(e)}")
        return None, None




def create_comprehensive_json_export():
    """Create comprehensive JSON export with all session data and localStorage"""
    try:
        # Get current session data
        current_session = {
            'session_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'session_id': get_session_id(),
                'export_version': '2.0',
                'authenticated_user': user_info if user_info else {}
            },
            'user_profile': {
                'name': user_info['name'] if user_info else '',
                'email': user_info['email'] if user_info else '',
                'role': user_info['role'] if user_info else '',
                'client': user_info['client'] if user_info else '',
                'access_level': user_info['access_level'] if user_info else '',
                'login_time': user_info['login_time'] if user_info else ''
            },
            'thread_info': {
                'thread_id': st.session_state.get('thread_id', ''),
                'user_thread_registry': st.session_state.get('user_thread_registry', {})
            },
            'conversation_data': {
                'history': st.session_state.get('history', []),
                'history_count': len(st.session_state.get('history', [])),
                'checkpoints': st.session_state.get('checkpoints', []),
                'last_checkpoint_turn': st.session_state.get('last_checkpoint_turn', 0)
            },
            'file_management': {
                'uploaded_file_ids': st.session_state.get('uploaded_file_ids', []),
                'file_count': len(st.session_state.get('uploaded_file_ids', []))
            },
            'system_state': {
                'checkpoint_pending': st.session_state.get('checkpoint_pending', False),
                'last_save': st.session_state.get('last_save', 0),
                'audit_log': st.session_state.get('audit_log', [])
            },
            'configuration': {
                'checkpoint_frequency': CHECKPOINT_FREQ,
                'summary_window': SUMMARY_WINDOW,
                'context_window': CONTEXT_WINDOW,
                'max_wait_time': MAX_WAIT,
                'polling_interval': POLLING_INTERVAL,
                'retry_attempts': RETRY_ATTEMPTS
            }
        }
        
        # Add localStorage simulation data
        session_id = get_session_id()
        stored_context_key = f"medical_context_{session_id}"
        
        current_session['localStorage_data'] = {
            'note': 'localStorage data retrieved client-side',
            'expected_keys': [stored_context_key],
            'context_structure': {
                'session_id': session_id,
                'timestamp': 'ISO timestamp',
                'history_count': 'number',
                'checkpoints': 'array',
                'last_checkpoint_turn': 'number',
                'recent_history': 'last 8 messages',
                'thread_id': 'string',
                'user_profile': 'object'
            }
        }
        
        return current_session
        
    except Exception as e:
        st.error(f"Error creating JSON export: {str(e)}")
        return {'error': f'Export failed: {str(e)}', 'timestamp': datetime.now().isoformat()}


def generate_json_download():
    """Generate JSON file and return download data"""
    try:
        # Get comprehensive data
        export_data = create_comprehensive_json_export()
        
        # Convert to JSON string with pretty formatting
        json_string = json.dumps(export_data, indent=2, default=str, ensure_ascii=False)
        json_bytes = json_string.encode('utf-8')
        
        # Generate filename
        user_name = user_info['name'] if user_info else 'anonymous'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sAÏmone_audit_{user_name}_{timestamp}.json"
        
        return json_bytes, filename
        
    except Exception as e:
        st.error(f"Error generating JSON: {str(e)}")
        return None, None

def _fetch_run_steps(thread_id, run_id):
    """Fetch all run steps from the OpenAI API and return parsed tool-call data.

    Returns a list of step dicts, each containing the full tool call details
    (function name, arguments, output/results) with no capping or truncation.
    """
    try:
        steps_page = openai.beta.threads.runs.steps.list(
            thread_id=thread_id,
            run_id=run_id,
            limit=100,
        )
        actions = []
        for step in steps_page.data:
            if step.type == "tool_calls":
                tool_calls_data = []
                for tc in step.step_details.tool_calls:
                    call_entry = {"id": tc.id, "type": tc.type}
                    if tc.type == "function":
                        call_entry["function_name"] = tc.function.name
                        try:
                            call_entry["arguments"] = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except (json.JSONDecodeError, TypeError):
                            call_entry["arguments"] = tc.function.arguments
                        # Full output returned to the agent
                        try:
                            call_entry["output"] = json.loads(tc.function.output) if tc.function.output else None
                        except (json.JSONDecodeError, TypeError):
                            call_entry["output"] = tc.function.output
                    elif tc.type == "file_search":
                        fs = getattr(tc, "file_search", None)
                        call_entry["results"] = []
                        if fs and hasattr(fs, "results") and fs.results:
                            for r in fs.results:
                                call_entry["results"].append({
                                    "file_id": getattr(r, "file_id", None),
                                    "file_name": getattr(r, "file_name", None),
                                    "score": getattr(r, "score", None),
                                    "content": getattr(r, "content", None),
                                })
                    elif tc.type == "code_interpreter":
                        ci = getattr(tc, "code_interpreter", None)
                        if ci:
                            call_entry["input"] = getattr(ci, "input", None)
                            call_entry["outputs"] = []
                            if hasattr(ci, "outputs") and ci.outputs:
                                for o in ci.outputs:
                                    call_entry["outputs"].append({
                                        "type": getattr(o, "type", None),
                                        "text": getattr(o, "text", None) if hasattr(o, "text") else None,
                                        "image_file_id": getattr(o.image, "file_id", None) if hasattr(o, "image") and o.image else None,
                                    })
                    elif tc.type == "web_search":
                        ws = getattr(tc, "web_search", None)
                        if ws and hasattr(ws, "results") and ws.results:
                            call_entry["results"] = []
                            for r in ws.results:
                                call_entry["results"].append({
                                    "url": getattr(r, "url", None),
                                    "title": getattr(r, "title", None),
                                    "snippet": getattr(r, "snippet", None),
                                })
                        else:
                            call_entry["raw"] = str(ws)
                    tool_calls_data.append(call_entry)

                actions.append({
                    "step_id": step.id,
                    "type": step.type,
                    "status": step.status,
                    "created_at": step.created_at,
                    "completed_at": getattr(step, "completed_at", None),
                    "tool_calls": tool_calls_data,
                })
            elif step.type == "message_creation":
                actions.append({
                    "step_id": step.id,
                    "type": step.type,
                    "status": step.status,
                    "created_at": step.created_at,
                    "completed_at": getattr(step, "completed_at", None),
                    "message_id": getattr(step.step_details.message_creation, "message_id", None),
                })
        return actions
    except Exception as e:
        return [{"error": f"Failed to fetch run steps: {str(e)}"}]


def _enrich_last_search_entry(thread_id):
    """Attach full run-steps data to the most recent search_history entry.

    Works for both improved_assistant_run() (which pre-stores _last_run_id/steps)
    and run_assistant() (falls back to querying the thread for the latest run).
    """
    if not st.session_state.get("search_history"):
        return

    entry = st.session_state["search_history"][-1]

    # Check if improved_assistant_run() already captured the data
    run_id = st.session_state.pop("_last_run_id", None)
    run_steps = st.session_state.pop("_last_run_steps", None)

    if not run_id:
        # Fallback: get the most recent completed run from the thread
        try:
            runs = openai.beta.threads.runs.list(
                thread_id=thread_id, limit=1, order="desc"
            )
            if runs.data:
                run_id = runs.data[0].id
        except Exception:
            return

    if run_id and not run_steps:
        run_steps = _fetch_run_steps(thread_id, run_id)

    entry["thread_id"] = thread_id
    entry["run_id"] = run_id
    entry["agent_actions"] = run_steps or []


def generate_search_history_json():
    """Generate a JSON file with the full uncapped search history for the session.

    Each entry includes the user query AND the complete agent actions:
    every tool call (function name, arguments, full output/results) the
    assistant executed during the corresponding run.
    """
    try:
        search_entries = st.session_state.get("search_history", [])

        # Build summary counts
        total_tool_calls = 0
        tool_type_counts = {}
        for entry in search_entries:
            for action in entry.get("agent_actions", []):
                for tc in action.get("tool_calls", []):
                    total_tool_calls += 1
                    fn = tc.get("function_name") or tc.get("type", "unknown")
                    tool_type_counts[fn] = tool_type_counts.get(fn, 0) + 1

        export_payload = {
            "export_metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "session_id": get_session_id(),
                "thread_id": st.session_state.get("thread_id", ""),
                "export_type": "search_history_full",
                "total_user_searches": len(search_entries),
                "total_agent_tool_calls": total_tool_calls,
                "tool_call_breakdown": tool_type_counts,
            },
            "user_profile": {
                "name": user_info.get("name", "") if user_info else "",
                "email": user_info.get("email", "") if user_info else "",
                "role": user_info.get("role", "") if user_info else "",
                "client": user_info.get("client", "") if user_info else "",
                "access_level": user_info.get("access_level", "") if user_info else "",
            },
            "searches": search_entries,
        }
        json_string = json.dumps(export_payload, indent=2, default=str, ensure_ascii=False)
        json_bytes = json_string.encode("utf-8")

        name = user_info.get("name", "anonymous") if user_info else "anonymous"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sAImone_search_history_{name}_{timestamp}.json"

        return json_bytes, filename
    except Exception as e:
        st.error(f"Error generating search history JSON: {str(e)}")
        return None, None


def create_download_button(file_data, filename, button_text, button_key, mime_type):
    """Create a styled download button matching sidebar theme"""
    if file_data and filename:
        # Create base64 encoded data
        b64_data = base64.b64encode(file_data).decode()
        
        # Custom download button with matching sidebar styling
        button_html = f"""
        <div style="margin: 0.5rem 0;">
            <a href="data:{mime_type};base64,{b64_data}" 
               download="{filename}"
               style="
                   display: inline-block;
                   width: 100%;
                   padding: 0.6rem 1rem;
                   background: linear-gradient(135deg, var(--black-tertiary) 0%, var(--navy-dark) 100%);
                   color: var(--white-primary);
                   text-decoration: none;
                   border-radius: 8px;
                   border: 2px solid var(--navy-light);
                   font-family: 'Source Sans Pro', sans-serif;
                   font-size: 0.9rem;
                   font-weight: 500;
                   text-align: center;
                   transition: all 0.3s ease;
                   box-shadow: 0 4px 12px var(--shadow-soft), inset 0 1px 2px rgba(255, 255, 255, 0.1), inset 0 -1px 2px rgba(0, 0, 0, 0.3);
                   position: relative;
                   overflow: hidden;
               "
               onmouseover="this.style.background='linear-gradient(135deg, var(--navy-light) 0%, var(--navy-medium) 100%)'; this.style.borderColor='var(--light-blue)'; this.style.transform='translateY(-2px)';"
               onmouseout="this.style.background='linear-gradient(135deg, var(--black-tertiary) 0%, var(--navy-dark) 100%)'; this.style.borderColor='var(--navy-light)'; this.style.transform='translateY(0)';"
               onmousedown="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px var(--shadow-soft), inset 0 2px 4px rgba(0, 0, 0, 0.4)';"
               onmouseup="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px var(--shadow-strong), inset 0 1px 3px rgba(255, 255, 255, 0.15), inset 0 -1px 3px rgba(0, 0, 0, 0.2)';"
            >
                {button_text}
            </a>
        </div>
        """
        
        st.markdown(button_html, unsafe_allow_html=True)
        return True
    else:
        st.error(f"Failed to generate {button_text.lower()}")
        return False

# --- SESSION PERSISTENCE WITH LOCALSTORAGE ---
def save_to_localstorage(key, data):
    json_data = json.dumps(data, default=str)
    components.html(f"""
    <script>
    localStorage.setItem('{key}', '{json_data}');
    parent.postMessage('saved', '*');
    </script>
    """, height=0)

def load_from_localstorage(key):
    result = components.html(f"""
    <script>
    const data = localStorage.getItem('{key}');
    if (data) {{
        parent.postMessage(data, '*');
    }} else {{
        parent.postMessage(null, '*');
    }}
    </script>
    """, height=0)
    if result:
        try:
            return json.loads(result)
        except:
            return None
    return None

def get_session_id():
    # Use authenticated user's email for session ID
    user_email = user_info['email'] if user_info else "anonymous"
    return f"medaffairs_{user_email}_{datetime.now().strftime('%Y%m%d')}"

def save_medical_context():
    session_id = get_session_id()
    context_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "history_count": len(st.session_state.get("history", [])),
        "checkpoints": st.session_state.get("checkpoints", []),
        "last_checkpoint_turn": st.session_state.get("last_checkpoint_turn", 0),
        "recent_history": st.session_state.get("history", [])[-8:],
        "thread_id": st.session_state.get("thread_id", ""),
        "user_profile": {
            "name": user_info['name'] if user_info else "",
            "email": user_info['email'] if user_info else "",
            "role": user_info['role'] if user_info else "",
            "client": user_info['client'] if user_info else "",
            "access_level": user_info['access_level'] if user_info else ""
        }
    }
    save_to_localstorage(f"medical_context_{session_id}", context_data)

def load_medical_context():
    session_id = get_session_id()
    stored_data = load_from_localstorage(f"medical_context_{session_id}")
    if stored_data:
        if "checkpoints" in stored_data:
            st.session_state["checkpoints"] = stored_data["checkpoints"]
        if "last_checkpoint_turn" in stored_data:
            st.session_state["last_checkpoint_turn"] = stored_data["last_checkpoint_turn"]
        if "recent_history" in stored_data and not st.session_state.get("history"):
            st.session_state["history"] = stored_data["recent_history"]
        if "thread_id" in stored_data and not st.session_state.get("thread_id"):
            st.session_state["thread_id"] = stored_data["thread_id"]
        if "user_profile" in stored_data:
            profile = stored_data["user_profile"]
            # Don't override authenticated user info with stored data
            # Just validate it matches current user
            if profile.get("email") == (user_info['email'] if user_info else ""):
                return True
        return True
    return False

# --- USER/THREAD LOGIC ---
user_id = user_info['email'] if user_info else "anonymous"
if "user_thread_registry" not in st.session_state:
    st.session_state["user_thread_registry"] = {}

# Check if user has an existing thread, and validate it still exists
if user_id in st.session_state["user_thread_registry"]:
    existing_thread_id = st.session_state["user_thread_registry"][user_id]
    # Validate the thread using cached validation (reduces API calls)
    is_valid, validation_error = validate_thread_exists_cached(existing_thread_id)
    if not is_valid:
        # Thread is invalid/expired - create a new one
        try:
            thread = openai.beta.threads.create()
            st.session_state["user_thread_registry"][user_id] = thread.id
            # Invalidate cache for old thread
            invalidate_thread_cache(existing_thread_id)
            # Clear history since thread was reset
            if "history" in st.session_state:
                st.session_state["history"] = []
        except Exception as e:
            st.error(f"Failed to create new thread: {e}")
else:
    # No existing thread - create new one
    try:
        thread = openai.beta.threads.create()
        st.session_state["user_thread_registry"][user_id] = thread.id
    except Exception as e:
        st.error(f"Failed to create thread: {e}")

thread_id = st.session_state["user_thread_registry"].get(user_id, "")
st.session_state["thread_id"] = thread_id

# --- SESSION STATE INIT WITH PERSISTENCE ---
context_loaded = load_medical_context()
if "history" not in st.session_state:
    st.session_state["history"] = []
if "checkpoints" not in st.session_state:
    st.session_state["checkpoints"] = []
if "last_checkpoint_turn" not in st.session_state:
    st.session_state["last_checkpoint_turn"] = 0
if "checkpoint_pending" not in st.session_state:
    st.session_state["checkpoint_pending"] = False
if "uploaded_file_ids" not in st.session_state:
    st.session_state["uploaded_file_ids"] = []
if "last_save" not in st.session_state:
    st.session_state["last_save"] = time.time()
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# Pre-populate user info from authentication
if user_info:
    st.session_state["user_name"] = user_info['name']
    st.session_state["user_role"] = user_info['role']
    st.session_state["user_client"] = user_info['client']
    st.session_state["user_email"] = user_info['email']
    st.session_state["user_access_level"] = user_info['access_level']

# --- Auto-save context every 30s if needed ---
if time.time() - st.session_state["last_save"] > 30 and st.session_state.get("history"):
    save_medical_context()
    st.session_state["last_save"] = time.time()

# --- CONTEXT ASSEMBLY, ASSISTANT RUN, CHECKPOINTS FUNCTIONS ---
def assemble_medical_context():
    """Assembly context with role-based initial message"""
    history = st.session_state.get("history", [])
    
    # Check if this is first interaction (no user messages yet)
    user_messages = [msg for msg in history if msg["role"] == "user"]
    is_first_interaction = len(user_messages) == 0
    
    context_parts = []
    
    # Add user context for first interaction only
    if is_first_interaction and user_info:
        # Define role perspectives for HOW to respond
        role_perspectives = {
            "Global Medical Affairs Director": "strategic and executive-focused, emphasizing frameworks, organizational impact, and high-level decision support",
            "Regional Medical Affairs Manager": "balanced strategic and tactical, focusing on regional implementation and team coordination", 
            "Medical Science Liaison (MSL)": "field-practical and scientifically detailed, emphasizing KOL engagement and territory insights",
            "Medical Information Specialist": "precise and compliance-focused, emphasizing accurate medical information and standard responses",
            "Medical Communications Manager": "strategic communications-focused, emphasizing publication planning and stakeholder messaging",
            "Medical Writing Lead": "technically detailed and document-focused, emphasizing clarity, accuracy, and regulatory compliance",
            "HEOR/RWE Manager": "data and evidence-focused, emphasizing quantitative analysis and value demonstration",
            "Clinical Affairs Manager": "operationally focused, balancing clinical science with trial management practicalities",
            "Regulatory Medical Writer": "highly technical and compliance-oriented, emphasizing regulatory requirements and submission standards",
            "Patient Advocacy Lead": "patient-centric and partnership-focused, emphasizing access, education, and stakeholder collaboration",
            "KOL Engagement Manager": "relationship and influence-focused, emphasizing strategic partnerships and thought leader management",
            "Digital Medical Strategy Lead": "innovation and technology-focused, emphasizing digital transformation and omnichannel engagement"
        }
        
        user_role = user_info.get('role', 'Medical Affairs Professional')
        perspective = role_perspectives.get(user_role, "comprehensive medical affairs support")
        
        context_parts.append("=== FIRST INTERACTION - USER PROFILE ===")
        context_parts.append(f"User: {user_info['name']} ({user_info['email']})")
        context_parts.append(f"Role: {user_role}")
        context_parts.append(f"Organization: {user_info.get('client', 'Not specified')}")
        context_parts.append("")
        context_parts.append("=== WELCOME MESSAGE INSTRUCTION ===")
        context_parts.append(f"This is {user_info['name']}'s first interaction as a {user_role}.")
        context_parts.append(f"Please provide a brief welcome (2-3 sentences) that:")
        context_parts.append(f"1. Acknowledges their role in {user_info.get('client', 'their organization')}")
        context_parts.append(f"2. Sets expectation that responses will be {perspective}")
        context_parts.append(f"3. Invites them to share their current medical affairs challenge or question")
        context_parts.append("")
        
        # Return just the welcome context for first interaction
        return "\n".join(context_parts)
    
    # For ongoing conversations: Return minimal context since OpenAI thread already has full history
    # This prevents double context assembly which causes timeouts and bloated messages
    if not history:
        return ""

    # Only add checkpoint summary if available (compact representation of prior conversation)
    if st.session_state.get("checkpoints"):
        return f"Prior session summary: {st.session_state['checkpoints'][-1]['summary']}"

    # No additional context needed - thread has full history
    return ""

def checkpoint(history_slice):
    """
    Create checkpoint summary using GPT-4.1.
    Optimized for GPT-4.1 ecosystem with circuit breaker resilience.
    """
    user_email = user_info['email'] if user_info else 'unknown'
    checkpoint_data = create_checkpoint_summary(history_slice, user_email)
    return checkpoint_data.get("summary", f"Summary unavailable. Session covers {len(history_slice)} exchanges.")
        
def improved_assistant_run(thread_id, assistant_id, message):
    """Run assistant with improved error handling for 400 errors and thread validation.

    Uses cached thread validation to reduce API calls and circuit breaker for resilience.
    """
    # Validate thread exists using cached validation (reduces redundant API calls)
    is_valid, validation_error = validate_thread_exists_cached(thread_id)
    if not is_valid:
        st.error(f"❌ Thread validation failed: {validation_error}")
        st.warning("💡 Please click 'Reset Session' in the sidebar to start fresh.")
        # Clear invalid thread from registry
        if "user_thread_registry" in st.session_state:
            user_id = st.session_state.get("user_email", "anonymous")
            if user_id in st.session_state["user_thread_registry"]:
                del st.session_state["user_thread_registry"][user_id]
        return f"Session error: {validation_error}"

    for attempt in range(RETRY_ATTEMPTS):
        try:
            context = assemble_medical_context()
            enhanced_message = f"{message}\n\n--- CONTEXT FOR CONTINUITY ---\n{context}" if context else message

            if attempt == 0:
                wait_for_idle_thread(thread_id, poll=POLLING_INTERVAL, timeout=MAX_WAIT)
                try:
                    openai.beta.threads.messages.create(
                        thread_id=thread_id,
                        role="user",
                        content=enhanced_message
                    )
                except openai.BadRequestError as e:
                    error_str = str(e).lower()
                    if "thread" in error_str:
                        st.error(f"❌ Thread error (400): {e}")
                        st.warning("💡 Please click 'Reset Session' to start fresh.")
                        return f"Thread error: {e}"
                    raise  # Re-raise for other BadRequestErrors
                except openai.NotFoundError as e:
                    st.error(f"❌ Thread not found: {e}")
                    st.warning("💡 Please click 'Reset Session' to start fresh.")
                    return f"Thread not found: {e}"

            try:
                run = openai.beta.threads.runs.create(
                    thread_id=thread_id,
                    assistant_id=assistant_id
                )
            except openai.BadRequestError as e:
                st.error(f"❌ Failed to create run (400): {e}")
                return f"Run creation failed: {e}"
            except openai.NotFoundError as e:
                st.error(f"❌ Thread or assistant not found: {e}")
                return f"Resource not found: {e}"

            start_time = time.time()
            status_placeholder = st.empty()

            while True:
                try:
                    run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                    elapsed = time.time() - start_time

                    if int(elapsed) % 5 == 0:
                        status_placeholder.info(f"🗽 Processing... ({elapsed:.0f}s) - Status: {run_status.status}")

                    if run_status.status == "requires_action":
                        _handle_function_call(run_status, thread_id, run.id)
                    elif run_status.status == "completed":
                        # Capture full run steps (tool calls + outputs) for search history export
                        st.session_state["_last_run_id"] = run.id
                        st.session_state["_last_run_steps"] = _fetch_run_steps(thread_id, run.id)

                        # Use run_id filtering to get the correct assistant message
                        messages = openai.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
                        status_placeholder.empty()
                        for m in messages.data:
                            if m.role == "assistant":
                                # Primary: match by run_id
                                if hasattr(m, 'run_id') and m.run_id == run.id:
                                    if m.content and len(m.content) > 0:
                                        return m.content[0].text.value
                                # Fallback: use timestamp comparison
                                elif m.created_at >= int(start_time):
                                    if m.content and len(m.content) > 0:
                                        return m.content[0].text.value
                        return "No response received"
                    elif run_status.status in ["failed", "cancelled", "expired"]:
                        status_placeholder.empty()
                        if attempt < RETRY_ATTEMPTS - 1:
                            st.warning(f"Run {run_status.status}, retrying...")
                            try:
                                openai.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                            except:
                                pass
                            break
                        else:
                            return f"Assistant run {run_status.status} after {RETRY_ATTEMPTS} attempts"
                    elif elapsed > MAX_WAIT:
                        status_placeholder.empty()
                        if attempt < RETRY_ATTEMPTS - 1:
                            st.warning(f"Timeout after {MAX_WAIT}s, cancelling and retrying...")
                            try:
                                openai.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                            except:
                                pass
                            break
                        else:
                            return f"Request timed out after {MAX_WAIT} seconds and {RETRY_ATTEMPTS} attempts."

                    time.sleep(POLLING_INTERVAL)

                except openai.RateLimitError:
                    st.warning("Rate limit hit, waiting 10 seconds...")
                    time.sleep(10)
                    continue
                except openai.BadRequestError as e:
                    st.error(f"❌ API error during processing (400): {e}")
                    return f"Processing error: {e}"

            time.sleep(3)

        except openai.BadRequestError as e:
            error_msg = str(e)
            st.error(f"❌ API error (400): {error_msg}")
            if "thread" in error_msg.lower():
                st.warning("💡 Please click 'Reset Session' to start fresh.")
                return f"Thread error: {error_msg}"
            if attempt < RETRY_ATTEMPTS - 1:
                st.warning(f"Retrying... (attempt {attempt + 1})")
                time.sleep(2)
                continue
            return f"API error after {RETRY_ATTEMPTS} attempts: {error_msg}"
        except openai.NotFoundError as e:
            st.error(f"❌ Resource not found: {e}")
            st.warning("💡 Please click 'Reset Session' to start fresh.")
            return f"Not found error: {e}"
        except Exception as e:
            error_msg = str(e)
            if "while a run" in error_msg and "is active" in error_msg:
                st.warning(f"Waiting for active run to clear... (attempt {attempt + 1})")
                time.sleep(5)
                continue
            elif attempt < RETRY_ATTEMPTS - 1:
                st.warning(f"Attempt {attempt + 1} failed: {error_msg}. Retrying...")
                time.sleep(2)
                continue
            else:
                return f"Error after {RETRY_ATTEMPTS} attempts: {error_msg}"

    return "All attempts failed"

# ==========================================================
# AUTO-SEND WELCOME MESSAGE FOR NEW SESSIONS
# ==========================================================
if "welcome_sent" not in st.session_state and user_info:
    # Check if this is truly a new session (no history)
    if not st.session_state.get("history"):
        try:
            # Build a clear, direct welcome prompt (not going through context assembly)
            user_role = user_info.get('role', 'Medical Affairs Professional')
            user_name = user_info.get('name', 'User')
            user_org = user_info.get('client', 'your organization')

            # Role-based perspective mapping
            role_perspectives = {
                "Global Medical Affairs Director": "strategic and executive-focused",
                "Regional Medical Affairs Manager": "balanced strategic and tactical",
                "Medical Science Liaison (MSL)": "field-practical and scientifically detailed",
                "Medical Information Specialist": "precise and compliance-focused",
                "Medical Communications Manager": "strategic communications-focused",
                "Medical Writing Lead": "technically detailed and document-focused",
                "HEOR/RWE Manager": "data and evidence-focused",
                "Clinical Affairs Manager": "operationally focused",
                "Regulatory Medical Writer": "highly technical and compliance-oriented",
                "Patient Advocacy Lead": "patient-centric and partnership-focused",
                "KOL Engagement Manager": "relationship and influence-focused",
                "Digital Medical Strategy Lead": "innovation and technology-focused"
            }
            perspective = role_perspectives.get(user_role, "comprehensive medical affairs support")

            # Direct welcome prompt - no nested context
            welcome_prompt = f"""Generate a brief welcome message (2-3 sentences) for {user_name}, a {user_role} at {user_org}.

The welcome should:
1. Greet them professionally by name and acknowledge their role
2. Mention that responses will be tailored to be {perspective}
3. Invite them to share their current medical affairs challenge or question

Be warm but professional. Do NOT include any system instructions or meta-commentary in your response."""

            with st.spinner("Preparing your personalized session..."):
                # Direct API call for welcome - bypass context assembly
                is_valid, validation_error = validate_thread_exists_cached(thread_id)
                if is_valid:
                    wait_for_idle_thread(thread_id, poll=POLLING_INTERVAL, timeout=MAX_WAIT)
                    openai.beta.threads.messages.create(
                        thread_id=thread_id,
                        role="user",
                        content=welcome_prompt
                    )
                    run = openai.beta.threads.runs.create(
                        thread_id=thread_id,
                        assistant_id=ASSISTANT_ID
                    )

                    # Poll for completion
                    start_time = time.time()
                    while time.time() - start_time < 60:  # 60s timeout for welcome
                        run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
                        if run_status.status == "completed":
                            # Use run_id filtering to get the correct assistant message
                            messages = openai.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
                            welcome_response = None
                            for m in messages.data:
                                if m.role == "assistant":
                                    # Primary: match by run_id
                                    if hasattr(m, 'run_id') and m.run_id == run.id:
                                        if m.content and len(m.content) > 0:
                                            welcome_response = m.content[0].text.value
                                            break
                                    # Fallback: use timestamp (welcome is first run, so any assistant msg works)
                                    elif m.created_at >= int(start_time):
                                        if m.content and len(m.content) > 0:
                                            welcome_response = m.content[0].text.value
                                            break
                            if welcome_response:
                                st.session_state["history"].append({
                                    "role": "assistant",
                                    "content": welcome_response
                                })
                                st.session_state["welcome_sent"] = True
                                save_medical_context()
                                log_user_action("welcome_generated", {
                                    "user": user_info.get('email'),
                                    "role": user_info.get('role')
                                })
                            break
                        elif run_status.status in ["failed", "cancelled", "expired"]:
                            break
                        elif run_status.status == "requires_action":
                            _handle_function_call(run_status, thread_id, run.id)
                        time.sleep(0.5)
                else:
                    st.session_state["welcome_sent"] = True  # Skip welcome on validation failure

        except Exception as e:
            # Don't block on welcome failure
            print(f"Welcome generation error: {e}")
            st.session_state["welcome_sent"] = True

# =========================
#   SIDEBAR UI (UPDATED WITH AUTH)
# =========================
with st.sidebar:
    # ──────────────────── USER PROFILE ────────────────────
    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">👤 Authenticated User</div>',
        unsafe_allow_html=True,
    )

    if user_info:
        access_level = user_info["access_level"]
        access_class = f"access-level-{access_level}"
        st.markdown(
            f"""
            <div class="user-info-banner">
                <strong>👋 {user_info['name']}</strong><br>
                <small>{user_info['role']}</small><br>
                <small>🏢 {user_info['client']}</small><br>
                <span class="{access_class}">Access: {access_level.title()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        restrictions = get_access_restrictions()
        if restrictions:
            st.caption("⚠️ Access restrictions: " + ", ".join(restrictions))

    # ───────────────────── NEW CHAT ────────────────────────
    if st.button("🆕 New Chat"):
        log_user_action("new_chat", "User started new chat session")
        st.session_state["history"] = []
        user_id = user_info["email"] if user_info else "anonymous"
        thread = openai.beta.threads.create()

        if "user_thread_registry" not in st.session_state:
            st.session_state["user_thread_registry"] = {}
        st.session_state["user_thread_registry"][user_id] = thread.id
        st.session_state["thread_id"] = thread.id
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close Auth section

        # ───────────────────── RESET SESSION ────────────────────────
    if st.button("🔁 Reset Session (delete thread)"):
        try:
            # 1) capture and pre-create a new thread so UI never goes threadless
            old_thread_id = st.session_state.get("thread_id")
            new_thread = openai.beta.threads.create()
            
            # 2) try to delete old thread (ignore failures)
            if old_thread_id:
                try:
                    openai.beta.threads.delete(old_thread_id)
                    log_user_action("thread_deleted", f"Deleted thread {old_thread_id}")
                except Exception as e:
                    log_user_action("thread_delete_failed", f"{e}")
            
            # 3) clear local session state (but keep registry shell)
            st.session_state["history"] = []
            st.session_state["checkpoints"] = []
            st.session_state["uploaded_file_ids"] = []
            st.session_state["last_checkpoint_turn"] = 0
            st.session_state["checkpoint_pending"] = False
            
            # 4) update registry and active thread
            user_id = user_info["email"] if user_info else "anonymous"
            if "user_thread_registry" not in st.session_state:
                st.session_state["user_thread_registry"] = {}
            st.session_state["user_thread_registry"][user_id] = new_thread.id
            st.session_state["thread_id"] = new_thread.id

            # 5) persist a clean context snapshot (overwrites localStorage key)
            save_medical_context()

            st.success("Session reset. New thread created.")
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")


# ───────────── Sidebar layout ─────────────
with st.sidebar:

    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">🔍 Deep Dive</div>',
        unsafe_allow_html=True,
    )

    # Text input
    deep_dive_term = st.text_input(
        "Term to expand on:",
        key="deep_dive_input",
        placeholder="e.g., KOL mapping"
    )

    # Send button
    if st.button("🎯 Expand", key="deep_dive_btn"):
        if deep_dive_term.strip():
            # Track search in full search history (uncapped)
            st.session_state["search_history"].append({
                "query": deep_dive_term.strip(),
                "source": "deep_dive",
                "timestamp": datetime.now().isoformat(),
                "user": user_info.get("email", "unknown") if user_info else "unknown",
                "thread_id": st.session_state.get("thread_id", ""),
            })
            # Create the silent prompt
            expand_prompt = f"Expand on: {deep_dive_term}. always run live data search and regulatory validation. Please provide the information in a matrix or tabular format if applicable."

            # Execute it
            st.session_state["silent_prompt_to_run"] = expand_prompt
            st.rerun()
        else:
            st.warning("Please enter a term")

    st.markdown("</div>", unsafe_allow_html=True)

    # ───────────── FACT CHECK FIELD ─────────────
    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">✓ Fact Check</div>',
        unsafe_allow_html=True,
    )

    # Text area for claim to fact-check
    fact_check_claim = st.text_area(
        "Claim to verify:",
        key="fact_check_input",
        placeholder="e.g., The NHS workforce is 21% non-British staff",
        height=100
    )

    # Send button
    if st.button("🔍 Verify", key="fact_check_btn"):
        if fact_check_claim.strip():
            # Track search in full search history (uncapped)
            st.session_state["search_history"].append({
                "query": fact_check_claim.strip(),
                "source": "fact_check",
                "timestamp": datetime.now().isoformat(),
                "user": user_info.get("email", "unknown") if user_info else "unknown",
                "thread_id": st.session_state.get("thread_id", ""),
            })
            # Create the fact-check prompt with specialized instructions
            fact_check_prompt = f"""Fact-check and validate the following claim:

"{fact_check_claim}"

Instructions:
- load and execute fact_verification_v1.json protocol
- Search multiple authoritative sources (regulatory databases, peer-reviewed literature, official statistics)
- Cross-validate against at least 2 independent primary sources
- If fact checking a scientific paper always load data_presentation.json
- Provide confidence level: High/Moderate/Low/Cannot Verify
- State exact verification date and sources accessed
- If claim contains time-sensitive information, verify current status
- Distinguish between VERIFIED FACTS vs INFERENCES
- Flag any discrepancies between sources
- Include hyperlinks to all sources used

Output format:
- VERDICT: [Verified/Partially Verified/Cannot Verify/Refuted]
- CONFIDENCE: [High/Moderate/Low]
- VERIFICATION DATE: [Today's date]
- SOURCES: [List with links]
- DETAILS: [Brief explanation of findings]
- DISCREPANCIES: [Note any conflicting information found]"""
        
            # Execute it
            st.session_state["silent_prompt_to_run"] = fact_check_prompt
            st.rerun()
        else:
            st.warning("Please enter a claim to verify")

    st.markdown("</div>", unsafe_allow_html=True)


    # ---------- INITIALISE STATE ----------
    if "uploaded_files_info" not in st.session_state:
        st.session_state["uploaded_files_info"] = []          # [{id,name,size,uploaded_at}]
    if "parsed_files" not in st.session_state:
        st.session_state["parsed_files"] = {}                 # {file_id: {"name", "summary"}}
    if "last_processed_upload" not in st.session_state:
         st.session_state["last_processed_upload"] = None      # (filename, size_bytes)
    # ---------- FILE‑UPLOAD HEADER ----------
    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">📁 File Upload</div>',
        unsafe_allow_html=True,
    )

    # ---------- ACCESS CHECK ----------
    if check_access_level("limited"):

        # ----- File‑uploader widget -----
        uploaded_file = st.file_uploader(
            "Upload documents for analysis",
            type=["pdf", "docx", "txt", "csv", "xlsx"],
            help="Files will be parsed and summarized upon upload",
        )

        if uploaded_file is not None:
            file_size_bytes = getattr(uploaded_file, "size", None)
            if file_size_bytes is None:
                file_size_bytes = len(uploaded_file.getvalue())
            file_size_mb = file_size_bytes / 1024 / 1024
            file_signature = (uploaded_file.name, file_size_bytes)
            st.caption(f"📄 {uploaded_file.name} ({file_size_mb:.1f} MB)")

            if st.session_state.get("last_processed_upload") != file_signature:
                st.session_state["last_processed_upload"] = file_signature
                
            # ----- Upload & parse button -----
                with st.spinner(f"🔄 Uploading {uploaded_file.name}…"):
                    upload_result = handle_file_upload(uploaded_file)

                if not upload_result.get("success"):
                    st.error(f"❌ Upload failed: {upload_result.get('error', 'Unknown error')}")
                    log_user_action("file_upload_failed", upload_result.get("error"))
                    st.session_state["last_processed_upload"] = None
                else:
                    file_id = upload_result["file_id"]
                    st.session_state["uploaded_files_info"].append(
                        {
                            "id": file_id,
                            "name": uploaded_file.name,
                            "size": upload_result["size_mb"],
                            "uploaded_at": datetime.now().isoformat(),
                        }
                    )
                    st.success(f"✅ {uploaded_file.name} uploaded successfully!")

                    # -------- One‑off parsing run --------
                    parse_prompt = (
                        f"I've uploaded the file '{uploaded_file.name}'. "
                        "Please provide: 1) brief acknowledgment, "
                        "2) 2–3 sentence summary, "
                        "3) medical‑affairs insights."
                    )
                    with st.spinner("🔍 Parsing file content…"):
                        parse_response = run_assistant(
                            user_input=parse_prompt,
                            output_type="brief_summary",
                            response_tone="professional",
                            compliance_level="strict",
                            thread_id=thread_id,
                            assistant_id=ASSISTANT_ID,
                            uploaded_file_ids=[file_id],      # only this file
                        )

                    if parse_response and not parse_response.startswith("❌"):
                        # cache summary
                        st.session_state["parsed_files"][file_id] = {
                            "name": uploaded_file.name,
                            "summary": parse_response,
                        }
                        # add to chat history
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["history"].extend(
                            [
                                {"role": "user", "content": f"[{ts}] [SYSTEM] File uploaded: {uploaded_file.name}"},
                                {"role": "assistant", "content": parse_response},
                            ]
                        )
                        save_medical_context()
                        st.info(f"📄 **File Summary:**\n\n{parse_response[:200]}…")
                        log_user_action("file_parsed", f"Parsed {uploaded_file.name}")
                    else:
                        st.warning("File uploaded but parsing failed. File is still available as reference.")
                        log_user_action("file_parse_failed", f"Failed to parse {uploaded_file.name}")
                        st.session_state["last_processed_upload"] = None

        else:
            st.session_state["last_processed_upload"] = None
        # ----- Show most recent uploads -----
        if st.session_state["uploaded_files_info"]:
            st.markdown("**📋 Reference Files:**")
            for file_info in st.session_state["uploaded_files_info"][-3:]:
                file_id = file_info['id']
                file_id_short = file_id[:8]
                confirm_key = f"confirm_delete_{file_id_short}"
        
                with st.expander(f"📄 {file_info['name']}"):
                    st.caption(f"Size: {file_info['size']:.1f} MB")
                    st.caption(f"Uploaded: {file_info['uploaded_at'][:16]}")
                    st.caption(f"ID: {file_id[:12]}…")
        
                    col1, col2, col3 = st.columns(3)
        
                    # Re-analyse
                    with col1:
                        if st.button("🔍", key=f"rean_{file_id_short}"):
                            with st.spinner("Re‑analysing…"):
                                reanalyse_prompt = (
                                    f"Please provide a fresh analysis of '{file_info['name']}' "
                                    "focusing on medical‑affairs insights, key data, and recommendations."
                                )
                                repl = run_assistant(
                                    user_input=reanalyse_prompt,
                                    output_type="detailed_analysis",
                                    response_tone="professional",
                                    compliance_level="strict",
                                    thread_id=thread_id,
                                    assistant_id=ASSISTANT_ID,
                                    uploaded_file_ids=[file_id],
                                )
                            if repl and not repl.startswith("❌"):
                                st.info(f"**Updated Analysis:**\n\n{repl[:300]}…")
                            else:
                                st.error("Re‑analysis failed")
        
                    # Copy ID
                    with col2:
                        if st.button("📋", key=f"copy_{file_id_short}"):
                            st.code(f"File ID: {file_id}")
        
                    # Delete with confirmation
                    with col3:
                        if st.session_state.get(confirm_key):
                            if st.button("⚠️ Confirm Delete", key=f"conf_{file_id_short}"):
                                try:
                                    st.session_state["uploaded_files_info"] = [
                                        f for f in st.session_state["uploaded_files_info"] if f["id"] != file_id
                                    ]
                                    st.session_state["parsed_files"].pop(file_id, None)
        
                                    import openai
                                    try:
                                        openai.files.delete(file_id)
                                        msg = "File removed from OpenAI storage"
                                    except Exception:
                                        msg = "File removed from session (will auto‑expire)"
        
                                    st.success(f"✅ {file_info['name']} deleted! {msg}")
                                    log_user_action("file_deleted", f"Deleted {file_info['name']}")
                                    st.session_state.pop(confirm_key)  # Reset confirmation
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Deletion failed: {e}")
                        else:
                            if st.button("🗑️ Delete", key=f"del_{file_id_short}"):
                                st.session_state[confirm_key] = True
                                st.rerun()
        else:
            st.warning("🔒 File upload requires limited access or higher")
        
        # Close sidebar section
        st.markdown("</div>", unsafe_allow_html=True)


    # ───────────────────── DOWNLOADS ───────────────────────
    if check_access_level("full"):
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">📥 Downloads</div>',
            unsafe_allow_html=True,
        )

        # PDF (instead of DOCX)
        if st.session_state.get("history"):
            pdf_data, pdf_filename = generate_pdf_download()
            if pdf_data and pdf_filename:
                if create_download_button(
                    pdf_data,
                    pdf_filename,
                    "📄 Download Session Report (PDF)",
                    "download_pdf",
                    "application/pdf",
                ):
                    log_user_action(
                        "download_pdf", f"Downloaded session report: {pdf_filename}"
                    )
            else:
                st.button(
                    "📄 Download Session Report (PDF)",
                    disabled=True,
                    help="Session data needed",
                )
        else:
            st.button(
                "📄 Download Session Report (PDF)",
                disabled=True,
                help="No conversation history",
            )

        # JSON
        json_data, json_filename = generate_json_download()
        if json_data and json_filename:
            if create_download_button(
                json_data,
                json_filename,
                "🔧 Download Audit Data (JSON)",
                "download_json",
                "application/json",
            ):
                log_user_action(
                    "download_json", f"Downloaded audit data: {json_filename}"
                )
        else:
            st.button(
                "🔧 Download Audit Data (JSON)",
                disabled=True,
                help="Export generation failed",
            )

        # Search History JSON
        if st.session_state.get("search_history"):
            sh_data, sh_filename = generate_search_history_json()
            if sh_data and sh_filename:
                if create_download_button(
                    sh_data,
                    sh_filename,
                    "🔎 Download Search History (JSON)",
                    "download_search_history",
                    "application/json",
                ):
                    log_user_action(
                        "download_search_history",
                        f"Downloaded search history: {sh_filename}",
                    )
            else:
                st.button(
                    "🔎 Download Search History (JSON)",
                    disabled=True,
                    help="Export generation failed",
                )
        else:
            st.button(
                "🔎 Download Search History (JSON)",
                disabled=True,
                help="No searches recorded yet",
            )

        st.markdown("</div>", unsafe_allow_html=True)  # close Downloads
    else:
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">📥 Downloads</div>',
            unsafe_allow_html=True,
        )
        st.warning("🔒 Download features require full access level")
        st.button("📄 Download Session Report (PDF)", disabled=True)
        st.button("🔧 Download Audit Data (JSON)", disabled=True)
        st.button("🔎 Download Search History (JSON)", disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────── QUICK ACTIONS (2×2 with tooltips) ───────────────
    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">⚡ Quick Actions</div>',
        unsafe_allow_html=True,
    )
    
    VALIDATION_PROMPT = textwrap.dedent("""
    [Prompt Start]

    Review the whole session, all the outputs and user inputs and extract all substantial factual claims.
    Do not use live data search, only show the substantial factual claims present in the session.

    For each claim, provide:
    1. The specific assertion (keep it to one sentence)
    2. Type: [Date/Number/Status/Attribution/Relationship]
    3. Stakes: [High/Medium/Low] based on likely external use
    
    Format as numbered list for easy reference to fact-check function, provide a separete list for the user and another for the system.
    
    
    Focus on claims that are:
    - Dates, numbers, voting records
    - Regulatory status/decisions  
    - Clinical trial outcomes
    - Scientific data
    - Competitive intelligence facts

    [Prompt End]
    """).strip()
    
    quick_actions = [
        ("Step by step", "proceed and use live web search to gather any information necessery for the next phase"),
        ("Increase detail", "Provide extra detail and deep dive to the last output, use live search to increase detail and refinement"),
        ("Claim listing", VALIDATION_PROMPT),
        ("Audit (ABPI)", "Generate a full-session audit pack (use live search for the ABPI code regulations), do a live check of the main claims and flag any error or inconsistency: timeline, internal/external refs, claims/citations, compact references, sign-off form."),
        ("Phase 2", "give me Phase 2, use live search and DB vs_693fe785b1a081918f82e9f903e008ed"),
        ("Phase 3", "give me phase 3, use live search and DB vs_693fe785b1a081918f82e9f903e008ed"),
        ("Phase 4", "give me phase 4, use live search and DB vs_693fe785b1a081918f82e9f903e008ed"),
        ("Phase 5", "give me phase 5, use live search and DB vs_693fe785b1a081918f82e9f903e008ed"),
    ]
    # Hover text for each button
    qa_help = {
        "1": "Break down the last request and execute stepwise.",
        "2": "Deepen the last output with live search and refinement.",
        "3": "Produce a structured, MAPS-style session summary.",
        "4": "Build a pharma-grade audit pack incl. citations & sign-off.",
        "A": "Strategic Framework",
        "B": "Tactical Framework",
        "C": "Tactical Devilerables",
        "D": "Risk and Mitigation",
    }
    
    rows = [quick_actions[i:i+2] for i in range(0, len(quick_actions), 2)]
    for r, pair in enumerate(rows):
        c1, c2 = st.columns([1, 1], gap="small")
        for col, (label, silent_prompt) in zip((c1, c2), pair):
            with col:
                disabled = (label == "Audit") and (not check_access_level("full"))
                tip = qa_help.get(label, "")
                # 1) native browser tooltip on hover (title=...)
                st.markdown(f'<div class="qa-tip" title="{escape(tip, quote=True)}">', unsafe_allow_html=True)
                # 2) Streamlit tooltip (help=...)
                if st.button(label, key=f"qa_{r}_{label}", disabled=disabled, use_container_width=True, help=tip):
                    if not disabled:
                        log_user_action("quick_action", f"Executed quick action: {label}")
                        # Track search in full search history (uncapped)
                        st.session_state["search_history"].append({
                            "query": label,
                            "source": "quick_action",
                            "timestamp": datetime.now().isoformat(),
                            "user": user_info.get("email", "unknown") if user_info else "unknown",
                            "thread_id": st.session_state.get("thread_id", ""),
                        })
                        st.session_state["silent_prompt_to_run"] = silent_prompt
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # close Quick Actions
    
# ─────────────── AUTOMATED FUNCTIONS (10×2) ────────────
    st.markdown(
        '<div class="sidebar-section"><div class="sidebar-title">🔧 Analysis Tools</div>',
        unsafe_allow_html=True,
    )

    LANDSCAPE_PROMPT = textwrap.dedent("""
    [Prompt Start]

    Inputs:
    - Domain/Topic: <DOMAIN/TOPIC>
    - Regions/Markets: <REGIONS>
    - Internal files: stakeholder_taxonomy_v2.json; tactics_taxonomy_v2.json; pillars_v2.json; metrics_v2.json; Data_presentation.json; authoritative_sources.json; <uploaded session files>
    
    Task:
    Produce an evidence-based landscape covering:
    1) Baseline & Impact: demand/usage (or epidemiology), burden (clinical/operational/economic).
    2) Paradigm/Workflow: current practice, sequencing/dependencies, biomarker/standards if relevant.
    3) Competitive Set: on-market vs roadmap (capabilities, MOA/tech, phase/readouts, differentiators/risks).
    4) Policy/Regulatory/Standards: current status, recent changes, key constraints.
    5) Economics: pricing/tariffs/TCO/reimbursement or procurement signals (as applicable).
    
    Live validation:
    - Use live search; prioritize primary/authoritative sources per authoritative_sources.json.
    - For each key claim, confirm with ≥2 independent sources where feasible; record verification date (today) for time-sensitive items.
    - Separate VERIFIED FACTS vs INFERENCES and tag confidence: High / Moderate / Needs verification. Flag unverifiable items.
    
    Data presentation (per Data_presentation.json):
    - Tables: clear headings/units; abbreviations defined in footnotes.
    - Clinical/scientific endpoints: n/N, effect size, CI, p-values; otherwise include sample size or data provenance.
    - Mark missing as NR/ND; include data cut-off/version dates on every key metric.
    - Avoid nested lists/overwrap in cells; split rows as needed.
    - After each table block, add a 3–5 line plain-text summary.
    
    Compliance (context-aware):
    - Apply relevant domain/region rules (e.g., FDA/EMA/NICE/MAPS/ICH for healthcare; GDPR/SEC/ISO/etc. where applicable). Distinguish approved vs investigational/off-label when relevant.
    
    Output:
    - Executive snapshot (10–12 bullets).
    - The tables above (Baseline & Impact; Paradigm; Competitive; Policy/Regulatory; Economics).
    - Notes on VERIFIED vs INFERENCES with confidence tags.
    - Primary references with access dates.
    
    [Prompt End]
        """).strip()

    COMPETITIVE_INTELLIGENCE_PROMPT = textwrap.dedent("""
    [Prompt Start]

    Inputs:
    - Domain/Topic: <DOMAIN/TOPIC>
    - Competitors/Products: <COMPETITORS>
    - Regions/Markets: <REGIONS>
    - Internal files: stakeholder_taxonomy_v2.json; tactics_taxonomy_v2.json; pillars_v2.json; metrics_v2.json; Data_presentation.json; authoritative_sources.json; <uploaded session files>
    
    Task:
    Produce a comprehensive real-time competitive intelligence report covering:
    1) Product Pipelines: current portfolio (approved products by indication/geography), pipeline assets (phase, indication, MOA/tech, expected timelines), recent regulatory milestones.
    2) Clinical Trial Tracking: active trials (NCT IDs, phase, endpoints, enrollment status, estimated completion), recent readouts (topline results, statistical significance, effect sizes), upcoming data catalysts.
    3) Publication Monitoring: peer-reviewed publications (key findings, journal impact, author affiliations), conference presentations (major congresses, abstract disclosures, presentation formats), real-world evidence studies.
    4) Conference Presence: congress participation (speaking slots, symposia, booth presence), scientific leadership visibility, partnership announcements.
    5) Digital Engagement: HCP-facing platforms (medcomms, educational initiatives, digital reach), social media presence, thought leader engagement patterns.
    6) Pricing/Access: launch pricing (by region/indication), reimbursement status (payer coverage, access restrictions), patient access programs, health economic value propositions.
    7) Strategic Movements: M&A activity, licensing deals, collaborations, manufacturing capacity changes, leadership changes, investor communications.
    8) Weekly Alert System: material changes flagged (regulatory, clinical, commercial), competitive threats identified, opportunity windows noted.
    
    Live validation:
    - Use live search extensively; prioritize primary sources: clinicaltrials.gov, FDA/EMA/PMDA databases, PubMed, company investor relations, major congress websites per authoritative_sources.json.
    - For time-sensitive items (trial results, regulatory decisions, pricing changes): confirm with ≥2 independent sources and record exact verification date.
    - Separate VERIFIED FACTS vs INFERENCES and tag confidence: High / Moderate / Needs verification. Flag items requiring follow-up monitoring.
    - Cross-reference claimed timelines against official regulatory/clinical trial databases.
    
    Competitive analysis framework:
    - Position each competitor relative to market leader/standard of care.
    - Identify differentiation vectors: efficacy profile, safety/tolerability, dosing convenience, biomarker selection, pricing/access.
    - Flag competitive advantages and vulnerabilities for each player.
    - Note head-to-head trial data where available.
    - Assess probability of success (technical, regulatory, commercial) for pipeline assets.
    
    Compliance:
    - Apply relevant regulatory context (FDA/EMA/PMDA approved vs investigational).
    - Distinguish between on-label and off-label usage patterns.
    - Note geographic regulatory variations.
    - Include medical/legal disclaimers for investigational agents.
    
    Output:
    - Executive summary (12–15 bullets highlighting key competitive dynamics and recent changes).
    - Competitive landscape table (all major players with product profiles, development status, market position).
    - Clinical catalyst calendar (upcoming data readouts, regulatory decisions, patent expiries for next 12–24 months).
    - Strategic SWOT per major competitor.
    - Priority watch list (competitors/assets requiring close monitoring with rationale).
    - Alert triggers (criteria for weekly monitoring: regulatory submissions, pivotal trial readouts, pricing/reimbursement changes, M&A).
    - Notes on VERIFIED vs INFERENCES with confidence tags.
    - Primary references with exact access dates and source types (regulatory filing, trial registry, peer-review, press release, etc.).
    
    [Prompt End]
        """).strip()

    Stress_test_Market_Access_Strategy_PROMPT = textwrap.dedent("""
    [Prompt Start]

    Load: `pillars_v2.json`, `metrics_v2.json`, `tactics_taxonomy_v2.json`, `stakeholder_taxonomy_v2.json`, `pricing_market_access_v2.json`

    Extract plan parameters from conversation and uploaded files.
    
    ## Process
    Run 1000-iteration Monte Carlo simulation to model:
    - Pillar success probabilities
    - Constraint frequency
    - Threshold breach risk
    - Cascade dynamics
    - Time pressure impact
    - Resource allocation sensitivity
    
    ## Output Structure
    
    **Strategic Assessment**
    Classify: Robust / Viable but fragile / Conditionally viable / Strategically fragile
    Brief explanation (2-3 sentences).
    
    **Threshold Warnings**
    List any thresholds at risk:
    - Name, direction (safe/approaching/at risk), frequency, reversibility
    
    **Primary Constraint**
    Single factor most often causing failure. Why it matters. Classification: Fatal/Manageable/Acceptable.
    
    **Secondary Constraints**
    Up to 3 additional constraints (one sentence each).
    
    **Cascade Analysis**
    Key cascades: Who→whom, velocity, leverage, implication.
    
    **Time Pressure**
    Critical windows: What, why it matters, cost of missing.
    
    **Competitive Dynamics**
    Scenarios with probability ranges, not predictions.
    
    **Robustness**
    Works across: Most/Some/Narrow futures. Key assumptions (2-3).
    
    **Adjustments**
    3-5 changes: CHANGE / WHY / TRADE-OFF / IMPACT
    
    **Resource Shifts**
    Major/Moderate/Minor shifts: FROM→TO with rationale.
    
    ## Style
    - Confident: thresholds, cascades, constraints
    - Humble: magnitude, exact timing
    - Use ranges (60-70%), not points (63.2%)
    - Scenarios for uncontrollable, recommendations for controllable
    - Decision-first, no statistical jargon
    
    Strategic direction for decisions, not tactical forecasting.
    
    [Prompt End]
        """).strip()

    
    if check_access_level("full"):
        # Create an expander for the automated functions
        with st.expander("View All Functions", expanded=False):
            automated_functions = [
                ("Landscape Analysis", "Landscape analysis, use landscape_analysis_v3.json protocol and pillars_v2.json, metrics_v2.json, tactics_taxonomy_v2.json, stakeholder_taxonomy_v2.json, pricing_market_access_v2.json. output fields: Use the fields described and mandated by the execution of landscape_analysis_v3.json. Use maximum context window dor this analysis and present in a matrix format whenever applicable.  …"),
                ("Competitive Intelligence", "COMPETITIVE_INTELLIGENCE_PROMPT"),
                ("Porter's Five Forces", "Porter's Five Forces analysis of the whole plan and situational analysis, use live data search for a complete analysis …"),
                ("Blue Ocean Strategy Canvas", "Blue Ocean Strategy Canvas of the whole plan, use live data search for a complete analysis …"),
                ("Stakeholder Map", "Dynamic stakeholder ecosystem mapping with: influence/interest matrix, engagement journey mapping, preferred communication channels, collaboration opportunities, advocacy potential, and automated relationship tracking, use live data search for a complete analysis …"),
                ("Stakeholder Power-Interest Grid", "Stakeholder Power-Interest Grid, use live data search for a complete analysis …"),
                ("KOL Identification & Tiering", "KOL Identification & Tiering, use live data search for a complete analysis, load and use kol.json, use real opinion leaders that fit the user request and best practices, validate before output …"),
                ("KOL Mapping & Engagement Plan", "KOL Mapping & Engagement Plan, use live data search for a complete analysis, use real opinion leaders that fit the user request and best practices, validate before output …"), 
                ("Omnichannel Strategy", "Deliver an omnichannel strategy plan based on MAPS framework and industry best practices, use live data to generate the best plan …"),
                ("Omnichannel Engagement Journey Map", "Omnichannel Engagement Journey Map, use live data search for a complete analysis …"),
                ("Medical Affairs ROI Dashboard", "Medical Affairs ROI Dashboard, use live data search for a complete analysis …"),
                ("Regulatory Pathway", "Use session context to determine regions, product type, and likely pathway complexity. Use live regulatory sources selectively.If access/HTA constraints are central in-session, incorporate pricing_market_access_v2.json for payer/HTA levers and timing dependencies. Translate outputs into pillar implications (pillars_v2.json) and MA actions constrained to tactics_taxonomy_v2.json with KPIs from metrics_v2.json. Output: region pathway table + evidence expectations + MA action list. …"),
                ("Performance Dashboard", "Real-time KPI tracking: medical impact metrics, stakeholder engagement scores, evidence generation progress, budget utilization, and predictive analytics for course correction …"),
                ("ROI Analysis", "Dynamic ROI modeling: evaluate and simulate the ROI per strategic pillar and individual tactic, use live data search for a complete analysis…"),
                ("Team Playbooks", "Interactive operational guides: role-specific workflows, decision trees, compliance checklists, best practice libraries, and continuous learning modules …"),
                ("Capability Building", "Competency development framework: skill gap analysis, personalized learning paths, certification tracking, and team performance optimization …"),
                ("Simulation Engine", "Analyze this (launch or planning) strategy using similarity matching (comparison_guide_similarity_engine.json). Find 3-4 comparable historical parallels (via a live search, with data that can be empirically used), extract empirical execution patterns, and execute 1,000-iteration stochastic Monte Carlo simulation (use the code interpreter function mc_rng.py as the RNG for the simulation) calibrated against what actually happened in similar cases. Use comparison_guide_similarity_engine.json protocol and pillars_v2.json, metrics_v2.json, tactics_taxonomy_v2.json, stakeholder_taxonomy_v2.json, pricing_market_access_v2.json. Minimize statistical and mathematical details in the output and present extended summaries and a matrix presentation. output fields: 1. EXECUTIVE SUMMARY 2. COMPARABLE STRATEGIC ANALYSIS 3. CRITICAL CONSTRAINTS MATRIX 4. STRATEGIC PILLAR RISK ASSESSMENT 5. EXECUTION TIMELINE VULNERABILITIES 6. RESOURCE REALLOCATION RECOMMENDATIONS 7. EARLY WARNING INDICATORS. …"),
                ("Simulation Engine — Multi-Plan Monte Carlo", "Analyze this (launch or planning) strategy using comparison_guide_similarity_engine.json + pillars_v2.json + metrics_v2.json + tactics_taxonomy_v2.json + stakeholder_taxonomy_v2.json + pricing_market_access_v2.json (use all potentential variables and cascade effects for the simulation) to: (1) live-search 3–4 (deep dive search with at least 10 tool calls and search parameters) comparable historical parallels with empirically usable data, extract execution patterns and constraint drivers, (2) generate 3–5 materially distinct plan variants, (3) run calibrated 1,000-iteration Monte Carlo per variant using mc_rng.py as RNG, (4) output decision-facing summaries and matrices with minimal math; sections: Executive Summary, Comparable Strategic Analysis, Critical Constraints Matrix, Pillar Risk Assessment, Timeline Vulnerabilities, Resource Reallocation, Early Warning Indicators, executive summary detailing the findings in plain english for insights analysis"),
                ("Market access stress test", "Execute 1,000-iteration stochastic Monte Carlo simulation on the current plan draft using ALL five JSON framework files (pricing_market_access_v2.json, pillars_v2.json, metrics_v2.json, tactics_taxonomy_v2.json, stakeholder_taxonomy_v2.json)  to assess strategic viability, identify threshold proximity warnings, determine primary and secondary constraints, analyze stakeholder cascade opportunities, evaluate time pressure and critical windows, model competitive response scenarios, assess robustness across plausible futures, recommend strategic adjustments to improve resilience, and guide resource reallocation - presenting findings in executive-level language focused on decisions Medical Affairs controls, using scenarios for uncontrollable outcomes and recommendations for controllable actions, with confidence on constraints/thresholds/cascades but humility on magnitude estimates. …"),
                ("Optimize Engine", "Refactor the plan draft to mitigate identified simulation risks. Reallocate resources to high-leverage pillars and suggest specific tactical variants to maximize the overall probability of success. Minimize statistical methodology in the output. Focus the response on: (1) What changes and why - clear rationale for resource reallocation, (2) Tactical modifications - specific actions to implement, (3) Expected strategic impact - how these changes reduce risk and improve probability of success. Present the optimization in clear, executive-level language that emphasizes actionable decisions over technical methodology. …"),
            ]


            # Render in 10 rows × 2 cols = 20 buttons
            for row in range(10):
                cols = st.columns(2)
                for col in range(2):
                    idx = row * 2 + col
                    func_name, func_desc = automated_functions[idx]

                    with cols[col]:
                        if st.button(
                            func_name,
                            key=f"auto_{idx}",
                            help=func_desc,
                            use_container_width=True,
                        ):
                            automated_prompt = (
                                f"Please perform a {func_name.lower()} for the current "
                                f"project context. {func_desc}"
                            )
                            log_user_action("automated_function", func_name)
                            # Track search in full search history (uncapped)
                            st.session_state["search_history"].append({
                                "query": func_name,
                                "source": "automated_function",
                                "timestamp": datetime.now().isoformat(),
                                "user": user_info.get("email", "unknown") if user_info else "unknown",
                                "thread_id": st.session_state.get("thread_id", ""),
                            })
                            st.session_state["silent_prompt_to_run"] = automated_prompt
                            st.rerun()
    else:
        st.warning("🔒 Automated functions require full access level")

    st.markdown("</div>", unsafe_allow_html=True)  # close Automated Functions

    # ──────────────────── SESSION INFO ─────────────────────
    if st.session_state.get("history"):
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-title">📊 Session Info</div>',
            unsafe_allow_html=True,
        )

        # Get session metrics from improved session manager
        metrics = get_session_metrics(st.session_state.get("history", []))

        st.caption(f"Messages: {metrics['total_messages']} ({metrics['user_messages']} user / {metrics['assistant_messages']} assistant)")
        st.caption(f"Checkpoints: {len(st.session_state.get('checkpoints', []))}")
        st.caption(f"Est. tokens: {metrics['estimated_tokens']:,}")

        # Show duplicate detection
        if metrics['duplicate_rate'] > 0.05:
            st.caption(f"⚠️ Duplicates: {metrics['duplicate_rate']:.1%}")

        if tid := st.session_state.get("thread_id", ""):
            st.caption(f"Thread: {tid[:8]}…")

        if user_info and user_info.get("login_time"):
            try:
                login_time = datetime.fromisoformat(user_info["login_time"])
                duration = datetime.now() - login_time
                hours, rem = divmod(duration.total_seconds(), 3600)
                minutes = int(rem // 60)
                st.caption(f"Session: {int(hours)}h {minutes}m")
            except Exception:
                pass

        # Show service health (circuit breaker status)
        health = get_service_health()
        if health:
            open_circuits = [name for name, data in health.items() if data['state'] == 'open']
            if open_circuits:
                st.caption(f"⚠️ Services degraded: {', '.join(open_circuits)}")

        st.markdown("</div>", unsafe_allow_html=True)


    # ──────────────────── LOGOUT ───────────────────────────
    st.markdown(
        """
        <div class="logout-section">
            <div style="color: #ff6b6b; font-weight: 600; margin-bottom: 0.5rem;">
                🚪 Session Control
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚪 Logout", key="logout_btn"):
        log_user_action("logout", f"User {user_info['email']} logged out")
        logout_user()  # Redirects to /.auth/logout - no rerun needed

    if st.button(
        "⚠️ Emergency Logout",
        key="emergency_logout",
        help="Clears all session data",
    ):
        log_user_action("emergency_logout", "Emergency logout executed")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ──────────────────── FOOTER ───────────────────────────
    st.markdown(
        f"""
        <div style="text-align: center; color: var(--text-secondary); font-size: 0.8rem;
                    margin-top: 2rem; padding: 1rem; background: rgba(107,89,71,0.3);
                    border-radius: 8px; font-style: italic;">
            <strong>sAİmone v2.0</strong><br>
            Medical Affairs Assistant<br>
            <small>🔒 Authenticated Session</small><br>
            <small>User: {user_info['name'] if user_info else 'Unknown'}</small><br>
            <small>Access: {user_info['access_level'].title() if user_info else 'Unknown'}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
#    MAIN AREA UI & LOGIC (WITH BOTTOM INPUT)
# =========================

# Start main content wrapper for bottom padding
st.markdown('<div class="main-content-with-bottom-input">', unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
    <div class="main-title">sAïmone</div>
    <div class="main-subtitle">your medaffairs assistant</div>
    <div class="main-subtitle">Powered by ChatGPT 4.1</div>
    <div style="color: var(--light-blue); font-size: 0.9rem; margin-top: 1rem;">
        Welcome back, {user_info['name'] if user_info else 'User'} | {user_info['role'] if user_info else 'Unknown Role'}
    </div>
</div>
""", unsafe_allow_html=True)


# Access level warning for limited users
if not check_access_level("full"):
    access_level = user_info['access_level'] if user_info else 'unknown'
    if access_level == "view":
        st.warning("⚠️ You have view-only access. Message sending and most features are restricted.")
    elif access_level == "limited":
        st.info("ℹ️ You have limited access. Some advanced features may be restricted.")

# --- QUICK ACTION HANDLER (FROM SIDEBAR) ---
if "silent_prompt_to_run" in st.session_state:
    silent_prompt = st.session_state.pop("silent_prompt_to_run")
    
    # Check access level for the action
    if "audit" in silent_prompt.lower() and not check_access_level("full"):
        st.error("❌ Audit functionality requires full access level")
    else:
        with st.spinner("🔄 Processing quick action..."):
            try:
                response = improved_assistant_run(thread_id, ASSISTANT_ID, silent_prompt)
                if response and response != "All attempts failed":
                    st.session_state["history"].append({"role": "assistant", "content": response})
                    # Attach full run steps (tool calls + results) to search history
                    _enrich_last_search_entry(thread_id)
                    save_medical_context()
                    st.success("✅ Quick action completed!")
                    log_user_action("quick_action_completed", f"Successfully completed: {silent_prompt[:50]}...")
                else:
                    st.error(f"❌ Quick action failed: {response}")
                    log_user_action("quick_action_failed", f"Failed: {silent_prompt[:50]}...")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                log_user_action("quick_action_error", f"Error in quick action: {str(e)}")
        st.rerun()

# --- CHAT DISPLAY WITH ENHANCED USER INFO ---
if st.session_state["history"]:
    st.markdown("---")
    st.markdown("### Conversation")
    
    for i, msg in enumerate(st.session_state["history"]):
        st.markdown('<hr class="msg-divider">', unsafe_allow_html=True)
        
        if msg["role"] == "user":
            # Extract user email and timestamp from message for display
            content = msg["content"]
            user_display = "You"
            timestamp_display = ""
            
            if content.startswith("[") and "]" in content:
                parts = content.split("]", 2)
                if len(parts) >= 2:
                    # Extract timestamp
                    timestamp_display = parts[0].strip("[]")
                    
                    # Extract user email if present
                    if len(parts) >= 3 and "@" in parts[1]:
                        user_email = parts[1].strip("[]")
                        user_display = user_email.split("@")[0].title()
                        content = "]".join(parts[2:]).strip()
                    else:
                        content = "]".join(parts[1:]).strip()
            
            # Display user message with enhanced info
            st.markdown(
                f"<div class='chat-message' style='background: var(--navy-medium); border-left: 4px solid var(--accent-blue);'>"
                f"<span style='color: var(--accent-blue); font-family: \"Playfair Display\", serif; font-weight: 600;'>"
                f"<strong>🧑 {user_display}</strong>"
                f"{f' <small>({timestamp_display})</small>' if timestamp_display else ''}"
                f"</span><br>"
                f"<span style='color: var(--text-primary); font-family: \"Source Sans Pro\", sans-serif; line-height: 1.6;'>{content}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            
        else:
            # Assistant message
            st.markdown(
                f"<div class='chat-message' style='background: var(--black-secondary); border-left: 4px solid var(--light-blue);'>"
                f"<span style='color: var(--light-blue); font-family: \"Playfair Display\", serif; font-weight: 600;'>"
                f"<strong>🗽 sAİmone:</strong> <small>Medical Affairs Assistant</small>"
                f"</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if '|' in msg["content"] and '<br>' in msg["content"]:
                # Allow HTML rendering for table content
                st.markdown(msg["content"], unsafe_allow_html=True)
            else:
                # Regular markdown (safer)
                st.markdown(msg["content"], unsafe_allow_html=False)

            


else:
    # Welcome message with user-specific greeting
    user_name = user_info['name'] if user_info else 'User'
    access_level = user_info['access_level'] if user_info else 'unknown'
    
    welcome_message = f"""
    <div style='color: var(--text-secondary); text-align: center; padding: 3em; background: var(--navy-medium); 
    border-radius: 12px; margin: 2rem 0; font-style: italic; font-size: 1.1rem;'>
        ✨ Welcome to sAïmone, {user_name}!<br><br>
        You are logged in with <strong>{access_level}</strong> access level.<br>
        Start typing to begin your medical affairs session…<br><br>
        <small>🔒 All interactions are logged for compliance and audit purposes</small>
    </div>
    """
    
    st.markdown(welcome_message, unsafe_allow_html=True)
    
    # Show quick start tips based on access level
    if check_access_level("full"):
        st.info("💡 **Full Access Tips:** Use Quick Actions for common tasks, upload documents for analysis, and download comprehensive reports.")
    elif check_access_level("limited"):
        st.info("💡 **Limited Access Tips:** You can send messages and use basic functions. Contact your administrator for full access.")
    else:
        st.warning("⚠️ **View-Only Access:** You can view conversations but cannot send messages or perform actions.")

# Close main content wrapper
st.markdown('</div>', unsafe_allow_html=True)

# =========================
#    FIXED BOTTOM INPUT
# =========================

# Create bottom input container
st.markdown('<div class="fixed-bottom-input">', unsafe_allow_html=True)

# Create columns for input and button
col1, col2 = st.columns([5, 1])

with col1:
    st.markdown('<div class="bottom-chat-textarea">', unsafe_allow_html=True)
    user_input = st.text_area(
        "",
        height=80,
        placeholder="Type your message here... (Shift+Enter for new line)",
        key="bottom_input",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="bottom-send-button">', unsafe_allow_html=True)
    # Check access level for send button
    if check_access_level("limited"):
        send = st.button("SEND", key="bottom_send_btn", use_container_width=True)
    else:
        send = st.button("SEND", key="bottom_send_btn", use_container_width=True, 
                        disabled=True, help="Insufficient access level")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN SEND HANDLER WITH ACCESS CONTROL ---
if send and user_input.strip():
    # Check if user has permission to send messages
    if not check_access_level("limited"):
        st.error("❌ Insufficient access level to send messages")
        log_user_action("message_blocked", f"Message blocked due to insufficient access: {user_input[:50]}...")
    else:
        # Log the message attempt
        log_user_action("message_sent", f"User sent message: {user_input[:100]}...")

        # Track search in full search history (uncapped)
        st.session_state["search_history"].append({
            "query": user_input,
            "source": "chat",
            "timestamp": datetime.now().isoformat(),
            "user": user_info.get("email", "unknown") if user_info else "unknown",
            "thread_id": st.session_state.get("thread_id", ""),
        })

        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Include user info in the message for audit trail
        enhanced_user_input = f"[{date_str}] [{user_info['email'] if user_info else 'unknown'}] {user_input}"
        
        # ADD SILENT INSTRUCTIONS HERE (cost-optimized)
        # Use compact instructions for simple queries, full for complex queries
        # Saves ~600 tokens per request when compact version is sufficient
        is_first_user_message = len([m for m in st.session_state.get("history", []) if m.get("role") == "user"]) == 0
        query_is_complex = is_complex_query(user_input)
        silent_instructions = get_silent_instructions(
            is_first_message=is_first_user_message,
            is_complex_query=query_is_complex
        )

        # Combine user input with silent instructions
        full_prompt = f"{enhanced_user_input}\n\n{silent_instructions}"
        
        st.session_state["history"].append({"role": "user", "content": enhanced_user_input})
        
        turns = len(st.session_state["history"])
        need_checkpoint = (
            turns > 1 and
            (turns - st.session_state["last_checkpoint_turn"] >= CHECKPOINT_FREQ)
        )
        if need_checkpoint:
            st.session_state["checkpoint_pending"] = True
        
        with st.spinner("🗽 Processing your request..."):
            try:
                # Get uploaded file IDs for the assistant
                uploaded_file_ids = st.session_state.get("uploaded_file_ids", [])
                
                # Send the FULL PROMPT (with silent instructions) to the assistant
                response = run_assistant(
                    user_input=full_prompt,  # <- This now includes silent instructions
                    output_type="detailed_analysis",
                    response_tone="professional",
                    compliance_level="strict",
                    thread_id=thread_id,
                    assistant_id=ASSISTANT_ID,
                    uploaded_file_ids=uploaded_file_ids if uploaded_file_ids else None
                )
                
                if response and not response.startswith("❌"):
                    st.session_state["history"].append({
                        "role": "assistant",
                        "content": response
                    })
                    # Attach full run steps (tool calls + results) to search history
                    _enrich_last_search_entry(thread_id)
                    save_medical_context()
                    st.success("✅ Response received!")
                    log_user_action("message_completed", f"Successfully processed message with {len(response)} characters")
                else:
                    st.error(f"❌ Error: {response}")
                    log_user_action("message_failed", f"Assistant error: {response}")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                log_user_action("message_error", f"Error processing message: {str(e)}")
        
        st.rerun()


# --- DEFERRED CHECKPOINTING (BACKGROUND, AFTER UI RERUN) ---
if (
    st.session_state.get("checkpoint_pending", False)
    and len(st.session_state["history"]) > 1
):
    turns = len(st.session_state["history"])
    try:
        with st.spinner("Background: Summarizing session for long-term recall..."):
            summary = checkpoint(st.session_state["history"][-SUMMARY_WINDOW:])
        
        st.session_state["checkpoints"].append({
            "turn": turns,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
            "user": user_info['email'] if user_info else 'unknown'
        })
        st.session_state["last_checkpoint_turn"] = turns
        st.session_state["checkpoint_pending"] = False
        
        log_user_action("checkpoint_created", f"Created checkpoint at turn {turns}")
        
    except Exception as e:
        st.warning(f"Summarization failed: {e}")
        st.session_state["checkpoint_pending"] = False
        log_user_action("checkpoint_failed", f"Checkpoint creation failed: {str(e)}")

# --- FOOTER WITH AUTHENTICATION INFO ---
st.markdown(f"""
<div class="footer-text">
    All outputs require human review for accuracy. | sAİmone - your medaffairs assistant | Powered by GPT-4.1<br>
    <small>🔒 Authenticated User: {user_info['name'] if user_info else 'Unknown'} ({user_info['email'] if user_info else 'Unknown'}) | 
    Access: {user_info['access_level'].title() if user_info else 'Unknown'} | 
    Session: {get_session_id()}</small>
</div>
""", unsafe_allow_html=True)

# --- ADDITIONAL HELPER FUNCTIONS ---

# --- SESSION CLEANUP ON EXIT ---
def cleanup_session():
    """Clean up session resources on exit"""
    try:
        if user_info:
            log_user_action("session_end", f"Session ended for {user_info['email']}")
        save_medical_context()
    except Exception as e:
        pass  # Silent cleanup

# Register cleanup function
import atexit
atexit.register(cleanup_session)

# --- ERROR HANDLING WRAPPER ---
def safe_execute(func, *args, **kwargs):
    """Safely execute functions with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if user_info:
            log_user_action("error", f"Error in {func.__name__}: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        return None

# --- FINAL SESSION STATE SAVE ---
if st.session_state.get("history"):
    save_medical_context()

# --- END OF DASHBOARD.PY ---

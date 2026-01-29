# auth_popup.py — Simplified honor system with cleaner disclaimer flow
# Relies on Azure Easy Auth blocking app access, uses honor system for user details

import json
import streamlit as st
from datetime import datetime
import os
import re
import html
import secrets


# -----------------------------
# 1) Pharma disclaimer (unchanged from your original)
# -----------------------------
def show_pharma_disclaimer():
    """Display the pharma compliance disclaimer with checkboxes."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #4a90e2;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    ">
        <h2 style="color: #87ceeb; text-align: center; margin-bottom: 1.5rem; font-family: 'Playfair Display', serif;">
            🧬 Pharmaceutical & Medical Communications Disclaimer
        </h2>
    </div>
    """, unsafe_allow_html=True)

    disclaimers = [
        {
            "key": "professional_use",
            "text": "I confirm I am a pharmaceutical or medical communications professional with appropriate credentials and training."
        },
        {
            "key": "regulatory_compliance",
            "text": "I understand this system assists with medical/market access strategy and must adhere to regulatory compliance per applicable guidelines (FDA, EMA, etc.)."
        },
        {
            "key": "medical_accuracy",
            "text": "I acknowledge that all medical information and outputs must be verified by qualified medical professionals before use in any official capacity."
        },
        {
            "key": "confidentiality",
            "text": "I agree to maintain confidentiality of any client or patient information and comply with all applicable data privacy regulations (GDPR, HIPAA, etc.)."
        },
        {
            "key": "mlr_review",
            "text": "I understand that all materials generated may require Medical, Legal, and Regulatory (MLR) review before external distribution."
        },
        {
            "key": "liability",
            "text": "I accept that this is an AI-assisted tool and final responsibility for decisions and outputs remains with qualified human professionals."
        },
    ]

    all_checked = True
    for item in disclaimers:
        checked = st.checkbox(
            item["text"],
            key=f"disc_{item['key']}",
            value=st.session_state.get(f"disc_{item['key']}", False)
        )
        if not checked:
            all_checked = False

    st.markdown("""
    ---
    **Additional Information:**
    - All sessions are logged for compliance and audit purposes  
    - Generated content is for internal use only unless approved through proper channels  
    - Users are responsible for ensuring compliance with local regulations  
    - System administrators may monitor usage for quality assurance
    - [Terms of Service](/terms) | [Privacy Policy](/privacy)
    """)

    return all_checked


# -----------------------------
# 2) Honor system user capture
# -----------------------------
def capture_user_info():
    """
    Honor system form for user details.
    Azure Easy Auth already blocks unauthorized access.
    """
    
    # Check if already captured
    if st.session_state.get("user_captured") and st.session_state.get("user_info"):
        return st.session_state["user_info"]
    
    # Honor system - manual entry
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1b263b 0%, #0d1b2a 100%); 
                padding: 2rem; border-radius: 12px; border: 1px solid #4a90e2; 
                margin-bottom: 2rem;">
        <h3 style="color: #87ceeb; margin-bottom: 1rem;">👤 Complete Your Profile</h3>
        <p style="color: #b0c4de;">Please provide your information to continue. 
        Access is already secured via Azure authentication.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("user_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Full Name *",
                placeholder="Dr. Jane Smith",
                help="Your full name as it should appear in reports"
            )
            email = st.text_input(
                "Email Address *",
                placeholder="jane.smith@pharma.com",
                help="Your professional email address"
            )
        
        with col2:
            role = st.selectbox(
                "Role *",
                [
                    "",
                    "Other/General",
                    "Global Medical Affairs Director",
                    "Regional Medical Affairs Manager",
                    "Medical Science Liaison (MSL)",
                    "Medical Information Specialist",
                    "Medical Communications Manager",
                    "Medical Writing Lead",
                    "HEOR/RWE Manager",
                    "Clinical Affairs Manager",
                    "Regulatory Medical Writer",
                    "Patient Advocacy Lead",
                    "KOL Engagement Manager",
                    "Digital Medical Strategy Lead"
                ],
                help="Your primary role"
            )
            organization = st.text_input(
                "Organization",
                placeholder="Pharma Corp",
                help="Your company or organization (optional)"
            )

        # Add Thread ID field for restoring previous sessions
        thread_id_input = st.text_input(
            "Restore from Thread ID (optional)",
            placeholder="paste existing thread id to resume an older session",
            help="If you have a prior thread id (from another session), paste it here to reuse it."
        )

        # Compliance certification
        st.markdown("### 📋 Compliance Certification")
        certify = st.checkbox(
            "I certify that the information provided above is accurate and matches my authenticated credentials",
            key="certify_info"
        )
        
        submitted = st.form_submit_button(
            "Continue",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            if not name or not email or not role or role == "":
                st.error("Please fill in all required fields (marked with *)")
            elif not certify:
                st.error("Please certify that your information is accurate")
            else:
                # Valid submission
                user_info = {
                    "name": name,
                    "email": email,
                    "role": role,
                    "organization": organization or "Not specified",
                    "method": "honor_system",
                    "certified_at": datetime.utcnow().isoformat() + "Z"
                }
                
                # If user provided a thread id, store it in session_state
                if thread_id_input and isinstance(thread_id_input, str) and thread_id_input.strip():
                    st.session_state["thread_id"] = thread_id_input.strip()
                
                st.session_state["user_info"] = user_info
                st.session_state["user_captured"] = True
                st.success("✅ Profile completed successfully")
                st.rerun()
    
    # If we get here, we're waiting for form submission
    st.stop()


# -----------------------------
# 3) Main authentication flow (cleaner from version 8)
# -----------------------------
def show_auth_popup():
    """
    Main entry point for authentication and disclaimers.
    Returns True when user is authenticated and disclaimers accepted.
    """
    
    # Check if already completed
    if (st.session_state.get("disclaimer_accepted") and 
        st.session_state.get("user_captured")):
        return True
    
    # Step 1: Capture user info
    user_info = capture_user_info()
    
    # Step 2: Show user info and disclaimers
    if user_info:
        # Display captured user info
        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown(f"**👤 User:** {user_info.get('name')}")
            with col2:
                st.markdown(f"**📧 Email:** {user_info.get('email')}")
            with col3:
                st.markdown(f"**Auth:** 🛡️ Azure")
            
            if user_info.get('role'):
                st.caption(f"Role: {user_info.get('role')} | Organization: {user_info.get('organization', 'N/A')}")
        
        # Step 3: Disclaimers
        if not st.session_state.get("disclaimer_accepted"):
            accepted = show_pharma_disclaimer()
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(
                    "✅ Accept Terms & Continue",
                    type="primary",
                    use_container_width=True,
                    disabled=not accepted
                ):
                    if accepted:
                        st.session_state["disclaimer_accepted"] = True
                        st.session_state["disclaimer_timestamp"] = datetime.utcnow().isoformat() + "Z"
                        
                        # Log the complete authentication event
                        log_user_action("authentication_complete", {
                            "user": user_info.get("name"),
                            "email": user_info.get("email"),
                            "method": user_info.get("method"),
                            "disclaimer_accepted": True
                        })
                        
                        st.rerun()
            
            with col2:
                if st.button("❌ Decline", use_container_width=True):
                    st.error("You must accept the terms to continue")
                    st.stop()
            
            st.stop()
    
    return True


# -----------------------------
# 4) Helper functions for main.py
# -----------------------------
def get_user_info():
    """Get user info for use in main app"""
    user_info = st.session_state.get("user_info", {})
    
    # Ensure all expected fields exist with defaults
    return {
        "name": user_info.get("name", "Unknown User"),
        "email": user_info.get("email", "unknown@example.com"),
        "role": user_info.get("role", "User"),
        "organization": user_info.get("organization", "Unknown"),
        "client": user_info.get("organization", "Unknown"),  # Alias for compatibility
        "tenant_id": user_info.get("id", "unknown"),
        "object_id": user_info.get("id", "unknown"),
        "access_level": "full",  # Default to full for authenticated users
        "disclaimer_accepted": st.session_state.get("disclaimer_accepted", False),
        "disclaimer_timestamp": st.session_state.get("disclaimer_timestamp"),
        "auth_method": user_info.get("method", "honor_system"),
        "login_time": st.session_state.get("login_time", datetime.now().isoformat()),
        "thread_id": st.session_state.get("thread_id", "")   # Expose thread ID
    }


def check_access_level(required_level="limited"):
    """Check if user has required access level"""
    if not st.session_state.get("disclaimer_accepted"):
        return False
    
    # All authenticated users get full access
    return True


def get_access_restrictions():
    """Get list of restrictions for current user"""
    # No restrictions for any role - everyone gets full access
    return []


def validate_session():
    """Check if session is valid"""
    return (st.session_state.get("user_captured") and 
            st.session_state.get("disclaimer_accepted"))


def logout_user():
    """Display logout link"""
    st.markdown("[🚪 Sign out](/.auth/logout?post_logout_redirect_uri=/)")


def log_user_action(action: str, meta=None):
    """Log user actions for audit trail"""
    try:
        user_info = get_user_info()
        
        if isinstance(meta, str):
            meta_dict = {"details": meta}
        else:
            meta_dict = meta or {}
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user_info.get("name"),
            "email": user_info.get("email"),
            "action": action,
            "meta": meta_dict,
            "auth_method": user_info.get("auth_method"),
            "session_id": st.session_state.get("session_id", "unknown")
        }
        
        # Store in session state
        if "audit_log" not in st.session_state:
            st.session_state["audit_log"] = []
        st.session_state["audit_log"].append(entry)
        
        # In production, also send to:
        # - Azure Application Insights
        # - Blob Storage
        # - Database
        
        # For debugging
        print(f"[AUDIT] {json.dumps(entry)}")
        
    except Exception as e:
        print(f"Audit log error: {e}")


def sync_with_cloud_db():
    """Placeholder for future cloud sync functionality"""
    # In future, this could sync user preferences, settings, etc.
    return True


# -----------------------------
# 5) Session initialization with stronger session IDs
# -----------------------------
if "session_id" not in st.session_state:
    # Use cryptographically secure random session ID
    st.session_state["session_id"] = f"session_{secrets.token_urlsafe(32)}"

if "login_time" not in st.session_state:
    st.session_state["login_time"] = datetime.now().isoformat()

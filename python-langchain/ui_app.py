"""
Streamlit UI wrapper for the LangGraph Event Planner app.
Professional, production-ready design with full functionality.

Run with: streamlit run ui_app.py
"""

import streamlit as st
import asyncio
import os
from pathlib import Path
from run_task import run_task


# Page configuration with professional styling
st.set_page_config(
    page_title="Event Planner Pro",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional design
st.markdown(
    """
    <style>
    /* Main page styling */
    .main { max-width: 1400px; }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e8eef5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    /* Status messages */
    .success-message {
        background: #f0f9ff;
        border-left: 4px solid #10b981;
        color: #065f46;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .info-message {
        background: #f3f4f6;
        border-left: 4px solid #3b82f6;
        color: #1f2937;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    /* Form styling */
    .form-group {
        margin-bottom: 1rem;
    }
    
    .form-label {
        font-weight: 500;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Download buttons */
    .download-section {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .file-item {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .file-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
    }
    
    /* Output text area */
    .output-container {
        background: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Task type badge */
    .task-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_file_bytes(file_path: str) -> bytes:
    """Read file as bytes for download."""
    with open(file_path, "rb") as f:
        return f.read()


def get_file_name(file_path: str) -> str:
    """Get just the filename."""
    return Path(file_path).name


def get_file_icon(file_path: str) -> str:
    """Get icon based on file extension."""
    ext = Path(file_path).suffix.lower()
    icons = {
        ".png": "🖼️",
        ".jpg": "🖼️",
        ".jpeg": "🖼️",
        ".txt": "📄",
        ".json": "⚙️",
        ".pdf": "📕",
    }
    return icons.get(ext, "📁")


def main():
    # Initialize session state
    if "result" not in st.session_state:
        st.session_state.result = None

    # Header
    st.markdown(
        """
        <div class="header-container">
            <h1 class="header-title">📅 Event Planner Pro</h1>
            <p class="header-subtitle">AI-Powered Event Content Generation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main layout
    col_form, col_action = st.columns([3, 1], gap="large")

    # ========== LEFT COLUMN: FORM ==========
    with col_form:
        st.markdown(
            '<div class="card"><div class="card-header">⚙️ Event Configuration</div>',
            unsafe_allow_html=True,
        )

        # Task Type Selection
        st.markdown(
            '<p class="section-header">Task Type</p>', unsafe_allow_html=True
        )
        task_type = st.selectbox(
            "Select the type of content to generate",
            [
                "flyer",
                "youth_registration_form",
                "basketball_clinic",
                "proposal_email",
            ],
            help="Choose what you want to create",
            label_visibility="collapsed",
        )

        # Display task description
        task_descriptions = {
            "flyer": "Create an attractive event flyer with QR code for registration",
            "youth_registration_form": "Design a registration form with QR code link",
            "basketball_clinic": "Generate specialized content for basketball events",
            "proposal_email": "Draft a professional proposal email to leadership",
        }
        st.info(task_descriptions.get(task_type, ""))

        # Event Details Section
        st.markdown(
            '<p class="section-header">Event Details</p>', unsafe_allow_html=True
        )

        col1, col2 = st.columns(2, gap="medium")

        with col1:
            event_name = st.text_input(
                "Event Name",
                value="Youth Basketball Clinic",
                placeholder="E.g., Youth Basketball Clinic",
            )
            event_date = st.text_input(
                "Event Date",
                value="2026-04-15",
                placeholder="YYYY-MM-DD",
                help="Date in YYYY-MM-DD format",
            )

        with col2:
            event_time = st.text_input(
                "Event Time",
                value="3:00 PM - 5:00 PM",
                placeholder="E.g., 3:00 PM - 5:00 PM",
            )
            location = st.text_input(
                "Location",
                value="City Sports Complex",
                placeholder="E.g., City Sports Complex",
            )

        # Form URL
        form_url = st.text_input(
            "Registration Form URL",
            value="https://forms.example.com/register",
            placeholder="https://forms.example.com/register",
            help="This URL will be embedded in QR codes",
        )

        # Additional Details (Collapsible)
        with st.expander("📝 Additional Details", expanded=False):
            description = st.text_area(
                "Event Description",
                value="Join us for an exciting youth basketball clinic!",
                placeholder="Describe your event...",
                height=80,
            )
            contact_email = st.text_input(
                "Contact Email",
                value="pastor@church.org",
                placeholder="pastor@church.org",
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== RIGHT COLUMN: ACTION ==========
    with col_action:
        st.markdown(
            '<div class="card" style="text-align: center;">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="card-header" style="justify-content: center;">Ready to Begin?</p>',
            unsafe_allow_html=True,
        )

        # Run button
        if st.button(
            "🚀 Generate Content",
            use_container_width=True,
            type="primary",
            key="run_button",
        ):
            # Build payload
            payload = {
                "task_type": task_type,
                "event_details": {
                    "event_name": event_name,
                    "event_date": event_date,
                    "event_time": event_time,
                    "location": location,
                    "form_url": form_url,
                    "description": description,
                    "contact_email": contact_email,
                },
            }

            # Run task with progress indicator
            progress_container = st.container()
            with progress_container:
                with st.spinner("⏳ Generating content..."):
                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            run_task(task_type, payload)
                        )
                        loop.close()

                        st.session_state.result = result

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.result = None

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== RESULTS SECTION ==========
    if st.session_state.result:
        result = st.session_state.result

        st.markdown("---")
        st.markdown(
            '<div class="card"><div class="card-header">✅ Generation Complete</div>',
            unsafe_allow_html=True,
        )

        # Task badge
        st.markdown(
            f"<span class='task-badge'>{task_type.replace('_', ' ').title()}</span>",
            unsafe_allow_html=True,
        )

        # Output section
        st.markdown(
            '<p class="section-header">AI-Generated Output</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="output-container">', unsafe_allow_html=True)
        st.text_area(
            "Content",
            value=result["final_text"],
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Files section
        if result["files"]:
            st.markdown(
                '<p class="section-header">📁 Generated Files</p>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="download-section">', unsafe_allow_html=True
            )

            for file_path in result["files"]:
                if os.path.exists(file_path):
                    file_name = get_file_name(file_path)
                    file_bytes = read_file_bytes(file_path)
                    file_icon = get_file_icon(file_path)

                    # Create a nice file item with download button
                    col_file, col_btn = st.columns([4, 1], gap="small")

                    with col_file:
                        st.markdown(
                            f"<div class='file-item'><span class='file-icon'>{file_icon}</span><span>{file_name}</span></div>",
                            unsafe_allow_html=True,
                        )

                    with col_btn:
                        st.download_button(
                            label="⬇️",
                            data=file_bytes,
                            file_name=file_name,
                            key=f"download_{file_path}",
                            use_container_width=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("ℹ️ No files were generated for this task type.")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("---")
        st.markdown(
            """
            <div class="card">
                <div style="text-align: center; padding: 2rem 0;">
                    <h3 style="color: #667eea; margin-bottom: 1rem;">👈 Ready to Generate?</h3>
                    <p style="color: #666; font-size: 0.95rem;">
                        Configure your event details on the left, select your task type, 
                        and click "Generate Content" to begin.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

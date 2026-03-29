import streamlit as st
import pandas as pd
import os
import datetime

# Retrieve submission summary from session state
summary = st.session_state.get('submission_summary', None)

def get_all_registrations():
    import sqlite3
    DB_PATH = "registrations.db"
    TABLE_NAME = "registrations"
    import pandas as pd
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    return df

st.set_page_config(page_title="Registration Summary", layout="wide")
st.markdown("<h1 style='color: purple; text-align: center;'>Registration Submitted</h1>", unsafe_allow_html=True)

if summary:
    st.success("🎉 Registration submitted successfully!")
    st.balloons()
    st.subheader("📄 Your Submission Summary")
    st.markdown(f"**Youth Name:** {summary.get('youth_first_last_name', '')}")
    st.markdown(f"**Age:** {summary.get('youth_age', '')}")
    st.markdown(f"**Parent/Guardian:** {summary.get('parent_first_last_name', '')}")
    st.markdown(f"**Phone:** {summary.get('parent_phone', '')}")
    st.markdown(f"**Transportation Needed:** {summary.get('transportation_needed', '')}")
    st.markdown(f"**Special Needs:** {summary.get('special_needs', 'None') or 'None'}")
    st.markdown(f"**Consent Given:** {'Yes' if summary.get('consent') else 'No'}")
    st.markdown(f"**Signature:** {summary.get('signature', '')}")
    st.markdown(f"**Date:** {summary.get('date', '')}")
    st.markdown(f"---")
    st.markdown(f"**Event:** {summary.get('event_name', '')}")
    st.markdown(f"**Event Date:** {summary.get('date_event', '')}")
    st.markdown(f"**Time:** {summary.get('time', '')}")
    st.markdown(f"**Location:** {summary.get('location', '')}")
    # Download CSV button for all registrations
    df = get_all_registrations()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download All Registrations (CSV)", csv, "registrations.csv", "text/csv")
    # Option to return to form
    if st.button("⬅️ Register Another Child"):
        st.session_state['submission_summary'] = None
        st.switch_page('Registration Form')
else:
    st.warning("No submission found. Please fill out the registration form first.")
    if st.button("Go to Registration Form"):
        st.switch_page('Registration Form')

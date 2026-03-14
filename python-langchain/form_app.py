
import streamlit as st
import json
import pandas as pd
from io import StringIO
import datetime
import sqlite3
import os

# Paste your app's form JSON here (from app_copy_2.py output)
form_json = '''{
  "title": "Spring Youth Basketball Clinic Registration Form",
  "event_details": {
    "event_name": "Spring Youth Basketball Clinic",
    "date": "April 15-17, 2026",
    "time": "6-8 PM daily",
    "location": "Louisville Community Church Gym",
    "max_participants": 20,
    "age_range": "12-17",
    "description": "Fun skills clinic with drills, games, and faith talks. Includes transportation help and special needs options."
  },
  "fields": [
    {"name": "youth_first_last_name", "label": "Youth First and Last Name", "type": "text", "required": true},
    {"name": "youth_age", "label": "Youth Age", "type": "number", "required": true},
    {"name": "parent_first_last_name", "label": "Parent/Guardian First and Last Name", "type": "text", "required": true},
    {"name": "parent_phone", "label": "Parent/Guardian Phone Number", "type": "tel", "required": true},
    {"name": "transportation_needed", "label": "Need transportation?", "type": "select", "required": true, "options": [{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}]},
    {"name": "special_needs", "label": "Special accommodations? (If yes, specify)", "type": "textarea", "required": false},
    {"name": "consent", "label": "I give permission for my child to participate.", "type": "checkbox", "required": true},
    {"name": "signature", "label": "Parent/Guardian Signature", "type": "text", "required": true},
    {"name": "date", "label": "Date", "type": "date", "required": true}
  ]
}'''

@st.cache_data
def load_form():
    return json.loads(form_json)

st.set_page_config(page_title="Youth Registration", layout="wide")

form_data = load_form()

# Initialize session state for page navigation
if 'show_summary' not in st.session_state:
    st.session_state['show_summary'] = False
if 'submission_summary' not in st.session_state:
    st.session_state['submission_summary'] = None

# Show summary page if submission was successful
if st.session_state.get('show_summary', False):
    st.markdown("<h1 style='color: purple; text-align: center;'>Registration Submitted</h1>", unsafe_allow_html=True)
    st.success("🎉 Registration submitted successfully!")
    st.balloons()
    st.subheader("📄 Your Submission Summary")
    summary = st.session_state.get('submission_summary', {})
    if summary:
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
    # Download feature removed: parents cannot download the dataset
    # Option to return to form
    if st.button("⬅️ Register Another Child"):
        st.session_state['show_summary'] = False
        st.session_state['submission_summary'] = None
        st.rerun()
    st.stop()  # Stop execution here, don't show form

# Show registration form
st.markdown("<h1 style='color: purple; text-align: center;'>Spring Youth Basketball Clinic Registration</h1>", unsafe_allow_html=True)

# st.title(form_data.get("title", "Youth Registration Form"))

if "description" in form_data:
    st.write(form_data["description"])
else:
    st.info("📋 Fill out the form below to register.")


# SQLite database setup
DB_PATH = "registrations.db"
TABLE_NAME = "registrations"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_table_if_not_exists(fields):
    # Build SQL for dynamic fields
    columns = [
        f"{f['name']} TEXT" if f["type"] not in ["number", "date", "checkbox"] else (
            f"{f['name']} INTEGER" if f["type"] == "number" else (
                f"{f['name']} BOOLEAN" if f["type"] == "checkbox" else f"{f['name']} TEXT"
            )
        )
        for f in fields
    ]
    # Add timestamp and event columns
    columns += [
        "timestamp TEXT",
        "event_name TEXT",
        "date_event TEXT",
        "time TEXT",
        "location TEXT",
        "max_participants INTEGER",
        "age_range TEXT",
        "description TEXT"
    ]
    sql = f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} (id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(columns)})"
    with get_db_connection() as conn:
        conn.execute(sql)
        conn.commit()

def insert_registration(submission):
    keys = list(submission.keys())
    values = [submission[k] for k in keys]
    placeholders = ",".join(["?" for _ in keys])
    sql = f"INSERT INTO {TABLE_NAME} ({', '.join(keys)}) VALUES ({placeholders})"
    with get_db_connection() as conn:
        conn.execute(sql, values)
        conn.commit()

def get_all_registrations():
    with get_db_connection() as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    return df

# Create table if needed
create_table_if_not_exists(form_data.get("fields", []))

with st.form(key="youth_registration"):
    form_values = {}
    for field in form_data.get("fields", []):
        f_name = field["name"]
        f_label = field["label"]
        f_type = field.get("type", "text")
        f_required = field.get("required", False)

        if f_type == "select":
            options = [opt.get("label", str(opt)) for opt in field.get("options", [])]
            form_values[f_name] = st.selectbox(f_label, options, key=f_name)
        elif f_type == "textarea":
            form_values[f_name] = st.text_area(f_label, key=f_name, height=100)
        elif f_type == "checkbox":
            form_values[f_name] = st.checkbox(f_label, key=f_name)
        elif f_type == "number":
            form_values[f_name] = st.number_input(f_label, min_value=0, max_value=100, step=1, key=f_name)
        elif f_type == "date":
            form_values[f_name] = st.date_input(f_label, value=datetime.date(2026, 4, 15), key=f_name)
        elif f_type == "tel":
            form_values[f_name] = st.text_input(f_label, placeholder="(123) 456-7890", help="Phone format: (123) 456-7890", key=f_name)
        elif f_type == "email":
            form_values[f_name] = st.text_input(f_label, placeholder="name@example.com", type="default", key=f_name)
        else:  # text default
            form_values[f_name] = st.text_input(f_label, key=f_name)
    # Submit button OUTSIDE loop
    col1, col2 = st.columns([4, 1])
    with col2:
        submitted = st.form_submit_button("✅ Register", use_container_width=True)




if submitted:
    # Basic validation
    missing = [name for name, val in form_values.items() if not val and any(f.get("required") for f in form_data["fields"] if f["name"] == name)]
    if missing:
        st.error(f"❌ Required fields missing: {', '.join(missing)}")
    else:
        # Add timestamp and event info
        submission = {
            **form_values,
            "timestamp": datetime.datetime.now().isoformat(),
            "event_name": form_data.get("event_details", {}).get("event_name", ""),
            "date_event": form_data.get("event_details", {}).get("date", ""),
            "time": form_data.get("event_details", {}).get("time", ""),
            "location": form_data.get("event_details", {}).get("location", ""),
            "max_participants": form_data.get("event_details", {}).get("max_participants", 0),
            "age_range": form_data.get("event_details", {}).get("age_range", ""),
            "description": form_data.get("event_details", {}).get("description", "")
        }
        insert_registration(submission)
        st.session_state['submission_summary'] = submission
        st.session_state['show_summary'] = True
        st.rerun()




# Sidebar for JSON update
with st.sidebar:
    st.image("NewburgCOCLogo.png", width=150)
    st.markdown("---")  # Adds a horizontal line for separation
    st.header("Event Details")
    # new_json = st.text_area("Paste new form JSON", value=form_json, height=300)
    st.write("Date: April 15-17, 2026")
    st.write("Time: 6-8 PM daily")
    st.write("Location: Newburg Church of Christ Gym")
    # if st.button("Reload Form"):
    #     st.cache_data.clear()
    #     st.rerun()
    # st.info("💡 Run `streamlit run form_app.py`")

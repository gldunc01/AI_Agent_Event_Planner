
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

st.title(form_data.get("title", "Youth Registration Form"))

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
        st.success("🎉 Registration submitted successfully!")
        st.balloons()
        # Show submitted data
        st.subheader("📄 Your Submission")
        st.json(submission)
        # Download CSV
        df = get_all_registrations()
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download All Registrations (CSV)", csv, "registrations.csv", "text/csv")


# Show past registrations from database
df = get_all_registrations()
if not df.empty:
    st.subheader(f"📊 All Registrations ({len(df)})")
    # Show all columns, or a subset if you want
    st.dataframe(df)

# Sidebar for JSON update
with st.sidebar:
    st.header("🔧 Form Config")
    new_json = st.text_area("Paste new form JSON", value=form_json, height=300)
    if st.button("Reload Form"):
        st.cache_data.clear()
        st.rerun()
    st.info("💡 Run `streamlit run form_app.py`")

# Example: How to Integrate Waiver Storage in Your Streamlit Form App

This shows how to connect the waiver feature to your `form_app.py`.

## Step 1: Import the Waiver Function

```python
import streamlit as st
import json
from pathlib import Path
import sys

# Add the python-langchain directory to path so we can import app_copy_2
sys.path.insert(0, str(Path(__file__).parent))
from app_copy_2 import save_signed_waiver
```

## Step 2: Load the Form Schema

```python
# Load form configuration
with open("current_event_form.json", "r") as f:
    form_config = json.load(f)

event_details = form_config["event_details"]
form_fields = form_config["fields"]
```

## Step 3: Render the Form with Waiver Section

```python
st.title(form_config["title"])
st.write(form_config["description"])

# Create a form for input
with st.form("registration_form"):
    form_data = {}
    current_section = None
    
    for field in form_fields:
        # Handle section headers
        if field.get("type") == "section_header":
            st.divider()
            st.subheader(field.get("section"))
            current_section = field.get("section")
            continue
        
        field_name = field["name"]
        field_label = field["label"]
        field_type = field["type"]
        required = field.get("required", False)
        
        # Render different field types
        if field_type == "text":
            form_data[field_name] = st.text_input(field_label, key=field_name)
        elif field_type == "number":
            form_data[field_name] = st.number_input(field_label, key=field_name)
        elif field_type == "tel":
            form_data[field_name] = st.text_input(field_label, key=field_name)
        elif field_type == "date":
            form_data[field_name] = st.date_input(field_label, key=field_name)
        elif field_type == "textarea":
            form_data[field_name] = st.text_area(field_label, key=field_name)
        elif field_type == "select":
            options = [opt["label"] for opt in field.get("options", [])]
            form_data[field_name] = st.selectbox(field_label, options, key=field_name)
        elif field_type == "checkbox":
            form_data[field_name] = st.checkbox(field_label, key=field_name)
    
    submitted = st.form_submit_button("✅ Submit Registration & Sign Waiver")

# Step 4: Process Form Submission
if submitted:
    # Validate required fields
    required_fields = [f["name"] for f in form_fields if f.get("required")]
    missing = [f for f in required_fields if not form_data.get(f)]
    
    if missing:
        st.error(f"❌ Missing required fields: {', '.join(missing)}")
    else:
        # Validate waiver acknowledgment
        if not form_data.get("waiver_acknowledgment"):
            st.error("❌ You must acknowledge the waiver to continue")
        else:
            try:
                # Save the standard registration (your existing code)
                # save_registration(form_data)  # Your existing function
                
                # Save the signed waiver
                waiver_result = save_signed_waiver(
                    participant_info=form_data,
                    event_name=event_details["event_name"]
                )
                
                st.success("✅ Registration Submitted & Waiver Signed!")
                st.write(f"**Thank you!** Your waiver has been recorded:")
                st.write(f"- **Event**: {waiver_result['event']}")
                st.write(f"- **Participant**: {waiver_result['participant']}")
                st.write(f"- **Recorded**: {waiver_result['timestamp']}")
                
                # Display waiver file location
                st.info(f"📁 Waiver saved to: `{waiver_result['file_path']}`")
                
            except Exception as e:
                st.error(f"❌ Error processing registration: {str(e)}")
```

## Step 5: Display Waiver PDF (Optional)

To show the waiver PDF in the form before submission:

```python
# Display waiver PDF section
if st.checkbox("📄 View Full Waiver PDF"):
    waiver_path = Path("waivers/Waiver.pdf")
    if waiver_path.exists():
        with open(waiver_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF",
                data=pdf_file.read(),
                file_name="Waiver.pdf",
                mime="application/pdf"
            )
        # Note: To display PDF preview, you'd need pdfplumber or similar
        st.info("Please review the waiver above before checking the acknowledgment box")
    else:
        st.warning("⚠️ Waiver PDF not found")
```

## Complete Example Form App

```python
import streamlit as st
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))
from app_copy_2 import save_signed_waiver

# Configure page
st.set_page_config(page_title="Registration Form", layout="wide")

# Load form configuration
form_config_path = Path("current_event_form.json")
if not form_config_path.exists():
    st.error("❌ Form configuration not found. Please run the event planner first.")
    st.stop()

with open(form_config_path, "r") as f:
    form_config = json.load(f)

event_details = form_config["event_details"]
form_fields = form_config["fields"]

# Display header
st.title(form_config["title"])
st.write(form_config["description"])

# Display event details
with st.expander("📌 Event Details"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Date", event_details["date"])
    col2.metric("⏰ Time", event_details["time"])
    col3.metric("📍 Location", event_details["location"])
    col4.metric("👥 Max Participants", event_details["max_participants"])

# Create registration form
with st.form("registration_form"):
    form_data = {}
    
    for field in form_fields:
        # Section headers
        if field.get("type") == "section_header":
            st.divider()
            st.subheader(f"📋 {field.get('section')}")
            continue
        
        field_name = field["name"]
        field_label = field["label"]
        field_type = field["type"]
        required_text = " *" if field.get("required") else ""
        
        # Render field types
        if field_type == "text":
            form_data[field_name] = st.text_input(
                field_label + required_text, 
                key=field_name
            )
        elif field_type == "number":
            form_data[field_name] = st.number_input(
                field_label + required_text,
                min_value=0,
                key=field_name
            )
        elif field_type == "tel":
            form_data[field_name] = st.text_input(
                field_label + required_text,
                key=field_name
            )
        elif field_type == "date":
            form_data[field_name] = str(st.date_input(
                field_label + required_text,
                value=datetime.now().date(),
                key=field_name
            ))
        elif field_type == "textarea":
            form_data[field_name] = st.text_area(
                field_label + required_text,
                key=field_name
            )
        elif field_type == "select":
            options = [opt["label"] for opt in field.get("options", [])]
            form_data[field_name] = st.selectbox(
                field_label + required_text,
                options,
                key=field_name
            )
        elif field_type == "checkbox":
            form_data[field_name] = st.checkbox(
                field_label + required_text,
                key=field_name
            )
    
    col1, col2 = st.columns(2)
    with col1:
        submitted = st.form_submit_button("✅ Submit Registration", use_container_width=True)
    with col2:
        reset = st.form_submit_button("🔄 Clear Form", use_container_width=True)

# Handle submission
if submitted:
    required_fields = [f["name"] for f in form_fields if f.get("required") and f.get("type") != "section_header"]
    missing = [f for f in required_fields if not form_data.get(f)]
    
    if missing:
        st.error(f"❌ Missing required fields: {', '.join(missing)}")
    elif not form_data.get("waiver_acknowledgment"):
        st.error("❌ Please acknowledge the waiver to continue")
    else:
        try:
            # Process waiver
            waiver_result = save_signed_waiver(
                participant_info=form_data,
                event_name=event_details["event_name"]
            )
            
            # Success message
            st.success("✅ Registration Submitted!")
            st.balloons()
            
            # Display confirmation
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 📝 Confirmation")
                st.write(f"**Youth**: {form_data.get('youth_first_last_name')}")
                st.write(f"**Parent**: {form_data.get('parent_first_last_name')}")
                st.write(f"**Event**: {event_details['event_name']}")
                st.write(f"**Date**: {event_details['date']}")
            
            with col2:
                st.write("### ✍️ Waiver Status")
                st.write(f"**Signed by**: {form_data.get('waiver_signature')}")
                st.write(f"**Date Signed**: {form_data.get('waiver_date')}")
                st.write(f"**Stored**: {waiver_result['timestamp']}")
            
            st.info(f"📁 Waiver record saved securely")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
```

This example shows:
- ✅ Loading the form schema
- ✅ Rendering all field types
- ✅ Handling the waiver section
- ✅ Saving signed waivers
- ✅ Displaying confirmation
- ✅ Error handling

Just adapt this to your existing `form_app.py` structure!

# Waiver Signature Feature Guide

## Overview
Your registration system now includes a **full waiver management system**:
- ✅ Registration form includes waiver acknowledgment section
- ✅ Waiver PDF is automatically stored and managed
- ✅ Signed waivers are stored locally with timestamps
- ✅ Each participant's waiver signature is recorded

## How It Works

### 1. **Automated Waiver Setup**
When you run the event planner and generate a registration form:
```
📋 Setting up waiver system...
📁 Waivers will be stored in: /path/to/waivers/
```

The system automatically:
- Creates a `waivers/` directory
- Copies your `Waiver.pdf` to the waivers folder
- Creates subdirectories for each event

### 2. **Registration Form Fields**
The form now includes:
```
- Waiver Acknowledgment (checkbox)
  "I have read and understand the liability waiver"
  
- Waiver Signature (text field)
  "Sign your name here (Parent/Guardian)"
  
- Waiver Date (date field)
  "Date signed"
```

These appear after the standard registration fields.

### 3. **Waiver Storage Structure**
Signed waivers are organized as:
```
waivers/
├── Waiver.pdf                          (Master copy)
├── basketball_clinic/
│   ├── waiver_john_doe_20260329_143022.json
│   ├── waiver_jane_smith_20260329_144515.json
│   └── waiver_mike_johnson_20260329_145830.json
└── youth_camp_2026/
    ├── waiver_alice_williams_20260329_150200.json
    └── waiver_bob_miller_20260329_151045.json
```

### 4. **Waiver Record Contents**
Each signed waiver JSON file contains:
```json
{
  "event": "Youth Basketball Clinic",
  "timestamp": "2026-03-29T14:30:22.123456",
  "youth_name": "John Doe",
  "youth_age": 14,
  "parent_name": "Jane Doe",
  "parent_phone": "(555) 123-4567",
  "waiver_signee": "Jane Doe",
  "waiver_date": "2026-03-29",
  "waiver_acknowledged": true,
  "original_signature_date": "2026-03-29"
}
```

## Integration with Your Streamlit Form

### Option 1: **Automatic Storage** (Recommended)
In your `form_app.py`, after form submission, add:

```python
from app_copy_2 import save_signed_waiver, process_waiver_submission

# After user submits registration form:
if form_submitted:
    # Save standard registration
    save_registration(registration_data)
    
    # Save waiver
    waiver_result = save_signed_waiver(
        participant_info=registration_data,
        event_name=event_details['event_name']
    )
    
    print(f"✅ Waiver saved: {waiver_result['file_path']}")
```

### Option 2: **Display Waiver PDF in Form**
To embed the waiver PDF in your Streamlit form:

```python
import streamlit as st
from pathlib import Path

# Display waiver PDF
waiver_path = Path("waivers/Waiver.pdf")
if waiver_path.exists():
    with open(waiver_path, "rb") as pdf_file:
        st.download_button(
            label="📥 Download Waiver PDF",
            data=pdf_file.read(),
            file_name="Waiver.pdf",
            mime="application/pdf"
        )
    
    # Show PDF preview (requires pdfplumber or similar)
    st.write("**Please read and acknowledge the following:**")
    # [PDF preview code here]
```

## Accessing Stored Waivers

### View All Waivers for an Event
```bash
ls waivers/basketball_clinic/
# Output:
# waiver_john_doe_20260329_143022.json
# waiver_jane_smith_20260329_144515.json
```

### Read a Specific Waiver
```python
import json

waiver_file = "waivers/basketball_clinic/waiver_john_doe_20260329_143022.json"
with open(waiver_file, 'r') as f:
    waiver_data = json.load(f)
    print(f"Signed by: {waiver_data['waiver_signee']}")
    print(f"Date: {waiver_data['waiver_date']}")
```

### Generate a Report of All Waivers
```python
import json
from pathlib import Path

waivers_dir = Path("waivers/basketball_clinic/")
waivers = []

for waiver_file in waivers_dir.glob("waiver_*.json"):
    with open(waiver_file) as f:
        waivers.append(json.load(f))

# Generate report
for waiver in waivers:
    print(f"{waiver['youth_name']} - Signed by {waiver['waiver_signee']}")
```

## Key Features

✅ **Automated**: Waivers are saved automatically when forms are submitted
✅ **Timestamped**: Every waiver has date/time stamp for records
✅ **Organized**: Waivers grouped by event in separate folders
✅ **Searchable**: JSON format makes waivers easy to search/filter
✅ **Audit Trail**: Complete participant info linked to waiver
✅ **Easy Export**: JSON files can be easily exported to Excel, databases, etc.

## File Locations

- **Master Waiver PDF**: `waivers/Waiver.pdf`
- **Event Waivers**: `waivers/{event_name_slug}/`
- **Individual Records**: `waiver_{participant_slug}_{timestamp}.json`

## Next Steps

1. **Update form_app.py** - Add waiver handling to form submission
2. **Configure Waiver PDF** - Ensure your Waiver.pdf is in the right location
3. **Test Registration** - Submit a test registration to verify waiver storage
4. **Backup Waivers** - Consider backing up the `waivers/` directory regularly
5. **Export Reports** - Generate waiver reports as needed for compliance

## Troubleshooting

### "Waiver PDF not found" warning
**Fix**: Copy your `Waiver.pdf` to one of these locations:
- `~/OneDrive/Desktop/Youth Ministry/Waiver.pdf`
- `~/Desktop/Waiver.pdf`
- Same folder as `app_copy_2.py`

### No waivers being saved
**Check**: 
- Verify `waiver_acknowledgment` checkbox is checked in form
- Ensure `waiver_signature` field is filled
- Check that `save_signed_waiver()` is being called after form submission

### Waivers saved to wrong location
**Fix**: Check the output messages during form generation - they show the exact waiver directory path

## Security Notes

- Waivers are stored locally on your server
- Consider encrypting the waivers directory if using cloud sync
- Regular backups are recommended
- JSON format allows easy privacy compliance reporting

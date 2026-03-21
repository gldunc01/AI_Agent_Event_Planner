# Streamlit UI for Event Planner - Quick Start

## ✅ What Was Built

I've created a **Streamlit UI wrapper** around your LangGraph event planning app with:

### Files Created:

1. **`run_task.py`** (102 lines)
   - Core wrapper function: `run_task(task_type: str, payload: dict) -> dict`
   - Initializes agents and graph once, then reuses them
   - Automatically detects newly created files
   - Returns: `{"final_text": str, "files": [list of absolute paths]}`

2. **`ui_app.py`** (193 lines)
   - Streamlit interface with sidebar configuration
   - Event details input form
   - Task type selector (flyer, form, clinic, email)
   - "Run Task" button
   - Text area showing final output
   - Download buttons for each generated file
   - Session state caching for re-downloads

3. **`STREAMLIT_README.md`**
   - Complete setup and usage guide
   - Troubleshooting tips
   - Architecture diagrams
   - Advanced usage examples

4. **Updated `requirements.txt`**
   - Added `streamlit>=1.28.0`

## 🚀 Quick Start

```bash
# From CodeYouAICapstone/python-langchain/
streamlit run ui_app.py
```

Browser opens → http://localhost:8501

## 📋 What the UI Does

| Feature | Details |
|---------|---------|
| **Task Types** | flyer, youth_registration_form, basketball_clinic, proposal_email |
| **Event Fields** | name, date, time, location, form_url, description, email |
| **Pipeline** | Researcher → Writer → Editor (3-stage async pipeline) |
| **File Outputs** | .png (flyer/QR), .txt (emails), .json (forms) |
| **Downloads** | One-click save to computer |

## 🔑 Key Features

✅ **No Code Duplication** — Imports & reuses all logic from `app_copy_2.py`
✅ **Async-Safe** — Wraps async functions with `asyncio.run()` for Streamlit
✅ **Auto File Detection** — Tracks files before/after execution
✅ **Session Caching** — Results persist during Streamlit session
✅ **Professional UI** — Sidebar form, text areas, progress spinners, download buttons

## 📦 Architecture

```
Streamlit App (ui_app.py)
    ↓
run_task(task_type, payload)  [run_task.py]
    ↓
LangGraph Pipeline  [app_copy_2.py]
    ├── Researcher Agent (research)
    ├── Writer Agent (generate content)
    └── Editor Agent (review & finalize)
    ↓
Generated Files
    ├── flayer.png (QR + event flyer)
    ├── event_qr.png (standalone QR)
    └── proposal_email_*.txt (timestamped)
```

## ✨ How It Works

1. **User fills form** (event details) → **Clicks "Run Task"**
2. **run_task.py initializes agents** (first call only, then cached)
3. **LangGraph runs 3-stage pipeline**:
   - **Researcher**: Gathers information
   - **Writer**: Creates content (form JSON, flyer design, or email draft)
   - **Editor**: Reviews and finalizes (may revise if needed)
4. **Files automatically created** during execution (QR codes, PNGs, emails)
5. **UI detects new files** and shows download buttons
6. **User downloads** directly from browser

## 🎯 Event Details Captured

The payload sent to LangGraph includes:
```python
{
    "task_type": "flyer",
    "event_details": {
        "event_name": "Youth Basketball Clinic",
        "event_date": "2026-04-15",
        "event_time": "3:00 PM - 5:00 PM",
        "location": "City Sports Complex",
        "form_url": "https://forms.example.com/register",
        "description": "Join us for...",
        "contact_email": "pastor@church.org"
    }
}
```

## 🔧 Environment Requirements

Ensure `.env` has:
```
GITHUB_TOKEN=<your-github-token>
```

Optional:
```
TAVILY_API_KEY=<optional-for-research-agent>
```

And `requirements.txt` has all dependencies installed.

## 👉 Next Steps

1. Navigate to `CodeYouAICapstone/python-langchain/`
2. Run: `streamlit run ui_app.py`
3. Fill in event details
4. Select a task type
5. Click "Run Task"
6. Wait ~30 seconds for pipeline completion
7. Download generated files

---

**All files are in:** `c:\Users\gldun\Downloads\repos\CodeYouAICapstone\python-langchain\`

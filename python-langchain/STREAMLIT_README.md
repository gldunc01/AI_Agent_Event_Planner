# Streamlit UI for Event Planner LangGraph App

This is a local Streamlit wrapper around the LangGraph event planning application.

## Files

- **`run_task.py`** — Core wrapper module that exposes the LangGraph logic as a synchronous `run_task(task_type, payload)` function
- **`ui_app.py`** — Streamlit UI application
- **`app_copy_2.py`** — Original LangGraph application (required dependency)

## Setup

### 1. Install Dependencies

Make sure all packages in `requirements.txt` are installed, including Streamlit:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Ensure you have a `.env` file with:

```
GITHUB_TOKEN=<your token>
TAVILY_API_KEY=<optional - for research agent>
```

### 3. Ensure Resources are in Place

The app expects these optional files in the same directory:
- `researcher.json` (with a "template" key for the researcher system prompt)
- `writer.json` (with a "template" key for the writer system prompt)
- `editor.json` (with a "template" key for the editor system prompt)

If these files don't exist, the app uses fallback prompts.

## Running the Streamlit App

From the `CodeYouAICapstone/python-langchain/` directory, run:

```bash
streamlit run ui_app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the UI

### 1. Select Task Type

Choose from the dropdown menu (sidebar):
- **flyer** — Generate event flyer with colors, QR code
- **youth_registration_form** — Generate registration form JSON schema and QR code
- **basketball_clinic** — Generate specialized flyer for basketball events
- **proposal_email** — Generate a proposal email to church leadership

### 2. Enter Event Details

In the sidebar, fill in:
- **Event Name** — e.g., "Youth Basketball Clinic"
- **Event Date** — Format: YYYY-MM-DD
- **Event Time** — e.g., "3:00 PM - 5:00 PM"
- **Location** — e.g., "City Sports Complex"
- **Registration Form URL** — Where people register (auto-linked in QR)
- **Additional Details** (optional) — Description and contact email

### 3. Click "Run Task"

The app will:
1. Initialize the LangGraph agents (researcher, writer, editor)
2. Run the pipeline: Researcher → Writer → Editor
3. Display the final output text in a text area
4. List all generated files (flyers, QR codes, emails)

### 4. Download Files

Each generated file gets a download button. Click to save directly to your computer.

## How It Works

### Architecture

```
┌─────────────────────────┐
│   Streamlit UI (ui_app.py)
│  - Event details form
│  - Task selector
│  - Download buttons
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  run_task() wrapper
│  (run_task.py)
│  - Async handler
│  - File tracking
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  LangGraph Pipeline
│  (app_copy_2.py)
│  1. Researcher Agent
│  2. Writer Agent
│  3. Editor Agent
└──────────┬──────────────┘
           │
           ↓
┌─────────────────────────┐
│  Generated Outputs
│  - PNG files (flyers)
│  - PNG files (QR codes)
│  - TXT files (emails)
│  - JSON schemas (forms)
└─────────────────────────┘
```

### Key Features

✅ **No Duplication** — Reuses `app_copy_2.py` directly; no code copying
✅ **Async Support** — Handles async/await properly with `asyncio.run()`
✅ **File Tracking** — Automatically detects newly created files
✅ **Download Integration** — Users can download file directly from browser
✅ **Session State** — Streamlit caches results so you can re-download files

## Troubleshooting

### "ModuleNotFoundError: No module named 'app_copy_2'"
Make sure you're running from the `CodeYouAICapstone/python-langchain/` directory.

### "Error running task: GITHUB_TOKEN not set"
Add `GITHUB_TOKEN` to your `.env` file (required for the LLM).

### Files not appearing in the download list
Check that the working directory is correct. Files are saved to the current working directory where you ran the Streamlit app.

### No output from the editor
This might mean the pipeline took too long or encountered an error in the agent. Check Streamlit console output for error messages.

## Example Payload (Advanced)

If you want to programmatically test `run_task()`:

```python
import asyncio
from run_task import run_task

async def test():
    payload = {
        "event_details": {
            "event_name": "Youth Basketball Clinic",
            "event_date": "2026-04-15",
            "event_time": "3:00 PM - 5:00 PM",
            "location": "City Sports Complex",
            "form_url": "https://forms.example.com/register"
        }
    }
    result = await run_task("flyer", payload)
    print(f"Final text: {result['final_text']}")
    print(f"Files: {result['files']}")

asyncio.run(test())
```

## Notes

- The app downloads files to your computer when you click the download button
- QR codes automatically link to the form URL you provide
- Flyer colors are determined by the AI/writer agent
- Emails are timestamped and saved locally
- The Streamlit session resets when the browser is refreshed

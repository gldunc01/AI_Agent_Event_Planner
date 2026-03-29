# Event Planner Application - Deployment Ready

## 🎯 Quick Start Workflow

### 1. Deploy Form App to Streamlit Cloud
- Follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Get your public URL (e.g., `https://ai-agent-event-planner.streamlit.app`)

### 2. Run Event Planner Locally
Use `app_copy_2.py` with your hosted URL:

```json
{
  "event_name": "My Youth Event",
  "event_date": "May 20, 2026",
  "event_time": "6:00 PM - 8:00 PM",
  "location": "Community Center",
  "description": "Join us for an amazing event!",
  "form_url": "https://fallback.url",
  "hosted_app_url": "https://ai-agent-event-planner.streamlit.app"
}
```

### 3. Generated Outputs
- 📧 **Proposal Email** → `proposal_email_*.txt`
- 📋 **Form Schema** → `current_event_form.json` (synced to form_app.py)
- 📱 **QR Code** → `event_qr.png` (links to hosted form_app.py!)
- 🎨 **Flyer** → `flyer.png` (with embedded QR code)

### 4. Share & Collect Registrations
- Share the flyer with your event details
- Users scan QR → Sent to YOUR deployed app
- Registrations saved to Streamlit Cloud's database automatically

---

## 📦 Files Included

- `form_app.py` - Streamlit registration form (deploy this)
- `app_copy_2.py` - Event planning backend (run locally)
- `requirements.txt` - All dependencies ✓
- `.streamlit/config.toml` - Streamlit configuration ✓
- `.env.example` - Environment variable template
- `DEPLOYMENT_GUIDE.md` - Step-by-step deployment instructions

---

## 🚀 Deployment Features

✅ **Automatic Form Syncing** - Changes in form schema update deployed app
✅ **QR Code to Hosted URL** - Direct link to your deployed registration form
✅ **Flyer with Embedded QR** - Professional marketing materials
✅ **Automatic Email Archive** - Proposal emails saved locally
✅ **Database Integration** - Registrations stored safely
✅ **Fully Automated Pipeline** - One command generates everything

---

## 🔧 Setup Checklist

- [ ] All files committed to GitHub
- [ ] Streamlit Community Cloud account created
- [ ] GitHub authorization completed
- [ ] First deployment successful
- [ ] Hosted URL obtained
- [ ] Local testing with hosted URL working

---

## 📞 Support

- Deployment help: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Streamlit docs: https://docs.streamlit.io/
- GitHub repo: https://github.com/gldunc01/AI_Agent_Event_Planner

Enjoy your automated event planning! 🎉

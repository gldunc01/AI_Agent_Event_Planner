# 🚀 Deployment Guide - Streamlit Community Cloud

## Overview
This guide will help you deploy the Youth Event Registration Form (`form_app.py`) to Streamlit Community Cloud, making it accessible via a public URL.

Once deployed, you'll get a URL like: `https://ai-agent-event-planner.streamlit.app`

---

## Step 1: Prepare Your GitHub Repository

Your repository is already set up at: `https://github.com/gldunc01/AI_Agent_Event_Planner`

Make sure all files are committed:
```bash
cd python-langchain
git add .
git commit -m "Add deployment configuration for Streamlit"
git push
```

---

## Step 2: Connect to Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
2. Click **"Sign up"** or **"Sign in"** with your GitHub account
3. Authorize Streamlit to access your GitHub repositories

---

## Step 3: Deploy the App

1. In Streamlit Cloud dashboard, click **"New app"**
2. Fill in the details:
   - **Repository**: `gldunc01/AI_Agent_Event_Planner`
   - **Branch**: `main` (or your main branch)
   - **Main file path**: `python-langchain/form_app.py`
3. Click **"Deploy"**

Streamlit will start building and deploying your app. This takes ~2-5 minutes.

Once complete, you'll get a URL: `https://YOUR_APP_NAME.streamlit.app`

---

## Step 4: Add Secrets (Environment Variables)

Your `form_app.py` doesn't need secrets (it only reads form data), but if you deploy `app_copy_2.py` separately, you'd need:

1. In Streamlit Cloud dashboard, go to your app settings ⚙️
2. Click **"Secrets"**
3. Add:
```
GITHUB_TOKEN = "your_github_token"
TAVILY_API_KEY = "your_tavily_key"  # Optional
```

---

## Step 5: Use the Hosted URL in Your Workflow

Once deployed, you'll have a public URL. Use it when running `app_copy_2.py`:

### Example with hosted URL:
```json
{
  "event_name": "Youth Basketball Clinic",
  "event_date": "April 15, 2026",
  "event_time": "2:00 PM - 4:00 PM",
  "location": "Community Center",
  "description": "Fun basketball skills training",
  "form_url": "https://fallback-form.example.com",
  "hosted_app_url": "https://ai-agent-event-planner.streamlit.app"
}
```

The QR code will now link directly to your deployed Streamlit app! 🎯

---

## Complete Workflow

1. **Run `app_copy_2.py`** with your event details (including `hosted_app_url`)
2. **Generate**: 
   - 📧 Proposal email
   - 📋 Registration form schema
   - 📱 QR code (linking to YOUR hosted app!)
   - 🎨 Flyer with embedded QR
3. **Share** the flyer and email with stakeholders
4. **Users scan QR** → Taken directly to your deployed `form_app.py` → Complete registration

---

## Useful URLs

- Your GitHub repo: https://github.com/gldunc01/AI_Agent_Event_Planner
- Streamlit Community Cloud: https://streamlit.io/cloud
- Deployed app (after deployment): `https://YOUR_APP_NAME.streamlit.app`

---

## Troubleshooting

**App won't deploy?**
- Check that `requirements.txt` has all dependencies ✓
- Verify `.streamlit/config.toml` exists ✓
- Make sure `python-langchain/form_app.py` is the exact file path

**Form not loading?**
- Make sure `current_event_form.json` is generated locally first
- Run `app_copy_2.py` to create the form file
- Check that files are in the same directory

**Secrets not working?**
- Restart the app after adding secrets
- Use correct format in Streamlit Cloud secrets UI

---

## File Structure

```
python-langchain/
├── form_app.py                    # Deployed Streamlit app
├── app_copy_2.py                  # Event planning backend
├── current_event_form.json        # Auto-generated form (local)
├── requirements.txt               # ✓ Dependencies
├── .streamlit/
│   └── config.toml               # ✓ Streamlit config
└── .env.example                  # ✓ Environment template
```

---

## Next Steps

1. ✅ Files are ready for deployment
2. ➡️ Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. ➡️ Deploy `form_app.py`
4. ➡️ Get your public URL
5. ➡️ Use it in `app_copy_2.py` as `hosted_app_url`
6. ➡️ Generate flyers with QR codes linking to your app!

Questions? Check the Streamlit docs: https://docs.streamlit.io/deploy/streamlit-community-cloud

Happy deploying! 🚀

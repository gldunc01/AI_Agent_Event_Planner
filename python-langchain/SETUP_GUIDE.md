# GitHub Waiver Storage Setup Guide

## Overview

Your youth registration app now supports **persistent GitHub-based waiver storage** for use on Streamlit Community Cloud. Waivers are automatically committed to your GitHub repository, ensuring they survive across app restarts and deployments.

## Quick Start (3 Steps)

### Step 1: Create GitHub Personal Access Token

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Fill in:
   - **Note**: `Streamlit Waiver Storage`
   - **Expiration**: 90 days (recommended)
   - **Scopes**: Check ✓ `repo` (full control of private repositories)
4. Click **"Generate token"**
5. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)
   - Save it somewhere safe temporarily

### Step 2: Configure Streamlit Secrets

#### For Streamlit Cloud:
1. Go to: https://share.streamlit.io/
2. Click on your app in the sidebar
3. Click the **⚙️ Settings** button (top-right)
4. Click **"Secrets"** tab
5. Paste this configuration:
   ```toml
   GITHUB_TOKEN = "ghp_your_token_here"
   GITHUB_REPO = "your-username/CodeYouAICapstone"
   ```
6. Replace:
   - `ghp_your_token_here` with your token from Step 1
   - `your-username` with your GitHub username
7. Click **"Save"** ✓

#### For Local Development:
1. Navigate to your project folder: `c:\Users\gldun\Downloads\repos\CodeYouAICapstone\python-langchain\`
2. Create folder `.streamlit` (if it doesn't exist)
3. Create file `.streamlit/secrets.toml` with:
   ```toml
   GITHUB_TOKEN = "ghp_your_token_here"
   GITHUB_REPO = "your-username/CodeYouAICapstone"
   ```
4. Save the file
5. **Important**: Add `.streamlit/secrets.toml` to `.gitignore` so you don't commit your token to GitHub

### Step 3: Install PyGithub

The app needs the `PyGithub` library. It's already added to `requirements.txt`.

For Streamlit Cloud:
- Automatically installed when you deploy

For local development:
```bash
pip install PyGithub
```

## Verification

After setup, test it by:

1. **Local**: Run `streamlit run form_app.py`
2. **Streamlit Cloud**: Refresh your app
3. **Test Registration**:
   - Fill out the form
   - Sign the waiver
   - Click "Submit Registration"
   - Should see: ✓ Success message with GitHub URL

4. **Verify in GitHub**:
   - Go to your repository
   - Navigate to `waivers/` folder
   - You should see files like: `waiver_participant_20260329_104634.json`

## What Gets Stored

When a youth signs a waiver, this information is committed to GitHub:

```json
{
  "event": "Event Name",
  "timestamp": "2026-03-29T10:46:34.123456",
  "youth_name": "Child Name",
  "youth_age": 14,
  "parent_name": "Parent/Guardian Name",
  "parent_phone": "+1-555-1234",
  "waiver_signee": "Signature Name",
  "waiver_date": "03-29-2026",
  "waiver_acknowledged": true
}
```

**File location in GitHub**: `waivers/{event_name}/waiver_participant_{timestamp}.json`

## Fallback Behavior

If GitHub integration fails (missing token, repo issues, etc.):
- Waivers fall back to **local storage** automatically
- ✓ Works on local development
- ⚠️ On Streamlit Cloud: files disappear when app restarts

## Troubleshooting

### Token Issues
- **Error**: `401 Unauthorized`
  - Solution: Verify token is correct in Streamlit Secrets
  - Check token hasn't expired (create new one if needed)

- **Error**: `404 Repository Not Found`
  - Solution: Verify `GITHUB_REPO` format is `owner/repo`
  - Make sure repository exists at `https://github.com/owner/repo`

### File Not Appearing in GitHub
- Waivers go to: `waivers/{event_name}/` folder
- Check if folder exists in repository
- If missing, check app logs for errors

### Streamlit Cloud Deployment
- After updating secrets, **restart** the app
  - Go to app page → Click menu (⋮) → "Reboot app"
  - Or: make a small code change and push to trigger redeploy

## Security Notes

⚠️ **KEEP YOUR TOKEN PRIVATE**:
- Never commit `.streamlit/secrets.toml` to GitHub
- Never share your token
- Tokens expire (use 90-day rotation recommended)
- If compromised, delete token immediately at https://github.com/settings/tokens

✓ **Good practices**:
- Store tokens in Streamlit Secrets, not in code
- Use `.gitignore` to exclude secrets files
- Rotate tokens every 90 days
- Use scoped tokens (only "repo" scope needed)

## Files Modified

- ✅ `form_app.py` - Updated with configuration documentation
- ✅ `app_copy_2.py` - GitHub storage implementation
- ✅ `requirements.txt` - Added PyGithub dependency
- ✅ `.streamlit/secrets.toml.example` - Configuration template
- ✅ `SETUP_GUIDE.md` - This file

## Questions?

If waivers aren't appearing:
1. Check Streamlit app logs for errors
2. Verify GitHub token is valid
3. Verify repository name format
4. Try local development with `.streamlit/secrets.toml` first
5. Check GitHub repository `waivers/` folder exists

## Next Steps

1. ✓ Create GitHub token
2. ✓ Configure Streamlit Secrets
3. ✓ Test with a test registration
4. ✓ Verify waiver appears in GitHub repository
5. ✓ Deploy to production!

---

**Status**: ✅ GitHub storage implementation is ready to use!

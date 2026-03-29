import os
import asyncio
import json
from typing import TypedDict, Annotated, Literal, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.types import Command
import qrcode  # pip install qrcode[pil]
from PIL import Image, ImageDraw, ImageFont  # For flyer PNG
import textwrap
from json_repair import repair_json
import re
import hashlib
import shutil
from pathlib import Path
import base64

load_dotenv()


# State definition for LangGraph: holds the message history for the agent pipeline.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    event_details: dict  # Event info passed through pipeline
    form_url: str  # URL where the form is hosted (for QR linking)
    qr_path: str  # Path to generated QR code

# Global agents
researcher_agent = None
writer_agent = None
editor_agent = None
llm = None  # Global LLM client

def truncate_messages(messages: list, max_messages: int = 5) -> list:
    if len(messages) <= max_messages:
        return messages
    return messages[:1] + messages[-(max_messages-1):]

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def print_banner(text: str, symbol: str = "="):
    """Fancy print headers."""
    line = symbol * 60
    print(f"\n{line}")
    print(f"{symbol*10} {text.center(40)} {symbol*10}")
    print(line)

def format_event_date(date_str: str) -> str:
    """
    Convert event date to MM-DD-YYYY format.
    
    Handles multiple input formats:
    - 2026-04-15 (ISO format)
    - April 15, 2026 (text format)
    - 04-15-2026 (already correct)
    - 4/15/2026 (slash format)
    
    Args:
        date_str: Date string in various formats
    
    Returns:
        Date string in MM-DD-YYYY format
    """
    if not date_str or date_str == "TBD":
        return "TBD"
    
    date_str = date_str.strip()
    
    try:
        # Try parsing ISO format (2026-04-15)
        if '-' in date_str and len(date_str) == 10:
            parts = date_str.split('-')
            if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 4:
                # YYYY-MM-DD format
                year, month, day = parts
                return f"{month.zfill(2)}-{day.zfill(2)}-{year}"
            elif len(parts) == 3 and parts[2].isdigit() and len(parts[2]) == 4:
                # MM-DD-YYYY or DD-MM-YYYY format - check if already correct
                month, day, year = parts
                if int(month) <= 12:  # Assume MM-DD-YYYY
                    return f"{month.zfill(2)}-{day.zfill(2)}-{year}"
                else:  # DD-MM-YYYY
                    return f"{day.zfill(2)}-{month.zfill(2)}-{year}"
        
        # Try parsing slash format (4/15/2026 or 04/15/2026)
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                if len(parts[2]) == 4:  # MM/DD/YYYY format
                    month, day, year = parts
                else:  # YYYY/MM/DD format
                    year, month, day = parts
                return f"{month.zfill(2)}-{day.zfill(2)}-{year}"
        
        # Try parsing text format (April 15, 2026)
        if ',' in date_str:
            from datetime import datetime as dt
            parsed = dt.strptime(date_str, "%B %d, %Y")
            return parsed.strftime("%m-%d-%Y")
        
    except Exception as e:
        print(f"⚠️ Could not parse date '{date_str}': {e}")
    
    return date_str


def generate_standardized_form(event_details: Dict) -> dict:
    """
    Generate the standardized youth registration form with event details.
    
    Form fields are fixed and consistent across all events.
    Only event details (name, date, time, location) change.
    
    Args:
        event_details: Dictionary with event info (event_name, event_date, event_time, location, etc.)
    
    Returns:
        Dictionary with complete form schema including fields and event details
    """
    event_name = event_details.get("event_name", "Youth Event")
    
    # Format the date to MM-DD-YYYY
    formatted_date = format_event_date(event_details.get("event_date", "TBD"))
    
    return {
        "title": f"{event_name} Registration Form",
        "description": f"Register for {event_name}. Please provide all required information.",
        "event_details": {
            "event_name": event_name,
            "date": formatted_date,
            "time": event_details.get("event_time", "TBD"),
            "location": event_details.get("location", "TBD"),
            "max_participants": event_details.get("max_participants", 30),
            "age_range": event_details.get("age_range", "12-17")
        },
        "fields": [
            {"name": "youth_first_last_name", "label": "Youth First and Last Name", "type": "text", "required": True},
            {"name": "youth_age", "label": "Youth Age", "type": "number", "required": True},
            {"name": "parent_first_last_name", "label": "Parent/Guardian First and Last Name", "type": "text", "required": True},
            {"name": "parent_phone", "label": "Parent/Guardian Phone Number", "type": "tel", "required": True},
            {"name": "transportation_needed", "label": "Need transportation?", "type": "select", "required": True, "options": [{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}]},
            {"name": "special_needs", "label": "Special accommodations? (If yes, specify)", "type": "textarea", "required": False},
            {"name": "consent", "label": "I give permission for my child to participate.", "type": "checkbox", "required": True},
            {"name": "signature", "label": "Parent/Guardian Signature", "type": "text", "required": True},
            {"name": "date", "label": "Date", "type": "date", "required": True},
            {"section": "WAIVER & LIABILITY RELEASE", "type": "section_header"},
            {"name": "waiver_acknowledgment", "label": "I have read and understand the liability waiver", "type": "checkbox", "required": True},
            {"name": "waiver_signature", "label": "Sign your name here (Parent/Guardian)", "type": "text", "required": True},
            {"name": "waiver_date", "label": "Date", "type": "date", "required": True}
        ]
    }


def setup_waiver_directory() -> str:
    """Create directory structure for storing waivers."""
    waivers_dir = os.path.join(os.path.dirname(__file__), "waivers")
    if not os.path.exists(waivers_dir):
        os.makedirs(waivers_dir)
    return waivers_dir


def copy_waiver_pdf(waiver_source: str = None) -> str:
    """
    Copy the waiver PDF to the waivers directory if it exists.
    
    Args:
        waiver_source: Path to the waiver PDF file. If None, looks for Waiver.pdf in common locations.
    
    Returns:
        Path to the waiver PDF in the waivers directory
    """
    waivers_dir = setup_waiver_directory()
    waiver_dest = os.path.join(waivers_dir, "Waiver.pdf")
    
    # If already copied, return path
    if os.path.exists(waiver_dest):
        return waiver_dest
    
    # If source provided, copy it
    if waiver_source and os.path.exists(waiver_source):
        shutil.copy2(waiver_source, waiver_dest)
        print(f"✅ Waiver PDF stored at: {waiver_dest}")
        return waiver_dest
    
    # Try common locations
    common_paths = [
        os.path.expanduser("~/OneDrive/Desktop/Youth Ministry/Waiver.pdf"),
        os.path.expanduser("~/Desktop/Waiver.pdf"),
        os.path.join(os.path.dirname(__file__), "Waiver.pdf"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            shutil.copy2(path, waiver_dest)
            print(f"✅ Waiver PDF stored at: {waiver_dest}")
            return waiver_dest
    
    print("⚠️ Waiver PDF not found - skipping waiver storage setup")
    return waiver_dest


def save_signed_waiver(participant_info: Dict, event_name: str) -> dict:
    """
    Save a record of the signed waiver with participant information.
    
    Args:
        participant_info: Dictionary with participant details including waiver_signature
        event_name: Name of the event
    
    Returns:
        Dictionary with waiver file path and details
    """
    waivers_dir = setup_waiver_directory()
    
    # Create event-specific subdirectory
    event_slug = re.sub(r'[^a-z0-9]+', '_', event_name.lower()).strip('_')
    event_waivers_dir = os.path.join(waivers_dir, event_slug)
    if not os.path.exists(event_waivers_dir):
        os.makedirs(event_waivers_dir)
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    youth_name = participant_info.get('youth_first_last_name', 'Unknown')
    youth_slug = re.sub(r'[^a-z0-9]+', '_', youth_name.lower()).strip('_')
    
    filename = f"waiver_{youth_slug}_{timestamp}.json"
    file_path = os.path.join(event_waivers_dir, filename)
    
    # Create waiver record
    waiver_record = {
        "event": event_name,
        "timestamp": datetime.now().isoformat(),
        "youth_name": participant_info.get('youth_first_last_name', 'N/A'),
        "youth_age": participant_info.get('youth_age', 'N/A'),
        "parent_name": participant_info.get('parent_first_last_name', 'N/A'),
        "parent_phone": participant_info.get('parent_phone', 'N/A'),
        "waiver_signee": participant_info.get('waiver_signature', 'N/A'),
        "waiver_date": participant_info.get('waiver_date', 'N/A'),
        "waiver_acknowledged": participant_info.get('waiver_acknowledgment', False),
        "original_signature_date": participant_info.get('date', 'N/A')
    }
    
    # Save to JSON
    with open(file_path, 'w') as f:
        json.dump(waiver_record, f, indent=2)
    
    print_banner("📋 WAIVER SIGNED & STORED", "✍️")
    print(f"✅ Waiver record saved: {file_path}")
    print(f"👤 Participant: {waiver_record['youth_name']}")
    print(f"📝 Signed by: {waiver_record['waiver_signee']}")
    
    return {
        "file_path": file_path,
        "event": event_name,
        "participant": youth_name,
        "timestamp": waiver_record['timestamp']
    }


# --- TOOL CANDIDATE ---
# This function is a good candidate to expose as a tool:
# - Input: flyer_data (dict with required fields), qr_path (path to QR PNG)
# - Output: flyer PNG file saved to disk
# - Used for: 'flyer' and 'basketball_clinic' tasks
def save_flyer_png_modern_clean(flyer_data: Dict, qr_path: str, output_path: str):
    """Modern clean design with centered header and card-style sections."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])
    
    # Create image with white background
    img = Image.new('RGB', (1100, 850), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 56)
        font_large = ImageFont.truetype("arial.ttf", 40)
        font_med = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Colored header bar
    draw.rectangle([(0, 0), (img.width, 140)], fill=primary_color)
    
    # Headline centered in header - wrapped for long text
    headline_wrapped = textwrap.fill(flyer_data['headline'], width=25)
    draw.text((60, 35), headline_wrapped, fill='white', font=font_huge)
    
    # Accent line separator
    draw.rectangle([(50, 155), (img.width - 50, 165)], fill=accent_color)
    
    # Subheadline
    subheadline_wrapped = textwrap.fill(flyer_data['subheadline'], width=40)
    draw.text((100, 185), subheadline_wrapped, fill=accent_color, font=font_med)
    
    # Details section with background cards
    y = 280
    details = [
        ("📅 When", flyer_data['date_time_line'][:50]),
        ("📍 Where", flyer_data['location_line'][:45]),
        ("ℹ️ About", flyer_data['body_blurb'][:80])
    ]
    
    for label, content in details:
        # Card background
        draw.rectangle([(60, y), (img.width - 60, y + 100)], outline=accent_color, width=2)
        draw.rectangle([(60, y), (200, y + 50)], fill=accent_color)
        
        # Label in colored box
        draw.text((70, y + 10), label, fill='white', font=font_med)
        
        # Content wrapped to fit
        wrapped = textwrap.fill(content, width=45)
        draw.text((70, y + 55), wrapped, fill=primary_color, font=font_small)
        
        y += 130
    
    # Call to action button
    cta_y = img.height - 150
    draw.rectangle([(100, cta_y), (img.width - 100, cta_y + 60)], fill=accent_color)
    cta_bbox = draw.textbbox((0, 0), flyer_data['call_to_action'], font=font_large)
    cta_width = cta_bbox[2] - cta_bbox[0]
    cta_x = (img.width - cta_width) // 2
    draw.text((cta_x, cta_y + 8), flyer_data['call_to_action'], fill='white', font=font_large)
    
    # QR code (bottom right corner)
    qr_img = Image.open(qr_path).resize((180, 180))
    img.paste(qr_img, (img.width - 220, img.height - 220))
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Modern Clean) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png_bold_vibrant(flyer_data: Dict, qr_path: str, output_path: str):
    """Bold vibrant design with diagonal accents and large typography."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])
    
    # Create image
    img = Image.new('RGB', (1100, 850), color=primary_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 72)
        font_large = ImageFont.truetype("arial.ttf", 44)
        font_med = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 22)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Large diagonal accent bar (top right)
    points = [(img.width, 0), (img.width, 200), (img.width - 300, 0)]
    draw.polygon(points, fill=accent_color)
    
    # Headline in white
    draw.text((60, 50), flyer_data['headline'], fill='white', font=font_huge)
    
    # Subheadline
    draw.text((60, 150), flyer_data['subheadline'], fill=accent_color, font=font_large)
    
    # Content section (white box with event details)
    content_y = 280
    draw.rectangle([(40, content_y), (img.width - 40, content_y + 380)], fill='white')
    
    # Draw event details inside white box
    draw.text((70, content_y + 20), "📅 " + flyer_data['date_time_line'], fill=primary_color, font=font_med)
    draw.text((70, content_y + 80), "📍 " + flyer_data['location_line'], fill=primary_color, font=font_med)
    
    # Body blurb wrapped - truncate if too long
    body_text = flyer_data['body_blurb'][:120]
    wrapped_body = textwrap.fill(body_text, width=60)
    draw.text((70, content_y + 150), wrapped_body, fill=primary_color, font=font_small)
    
    # CTA in accent color
    draw.text((70, content_y + 300), flyer_data['call_to_action'], fill=accent_color, font=font_large)
    
    # QR code (bottom right in colored box)
    qr_box_x, qr_box_y = img.width - 240, img.height - 240
    draw.rectangle([(qr_box_x - 20, qr_box_y - 20), (qr_box_x + 200, qr_box_y + 200)], fill=accent_color)
    qr_img = Image.open(qr_path).resize((180, 180))
    img.paste(qr_img, (qr_box_x, qr_box_y))
    
    # Footer text
    draw.text((60, img.height - 40), "Scan QR to Register Instantly!", fill='white', font=font_small)
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Bold Vibrant) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png_professional_business(flyer_data: Dict, qr_path: str, output_path: str):
    """Professional business style with two-column layout."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])
    
    img = Image.new('RGB', (1100, 850), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 60)
        font_large = ImageFont.truetype("arial.ttf", 40)
        font_med = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Left column (primary color)
    draw.rectangle([(0, 0), (500, img.height)], fill=primary_color)
    
    # Headline on left
    wrapped_headline = textwrap.fill(flyer_data['headline'], width=20)
    draw.text((30, 40), wrapped_headline, fill='white', font=font_huge)
    
    # Subheadline
    draw.text((30, 250), flyer_data['subheadline'], fill=accent_color, font=font_large)
    
    # Left column details
    draw.text((30, 380), "Date & Time:", fill=accent_color, font=font_med)
    draw.text((30, 430), flyer_data['date_time_line'], fill='white', font=font_small)
    
    draw.text((30, 520), "Location:", fill=accent_color, font=font_med)
    draw.text((30, 570), flyer_data['location_line'], fill='white', font=font_small)
    
    # Right column (white)
    draw.rectangle([(500, 0), (img.width, img.height)], fill='white')
    
    # Right column details
    draw.text((550, 100), "About This Event:", fill=primary_color, font=font_large)
    wrapped_body = textwrap.fill(flyer_data['body_blurb'], width=45)
    draw.text((550, 180), wrapped_body, fill=(80, 80, 80), font=font_small)
    
    # CTA box
    draw.rectangle([(550, 400), (img.width - 30, 480)], fill=accent_color)
    cta_bbox = draw.textbbox((0, 0), flyer_data['call_to_action'], font=font_med)
    cta_width = cta_bbox[2] - cta_bbox[0]
    cta_x = 550 + (img.width - 580 - cta_width) // 2
    draw.text((cta_x, 415), flyer_data['call_to_action'], fill='white', font=font_med)
    
    # QR code on right
    qr_img = Image.open(qr_path).resize((220, 220))
    img.paste(qr_img, (img.width - 280, img.height - 260))
    
    # Small footer
    draw.text((550, img.height - 50), "Scan QR Code Above to Register", fill=(150, 150, 150), font=font_small)
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Professional) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png_retro_playful(flyer_data: Dict, qr_path: str, output_path: str):
    """Retro 90s playful style with bouncy elements and rounded corners."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])
    secondary = tuple(min(255, c + 40) for c in primary_color)  # Lighter shade
    
    img = Image.new('RGB', (1100, 850), color=secondary)
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 70)
        font_large = ImageFont.truetype("arial.ttf", 44)
        font_med = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Playful circles in corners
    draw.ellipse([(20, 20), (200, 200)], outline=accent_color, width=8)
    draw.ellipse([(img.width - 200, img.height - 200), (img.width - 20, img.height - 20)], outline=primary_color, width=8)
    
    # Central content box with thick border
    box_x1, box_y1 = 80, 120
    box_x2, box_y2 = img.width - 80, img.height - 160
    draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], fill=(255, 255, 255), outline=primary_color, width=6)
    
    # Headline with shadow effect
    draw.text((100, 140), flyer_data['headline'], fill=primary_color, font=font_huge)
    draw.text((102, 142), flyer_data['headline'], fill=accent_color, font=font_huge)
    
    # Subheadline
    draw.text((100, 240), flyer_data['subheadline'], fill=accent_color, font=font_large)
    
    # Event details in colored boxes
    y = 330
    details = [
        ("📅", flyer_data['date_time_line']),
        ("📍", flyer_data['location_line']),
        ("ℹ️", flyer_data['body_blurb'][:60] + "...")
    ]
    
    for icon, content in details:
        draw.rectangle([(100, y), (img.width - 100, y + 50)], fill=accent_color, outline=primary_color, width=3)
        draw.text((110, y + 8), icon + " " + content, fill='white', font=font_med)
        y += 70
    
    # CTA with burst effect
    cta_y = img.height - 130
    draw.rectangle([(150, cta_y), (img.width - 150, cta_y + 70)], fill=primary_color)
    for i in range(0, 20):
        angle = i * 18
        draw.text((img.width // 2, cta_y + 20), "⭐", fill=accent_color, font=font_small)
    
    cta_bbox = draw.textbbox((0, 0), flyer_data['call_to_action'], font=font_large)
    cta_width = cta_bbox[2] - cta_bbox[0]
    cta_x = (img.width - cta_width) // 2
    draw.text((cta_x, cta_y + 10), flyer_data['call_to_action'], fill='white', font=font_large)
    
    # QR code with decorative frame
    qr_box_x, qr_box_y = img.width - 260, 40
    draw.rectangle([(qr_box_x - 10, qr_box_y - 10), (qr_box_x + 210, qr_box_y + 210)], fill=accent_color, width=4)
    qr_img = Image.open(qr_path).resize((180, 180))
    img.paste(qr_img, (qr_box_x, qr_box_y))
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Retro Playful) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png_sporty_dynamic(flyer_data: Dict, qr_path: str, output_path: str):
    """High-energy sports style with diagonal stripes and action vibes."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])
    
    img = Image.new('RGB', (1100, 850), color=(20, 20, 20))  # Dark background
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 80)
        font_large = ImageFont.truetype("arial.ttf", 48)
        font_med = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Diagonal stripes background
    stripe_width = 40
    for i in range(0, img.width + img.height, stripe_width * 2):
        draw.line([(i, 0), (i - img.height, img.height)], fill=primary_color, width=stripe_width)
    
    # Energy band across top
    draw.rectangle([(0, 0), (img.width, 100)], fill=accent_color)
    
    # Headline in white with thick letters
    draw.text((50, 20), flyer_data['headline'], fill='white', font=font_huge)
    
    # Main content area with semi-transparent overlay
    overlay = Image.new('RGBA', (img.width, img.height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, 130), (img.width, img.height - 50)], fill=(255, 255, 255, 25))
    img.paste(overlay, (0, 0), overlay)
    
    # Event details in bold
    draw.text((60, 150), "🗓️  " + flyer_data['date_time_line'], fill=accent_color, font=font_large)
    draw.text((60, 240), "📍 " + flyer_data['location_line'], fill=accent_color, font=font_large)
    
    # Body wrapped - truncate if too long
    body_text = flyer_data['body_blurb'][:100]
    wrapped_body = textwrap.fill(body_text, width=50)
    draw.text((60, 350), wrapped_body, fill='white', font=font_med)
    
    # Action button (bold CTA)
    draw.rectangle([(100, 580), (img.width - 100, 680)], fill=accent_color, outline='white', width=5)
    cta_bbox = draw.textbbox((0, 0), flyer_data['call_to_action'], font=font_huge)
    cta_width = cta_bbox[2] - cta_bbox[0]
    cta_x = (img.width - cta_width) // 2
    draw.text((cta_x, 595), flyer_data['call_to_action'], fill='white', font=font_huge)
    
    # QR corner badge
    qr_img = Image.open(qr_path).resize((160, 160))
    qr_bg = Image.new('RGB', (190, 190), accent_color)
    qr_bg.paste(qr_img, (15, 15))
    img.paste(qr_bg, (img.width - 210, img.height - 190))
    
    draw.text((img.width - 200, img.height - 30), "SCAN TO REGISTER", fill='white', font=font_small)
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Sporty Dynamic) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png_minimalist_cool(flyer_data: Dict, qr_path: str, output_path: str):
    """Ultra-clean minimalist design with lots of whitespace and typography focus."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    
    img = Image.new('RGB', (1100, 850), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 90)
        font_large = ImageFont.truetype("arial.ttf", 42)
        font_med = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 22)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Single color accent bar (very thin)
    draw.rectangle([(0, 200), (img.width, 210)], fill=primary_color)
    
    # Minimalist headline (centered, huge)
    headline_lines = flyer_data['headline'].split('\n') if '\n' in flyer_data['headline'] else [flyer_data['headline']]
    y_pos = 80
    for line in headline_lines:
        bbox = draw.textbbox((0, 0), line, font=font_huge)
        line_width = bbox[2] - bbox[0]
        x_pos = (img.width - line_width) // 2
        draw.text((x_pos, y_pos), line, fill=primary_color, font=font_huge)
        y_pos += 100
    
    # Plenty of whitespace
    y_pos = 300
    
    # Details - one per line, left aligned
    draw.text((100, y_pos), flyer_data['date_time_line'], fill=(80, 80, 80), font=font_med)
    y_pos += 80
    draw.text((100, y_pos), flyer_data['location_line'], fill=(80, 80, 80), font=font_med)
    y_pos += 100
    
    # Body description - truncate if too long
    body_text = flyer_data['body_blurb'][:100]
    wrapped_body = textwrap.fill(body_text, width=45)
    draw.text((100, y_pos), wrapped_body, fill=(120, 120, 120), font=font_small)
    
    # CTA at bottom - simple and elegant
    draw.text((100, img.height - 100), flyer_data['call_to_action'], fill=primary_color, font=font_large)
    
    # QR code bottom right - no border, just QR
    qr_img = Image.open(qr_path).resize((200, 200))
    img.paste(qr_img, (img.width - 250, img.height - 250))
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Minimalist Cool) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png_festival_fun(flyer_data: Dict, qr_path: str, output_path: str):
    """Festival/carnival style with colorful sections and playful layout."""
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])
    
    img = Image.new('RGB', (1100, 850), color=(255, 220, 100))  # Warm yellow base
    draw = ImageDraw.Draw(img)
    
    try:
        font_huge = ImageFont.truetype("arial.ttf", 76)
        font_large = ImageFont.truetype("arial.ttf", 46)
        font_med = ImageFont.truetype("arial.ttf", 30)
        font_small = ImageFont.truetype("arial.ttf", 24)
    except:
        font_huge = font_large = font_med = font_small = ImageFont.load_default()
    
    # Colorful header banner with gradient effect (simulated with rectangles)
    draw.rectangle([(0, 0), (img.width, 30)], fill=primary_color)
    draw.rectangle([(0, 30), (img.width, 60)], fill=accent_color)
    draw.rectangle([(0, 60), (img.width, 90)], fill=primary_color)
    
    # Headline with background
    draw.rectangle([(40, 110), (img.width - 40, 190)], fill=(255, 255, 255), outline=primary_color, width=4)
    draw.text((60, 125), flyer_data['headline'], fill=primary_color, font=font_huge)
    
    # Subheadline
    draw.text((60, 210), flyer_data['subheadline'], fill=accent_color, font=font_large)
    
    # Three info boxes with different colors
    boxes = [
        (60, 280, primary_color, "📅", flyer_data['date_time_line']),
        (360, 280, accent_color, "📍", flyer_data['location_line']),
        (660, 280, primary_color, "🎉", "Join us for fun!")
    ]
    
    for box_x, box_y, color, icon, text in boxes:
        draw.rectangle([(box_x, box_y), (box_x + 280, box_y + 150)], fill=color, outline='white', width=3)
        draw.text((box_x + 20, box_y + 20), icon, fill='white', font=font_huge)
        wrapped_text = textwrap.fill(text, width=20)
        draw.text((box_x + 20, box_y + 70), wrapped_text, fill='white', font=font_med)
    
    # Large body area
    draw.rectangle([(60, 470), (img.width - 60, 680)], fill='white', outline=primary_color, width=4)
    body_text = flyer_data['body_blurb'][:120]
    wrapped_body = textwrap.fill(body_text, width=55)
    draw.text((80, 490), wrapped_body, fill=(40, 40, 40), font=font_med)
    
    # CTA button
    draw.rectangle([(150, 720), (img.width - 150, 800)], fill=primary_color, outline=accent_color, width=5)
    cta_bbox = draw.textbbox((0, 0), flyer_data['call_to_action'], font=font_large)
    cta_width = cta_bbox[2] - cta_bbox[0]
    cta_x = (img.width - cta_width) // 2
    draw.text((cta_x, 740), flyer_data['call_to_action'], fill='white', font=font_large)
    
    # QR code with festive frame
    qr_img = Image.open(qr_path).resize((140, 140))
    img.paste(qr_img, (img.width - 180, 20))
    
    img.save(output_path)
    print(f"✅ FLYER SAVED (Festival Fun) to: {os.path.abspath(output_path)}")
    return output_path

def save_flyer_png(flyer_data: Dict, qr_path: str, output_path: str = "flyer.png"):
    """Generate professional flyer with rotating design templates."""
    print_banner("🎨 GENERATING FLYER PNG", "🖼️")

    # If flyer_data is a list, use the first element
    if isinstance(flyer_data, list):
        if len(flyer_data) == 0:
            raise ValueError("flyer_data is an empty list!")
        flyer_data = flyer_data[0]

    # Validate flyer_data is a dict
    if not isinstance(flyer_data, dict):
        raise ValueError(f"Expected flyer_data to be a dict, got {type(flyer_data).__name__}: {str(flyer_data)[:200]}")

    # Validate required fields
    required_fields = ['headline', 'subheadline', 'date_time_line', 'location_line', 'body_blurb', 'call_to_action', 'color_scheme']
    missing_fields = [f for f in required_fields if f not in flyer_data]
    if missing_fields:
        raise ValueError(f"Missing required fields in flyer JSON: {missing_fields}")

    # Validate color_scheme has required colors
    if not all(k in flyer_data['color_scheme'] for k in ['primary', 'accent']):
        raise ValueError("color_scheme must have 'primary' and 'accent' hex colors")

    # Select design based on design_style in flyer_data, or rotate through designs
    design_style = flyer_data.get('design_style', '').lower()
    
    designs = [
        save_flyer_png_modern_clean,
        save_flyer_png_bold_vibrant,
        save_flyer_png_professional_business,
        save_flyer_png_retro_playful,
        save_flyer_png_sporty_dynamic,
        save_flyer_png_minimalist_cool,
        save_flyer_png_festival_fun
    ]
    
    # Map design names or use hash-based rotation for variety
    if design_style == 'bold' or design_style == 'vibrant':
        chosen_design = save_flyer_png_bold_vibrant
    elif design_style == 'professional' or design_style == 'business':
        chosen_design = save_flyer_png_professional_business
    elif design_style == 'retro' or design_style == 'playful':
        chosen_design = save_flyer_png_retro_playful
    elif design_style == 'sporty' or design_style == 'dynamic':
        chosen_design = save_flyer_png_sporty_dynamic
    elif design_style == 'minimalist' or design_style == 'cool':
        chosen_design = save_flyer_png_minimalist_cool
    elif design_style == 'festival' or design_style == 'fun':
        chosen_design = save_flyer_png_festival_fun
    else:
        # Use hash of headline to consistently pick same design for same event
        headline_hash = int(hashlib.md5(flyer_data['headline'].encode()).hexdigest(), 16)
        chosen_design = designs[headline_hash % len(designs)]
    
    print(f"📐 Using design template: {chosen_design.__name__.replace('save_flyer_png_', '').replace('_', ' ').title()}")
    return chosen_design(flyer_data, qr_path, output_path)

async def email_generation_node(state: State) -> Command[Literal["form_generation", "__end__"]]:
    print_banner("📧 STEP 1: PROPOSAL EMAIL")
    
    event_details = state.get("event_details", {})
    truncated_messages = truncate_messages(state["messages"])
    print(f"✉️ Generating proposal email for: {event_details.get('event_name', 'Unnamed Event')}")
    
    # Create a concise email generation prompt
    email_task_prompt = f"""
    Write a SHORT, friendly proposal email to church leadership.
    
    STRICT REQUIREMENT: Start with ONLY "Good evening Brothers," - do not add Sisters or any variation.
    
    Keep it brief (under 200 words):
    - What: Event name and purpose
    - When: Date and time
    - Where: Location
    - Cost: Per person cost, youth payment amount, church subsidy needed
    - Action: Request approval
    
    Tone: Warm, genuine, conversational - like talking to friends.
    
    Format:
    --- EMAIL BODY START ---
    Good evening Brothers,
    
    [Brief 2-3 sentence intro]
    
    [Event details and cost breakdown in 2-3 lines]
    
    [One sentence closing request]
    
    Blessings,
    [Your name]
    --- EMAIL BODY END ---
    
    After writing, call save_proposal_email with:
    - email_body: your email text
    - event_name: "{event_details.get('event_name', 'Unnamed Event')}"
    - recipient: "pastor@church.org"
    
    Event Details:
    {json.dumps(event_details, indent=2)}
    """
    
    try:
        # Create email agent with enhanced prompt
        email_agent = create_agent(llm, tools=[save_proposal_email], system_prompt=email_task_prompt)
        response = await email_agent.ainvoke({"messages": truncated_messages})
        
        # Extract email content and check if tool was called
        email_content = None
        tool_result = None
        tool_used = False
        
        for msg in response["messages"]:
            if msg.type == "ai":
                email_content = msg.content
            # Check for tool use in ToolUseBlock or other message types
            if hasattr(msg, 'name') and msg.name == "save_proposal_email":
                tool_used = True
            # Look for tool results
            if hasattr(msg, 'content') and isinstance(msg.content, str) and "file_path" in msg.content:
                tool_result = msg.content
                tool_used = True
        
        # Display the email content
        if email_content:
            print("\n📄 GENERATED EMAIL:")
            print("="*60)
            print(email_content)
            print("="*60)
        else:
            print("⚠️ No email content generated")
        
        # If tool was used, confirm
        if tool_used:
            print("✅ Email archived successfully")
        else:
            # Fallback: Extract email text and save manually
            print("⚠️ Extracting email from output and saving manually...")
            if email_content and "EMAIL BODY" in email_content:
                # Try to extract the email between markers
                start_marker = "--- EMAIL BODY START ---"
                end_marker = "--- EMAIL BODY END ---"
                start_idx = email_content.find(start_marker)
                end_idx = email_content.find(end_marker)
                
                if start_idx != -1 and end_idx != -1:
                    extracted_email = email_content[start_idx + len(start_marker):end_idx].strip()
                    if extracted_email:
                        result = save_proposal_email(
                            email_body=extracted_email,
                            event_name=event_details.get('event_name', 'Unnamed Event'),
                            recipient="pastor@church.org"
                        )
                        print("✅ Email saved via fallback method")
                        tool_used = True
            
            if not tool_used:
                print("⚠️ Email generated but not archived - see content above")
        
    except Exception as e:
        print(f"❌ Error generating email: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n➡️ Moving to form generation...")
    return Command(update={"messages": response["messages"], "event_details": event_details}, goto="form_generation")

async def form_generation_node(state: State) -> Command[Literal["flyer_generation", "__end__"]]:
    print_banner("📋 STEP 2: REGISTRATION FORM")
    
    event_details = state.get("event_details", {})
    form_url = event_details.get("form_url", "https://forms.example.com/register")
    
    # Check if hosted_app_url is provided - use it instead of form_url
    hosted_app_url = event_details.get("hosted_app_url")
    if hosted_app_url:
        form_url = hosted_app_url
        print(f"🌐 Using hosted app URL: {form_url}")
    else:
        print(f"🔗 Form will link to: {form_url}")
    
    truncated_messages = truncate_messages(state["messages"])
    
    # Setup waiver system
    print("\n📋 Setting up waiver system...")
    waiver_dir = setup_waiver_directory()
    waiver_pdf_path = copy_waiver_pdf()
    print(f"📁 Waivers will be stored in: {waiver_dir}")
    
    # Generate the standardized form with event details
    form_schema = generate_standardized_form(event_details)
    
    # Display form schema
    print("\n📋 FORM SCHEMA GENERATED:")
    print("="*60)
    print(json.dumps(form_schema, indent=2))
    print("="*60)
    
    # Save form schema to file for form_app.py to load automatically
    form_file_path = os.path.join(os.path.dirname(__file__), "current_event_form.json")
    with open(form_file_path, 'w') as f:
        json.dump(form_schema, f, indent=2)
    
    print(f"\n💾 Form schema saved to: {form_file_path}")
    print("   form_app.py will automatically load this!")
    
    # Generate QR code linking to form_url (or hosted_app_url if provided)
    qr_path = create_qr_png(form_url)
    
    print(f"✅ Form generated with QR code: {qr_path}")
    print("📤 To sync to your hosted app, push current_event_form.json to GitHub manually")
    
    print("\n➡️ Moving to flyer generation...")
    return Command(
        update={
            "messages": truncated_messages,
            "event_details": event_details,
            "form_url": form_url,
            "qr_path": qr_path
        },
        goto="flyer_generation"
    )

async def flyer_generation_node(state: State) -> Command[Literal["__end__"]]:
    print_banner("🎨 STEP 3: FLYER WITH CREATIVE DESIGNS")
    
    event_details = state.get("event_details", {})
    form_url = state.get("form_url", event_details.get("form_url", "https://forms.example.com/register"))
    qr_path = state.get("qr_path")
    truncated_messages = truncate_messages(state["messages"])
    
    print(f"🎯 Creating flyer for: {event_details.get('event_name', 'Unnamed Event')}")
    if qr_path:
        print(f"📱 Using existing QR code: {qr_path}")
    else:
        print(f"📱 Will generate new QR linking to: {form_url}")
    
    # STEP 1: Designer agent generates multiple creative variations
    print_banner("👨‍🎨 FLYER DESIGNER: Generating Creative Concepts", "🎨")
    
    designer_prompt = f"""
    You are a creative flyer designer for youth events. Your job is to generate multiple creative design concepts.
    
    TASK: Call the design_flyer_variations tool with these event details:
    {json.dumps(event_details, indent=2)}
    
    This will generate 5 different creative design variations (modern, retro, sporty, minimalist, festival).
    Review them and be ready to recommend the best one to the editor.
    
    After calling the tool, provide a brief recommendation about which design you think would work best for this event and why.
    """
    
    designer_agent = create_agent(llm, tools=[design_flyer_variations], system_prompt=designer_prompt)
    designer_response = await designer_agent.ainvoke({"messages": truncated_messages})
    
    # Extract the variations from the designer's response
    design_variations = None
    designer_output = None
    for msg in designer_response["messages"]:
        if msg.type == "ai":
            designer_output = msg.content
    
    print("\n👨‍🎨 DESIGNER RECOMMENDATIONS:")
    if designer_output:
        print("="*60)
        print(designer_output[:600] + ("..." if len(designer_output) > 600 else ""))
        print("="*60)
    
    # Extract variations from tool call results if available
    tool_msgs = [m for m in designer_response["messages"] if hasattr(m, 'name') and m.name == "design_flyer_variations"]
    if tool_msgs and hasattr(tool_msgs[-1], 'content'):
        try:
            variations_data = json.loads(tool_msgs[-1].content)
            if 'variations' in variations_data:
                design_variations = variations_data['variations']
        except:
            pass
    
    # If we couldn't extract variations, generate them directly
    if not design_variations:
        print("⚠️ Generating variations directly...")
        from langchain.tools import tool
        @tool
        def get_variations(event_details):
            result = design_flyer_variations(event_details)
            return result
        
        var_result = await get_variations.ainvoke(event_details)
        design_variations = var_result.get('variations')
    
    if not design_variations:
        print("❌ Could not generate design variations")
        design_variations = []
    
    # STEP 2: Editor agent selects and refines the best design
    print_banner("✏️ FLYER EDITOR: Selecting Best Design", "📝")
    
    editor_prompt = f"""
    You are a flyer editor. The designer has created these flyer design variations:
    
    {json.dumps(design_variations, indent=2)}
    
    TASK:
    1. Analyze which design best fits this youth event
    2. You can slightly refine it (improve copy, adjust colors if needed)
    3. Call the select_and_render_flyer tool with your chosen design
    
    Consider:
    - The event type and target audience
    - Which design will grab attention and encourage registration
    - The energy level and vibe that matches the event
    
    After analyzing, call select_and_render_flyer with the best design (you can refine the copy if it will be better).
    
    Event Details:
    {json.dumps(event_details, indent=2)}
    
    Form URL for QR code: {form_url}
    """
    
    editor_agent = create_agent(llm, tools=[select_and_render_flyer], system_prompt=editor_prompt)
    editor_response = await editor_agent.ainvoke({"messages": designer_response["messages"]})
    
    # Extract editor output
    editor_output = None
    for msg in editor_response["messages"]:
        if msg.type == "ai":
            editor_output = msg.content
    
    if editor_output:
        print("\n✏️ EDITOR SELECTION:")
        print("="*60)
        print(editor_output[:600] + ("..." if len(editor_output) > 600 else ""))
        print("="*60)
    
    print("\n✅ Creative flyer generated with embedded QR code!")
    print("🎨 Design variations explored and best option selected")
    print("📁 All files exported to current directory")
    
    return Command(
        update={
            "messages": editor_response["messages"],
            "event_details": event_details,
            "form_url": form_url,
            "qr_path": qr_path
        },
        goto="__end__"
    )


def get_task_prompt(task_type: str, payload: Dict[str, Any]) -> str:
    # Simplified - now only used internally
    event_details = payload.get("event_details", {})
    
    # Proposal email prompt
    email_prompt = f"""
    Write a friendly, conversational proposal email to church leadership.
    Output PLAIN TEXT email body ONLY (no JSON), ready to send.

    Tone & Style:
    - Warm, personal greeting (e.g., "Good evening Brothers," or similar)
    - Conversational and genuine - not corporate or stuffy
    - Like talking to community members, not a formal business letter
    - Clear, friendly, and respectful
    
    Structure:
    - Subject line as first line: "Subject: <title>"
    - Personal greeting
    - Event details and purpose (what, when, where, why it's great)
    - Simple cost breakdown (total per person, youth pays, church covers difference)
    - Specific dates and logistics
    - Warm closing with signature
    - Friendly tone throughout
    
    After writing, call the save_proposal_email tool to archive it.
    
    Event: {json.dumps(event_details, indent=2)}
    """
    return email_prompt

def extractjsonfromtext(text: str) -> dict:
    """Extract JSON from LLM output with multiple strategies."""
    # Strategy 1: Try direct JSON parsing
    try:
        return json.loads(text)
    except:
        pass
    
    # Strategy 2: Look for JSON between triple backticks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except:
            pass
    
    # Strategy 3: Find first { and match braces
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    potential_json = text[start_idx:i+1]
                    try:
                        return json.loads(potential_json)
                    except:
                        pass
    
    # Strategy 4: Use repair_json library as fallback
    try:
        repaired = repair_json(text)
        if repaired:
            result = json.loads(repaired)
            if isinstance(result, dict):
                return result
    except:
        pass
    
    raise ValueError("No valid JSON found")

# def extract_json_from_text(text: str) -> dict:
#     """Extract JSON from LLM output with multiple strategies."""
#     import re
    
#     # Strategy 1: Look for JSON between triple backticks
#     json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
#     if json_match:
#         try:
#             return json.loads(json_match.group(1).strip())
#         except:
#             pass
    
#     # Strategy 2: Find first { and match braces
#     start_idx = text.find('{')
#     if start_idx != -1:
#         brace_count = 0
#         for i in range(start_idx, len(text)):
#             if text[i] == '{':
#                 brace_count += 1
#             elif text[i] == '}':
#                 brace_count -= 1
#                 if brace_count == 0:
#                     potential_json = text[start_idx:i+1]
#                     try:
#                         return json.loads(potential_json)
#                     except:
#                         pass
    
#     # Strategy 3: Try multiple regex patterns
#     patterns = [
#         r'\{[\s\S]*\}',  # Greedy match
#         r'(?<=\n)\{[\s\S]*?\}(?=\n)',  # Match between newlines
#     ]
#     for pattern in patterns:
#         matches = re.findall(pattern, text)
#         for match in matches:
#             try:
#                 return json.loads(match)
#             except:
#                 continue
    
#     raise ValueError("No valid JSON found in output")


# --- TOOL CANDIDATE ---
# This function is a good candidate to expose as a tool:
# - Input: form_url (string)
# - Output: path to saved QR PNG
# - Used for: all tasks that need a QR code (flyer, basketball_clinic, youth_registration_form)
def create_qr_png(form_url: str, output_path: str = "event_qr.png") -> str:
    print_banner("📱 GENERATING QR CODE")
    print(f"🔗 Linking to: {form_url}")
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(form_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    
    print(f"✅ QR SAVED to: {os.path.abspath(output_path)}")  # FULL PATH
    print("📲 Ready for flyers/texts!")
    return output_path


# --- LANGGRAPH TOOL: Process Waiver Submission ---
# This tool processes and stores signed waiver information from registrations
@tool
def process_waiver_submission(participant_data: dict, event_name: str = "Youth Event") -> dict:
    """
    Process and store a signed waiver from a participant.
    
    This tool takes registration data (including waiver signature) and stores it securely.
    
    Args:
        participant_data: Dictionary with participant info including:
            - youth_first_last_name: Youth's full name
            - youth_age: Youth's age
            - parent_first_last_name: Parent/Guardian name
            - parent_phone: Parent/Guardian phone
            - waiver_signature: Name as signed on waiver
            - waiver_date: Date waiver was signed
            - waiver_acknowledgment: Boolean - waiver was acknowledged
        event_name: Name of the event
    
    Returns:
        Dictionary with waiver storage details and confirmation
    """
    result = save_signed_waiver(participant_data, event_name)
    return result


# --- LANGGRAPH TOOL: Flyer Design Variations ---
# This tool generates multiple creative flyer design concepts for the editor to choose from
@tool
def design_flyer_variations(event_details: dict, event_name: str = "Youth Event") -> dict:
    """
    Generate multiple creative flyer design variations to choose from.
    
    This tool creates several distinct design concepts in JSON format that capture
    different creative approaches - modern, playful, sporty, minimalist, etc.
    
    Args:
        event_details: Dictionary with event info (name, date, time, location, description)
        event_name: Name of the event (default: "Youth Event")
    
    Returns:
        Dictionary with:
        - "variations": List of 5 different flyer design concepts as JSON objects
        - "descriptions": Brief description of each design's style and vibe
    """
    
    # Format the date for flyer display
    formatted_date = format_event_date(event_details.get('event_date', 'Coming Soon'))
    date_time_line = formatted_date + " • " + event_details.get('event_time', 'TBA')
    
    design_variations = [
        {
            "name": "Modern Clean",
            "vibe": "Professional, polished, modern with clean lines",
            "design_style": "modern",
            "headline": f"🎯 {event_details.get('event_name', event_name)}",
            "subheadline": "Get ready for an amazing experience!",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": "Join us for an incredible event featuring fun, friends, and unforgettable memories. Perfect for ages 12-18!",
            "call_to_action": "Register Now!",
            "color_scheme": {"primary": "#2E7D32", "accent": "#FFC107"}
        },
        {
            "name": "Retro Playful",
            "vibe": "Fun, nostalgic 90s style with bouncy energy",
            "design_style": "retro",
            "headline": f"🎉 {event_details.get('event_name', event_name).upper()}!",
            "subheadline": "This is going to be LIT! ✨",
            "date_time_line": date_time_line,
            "location_line": f"📍 {event_details.get('location', 'Location TBA')}",
            "body_blurb": "Get hyped! This event is packed with activities, games, friends, and good vibes. You don't want to miss this!",
            "call_to_action": "Let's Go! 🚀",
            "color_scheme": {"primary": "#FF6B6B", "accent": "#4ECDC4"}
        },
        {
            "name": "Sporty Dynamic",
            "vibe": "High-energy, athletic, action-packed vibes",
            "design_style": "sporty",
            "headline": f"⚡ {event_details.get('event_name', event_name)}",
            "subheadline": "Game On!",
            "date_time_line": date_time_line,
            "location_line": f"🏟️ {event_details.get('location', 'Location TBA')}",
            "body_blurb": f"Ready to bring your A-game? Join us for {event_details.get('event_name', 'this event')} and show what you've got!",
            "call_to_action": "Sign Me Up! 🏆",
            "color_scheme": {"primary": "#000000", "accent": "#FF6B35"}
        },
        {
            "name": "Minimalist Cool",
            "vibe": "Ultra-modern, sophisticated, clean aesthetic",
            "design_style": "minimalist",
            "headline": f"{event_details.get('event_name', event_name)}",
            "subheadline": "Be there.",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": "An unforgettable experience awaits. Simple. Elegant. Powerful.",
            "call_to_action": "Join Us",
            "color_scheme": {"primary": "#1A1A1A", "accent": "#00D9FF"}
        },
        {
            "name": "Festival Fun",
            "vibe": "Colorful, celebratory, carnival atmosphere",
            "design_style": "festival",
            "headline": f"🎪 {event_details.get('event_name', event_name)} 🎪",
            "subheadline": "The Event of the Year!",
            "date_time_line": f"📅 {date_time_line}",
            "location_line": f"🎟️ {event_details.get('location', 'Location TBA')}",
            "body_blurb": "Music, games, food, friends, and NON-STOP FUN! This is the event everyone will be talking about!",
            "call_to_action": "Get Your Ticket! 🎟️",
            "color_scheme": {"primary": "#FF1493", "accent": "#00CED1"}
        }
    ]
    
    descriptions = [
        "Design 1: Clean, professional look - great for all-ages events",
        "Design 2: Fun and playful 90s vibe - perfect for getting kids excited",
        "Design 3: High-energy sports style - ideal for athletic/competitive events",
        "Design 4: Sophisticated modern - works for any event that wants elegance",
        "Design 5: Festive carnival - maximum energy and fun for youth events"
    ]
    
    return {
        "variations": design_variations,
        "descriptions": descriptions,
        "note": "Pick your favorite design number (1-5) or mix & match elements!"
    }


# --- LANGGRAPH TOOL: Select and Render Flyer ---
# This tool takes a selected design variation and renders the final flyer
@tool
def select_and_render_flyer(selected_design: dict, form_url: str = "https://forms.example.com/register") -> dict:
    """
    Select a flyer design variation, optionally refine it, and render the final flyer PNG.
    
    This tool takes a selected design concept and generates:
    - A QR code PNG linking to the form_url
    - A flyer PNG with the selected design
    
    Args:
        selected_design: The chosen design variation dict with all design fields
        form_url: URL to embed in the QR code for registration
    
    Returns:
        Dictionary with:
        - "flyer_path": Absolute path to generated flyer PNG
        - "qr_path": Absolute path to generated QR code PNG
        - "design_used": Name of the design style used
    """
    
    print_banner("🎨 RENDERING FINAL FLYER", "✨")
    
    # Validate design has required fields
    required = ['headline', 'subheadline', 'date_time_line', 'location_line', 'body_blurb', 'call_to_action', 'color_scheme']
    missing = [f for f in required if f not in selected_design]
    if missing:
        raise ValueError(f"Selected design missing fields: {missing}")
    
    # Generate QR code
    qr_path = create_qr_png(form_url)
    
    # Generate flyer with selected design
    flyer_path = save_flyer_png(selected_design, qr_path)
    
    design_name = selected_design.get('name', 'Custom')
    
    return {
        "flyer_path": flyer_path,
        "qr_path": qr_path,
        "design_used": design_name
    }


# --- LANGGRAPH TOOL: Flyer Package Generation ---
# This tool is exposed to the writer agent and can be called during the writing phase.
# It encapsulates the logic for generating both QR codes and flyer PNGs.
@tool
def generate_flyer_package(flyer_data: dict, form_url: str = "https://forms.example.com/register") -> dict:
    """
    Generate a complete flyer package including PNG and QR code.
    
    This tool takes flyer content data and generates:
    - A QR code PNG linking to the provided form_url
    - A flyer PNG with all the event details
    
    Args:
        flyer_data: Dictionary with required fields:
            - headline: Main event title (str)
            - subheadline: Subtitle (str)
            - date_time_line: When the event is (str)
            - location_line: Where the event is (str)
            - body_blurb: Event description (str)
            - call_to_action: What to do next (str)
            - color_scheme: Dict with 'primary' and 'accent' hex colors (dict)
        form_url: URL to embed in the QR code for registration (str, default: https://forms.example.com/register)
    
    Returns:
        Dictionary with:
        - "flyer_path": Absolute path to generated flyer PNG
        - "qr_path": Absolute path to generated QR PNG
    """
    qr_path = create_qr_png(form_url)
    flyer_path = save_flyer_png(flyer_data, qr_path)
    return {
        "flyer_path": flyer_path,
        "qr_path": qr_path
    }


# --- LANGGRAPH TOOL: Registration Form Builder ---
# This tool is exposed to the writer agent and can be called during the writing phase.
# It encapsulates the logic for extracting form JSON and generating QR codes.
@tool
def build_registration_form(llm_output: str, form_url: str = "https://forms.example.com/register") -> dict:
    """
    Build and validate a registration form schema with QR code.
    
    This tool takes LLM-generated form schema content and generates:
    - A parsed and validated form schema dictionary
    - A QR code PNG linking to the provided form_url
    
    Args:
        llm_output: The LLM-generated form schema as a string (should be valid JSON)
        form_url: URL to embed in the QR code for registration (str, default: https://forms.example.com/register)
    
    Returns:
        Dictionary with:
        - "form_schema": Parsed form schema dictionary with fields (title, description, fields)
        - "qr_path": Absolute path to generated QR code PNG
    """
    form_schema = extractjsonfromtext(llm_output)
    qr_path = create_qr_png(form_url)
    return {
        "form_schema": form_schema,
        "qr_path": qr_path
    }


# --- LANGGRAPH TOOL: Proposal Email Save ---
# This tool is exposed to the editor agent and can be called at the end of email refinement.
# It encapsulates the logic for exporting proposal emails to timestamped files.
@tool
def save_proposal_email(email_body: str, event_name: str = "Unnamed Event", recipient: str = "pastor@church.org") -> dict:
    """
    Save a proposal email to a timestamped text file for archival.
    
    This tool takes the finalized email text and saves it with recipient info
    to a text file for record-keeping and future reference.
    
    Args:
        email_body: The complete email text (as a string)
        event_name: Name of the event (used for filename slug, default: "Unnamed Event")
        recipient: Email recipient address (default: "pastor@church.org")
    
    Returns:
        Dictionary with:
        - "file_path": Absolute path to saved email file
        - "recipient": The recipient email address
        - "event_name": The event name used in the filename
    """
    # Create a safe slug from event_name
    slug = re.sub(r'[^a-z0-9]+', '_', event_name.lower()).strip('_')
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename
    filename = f"proposal_email_{slug}_{timestamp}.txt"
    file_path = os.path.abspath(filename)
    
    # Write file with recipient header
    with open(file_path, 'w') as f:
        f.write(f"To: {recipient}\n")
        f.write(f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
        f.write(f"Event: {event_name}\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write(email_body)
    
    print_banner("💾 EMAIL ARCHIVED", "📧")
    print(f"✅ Email saved to: {file_path}")
    print(f"📬 Recipient: {recipient}")
    print(f"📅 Event: {event_name}")
    
    return {
        "file_path": file_path,
        "recipient": recipient,
        "event_name": event_name
    }


async def main():
    global researcher_agent, writer_agent, editor_agent, llm
    
    print_banner("🚀 UNIFIED EVENT PLANNER v3.0", "✨")
    print("📝 Workflow: Proposal Email → Registration Form → Flyer with QR")
    
    # API checks
    missing = []
    if not os.getenv("GITHUB_TOKEN"): missing.append("GITHUB_TOKEN")
    
    if missing:
        print("⚠️", ", ".join(missing), "→ Add to .env")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0.3,
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv("GITHUB_TOKEN"))
    
    # Base prompts
    base_prompts = {
        "email": "You are a professional proposal writer for youth ministry events.",
        "form": "You are a form designer for youth ministry registration.",
        "flyer": "You are a creative flyer designer for youth events."
    }
    
    # Try to load from external files
    for fname, key in [("email_system.txt", "email"), ("form_system.txt", "form"), ("flyer_system.txt", "flyer")]:
        try:
            with open(fname) as f:
                base_prompts[key] = f.read()
        except:
            pass
    
    # Build the unified graph nodes (each node creates its own agent with detailed prompts)
    builder = StateGraph(State)
    builder.add_node("email_generation", email_generation_node)
    builder.add_node("form_generation", form_generation_node)
    builder.add_node("flyer_generation", flyer_generation_node)
    
    builder.add_edge(START, "email_generation")
    
    graph = builder.compile()
    
    print("\n📋 INPUT FORMAT (JSON with event details):")
    print("""
    {
      "event_name": "Youth Basketball Clinic",
      "event_date": "April 15, 2026",
      "event_time": "2:00 PM - 4:00 PM",
      "location": "Community Center",
      "description": "Fun basketball skills training",
      "form_url": "https://forms.example.com/basketball-2026",
      "hosted_app_url": "https://ai-agent-event-planner.streamlit.app (OPTIONAL)"
    }
    """)
    
    print("\n💡 HOSTED APP URL:")
    print("   If you provide 'hosted_app_url', it will override 'form_url' in the QR code and flyer")
    print("   This should be your deployed Streamlit app URL")
    
    print("\n🎯 The system will automatically:")
    print("  1. Generate a proposal email")
    print("  2. Build a registration form")
    print("  3. Create a QR code linking to the form")
    print("  4. CREATIVE PROCESS: Designer generates 5 design concepts")
    print("  5. SELECTION PROCESS: Editor picks the best design and renders final flyer")
    print("  6. Output flyer with embedded QR code in multiple creative styles!")
    
    while True:
        try:
            payload_str = input("\n📝 Event details JSON (or 'quit'): ").strip()
            if payload_str.lower() in ['quit', 'exit', 'q']: 
                break
            
            payload = json.loads(payload_str)
            event_details = payload
            
            # Validate required fields
            required = ["event_name", "form_url"]
            missing = [f for f in required if f not in event_details]
            if missing:
                print(f"❌ Missing fields: {', '.join(missing)}")
                continue
            
            print(f"\n🎯 Processing: {event_details['event_name']}")
            print_banner("⚙️ PIPELINE STARTING")
            
            # Run the unified pipeline
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=json.dumps(event_details))],
                "event_details": event_details,
                "form_url": event_details.get("form_url"),
                "qr_path": ""
            })
            
            print_banner("✅ COMPLETE PIPELINE FINISHED")
            print("\n📁 All outputs generated:")
            print("  📧 proposal_email_*.txt")
            print("  📋 Registration form schema")
            print("  📱 event_qr.png")
            print("  🎨 flyer.png")
            
        except json.JSONDecodeError:
            print("❌ Invalid JSON format")
        except Exception as e:
            print(f"💥 Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

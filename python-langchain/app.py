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
from html2image import Html2Image  # For HTML to PNG conversion
import random

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

def truncate_messages(messages: list, max_messages: int = 3, keep_first: bool = True) -> list:
    """
    Truncate messages to prevent token limit issues with GPT-4 mini.
    
    Args:
        messages: List of messages to truncate
        max_messages: Maximum number of messages to keep (default: 3 for GPT-4 mini token limits)
        keep_first: Whether to keep the first message (usually the initial context)
    
    Returns:
        Truncated message list
    """
    if len(messages) <= max_messages:
        return messages
    
    # Keep first message (context) + most recent messages to stay under token limit
    if keep_first and len(messages) > 1:
        # Keep 1st message + last (max_messages-1) messages
        return messages[:1] + messages[-(max_messages-1):]
    else:
        # Just keep the last max_messages
        return messages[-max_messages:]

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_random_color_pair() -> tuple:
    """Generate random but complementary color pair (primary, accent)."""
    color_palettes = [
        ("#FF6B6B", "#4ECDC4"),  # Coral & Teal
        ("#6C5CE7", "#A29BFE"),  # Purple & Light Purple
        ("#00B894", "#FDCB6E"),  # Green & Yellow
        ("#E17055", "#74B9FF"),  # Red & Blue
        ("#0984E3", "#FD79A8"),  # Blue & Pink
        ("#27AE60", "#F39C12"),  # Dark Green & Orange
        ("#9B59B6", "#E91E63"),  # Purple & Magenta
        ("#1ABC9C", "#E74C3C"),  # Turquoise & Red
        ("#3498DB", "#2ECC71"),  # Sky Blue & Green
        ("#F94937", "#FFC107"),  # Deep Red & Amber
        ("#FF7675", "#A0522D"),  # Light Red & Brown
        ("#FF85C0", "#6C63FF"),  # Pink & Indigo
        ("#00D2FC", "#3A86FF"),  # Cyan & Blue
        ("#FF006E", "#FFBE0B"),  # Hot Pink & Yellow
        ("#8338EC", "#FB5607"),  # Purple & Orange
    ]
    return random.choice(color_palettes)

def generate_random_layout_variant() -> str:
    """Return a random layout style."""
    return random.choice([
        "gradient_top",
        "sidebar_left",
        "centered_bold",
        "split_diagonal",
        "layered_cards",
        "asymmetric",
        "minimalist_left",
        "full_splash",
        "ribbon_header",
        "hexagon_accent",
        "geometric_bg",
        "bubble_accent"
    ])

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
            {"name": "date", "label": "Date", "type": "date", "required": True}
        ]
    }
  

# ============================================================================
# HTML/CSS-BASED FLYER GENERATION SYSTEM
# ============================================================================

def generate_html_flyer(flyer_data: Dict, qr_base64: str, layout: str = None) -> str:
    """Generate HTML/CSS flyer based on layout style with randomization."""
    
    if not layout:
        layout = generate_random_layout_variant()
    
    primary = flyer_data['color_scheme']['primary']
    accent = flyer_data['color_scheme']['accent']
    
    # Escape HTML special characters
    headline = flyer_data['headline'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    subheadline = flyer_data['subheadline'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    date_time = flyer_data['date_time_line'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    location = flyer_data['location_line'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    body = flyer_data['body_blurb'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    cta = flyer_data['call_to_action'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    base_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Event Flyer</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Arial', 'Segoe UI', sans-serif; background: white; }}
            .flyer {{ width: 1100px; height: 850px; position: relative; overflow: hidden; }}
        </style>
    </head>
    <body>
        <div class="flyer">
    """
    
    if layout == "gradient_top":
        html = base_html + f"""
            <style>
                .gradient-bg {{ background: linear-gradient(135deg, {primary} 0%, {accent} 100%); height: 280px; display: flex; flex-direction: column; justify-content: center; padding: 40px; color: white; }}
                .headline {{ font-size: 56px; font-weight: bold; margin-bottom: 15px; }}
                .subheadline {{ font-size: 32px; opacity: 0.9; }}
                .content {{ padding: 40px; }}
                .info-box {{ margin: 20px 0; padding: 15px; background: {accent}22; border-left: 5px solid {accent}; border-radius: 5px; }}
                .info-title {{ font-weight: bold; color: {primary}; font-size: 18px; }}
                .info-text {{ color: #333; font-size: 16px; margin-top: 5px; }}
                .body-text {{ color: #555; font-size: 16px; line-height: 1.6; margin: 20px 0; }}
                .cta {{ background: {accent}; color: white; padding: 20px 30px; font-size: 24px; font-weight: bold; border-radius: 10px; margin: 20px 0; display: inline-block; }}
                .qr {{ position: absolute; bottom: 20px; right: 20px; width: 180px; height: 180px; background: white; padding: 10px; border-radius: 5px; }}
            </style>
            <div class="gradient-bg">
                <div class="headline">{headline}</div>
                <div class="subheadline">{subheadline}</div>
            </div>
            <div class="content">
                <div class="info-box">
                    <div class="info-title">📅 When</div>
                    <div class="info-text">{date_time}</div>
                </div>
                <div class="info-box">
                    <div class="info-title">📍 Where</div>
                    <div class="info-text">{location}</div>
                </div>
                <div class="body-text">{body}</div>
                <div class="cta">{cta}</div>
            </div>
            <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
        </div>
        </body>
        </html>
        """
    
    elif layout == "sidebar_left":
        html = base_html + f"""
            <style>
                .sidebar {{ width: 45%; height: 100%; background: linear-gradient(180deg, {primary} 0%, {accent} 100%); color: white; padding: 40px; display: flex; flex-direction: column; justify-content: space-between; float: left; }}
                .main {{ width: 55%; height: 100%; padding: 40px; overflow-y: auto; float: right; }}
                .headline {{ font-size: 48px; font-weight: bold; margin-bottom: 20px; }}
                .subheadline {{ font-size: 28px; opacity: 0.9; margin-bottom: 40px; }}
                .info-text {{ font-size: 18px; margin: 10px 0; }}
                .main-headline {{ font-size: 32px; color: {primary}; font-weight: bold; margin-bottom: 20px; }}
                .body-text {{ color: #555; font-size: 16px; line-height: 1.6; margin: 15px 0; }}
                .cta {{ background: {accent}; color: white; padding: 15px 25px; font-size: 20px; font-weight: bold; border-radius: 5px; margin-top: 20px; display: inline-block; }}
                .qr {{ width: 150px; height: 150px; padding: 8px; background: white; border-radius: 3px; margin-top: auto; }}
            </style>
            <div class="sidebar">
                <div>
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                </div>
                <div>
                    <div class="info-text">📅 {date_time}</div>
                    <div class="info-text">📍 {location}</div>
                </div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
            <div class="main">
                <div class="main-headline">About This Event</div>
                <div class="body-text">{body}</div>
                <div class="cta">{cta}</div>
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "centered_bold":
        html = base_html + f"""
            <style>
                .background {{ width: 100%; height: 100%; background: {primary}; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: white; padding: 40px; }}
                .headline {{ font-size: 72px; font-weight: bold; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
                .subheadline {{ font-size: 36px; margin-bottom: 40px; }}
                .content-box {{ background: white; color: #333; padding: 30px; border-radius: 15px; max-width: 700px; margin: 20px auto; }}
                .info-item {{ margin: 15px 0; font-size: 18px; }}
                .body-text {{ color: #666; font-size: 16px; line-height: 1.6; margin: 15px 0; }}
                .cta {{ background: {accent}; color: white; padding: 18px 35px; font-size: 24px; font-weight: bold; border-radius: 10px; margin-top: 20px; display: inline-block; }}
                .qr {{ position: absolute; bottom: 30px; right: 30px; width: 160px; height: 160px; background: white; padding: 10px; border-radius: 5px; }}
            </style>
            <div class="background">
                <div class="headline">{headline}</div>
                <div class="subheadline">{subheadline}</div>
                <div class="content-box">
                    <div class="info-item">📅 {date_time}</div>
                    <div class="info-item">📍 {location}</div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                </div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "split_diagonal":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; position: relative; display: flex; }}
                .left {{ width: 50%; background: {primary}; padding: 40px; color: white; display: flex; flex-direction: column; justify-content: center; clip-path: polygon(0 0, 100% 0, 85% 100%, 0 100%); }}
                .right {{ width: 50%; background: #f5f5f5; padding: 40px; margin-left: -50px; padding-left: 80px; }}
                .headline {{ font-size: 52px; font-weight: bold; margin-bottom: 20px; }}
                .subheadline {{ font-size: 28px; opacity: 0.95; }}
                .right-headline {{ font-size: 32px; color: {primary}; font-weight: bold; margin-bottom: 20px; }}
                .info-item {{ margin: 15px 0; font-size: 17px; color: #333; }}
                .body-text {{ color: #666; font-size: 16px; line-height: 1.6; margin: 20px 0; }}
                .cta {{ background: {accent}; color: white; padding: 16px 30px; font-size: 20px; font-weight: bold; border-radius: 8px; margin-top: 20px; display: inline-block; }}
                .qr {{ position: absolute; bottom: 25px; right: 25px; width: 140px; height: 140px; background: white; padding: 8px; border-radius: 4px; }}
            </style>
            <div class="container">
                <div class="left">
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                </div>
                <div class="right">
                    <div class="right-headline">Event Details</div>
                    <div class="info-item">📅 {date_time}</div>
                    <div class="info-item">📍 {location}</div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                </div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "layered_cards":
        html = base_html + f"""
            <style>
                .background {{ width: 100%; height: 100%; background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%); padding: 30px; }}
                .header-card {{ background: {primary}; color: white; padding: 35px; border-radius: 15px; margin-bottom: -20px; position: relative; z-index: 3; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                .headline {{ font-size: 48px; font-weight: bold; margin-bottom: 10px; }}
                .subheadline {{ font-size: 24px; opacity: 0.9; }}
                .content-card {{ background: white; padding: 40px; border-radius: 15px; margin-top: 30px; position: relative; z-index: 2; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .info-box {{ padding: 15px; background: {accent}15; border-left: 4px solid {accent}; border-radius: 5px; }}
                .info-label {{ font-weight: bold; color: {primary}; font-size: 16px; }}
                .info-value {{ color: #666; font-size: 15px; margin-top: 5px; }}
                .body-text {{ color: #555; font-size: 16px; line-height: 1.6; margin: 20px 0; }}
                .cta {{ background: {accent}; color: white; padding: 18px 35px; font-size: 22px; font-weight: bold; border-radius: 10px; margin: 20px 0; display: inline-block; }}
                .qr {{ position: absolute; bottom: 30px; right: 30px; width: 150px; height: 150px; background: white; padding: 8px; border-radius: 5px; }}
            </style>
            <div class="background">
                <div class="header-card">
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                </div>
                <div class="content-card">
                    <div class="info-grid">
                        <div class="info-box">
                            <div class="info-label">📅 Date & Time</div>
                            <div class="info-value">{date_time}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">📍 Location</div>
                            <div class="info-value">{location}</div>
                        </div>
                    </div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                    <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
                </div>
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "asymmetric":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; background: white; position: relative; }}
                .accent-shape {{ position: absolute; background: {accent}; opacity: 0.1; width: 300px; height: 500px; border-radius: 150px 0 0 150px; right: 0; top: 0; }}
                .header {{ padding: 40px 50px; position: relative; z-index: 2; }}
                .headline {{ font-size: 54px; font-weight: bold; color: {primary}; margin-bottom: 10px; }}
                .subheadline {{ font-size: 28px; color: {accent}; }}
                .main {{ padding: 20px 50px; position: relative; z-index: 2; }}
                .info-block {{ margin: 20px 0; }}
                .info-label {{ font-weight: bold; color: {primary}; font-size: 17px; }}
                .info-value {{ color: #666; font-size: 16px; margin-top: 5px; }}
                .body-text {{ color: #555; font-size: 16px; line-height: 1.6; margin: 25px 0; }}
                .cta {{ background: linear-gradient(135deg, {primary}, {accent}); color: white; padding: 18px 40px; font-size: 22px; font-weight: bold; border-radius: 12px; display: inline-block; }}
                .qr {{ position: absolute; bottom: 40px; right: 40px; width: 170px; height: 170px; background: white; padding: 10px; border-radius: 5px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            </style>
            <div class="container">
                <div class="accent-shape"></div>
                <div class="header">
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                </div>
                <div class="main">
                    <div class="info-block">
                        <div class="info-label">📅</div>
                        <div class="info-value">{date_time}</div>
                    </div>
                    <div class="info-block">
                        <div class="info-label">📍</div>
                        <div class="info-value">{location}</div>
                    </div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                </div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "minimalist_left":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; display: flex; background: white; }}
                .left {{ width: 40%; background: {primary}; padding: 50px; color: white; display: flex; flex-direction: column; justify-content: space-between; }}
                .right {{ width: 60%; padding: 50px; display: flex; flex-direction: column; justify-content: center; }}
                .headline {{ font-size: 44px; font-weight: 900; margin-bottom: 20px; letter-spacing: -1px; }}
                .subheadline {{ font-size: 24px; opacity: 0.9; font-weight: 300; }}
                .right-headline {{ font-size: 36px; color: {primary}; font-weight: bold; margin-bottom: 30px; }}
                .info-item {{ font-size: 18px; margin: 15px 0; color: #444; }}
                .body-text {{ color: #666; font-size: 16px; line-height: 1.7; margin: 20px 0; }}
                .cta {{ background: {accent}; color: white; padding: 16px 32px; font-size: 18px; font-weight: bold; border-radius: 8px; display: inline-block; margin-top: 20px; }}
                .qr {{ width: 120px; height: 120px; background: white; padding: 8px; border-radius: 4px; margin-top: auto; }}
            </style>
            <div class="container">
                <div class="left">
                    <div>
                        <div class="headline">{headline}</div>
                        <div class="subheadline">{subheadline}</div>
                    </div>
                    <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
                </div>
                <div class="right">
                    <div class="right-headline">Event Details</div>
                    <div class="info-item">📅 {date_time}</div>
                    <div class="info-item">📍 {location}</div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                </div>
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "full_splash":
        html = base_html + f"""
            <style>
                .splash {{ width: 100%; height: 100%; background: linear-gradient(135deg, {primary} 0%, {accent} 100%); display: flex; flex-direction: column; justify-content: space-between; padding: 50px; color: white; text-align: center; }}
                .headline {{ font-size: 68px; font-weight: 900; margin-bottom: 15px; text-shadow: 2px 2px 6px rgba(0,0,0,0.3); }}
                .subheadline {{ font-size: 32px; opacity: 0.95; margin-bottom: 40px; }}
                .content {{ background: rgba(255,255,255,0.95); color: #333; padding: 30px; border-radius: 15px; margin: 20px auto; max-width: 600px; }}
                .info-item {{ font-size: 18px; margin: 10px; }}
                .body-text {{ color: #666; font-size: 16px; line-height: 1.6; margin: 15px 0; }}
                .cta {{ background: {primary}; color: white; padding: 16px 35px; font-size: 22px; font-weight: bold; border-radius: 10px; margin-top: 15px; display: inline-block; }}
                .qr {{ position: absolute; top: 20px; right: 20px; width: 150px; height: 150px; background: white; padding: 10px; border-radius: 5px; }}
            </style>
            <div class="splash">
                <div class="headline">{headline}</div>
                <div class="subheadline">{subheadline}</div>
                <div class="content">
                    <div class="info-item">📅 {date_time}</div>
                    <div class="info-item">📍 {location}</div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                </div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "ribbon_header":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; background: #fafafa; padding: 40px; }}
                .ribbon {{ background: {primary}; color: white; padding: 30px; margin: -40px -40px 40px -40px; position: relative; }}
                .ribbon::after {{ content: ''; position: absolute; bottom: -15px; left: 0; right: 0; height: 15px; background: inherit; clip-path: polygon(0 0, 0 50%, 50% 100%, 100% 50%, 100% 0); }}
                .headline {{ font-size: 52px; font-weight: bold; margin-bottom: 10px; }}
                .subheadline {{ font-size: 26px; opacity: 0.9; }}
                .main {{ position: relative; z-index: 1; padding-top: 25px; }}
                .info-section {{ margin: 25px 0; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid {accent}; }}
                .info-label {{ font-weight: bold; color: {primary}; font-size: 16px; }}
                .info-value {{ color: #666; margin-top: 5px; }}
                .body-text {{ color: #555; font-size: 16px; line-height: 1.6; margin: 20px 0; }}
                .cta {{ background: {accent}; color: white; padding: 18px 35px; font-size: 22px; font-weight: bold; border-radius: 10px; display: inline-block; margin: 20px 0; }}
                .qr {{ position: absolute; bottom: 40px; right: 40px; width: 160px; height: 160px; background: white; padding: 8px; border-radius: 5px; }}
            </style>
            <div class="ribbon">
                <div class="headline">{headline}</div>
                <div class="subheadline">{subheadline}</div>
            </div>
            <div class="main">
                <div class="info-section">
                    <div class="info-label">📅 When</div>
                    <div class="info-value">{date_time}</div>
                </div>
                <div class="info-section">
                    <div class="info-label">📍 Where</div>
                    <div class="info-value">{location}</div>
                </div>
                <div class="body-text">{body}</div>
                <div class="cta">{cta}</div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "hexagon_accent":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; background: linear-gradient(to right, #f0f0f0 50%, white 50%); padding: 40px; }}
                .hex {{ position: absolute; right: 30px; top: 30px; width: 200px; height: 200px; background: {accent}; clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%); opacity: 0.2; }}
                .content {{ max-width: 600px; position: relative; z-index: 1; }}
                .headline {{ font-size: 56px; font-weight: bold; color: {primary}; margin-bottom: 10px; }}
                .subheadline {{ font-size: 28px; color: {accent}; margin-bottom: 40px; }}
                .info-item {{ margin: 18px 0; font-size: 18px; color: #555; }}
                .body-text {{ color: #666; font-size: 16px; line-height: 1.7; margin: 25px 0; }}
                .cta {{ background: {primary}; color: white; padding: 18px 40px; font-size: 22px; font-weight: bold; border-radius: 10px; display: inline-block; }}
                .qr {{ position: absolute; bottom: 40px; right: 40px; width: 160px; height: 160px; background: white; padding: 8px; border-radius: 5px; }}
            </style>
            <div class="hex"></div>
            <div class="container">
                <div class="content">
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                    <div class="info-item">📅 {date_time}</div>
                    <div class="info-item">📍 {location}</div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                </div>
                <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "geometric_bg":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; background: white; position: relative; overflow: hidden; }}
                .geo1 {{ position: absolute; width: 300px; height: 300px; background: {primary}; opacity: 0.08; top: -50px; right: -50px; transform: rotate(45deg); }}
                .geo2 {{ position: absolute; width: 250px; height: 250px; background: {accent}; opacity: 0.08; bottom: -30px; left: -30px; border-radius: 50%; }}
                .geo3 {{ position: absolute; width: 200px; height: 300px; background: {accent}; opacity: 0.06; top: 50%; right: 10%; clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%); }}
                .content {{ position: relative; z-index: 1; padding: 50px; }}
                .headline {{ font-size: 56px; font-weight: bold; color: {primary}; margin-bottom: 10px; }}
                .subheadline {{ font-size: 28px; color: {accent}; margin-bottom: 40px; }}
                .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
                .info-box {{ padding: 15px; background: {accent}15; border-radius: 8px; }}
                .info-label {{ font-weight: bold; color: {primary}; }}
                .info-value {{ color: #666; margin-top: 5px; font-size: 15px; }}
                .body-text {{ color: #555; font-size: 16px; line-height: 1.7; margin: 25px 0; }}
                .cta {{ background: linear-gradient(135deg, {primary}, {accent}); color: white; padding: 18px 40px; font-size: 22px; font-weight: bold; border-radius: 10px; display: inline-block; }}
                .qr {{ position: absolute; bottom: 30px; right: 30px; width: 150px; height: 150px; background: white; padding: 8px; border-radius: 5px; }}
            </style>
            <div class="container">
                <div class="geo1"></div>
                <div class="geo2"></div>
                <div class="geo3"></div>
                <div class="content">
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                    <div class="info-grid">
                        <div class="info-box">
                            <div class="info-label">📅 When</div>
                            <div class="info-value">{date_time}</div>
                        </div>
                        <div class="info-box">
                            <div class="info-label">📍 Where</div>
                            <div class="info-value">{location}</div>
                        </div>
                    </div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                    <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
                </div>
            </div>
        </div>
        </body>
        </html>
        """
    
    elif layout == "bubble_accent":
        html = base_html + f"""
            <style>
                .container {{ width: 100%; height: 100%; background: {primary}; padding: 40px; margin: 0; }}
                .bubble {{ position: absolute; background: {accent}; opacity: 0.2; border-radius: 50%; }}
                .bubble1 {{ width: 250px; height: 250px; top: 50px; left: 50px; }}
                .bubble2 {{ width: 200px; height: 200px; bottom: 100px; right: 80px; }}
                .bubble3 {{ width: 150px; height: 150px; top: 60%; left: 70%; }}
                .card {{ background: white; color: #333; padding: 50px; border-radius: 20px; margin: 40px; position: relative; z-index: 1; }}
                .headline {{ font-size: 52px; font-weight: bold; color: {primary}; margin-bottom: 10px; }}
                .subheadline {{ font-size: 28px; color: {accent}; margin-bottom: 30px; }}
                .info-item {{ font-size: 18px; margin: 15px 0; color: #555; }}
                .body-text {{ color: #666; font-size: 16px; line-height: 1.7; margin: 25px 0; }}
                .cta {{ background: {accent}; color: white; padding: 18px 40px; font-size: 22px; font-weight: bold; border-radius: 12px; display: inline-block; }}
                .qr {{ position: absolute; bottom: 30px; right: 30px; width: 160px; height: 160px; background: white; padding: 8px; border-radius: 5px; }}
            </style>
            <div class="container">
                <div class="bubble bubble1"></div>
                <div class="bubble bubble2"></div>
                <div class="bubble bubble3"></div>
                <div class="card">
                    <div class="headline">{headline}</div>
                    <div class="subheadline">{subheadline}</div>
                    <div class="info-item">📅 {date_time}</div>
                    <div class="info-item">📍 {location}</div>
                    <div class="body-text">{body}</div>
                    <div class="cta">{cta}</div>
                    <img class="qr" src="data:image/png;base64,{qr_base64}" alt="QR Code">
                </div>
            </div>
        </div>
        </body>
        </html>
        """
    
    else:
        # Fallback to gradient_top
        return generate_html_flyer(flyer_data, qr_base64, "gradient_top")
    
    return html


def save_flyer_png(flyer_data: Dict, qr_path: str, output_path: str = "flyer.png"):
    """Generate professional flyer using HTML/CSS converted to PNG."""
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

    # Convert QR code to base64
    with open(qr_path, 'rb') as f:
        qr_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Get randomized layout
    layout = generate_random_layout_variant()
    print(f"🎨 Using layout: {layout.replace('_', ' ').title()}")
    
    # Generate HTML
    html_content = generate_html_flyer(flyer_data, qr_base64, layout)
    
    # Convert HTML to PNG using html2image
    try:
        hti = Html2Image()
        # Fix: use html_str instead of html_string
        hti.screenshot(html_str=html_content, save_as=output_path, size=(1100, 850))
        print(f"✅ FLYER SAVED to: {os.path.abspath(output_path)}")
        print(f"📐 Layout: {layout} | Design: HTML/CSS")
        return output_path
    except Exception as e:
        print(f"❌ Error converting HTML to PNG: {e}")
        # Fallback: save HTML for debugging with UTF-8 encoding
        html_file = output_path.replace('.png', '.html')
        # Use UTF-8 encoding to handle emojis and special characters
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"💾 Saved HTML for debugging: {html_file}")
        raise

async def email_generation_node(state: State) -> Command[Literal["form_generation", "__end__"]]:
    print_banner("📧 STEP 1: PROPOSAL EMAIL")
    
    event_details = state.get("event_details", {})
    truncated_messages = truncate_messages(state["messages"], max_messages=2)
    print(f"✉️ Generating proposal email for: {event_details.get('event_name', 'Unnamed Event')}")
    print(f"📊 Message history: {len(state['messages'])} → {len(truncated_messages)} messages")
    
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
            print(email_content[:500])  # Truncate display
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
    
    # Truncate response messages to keep downstream token usage low
    final_messages = truncate_messages(response.get("messages", truncated_messages), max_messages=2)
    print("\n➡️ Moving to form generation...")
    return Command(
        update={"messages": final_messages, "event_details": event_details},
        goto="form_generation"
    )

async def form_generation_node(state: State) -> Command[Literal["flyer_generation", "__end__"]]:
    print_banner("📋 STEP 2: REGISTRATION FORM")
    
    event_details = state.get("event_details", {})
    truncated_messages = truncate_messages(state["messages"], max_messages=2)
    print(f"📊 Message history: {len(state['messages'])} → {len(truncated_messages)} messages")
    
    form_url = event_details.get("form_url", "https://forms.example.com/register")
    
    # Check if hosted_app_url is provided - use it instead of form_url
    hosted_app_url = event_details.get("hosted_app_url")
    if hosted_app_url:
        form_url = hosted_app_url
        print(f"🌐 Using hosted app URL: {form_url}")
    else:
        print(f"🔗 Form will link to: {form_url}")
    
    # Generate the standardized form with event details
    form_schema = generate_standardized_form(event_details)
    
    # Display form schema
    print("\n📋 FORM SCHEMA GENERATED:")
    print("="*60)
    print(json.dumps(form_schema, indent=2)[:500])  # Truncate display
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
    truncated_messages = truncate_messages(state["messages"], max_messages=2)
    print(f"📊 Message history: {len(state['messages'])} → {len(truncated_messages)} messages")
    
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
    # Use minimal message history for designers
    designer_response = await designer_agent.ainvoke({"messages": truncate_messages(truncated_messages, max_messages=1)})
    
    # Extract the variations from the designer's response
    design_variations = None
    designer_output = None
    for msg in designer_response["messages"]:
        if msg.type == "ai":
            designer_output = msg.content
    
    print("\n👨‍🎨 DESIGNER RECOMMENDATIONS:")
    if designer_output:
        print("="*60)
        print(designer_output[:400] + ("..." if len(designer_output) > 400 else ""))
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
    # Use minimal message history for editor
    editor_response = await editor_agent.ainvoke({"messages": truncate_messages(truncated_messages, max_messages=1)})
    
    # Extract editor output
    editor_output = None
    for msg in editor_response["messages"]:
        if msg.type == "ai":
            editor_output = msg.content
    
    if editor_output:
        print("\n✏️ EDITOR SELECTION:")
        print("="*60)
        print(editor_output[:400] + ("..." if len(editor_output) > 400 else ""))
        print("="*60)
    
    print("\n✅ Creative flyer generated with embedded QR code!")
    print("🎨 Design variations explored and best option selected")
    print("📁 All files exported to current directory")
    
    return Command(
        update={
            "messages": truncate_messages(editor_response["messages"], max_messages=2),
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
    
    # Generate random color pairs for each variation for more diversity
    color_pair_1 = generate_random_color_pair()
    color_pair_2 = generate_random_color_pair()
    color_pair_3 = generate_random_color_pair()
    color_pair_4 = generate_random_color_pair()
    color_pair_5 = generate_random_color_pair()
    
    design_variations = [
        {
            "name": "Gradient Splash",
            "vibe": "Modern gradient background with bold typography",
            "design_style": "gradient_top",
            "headline": f"🎯 {event_details.get('event_name', event_name)}",
            "subheadline": "Get ready for an amazing experience!",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": "Join us for an incredible event featuring fun, friends, and unforgettable memories. Perfect for ages 12-18!",
            "call_to_action": "Register Now!",
            "color_scheme": {"primary": color_pair_1[0], "accent": color_pair_1[1]}
        },
        {
            "name": "Sidebar Energy",
            "vibe": "Eye-catching sidebar with modern layout",
            "design_style": "sidebar_left",
            "headline": f"🎉 {event_details.get('event_name', event_name).upper()}!",
            "subheadline": "This is going to be amazing! ✨",
            "date_time_line": date_time_line,
            "location_line": f"📍 {event_details.get('location', 'Location TBA')}",
            "body_blurb": "Get excited! This event is packed with activities, games, friends, and good vibes. You don't want to miss this!",
            "call_to_action": "Join Us! 🚀",
            "color_scheme": {"primary": color_pair_2[0], "accent": color_pair_2[1]}
        },
        {
            "name": "Bold Center",
            "vibe": "Centered bold design with impact",
            "design_style": "centered_bold",
            "headline": f"⚡ {event_details.get('event_name', event_name)}",
            "subheadline": "Be There!",
            "date_time_line": date_time_line,
            "location_line": f"📍 {event_details.get('location', 'Location TBA')}",
            "body_blurb": f"Ready for an incredible experience? Join us for {event_details.get('event_name', 'this event')} and make memories that last!",
            "call_to_action": "Sign Me Up! 🏆",
            "color_scheme": {"primary": color_pair_3[0], "accent": color_pair_3[1]}
        },
        {
            "name": "Minimalist Cool",
            "vibe": "Clean, sophisticated, and modern",
            "design_style": "minimalist_left",
            "headline": f"{event_details.get('event_name', event_name)}",
            "subheadline": "Be there.",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": "An unforgettable experience awaits. Simple. Elegant. Powerful.",
            "call_to_action": "Join Us",
            "color_scheme": {"primary": color_pair_4[0], "accent": color_pair_4[1]}
        },
        {
            "name": "Dynamic Geometric",
            "vibe": "Modern with geometric shapes and visual interest",
            "design_style": "geometric_bg",
            "headline": f"🎪 {event_details.get('event_name', event_name)} 🎪",
            "subheadline": "Event of the Season!",
            "date_time_line": f"📅 {date_time_line}",
            "location_line": f"📍 {event_details.get('location', 'Location TBA')}",
            "body_blurb": "Experience something amazing with friends, games, and non-stop fun! This is the event everyone will be talking about!",
            "call_to_action": "Get Your Spot! 🎟️",
            "color_scheme": {"primary": color_pair_5[0], "accent": color_pair_5[1]}
        }
    ]
    
    descriptions = [
        "Design 1: Modern gradient - clean and contemporary look",
        "Design 2: Sidebar energy - bold visual divide for impact",
        "Design 3: Bold center - maximum impact with centered design",
        "Design 4: Minimalist cool - sophisticated and elegant",
        "Design 5: Geometric - modern shapes and visual depth"
    ]
    
    return {
        "variations": design_variations,
        "descriptions": descriptions,
        "note": "Each design has a unique color palette and layout. Pick your favorite or mix & match!"
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

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
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None
from PIL import Image, ImageDraw, ImageFont  # For flyer PNG
import textwrap
from json_repair import repair_json
import re
import hashlib
import shutil
from pathlib import Path
import base64
try:
    import tiktoken
except ImportError:
    tiktoken = None

# HTML to PNG conversion
try:
    from html2image import Html2Image
except ImportError:
    Html2Image = None

load_dotenv()

# Ensure .env is loaded from project directory (fallback if not already loaded)
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
if env_path.exists() and not os.getenv("TAVILY_API_KEY"):
    load_dotenv(dotenv_path=env_path)


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

def count_tokens(text, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken. Handles LangChain message objects."""
    # Extract content from LangChain message objects
    if hasattr(text, "content"):
        # Handle HumanMessage, AIMessage, etc.
        text = text.content
    elif hasattr(text, "__str__"):
        # Fallback to string representation
        text = str(text)
    
    # Ensure we have a string
    if not isinstance(text, str):
        text = str(text)
    
    if tiktoken is None:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback estimation
        return len(text) // 4

def truncate_messages(messages: list, max_tokens: int = 6000, model: str = "gpt-4o-mini") -> list:
    """
    Truncate messages to stay within token limit for gpt-4o-mini (8000 max).
    Keeps system message (first) and recent messages, removes older ones first.
    Target: 6000 tokens to leave room for response.
    """
    if not messages:
        return messages
    
    # Calculate total tokens in all messages
    total_tokens = sum(count_tokens(msg) for msg in messages)
    
    # If under limit, return as-is
    if total_tokens <= max_tokens:
        return messages
    
    print(f"⚠️  Message size too large ({total_tokens} tokens). Truncating to stay under {max_tokens}...")
    
    # Keep first message (system/context) and progressively add recent ones until we hit the limit
    if len(messages) <= 1:
        return messages
    
    # Start with the first message
    truncated = [messages[0]]
    current_tokens = count_tokens(messages[0])
    
    # Add recent messages from the end, working backwards
    for msg in reversed(messages[1:]):
        msg_tokens = count_tokens(msg)
        if current_tokens + msg_tokens <= max_tokens:
            truncated.insert(1, msg)  # Insert after first message
            current_tokens += msg_tokens
        else:
            # Can't fit more, stop here
            break
    
    print(f"✂️  Reduced from {len(messages)} messages to {len(truncated)} messages ({current_tokens} tokens)")
    return truncated

def convert_html_to_png(html_file_path: str, output_png_path: str = None) -> str:
    """
    Convert an HTML flyer to PNG image.
    
    Args:
        html_file_path: Path to the HTML flyer file
        output_png_path: Path where PNG should be saved (auto-generated if not provided)
    
    Returns:
        Path to the generated PNG file
    
    Raises:
        RuntimeError: If html2image is not installed or conversion fails
    """
    if not Html2Image:
        raise RuntimeError(
            "html2image is not installed. Install with: pip install html2image\n"
            "You may also need: pip install pillow\n"
            "Note: Requires Chrome or Firefox installed on your system."
        )
    
    # Generate output path if not provided
    if output_png_path is None:
        html_path = Path(html_file_path)
        output_png_path = str(html_path.parent / f"{html_path.stem}_flyer.png")
    
    try:
        # Initialize HTML to Image converter
        hti = Html2Image()
        
        # Set options for better quality
        # Size: 1000x1200 pixels (standard flyer size)
        # Quality: high detail
        hti.draw_method = 'svg'  # Better rendering
        
        # Set output directory and filename separately
        output_path_obj = Path(output_png_path)
        hti.output_path = str(output_path_obj.parent)
        filename = output_path_obj.name
        
        # Convert HTML to PNG
        hti.screenshot(
            url=f"file://{Path(html_file_path).absolute()}",
            save_as=filename,
            size=(1000, 1200)  # width x height (adjust as needed)
        )
        
        print(f"✓ HTML flyer converted to PNG: {output_png_path}")
        return output_png_path
    
    except FileNotFoundError as e:
        raise RuntimeError(f"HTML file not found: {html_file_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to convert HTML to PNG: {str(e)}") from e


def convert_html_to_png_batch(html_files: list, output_dir: str = None) -> list:
    """
    Convert multiple HTML flyers to PNG images in batch.
    
    Args:
        html_files: List of HTML file paths
        output_dir: Directory where PNGs should be saved (default: same as HTML)
    
    Returns:
        List of paths to generated PNG files
    """
    png_paths = []
    for html_file in html_files:
        output_path = None
        if output_dir:
            html_name = Path(html_file).stem
            output_path = str(Path(output_dir) / f"{html_name}_flyer.png")
        
        try:
            png_path = convert_html_to_png(html_file, output_path)
            png_paths.append(png_path)
        except RuntimeError as e:
            print(f"⚠ Error converting {html_file}: {str(e)}")
            continue
    
    return png_paths


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


def enrich_flyer_design_with_tavily(event_details: dict) -> dict:
    """
    Enrich event details with DESIGN PATTERNS from Tavily search results.
    
    Extracts actionable design data: color trends, layout patterns, typography styles,
    and visual hierarchy recommendations from real-world examples.
    
    Args:
        event_details: Dictionary with event info (event_name, description, etc.)
    
    Returns:
        Updated event_details with 'design_system' containing:
        - colors: Extracted brand colors from references
        - layout_type: Recommended layout pattern
        - typography_vibe: Text hierarchy style
        - visual_elements: Key design patterns
        - design_references: Actual URLs analyzed
    """
    # Default design system (fallback)
    default_design_system = {
        "colors": {
            "primary": "#2E7D32",
            "accent": "#FFC107",
            "secondary": "#1976D2",
            "neutral": "#F5F5F5"
        },
        "layout_type": "hero_centered",
        "typography_vibe": "modern_bold",
        "visual_elements": ["gradient_background", "emoji_accents", "card_layout"],
        "design_references": []
    }
    
    if not os.getenv("TAVILY_API_KEY") or TavilyClient is None:
        event_details["design_system"] = default_design_system
        if TavilyClient is None:
            print("⚠️  Tavily library not available. Install with: pip install tavily-python")
        else:
            print("⚠️  TAVILY_API_KEY not set in environment. Using default design system.")
        return event_details
    
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        print(f"✅ Tavily client initialized with API key")
        
        # Build targeted search for actual design references
        event_name = event_details.get("event_name", "event")
        description = event_details.get("description", "")
        event_type = event_details.get("event_type", "youth").lower()
        
        # Search for specific design patterns, not generic inspiration
        search_queries = [
            f"{event_type} event flyer 2025 design trends color palette",
            f"modern promotional poster {event_name.split()[0] if event_name else 'event'} design patterns",
            f"youth engagement event visual hierarchy design examples"
        ]
        
        all_results = []
        design_refs = []
        
        for query in search_queries:
            try:
                response = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=3,
                    include_answer=False
                )
                
                if response.get("results"):
                    all_results.extend(response["results"][:3])
                    for r in response["results"][:3]:
                        if r.get("url"):
                            design_refs.append(r["url"])
            except Exception as query_error:
                print(f"  Query '{query[:50]}...' failed: {str(query_error)[:80]}")
                pass
        
        # EXTRACT DESIGN PATTERNS from results
        design_system = extract_design_patterns(all_results, event_details)
        design_system["design_references"] = design_refs[:5]
        
        event_details["design_system"] = design_system
        print(f"✅ Design system extracted from {len(all_results)} references via Tavily")
        
    except Exception as e:
        print(f"⚠️  Tavily enrichment failed: {str(e)}")
        print(f"     Traceback: {type(e).__name__}")
        event_details["design_system"] = default_design_system
    
    return event_details


def extract_design_patterns(search_results: list, event_details: dict) -> dict:
    """
    Extract actionable design patterns from Tavily search results.
    
    Analyzes URLs, titles, and descriptions to infer:
    - Color trends (hex codes when found)
    - Layout patterns (hero, grid, split, etc.)
    - Typography vibes (bold, minimal, playful, etc.)
    - Visual elements used
    
    Args:
        search_results: List of Tavily search result dicts
        event_details: Event info to contextualize patterns
    
    Returns:
        Design system dict with extracted patterns
    """
    color_palette = extract_colors_from_text(search_results)
    layout_type = infer_layout_pattern(search_results, event_details)
    typography_vibe = infer_typography_style(search_results, event_details)
    visual_elements = extract_visual_elements(search_results)
    
    return {
        "colors": color_palette,
        "layout_type": layout_type,
        "typography_vibe": typography_vibe,
        "visual_elements": visual_elements,
        "design_references": []
    }


def extract_colors_from_text(search_results: list) -> dict:
    """
    Extract color information from search result titles and snippets.
    """
    color_keywords = {
        "vibrant": {"primary": "#FF6B6B", "accent": "#4ECDC4"},
        "modern": {"primary": "#1A1A2E", "accent": "#FFC107"},
        "playful": {"primary": "#FF1493", "accent": "#FFD700"},
        "professional": {"primary": "#1E3A5F", "accent": "#E74C3C"},
        "energetic": {"primary": "#FF6B35", "accent": "#000000"},
        "gradient": {"primary": "#FF6B6B", "accent": "#4ECDC4"},
        "bold": {"primary": "#E74C3C", "accent": "#000000"},
        "minimal": {"primary": "#2C3E50", "accent": "#ECF0F1"},
        "bright": {"primary": "#FFD700", "accent": "#FF6347"},
        "cool": {"primary": "#00B4D8", "accent": "#0077B6"}
    }
    
    # Default color scheme
    palette = {"primary": "#2E7D32", "accent": "#FFC107", "secondary": "#1976D2", "neutral": "#F5F5F5"}
    
    # Scan titles/snippets for color keywords
    combined_text = " ".join([r.get("title", "") + " " + r.get("snippet", "") for r in search_results]).lower()
    
    # Check for keywords and apply matching palette
    for keyword, colors in color_keywords.items():
        if keyword in combined_text:
            palette["primary"] = colors["primary"]
            palette["accent"] = colors["accent"]
            break
    
    return palette


def infer_layout_pattern(search_results: list, event_details: dict) -> str:
    """
    Infer recommended layout pattern from search results.
    
    Options: hero_centered, split_layout, grid_cards, minimal_text, full_bleed, layered
    """
    combined_text = " ".join([r.get("title", "") + " " + r.get("snippet", "") for r in search_results]).lower()
    
    event_type = event_details.get("event_type", "").lower()
    
    # Pattern inference
    if "card" in combined_text or "grid" in combined_text:
        return "grid_cards"
    elif "split" in combined_text or "two-column" in combined_text:
        return "split_layout"
    elif "minimal" in combined_text or "whitespace" in combined_text:
        return "minimal_text"
    elif "layered" in combined_text or "overlay" in combined_text:
        return "layered"
    elif "full" in combined_text or "background" in combined_text:
        return "full_bleed"
    else:
        # Default based on event type
        if event_type in ["sports", "music", "dance"]:
            return "full_bleed"
        elif event_type in ["workshop", "seminar", "training"]:
            return "split_layout"
        else:
            return "hero_centered"


def infer_typography_style(search_results: list, event_details: dict) -> str:
    """
    Infer typography style from search results.
    
    Options: modern_bold, minimal_elegant, playful_fun, energetic_action, business_formal
    """
    combined_text = " ".join([r.get("title", "") + " " + r.get("snippet", "") for r in search_results]).lower()
    
    if "playful" in combined_text or "fun" in combined_text or "quirky" in combined_text:
        return "playful_fun"
    elif "minimal" in combined_text or "elegant" in combined_text or "sophisticated" in combined_text:
        return "minimal_elegant"
    elif "bold" in combined_text or "energetic" in combined_text or "action" in combined_text:
        return "energetic_action"
    elif "formal" in combined_text or "professional" in combined_text or "corporate" in combined_text:
        return "business_formal"
    else:
        return "modern_bold"


def extract_visual_elements(search_results: list) -> list:
    """
    Extract key visual elements from search results.
    """
    combined_text = " ".join([r.get("title", "") + " " + r.get("snippet", "") for r in search_results]).lower()
    
    elements = []
    element_keywords = {
        "gradient": "gradient_background",
        "emoji": "emoji_accents",
        "icon": "icon_library",
        "card": "card_layout",
        "shadow": "drop_shadows",
        "border": "bold_borders",
        "circle": "circular_elements",
        "line": "geometric_lines",
        "image": "hero_image",
        "video": "video_background"
    }
    
    for keyword, element in element_keywords.items():
        if keyword in combined_text:
            elements.append(element)
    
    # Ensure we have at least some elements
    if not elements:
        elements = ["gradient_background", "emoji_accents", "card_layout"]
    
    return elements[:5]  # Limit to 5 elements

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


# --- HTML/CSS FLYER GENERATION ---
# Modern approach: Generate responsive HTML/CSS flyers instead of static PNGs
# Much better visuals, easier to customize, and can be embedded anywhere

def generate_html_flyer(flyer_data: Dict, qr_path: str, design_system: dict = None, output_path: str = "flyer.html") -> str:
    """
    Generate a professional HTML/CSS flyer with reference-informed design.
    
    Args:
        flyer_data: Dictionary with flyer content
        qr_path: Path to QR code image
        design_system: Design patterns from Tavily (colors, layout, typography)
        output_path: Where to save the HTML file
    
    Returns:
        Path to generated HTML file
    """
    if design_system is None:
        design_system = {
            "colors": {"primary": "#2E7D32", "accent": "#FFC107", "secondary": "#1976D2", "neutral": "#F5F5F5"},
            "layout_type": "hero_centered",
            "typography_vibe": "modern_bold",
            "visual_elements": ["gradient_background", "emoji_accents", "card_layout"]
        }
    
    # Load QR code as base64 for embedding
    qr_base64 = ""
    try:
        with open(qr_path, "rb") as img_file:
            qr_base64 = base64.b64encode(img_file.read()).decode()
    except:
        qr_base64 = ""
    
    colors = design_system.get("colors", {})
    layout = design_system.get("layout_type", "hero_centered")
    typo_vibe = design_system.get("typography_vibe", "modern_bold")
    
    # Select layout renderer based on design_system
    if layout == "split_layout":
        html = render_split_layout_flyer(flyer_data, colors, qr_base64)
    elif layout == "grid_cards":
        html = render_grid_cards_flyer(flyer_data, colors, qr_base64)
    elif layout == "minimal_text":
        html = render_minimal_flyer(flyer_data, colors, qr_base64)
    elif layout == "full_bleed":
        html = render_fullbleed_flyer(flyer_data, colors, qr_base64, typo_vibe)
    else:  # hero_centered (default)
        html = render_hero_centered_flyer(flyer_data, colors, qr_base64, typo_vibe)
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML FLYER SAVED to: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


def render_hero_centered_flyer(flyer_data: Dict, colors: dict, qr_base64: str, typo_vibe: str = "modern_bold") -> str:
    """
    Render hero-centered layout with prominent headline and stacked content.
    """
    primary = colors.get("primary", "#2E7D32")
    accent = colors.get("accent", "#FFC107")
    neutral = colors.get("neutral", "#F5F5F5")
    secondary = colors.get("secondary", "#1976D2")
    
    font_weight = "900" if "bold" in typo_vibe else "600"
    letter_spacing = "2px" if "bold" in typo_vibe else "0px"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{flyer_data.get('headline', 'Event Flyer')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, {primary}15 0%, {accent}15 100%);
            padding: 20px;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .flyer {{
            width: 100%;
            max-width: 900px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            overflow: hidden;
        }}
        
        .hero {{
            background: linear-gradient(135deg, {primary} 0%, {primary}dd 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 400px;
            height: 400px;
            background: {accent}20;
            border-radius: 50%;
        }}
        
        .hero-content {{
            position: relative;
            z-index: 1;
        }}
        
        .headline {{
            font-size: 3.5em;
            font-weight: {font_weight};
            letter-spacing: {letter_spacing};
            margin-bottom: 15px;
            line-height: 1.2;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .subheadline {{
            font-size: 1.4em;
            font-weight: 300;
            color: {accent};
            margin-bottom: 30px;
        }}
        
        .accent-line {{
            height: 4px;
            width: 100px;
            background: {accent};
            margin: 20px auto 30px;
            border-radius: 2px;
        }}
        
        .content {{
            padding: 50px 40px;
        }}
        
        .details-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-bottom: 40px;
        }}
        
        .detail-card {{
            background: {neutral};
            padding: 25px;
            border-radius: 12px;
            border-left: 4px solid {accent};
        }}
        
        .detail-label {{
            font-size: 0.9em;
            color: {primary};
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }}
        
        .detail-value {{
            font-size: 1.2em;
            color: #333;
            font-weight: 600;
        }}
        
        .description {{
            background: {secondary}10;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 40px;
            line-height: 1.6;
            font-size: 1.05em;
            color: #555;
        }}
        
        .cta-section {{
            display: flex;
            gap: 20px;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}
        
        .cta-button {{
            flex: 1;
            min-width: 250px;
            padding: 20px 30px;
            background: linear-gradient(135deg, {accent} 0%, {primary} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.3em;
            font-weight: 700;
            cursor: pointer;
            text-align: center;
            box-shadow: 0 10px 30px {accent}40;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }}
        
        .cta-button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 15px 40px {accent}60;
        }}
        
        .qr-section {{
            flex: 0 0 180px;
            text-align: center;
        }}
        
        .qr-code {{
            width: 150px;
            height: 150px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .qr-label {{
            font-size: 0.8em;
            color: #999;
            margin-top: 8px;
        }}
        
        @media (max-width: 768px) {{
            .headline {{ font-size: 2.5em; }}
            .subheadline {{ font-size: 1.1em; }}
            .details-grid {{ grid-template-columns: 1fr; }}
            .cta-section {{ flex-direction: column; }}
            .cta-button {{ min-width: 100%; }}
            .qr-section {{ margin-top: 20px; }}
            .content {{ padding: 30px 20px; }}
        }}
    </style>
</head>
<body>
    <div class="flyer">
        <div class="hero">
            <div class="hero-content">
                <h1 class="headline">{flyer_data.get('headline', 'Event')}</h1>
                <p class="subheadline">{flyer_data.get('subheadline', '')}</p>
                <div class="accent-line"></div>
            </div>
        </div>
        
        <div class="content">
            <div class="details-grid">
                <div class="detail-card">
                    <div class="detail-label">📅 When</div>
                    <div class="detail-value">{flyer_data.get('date_time_line', 'TBA')}</div>
                </div>
                <div class="detail-card">
                    <div class="detail-label">📍 Where</div>
                    <div class="detail-value">{flyer_data.get('location_line', 'TBA')}</div>
                </div>
            </div>
            
            <div class="description">
                {flyer_data.get('body_blurb', 'Join us for an amazing event!')}
            </div>
            
            <div class="cta-section">
                <a class="cta-button" href="#register">{flyer_data.get('call_to_action', 'Register Now')}</a>
                <div class="qr-section">
                    {'<img src="data:image/png;base64,' + qr_base64 + '" class="qr-code" alt="QR Code">' if qr_base64 else ''}
                    <p class="qr-label">Scan to register</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html


def render_split_layout_flyer(flyer_data: Dict, colors: dict, qr_base64: str) -> str:
    """
    Render split-layout: text on left, visual/QR on right.
    """
    primary = colors.get("primary", "#2E7D32")
    accent = colors.get("accent", "#FFC107")
    neutral = colors.get("neutral", "#F5F5F5")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{flyer_data.get('headline', 'Event Flyer')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, {primary}20 0%, {accent}20 100%);
            padding: 20px;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .flyer {{
            width: 100%;
            max-width: 1000px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 30px 80px rgba(0,0,0,0.12);
            overflow: hidden;
            display: grid;
            grid-template-columns: 1fr 1fr;
            min-height: 600px;
        }}
        
        .left-section {{
            background: linear-gradient(135deg, {primary}f5 0%, {primary}dd 100%);
            color: white;
            padding: 50px 40px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        
        .headline {{
            font-size: 2.8em;
            font-weight: 900;
            margin-bottom: 15px;
            line-height: 1.2;
        }}
        
        .subheadline {{
            font-size: 1.3em;
            font-weight: 300;
            margin-bottom: 30px;
            color: {accent};
        }}
        
        .detail {{
            margin-bottom: 20px;
            font-size: 1.05em;
            line-height: 1.5;
        }}
        
        .detail-icon {{
            font-size: 1.3em;
            margin-right: 10px;
        }}
        
        .right-section {{
            background: {neutral};
            padding: 50px 40px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .description {{
            font-size: 1.1em;
            line-height: 1.7;
            color: #555;
            margin-bottom: 30px;
        }}
        
        .cta-button {{
            padding: 18px 30px;
            background: linear-gradient(135deg, {accent} 0%, {primary} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 700;
            cursor: pointer;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 10px 30px {accent}40;
        }}
        
        .cta-button:hover {{
            transform: translateY(-2px);
        }}
        
        .qr-container {{
            text-align: center;
        }}
        
        .qr-code {{
            width: 140px;
            height: 140px;
            margin: 0 auto 10px;
            background: white;
            padding: 8px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .qr-label {{
            font-size: 0.85em;
            color: #999;
        }}
        
        @media (max-width: 768px) {{
            .flyer {{ grid-template-columns: 1fr; }}
            .headline {{ font-size: 2em; }}
            .left-section, .right-section {{ padding: 30px; }}
        }}
    </style>
</head>
<body>
    <div class="flyer">
        <div class="left-section">
            <h1 class="headline">{flyer_data.get('headline', 'Event')}</h1>
            <p class="subheadline">{flyer_data.get('subheadline', '')}</p>
            <div class="detail">
                <span class="detail-icon">📅</span>
                <span>{flyer_data.get('date_time_line', 'TBA')}</span>
            </div>
            <div class="detail">
                <span class="detail-icon">📍</span>
                <span>{flyer_data.get('location_line', 'TBA')}</span>
            </div>
        </div>
        
        <div class="right-section">
            <div class="description">
                {flyer_data.get('body_blurb', 'Join us for an amazing event!')}
            </div>
            
            <button class="cta-button">{flyer_data.get('call_to_action', 'Register Now')}</button>
            
            <div class="qr-container">
                {'<img src="data:image/png;base64,' + qr_base64 + '" class="qr-code" alt="QR Code">' if qr_base64 else ''}
                <p class="qr-label">Scan to join</p>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html


def render_grid_cards_flyer(flyer_data: Dict, colors: dict, qr_base64: str) -> str:
    """
    Render grid-card layout with event details in distinct cards.
    """
    primary = colors.get("primary", "#2E7D32")
    accent = colors.get("accent", "#FFC107")
    neutral = colors.get("neutral", "#F5F5F5")
    secondary = colors.get("secondary", "#1976D2")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{flyer_data.get('headline', 'Event Flyer')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, {primary}15 0%, {secondary}15 100%);
            padding: 30px 20px;
            min-height: 100vh;
        }}
        
        .flyer {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 50px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.12);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 50px;
        }}
        
        .headline {{
            font-size: 3em;
            font-weight: 900;
            color: {primary};
            margin-bottom: 10px;
        }}
        
        .subheadline {{
            font-size: 1.4em;
            color: {accent};
            font-weight: 300;
        }}
        
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .card {{
            background: linear-gradient(135deg, {primary}08 0%, {accent}08 100%);
            border: 2px solid {primary};
            border-radius: 12px;
            padding: 25px;
            text-align: center;
        }}
        
        .card-icon {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .card-label {{
            font-size: 0.85em;
            color: {primary};
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        
        .card-value {{
            font-size: 1.15em;
            color: #333;
            font-weight: 600;
        }}
        
        .description {{
            background: {neutral};
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 40px;
            line-height: 1.7;
            font-size: 1.05em;
            color: #555;
        }}
        
        .footer {{
            display: flex;
            gap: 20px;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .cta-button {{
            flex: 1;
            min-width: 200px;
            padding: 18px 30px;
            background: linear-gradient(135deg, {accent} 0%, {primary} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 10px 30px {accent}40;
        }}
        
        .cta-button:hover {{
            transform: translateY(-2px);
        }}
        
        .qr-container {{
            text-align: center;
        }}
        
        .qr-code {{
            width: 130px;
            height: 130px;
            background: white;
            padding: 8px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .qr-label {{
            font-size: 0.8em;
            color: #999;
            margin-top: 8px;
        }}
        
        @media (max-width: 768px) {{
            .flyer {{ padding: 30px; }}
            .headline {{ font-size: 2em; }}
            .cards-grid {{ grid-template-columns: 1fr; }}
            .footer {{ flex-direction: column; }}
            .cta-button {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="flyer">
        <div class="header">
            <h1 class="headline">{flyer_data.get('headline', 'Event')}</h1>
            <p class="subheadline">{flyer_data.get('subheadline', '')}</p>
        </div>
        
        <div class="cards-grid">
            <div class="card">
                <div class="card-icon">📅</div>
                <div class="card-label">Date & Time</div>
                <div class="card-value">{flyer_data.get('date_time_line', 'TBA')}</div>
            </div>
            <div class="card">
                <div class="card-icon">📍</div>
                <div class="card-label">Location</div>
                <div class="card-value">{flyer_data.get('location_line', 'TBA')}</div>
            </div>
            <div class="card">
                <div class="card-icon">🚀</div>
                <div class="card-label">Action</div>
                <div class="card-value">Register Now</div>
            </div>
        </div>
        
        <div class="description">
            {flyer_data.get('body_blurb', 'Join us for an amazing event!')}
        </div>
        
        <div class="footer">
            <button class="cta-button">{flyer_data.get('call_to_action', 'Register Now')}</button>
            <div class="qr-container">
                {'<img src="data:image/png;base64,' + qr_base64 + '" class="qr-code" alt="QR Code">' if qr_base64 else ''}
                <p class="qr-label">Scan to join</p>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html


def render_minimal_flyer(flyer_data: Dict, colors: dict, qr_base64: str) -> str:
    """
    Render ultra-minimalist design with lots of whitespace.
    """
    primary = colors.get("primary", "#2E7D32")
    accent = colors.get("accent", "#FFC107")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{flyer_data.get('headline', 'Event Flyer')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background: white;
            padding: 60px 40px;
            text-align: center;
        }}
        
        .headline {{
            font-size: 4em;
            font-weight: 300;
            margin-bottom: 20px;
            color: {primary};
            letter-spacing: -2px;
        }}
        
        .divider {{
            width: 100px;
            height: 3px;
            background: {accent};
            margin: 30px auto 40px;
        }}
        
        .content {{
            max-width: 600px;
            margin: 0 auto;
        }}
        
        .detail {{
            margin: 40px 0;
            font-size: 1.1em;
            color: #777;
        }}
        
        .description {{
            margin: 40px 0 60px;
            font-size: 1.05em;
            line-height: 1.8;
            color: #555;
        }}
        
        .cta-button {{
            padding: 16px 40px;
            background: {primary};
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            margin-bottom: 40px;
            text-decoration: none;
            display: inline-block;
        }}
        
        .cta-button:hover {{
            background: {accent};
        }}
        
        .qr-section {{
            margin-top: 60px;
            padding-top: 40px;
            border-top: 1px solid #eee;
        }}
        
        .qr-code {{
            width: 150px;
            height: 150px;
            margin: 0 auto;
            background: white;
            padding: 8px;
            border-radius: 4px;
        }}
        
        .qr-label {{
            font-size: 0.9em;
            color: #999;
            margin-top: 15px;
        }}
    </style>
</head>
<body>
    <h1 class="headline">{flyer_data.get('headline', 'Event')}</h1>
    <div class="divider"></div>
    
    <div class="content">
        <div class="detail">📅 {flyer_data.get('date_time_line', 'TBA')}</div>
        <div class="detail">📍 {flyer_data.get('location_line', 'TBA')}</div>
        
        <p class="description">
            {flyer_data.get('body_blurb', 'Join us for an amazing event!')}
        </p>
        
        <a class="cta-button" href="#">{flyer_data.get('call_to_action', 'Register Now')}</a>
        
        <div class="qr-section">
            {'<img src="data:image/png;base64,' + qr_base64 + '" class="qr-code" alt="QR Code">' if qr_base64 else ''}
            <p class="qr-label">Scan to register</p>
        </div>
    </div>
</body>
</html>"""
    return html


def render_fullbleed_flyer(flyer_data: Dict, colors: dict, qr_base64: str, typo_vibe: str = "modern_bold") -> str:
    """
    Full-bleed background with text overlay - perfect for sporty/energetic events.
    """
    primary = colors.get("primary", "#FF6B35")
    accent = colors.get("accent", "#000000")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{flyer_data.get('headline', 'Event Flyer')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Impact', 'Arial Black', sans-serif;
            background: linear-gradient(135deg, {primary} 0%, {primary}dd 100%);
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        
        .flyer {{
            width: 100%;
            max-width: 1000px;
            height: 600px;
            background: linear-gradient(135deg, {primary} 0%, {primary}dd 100%);
            border-radius: 20px;
            padding: 50px;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
            box-shadow: 0 30px 80px rgba(0,0,0,0.3);
        }}
        
        .flyer::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 600px;
            height: 600px;
            background: {accent}15;
            border-radius: 50%;
            z-index: 1;
        }}
        
        .content {{
            position: relative;
            z-index: 2;
        }}
        
        .headline {{
            font-size: 4.5em;
            font-weight: 900;
            line-height: 1.1;
            margin-bottom: 20px;
            text-shadow: 0 3px 10px rgba(0,0,0,0.3);
            letter-spacing: 2px;
        }}
        
        .subheadline {{
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 30px;
            color: {accent};
            text-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .details {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
            font-size: 1.3em;
        }}
        
        .description {{
            font-size: 1.1em;
            line-height: 1.6;
            margin-bottom: 30px;
            max-width: 600px;
        }}
        
        .footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 30px;
        }}
        
        .cta-button {{
            background: {accent};
            color: {primary};
            border: none;
            padding: 20px 40px;
            border-radius: 10px;
            font-size: 1.5em;
            font-weight: 900;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: all 0.3s;
        }}
        
        .cta-button:hover {{
            transform: translateY(-3px);
        }}
        
        .qr-box {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .qr-code {{
            width: 120px;
            height: 120px;
        }}
        
        .qr-label {{
            font-size: 0.8em;
            color: {primary};
            margin-top: 8px;
            font-weight: 700;
        }}
        
        @media (max-width: 768px) {{
            .flyer {{ height: auto; padding: 30px; }}
            .headline {{ font-size: 2.5em; }}
            .subheadline {{ font-size: 1.3em; }}
            .details {{ grid-template-columns: 1fr; }}
            .footer {{ flex-direction: column; }}
            .cta-button {{ width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="flyer">
        <div class="content">
            <h1 class="headline">{flyer_data.get('headline', 'Event')}</h1>
            <p class="subheadline">{flyer_data.get('subheadline', '')}</p>
            
            <div class="details">
                <div>📅 {flyer_data.get('date_time_line', 'TBA')}</div>
                <div>📍 {flyer_data.get('location_line', 'TBA')}</div>
            </div>
            
            <p class="description">{flyer_data.get('body_blurb', 'Join us for an amazing event!')}</p>
        </div>
        
        <div class="footer">
            <button class="cta-button">{flyer_data.get('call_to_action', 'Register Now')}</button>
            <div class="qr-box">
                {'<img src="data:image/png;base64,' + qr_base64 + '" class="qr-code" alt="QR Code">' if qr_base64 else ''}
                <p class="qr-label">SCAN</p>
            </div>
        </div>
    </div>
</body>
</html>"""
    return html


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
    # Use aggressive truncation for email generation (only 1500 tokens for messages)
    truncated_messages = truncate_messages(state["messages"], max_tokens=1500)
    print(f"✉️ Generating proposal email for: {event_details.get('event_name', 'Unnamed Event')}")
    
    email_task_prompt = f"""Write a SHORT proposal email to church leadership.
Start: "Good evening Brothers,"
Brief (200 words max): Event name, date, location, cost per person ${event_details.get('cost_per_person', '0')}, church subsidy ${event_details.get('church_subsidy', '0')}, request approval.
End: "Blessings, [Your name]"
Tone: Warm, conversational.

Format:
--- EMAIL BODY START ---
Good evening Brothers,
[2-3 sentences]
[Details & costs]
[Closing request]
Blessings,
[Your name]
--- EMAIL BODY END ---

Call save_proposal_email with the email body and event_name "{event_details.get('event_name', 'Unnamed Event')}"."""
    
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
    # Use very aggressive truncation for flyer generation (only 1500 tokens for messages)
    truncated_messages = truncate_messages(state["messages"], max_tokens=1500)
    
    # Enrich event details with Tavily design inspiration
    event_details = enrich_flyer_design_with_tavily(event_details)
    
    print(f"🎯 Creating flyer for: {event_details.get('event_name', 'Unnamed Event')}")
    if qr_path:
        print(f"📱 Using existing QR code: {qr_path}")
    else:
        print(f"📱 Will generate new QR linking to: {form_url}")
    
    # STEP 1: Designer agent generates multiple creative variations
    print_banner("👨‍🎨 FLYER DESIGNER: Generating Creative Concepts", "🎨")
    
    # Create a compact event summary - no raw history at all
    event_summary = {
        "eventname": event_details.get("event_name", "Unnamed Event"),
        "date": event_details.get("date", "TBD"),
        "time": event_details.get("time", "TBA"),
        "location": event_details.get("location", "TBD"),
        "description": event_details.get("description", "")[:500],
        "design_notes": str(event_details.get("design_inspiration_notes", ""))[:500],
    }
    
    designer_prompt = f"""You are a creative flyer designer. Generate 5 design variations for the event provided.
Call the design_flyer_variations tool with your recommended design."""
    
    designer_agent = create_agent(llm, tools=[design_flyer_variations], system_prompt=designer_prompt)
    designer_response = await designer_agent.ainvoke({
        "messages": [HumanMessage(content=json.dumps(event_summary))]
    })
    
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
    
    editor_prompt = f"""You are a flyer editor. Select the best design variation.
Event: {event_details.get('event_name', 'Unnamed')} | Audience: {event_details.get('target_audience', 'Youth')}
Call select_and_render_flyer with best design. QR URL: {form_url}"""
    
    editor_agent = create_agent(llm, tools=[select_and_render_flyer], system_prompt=editor_prompt)
    # Use aggressive truncation for editor too (only 1000 tokens for messages)
    truncated_editor_messages = truncate_messages(designer_response["messages"], max_tokens=1000)
    editor_response = await editor_agent.ainvoke({"messages": truncated_editor_messages})
    
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
# # This tool processes and stores signed waiver information from registrations
# @tool
# def process_waiver_submission(participant_data: dict, event_name: str = "Youth Event") -> dict:
#     """
#     Process and store a signed waiver from a participant.
    
#     This tool takes registration data (including waiver signature) and stores it securely.
    
#     Args:
#         participant_data: Dictionary with participant info including:
#             - youth_first_last_name: Youth's full name
#             - youth_age: Youth's age
#             - parent_first_last_name: Parent/Guardian name
#             - parent_phone: Parent/Guardian phone
#             - waiver_signature: Name as signed on waiver
#             - waiver_date: Date waiver was signed
#             - waiver_acknowledgment: Boolean - waiver was acknowledged
#         event_name: Name of the event
    
#     Returns:
#         Dictionary with waiver storage details and confirmation
#     """
#     result = save_signed_waiver(participant_data, event_name)
#     return result


# --- LANGGRAPH TOOL: Flyer Design Variations ---
# This tool generates multiple creative flyer design concepts for the editor to choose from
# Remove all old PNG functions - they were replaced by HTML generators above
# save_flyer_png_bold_vibrant, save_flyer_png_professional_business, etc. no longer needed

@tool
def design_flyer_variations(event_details: dict, event_name: str = "Youth Event") -> dict:
    """
    Generate multiple creative flyer design variations INFORMED BY TAVILY DESIGN SYSTEM.
    
    Uses extracted design_system (colors, layouts, typography from Tavily) to create
    5 distinct design concepts that are reference-informed, not template-based.
    
    Args:
        event_details: Dictionary with event info including 'design_system' from Tavily
        event_name: Name of the event (default: "Youth Event")
    
    Returns:
        Dictionary with:
        - "variations": List of 5 reference-informed design concepts
        - "descriptions": Explanations for each design's pattern and inspiration
        - "design_system_used": The extracted design system that informed the variations
    """
    
    # Get design system from Tavily enrichment (or fall back to defaults)
    design_system = event_details.get("design_system", {
        "colors": {"primary": "#2E7D32", "accent": "#FFC107", "secondary": "#1976D2", "neutral": "#F5F5F5"},
        "layout_type": "hero_centered",
        "typography_vibe": "modern_bold",
        "visual_elements": ["gradient_background", "emoji_accents", "card_layout"]
    })
    
    colors = design_system.get("colors", {})
    primary_color = colors.get("primary", "#2E7D32")
    accent_color = colors.get("accent", "#FFC107")
    secondary_color = colors.get("secondary", "#1976D2")
    layout_type = design_system.get("layout_type", "hero_centered")
    visual_elements = design_system.get("visual_elements", [])
    
    # Format date
    formatted_date = format_event_date(event_details.get('event_date', 'Coming Soon'))
    date_time_line = formatted_date + " • " + event_details.get('event_time', 'TBA')
    
    # Create 5 variations based on REFERENCE-INFORMED LAYOUTS
    design_variations = [
        {
            "name": "Hero Centered (Reference-Informed)",
            "vibe": f"Professional hero layout using extracted color system: {primary_color} + {accent_color}",
            "design_style": "hero_centered",
            "layout_type": "hero_centered",
            "headline": event_details.get('event_name', event_name),
            "subheadline": "Get ready for an amazing experience!",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": event_details.get('description', f'Join us for this incredible event featuring fun, friends, and unforgettable memories.'),
            "call_to_action": "Register Now!",
            "color_scheme": {"primary": primary_color, "accent": accent_color, "secondary": secondary_color, "neutral": colors.get("neutral", "#F5F5F5")}
        },
        {
            "name": "Split Layout (Design Reference)",
            "vibe": f"Two-column design from references: visual left ({primary_color}), content right",
            "design_style": "split_layout",
            "layout_type": "split_layout",
            "headline": event_details.get('event_name', event_name),
            "subheadline": "Be part of something special",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": event_details.get('description', 'Discover an unforgettable experience'),
            "call_to_action": "Join Us!",
            "color_scheme": {"primary": primary_color, "accent": accent_color, "secondary": secondary_color, "neutral": colors.get("neutral", "#F5F5F5")}
        },
        {
            "name": "Grid Cards Layout (Trending Pattern)",
            "vibe": f"Modern grid-card system from current design trends, colors: {accent_color} accents",
            "design_style": "grid_cards",
            "layout_type": "grid_cards",
            "headline": event_details.get('event_name', event_name),
            "subheadline": "Everything you need to know",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": event_details.get('description', 'Connect, engage, and grow with us'),
            "call_to_action": "Register Today",
            "color_scheme": {"primary": primary_color, "accent": accent_color, "secondary": secondary_color, "neutral": colors.get("neutral", "#F5F5F5")}
        },
        {
            "name": "Minimal Elegant (Sophisticated Reference)",
            "vibe": f"Ultra-clean design with typography focus and {primary_color} accents",
            "design_style": "minimal_elegant",
            "layout_type": "minimal_text",
            "headline": event_details.get('event_name', event_name),
            "subheadline": "attend",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": event_details.get('description', 'A refined experience awaits'),
            "call_to_action": "Learn More",
            "color_scheme": {"primary": primary_color, "accent": accent_color, "secondary": secondary_color, "neutral": colors.get("neutral", "#F5F5F5")}
        },
        {
            "name": "Full-Bleed Energetic (Dynamic Reference)",
            "vibe": f"High-impact full-background using vibrant research colors: {primary_color}",
            "design_style": "full_bleed",
            "layout_type": "full_bleed",
            "headline": event_details.get('event_name', event_name).upper(),
            "subheadline": "Don't Miss Out!",
            "date_time_line": date_time_line,
            "location_line": event_details.get('location', 'Location TBA'),
            "body_blurb": event_details.get('description', 'Join thousands for an incredible experience'),
            "call_to_action": "Claim Your Spot",
            "color_scheme": {"primary": primary_color, "accent": accent_color, "secondary": secondary_color, "neutral": colors.get("neutral", "#F5F5F5")}
        }
    ]
    
    # Descriptions explaining the design references
    descriptions = [
        f"Design 1: Hero-centered layout from modern promotional trends. Primary color {primary_color} establishes brand presence.",
        f"Design 2: Split-column pattern from successful event promotions. Combines visual impact with content clarity.",
        f"Design 3: Grid-card system trending in 2024-2025 youth engagement. Organized info architecture with {accent_color} accents.",
        f"Design 4: Minimalist approach inspired by high-end event design references. Whitespace + typography.",
        f"Design 5: Full-bleed dynamic layout ideal for energetic events. Extracted palette creates strong visual impact."
    ]
    
    ref_notes = f"\nDesign patterns extracted from {len(design_system.get('design_references', []))} reference sources via Tavily"
    
    return {
        "variations": design_variations,
        "descriptions": descriptions,
        "design_system_used": design_system,
        "note": f"Variations use reference-informed colors, layouts, and typography patterns.{ref_notes}"
    }


# --- LANGGRAPH TOOL: Select and Render Flyer ---
# This tool takes a selected design variation and renders the final flyer
@tool
def select_and_render_flyer(selected_design: dict, form_url: str = "https://forms.example.com/register") -> dict:
    """
    Select a flyer design variation and render as PNG.
    
    This tool takes a reference-informed design concept and generates:
    - A QR code PNG linking to the form_url
    - A PNG flyer image based on the selected layout_type
    
    The PNG is ready for email, print, social media, and web use.
    
    Args:
        selected_design: The chosen design variation dict with layout_type, colors, etc.
        form_url: URL to embed in the QR code for registration
    
    Returns:
        Dictionary with:
        - "flyer_path": Absolute path to generated flyer PNG
        - "qr_path": Absolute path to generated QR PNG
        - "design_used": Name of the design style used
        - "layout_type": Which layout pattern was used
    """
    
    print_banner("🎨 RENDERING FINAL FLYER (PNG)", "✨")
    
    # Validate design has required fields
    required = ['headline', 'subheadline', 'date_time_line', 'location_line', 'body_blurb', 'call_to_action', 'color_scheme']
    missing = [f for f in required if f not in selected_design]
    if missing:
        raise ValueError(f"Selected design missing fields: {missing}")
    
    # Generate QR code
    qr_path = create_qr_png(form_url)
    
    # Extract layout type from design (or use default)
    layout_type = selected_design.get('layout_type', 'hero_centered')
    design_system = {
        "colors": selected_design.get('color_scheme', {}),
        "layout_type": layout_type,
        "typography_vibe": selected_design.get('design_style', 'modern_bold'),
        "visual_elements": ["gradient_background", "emoji_accents", "card_layout"]
    }
    
    # Step 1: Generate HTML flyer as intermediate step
    html_path = generate_html_flyer(selected_design, qr_path, design_system)
    
    # Step 2: Convert HTML to PNG
    try:
        png_path = convert_html_to_png(html_path)
        print(f"✅ PNG flyer generated: {png_path}")
    except RuntimeError as e:
        print(f"⚠️ PNG conversion failed: {str(e)}")
        print(f"Falling back to HTML: {html_path}")
        png_path = html_path  # Fallback to HTML if PNG fails
    
    design_name = selected_design.get('name', 'Custom')
    
    return {
        "flyer_path": png_path,
        "qr_path": qr_path,
        "design_used": design_name,
        "layout_type": layout_type
    }


# --- LANGGRAPH TOOL: Convert HTML Flyer to PNG ---
# This tool converts an existing HTML flyer to PNG format for email/print/social media
@tool
def convert_flyer_html_to_png(html_flyer_path: str, output_png_path: str = None) -> dict:
    """
    Convert an HTML flyer to PNG image format.
    
    This tool takes a generated HTML flyer and converts it to a PNG image
    for use in email attachments, print materials, or social media.
    
    Args:
        html_flyer_path: Absolute path to the HTML flyer file
        output_png_path: Optional custom path for PNG output (auto-generated if not provided)
    
    Returns:
        Dictionary with:
        - "png_path": Absolute path to generated PNG file
        - "html_path": Original HTML path
        - "format": "PNG (from HTML)"
        - "status": "success" or error message
    """
    try:
        png_path = convert_html_to_png(html_flyer_path, output_png_path)
        return {
            "png_path": png_path,
            "html_path": html_flyer_path,
            "format": "PNG (from HTML)",
            "status": "success"
        }
    except RuntimeError as e:
        return {
            "png_path": None,
            "html_path": html_flyer_path,
            "format": "PNG (from HTML)",
            "status": f"error: {str(e)}"
        }


# --- LANGGRAPH TOOL: Generate Flyer with Both HTML and PNG ---
# This tool generates a flyer and optionally converts it to PNG
@tool
def generate_flyer_with_png_option(flyer_data: dict, form_url: str = "https://forms.example.com/register", include_png: bool = False) -> dict:
    """
    Generate a flyer with optional PNG conversion.
    
    This tool generates an HTML flyer and optionally converts it to PNG format
    for maximum flexibility in distribution channels.
    
    Args:
        flyer_data: Dictionary with flyer content (headline, subheadline, etc.)
        form_url: URL to embed in QR code
        include_png: If True, also generate PNG version of the flyer (bool, default: False)
    
    Returns:
        Dictionary with:
        - "html_path": Path to HTML flyer
        - "png_path": Path to PNG flyer (None if include_png=False)
        - "qr_path": Path to QR code PNG
        - "formats": List of generated formats ["HTML", "PNG", "QR"]
    """
    qr_path = create_qr_png(form_url)
    colors = flyer_data.get("color_scheme", {"primary": "#2E7D32", "accent": "#FFC107"})
    design_system = {"colors": colors, "layout_type": "hero_centered", "typography_vibe": "modern_bold"}
    html_path = generate_html_flyer(flyer_data, qr_path, design_system)
    
    formats = ["HTML", "QR"]
    png_path = None
    
    if include_png:
        try:
            png_path = convert_html_to_png(html_path)
            formats.append("PNG")
            print(f"✅ PNG version also generated: {png_path}")
        except RuntimeError as e:
            print(f"⚠️ PNG generation skipped: {str(e)}")
    
    return {
        "html_path": html_path,
        "png_path": png_path,
        "qr_path": qr_path,
        "formats": formats
    }


# --- LANGGRAPH TOOL: Flyer Package Generation ---
# This tool is exposed to the writer agent and can be called during the writing phase.
# It encapsulates the logic for generating QR code and PNG flyer.
@tool
def generate_flyer_package(flyer_data: dict, form_url: str = "https://forms.example.com/register") -> dict:
    """
    Generate a complete flyer package: PNG flyer + QR code.
    
    This tool takes flyer content data and generates:
    - A QR code PNG linking to the provided form_url
    - A professional PNG flyer image with reference-informed design
    
    The PNG flyer is ready for email, web, print, and social media.
    
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
    # Extract color_scheme from flyer_data if available
    colors = flyer_data.get("color_scheme", {"primary": "#2E7D32", "accent": "#FFC107"})
    design_system = {"colors": colors, "layout_type": "hero_centered", "typography_vibe": "modern_bold"}
    
    # Step 1: Generate HTML as intermediate
    html_path = generate_html_flyer(flyer_data, qr_path, design_system)
    
    # Step 2: Convert to PNG
    try:
        flyer_path = convert_html_to_png(html_path)
        print(f"✅ PNG flyer generated: {flyer_path}")
    except RuntimeError as e:
        print(f"⚠️ PNG conversion failed: {str(e)}")
        print(f"Returning HTML instead: {html_path}")
        flyer_path = html_path  # Fallback to HTML if PNG fails
    
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
            event_details = payload.get("event_details", payload)
            
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

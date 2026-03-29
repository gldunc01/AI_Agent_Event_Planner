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


# --- TOOL CANDIDATE ---
# This function is a good candidate to expose as a tool:
# - Input: flyer_data (dict with required fields), qr_path (path to QR PNG)
# - Output: flyer PNG file saved to disk
# - Used for: 'flyer' and 'basketball_clinic' tasks
def save_flyer_png(flyer_data: Dict, qr_path: str, output_path: str = "flyer.png"):
    """Generate simple flyer PNG from JSON + QR."""
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

    # Convert hex colors to RGB tuples
    primary_color = hex_to_rgb(flyer_data['color_scheme']['primary'])
    accent_color = hex_to_rgb(flyer_data['color_scheme']['accent'])

    # Canvas: 1100x800 landscape
    img = Image.new('RGB', (1100, 800), color=primary_color)
    draw = ImageDraw.Draw(img)

    # Try to load font, fallback
    try:
        font_large = ImageFont.truetype("arial.ttf", 54)
        font_med = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 26)
    except:
        font_large = ImageFont.load_default()
        font_med = font_small = font_large

    # Layout variables
    left_margin = 60
    top_margin = 60
    col_width = 650
    y = top_margin

    # Headline
    draw.text((left_margin, y), flyer_data['headline'], fill='white', font=font_large)
    y += 90

    # Subheadline
    draw.text((left_margin, y), flyer_data['subheadline'], fill=accent_color, font=font_med)
    y += 60

    # Details (white text on primary bg)
    details = [
        flyer_data['date_time_line'],
        flyer_data['location_line'],
        flyer_data['body_blurb']
    ]
    for line in details:
        wrapped = textwrap.fill(line, width=45)
        draw.text((left_margin, y), wrapped, fill='white', font=font_small)
        y += 48

    # Call to action (accent color)
    draw.text((left_margin, y), flyer_data['call_to_action'], fill=accent_color, font=font_med)
    y += 80

    # QR code (bottom right)
    qr_img = Image.open(qr_path).resize((220, 220))
    img.paste(qr_img, (img.width - 260, img.height - 260))

    # Footer
    draw.text((left_margin, img.height - 60), "Scan QR to Register!", fill='white', font=font_small)

    img.save(output_path)
    print(f"✅ FLYER SAVED to: {os.path.abspath(output_path)}")
    print("📱 View in any image viewer/PDF converter")
    return output_path

async def email_generation_node(state: State) -> Command[Literal["form_generation", "__end__"]]:
    print_banner("📧 STEP 1: PROPOSAL EMAIL")
    
    event_details = state.get("event_details", {})
    truncated_messages = truncate_messages(state["messages"])
    print(f"✉️ Generating proposal email for: {event_details.get('event_name', 'Unnamed Event')}")
    
    # Create a detailed email generation prompt
    email_task_prompt = f"""
    Write a professional proposal email to church leadership about this youth event.
    
    TASK: Generate a complete proposal email with:
    - Subject line: "Subject: <meaningful title>"
    - Event overview and purpose
    - Date, time, and location details
    - Expected attendance and logistics
    - Budget considerations and approval request
    - Safety and supervision plan
    - Professional, persuasive tone
    
    STEP 1: Write the complete email body first (PLAIN TEXT ONLY, no JSON)
    
    STEP 2: Then call the save_proposal_email tool with:
    - email_body: your complete email text
    - event_name: "{event_details.get('event_name', 'Unnamed Event')}"
    - recipient: "pastor@church.org"
    
    Event Details:
    {json.dumps(event_details, indent=2)}
    """
    
    # Create email agent with enhanced prompt
    email_agent = create_agent(llm, tools=[save_proposal_email], system_prompt=email_task_prompt)
    response = await email_agent.ainvoke({"messages": truncated_messages})
    
    # Extract and display email content
    email_content = None
    tool_used = False
    for msg in response["messages"]:
        if msg.type == "ai":
            email_content = msg.content
        if hasattr(msg, 'name') and msg.name == "save_proposal_email":
            tool_used = True
    
    if email_content:
        print("\n📄 GENERATED EMAIL:")
        print("="*60)
        print(email_content[:500] + ("..." if len(email_content) > 500 else ""))
        print("="*60)
    
    if tool_used:
        print("✅ Email archived successfully")
    else:
        print("⚠️ Note: Tool may not have been called - email shown above for reference")
    
    print("\n➡️ Moving to form generation...")
    return Command(update={"messages": response["messages"], "event_details": event_details}, goto="form_generation")

async def form_generation_node(state: State) -> Command[Literal["flyer_generation", "__end__"]]:
    print_banner("📋 STEP 2: REGISTRATION FORM")
    
    event_details = state.get("event_details", {})
    form_url = event_details.get("form_url", "https://forms.example.com/register")
    truncated_messages = truncate_messages(state["messages"])
    
    print(f"🔗 Form will link to: {form_url}")
    
    # Create writer agent with form-specific prompt
    form_task_prompt = f"""
    Design a mobile-friendly registration form for this youth event.
    
    TASK: Create a form JSON schema with these required fields:
    {{
      "title": "string",
      "description": "string", 
      "fields": [{{"name": "str", "label": "str", "type": "text|number|tel|textarea|select", "required": bool, "options": []}}]
    }}
    
    REQUIREMENTS:
    - Title: "{event_details.get('event_name', 'Event')} Registration"
    - Description: Brief description of the event
    - Include fields for: youth_first_name, youth_last_name, youth_age, parent_first_name, parent_last_name, parent_phone, youth_phone (optional), special_accommodations, transportation_needed
    - Make it mobile-friendly and easy to fill
    
    STEP 1: Design the complete form JSON schema (make sure all fields are valid)
    
    STEP 2: Call the build_registration_form tool with:
    - llm_output: the complete JSON schema as a string
    - form_url: "{form_url}"
    
    Event Details:
    {json.dumps(event_details, indent=2)}
    """
    
    form_agent = create_agent(llm, tools=[build_registration_form], system_prompt=form_task_prompt)
    response = await form_agent.ainvoke({"messages": truncated_messages})
    
    # Extract form schema and QR path from tool results
    form_schema = None
    qr_path = None
    form_output = None
    
    try:
        from langchain_core.messages import ToolMessage
        for msg in response["messages"]:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, dict) and "form_schema" in content:
                    form_schema = content.get("form_schema")
                    qr_path = content.get("qr_path")
                    break
                elif isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if "form_schema" in parsed:
                            form_schema = parsed.get("form_schema")
                            qr_path = parsed.get("qr_path")
                            break
                    except:
                        pass
            elif msg.type == "ai":
                form_output = msg.content
    except:
        pass
    
    # Display form schema
    if form_schema:
        print("\n📋 FORM SCHEMA GENERATED:")
        print("="*60)
        print(json.dumps(form_schema, indent=2))
        print("="*60)
    elif form_output:
        print("\n📋 FORM OUTPUT (for reference):")
        print("="*60)
        print(form_output[:500] + ("..." if len(form_output) > 500 else ""))
        print("="*60)
    
    if qr_path:
        print(f"✅ Form generated with QR code: {qr_path}")
    else:
        print("⚠️ QR code path not extracted, will generate new one")
        qr_path = create_qr_png(form_url)
    
    print("\n➡️ Moving to flyer generation...")
    return Command(
        update={
            "messages": response["messages"],
            "event_details": event_details,
            "form_url": form_url,
            "qr_path": qr_path
        },
        goto="flyer_generation"
    )

async def flyer_generation_node(state: State) -> Command[Literal["__end__"]]:
    print_banner("🎨 STEP 3: FLYER WITH QR CODE")
    
    event_details = state.get("event_details", {})
    form_url = state.get("form_url", event_details.get("form_url", "https://forms.example.com/register"))
    qr_path = state.get("qr_path")
    truncated_messages = truncate_messages(state["messages"])
    
    print(f"🎯 Creating flyer for: {event_details.get('event_name', 'Unnamed Event')}")
    if qr_path:
        print(f"📱 Using existing QR code: {qr_path}")
    else:
        print(f"📱 Will generate new QR linking to: {form_url}")
    
    # Create writer agent with flyer-specific prompt
    flyer_task_prompt = f"""
    Create an attractive and eye-catching flyer for this youth event.
    
    TASK: Design a flyer JSON schema with these required fields:
    {{
      "headline": "str (main event title, short and catchy)",
      "subheadline": "str (tagline or subtitle)", 
      "date_time_line": "str (when is the event?)",
      "location_line": "str (where is the event?)",
      "body_blurb": "str (short description of the event)",
      "call_to_action": "str (what should people do? e.g., 'Scan QR to Register!')",
      "color_scheme": {{"primary": "#hex (background color)", "secondary": "#hex", "accent": "#hex (highlight color)"}}
    }}
    
    REQUIREMENTS:
    - Use bright, engaging colors (primary should be vibrant)
    - Mention that scanning the QR code registers attendees
    - Keep text short and energetic
    - Target audience: youth and parents
    - Include date/time/location clearly
    
    STEP 1: Create the complete flyer JSON design
    
    STEP 2: Call the generate_flyer_package tool with:
    - flyer_data: the JSON object you designed
    - form_url: "{form_url}"
    
    Event Details:
    {json.dumps(event_details, indent=2)}
    """
    
    flyer_agent = create_agent(llm, tools=[generate_flyer_package], system_prompt=flyer_task_prompt)
    response = await flyer_agent.ainvoke({"messages": truncated_messages})
    
    # Extract flyer output
    flyer_output = None
    for msg in response["messages"]:
        if msg.type == "ai":
            flyer_output = msg.content
    
    if flyer_output:
        print("\n🎨 FLYER OUTPUT:")
        print("="*60)
        print(flyer_output[:500] + ("..." if len(flyer_output) > 500 else ""))
        print("="*60)
    
    print("✅ Flyer generated with embedded QR code")
    print("\n📁 All files exported to current directory")
    
    return Command(
        update={
            "messages": response["messages"],
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
    Write a professional proposal email to church leadership.
    Output PLAIN TEXT email body ONLY (no JSON), ready to send.

    Structure:
    - Subject line as first line: "Subject: <title>"
    - Include: event details, safety/logistics, budget, approval ask
    - Professional tone, clear and concise
    
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
      "form_url": "https://forms.example.com/basketball-2026"
    }
    """)
    
    print("\n🎯 The system will automatically:")
    print("  1. Generate a proposal email")
    print("  2. Build a registration form")
    print("  3. Create a QR code linking to the form")
    print("  4. Generate a flyer with the QR code embedded")
    
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

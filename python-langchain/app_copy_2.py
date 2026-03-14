import os
import asyncio
import json
from typing import TypedDict, Annotated, Literal, Dict, Any
from dotenv import load_dotenv
from langchain.agents import create_agent
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

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Global agents
researcher_agent = None
writer_agent = None
editor_agent = None

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

async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    print_banner("🔍 RESEARCH PHASE")
    
    truncated_messages = truncate_messages(state["messages"])
    print(f"📄 Feeding {len(truncated_messages)} messages to researcher...")
    
    response = await researcher_agent.ainvoke({"messages": truncated_messages})
    
    print("\n📋 RESEARCH SUMMARY:")
    for msg in response["messages"][-3:]:  # Last 3
        if msg.type == "ai":
            print(f"• {msg.content[:150]}...")
        elif hasattr(msg, 'name') and msg.name:
            print(f"• Tool '{msg.name}' used")
    
    print("\n➡️ Passing to WRITER...")
    return Command(update={"messages": response["messages"]}, goto="writer")

async def writer_node(state: State) -> Command[Literal["editor", "__end__"]]:
    print_banner("✍️ WRITING PHASE")
    
    truncated_messages = truncate_messages(state["messages"])
    print(f"📝 Writer received context ({len(truncated_messages)} msgs)")
    
    response = await writer_agent.ainvoke({"messages": truncated_messages})
    output = response["messages"][-1].content
    
    print("\n📄 WRITER OUTPUT:")
    print("="*50)
    print(output)
    print("="*50)
    
    print("\n➡️ Sending to EDITOR for review...")
    return Command(update={"messages": response["messages"]}, goto="editor")

async def editor_node(state: State) -> Command[Literal["__end__", "writer"]]:
    print_banner("🔍 EDITOR REVIEW")
    
    truncated_messages = truncate_messages(state["messages"])
    print("📋 Editor checking grammar, accuracy, completeness...")
    
    response = await editor_agent.ainvoke({"messages": truncated_messages})
    feedback = response["messages"][-1].content
    
    print("\n👀 EDITOR FEEDBACK:")
    print("="*50)
    print(feedback)
    print("="*50)
    
    if "REVISE" in feedback.upper() or "REVISE" in feedback.lower():
        print("\n🔄 EDITOR SAYS: REVISE → Back to WRITER")
        return Command(update={"messages": response["messages"]}, goto="writer")
    
    print("\n✅ EDITOR APPROVES! Final output ready.")
    print("\n📁 CHECK YOUR LOCAL FILES FOR EXPORTS ↓")
    return Command(update={"messages": response["messages"]}, goto="__end__")


def get_task_prompt(task_type: str, payload: Dict[str, Any]) -> str:
        prompts = {
        "youth_registration_form": """
    You are a form designer for youth ministry events.
    Output VALID JSON schema ONLY for a mobile-friendly registration form.

    {
      "title": "string",
      "description": "string", 
      "fields": [{"name": "str", "label": "str", "type": "text|number|tel|textarea|select", "required": bool, "options": []}]
    }

    Include: youth first/last/age, parent first/last/phone, youth_phone(opt), accommodations, transportation.
    """,
        
        "flyer": """
    Create flyer content JSON for youth events. Respond with ONLY valid JSON. No text, explanations, markdown, or code blocks before/after. Example: {\"headline\": \"...\"}.

    {
      "headline": "str",
      "subheadline": "str", 
      "date_time_line": "str",
      "location_line": "str",
      "body_blurb": "str",
      "call_to_action": "str",
      "color_scheme": {"primary": "#hex", "secondary": "#hex", "accent": "#hex"}
    }

    Short, energetic text. Mention QR registration.
    """,
        
        "proposal_email": """
    Write a professional proposal email to church leadership.
    Output PLAIN TEXT email body ONLY (no JSON), ready for Outlook.

    Include: event details, safety/logistics, budget, approval ask.
    Subject line in first line as "Subject: Title".
    """
        }
        base_prompt = prompts.get(task_type, "You are a skilled content writer.")
        return base_prompt + f"\n\nEvent details: {json.dumps(payload, indent=2)}"

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

async def main():
    global researcher_agent, writer_agent, editor_agent
    
    print_banner("🚀 EVENT PLANNER ASSISTANT v2.1 (DEBUG MODE)", "✨")
    
    # API checks
    missing = []
    if not os.getenv("GITHUB_TOKEN"): missing.append("GITHUB_TOKEN")
    if not os.getenv("TAVILY_API_KEY"): missing.append("TAVILY_API_KEY (optional)")
    
    if missing:
        print("⚠️", ", ".join(missing), "→ Add to .env")
    
    llm = ChatOpenAI(model="openai/gpt-4o-mini", temperature=0.3,
                    base_url="https://models.github.ai/inference",
                    api_key=os.getenv("GITHUB_TOKEN"))
    
    # Load prompts
    prompts = {"researcher": "You are a research assistant.", "writer": "", "editor": "You are an editor."}
    for fname, key in [("researcher.json", "researcher"), ("writer.json", "writer"), ("editor.json", "editor")]:
        try:
            prompts[key] = json.load(open(fname))["template"]
        except:
            print(f"⚠️ {fname} missing → using fallback")
    
    # Researcher
    researcher_tools = []
    if os.getenv("TAVILY_API_KEY"):
        research_client = MultiServerMCPClient({
            "tavily": {"transport": "http", "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}"}
        })
        researcher_tools = await research_client.get_tools()
        print(f"🔍 Researcher tools loaded: {len(researcher_tools)}")
    
    researcher_agent = create_agent(llm, tools=researcher_tools, system_prompt=prompts["researcher"])
    editor_agent = create_agent(llm, tools=[], system_prompt=prompts["editor"])
    
    # Graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node).add_node("writer", writer_node).add_node("editor", editor_node)
    builder.add_edge(START, "researcher")
    graph = builder.compile()
    
    print("\n📋 USAGE: Paste JSON with 'task_type':")
    print("  • 'youth_registration_form' → Form JSON + QR")
    print("  • 'flyer' → Flyer JSON + PNG + QR") 
    print("  • 'proposal_email' → Ready-to-send email")
    print("\n🎯 Basketball clinic example ready to copy ↓")
    
    while True:
        try:
            payload_str = input("\n📝 JSON payload (or 'quit'): ").strip()
            if payload_str.lower() in ['quit', 'exit', 'q']: break
            
            payload = json.loads(payload_str)
            task_type = payload.get("task_type")
            if not task_type:
                print("❌ Add 'task_type' to JSON")
                continue
            
            print(f"\n🎯 TASK: {task_type.upper()}")
            print(f"📊 Event: {payload.get('event_details', {}).get('event_name', 'Unnamed')}")
            
            # Dynamic writer
            writer_prompt = get_task_prompt(task_type, payload)
            writer_agent = create_agent(llm, tools=[], system_prompt=writer_prompt)
            
            # RUN PIPELINE
            print_banner("⚙️ PIPELINE STARTING")
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=f"TASK={task_type}\n{payload_str}")]
            })
            
            output = result["messages"][-1].content
            print_banner("✅ PIPELINE COMPLETE")
            
            # POST-PROCESS EXPORTS
            event_details = payload.get("event_details", {})
            form_url = event_details.get("form_url", "https://forms.example.com/register")
            
            if task_type in ["flyer", "basketball_clinic"]:
                print_banner("🎨 FLYER EXPORT MODE")
                print("📄 Raw writer output:")
                print(output[:500] + "..." if len(output) > 500 else output)
                
                # Always generate QR
                qr_path = create_qr_png(form_url)
                
                # Try to extract JSON
                try:
                    flyer_data = extractjsonfromtext(output)
                    print("✅ Parsed JSON from output!")
                    flyer_path = save_flyer_png(flyer_data, qr_path)
                    print(f"\n🎉 FULL FLYER PACKAGE EXPORTED:")
                    print(f"  🖼️  Flyer: {flyer_path}")
                    print(f"  📱  QR Code: {qr_path}")
                except (ValueError, KeyError, TypeError) as e:
                    print(f"⚠️ Error processing flyer: {e}")
                    print(f"📱 QR Code ready: {qr_path}")
                    print("\n💡 Extracted data check:")
                    try:
                        data = extractjsonfromtext(output)
                        print(f"  Type: {type(data)}")
                        print(f"  Content: {str(data)[:300]}")
                    except:
                        print(f"  Could not extract JSON at all")
                    print("\n💡 Could not auto-extract valid flyer JSON. Please copy and paste ONLY the JSON portion below (edit if needed), then press Enter:")
                    print("----- COPY BELOW THIS LINE -----")
                    print(output)
                    print("----- END -----")
                    flyer_json_str = input("Paste valid JSON here (or leave blank to skip flyer export): ").strip()
                    if flyer_json_str:
                        try:
                            flyer_data = json.loads(flyer_json_str)
                            flyer_path = save_flyer_png(flyer_data, qr_path)
                            print(f"\n🎉 FULL FLYER PACKAGE EXPORTED (manual):")
                            print(f"  🖼️  Flyer: {flyer_path}")
                            print(f"  📱  QR Code: {qr_path}")
                        except Exception as e2:
                            print(f"❌ Still invalid: {e2}")
                
            elif task_type == "youth_registration_form":
                qr_path = create_qr_png(form_url)
                print("\n📋 Form JSON (copy to SurveyHeart/Streamlit):")
                try:
                    form_data = extractjsonfromtext(output)
                    print(json.dumps(form_data, indent=2))
                except:
                    print(output)
            
            elif task_type == "proposal_email":
                print("\n📧 EMAIL READY - COPY TO OUTLOOK/GMAIL:")
                print("-"*50)
                print(output)
                print("-"*50)
            
            else:
                print("\n📋 OUTPUT (copy as needed):")
                print(output)
                
        except json.JSONDecodeError:
            print("❌ Invalid JSON format")
        except Exception as e:
            print(f"💥 Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

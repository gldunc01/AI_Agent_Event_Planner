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
import qrcode

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Global agents
researcher_agent = None
writer_agent = None
editor_agent = None

def truncate_messages(messages: list, max_messages: int = 5) -> list:
    """Keep only the most recent messages to avoid token limit."""
    if len(messages) <= max_messages:
        return messages
    return messages[:1] + messages[-(max_messages-1):]

async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    
    truncated_messages = truncate_messages(state["messages"])
    response = await researcher_agent.ainvoke({"messages": truncated_messages})
    
    # Debug research output
    print("\n--- Research Results ---")
    for msg in response["messages"]:
        if msg.type == "tool":
            content_preview = str(msg.content)[:300] + "..." if len(str(msg.content)) > 300 else str(msg.content)
            print(f"Tool '{msg.name}': {content_preview}")
        elif msg.type == "ai":
            print(f"\nResearcher: {msg.content[:200]}...")
    
    print("\n" + "="*50 + "\n")
    return Command(update={"messages": response["messages"]}, goto="writer")

async def writer_node(state: State) -> Command[Literal["editor", "__end__"]]:
    print("\n" + "="*50)
    print("WRITER NODE")
    print("="*50)
    
    truncated_messages = truncate_messages(state["messages"])
    response = await writer_agent.ainvoke({"messages": truncated_messages})
    
    final_message = response["messages"][-1]
    print(f"\nWriter Output:\n{final_message.content}")
    
    print("\n" + "="*50 + "\n")
    return Command(update={"messages": response["messages"]}, goto="editor")

async def editor_node(state: State) -> Command[Literal["__end__", "writer"]]:
    print("\n" + "="*50)
    print("EDITOR NODE")
    print("="*50)
    
    truncated_messages = truncate_messages(state["messages"])
    response = await editor_agent.ainvoke({"messages": truncated_messages})
    
    final_message = response["messages"][-1].content
    print(f"\nEditor Feedback:\n{final_message}")
    
    # Check if editor wants revision
    if "REVISE" in final_message.upper():
        print("\n⚠️ REVISION REQUESTED - back to writer")
        return Command(update={"messages": response["messages"]}, goto="writer")
    
    print("\n✓ APPROVED - workflow complete")
    print("="*50 + "\n")
    return Command(update={"messages": response["messages"]}, goto="__end__")

def get_task_prompt(task_type: str, payload: Dict[str, Any]) -> str:
    """Dynamic system prompts based on task_type."""
    
    prompts = {
        "youth_registration_form": """
You are a form designer for youth ministry events.
Output VALID JSON schema ONLY for a mobile-friendly registration form.

{
  "title": "string",
  "description": "string", 
  "fields": [{"name": "str", "label": "str", "type": "text|number|tel|textarea|select", "required": bool, "options": []}]
}

Include: youth_first,last name/age, parent first,last/phone, youth_phone(opt), accommodations(textarea), transportation(select).
""",
        
        "flyer": """
Create flyer content JSON for youth events. Output JSON ONLY.

{
  "headline": "str",
  "subheadline": "str",
  "date_time_line": "str",
  "location_line": "str", 
  "body_blurb": "str",
  "call_to_action": "str",
  "color_scheme": {"primary": "#hex", "secondary": "#hex", "accent": "#hex"}
}

Short, energetic text for teens/parents. Mention QR registration.
""",
        
        "proposal_email": """
Write a professional proposal email to church leadership.
Output PLAIN TEXT email body ONLY (no JSON), ready for Outlook.

Include: event details, safety/logistics, budget, approval ask.
Subject line in first line as "Subject: Title".
Warm but formal tone.
"""
    }
    
    base_prompt = prompts.get(task_type, "You are a skilled content writer. Create structured output based on event details.")
    return base_prompt + f"\n\nEvent details: {json.dumps(payload, indent=2)}"

def create_qr_png(form_url: str, output_path: str = "qr.png") -> str:
    """Generate QR code PNG for form URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(form_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_path)
    print(f"✅ QR code saved: {output_path}")
    return output_path

async def main():
    global researcher_agent, writer_agent, editor_agent
    
    # API key checks
    if not os.getenv("GITHUB_TOKEN"):
        print("❌ GITHUB_TOKEN required (.env)")
        return
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY required (tavily.com)")
        return
    
    # LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    
    # Load base prompts
    try:
        with open("templates/researcher.json", "r") as f: researcher_prompt = json.load(f)["template"]
        with open("templates/writer.json", "r") as f: writer_base_prompt = json.load(f)["template"]  
        with open("templates/editor.json", "r") as f: editor_prompt = json.load(f)["template"]
    except FileNotFoundError:
        print("⚠️ Using fallback prompts (create JSON files)")
        researcher_prompt = writer_base_prompt = editor_prompt = "You are a helpful assistant."
    
    # Researcher tools (Tavily optional)
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        research_client = MultiServerMCPClient({
            "tavily": {"transport": "http", "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}"}
        })
        researcher_tools = await research_client.get_tools()
    else:
        researcher_tools = []
    
    researcher_agent = create_agent(llm, tools=researcher_tools, system_prompt=researcher_prompt)
    editor_agent = create_agent(llm, tools=[], system_prompt=editor_prompt)
    
    # Build graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node) 
    builder.add_node("editor", editor_node)
    
    builder.add_edge(START, "researcher")
    graph = builder.compile()
    
    print("\n" + "="*60)
    print("🚀 EVENT PLANNER ASSISTANT v2.0")
    print("Paste JSON payload with 'task_type' (youth_registration_form | flyer | proposal_email)")
    print("Example: See basketball clinic sample")
    print("="*60)
    
    # Interactive input
    while True:
        try:
            payload_str = input("\n📝 Event JSON (or 'quit'): ").strip()
            if payload_str.lower() in ['quit', 'exit', 'q']:
                break
                
            payload = json.loads(payload_str)
            task_type = payload.get("task_type")
            event_details = payload.get("event_details", {})
            
            if not task_type:
                print("❌ Missing 'task_type'")
                continue
            
            # Dynamic writer prompt
            writer_prompt = get_task_prompt(task_type, payload)
            writer_agent = create_agent(llm, tools=[], system_prompt=writer_prompt)
            
            # Run pipeline
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=f"Task: {task_type}\nDetails: {json.dumps(payload)}")]
            })
            
            output = result["messages"][-1].content
            print("\n🎉 RESULT:\n", output)
            
            # Post-process based on task_type
            if task_type == "youth_registration_form" and event_details.get("form_url"):
                qr_path = create_qr_png(event_details["form_url"])
                print(f"\n📱 Form ready + QR: {qr_path}")
            
            elif task_type == "flyer":
                print("\n🖼️ Use this JSON in Canva/ReportLab + embed QR")
            
        except json.JSONDecodeError:
            print("❌ Invalid JSON")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
    
    print("Orchestration complete!")

if __name__ == "__main__":
    asyncio.run(main())

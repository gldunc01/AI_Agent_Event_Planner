"""
Task runner that bridges UI (ui_app.py) with the LangGraph pipeline (app_copy_2.py)

This module provides a `run_task` function that:
1. Takes task_type and payload from the UI
2. Runs the appropriate LangGraph pipeline
3. Returns results formatted for Streamlit display
"""

import os
import asyncio
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

import app_copy_2
from app_copy_2 import (
    State,
    email_generation_node,
    form_generation_node,
    flyer_generation_node,
    generate_standardized_form,
    create_qr_png,
    save_proposal_email,
    generate_flyer_package,
)

load_dotenv()


async def run_task(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a task based on task_type and return formatted results.
    
    Args:
        task_type: One of 'flyer', 'youth_registration_form', 'basketball_clinic', 'proposal_email'
        payload: Dictionary with 'event_details' and other task-specific data
    
    Returns:
        Dictionary with:
        - 'final_text': Generated content as string
        - 'files': List of file paths generated
    """
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN")
    )
    
    # Make llm available to nodes
    app_copy_2.llm = llm
    
    event_details = payload.get("event_details", {})
    form_url = event_details.get("form_url", "https://forms.example.com/register")
    
    files_generated = []
    final_text = ""
    
    try:
        if task_type == "proposal_email":
            # Run just the email generation node
            print("📧 Generating proposal email...")
            
            builder = StateGraph(State)
            builder.add_node("email_generation", email_generation_node)
            builder.add_edge(START, "email_generation")
            graph = builder.compile()
            
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=json.dumps(event_details))],
                "event_details": event_details,
                "form_url": form_url,
                "qr_path": ""
            })
            
            # Find generated email file
            for file in Path(".").glob("proposal_email_*.txt"):
                files_generated.append(str(file.absolute()))
                with open(file, 'r') as f:
                    final_text = f.read()
            
            if not final_text:
                final_text = "Email generated successfully. Check files for details."
        
        elif task_type == "youth_registration_form":
            # Run form generation node
            print("📋 Generating registration form...")
            
            builder = StateGraph(State)
            builder.add_node("form_generation", form_generation_node)
            builder.add_edge(START, "form_generation")
            graph = builder.compile()
            
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=json.dumps(event_details))],
                "event_details": event_details,
                "form_url": form_url,
                "qr_path": ""
            })
            
            # Find generated files
            for file in Path(".").glob("current_event_form.json"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("event_qr.png"):
                files_generated.append(str(file.absolute()))
            
            form_schema = generate_standardized_form(event_details)
            final_text = f"Registration Form Generated:\n\n{json.dumps(form_schema, indent=2)}"
        
        elif task_type == "flyer":
            # Run full pipeline: email → form → flyer
            print("🎨 Generating complete event package (email, form, flyer)...")
            
            builder = StateGraph(State)
            builder.add_node("email_generation", email_generation_node)
            builder.add_node("form_generation", form_generation_node)
            builder.add_node("flyer_generation", flyer_generation_node)
            
            builder.add_edge(START, "email_generation")
            
            graph = builder.compile()
            
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=json.dumps(event_details))],
                "event_details": event_details,
                "form_url": form_url,
                "qr_path": ""
            })
            
            # Collect all generated files
            for file in Path(".").glob("proposal_email_*.txt"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("current_event_form.json"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("event_qr.png"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("flyer.png"):
                files_generated.append(str(file.absolute()))
            
            final_text = "Complete event package generated:\n✅ Proposal email\n✅ Registration form\n✅ QR code\n✅ Flyer with QR code"
        
        elif task_type == "basketball_clinic":
            # Similar to flyer, but specialized for basketball
            print("🏀 Generating basketball clinic event package...")
            
            # Add basketball-specific details
            event_details["description"] = event_details.get("description", "Basketball clinic for youth")
            
            builder = StateGraph(State)
            builder.add_node("email_generation", email_generation_node)
            builder.add_node("form_generation", form_generation_node)
            builder.add_node("flyer_generation", flyer_generation_node)
            
            builder.add_edge(START, "email_generation")
            
            graph = builder.compile()
            
            result = await graph.ainvoke({
                "messages": [HumanMessage(content=json.dumps(event_details))],
                "event_details": event_details,
                "form_url": form_url,
                "qr_path": ""
            })
            
            # Collect all generated files
            for file in Path(".").glob("proposal_email_*.txt"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("current_event_form.json"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("event_qr.png"):
                files_generated.append(str(file.absolute()))
            for file in Path(".").glob("flyer.png"):
                files_generated.append(str(file.absolute()))
            
            final_text = "Basketball clinic event package generated:\n✅ Proposal email\n✅ Basketball registration form\n✅ QR code\n✅ Flyer with QR code"
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return {
            "final_text": final_text,
            "files": files_generated
        }
    
    except Exception as e:
        error_msg = f"Error generating {task_type}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "final_text": error_msg,
            "files": []
        }


def _get_latest_files(directory: str = ".") -> list:
    """Get list of recently modified files (last 5 minutes)."""
    import os
    import time

    now = time.time()
    cutoff = now - (5 * 60)  # Last 5 minutes
    latest = []

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            mtime = os.path.getmtime(fpath)
            if mtime > cutoff:
                latest.append(os.path.abspath(fpath))

    return latest


# For testing
if __name__ == "__main__":

    async def test():
        payload = {
            "event_details": {
                "event_name": "Youth Basketball Clinic",
                "event_date": "2026-04-15",
                "event_time": "3:00 PM - 5:00 PM",
                "location": "City Sports Complex",
                "form_url": "https://forms.example.com/register",
            },
        }
        result = await run_task("proposal_email", payload)
        print("Result:", result)

    asyncio.run(test())

"""
Wrapper module to expose LangGraph app logic as a simple async function.
This module re-uses the agents and graph from app_copy_2.py.
"""

import os
import asyncio
import json
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
import app_copy_2
from app_copy_2 import (
    State,
    researcher_node,
    writer_node,
    editor_node,
    get_task_prompt,
    generate_flyer_package,
    build_registration_form,
    save_proposal_email,
)

load_dotenv()

# Global graph (initialized once)
_graph = None


async def _initialize_agents():
    """Initialize agents and graph (one time)."""
    global _graph

    if _graph is not None:
        return  # Already initialized

    print("[run_task] Initializing agents...")

    # Load LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN"),
    )

    # Load prompts from files (or use fallbacks)
    prompts = {
        "researcher": "You are a research assistant.",
        "writer": "",
        "editor": "You are an editor.",
    }
    for fname, key in [
        ("researcher.json", "researcher"),
        ("writer.json", "writer"),
        ("editor.json", "editor"),
    ]:
        try:
            prompts[key] = json.load(open(fname))["template"]
        except:
            pass  # Use fallback

    # Tool lists
    writer_tools = [generate_flyer_package, build_registration_form]
    editor_tools = [save_proposal_email]
    researcher_tools = []

    # Create agents and set them globally in app_copy_2
    app_copy_2.researcher_agent = create_agent(
        llm, tools=researcher_tools, system_prompt=prompts["researcher"]
    )
    app_copy_2.editor_agent = create_agent(
        llm, tools=editor_tools, system_prompt=prompts["editor"]
    )

    # Build graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("editor", editor_node)
    builder.add_edge(START, "researcher")
    _graph = builder.compile()

    print("[run_task] Agents initialized ✓")


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


async def run_task(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a LangGraph task and return the final output and generated files.

    Args:
        task_type: One of ["youth_registration_form", "flyer", "basketball_clinic", "proposal_email"]
        payload: Dictionary with "event_details" key containing:
            - event_name: str
            - event_date: str
            - event_time: str
            - location: str
            - form_url: str (optional, defaults to "https://forms.example.com/register")

    Returns:
        Dictionary with:
            - "final_text": str (the final AI/editor output)
            - "files": list (absolute paths to generated files)
    """

    # Initialize agents if needed
    await _initialize_agents()

    print(f"[run_task] Running task: {task_type}")

    # Get files before execution
    files_before = set(
        os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f)
    )

    try:
        # Create a writer agent with task-specific prompt
        writer_tools = [generate_flyer_package, build_registration_form]
        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.3,
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("GITHUB_TOKEN"),
        )
        writer_prompt = get_task_prompt(task_type, payload)
        app_copy_2.writer_agent = create_agent(
            llm, tools=writer_tools, system_prompt=writer_prompt
        )

        # Run the graph
        result = await _graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"TASK={task_type}\n{json.dumps(payload)}"
                    )
                ]
            }
        )

        # Extract final text
        final_text = result["messages"][-1].content

        # Get files created after execution
        files_after = set(
            os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f)
        )
        new_files = list(files_after - files_before)

        return {"final_text": final_text, "files": new_files}

    except Exception as e:
        raise Exception(f"Error running task '{task_type}': {str(e)}")


# For testing
if __name__ == "__main__":

    async def test():
        payload = {
            "task_type": "proposal_email",
            "event_details": {
                "event_name": "Youth Basketball Clinic",
                "event_date": "2026-04-15",
                "event_time": "3:00 PM - 5:00 PM",
                "location": "City Sports Complex",
                "form_url": "https://forms.example.com/register",
            },
        }
        result = await run_task(payload["task_type"], payload)
        print("Result:", result)

    asyncio.run(test())

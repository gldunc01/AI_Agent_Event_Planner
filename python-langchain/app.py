import os
import asyncio
import json
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.types import Command

load_dotenv()

# 1. Define shared state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Global variables for agents (will be set in main)
researcher_agent = None
writer_agent = None
editor_agent = None


def truncate_messages(messages: list, max_messages: int = 5) -> list:
    """Keep only the most recent messages to avoid token limit."""
    if len(messages) <= max_messages:
        return messages
    # Keep the first message (system context) and last N-1 messages
    return messages[:1] + messages[-(max_messages-1):]


async def researcher_node(state: State) -> Command[Literal["writer", "__end__"]]:
    """Research node that hands off to writer."""
    print("\n" + "="*50)
    print("RESEARCHER NODE")
    print("="*50)
    
    # Truncate messages to avoid token limit
    truncated_messages = truncate_messages(state["messages"])
    
    # Get the researcher agent from the closure
    response = await researcher_agent.ainvoke({"messages": truncated_messages})
    
    # Debug: Print search results and tool usage
    print("\n--- Research Results ---")
    for msg in response["messages"]:
        # Check for tool calls (AI messages with tool_calls)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"\nTool Called: {tool_call.get('name', 'Unknown')}")
                print(f"Arguments: {tool_call.get('args', {})}")
        
        # Check for tool responses (ToolMessage)
        if msg.type == "tool":
            print(f"\nTool Response from: {getattr(msg, 'name', 'Unknown Tool')}")
            content_preview = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
            print(f"Content: {content_preview}")
        
        # Print AI responses (but not tool calls)
        if msg.type == "ai" and not hasattr(msg, 'tool_calls'):
            print(f"\nResearcher Response:")
            print(f"{msg.content}")
    
    print("\n" + "="*50 + "\n")
    
    # Native handoff: explicitly tell the graph to move to 'writer'
    return Command(
        update={"messages": response["messages"]},
        goto="writer"
    )


async def writer_node(state: State) -> Command[Literal["editor", "__end__"]]:
    """Writer node that receives research and hands off to editor."""
    print("\n" + "="*50)
    print("WRITER NODE")
    print("="*50)
    
    # Truncate messages to avoid token limit
    truncated_messages = truncate_messages(state["messages"])
    
    # Get the writer agent from the closure
    response = await writer_agent.ainvoke({"messages": truncated_messages})
    
    # print("\n--- Writer Output ---")
    # for msg in response["messages"]:
    #     if msg.type == "ai" and not hasattr(msg, 'tool_calls'):
    #         print(f"\nWriter Response:")
    #         print(f"{msg.content}")

    # Print the written content
    final_message = response["messages"][-1]
    print(f"\nWriter Output:")
    print(f"{final_message.content}")
    
    print("\n" + "="*50 + "\n")
    
    # Native handoff: move to editor
    return Command(
        update={"messages": response["messages"]},
        goto="editor"
    )


async def editor_node(state: State) -> Command[Literal["__end__"]]:
    """Editor node that finalizes the content."""
    print("\n" + "="*50)
    print("EDITOR NODE")
    print("="*50)
    
    # Truncate messages to avoid token limit
    truncated_messages = truncate_messages(state["messages"])
    
    # Get the editor agent from the closure
    response = await editor_agent.ainvoke({"messages": truncated_messages})
    
    # print("\n--- Editor Output ---")
    # for msg in response["messages"]:
    #     if msg.type == "ai" and not hasattr(msg, 'tool_calls'):
    #         print(f"\nEditor Response:")
    #         print(f"{msg.content}")

    # Debug: Print editor feedback
    final_message = response["messages"][-1]
    print(f"\nEditor Feedback:")
    print(f"{final_message.content}")
    
    # Example logic: if editor finds an error, hand back to writer
    if "REVISE" in str(final_message.content):
        print("\n⚠️  Editor requested REVISION - routing back to writer")
    
        print("\n" + "="*50 + "\n")
    
        # End the workflow
        return Command(
            update={"messages": response["messages"]},
            goto="writer"  # For testing, we can route back to writer to see the loop
        )
    
    print("\n✓ Editor approved - workflow complete")
    print("="*50 + "\n")
    
    return Command(
        update={"messages": response["messages"]},
        goto="__end__"
    )



async def main():
    """Run the multi-agent content creation workflow."""
    global researcher_agent, writer_agent, editor_agent
    
    # Check for required API keys
    if not os.getenv("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN not found.")
        print("Add GITHUB_TOKEN=your-token to a .env file")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found.")
        print("Add TAVILY_API_KEY=your-key to a .env file")
        print("Get your API key from: https://app.tavily.com/")
        return
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        base_url="https://models.github.ai/inference",
        api_key=os.getenv("GITHUB_TOKEN")
    )

    # Load prompts from your local filesystem
    with open("templates/researcher.json", "r") as f:
        researcher_data = json.load(f)
        researcher_prompt = researcher_data.get("template", "You are a helpful research assistant.")

    with open("templates/writer.json", "r") as f:
        writer_data = json.load(f)
        writer_prompt = writer_data.get("template", "You are a helpful writing assistant.")

    with open("templates/editor.json", "r") as f:
        editor_data = json.load(f)
        editor_prompt = editor_data.get("template", "You are a helpful editing assistant.")

    # Get Tavily API key from environment
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Create MCP client for Tavily
    research_client = MultiServerMCPClient({
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}",
        }
    })
    
    # Get tools from the client (await because it's async)
    researcher_tools = await research_client.get_tools()
    
    print(f"Research tools: {[tool.name for tool in researcher_tools]}")

    # Create agents using create_agent (new API)
    researcher_agent = create_agent(
        llm, 
        tools=researcher_tools, 
        system_prompt=researcher_prompt
    )
    
    # # Create writer agent (without tools for now)
    # writer_prompt = "You are a professional content writer. Take the research findings and write engaging, well-structured content."
    writer_agent = create_agent(
        llm,
        tools=[],
        system_prompt=writer_prompt
    )
    
    # # Create editor agent (without tools for now)
    # editor_prompt = "You are a professional editor. Review and refine the content for clarity, grammar, and impact."
    editor_agent = create_agent(
        llm,
        tools=[],
        system_prompt=editor_prompt
    )

    # Build the Graph
    builder = StateGraph(State)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("editor", editor_node)
    
    # Set the entry point and edges
    builder.add_edge(START, "researcher")
    # builder.add_edge("researcher", "writer")
    # builder.add_edge("writer", "editor")
    
    graph = builder.compile()
    
    # Run the workflow
    print("\n" + "="*50)
    print("Starting Multi-Agent Content Creation Workflow")
    print("="*50 + "\n")
    
    user_input = input("Enter the topic that you would like to research: ")
    initial_message = HumanMessage(content=user_input)
    result = await graph.ainvoke({"messages": [initial_message]})
    
    print("\n" + "="*50)
    print("Workflow Complete")
    print("="*50 + "\n")
    print("Final Output:")
    print(result["messages"][-1].content if result["messages"] else "No output")
    print("\nOrchestration setup complete!")


if __name__ == "__main__":
    asyncio.run(main())

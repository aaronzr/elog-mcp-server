import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from contextlib import AsyncExitStack

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessageChunk
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Point to where your MCP server is running
MCP_ENDPOINT = "http://localhost:9000/mcp"

# === System Prompt ===
system_prompt = """
You are an intelligent AI assistant who answers questions about the electronic logbook (ELOG) used at SLAC National Accelerator Laboratory.
Use the retriever tool to look up facts. Always cite specific document sources in your responses.
"""

# === Agent State ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# === Graph Node: LLM ===
def should_continue(state: AgentState):
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and len(last.tool_calls) > 0

def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = AIMessageChunk("")
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        response += chunk
    return {"messages": [response]}


# === Graph Node: Tools ===
def create_tool_node(tools_dict):
    async def take_action(state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            print(f"Calling Tool: {tool_name} with args: {tool_args}")

            tool = tools_dict.get(tool_name)
            if tool is None:
                result = f"Tool {tool_name} not found."
            else:
                result = await tool.ainvoke(tool_args)

            results.append(
                ToolMessage(tool_call_id=call["id"], name=tool_name, content=str(result))
            )

        return {"messages": results}

    return take_action


# === Setup MCP Session ===
async def setup_mcp():
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()

    read, write, _ = await exit_stack.enter_async_context(
        streamablehttp_client(MCP_ENDPOINT)
    )
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()

    tools = await load_mcp_tools(session)
    return tools, session, exit_stack


# === Main Agent Runner ===
async def main():
    load_dotenv()
    
    tools, session, exit_stack = await setup_mcp()

    try:
        tools_dict = {t.name: t for t in tools}

        global llm  # Bound model must be created *after* tools are available
        llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)

        # Construct LangGraph
        graph = StateGraph(AgentState)
        graph.add_node("llm", call_llm)
        graph.add_node("retriever_agent", create_tool_node(tools_dict))
        graph.set_entry_point("llm")

        graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
        graph.add_edge("retriever_agent", "llm")

        agent = graph.compile()

        # Interactive loop
        print("=== RAG AGENT ===")
        while True:
            user_input = input("\n\nWhat is your question (or 'exit')? ")
            if user_input.lower() in ("exit", "quit"):
                break
            if user_input.lower() in ("history", "hist", "messages", "msg"):
                print(agent.get_state_history())
                continue
            print("\n=== ANSWER ===\n")
            await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})

    finally:
        await exit_stack.aclose()


# === Run ===
if __name__ == "__main__":
    asyncio.run(main())

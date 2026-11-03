"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from contextlib import ExitStack
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

# TODO: Create the BedrockAgentCoreApp instance
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in the infrastructure setup section of the project instructions.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# This starter uses an unsigned MCP connection and therefore assumes the
# project Gateway is configured with the NONE authorizer.
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = "https://customersupportgateway-fcpmoa2e9p.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "U19WXDH0DK"         
REGION      = "us-east-1"      
MEMORY_ID   = "CustomerSupportMemory-Ity05i69Og"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

model_id = "global.amazon.nova-2-lite-v1:0"

# TODO: Create the BedrockModel instance
model = BedrockModel(model_id=model_id)

# TODO: Create the MemoryClient instance
memory_client = MemoryClient(region_name=REGION)

# TODO: Create the boto3 bedrock-agent-runtime client
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


def _extract_text(value):
    """Best-effort extraction of plain text from nested Strands or memory payloads."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _extract_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "content", "summary", "memory", "value", "message"):
            if key in value:
                text = _extract_text(value[key])
                if text:
                    return text
        for nested_value in value.values():
            text = _extract_text(nested_value)
            if text:
                return text
        return ""
    return ""


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Read the namespace from strategy["namespaceTemplates"][0].
#      For compatibility with older AgentCore responses, fall back to
#      strategy["namespaces"][0] when namespaceTemplates is absent.
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces: Dict[str, str] = {}
    for strategy in strategies:
        namespace_templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
        if not namespace_templates:
            continue
        namespace = namespace_templates[0]
        if not namespace:
            continue
        strategy_type = strategy.get("strategyType") or strategy.get("type") or strategy.get("name")
        if strategy_type:
            namespaces[strategy_type] = namespace
    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_id = memory_id
        self.memory_client = memory_client
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = getattr(event.agent, "messages", []) or []
        latest_message = getattr(event, "message", None) or (messages[-1] if messages else None)
        if not isinstance(latest_message, dict):
            return

        if latest_message.get("role") != "user":
            return

        query = _extract_text(latest_message)
        if not query:
            return

        collected: list[str] = []
        for strategy_name, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            memories = self.memory_client.retrieve_memories(
                self.memory_id,
                namespace,
                query,
                top_k=5,
            )
            for memory in memories:
                memory_text = _extract_text(memory)
                if not memory_text:
                    continue
                entry = f"[{strategy_name}] {memory_text}"
                if entry not in collected:
                    collected.append(entry)

        if not collected:
            return

        prefixed_message = "Customer Context:\n" + "\n".join(collected) + "\n\n" + query

        content = latest_message.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    block["text"] = prefixed_message
                    return
            content.insert(0, {"text": prefixed_message})
        else:
            latest_message["content"] = [{"text": prefixed_message}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = getattr(event.agent, "messages", []) or []
        user_query = None
        agent_response = None

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role == "user" and user_query is None:
                text = _extract_text(message)
                if text:
                    user_query = text
                    break

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role == "assistant" and agent_response is None:
                text = _extract_text(message)
                if text:
                    agent_response = text
                    break

        if user_query and agent_response:
            self.memory_client.create_event(
                self.memory_id,
                self.actor_id,
                self.session_id,
                [(user_query, "USER"), (agent_response, "ASSISTANT")],
            )

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    response = _bedrock_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
    )

    retrieval_results = response.get("retrievalResults") or []
    if not retrieval_results:
        return "No relevant information found in the knowledge base."

    chunks: list[str] = []
    for item in retrieval_results:
        content = item.get("content") or {}
        chunk = content.get("text") or item.get("text") or _extract_text(item)
        if chunk:
            chunks.append(str(chunk))

    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Calculation breakdown, or tier-discount details if the interpreter is unavailable
    """
    code = f"""
import json
import math

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}
loyalty_points = {loyalty_points}
tier = {tier!r}
order_total = {order_total}
product_category = {product_category!r}

points_redeemed = min((loyalty_points // 500) * 500, max(int(order_total * 0.5), 0))
subtotal_after_points = max(order_total - points_redeemed, 0)
discount_rate = tier_rates.get(tier, 0.0)
tier_discount = subtotal_after_points * discount_rate
final_total = max(subtotal_after_points - tier_discount, 0)
total_savings = order_total - final_total
points_earned = int(order_total * earn_rates.get(product_category, 1))
remaining_points = max(loyalty_points - points_redeemed + points_earned, 0)
result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_percent": round(discount_rate * 100, 2),
    "tier_discount_amount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
    "product_category": product_category,
    "tier": tier,
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

        first_result = next(
            (
                event["result"]
                for event in response.get("stream", [])
                if isinstance(event, dict) and "result" in event
            ),
            None,
        )
        if first_result is None:
            raise RuntimeError("Code Interpreter response did not contain a result event.")

        if isinstance(first_result, dict):
            content = first_result.get("content", [])
            output_text = next(
                (
                    item["text"]
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                ),
                None,
            ) if isinstance(content, list) else None
            if output_text is not None:
                return json.dumps(json.loads(output_text))
            if any(
                key in first_result
                for key in ("points_redeemed", "tier_discount_percent", "final_total")
            ):
                return json.dumps(first_result)
            raise RuntimeError("Code Interpreter result did not contain JSON output.")

        if isinstance(first_result, str):
            return json.dumps(json.loads(first_result))
        raise RuntimeError("Code Interpreter returned an unsupported result format.")

    except Exception:
        tier_rate = {"Silver": 0.0, "Gold": 0.10, "Platinum": 0.15}.get(tier, 0.0)
        tier_discount = order_total * tier_rate
        result = {
            "tier_discount_percent": round(tier_rate * 100, 2),
            "tier_discount_amount": round(tier_discount, 2),
            "tier": tier,
        }
        return json.dumps(result)


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id") or payload.get("actor_id") or "anonymous-user"
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(actor_id, session_id, memory_client, MEMORY_ID)
        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]
        gateway_tools_available = False

        with ExitStack() as stack:
            try:
                gateway_client = stack.enter_context(
                    MCPClient(lambda: streamable_http_client(url=GATEWAY_URL))
                )
                gateway_tools = gateway_client.list_tools_sync()
                tools.extend(gateway_tools)
                gateway_tools_available = bool(gateway_tools)
                if not gateway_tools_available:
                    logger.warning("AgentCore Gateway connected but returned no tools.")
            except Exception:
                logger.exception("AgentCore Gateway unavailable; continuing with base tools.")

            system_prompt = (
                "You are a helpful customer support AI for an online store. "
                "Use the knowledge base, loyalty calculator, gateway tools, and browser "
                "to answer product questions, loyalty rewards, order questions, and refunds. "
                "For live web tasks, browser use is mandatory and takes priority: whenever "
                "the user asks you to visit a URL, inspect a website, or report live page "
                "details such as its title, you MUST call the browser tool to navigate to "
                "the requested URL and inspect the page before answering. Never give an "
                "accessibility fallback or answer from general knowledge before attempting "
                "the browser. Say the page could not be accessed only if the browser tool "
                "was actually called and failed. For non-live tasks, use the relevant "
                "knowledge-base, loyalty, or Gateway tools as appropriate."
            )
            if not gateway_tools_available:
                system_prompt += (
                    " Order-tracking and refund Gateway tools are unavailable for this request. "
                    "Do not claim to have looked up live account details, tracked an order, or "
                    "submitted a refund. Explain that these actions are temporarily unavailable. "
                    "You may still answer general questions using the available knowledge base "
                    "and other tools."
                )
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=system_prompt,
            )

            result = agent(user_input)
            content = result.message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]
            return _extract_text(result.message)
    except Exception as exc:
        logger.exception("Agent invocation failed: %s", exc)
        return "I couldn't process that request right now. Please try again later."


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()

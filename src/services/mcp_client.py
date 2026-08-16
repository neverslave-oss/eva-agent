"""
mcp_client.py — Thin MCP client for coding agent discovery (ADR-015).
Connects to MCP servers (HTTP or stdio) and maps their tools to kernel-evolving skill schema.
"""
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Minimal MCP client. Connects to an MCP server and:
    - Lists available tools (tools/list)
    - Calls a tool (tools/call)
    - Maps tools to kernel-evolving's virtual skill schema
    """

    def __init__(self, endpoint: str, agent_type: str = "unknown"):
        self.endpoint = endpoint.rstrip("/")
        self.agent_type = agent_type
        self._tools_cache: Optional[list] = None

    def list_tools(self) -> list:
        """Fetch tools/list from MCP server. Returns list of tool dicts."""
        # Try HTTP MCP (most common for local agents)
        try:
            req = urllib.request.Request(
                f"{self.endpoint}/tools/list",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                tools = data.get("tools", [])
                self._tools_cache = tools
                return tools
        except Exception as e:
            logger.debug("[MCPClient] list_tools failed for %s: %s", self.endpoint, e)
            return []

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool via MCP tools/call. Returns string result."""
        body = json.dumps({"name": name, "arguments": arguments}).encode()
        try:
            req = urllib.request.Request(
                f"{self.endpoint}/tools/call",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
                # MCP returns content as list of {type, text} blocks
                content = data.get("content", [])
                if isinstance(content, list):
                    return "\n".join(
                        c.get("text", str(c)) for c in content
                        if isinstance(c, dict)
                    )
                return str(content)
        except Exception as e:
            logger.warning("[MCPClient] call_tool %s failed: %s", name, e)
            return f"[mcp_error] {e}"

    def to_virtual_skills(self) -> list[dict]:
        """Map MCP tools to kernel-evolving virtual skill descriptors."""
        tools = self._tools_cache or self.list_tools()
        skills = []
        for tool in tools:
            skills.append({
                "name": f"mcp/{self.agent_type}/{tool.get('name', 'unknown')}",
                "description": tool.get("description", ""),
                "source": "mcp",
                "agent_type": self.agent_type,
                "mcp_endpoint": self.endpoint,
                "mcp_tool_name": tool.get("name", ""),
                "input_schema": tool.get("inputSchema", {}),
            })
        return skills

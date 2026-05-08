// M11 — agent graph trace state types.
// The AgentTrace canvas has three kinds of nodes (coordinator, the four
// subagents, and the five tools) and each can be in one of four states.

export type NodeState = "idle" | "active" | "done" | "error"

export type NodeStates = Record<string, NodeState>

// Five tool ids — must match the LangChain @tool function names that fire
// `on_tool_start` / `on_tool_end` events. Verified against
// packages/agents/vanguard_agents/tools/research.py and
// apps/mcp_server/tools/action.py.
export const TOOL_IDS = [
  "web_search",
  "db_query",
  "send_email",
  "create_ticket",
  "log_action",
] as const

export type ToolId = (typeof TOOL_IDS)[number]

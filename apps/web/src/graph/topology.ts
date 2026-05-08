// Static graph topology for the M11 AgentTrace canvas.
// Hardcoded rather than introspected from LangGraph at runtime: the four
// subagents and five tools are stable, and shipping the topology over SSE
// every turn would be wasteful. If the agent graph gains a node, edit here.

export type NodeKind = "coordinator" | "subagent" | "tool"

export interface TraceNode {
  id: string
  label: string
  kind: NodeKind
  x: number
  y: number
}

export interface TraceEdge {
  id: string
  source: string
  target: string
}

// Layout: coordinator on top, four subagents in a row underneath, tools
// hanging off research / action. x grows right; y grows down.
export const TRACE_NODES: TraceNode[] = [
  { id: "coordinator", label: "Coordinator", kind: "coordinator", x: 280, y: 0 },

  { id: "support",  label: "Support",  kind: "subagent", x: 20,  y: 110 },
  { id: "research", label: "Research", kind: "subagent", x: 200, y: 110 },
  { id: "action",   label: "Action",   kind: "subagent", x: 380, y: 110 },
  { id: "clarify",  label: "Clarify",  kind: "subagent", x: 560, y: 110 },

  { id: "web_search",    label: "web_search",    kind: "tool", x: 0,   y: 230 },
  { id: "db_query",      label: "db_query",      kind: "tool", x: 140, y: 230 },
  { id: "send_email",    label: "send_email",    kind: "tool", x: 310, y: 230 },
  { id: "create_ticket", label: "create_ticket", kind: "tool", x: 450, y: 230 },
  { id: "log_action",    label: "log_action",    kind: "tool", x: 590, y: 230 },
]

export const TRACE_EDGES: TraceEdge[] = [
  // coordinator → subagents
  { id: "e_c_support", source: "coordinator", target: "support" },
  { id: "e_c_research", source: "coordinator", target: "research" },
  { id: "e_c_action", source: "coordinator", target: "action" },
  { id: "e_c_clarify", source: "coordinator", target: "clarify" },

  // research → its two tools
  { id: "e_r_websearch", source: "research", target: "web_search" },
  { id: "e_r_dbquery", source: "research", target: "db_query" },

  // action → its three tools
  { id: "e_a_email", source: "action", target: "send_email" },
  { id: "e_a_ticket", source: "action", target: "create_ticket" },
  { id: "e_a_log", source: "action", target: "log_action" },
]

export const TRACE_NODE_IDS: string[] = TRACE_NODES.map((n) => n.id)

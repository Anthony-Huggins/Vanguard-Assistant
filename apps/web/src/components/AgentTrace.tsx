// M11 — live LangGraph trace canvas.
// Renders a static topology (coordinator → 4 subagents → 5 tools) on top of
// React Flow and overlays runtime state from the Zustand store. Each
// agent_start / agent_end / tool_start / tool_end SSE event has already
// flowed through the event reducer by the time we render, so this component
// is purely presentational.

import { useMemo, useState } from "react"
import { Background, ReactFlow, type Edge, type Node } from "@xyflow/react"

import { TRACE_EDGES, TRACE_NODES, type TraceNode } from "@/graph/topology"
import { useChatStore } from "@/lib/store"
import type { NodeState, NodeStates } from "@/types/trace"

const KIND_COLORS: Record<TraceNode["kind"], string> = {
  coordinator: "#7c3aed", // violet-600
  subagent: "#2563eb", // blue-600
  tool: "#0d9488", // teal-600
}

const STATE_BG: Record<NodeState, string> = {
  idle: "#ffffff",
  active: "#fef3c7", // amber-100, paired with the pulse ring
  done: "#dcfce7", // green-100
  error: "#fee2e2", // red-100
}

const STATE_RING: Record<NodeState, string> = {
  idle: "transparent",
  active: "0 0 0 3px #f59e0b", // amber-500
  done: "0 0 0 2px #16a34a", // green-600
  error: "0 0 0 2px #dc2626", // red-600
}

function nodeStyle(kind: TraceNode["kind"], state: NodeState): React.CSSProperties {
  return {
    background: STATE_BG[state],
    color: "#0f172a",
    border: `1.5px solid ${KIND_COLORS[kind]}`,
    borderRadius: 8,
    padding: "6px 10px",
    fontSize: 12,
    fontWeight: 500,
    boxShadow: STATE_RING[state] !== "transparent" ? STATE_RING[state] : undefined,
    transition: "background 200ms, box-shadow 200ms",
  }
}

function buildNodes(states: NodeStates): Node[] {
  return TRACE_NODES.map((n) => ({
    id: n.id,
    position: { x: n.x, y: n.y },
    data: { label: n.label },
    style: nodeStyle(n.kind, states[n.id] ?? "idle"),
    draggable: false,
    selectable: false,
  }))
}

function buildEdges(states: NodeStates): Edge[] {
  return TRACE_EDGES.map((e) => {
    const upstream = states[e.source] ?? "idle"
    const downstream = states[e.target] ?? "idle"
    const lit = upstream !== "idle" && downstream !== "idle"
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      animated: downstream === "active",
      style: {
        stroke: lit ? "#16a34a" : "#cbd5e1",
        strokeWidth: lit ? 2 : 1,
      },
    }
  })
}

function selectLatestRoutingReason(): string {
  const msgs = useChatStore.getState().messages
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]
    if (m.kind === "streaming" || m.kind === "assistant") {
      return m.routingReason ?? ""
    }
  }
  return ""
}

export function AgentTrace() {
  const [open, setOpen] = useState(true)
  const states = useChatStore((s) => s.nodeStates)
  const reason = selectLatestRoutingReason()

  const nodes = useMemo(() => buildNodes(states), [states])
  const edges = useMemo(() => buildEdges(states), [states])

  return (
    <div className="border-border bg-card flex flex-col rounded-md border">
      <div className={open ? "flex items-center justify-between border-b px-3 py-2" : "flex items-center justify-between px-3 py-2"}>
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold tracking-wide uppercase">
            Agent trace
          </span>
          {/* toggle switch */}
          <button
            role="switch"
            aria-checked={open}
            onClick={() => setOpen((v) => !v)}
            className={`relative inline-flex h-4 w-8 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus-visible:outline-none ${open ? "bg-primary" : "bg-input"}`}
          >
            <span
              className={`pointer-events-none block h-3 w-3 rounded-full bg-white shadow-lg transition-transform ${open ? "translate-x-4" : "translate-x-0"}`}
            />
          </button>
        </div>
        <span
          className="text-muted-foreground truncate text-xs"
          title={reason}
          data-testid="agent-trace-reason"
        >
          {reason ? `Routing: ${reason}` : "Idle"}
        </span>
      </div>
      {open && (
        <div className="h-[320px] w-full" data-testid="agent-trace-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.05 }}
            panOnDrag={false}
            zoomOnScroll={false}
            zoomOnPinch={false}
            zoomOnDoubleClick={false}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable={false}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={16} size={1} color="#e2e8f0" />
          </ReactFlow>
        </div>
      )}
    </div>
  )
}

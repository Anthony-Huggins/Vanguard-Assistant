// Zustand store for the chat view. Models the streaming lifecycle of an
// assistant turn (pending → streaming → assistant) plus a separate
// pendingConfirmation slot for action-tool human-in-the-loop modals.

import { create } from "zustand"

import { TRACE_NODE_IDS } from "@/graph/topology"
import type { AgentName, Source } from "@/types/api"
import type { NodeState, NodeStates } from "@/types/trace"

export interface UserMessage {
  kind: "user"
  content: string
}

export interface PendingMessage {
  kind: "pending"
}

export interface ToolCallTrace {
  tool: string
  args: Record<string, unknown>
  status: "running" | "done"
  outputPreview?: string
}

export interface StreamingMessage {
  kind: "streaming"
  content: string
  agent: AgentName | null
  routingReason: string
  sources: Source[]
  tools: ToolCallTrace[]
}

export interface AssistantMessage {
  kind: "assistant"
  content: string
  sources: Source[]
  agent: AgentName
  routingReason: string
  tools: ToolCallTrace[]
}

export interface ErrorMessage {
  kind: "error"
  content: string
}

// Hydrated from the LangGraph checkpointer when switching threads.  The
// checkpointer stores raw BaseMessages, so we lose the structured agent /
// sources / tools metadata.  Renders as a plain card without a badge.
export interface HistoricalAssistantMessage {
  kind: "historical_assistant"
  content: string
}

export type ChatMessage =
  | UserMessage
  | PendingMessage
  | StreamingMessage
  | AssistantMessage
  | HistoricalAssistantMessage
  | ErrorMessage

export interface PendingConfirmation {
  question: string
  tool: string
  args: Record<string, unknown>
}

interface ChatState {
  threadId: string | null
  messages: ChatMessage[]
  pendingConfirmation: PendingConfirmation | null

  // M11 — graph-trace state. Keyed by node id from `TRACE_NODE_IDS`.
  // Reset to all-idle at the start of every turn.
  nodeStates: NodeStates

  // Supervisor toggle — sent with each chat request so the backend picks
  // the right compiled graph variant.
  supervisorEnabled: boolean
  setSupervisorEnabled: (v: boolean) => void

  setThreadId: (id: string) => void
  appendUser: (content: string) => void
  appendPending: () => void

  // Hydrate from GET /api/threads/{id} when switching threads.  Replaces
  // any in-flight messages with the rebuilt assistant/user history.
  setMessages: (messages: ChatMessage[]) => void

  // Streaming-specific mutations — applied by the event reducer.
  startStreaming: (agent: AgentName, routingReason: string) => void
  appendToken: (text: string) => void
  setSources: (sources: Source[]) => void
  pushToolStart: (tool: string, args: Record<string, unknown>) => void
  resolveToolEnd: (tool: string, outputPreview: string) => void

  // M11 — node-state transitions. The event reducer calls these directly
  // for both agent_* (coordinator + subagents) and tool_* events.
  setNodeState: (id: string, state: NodeState) => void
  resetNodeStates: () => void

  finalizeStreaming: () => void
  setError: (msg: string) => void
  setPendingConfirmation: (c: PendingConfirmation | null) => void

  reset: () => void
}

function freshNodeStates(): NodeStates {
  const out: NodeStates = {}
  for (const id of TRACE_NODE_IDS) out[id] = "idle"
  return out
}

function findLastIndex<T>(
  arr: T[],
  pred: (t: T) => boolean,
): number {
  for (let i = arr.length - 1; i >= 0; i--) if (pred(arr[i])) return i
  return -1
}

export const useChatStore = create<ChatState>((set) => ({
  threadId: null,
  messages: [],
  pendingConfirmation: null,
  nodeStates: freshNodeStates(),
  supervisorEnabled: false,

  setSupervisorEnabled: (v) => set({ supervisorEnabled: v }),

  setThreadId: (id) => set({ threadId: id }),

  appendUser: (content) =>
    set((s) => ({ messages: [...s.messages, { kind: "user", content }] })),

  appendPending: () =>
    set((s) => ({ messages: [...s.messages, { kind: "pending" }] })),

  setMessages: (messages) => set({ messages, pendingConfirmation: null }),

  startStreaming: (agent, routingReason) =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(next, (m) => m.kind === "pending")
      const stream: StreamingMessage = {
        kind: "streaming",
        content: "",
        agent,
        routingReason,
        sources: [],
        tools: [],
      }
      if (idx >= 0) next[idx] = stream
      else next.push(stream)
      return { messages: next }
    }),

  appendToken: (text) =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(next, (m) => m.kind === "streaming")
      if (idx < 0) return s
      const cur = next[idx] as StreamingMessage
      next[idx] = { ...cur, content: cur.content + text }
      return { messages: next }
    }),

  setSources: (sources) =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(
        next,
        (m) => m.kind === "streaming" || m.kind === "assistant",
      )
      if (idx < 0) return s
      const cur = next[idx]
      if (cur.kind === "streaming" || cur.kind === "assistant") {
        next[idx] = { ...cur, sources }
      }
      return { messages: next }
    }),

  pushToolStart: (tool, args) =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(next, (m) => m.kind === "streaming")
      if (idx < 0) return s
      const cur = next[idx] as StreamingMessage
      next[idx] = {
        ...cur,
        tools: [...cur.tools, { tool, args, status: "running" }],
      }
      return { messages: next }
    }),

  resolveToolEnd: (tool, outputPreview) =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(next, (m) => m.kind === "streaming")
      if (idx < 0) return s
      const cur = next[idx] as StreamingMessage
      // Mark the most recent running invocation of `tool` as done.
      const tools = [...cur.tools]
      const tIdx = findLastIndex(
        tools,
        (t) => t.tool === tool && t.status === "running",
      )
      if (tIdx >= 0) {
        tools[tIdx] = { ...tools[tIdx], status: "done", outputPreview }
      }
      next[idx] = { ...cur, tools }
      return { messages: next }
    }),

  finalizeStreaming: () =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(next, (m) => m.kind === "streaming")
      if (idx < 0) return s
      const cur = next[idx] as StreamingMessage
      // Demote pending agent → null is impossible at finalize time; if
      // it's still null (shouldn't happen) just drop the bubble.
      if (cur.agent == null) {
        next.splice(idx, 1)
      } else {
        const finalized: AssistantMessage = {
          kind: "assistant",
          content: cur.content,
          sources: cur.sources,
          agent: cur.agent,
          routingReason: cur.routingReason,
          tools: cur.tools,
        }
        next[idx] = finalized
      }
      return { messages: next }
    }),

  setError: (content) =>
    set((s) => {
      const next = [...s.messages]
      const idx = findLastIndex(
        next,
        (m) => m.kind === "pending" || m.kind === "streaming",
      )
      const err: ErrorMessage = { kind: "error", content }
      if (idx >= 0) next[idx] = err
      else next.push(err)
      return { messages: next }
    }),

  setPendingConfirmation: (c) => set({ pendingConfirmation: c }),

  setNodeState: (id, state) =>
    set((s) => {
      if (!(id in s.nodeStates)) return s
      if (s.nodeStates[id] === state) return s
      return { nodeStates: { ...s.nodeStates, [id]: state } }
    }),

  resetNodeStates: () => set({ nodeStates: freshNodeStates() }),

  reset: () =>
    set({
      messages: [],
      pendingConfirmation: null,
      nodeStates: freshNodeStates(),
    }),
}))

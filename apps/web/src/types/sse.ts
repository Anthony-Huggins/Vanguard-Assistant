// SSE event taxonomy emitted by /api/chat/stream and /api/chat/resume.
// Mirrors the dispatch table in apps/api/streaming.py — keep in sync.

import type { AgentName, Source } from "@/types/api"

// Node names that fire agent_start / agent_end. Wider than AgentName because
// the coordinator pulses on every turn for the M11 graph viz, even though
// it never owns the assistant's reply text.
export type TracedNodeName = AgentName | "coordinator"

export interface AgentStartEvent {
  type: "agent_start"
  data: { agent: TracedNodeName; routing_reason: string }
}

export interface AgentEndEvent {
  type: "agent_end"
  data: { agent: TracedNodeName }
}

export interface TokenEvent {
  type: "token"
  data: { text: string }
}

export interface ToolStartEvent {
  type: "tool_start"
  data: { tool: string; args: Record<string, unknown> }
}

export interface ToolEndEvent {
  type: "tool_end"
  data: { tool: string; output_preview: string }
}

export interface CitationEvent {
  type: "citation"
  data: { sources: Source[] }
}

export interface ConfirmationRequiredEvent {
  type: "confirmation_required"
  data: {
    question: string
    tool: string
    args: Record<string, unknown>
  }
}

export interface DoneEvent {
  type: "done"
  data: Record<string, never>
}

export interface ErrorEvent {
  type: "error"
  data: { message: string }
}

export type StreamEvent =
  | AgentStartEvent
  | AgentEndEvent
  | TokenEvent
  | ToolStartEvent
  | ToolEndEvent
  | CitationEvent
  | ConfirmationRequiredEvent
  | DoneEvent
  | ErrorEvent

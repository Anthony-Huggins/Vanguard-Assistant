// Single fold function from translated SSE events into the Zustand chat
// store.  Pulled out as its own module so the wiring stays testable
// without React/JSDOM in the loop — see eventReducer.test.ts.

import type { StreamEvent } from "@/types/sse"
import { useChatStore } from "@/lib/store"
import type { AgentName } from "@/types/api"

export type ChatStoreApi = typeof useChatStore

const SUBAGENT_NAMES: ReadonlySet<AgentName> = new Set([
  "support",
  "research",
  "action",
  "clarify",
])

function isSubagent(name: string): name is AgentName {
  return SUBAGENT_NAMES.has(name as AgentName)
}

export function applyStreamEvent(event: StreamEvent, api: ChatStoreApi = useChatStore) {
  const s = api.getState()
  switch (event.type) {
    case "agent_start":
      // M11 — pulse the node in the trace canvas. Coordinator pulses too
      // even though it never owns the assistant's reply.
      s.setNodeState(event.data.agent, "active")
      if (isSubagent(event.data.agent)) {
        s.startStreaming(event.data.agent, event.data.routing_reason)
      }
      return
    case "agent_end":
      s.setNodeState(event.data.agent, "done")
      return
    case "token":
      s.appendToken(event.data.text)
      return
    case "citation":
      s.setSources(event.data.sources)
      return
    case "tool_start":
      s.setNodeState(event.data.tool, "active")
      s.pushToolStart(event.data.tool, event.data.args)
      return
    case "tool_end":
      s.setNodeState(event.data.tool, "done")
      s.resolveToolEnd(event.data.tool, event.data.output_preview)
      return
    case "confirmation_required":
      // Finalize the in-progress streaming bubble (the action agent's
      // reasoning text) before opening the modal so it doesn't stay
      // stuck in "streaming" state while the user reads the dialog.
      s.finalizeStreaming()
      s.setPendingConfirmation(event.data)
      return
    case "done":
      s.finalizeStreaming()
      return
    case "error":
      // Mark the active node red if we can identify one; otherwise just
      // stash the error message.
      for (const [id, st] of Object.entries(s.nodeStates)) {
        if (st === "active") s.setNodeState(id, "error")
      }
      s.setError(event.data.message)
      return
  }
}

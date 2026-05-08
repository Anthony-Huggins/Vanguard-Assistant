// Smoke test the SSE → store fold.  Pinned shape: a happy-path support
// turn folds into a single assistant bubble with the right content,
// sources, and agent.  No React in the loop — this is pure store logic.

import { beforeEach, describe, expect, it } from "vitest"

import { applyStreamEvent } from "@/lib/eventReducer"
import { useChatStore } from "@/lib/store"
import type { AssistantMessage, StreamingMessage } from "@/lib/store"

describe("applyStreamEvent", () => {
  beforeEach(() => {
    useChatStore.getState().reset()
    useChatStore.getState().appendUser("test question")
    useChatStore.getState().appendPending()
  })

  it("folds a happy-path support turn into a finalized assistant message", () => {
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "support", routing_reason: "RAG topic" },
    })
    applyStreamEvent({ type: "token", data: { text: "Hello" } })
    applyStreamEvent({ type: "token", data: { text: " world" } })
    applyStreamEvent({
      type: "citation",
      data: {
        sources: [
          { source: "fees.pdf", chunk_id: "12", page: 3, score: 0.5 },
        ],
      },
    })
    applyStreamEvent({ type: "agent_end", data: { agent: "support" } })
    applyStreamEvent({ type: "done", data: {} })

    const msgs = useChatStore.getState().messages
    expect(msgs).toHaveLength(2)
    const last = msgs[msgs.length - 1] as AssistantMessage
    expect(last.kind).toBe("assistant")
    expect(last.content).toBe("Hello world")
    expect(last.agent).toBe("support")
    expect(last.routingReason).toBe("RAG topic")
    expect(last.sources).toHaveLength(1)
  })

  it("tracks tool_start / tool_end into the streaming bubble", () => {
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "research", routing_reason: "needs web" },
    })
    applyStreamEvent({
      type: "tool_start",
      data: { tool: "web_search", args: { q: "vfiax returns" } },
    })
    applyStreamEvent({
      type: "tool_end",
      data: { tool: "web_search", output_preview: "..." },
    })

    const msgs = useChatStore.getState().messages
    const stream = msgs[msgs.length - 1] as StreamingMessage
    expect(stream.kind).toBe("streaming")
    expect(stream.tools).toHaveLength(1)
    expect(stream.tools[0].tool).toBe("web_search")
    expect(stream.tools[0].status).toBe("done")
  })

  it("stores a pending confirmation instead of finalizing on confirmation_required", () => {
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "action", routing_reason: "user wants email" },
    })
    applyStreamEvent({ type: "token", data: { text: "Sending…" } })
    applyStreamEvent({
      type: "confirmation_required",
      data: {
        question: "Send email to x@y.com?",
        tool: "send_email",
        args: { recipient: "x@y.com" },
      },
    })

    const state = useChatStore.getState()
    expect(state.pendingConfirmation?.tool).toBe("send_email")
    // Streaming bubble is still in flight — not finalized yet.
    const last = state.messages[state.messages.length - 1] as StreamingMessage
    expect(last.kind).toBe("streaming")
  })

  it("converts the in-flight bubble to an error on error event", () => {
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "support", routing_reason: "" },
    })
    applyStreamEvent({ type: "error", data: { message: "boom" } })

    const msgs = useChatStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.kind).toBe("error")
    if (last.kind === "error") expect(last.content).toBe("boom")
  })

  // M11 — node-state tracking for the AgentTrace canvas.

  it("M11 — folds agent_start/agent_end into nodeStates for coordinator and subagent", () => {
    useChatStore.getState().resetNodeStates()
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "coordinator", routing_reason: "" },
    })
    let states = useChatStore.getState().nodeStates
    expect(states.coordinator).toBe("active")

    applyStreamEvent({
      type: "agent_end",
      data: { agent: "coordinator" },
    })
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "support", routing_reason: "RAG topic" },
    })
    applyStreamEvent({ type: "agent_end", data: { agent: "support" } })

    states = useChatStore.getState().nodeStates
    expect(states.coordinator).toBe("done")
    expect(states.support).toBe("done")
    expect(states.research).toBe("idle")
    expect(states.action).toBe("idle")
  })

  it("M11 — does NOT start a streaming bubble for coordinator's agent_start", () => {
    useChatStore.getState().resetNodeStates()
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "coordinator", routing_reason: "" },
    })
    // The pending bubble from beforeEach is still pending; coordinator
    // shouldn't have promoted it to streaming.
    const msgs = useChatStore.getState().messages
    const last = msgs[msgs.length - 1]
    expect(last.kind).toBe("pending")
  })

  it("M11 — folds tool_start/tool_end into nodeStates", () => {
    useChatStore.getState().resetNodeStates()
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "research", routing_reason: "" },
    })
    applyStreamEvent({
      type: "tool_start",
      data: { tool: "web_search", args: {} },
    })
    expect(useChatStore.getState().nodeStates.web_search).toBe("active")

    applyStreamEvent({
      type: "tool_end",
      data: { tool: "web_search", output_preview: "..." },
    })
    expect(useChatStore.getState().nodeStates.web_search).toBe("done")
  })

  it("M11 — error event flips active node(s) to error state", () => {
    useChatStore.getState().resetNodeStates()
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "action", routing_reason: "" },
    })
    applyStreamEvent({ type: "tool_start", data: { tool: "send_email", args: {} } })
    applyStreamEvent({ type: "error", data: { message: "smtp down" } })

    const states = useChatStore.getState().nodeStates
    expect(states.action).toBe("error")
    expect(states.send_email).toBe("error")
    // Idle nodes stay idle.
    expect(states.support).toBe("idle")
  })

  it("M11 — resetNodeStates clears all back to idle", () => {
    applyStreamEvent({
      type: "agent_start",
      data: { agent: "support", routing_reason: "" },
    })
    applyStreamEvent({ type: "agent_end", data: { agent: "support" } })
    expect(useChatStore.getState().nodeStates.support).toBe("done")

    useChatStore.getState().resetNodeStates()
    const states = useChatStore.getState().nodeStates
    for (const v of Object.values(states)) expect(v).toBe("idle")
  })
})

// M11 smoke test — verifies the AgentTrace renders, surfaces the routing
// reason from the most recent streaming/assistant message, and falls back
// to "Idle" when there's nothing in flight. React Flow itself is expensive
// to render under jsdom; we only assert the surrounding chrome we own.

import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it } from "vitest"

import { AgentTrace } from "@/components/AgentTrace"
import { useChatStore } from "@/lib/store"

// React Flow uses ResizeObserver internally; jsdom doesn't provide one.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
;(globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }).ResizeObserver =
  ResizeObserverStub

// Stub DOMMatrixReadOnly which @xyflow/react also needs for transforms.
if (typeof globalThis.DOMMatrixReadOnly === "undefined") {
  ;(globalThis as unknown as { DOMMatrixReadOnly: unknown }).DOMMatrixReadOnly = class {
    m22 = 1
    constructor() {}
  }
}

describe("AgentTrace", () => {
  beforeEach(() => {
    useChatStore.getState().reset()
  })

  it("shows 'Idle' when no streaming/assistant message exists", () => {
    render(<AgentTrace />)
    expect(screen.getByTestId("agent-trace-reason")).toHaveTextContent("Idle")
  })

  it("surfaces the routing reason from the most recent streaming message", () => {
    useChatStore.getState().appendUser("hi")
    useChatStore.getState().appendPending()
    useChatStore.getState().startStreaming("support", "user asked about fees")

    render(<AgentTrace />)
    const reason = screen.getByTestId("agent-trace-reason")
    expect(reason).toHaveTextContent("Routing: user asked about fees")
  })

  it("renders the canvas container with expected test id", () => {
    render(<AgentTrace />)
    expect(screen.getByTestId("agent-trace-canvas")).toBeInTheDocument()
  })

  it("does not throw when the store has node states across all four states", () => {
    useChatStore.getState().setNodeState("coordinator", "done")
    useChatStore.getState().setNodeState("support", "active")
    useChatStore.getState().setNodeState("send_email", "error")
    // Render shouldn't throw — the styling map covers all four NodeStates.
    expect(() => render(<AgentTrace />)).not.toThrow()
  })
})

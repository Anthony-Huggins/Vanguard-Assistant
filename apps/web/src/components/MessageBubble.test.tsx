import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { MessageBubble } from "@/components/MessageBubble"
import type { AssistantMessage, UserMessage } from "@/lib/store"

describe("MessageBubble", () => {
  it("renders the user's message text", () => {
    const m: UserMessage = { kind: "user", content: "What does Vanguard charge?" }
    render(<MessageBubble message={m} />)
    expect(screen.getByText("What does Vanguard charge?")).toBeInTheDocument()
  })

  it("renders an assistant reply with sources and an agent badge", () => {
    const m: AssistantMessage = {
      kind: "assistant",
      content: "Vanguard's expense ratios average around 0.04%.",
      sources: [
        { source: "vanguard-fees.pdf", chunk_id: "12", page: 3, score: 0.61 },
      ],
      agent: "support",
      routingReason: "user asked about fees",
      tools: [],
    }
    render(<MessageBubble message={m} />)
    expect(
      screen.getByText("Vanguard's expense ratios average around 0.04%."),
    ).toBeInTheDocument()
    expect(screen.getByText("Support Agent")).toBeInTheDocument()
    expect(screen.getByText("vanguard-fees.pdf")).toBeInTheDocument()
    expect(screen.getByText(/chunk_id: 12/)).toBeInTheDocument()
  })
})

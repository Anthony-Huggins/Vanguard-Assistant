import { describe, expect, it } from "vitest"

import { hydrateMessages } from "@/lib/threadHydration"

describe("hydrateMessages", () => {
  it("converts human/ai roles and drops tool/system", () => {
    const out = hydrateMessages([
      { role: "human", content: "What does Vanguard charge?" },
      {
        role: "ai",
        content:
          "[Support Agent] Around 0.04%.\n\nSources: [fees.pdf, chunk_id: 12]",
      },
      { role: "tool", content: "internal" },
      { role: "system", content: "internal" },
    ])

    expect(out).toHaveLength(2)
    expect(out[0].kind).toBe("user")
    if (out[0].kind === "user") {
      expect(out[0].content).toBe("What does Vanguard charge?")
    }
    expect(out[1].kind).toBe("historical_assistant")
    if (out[1].kind === "historical_assistant") {
      expect(out[1].content).toBe("Around 0.04%.")
    }
  })
})

// Convert raw ThreadMessages from GET /api/threads/{id} into the store's
// ChatMessage shapes.  The checkpointer only stores BaseMessages, so we
// lose structured metadata (agent, sources, tools) on rehydration —
// those messages render as `historical_assistant` (a plain card with no
// badge or sources block).

import type { ChatMessage } from "@/lib/store"
import type { ThreadMessage } from "@/types/api"

// Mirror of the regexes the API uses to decorate Support agent replies.
// Keep in sync with apps/api/routers/chat.py.
const BADGE_RE = /^\[[^\]]+\](?:\s*\(\d+\s+tool\s+calls?\))?\s*/
const SOURCES_RE = /\n\nSources:\s*(?:\[[^\]]+\]\s*)+\s*$/

function stripDecorations(text: string): string {
  return text.replace(BADGE_RE, "").replace(SOURCES_RE, "").trimEnd()
}

export function hydrateMessages(raw: ThreadMessage[]): ChatMessage[] {
  const out: ChatMessage[] = []
  for (const m of raw) {
    if (m.role === "human") {
      out.push({ kind: "user", content: m.content })
    } else if (m.role === "ai") {
      out.push({
        kind: "historical_assistant",
        content: stripDecorations(m.content),
      })
    }
    // tool / system messages are internal — drop them from the UI.
  }
  return out
}

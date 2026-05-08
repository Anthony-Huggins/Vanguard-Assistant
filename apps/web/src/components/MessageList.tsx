import { useEffect, useRef } from "react"

import { MessageBubble } from "@/components/MessageBubble"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useChatStore } from "@/lib/store"

export function MessageList() {
  const messages = useChatStore((s) => s.messages)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages.length])

  if (messages.length === 0) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center text-sm">
        Ask about Vanguard funds, fees, or open a support ticket.
      </div>
    )
  }

  return (
    <ScrollArea className="h-full pr-3">
      <div className="flex flex-col gap-3 py-2">
        {messages.map((m, i) => (
          <MessageBubble key={i} message={m} />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  )
}

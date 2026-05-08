import { useEffect, useRef, useState } from "react"

import { AgentTrace } from "@/components/AgentTrace"
import { ConfirmationDialog } from "@/components/ConfirmationDialog"
import { MessageList } from "@/components/MessageList"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { getThread } from "@/lib/api"
import { applyStreamEvent } from "@/lib/eventReducer"
import { resumeChatStream, startChatStream } from "@/lib/sseClient"
import { hydrateMessages } from "@/lib/threadHydration"
import { useChatStore } from "@/lib/store"

interface ChatProps {
  threadId: string
  onTurnDone?: () => void
}

export function Chat({ threadId, onTurnDone }: ChatProps) {
  const [draft, setDraft] = useState("")
  const [busy, setBusy] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const setMessages = useChatStore((s) => s.setMessages)
  const appendUser = useChatStore((s) => s.appendUser)
  const appendPending = useChatStore((s) => s.appendPending)
  const setError = useChatStore((s) => s.setError)
  const pendingConfirmation = useChatStore((s) => s.pendingConfirmation)
  const setPendingConfirmation = useChatStore((s) => s.setPendingConfirmation)
  const resetNodeStates = useChatStore((s) => s.resetNodeStates)

  // Hydrate from the API when the thread id changes (or on first mount).
  // 404 = brand-new thread → start empty.
  useEffect(() => {
    let cancelled = false
    setMessages([])
    getThread(threadId)
      .then((detail) => {
        if (cancelled) return
        setMessages(hydrateMessages(detail.messages))
      })
      .catch(() => {
        if (cancelled) return
        // 404s and network errors both leave the chat empty; the next turn
        // will create the thread on the backend.
        setMessages([])
      })
    return () => {
      cancelled = true
    }
  }, [threadId, setMessages])

  // Cancel any in-flight stream when the component unmounts (or thread switches).
  useEffect(() => () => abortRef.current?.abort(), [])

  async function send(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    const message = draft.trim()
    if (!message || busy) return

    setDraft("")
    setBusy(true)
    appendUser(message)
    appendPending()
    resetNodeStates()

    const ctrl = new AbortController()
    abortRef.current?.abort()
    abortRef.current = ctrl

    try {
      await startChatStream(
        { threadId, message },
        {
          onEvent: applyStreamEvent,
          onError: (err) => {
            const msg = err instanceof Error ? err.message : String(err)
            setError(msg)
          },
          signal: ctrl.signal,
        },
      )
    } finally {
      setBusy(false)
      onTurnDone?.()
    }
  }

  async function answerConfirmation(answer: "yes" | "no") {
    setPendingConfirmation(null)
    setBusy(true)
    appendPending()
    resetNodeStates()

    const ctrl = new AbortController()
    abortRef.current?.abort()
    abortRef.current = ctrl

    try {
      await resumeChatStream(
        { threadId, answer },
        {
          onEvent: applyStreamEvent,
          onError: (err) => {
            const msg = err instanceof Error ? err.message : String(err)
            setError(msg)
          },
          signal: ctrl.signal,
        },
      )
    } finally {
      setBusy(false)
      onTurnDone?.()
    }
  }

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="min-h-0 flex-1">
        <MessageList />
      </div>
      <AgentTrace />
      <form onSubmit={send} className="flex gap-2">
        <Input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type a message…"
          disabled={busy}
          autoFocus
        />
        <Button type="submit" disabled={busy || draft.trim().length === 0}>
          Send
        </Button>
      </form>
      <ConfirmationDialog
        pending={pendingConfirmation}
        onAnswer={answerConfirmation}
      />
    </div>
  )
}

// Thin wrapper around @microsoft/fetch-event-source.  Native EventSource is
// GET-only; we need POST + headers, plus the option to abort cleanly when
// the user navigates away.

import { fetchEventSource } from "@microsoft/fetch-event-source"

import type { StreamEvent } from "@/types/sse"

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "")

interface StreamCallbacks {
  onEvent: (event: StreamEvent) => void
  onError?: (error: unknown) => void
  signal?: AbortSignal
}

async function streamPost(
  path: string,
  body: unknown,
  { onEvent, onError, signal }: StreamCallbacks,
): Promise<void> {
  await fetchEventSource(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
    // Don't auto-reconnect — chat turns are not idempotent.
    openWhenHidden: true,
    async onopen(res) {
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`
        try {
          const j = await res.json()
          if (j?.detail) detail = String(j.detail)
        } catch {
          /* not json */
        }
        throw new Error(detail)
      }
    },
    onmessage(msg) {
      if (!msg.event) return
      try {
        const data = msg.data ? JSON.parse(msg.data) : {}
        onEvent({ type: msg.event, data } as StreamEvent)
      } catch (e) {
        onError?.(e)
      }
    },
    onerror(err) {
      // Throwing aborts the underlying retry loop — without this, the
      // library will retry on connection errors which we don't want.
      throw err
    },
    onclose() {
      // Default close behavior is fine; nothing to do.
    },
  }).catch((err) => {
    if ((err as { name?: string }).name === "AbortError") return
    onError?.(err)
  })
}

export interface StartStreamArgs {
  threadId: string
  message: string
  supervisorEnabled?: boolean
}

export function startChatStream(
  args: StartStreamArgs,
  callbacks: StreamCallbacks,
): Promise<void> {
  return streamPost(
    "/api/chat/stream",
    {
      thread_id: args.threadId,
      message: args.message,
      supervisor_enabled: args.supervisorEnabled ?? false,
    },
    callbacks,
  )
}

export interface ResumeStreamArgs {
  threadId: string
  answer: "yes" | "no"
  supervisorEnabled?: boolean
}

export function resumeChatStream(
  args: ResumeStreamArgs,
  callbacks: StreamCallbacks,
): Promise<void> {
  return streamPost(
    "/api/chat/resume",
    {
      thread_id: args.threadId,
      answer: args.answer,
      supervisor_enabled: args.supervisorEnabled ?? false,
    },
    callbacks,
  )
}

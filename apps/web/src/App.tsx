// Top-level shell: thread sidebar (M9) + chat pane.  ``threadId`` is
// always reflected in ``?thread=`` so a reload preserves the active
// conversation; clicking a sidebar item or "+ New thread" updates both.

import { useCallback, useEffect, useState } from "react"

import { Chat } from "@/components/Chat"
import { EvalDashboard } from "@/components/EvalDashboard"
import { ThreadSidebar } from "@/components/ThreadSidebar"
import { useChatStore } from "@/lib/store"
import type { ThreadSummary } from "@/types/api"

function readThreadParam(): string | null {
  return new URL(window.location.href).searchParams.get("thread")
}

function readViewParam(): string {
  return new URL(window.location.href).searchParams.get("view") ?? "chat"
}

function writeThreadParam(threadId: string) {
  const url = new URL(window.location.href)
  url.searchParams.set("thread", threadId)
  window.history.replaceState({}, "", url.toString())
}

function writeViewParam(view: string) {
  const url = new URL(window.location.href)
  url.searchParams.set("view", view)
  window.history.replaceState({}, "", url.toString())
}

function App() {
  const [threadId, setThreadId] = useState<string | null>(null)
  const [threads, setThreads] = useState<ThreadSummary[]>([])
  const [refreshKey, setRefreshKey] = useState(0)
  const [view, setView] = useState<string>(readViewParam())
  const setStoreThreadId = useChatStore((s) => s.setThreadId)

  // First mount: read or generate the thread id and pin it to the URL.
  useEffect(() => {
    const existing = readThreadParam()
    const id = existing ?? crypto.randomUUID()
    if (!existing) writeThreadParam(id)
    setThreadId(id)
    setStoreThreadId(id)
  }, [setStoreThreadId])

  const handleSelect = useCallback(
    (id: string) => {
      writeThreadParam(id)
      setThreadId(id)
      setStoreThreadId(id)
    },
    [setStoreThreadId],
  )

  const handleNewThread = useCallback(() => {
    handleSelect(crypto.randomUUID())
  }, [handleSelect])

  const refreshSidebar = useCallback(() => {
    setRefreshKey((k) => k + 1)
  }, [])

  if (!threadId) return null

  return (
    <div className="bg-background text-foreground flex h-screen flex-col">
      <header className="flex shrink-0 items-center justify-between border-b px-6 py-3">
        <h1 className="text-base font-semibold">Vanguard Assistant</h1>
        <div className="flex items-center gap-4">
          <nav className="flex gap-2">
            <button
              onClick={() => { setView("chat"); writeViewParam("chat") }}
              className={`rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                view === "chat"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              Chat
            </button>
            <button
              onClick={() => { setView("evals"); writeViewParam("evals") }}
              className={`flex items-center gap-1.5 rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                view === "evals"
                  ? "bg-primary text-primary-foreground"
                  : "border border-border text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>
              </svg>
              RAG Evals
            </button>
          </nav>
          <span className="text-muted-foreground font-mono text-xs">
            thread {threadId.slice(0, 8)}
          </span>
        </div>
      </header>
      <div className="flex min-h-0 flex-1">
        {view === "evals" ? (
          <main className="min-h-0 flex-1 overflow-y-auto">
            <EvalDashboard />
          </main>
        ) : (
          <>
            <ThreadSidebar
              activeThreadId={threadId}
              onSelect={handleSelect}
              onNewThread={handleNewThread}
              threads={threads}
              setThreads={setThreads}
              refreshKey={refreshKey}
            />
            <main className="flex min-w-0 flex-1 flex-col px-6 py-4">
              <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col">
                {/* key={threadId} forces remount on switch — clean slate for the
                    store hydration + abort of any in-flight stream. */}
                <Chat
                  key={threadId}
                  threadId={threadId}
                  onTurnDone={refreshSidebar}
                />
              </div>
            </main>
          </>
        )}
      </div>
    </div>
  )
}

export default App

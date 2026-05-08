// Sidebar listing past threads (M9).  Pulls from GET /api/threads on mount
// and after each turn finishes (the parent calls `refresh()` on `done`).
// Clicking a thread sets `?thread=...` in the URL — App.tsx watches that
// param and re-renders the chat with the new id.

import { useEffect } from "react"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { listThreads } from "@/lib/api"
import { cn } from "@/lib/utils"
import type { ThreadSummary } from "@/types/api"

interface ThreadSidebarProps {
  activeThreadId: string
  onSelect: (threadId: string) => void
  onNewThread: () => void
  threads: ThreadSummary[]
  setThreads: (next: ThreadSummary[]) => void
  refreshKey: number
}

function relativeTime(iso: string): string {
  const now = Date.now()
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ""
  const diff = Math.max(0, now - then)
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (diff < minute) return "just now"
  if (diff < hour) return `${Math.floor(diff / minute)}m ago`
  if (diff < day) return `${Math.floor(diff / hour)}h ago`
  return `${Math.floor(diff / day)}d ago`
}

export function ThreadSidebar({
  activeThreadId,
  onSelect,
  onNewThread,
  threads,
  setThreads,
  refreshKey,
}: ThreadSidebarProps) {
  useEffect(() => {
    let cancelled = false
    listThreads()
      .then((rows) => {
        if (!cancelled) setThreads(rows)
      })
      .catch(() => {
        // Sidebar errors are silent — failing to list threads shouldn't
        // block the chat itself.  Refreshing the page will retry.
      })
    return () => {
      cancelled = true
    }
  }, [refreshKey, setThreads])

  return (
    <aside className="bg-muted/30 flex h-full w-64 flex-col border-r">
      <div className="border-b p-3">
        <Button
          variant="default"
          size="sm"
          className="w-full"
          onClick={onNewThread}
        >
          + New thread
        </Button>
      </div>
      <ScrollArea className="flex-1">
        {threads.length === 0 ? (
          <div className="text-muted-foreground p-3 text-xs">
            No threads yet — send a message to start one.
          </div>
        ) : (
          <ul className="flex flex-col">
            {threads.map((t, i) => {
              const isActive = t.thread_id === activeThreadId
              return (
                <li key={t.thread_id} className="flex flex-col">
                  <button
                    type="button"
                    onClick={() => onSelect(t.thread_id)}
                    className={cn(
                      "hover:bg-accent flex flex-col items-start gap-0.5 px-3 py-2 text-left transition-colors",
                      isActive && "bg-accent",
                    )}
                  >
                    <span className="line-clamp-1 text-sm">
                      {t.display_name ?? t.thread_id.slice(0, 8)}
                    </span>
                    <span className="text-muted-foreground font-mono text-[10px]">
                      {relativeTime(t.last_updated)} · {t.message_count} msg
                    </span>
                  </button>
                  {i < threads.length - 1 && <Separator />}
                </li>
              )
            })}
          </ul>
        )}
      </ScrollArea>
    </aside>
  )
}

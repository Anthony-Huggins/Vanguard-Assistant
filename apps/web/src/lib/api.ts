// Thin fetch wrapper. Vite's dev proxy forwards `/api/*` → http://localhost:8000
// (see vite.config.ts), and prod will use the same relative path against the
// API container. Replaced in M8 by an SSE client for /api/chat/stream.

import type {
  ChatRequest,
  ChatResponse,
  ThreadDetail,
  ThreadSummary,
} from "@/types/api"

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "")

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* body wasn't json — keep status text */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export function postChat(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(body),
  })
}

export function listThreads(): Promise<ThreadSummary[]> {
  return request<ThreadSummary[]>("/api/threads")
}

export function getThread(threadId: string): Promise<ThreadDetail> {
  return request<ThreadDetail>(`/api/threads/${encodeURIComponent(threadId)}`)
}

export function getLatestEval<T = unknown>(): Promise<T> {
  return request<T>("/api/evals/latest")
}

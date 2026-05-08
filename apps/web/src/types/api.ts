// Hand-written mirror of `apps/api/schemas.py`. Keep in sync — when the
// FastAPI Pydantic models change, update these too. (The plan deliberately
// skips an OpenAPI codegen step in M7; volume of types is small.)

export type AgentName = "support" | "research" | "action" | "clarify"

export interface Source {
  source: string
  chunk_id: string
  page: number | null
  score: number
}

export interface ChatRequest {
  thread_id: string
  message: string
}

export interface ChatResponse {
  reply: string
  sources: Source[]
  agent_used: AgentName
  routing_reason: string
}

export interface ThreadSummary {
  thread_id: string
  display_name: string | null
  last_updated: string
  message_count: number
}

export interface ThreadMessage {
  role: string
  content: string
}

export interface ThreadDetail {
  thread_id: string
  messages: ThreadMessage[]
}

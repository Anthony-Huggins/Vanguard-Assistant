import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { SourcesList } from "@/components/SourcesList"
import { cn } from "@/lib/utils"
import type { AgentName } from "@/types/api"
import type { ChatMessage, ToolCallTrace } from "@/lib/store"

interface MessageBubbleProps {
  message: ChatMessage
}

const AGENT_LABEL: Record<AgentName, string> = {
  support: "Support Agent",
  research: "Research Agent",
  action: "Action Agent",
  clarify: "Clarify Agent",
}

function ToolRibbon({ tools }: { tools: ToolCallTrace[] }) {
  if (tools.length === 0) return null
  return (
    <div className="mb-2 flex flex-wrap gap-1.5">
      {tools.map((t, i) => (
        <Badge
          key={`${t.tool}-${i}`}
          variant={t.status === "running" ? "default" : "secondary"}
          className="font-mono text-xs"
          title={t.outputPreview ?? ""}
        >
          {t.status === "running" ? "▸ " : "✓ "}
          {t.tool}
        </Badge>
      ))}
    </div>
  )
}

export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.kind === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-primary text-primary-foreground max-w-[80%] rounded-2xl rounded-br-sm px-4 py-2.5 whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.kind === "pending") {
    return (
      <div className="flex justify-start">
        <Card className="bg-muted/40 max-w-[85%] py-2 shadow-none">
          <CardContent className="px-4 py-1">
            <span className="text-muted-foreground text-sm">thinking…</span>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (message.kind === "error") {
    return (
      <div className="flex justify-start">
        <Card
          className={cn(
            "border-destructive/40 bg-destructive/5 max-w-[85%] py-2 shadow-none",
          )}
        >
          <CardContent className="px-4 py-1">
            <div className="text-destructive text-sm font-medium">
              Request failed
            </div>
            <div className="text-muted-foreground mt-1 text-sm">
              {message.content}
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (message.kind === "streaming") {
    return (
      <div className="flex justify-start">
        <Card className="bg-card max-w-[85%] py-3 shadow-sm">
          <CardContent className="px-4 py-0">
            <ToolRibbon tools={message.tools} />
            <div className="text-foreground whitespace-pre-wrap">
              {message.content}
              <span className="ml-0.5 inline-block h-4 w-1 animate-pulse bg-current align-middle" />
            </div>
            <SourcesList sources={message.sources} />
            {message.agent && (
              <div className="text-muted-foreground mt-3 flex items-center gap-2 text-xs">
                <Badge variant="outline" className="font-medium">
                  {AGENT_LABEL[message.agent]}
                </Badge>
                {message.routingReason && (
                  <span className="italic">"{message.routingReason}"</span>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    )
  }

  if (message.kind === "historical_assistant") {
    return (
      <div className="flex justify-start">
        <Card className="bg-card max-w-[85%] py-3 shadow-sm">
          <CardContent className="px-4 py-0">
            <div className="text-foreground whitespace-pre-wrap">
              {message.content}
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // assistant (finalized)
  return (
    <div className="flex justify-start">
      <Card className="bg-card max-w-[85%] py-3 shadow-sm">
        <CardContent className="px-4 py-0">
          <ToolRibbon tools={message.tools} />
          <div className="text-foreground whitespace-pre-wrap">
            {message.content}
          </div>
          <SourcesList sources={message.sources} />
          <div className="text-muted-foreground mt-3 flex items-center gap-2 text-xs">
            <Badge variant="outline" className="font-medium">
              {AGENT_LABEL[message.agent]}
            </Badge>
            {message.routingReason && (
              <span className="italic">"{message.routingReason}"</span>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

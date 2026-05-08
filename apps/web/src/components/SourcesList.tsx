import { Badge } from "@/components/ui/badge"
import type { Source } from "@/types/api"

interface SourcesListProps {
  sources: Source[]
}

export function SourcesList({ sources }: SourcesListProps) {
  if (sources.length === 0) return null
  return (
    <div className="mt-3 flex flex-col gap-1.5">
      <div className="text-muted-foreground text-xs uppercase tracking-wide">
        Sources
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((s, i) => (
          <Badge
            key={`${s.source}-${s.chunk_id}-${i}`}
            variant="secondary"
            className="font-mono text-xs"
            title={`score ${s.score.toFixed(2)}${
              s.page != null ? ` · page ${s.page}` : ""
            }`}
          >
            {s.source}
            <span className="text-muted-foreground ml-1.5">
              chunk_id: {s.chunk_id}
            </span>
          </Badge>
        ))}
      </div>
    </div>
  )
}

// M12 — Ragas eval results dashboard.
// Fetches /api/evals/latest and renders a summary header + per-question table.
// Accessed via ?view=evals in the URL.

import { useEffect, useState } from "react"

import { getLatestEval } from "@/lib/api"

interface EvalRow {
  question: string
  ground_truth: string
  answer: string
  sources: string[]
  expected_sources: string[]
  faithfulness: number
  answer_relevancy: number
  context_precision: number
  retrieval_precision: number
  retrieval_recall: number
  citation_hit: boolean
}

interface EvalRun {
  timestamp: string
  summary: {
    faithfulness: number
    answer_relevancy: number
    context_precision: number
    retrieval_precision: number
    retrieval_recall: number
    citation_hit_rate: number
  }
  rows: EvalRow[]
}

const METRIC_DESCRIPTIONS: Record<string, string> = {
  Faithfulness:
    "Is the answer grounded in the retrieved context? Measures whether the response makes claims supported by the source chunks (LLM-judged).",
  "Answer Relevancy":
    "Does the answer actually address the question asked? Penalises off-topic or incomplete responses (LLM-judged).",
  "Context Precision":
    "Are the most relevant chunks ranked at the top of the retrieved list? High precision means the best evidence came first.",
  "Retrieval Precision":
    "What fraction of the retrieved chunks came from an expected source document? Measures how much noise is in the retrieval.",
  "Retrieval Recall":
    "What fraction of the expected source documents were retrieved at all? Measures whether the retriever found the right docs.",
  "Citation Hit":
    "Did at least one expected source document appear in the retrieved chunks? Binary pass/fail per question.",
}

function InfoTooltip({ metric }: { metric: string }) {
  const desc = METRIC_DESCRIPTIONS[metric]
  if (!desc) return null
  return (
    <span className="group relative ml-1 inline-block align-middle">
      <span className="text-muted-foreground/60 hover:text-muted-foreground flex h-3.5 w-3.5 cursor-default items-center justify-center rounded-full border border-current text-[9px] font-bold leading-none">
        i
      </span>
      <span className="pointer-events-none absolute top-full left-1/2 z-50 mt-1.5 w-56 -translate-x-1/2 rounded bg-gray-900 px-2.5 py-2 text-xs text-gray-50 shadow-lg opacity-0 transition-opacity group-hover:opacity-100">
        {desc}
      </span>
    </span>
  )
}

function Score({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 80 ? "text-green-600" : pct >= 60 ? "text-amber-500" : "text-red-500"
  return <span className={`font-mono font-semibold ${color}`}>{pct}%</span>
}

export function EvalDashboard() {
  const [run, setRun] = useState<EvalRun | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getLatestEval<EvalRun>()
      .then(setRun)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return <div className="text-muted-foreground p-8 text-sm">Loading eval results…</div>
  }

  if (error) {
    return (
      <div className="p-8">
        <p className="text-destructive text-sm font-medium">Failed to load eval results</p>
        <p className="text-muted-foreground mt-1 text-xs">{error}</p>
        <p className="text-muted-foreground mt-3 text-xs">
          Run{" "}
          <code className="bg-muted rounded px-1 py-0.5">
            ./.venv/Scripts/python.exe packages/eval/run_eval.py
          </code>{" "}
          to generate results.
        </p>
      </div>
    )
  }

  if (!run) return null

  const { summary, rows, timestamp } = run
  const ts = timestamp.replace(/T/, " ").replace(/Z$/, " UTC")

  return (
    <div className="flex flex-col gap-6 p-6">
      <div>
        <h2 className="text-lg font-semibold">Ragas Eval Results</h2>
        <p className="text-muted-foreground text-xs">Run at {ts} · {rows.length} questions</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[
          { label: "Faithfulness", value: summary.faithfulness },
          { label: "Answer Relevancy", value: summary.answer_relevancy },
          { label: "Context Precision", value: summary.context_precision },
          { label: "Retrieval Precision", value: summary.retrieval_precision },
          { label: "Retrieval Recall", value: summary.retrieval_recall },
          { label: "Citation Hit", value: summary.citation_hit_rate },
        ].map(({ label, value }) => (
          <div key={label} className="border-border bg-card rounded-md border p-3">
            <p className="text-muted-foreground flex items-center gap-1 text-xs">
              {label}
              <InfoTooltip metric={label} />
            </p>
            <p className="mt-1 text-2xl">
              <Score value={value} />
            </p>
          </div>
        ))}
      </div>

      {/* Per-question table */}
      <div className="border-border overflow-x-auto rounded-md border">
        <table className="w-full text-xs">
          <thead className="overflow-visible">
            <tr className="bg-muted text-muted-foreground border-b text-left overflow-visible">
              <th className="px-3 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">Question</th>
              {(["Faithfulness", "Answer Relevancy", "Context Precision", "Retrieval Precision", "Retrieval Recall", "Citation Hit"] as const).map((label) => (
                <th key={label} className="px-3 py-2 font-medium whitespace-nowrap">
                  {label}
                  <InfoTooltip metric={label} />
                </th>
              ))}
              <th className="px-3 py-2 font-medium">Sources Retrieved</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-b last:border-0 hover:bg-muted/30 transition-colors"
              >
                <td className="text-muted-foreground px-3 py-2">{i + 1}</td>
                <td className="max-w-[280px] px-3 py-2">
                  <span title={row.question} className="line-clamp-2">
                    {row.question}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <Score value={row.faithfulness} />
                </td>
                <td className="px-3 py-2">
                  <Score value={row.answer_relevancy} />
                </td>
                <td className="px-3 py-2">
                  <Score value={row.context_precision} />
                </td>
                <td className="px-3 py-2">
                  <Score value={row.retrieval_precision} />
                </td>
                <td className="px-3 py-2">
                  <Score value={row.retrieval_recall} />
                </td>
                <td className="px-3 py-2">
                  {row.citation_hit ? (
                    <span className="text-green-600 font-bold">✓</span>
                  ) : (
                    <span className="text-red-500 font-bold">✗</span>
                  )}
                </td>
                <td className="text-muted-foreground max-w-[200px] truncate px-3 py-2">
                  {row.sources.join(", ") || "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

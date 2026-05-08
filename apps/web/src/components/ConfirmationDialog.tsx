// Modal that surfaces an Action-agent confirmation question.  The
// underlying `Dialog` is `@base-ui/react`-controlled — passing `open`
// drives the overlay; the user's choice is reported via `onAnswer`.

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import type { PendingConfirmation } from "@/lib/store"

interface ConfirmationDialogProps {
  pending: PendingConfirmation | null
  onAnswer: (answer: "yes" | "no") => void
}

export function ConfirmationDialog({
  pending,
  onAnswer,
}: ConfirmationDialogProps) {
  return (
    <Dialog open={pending != null} onOpenChange={(open) => {
      if (!open && pending != null) onAnswer("no")
    }}>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Confirm action</DialogTitle>
          <DialogDescription>
            {pending?.question ?? ""}
          </DialogDescription>
        </DialogHeader>
        {pending != null && (
          <div className="bg-muted/40 -mx-1 rounded-md px-3 py-2 font-mono text-xs">
            <div className="text-muted-foreground">tool</div>
            <div className="font-medium">{pending.tool}</div>
            {Object.keys(pending.args).length > 0 && (
              <>
                <div className="text-muted-foreground mt-2">args</div>
                <pre className="overflow-x-auto whitespace-pre-wrap break-all">
                  {JSON.stringify(pending.args, null, 2)}
                </pre>
              </>
            )}
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onAnswer("no")}>
            Deny
          </Button>
          <Button onClick={() => onAnswer("yes")}>Confirm</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

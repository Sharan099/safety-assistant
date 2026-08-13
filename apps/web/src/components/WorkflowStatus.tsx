// UI_UX_DESIGN_BRIEF_LEVEL3.md Section 5: "Use a visible workflow... The
// engineer must always know: 1. where they are; 2. what has been checked;
// 3. what remains; 4. what evidence exists." Not a wizard — every step
// stays independently visible regardless of order completed.

export type StepState = "DONE" | "PENDING" | "NOT_APPLICABLE";

export interface WorkflowStep {
  label: string;
  state: StepState;
}

const SYMBOL: Record<StepState, string> = { DONE: "✓", PENDING: "○", NOT_APPLICABLE: "–" };
const CLASS: Record<StepState, string> = {
  DONE: "text-emerald-400",
  PENDING: "text-neutral-600",
  NOT_APPLICABLE: "text-neutral-700",
};

export function WorkflowStatus({ steps }: { steps: WorkflowStep[] }) {
  return (
    <ol className="space-y-1">
      {steps.map((step, i) => (
        <li key={step.label} className={`flex items-center gap-2 text-xs ${CLASS[step.state]}`}>
          <span className="w-4 text-right font-mono text-[10px] text-neutral-700">{String(i + 1).padStart(2, "0")}</span>
          <span aria-hidden>{SYMBOL[step.state]}</span>
          <span className={step.state === "DONE" ? "text-neutral-300" : ""}>{step.label}</span>
        </li>
      ))}
    </ol>
  );
}

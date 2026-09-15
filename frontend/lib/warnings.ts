/** Pipeline warnings are written for logs; the engineer sees what it means and what to do. */
export function humanizeWarning(raw: string): string {
  if (raw.startsWith("generation unavailable (LLMRateLimited)")) {
    return "The answer model is busy (rate limited), so only the retrieved evidence is shown. Ask again in a moment.";
  }
  if (raw.startsWith("generation unavailable")) {
    return "The answer model did not respond, so only the retrieved evidence is shown. Ask again in a moment.";
  }
  if (raw.startsWith("not sent to the answer model")) {
    return `Withheld from the answer model (data-class policy) — shown as evidence only: ${raw.split(": ").slice(1).join(": ")}`;
  }
  if (raw.startsWith("prompt-injection pattern detected")) {
    return "Instructions embedded in the question were ignored; only the regulatory content was answered.";
  }
  if (raw.startsWith("evidence-only") || raw.includes("evidence-only mode")) {
    return "Only the retrieved evidence is shown for this turn.";
  }
  if (raw.startsWith("contains a value derived")) {
    return "Contains a value computed from the cited limits (working shown) — verify before use.";
  }
  if (raw.startsWith("weak evidence")) {
    return "No source matched this question closely; check the evidence before relying on the answer.";
  }
  if (raw.startsWith("evidence spans several versions")) {
    return raw.replace("evidence spans several versions of", "The evidence spans several versions of");
  }
  if (raw.includes("claim(s) removed")) {
    return "Part of the draft could not be verified against the evidence and was removed.";
  }
  if (raw.startsWith("the model reports")) {
    return "The sources cover this question only partly; the uncovered part is stated in the answer.";
  }
  return raw;
}

/** Lines of an excerpt that carry the answer: shared numbers first, then shared content words. */
export function supportingLines(excerpt: string, answer: string): Set<number> {
  const numbers = new Set((answer.match(/\d+(?:[.,]\d+)?/g) ?? []).map((n) => n.replace(",", ".")));
  const words = new Set(
    (answer.toLowerCase().match(/[a-z][a-z-]{3,}/g) ?? []).filter((w) => !STOP.has(w)),
  );
  const lines = excerpt.split("\n");
  const hits = new Set<number>();
  lines.forEach((line, i) => {
    const lineNumbers = (line.match(/\d+(?:[.,]\d+)?/g) ?? []).map((n) => n.replace(",", "."));
    // a clause number like "5.2.1.4" is not evidence of a value; the shared number must be a value
    if (lineNumbers.some((n) => numbers.has(n) && !isClauseNumber(n, line))) hits.add(i);
    if (!hits.has(i)) {
      const lineWords = new Set((line.toLowerCase().match(/[a-z][a-z-]{3,}/g) ?? []));
      let shared = 0;
      for (const w of lineWords) if (words.has(w)) shared += 1;
      if (shared >= 3) hits.add(i);
    }
  });
  return hits;
}

function isClauseNumber(n: string, line: string): boolean {
  // "5.2.1.4." at the start of a line is a paragraph number, not a limit
  return new RegExp(`^\\s*${n.replace(/\./g, "\\.")}\\.?\\s`).test(line) || line.trim() === n || line.trim() === `${n}.`;
}

const STOP = new Set(
  "shall that with this from than which their there these those have been will into than also when where what does under over about after before between each such only other more most".split(" "),
);

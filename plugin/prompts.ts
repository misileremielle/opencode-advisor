/**
 * Prompt construction for the advisor agent.
 *
 * The advisor is a real OpenCode agent with its own persistent session, so its
 * system prompt is omp's verbatim (see omp-prompts.ts) followed by the same
 * blocks omp appends: the project's context files, WATCHDOG guidance, shared
 * instructions, and the advisor's own specialization. Each review sends the transcript update as
 * the session's next user message, framed exactly as omp frames it.
 */
import type { Note } from "./guard.ts"
import { OMP_BOUNDARY_GUIDANCE, renderSystem } from "./omp-prompts.ts"

export const ADVISOR_GUIDANCE = "weigh, don't blindly obey"

/** The advisor agent's system prompt, in omp's order. */
export function buildSystemPrompt(opts: {
  maxNotes?: number
  /** omp's `context-files.md` block (the project's AGENTS.md files). */
  projectContext?: string
  sharedInstructions?: string
  advisorInstructions?: string
  watchdogBlocks: string[]
}): string {
  const parts = [renderSystem(opts.maxNotes)]
  if (opts.projectContext?.trim()) parts.push(opts.projectContext.trim())
  for (const block of opts.watchdogBlocks) if (block.trim()) parts.push(block.trim())
  if (opts.sharedInstructions?.trim()) parts.push(opts.sharedInstructions.trim())
  if (opts.advisorInstructions?.trim()) parts.push(opts.advisorInstructions.trim())
  return parts.join("\n\n")
}

/** omp's header for a batch delivered after the primary has answered. */
export const BOUNDARY_GUIDANCE = OMP_BOUNDARY_GUIDANCE.trim()

/** Render notes as the agent-facing `<advisory>` blocks omp uses. */
/**
 * Appended to a steered note when the agent had already written its final
 * answer, which is the one case where acting on a note buries that answer: the
 * reader has to scroll past the whole exchange to find what was concluded.
 */
/**
 * Appended to every delivered advisory. Self-guarding on purpose: raise time
 * cannot know delivery-time context — the household session delivered a nit
 * raised 14 hours earlier, and another raised 80 s before the answer it
 * followed — so the hint must not assert that an answer exists, only instruct
 * conditionally. Mid-turn the condition is false and the line is inert.
 */
export const RESTATE_HINT =
  '<advisory-closeout>If you have already written a final answer and this note changes it, finish with a complete restatement of the corrected answer. If it changes nothing, acknowledge it in at most one line or not at all; never repost an unchanged answer.</advisory-closeout>'

export function formatAdvisoryBatch(notes: Note[], advisorName?: string, opts: { boundary?: boolean } = {}): string {
  const body =
    notes
      .map((n) => {
        const severity = n.severity ? ` severity="${n.severity}"` : ""
        const who = advisorName ? ` advisor="${escapeAttr(advisorName)}"` : ""
        return `<advisory${who}${severity} guidance="${ADVISOR_GUIDANCE}">\n${escapeXml(n.note)}\n</advisory>`
      })
      .join("\n") +
    "\n" +
    RESTATE_HINT
  return opts.boundary ? `${BOUNDARY_GUIDANCE}\n\n${body}` : body
}

function escapeXml(text: string): string {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
}

function escapeAttr(text: string): string {
  return escapeXml(text).replaceAll('"', "&quot;")
}

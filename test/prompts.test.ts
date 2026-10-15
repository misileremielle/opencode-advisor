import { describe, expect, test } from "bun:test"
import { OMP_ADVISE_TOOL, OMP_BOUNDARY_GUIDANCE, OMP_SYSTEM, renderContextFiles, renderSystem } from "../plugin/omp-prompts.ts"
import { BOUNDARY_GUIDANCE, buildSystemPrompt, formatAdvisoryBatch } from "../plugin/prompts.ts"

describe("omp prompts", () => {
  test("system.md renders its note budget, defaulting to omp's 4", () => {
    expect(renderSystem(2)).toContain("max 2 non-blockers/update")
    expect(renderSystem(undefined)).toContain("max 4 non-blockers/update")
    expect(renderSystem(3)).not.toContain("{{")
  })

  test("the verbatim prompts are omp's", () => {
    expect(OMP_SYSTEM.startsWith("<conventions>")).toBe(true)
    expect(OMP_SYSTEM).toContain("Default read-only: `read`, `grep`, `glob`")
    expect(OMP_SYSTEM.trimEnd().endsWith("Offer better designs, not only warning.")).toBe(true)
    expect(OMP_ADVISE_TOOL).toBe(
      "Watched agent: send 1 concrete, terse advice.\nUse sparingly; stay silent when nothing matters.\nCall to avert likely-wrong or materially wasteful work.",
    )
    expect(OMP_BOUNDARY_GUIDANCE).toContain("aggregated review from other models")
  })

  test("context-files.md renders each file inside omp's project-context framing", () => {
    const block = renderContextFiles([
      { path: "/r/AGENTS.md", content: "RULE ONE" },
      { path: "/r/sub/AGENTS.md", content: "RULE TWO" },
    ])!
    expect(block.startsWith("<project-context>")).toBe(true)
    expect(block).toContain("binding on driving agent")
    expect(block).toContain('<file path="/r/AGENTS.md">\nRULE ONE\n</file>')
    expect(block.indexOf("RULE ONE")).toBeLessThan(block.indexOf("RULE TWO"))
    expect(block).not.toContain("{{")
    expect(renderContextFiles([])).toBeUndefined()
  })
})

describe("buildSystemPrompt", () => {
  test("is omp's system prompt followed by context files, watchdog, shared and advisor blocks, in omp's order", () => {
    const prompt = buildSystemPrompt({
      maxNotes: 2,
      projectContext: "@@CONTEXT@@",
      sharedInstructions: "@@SHARED@@",
      advisorInstructions: "@@OWN@@",
      watchdogBlocks: ["@@WATCHDOG@@"],
    })
    expect(prompt.startsWith(renderSystem(2))).toBe(true)
    const order = ["@@CONTEXT@@", "@@WATCHDOG@@", "@@SHARED@@", "@@OWN@@"].map((t) => prompt.indexOf(t))
    expect(order).toEqual([...order].sort((x, y) => x - y))
    expect(order.every((i) => i > 0)).toBe(true)
  })
})

describe("formatAdvisoryBatch", () => {
  test("renders omp-style advisory elements with escaping", () => {
    const text = formatAdvisoryBatch([{ note: "Use A < B & C", severity: "blocker" }], "Security")
    expect(text).toContain('severity="blocker"')
    expect(text).toContain('advisor="Security"')
    expect(text).toContain("A &lt; B &amp; C")
    expect(text).toContain('guidance="weigh, don\'t blindly obey"')
  })

  test("every batch carries the conditional close-out hint", () => {
    const text = formatAdvisoryBatch([{ note: "x", severity: "nit" }], "Advisor")
    expect(text).toContain("<advisory-closeout>If you have already written a final answer and this note changes it")
    expect(text).toContain("never repost an unchanged answer")
    expect(text.endsWith("</advisory-closeout>")).toBe(true)
  })

  test("a boundary batch is prefaced by omp's boundary guidance", () => {
    const text = formatAdvisoryBatch([{ note: "x", severity: "concern" }], "Advisor", { boundary: true })
    expect(text.startsWith(BOUNDARY_GUIDANCE)).toBe(true)
    expect(formatAdvisoryBatch([{ note: "x" }], "Advisor").includes(BOUNDARY_GUIDANCE)).toBe(false)
  })
})

import { describe, expect, test } from "bun:test"
import { STALE_AFTER_MS, StaleNoteFilter } from "../plugin/stale.ts"

const note = (id: string, text: string) => ({ id, role: "user", content: [{ type: "text", text }] })
const reply = (id: string, text: string) => ({ id, role: "assistant", content: [{ type: "text", text }] })
const roles = (messages: { role: string }[]) => messages.map((m) => m.role)

describe("StaleNoteFilter", () => {
  test("keeps a note that reaches the model promptly, and keeps it on later steps", () => {
    const filter = new StaleNoteFilter()
    filter.track("m1", "<advisory>a</advisory>", 0)
    const first = [note("u", "hi"), note("m1", "<advisory>a</advisory>")]
    expect(filter.filter(first, 10_000)).toBe(0)
    expect(first).toHaveLength(2)
    const later = [note("u", "hi"), note("m1", "<advisory>a</advisory>")]
    expect(filter.filter(later, STALE_AFTER_MS * 10)).toBe(0)
    expect(later).toHaveLength(2)
  })

  test("drops a note that first reaches the model late, and keeps dropping it", () => {
    const filter = new StaleNoteFilter()
    filter.track("m1", "<advisory>a</advisory>", 0)
    const first = [note("u", "hi"), note("m1", "<advisory>a</advisory>")]
    expect(filter.filter(first, STALE_AFTER_MS + 1)).toBe(1)
    expect(first.map((m) => m.id)).toEqual(["u"])
    const later = [note("u", "hi"), note("m1", "<advisory>a</advisory>")]
    expect(filter.filter(later, STALE_AFTER_MS + 5_000)).toBe(1)
    expect(later).toHaveLength(1)
  })

  test("matches by text when the request carries no id", () => {
    const filter = new StaleNoteFilter()
    filter.track("m1", "<advisory>late</advisory>", 0)
    const messages = [{ role: "user", content: "<advisory>late</advisory>" }, { role: "user", content: "keep" }]
    expect(filter.filter(messages, STALE_AFTER_MS + 1)).toBe(1)
    expect(messages).toEqual([{ role: "user", content: "keep" }])
  })

  test("leaves untracked messages and non-arrays alone", () => {
    const filter = new StaleNoteFilter()
    filter.track("m1", "x", 0)
    const messages = [note("u", "other")]
    expect(filter.filter(messages, STALE_AFTER_MS * 2)).toBe(0)
    expect(messages).toHaveLength(1)
    expect(filter.filter(undefined)).toBe(0)
  })

  // The shape of the observed failures: a queued note promoted at the idle
  // boundary is the only input after the final answer. Dropping it sent a
  // conversation ending on that answer, and the provider returned 400
  // ("assistant message prefill").
  test("keeps a late note that is the only input after the final answer, and keeps it on later steps", () => {
    const filter = new StaleNoteFilter()
    filter.track("m1", "<advisory>late</advisory>", 0)
    const first = [note("u", "fix foo"), reply("a1", "Done."), note("m1", "<advisory>late</advisory>")]
    expect(filter.filter(first, STALE_AFTER_MS * 10)).toBe(0)
    expect(roles(first)).toEqual(["user", "assistant", "user"])
    const later = [...first, reply("a2", "Restated answer.")]
    expect(filter.filter(later, STALE_AFTER_MS * 11)).toBe(0)
    expect(later).toHaveLength(4)
  })

  test("never leaves the request ending on an assistant message", () => {
    // Two late notes at the end: one may go, one must stay.
    const two = new StaleNoteFilter()
    two.track("m1", "<advisory>one</advisory>", 0)
    two.track("m2", "<advisory>two</advisory>", 0)
    const both = [note("u", "fix foo"), reply("a1", "Done."), note("m1", "<advisory>one</advisory>"), note("m2", "<advisory>two</advisory>")]
    expect(two.filter(both, STALE_AFTER_MS * 10)).toBe(1)
    expect(roles(both)).toEqual(["user", "assistant", "user"])

    // A trailing system update is not a turn the model can answer.
    const sys = new StaleNoteFilter()
    sys.track("m1", "<advisory>late</advisory>", 0)
    const withSystem = [reply("a1", "Done."), note("m1", "<advisory>late</advisory>"), { role: "system", content: "skills changed" }]
    expect(sys.filter(withSystem, STALE_AFTER_MS * 10)).toBe(0)
    expect(withSystem).toHaveLength(3)

    // Nothing at all before the note: dropping it would empty the request.
    const alone = new StaleNoteFilter()
    alone.track("m1", "<advisory>late</advisory>", 0)
    const only = [note("m1", "<advisory>late</advisory>")]
    expect(alone.filter(only, STALE_AFTER_MS * 10)).toBe(0)
    expect(only).toHaveLength(1)
  })

  test("still drops a late note when another input follows it", () => {
    const filter = new StaleNoteFilter()
    filter.track("m1", "<advisory>late</advisory>", 0)
    const messages = [reply("a1", "Done."), note("m1", "<advisory>late</advisory>"), note("u2", "and also this")]
    expect(filter.filter(messages, STALE_AFTER_MS + 1)).toBe(1)
    expect(messages.map((m) => m.id)).toEqual(["a1", "u2"])
  })
})

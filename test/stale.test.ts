import { describe, expect, test } from "bun:test"
import { STALE_AFTER_MS, StaleNoteFilter } from "../plugin/stale.ts"

const note = (id: string, text: string) => ({ id, role: "user", content: [{ type: "text", text }] })

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
})

/**
 * Late-advice filter.
 *
 * A queued note waits in the session inbox until the next turn boundary, and on
 * the household sessions the median wait was about seven minutes: by then most
 * concerns describe work that is already done, and answering them costs the
 * agent a whole extra reply. The plugin cannot intercept promotion itself, so
 * the filter works on the request instead: the first time a tracked note shows
 * up in a primary request, it is kept if it arrived within `STALE_AFTER_MS` of
 * being injected and hidden from the model otherwise. The verdict is sticky, so
 * a kept note stays and a dropped one never reappears on a later step. Blockers
 * are never tracked. The message itself stays in the transcript.
 */

export const STALE_AFTER_MS = 120_000
const MAX_TRACKED = 500

interface Tracked {
  injectedAt: number
  text: string
  firstSeen?: number
  drop?: boolean
}

function messageText(message: unknown): string | undefined {
  const content = (message as { content?: unknown } | null)?.content
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return undefined
  const parts = content
    .map((part) => (typeof (part as { text?: unknown })?.text === "string" ? (part as { text: string }).text : undefined))
    .filter((text): text is string => text !== undefined)
  return parts.length > 0 ? parts.join("") : undefined
}

export class StaleNoteFilter {
  #tracked = new Map<string, Tracked>()
  #byText = new Map<string, string>()

  track(id: string, text: string, now: number = Date.now()): void {
    this.#tracked.set(id, { injectedAt: now, text })
    this.#byText.set(text, id)
    while (this.#tracked.size > MAX_TRACKED) {
      const oldest = this.#tracked.keys().next().value as string
      const entry = this.#tracked.get(oldest)
      this.#tracked.delete(oldest)
      if (entry && this.#byText.get(entry.text) === oldest) this.#byText.delete(entry.text)
    }
  }

  /** Remove late notes from a request's messages in place; returns how many were removed. */
  filter(messages: unknown, now: number = Date.now()): number {
    if (!Array.isArray(messages) || this.#tracked.size === 0) return 0
    let removed = 0
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i] as { id?: unknown } | null
      const id = typeof message?.id === "string" && this.#tracked.has(message.id) ? message.id : undefined
      const text = id === undefined ? messageText(message) : undefined
      const key = id ?? (text !== undefined ? this.#byText.get(text) : undefined)
      if (key === undefined) continue
      const entry = this.#tracked.get(key)
      if (!entry) continue
      if (entry.firstSeen === undefined) {
        entry.firstSeen = now
        entry.drop = now - entry.injectedAt > STALE_AFTER_MS
      }
      if (entry.drop) {
        messages.splice(i, 1)
        removed++
      }
    }
    return removed
  }
}

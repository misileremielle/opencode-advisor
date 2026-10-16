import { describe, expect, test } from "bun:test"
import type { AdvisorConfig } from "../plugin/config.ts"
import { AdvisorEngine, type AdvisorRunResult, type EngineHost, type InjectInput, type ModelRef, type NotifyInput, type PersistedCounters } from "../plugin/engine.ts"
import type { SessionMessage } from "../plugin/transcript.ts"
import { mkdtempSync } from "node:fs"
import * as engineOs from "node:os"
import * as enginePath from "node:path"
import { readFileSync } from "node:fs"

const notesBase = mkdtempSync(enginePath.join(engineOs.tmpdir(), "advisor-engine-notes-"))

function makeConfig(over: Partial<AdvisorConfig> = {}): AdvisorConfig {
  return {
    enabled: true,
    model: "p/m",
    advisors: [{ name: "Advisor", slug: "advisor", enabled: true, tools: [] }],
    immuneTurns: 3,
    includeThinking: true,
    maxTranscriptChars: 60_000,
    requestTimeoutMs: 45_000,
    contextLine: true,
    notify: "off",
    watchdogBlocks: [],
    warnings: [],
    ...over,
  }
}

interface FakeHost extends EngineHost {
  injections: InjectInput[]
  prompts: string[]
  responses: string[]
  listCalls: number
  persisted: PersistedCounters[]
  logs: string[]
  model?: ModelRef
  /** Every advisor run fails with this error. */
  generateError?: string
  /** Overrides the scripted advisor run. */
  run?: (input: { advisorSessionID: string; text: string; signal: AbortSignal }) => Promise<AdvisorRunResult>
  /** Advisor sessions opened, as `parentID/slug`. */
  advisorSessions: string[]
  /** What `advise` answered the fake advisor. */
  acks: string[]
  engine?: AdvisorEngine
  /** The first N inject calls throw, to test delivery fallbacks. */
  injectFailures?: number
  session?: { agent?: string; location?: { directory?: string } }
  agents?: { id: string; mode?: string; hidden?: boolean }[]
}

function makeHost(directory: string, messages: SessionMessage[]): FakeHost {
  const host: FakeHost = {
    directory,
    injections: [],
    prompts: [],
    responses: [],
    listCalls: 0,
    persisted: [],
    notesBase,
    logs: [],
    advisorSessions: [],
    acks: [],
    persistCounters(_sessionID, counters) {
      host.persisted.push(counters)
    },
    async listMessages() {
      host.listCalls += 1
      return messages
    },
    async createAdvisorSession({ parentID, advisor }) {
      host.advisorSessions.push(`${parentID}/${advisor.slug}`)
      return `adv_${parentID}_${advisor.slug}`
    },
    async runAdvisor(input) {
      // A scripted advisor: each response is the notes it raises through
      // the real `advise` path, one call per note.
      host.prompts.push(input.text)
      if (host.run) return host.run(input)
      if (host.generateError) return { ok: false, error: host.generateError }
      const reply = JSON.parse(host.responses.shift() ?? '{"notes":[]}') as { notes?: { note: string; severity?: string }[] }
      for (const note of reply.notes ?? []) host.acks.push(await host.engine!.advise(input.advisorSessionID, note))
      return { ok: true }
    },
    async inject(input) {
      if (host.injectFailures && host.injectFailures > 0) {
        host.injectFailures -= 1
        throw new Error("injection rejected")
      }
      host.injections.push(input)
      return `msg_${host.injections.length}`
    },
    async getSession() {
      return host.session
    },
    async listAgents() {
      return host.agents ?? []
    },
    async resolveModel(): Promise<ModelRef> {
      return host.model ?? { providerID: "p", id: "m" }
    },
    log(level, message, data) {
      host.logs.push(`${level}: ${message}${data ? ` ${JSON.stringify(data)}` : ""}`)
    },
  }
  return host
}

/** An engine the fake advisor can call back into, as the real `advise` tool does. */
class TestEngine extends AdvisorEngine {
  constructor(config: AdvisorConfig, host: FakeHost) {
    super(config, host)
    host.engine = this
  }
}

const user: SessionMessage = { id: "m1", type: "user", text: "do the thing" }
const terminal: SessionMessage = { id: "m2", type: "assistant", finish: "stop", content: [{ type: "text", text: "done" }] }
const midwork: SessionMessage = {
  id: "m3",
  type: "assistant",
  finish: "tool-calls",
  content: [{ type: "tool", name: "edit", state: { status: "completed", input: {}, content: [] } }],
}

describe("AdvisorEngine", () => {
  test("a concern after a terminal answer wakes the agent", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"Off-by-one in foo.ts"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    expect(host.injections).toHaveLength(1)
    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.resume).toBe(true)
    expect(host.injections[0]!.text).toContain('severity="concern"')
    expect(host.injections[0]!.text).toContain("Off-by-one in foo.ts")
    expect(host.injections[0]!.metadata?.advisor).toBeDefined()
    engine.dispose()
  })

  test("a queued concern is hidden from the model when it arrives late; a blocker never is", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"Late concern"}]}')
    // A failed wake falls back to queue, which is the late-arriving path.
    host.injectFailures = 1
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections[0]!.delivery).toBe("queue")
    // Another input follows the note, so hiding it cannot leave the request
    // ending on an assistant message (stale.test.ts covers that case).
    const next = { id: "msg_2", role: "user", content: [{ type: "text", text: "next" }] }
    const late = [{ id: "msg_1", role: "user", content: [{ type: "text", text: host.injections[0]!.text }] }, next]
    expect(engine.staleNotes.filter(late, Date.now() + 10 * 60_000)).toBe(1)
    engine.dispose()

    const host2 = makeHost("/repo", [user, terminal])
    host2.responses.push('{"notes":[{"severity":"blocker","note":"Late blocker"}]}')
    const engine2 = new TestEngine(makeConfig(), host2)
    await engine2.review("s1", false)
    const kept = [{ id: "msg_1", role: "user", content: [{ type: "text", text: host2.injections[0]!.text }] }, next]
    expect(engine2.staleNotes.filter(kept, Date.now() + 10 * 60_000)).toBe(0)
    engine2.dispose()
  })

  test("steers a concern while work is still streaming", async () => {
    const host = makeHost("/repo", [user, midwork])
    host.responses.push('{"notes":[{"severity":"concern","note":"Wrong path"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)

    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.resume).toBe(true)
    engine.dispose()
  })

  test("does nothing while disabled", async () => {
    const host = makeHost("/repo", [user, terminal])
    const engine = new TestEngine(makeConfig({ enabled: false }), host)

    await engine.review("s1", false)

    expect(host.listCalls).toBe(0)
    expect(host.injections).toHaveLength(0)
    engine.dispose()
  })

  test("suppresses content-free and repeated notes", async () => {
    const messages: SessionMessage[] = [user, terminal]
    const host = makeHost("/repo", messages)
    host.responses.push('{"notes":[{"note":"LGTM"},{"severity":"concern","note":"Real issue"}]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(1)
    expect(host.injections[0]!.text).toContain("Real issue")

    // A second review sees the same note and drops it as a duplicate.
    messages.push({ id: "m4", type: "assistant", finish: "stop", content: [{ type: "text", text: "more" }] })
    host.responses.push('{"notes":[{"severity":"concern","note":"real issue"}]}')
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(1)
    engine.dispose()
  })

  test("respects the per-advisor note budget", async () => {
    const host = makeHost("/repo", [user, midwork])
    host.responses.push(
      '{"notes":[{"severity":"concern","note":"one"},{"severity":"concern","note":"two"},{"severity":"concern","note":"three"}]}',
    )
    const config = makeConfig({
      advisors: [{ name: "Advisor", slug: "advisor", enabled: true, tools: [], maxNotesPerUpdate: 2 }],
    })
    const engine = new TestEngine(config, host)

    await engine.review("s1", false)

    const text = host.injections.flatMap((i) => i.text).join("\n")
    // [ >] excludes <advisory-closeout>, which rides the batch but is not a note.
    expect(text.match(/<advisory[ >]/g)?.length).toBe(2)
    engine.dispose()
  })

  test("seeds to the last user turn so enabling mid-session does not replay history", async () => {
    const messages: SessionMessage[] = [
      { id: "old1", type: "user", text: "old" },
      { id: "old2", type: "assistant", finish: "stop", content: [{ type: "text", text: "old done" }] },
      user,
      terminal,
    ]
    const host = makeHost("/repo", messages)
    host.responses.push('{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    const prompt = host.prompts[0]!
    expect(prompt).toContain("do the thing")
    expect(prompt).not.toContain("old done")
    engine.dispose()
  })

  test("status separates the session switch from the roster switch", () => {
    const host = makeHost("/repo", [user, terminal])
    const engine = new TestEngine(makeConfig({ enabled: false }), host)

    let status = engine.status("s1")
    expect(status.enabled).toBe(false)
    expect(status.defaultEnabled).toBe(false)
    expect(status.override).toBeUndefined()
    expect(status.advisors[0]!.rosterEnabled).toBe(true)
    expect(status.advisors[0]!.active).toBe(false)

    engine.setSessionEnabled("s1", true)
    status = engine.status("s1")
    expect(status.enabled).toBe(true)
    expect(status.override).toBe(true)
    expect(status.advisors[0]!.active).toBe(true)

    engine.setSessionEnabled("s1", undefined, false)
    status = engine.status("s1")
    expect(status.enabled).toBe(false)
    expect(status.override).toBeUndefined()
    engine.dispose()
  })

  test("persists a session override when one is set", () => {
    const host = makeHost("/repo", [user, terminal])
    const calls: [string, boolean | undefined][] = []
    host.onSessionOverride = (sessionID, enabled) => {
      calls.push([sessionID, enabled])
    }
    const engine = new TestEngine(makeConfig(), host)

    engine.setSessionEnabled("s1", true)
    engine.setSessionEnabled("s1", undefined, false)
    expect(calls).toEqual([["s1", true]])
    engine.dispose()
  })

  test("records a review pass even when it raises no notes", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    const status = engine.status("s1")
    expect(status.reviews).toBe(1)
    expect(status.lastNoteCount).toBe(0)
    expect(status.lastReviewAt).toBeGreaterThan(0)
    expect(status.lastError).toBeUndefined()
    engine.dispose()
  })

  test("marks a mid-turn pass as in progress", async () => {
    const messages: SessionMessage[] = [user, terminal]
    const host = makeHost("/repo", messages)
    host.responses.push('{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)
    expect(engine.status("s1").lastOutcome).toBe("no notes (in progress)")

    messages.push({ id: "m4", type: "assistant", finish: "stop", content: [{ type: "text", text: "settled" }] })
    host.responses.push('{"notes":[]}')
    await engine.review("s1", false)
    expect(engine.status("s1").lastOutcome).toBe("no notes")
    engine.dispose()
  })

  test("notifies once when a note is delivered", async () => {
    const host = makeHost("/repo", [user, terminal])
    const calls: NotifyInput[] = []
    host.notify = (input) => {
      calls.push(input)
    }
    host.responses.push('{"notes":[{"severity":"concern","note":"Guard the empty case"}]}')
    const engine = new TestEngine(makeConfig({ notify: "away" }), host)

    await engine.review("s1", false)

    expect(calls).toHaveLength(1)
    expect(calls[0]!.title).toBe("Advisor · concern")
    expect(calls[0]!.body).toContain("Guard the empty case")
    expect(calls[0]!.showWhenFocused).toBe(false)
    expect(calls[0]!.directory).toBe("/repo")
    engine.dispose()
  })

  test("never notifies when notifications are off", async () => {
    const host = makeHost("/repo", [user, terminal])
    let calls = 0
    host.notify = () => {
      calls += 1
    }
    host.responses.push('{"notes":[{"severity":"blocker","note":"Stop"}]}')
    const engine = new TestEngine(makeConfig({ notify: "off" }), host)

    await engine.review("s1", false)

    expect(calls).toBe(0)
    engine.dispose()
  })

  test("notify always asks to show while focused", async () => {
    const host = makeHost("/repo", [user, terminal])
    const calls: NotifyInput[] = []
    host.notify = (input) => {
      calls.push(input)
    }
    host.responses.push('{"notes":[{"severity":"concern","note":"Note"}]}')
    const engine = new TestEngine(makeConfig({ notify: "always" }), host)

    await engine.review("s1", false)

    expect(calls[0]!.showWhenFocused).toBe(true)
    engine.dispose()
  })

  test("counts the notes a review pass raises", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"Guard the empty case"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    expect(engine.status("s1").lastNoteCount).toBe(1)
    expect(engine.status("s1").lastOutcome).toBe("1 notes")
    engine.dispose()
  })

  test("a blocker pages even while the user is looking", async () => {
    const host = makeHost("/repo", [user, terminal])
    const calls: NotifyInput[] = []
    host.notify = (input) => {
      calls.push(input)
    }
    host.responses.push('{"notes":[{"severity":"blocker","note":"This drops the last write"}]}')
    const engine = new TestEngine(makeConfig({ notify: "away" }), host)

    await engine.review("s1", false)

    expect(calls).toHaveLength(1)
    expect(calls[0]!.title).toBe("Advisor · blocker")
    expect(calls[0]!.showWhenFocused).toBe(true)
    engine.dispose()
  })

  test("suppresses a reworded repeat of an earlier note", async () => {
    const messages: SessionMessage[] = [user, midwork]
    const host = makeHost("/repo", messages)
    const engine = new TestEngine(makeConfig(), host)

    host.responses.push('{"notes":[{"severity":"concern","note":"The emit URL is built from the origin so a path prefix is dropped"}]}')
    await engine.review("s1", true)
    expect(host.injections).toHaveLength(1)

    messages.push({ id: "m9", type: "assistant", finish: "stop", content: [{ type: "text", text: "more" }] })
    host.responses.push('{"notes":[{"severity":"concern","note":"Emit URL built from origin drops the path prefix"}]}')
    await engine.review("s1", true)

    expect(host.injections).toHaveLength(1)
    engine.dispose()
  })

  test("caps delivered blockers per user turn", async () => {
    const messages = [user, midwork]
    const host = makeHost("/repo", messages)
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 3; i++) {
      // Reviews are incremental, so each pass needs new work to review.
      if (i > 0) messages.push({ ...midwork, id: `step${i}` })
      host.responses.push(`{"notes":[{"severity":"blocker","note":"blocker number ${i}"}]}`)
      await engine.review("s1", true)
    }
    expect(host.injections).toHaveLength(2)
    engine.dispose()
  })

  test("says so once when the configured model cannot be resolved", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.model = { providerID: "p", id: "m", warning: 'the configured model "opencode-go/GLM-5.3-Flash" is not in provider "opencode-go"' }
    host.responses.push('{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)

    const notices = host.injections.filter((i) => (i.metadata as { advisor?: { kind?: string } })?.advisor?.kind === "notice")
    expect(notices).toHaveLength(1)
    expect(notices[0]!.delivery).toBe("queue")
    expect(notices[0]!.resume).toBe(false)
    expect(notices[0]!.text).toContain("not in provider")
    expect(engine.status("s1").modelWarning).toContain("not in provider")

    host.responses.push('{"notes":[]}')
    await engine.review("s1", false)
    expect(host.injections.filter((i) => (i.metadata as { advisor?: { kind?: string } })?.advisor?.kind === "notice")).toHaveLength(1)
    engine.dispose()
  })

  test("says so once when the reviewer's model call fails", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.generateError = "Model unavailable: opencode-go/GLM-5.3-Flash"
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)

    const notices = host.injections.filter((i) => (i.metadata as { advisor?: { kind?: string } })?.advisor?.kind === "notice")
    expect(notices).toHaveLength(1)
    expect(notices[0]!.text).toContain("Model unavailable")

    await engine.review("s1", false)
    expect(host.injections).toHaveLength(1)
    expect(engine.status("s1").lastOutcome).toBe("advisor error")
    engine.dispose()
  })

  test("says nothing extra when the model resolves cleanly", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(0)
    expect(engine.status("s1").modelWarning).toBeUndefined()
    engine.dispose()
  })

  test("a nit does not survive a settled turn", async () => {
    const messages = [user, terminal]
    const host = makeHost("/repo", messages)
    const engine = new TestEngine(makeConfig(), host)

    host.responses.push('{"notes":[{"severity":"nit","note":"Consider renaming the flag"}]}')
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(0)

    // The same note while a turn is running still shapes the next step.
    messages.push({ ...midwork, id: "later" })
    host.responses.push('{"notes":[{"severity":"nit","note":"Consider renaming the flag"}]}')
    await engine.review("s1", true)
    expect(host.injections).toHaveLength(1)
    engine.dispose()
  })

  test("a concern still lands after the turn settles", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The retry loop can spin forever"}]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(1)
    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.resume).toBe(true)
    engine.dispose()
  })

  test("skips sessions run by auxiliary agents", async () => {
    const roster = [
      { id: "build", mode: "primary", hidden: false },
      { id: "explore", mode: "subagent", hidden: false },
      { id: "historian", mode: "primary", hidden: true },
      { id: "dreamer-memory-mapper", mode: "primary", hidden: true },
    ]
    for (const agent of ["historian", "dreamer-memory-mapper", "explore"]) {
      const host = makeHost("/repo", [user, terminal])
      host.agents = roster
      host.session = { agent, location: { directory: "/repo" } }
      host.responses.push('{"notes":[{"severity":"concern","note":"should not run"}]}')
      const engine = new TestEngine(makeConfig(), host)
      await engine.review("s1", false)
      expect(host.prompts).toHaveLength(0)
      expect(host.injections).toHaveLength(0)
      engine.dispose()
    }
  })

  test("reviews a session run by a visible primary agent", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.agents = [{ id: "build", mode: "primary", hidden: false }]
    host.session = { agent: "build", location: { directory: "/repo" } }
    host.responses.push('{"notes":[{"severity":"concern","note":"real advice"}]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(1)
    engine.dispose()
  })

  test("skips sessions that belong to another location", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.agents = [{ id: "build", mode: "primary", hidden: false }]
    host.session = { agent: "build", location: { directory: "/elsewhere" } }
    host.responses.push('{"notes":[{"severity":"concern","note":"wrong repo"}]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(0)
    engine.dispose()
  })

  test("reviews everything when the agent roster is unusable", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.agents = []
    host.session = { agent: "build", location: { directory: "/repo" } }
    host.responses.push('{"notes":[{"severity":"concern","note":"still reviewed"}]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)
    expect(host.injections).toHaveLength(1)
    engine.dispose()
  })

  test("a plugin shutdown does not page the user", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.run = ({ signal }) =>
      new Promise<AdvisorRunResult>((_resolve, reject) => {
        signal.addEventListener("abort", () =>
          reject(new Error("Connection lost while reading the response: ECONNRESET")),
        )
      })
    const engine = new TestEngine(makeConfig(), host)

    const pending = engine.review("s1", false)
    await new Promise((resolve) => setTimeout(resolve, 5))
    engine.dispose()
    await pending

    // Our own abort is not the endpoint failing: no notice at all, and no
    // half-finished counters written over the stored ones.
    expect(host.injections).toHaveLength(0)
    expect(host.persisted).toHaveLength(0)
  })

  test("counters can be seeded, so a reload does not read as never-reviewed", () => {
    const host = makeHost("/repo", [user, terminal])
    const engine = new TestEngine(makeConfig(), host)
    engine.seedCounters("s1", {
      reviews: 50,
      notesDelivered: 37,
      lastNoteCount: 0,
      lastReviewAt: 1_700_000_000_000,
      lastOutcome: "no notes",
    })

    const status = engine.status("s1")
    expect(status.reviews).toBe(50)
    expect(status.notesDelivered).toBe(37)
    expect(status.lastOutcome).toBe("no notes")
    engine.dispose()
  })

  test("a completed pass persists its counters", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)

    const last = host.persisted.at(-1)
    expect(last?.reviews).toBe(1)
    expect(last?.lastOutcome).toBe("no notes")
    engine.dispose()
  })

  test("a failed pass persists its failure, not just its success", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.generateError = "virtual key is required"
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)

    expect(host.persisted.at(-1)?.lastError).toContain("virtual key is required")
    engine.dispose()
  })

  test("command replies are always queued, so OpenChamber can surface them", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[]}', '{"notes":[]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.reply("s1", "status text")
    expect(host.injections).toHaveLength(1)
    expect(host.injections[0]!.delivery).toBe("queue")
    expect(host.injections[0]!.resume).toBe(false)

    // Even mid-turn, when the old logic would have steered the reply into the
    // running turn — invisible in OpenChamber, which is exactly what made
    // /advisor status vanish while the session was active.
    await engine.review("s1", true)
    await engine.reply("s1", "status text")
    expect(host.injections[1]!.delivery).toBe("queue")
    engine.dispose()
  })

  test("a reply that cannot be delivered is reported, not thrown", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.inject = async () => {
      throw new Error("session closed")
    }
    const engine = new TestEngine(makeConfig(), host)
    await engine.reply("s1", "status text")

    expect(host.logs.some((l) => l.includes("advisor reply was not delivered"))).toBe(true)
    engine.dispose()
  })

  test("notes and notices carry a raisedAt stamp", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"stamped"}]}')
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", false)

    const meta = host.injections[0]!.metadata as { advisor?: { raisedAt?: number } }
    expect(typeof meta.advisor?.raisedAt).toBe("number")
    engine.dispose()
  })

  test("a blocker wakes an idle session", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"blocker","note":"Stop and fix the schema"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    expect(host.injections).toHaveLength(1)
    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.resume).toBe(true)
    engine.dispose()
  })

  test("wake-ups on settled turns stop at the per-turn steer cap", async () => {
    // Each woken run adds an assistant answer and no user message, so the cap
    // is never reset by the advisor's own wake-ups.
    const transcript: SessionMessage[] = [user, terminal]
    const host = makeHost("/repo", transcript)
    const findings = [
      "The retry loop in fetch.ts never backs off",
      "config.ts reads PORT before dotenv loads",
      "Migration 7 drops the users index",
      "The cache key ignores the locale header",
      "Logger writes secrets from the auth payload",
      "Test fixture leaks a temp directory per run",
    ]
    for (const note of findings) host.responses.push(JSON.stringify({ notes: [{ severity: "concern", note }] }))
    const engine = new TestEngine(makeConfig({ immuneTurns: 0 }), host)

    for (let i = 0; i < 6; i++) {
      await engine.review("s1", false)
      transcript.push({ id: `a${i}`, type: "assistant", finish: "stop", content: [{ type: "text", text: `answer ${i}` }] })
    }

    const wakes = host.injections.filter((inj) => inj.resume)
    expect(wakes).toHaveLength(4)
    expect(host.injections.slice(4).every((inj) => inj.delivery === "queue" && inj.resume === false)).toBe(true)
    engine.dispose()
  })

  test("a mid-turn concern steers the running turn", async () => {
    const host = makeHost("/repo", [user, midwork])
    host.responses.push('{"notes":[{"severity":"concern","note":"Check the guard"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)

    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.resume).toBe(true)
    engine.dispose()
  })

  test("a steered note on an already-answered turn asks for a restatement", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The port is wrong"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)

    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.text).toContain("advisory-closeout")
    engine.dispose()
  })

  test("a steered note mid-work carries the conditional hint too", async () => {
    const host = makeHost("/repo", [user, midwork])
    host.responses.push('{"notes":[{"severity":"concern","note":"Check the guard"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)

    expect(host.injections[0]!.delivery).toBe("steer")
    expect(host.injections[0]!.text).toContain("If you have already written a final answer")
    engine.dispose()
  })

  test("a delivered note lands in the notes bridge for the panel", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The port is wrong"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    const st = engine.status("s1")
    console.log("[bridge] injections:", host.injections.length, "| reviews:", st.reviews, "| outcome:", st.lastOutcome, "| err:", st.lastError)
    const { notesFilePath } = await import("../plugin/notes-file.ts")
    const lines = readFileSync(notesFilePath(notesBase, "/repo"), "utf8").trim().split("\n")
    const parsed = JSON.parse(lines.at(-1)!)
    expect(parsed.sessionID).toBe("s1")
    expect(parsed.advisor).toBe("Advisor")
    expect(parsed.severity).toBe("concern")
    expect(parsed.note).toBe("The port is wrong")
    expect(typeof parsed.t).toBe("number")
    engine.dispose()
  })

  test("a wake-up note on a settled turn also asks for a restatement", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The port is wrong"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    expect(host.injections[0]!.resume).toBe(true)
    expect(host.injections[0]!.text).toContain("advisory-closeout")
    engine.dispose()
  })

  test("a failed steer injection falls back to queue and keeps the close-out hint", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The port is wrong"}]}')
    host.injectFailures = 1
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)

    expect(host.injections).toHaveLength(1)
    expect(host.injections[0]!.delivery).toBe("queue")
    expect(host.injections[0]!.resume).toBe(false)
    expect(host.injections[0]!.text).toContain("advisory-closeout")
    expect(host.logs.some((l) => l.includes("retrying queued"))).toBe(true)
    engine.dispose()
  })

  test("a failed queue delivery leaves the note unrouted for a later pass", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The port is wrong"}]}')
    host.injectFailures = 99
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    expect(host.injections).toHaveLength(0)
    expect(host.logs.some((l) => l.includes("was not delivered"))).toBe(true)
    // Nothing was delivered, so the counters must not claim otherwise.
    expect(host.persisted.at(-1)?.notesDelivered ?? 0).toBe(0)
    engine.dispose()
  })
})

describe("AdvisorEngine catch-up", () => {
  const stepEnded = (sessionID = "s1") => ({ type: "session.step.ended", data: { sessionID, finish: "tool-calls" } })

  /** Let the wait register: `catchUp` checks reviewability before it holds. */
  const tick = () => new Promise((resolve) => setTimeout(resolve, 0))

  /** Fail the test instead of hanging when a wait never ends. */
  function within<T>(promise: Promise<T>, ms = 2_000): Promise<T | "timeout"> {
    return Promise.race([promise, new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), ms))])
  }

  function makeCatchUpHost() {
    const host = makeHost("/repo", [user, midwork]) as FakeHost & { interrupts: string[] }
    host.interrupts = []
    host.interrupt = async (sessionID) => {
      host.interrupts.push(sessionID)
    }
    return host
  }

  test("does not wait while fewer than 5 steps are unreviewed", async () => {
    const host = makeCatchUpHost()
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 4; i++) engine.onEvent(stepEnded())

    expect(await within(engine.catchUp("s1"), 50)).toBeUndefined()
    expect(engine.status("s1").catchingUp).toBe(false)
    engine.dispose()
  })

  test("waits at 5 unreviewed steps and releases once every step is reviewed", async () => {
    const host = makeCatchUpHost()
    let answer!: (result: AdvisorRunResult) => void
    host.run = () => new Promise<AdvisorRunResult>((resolve) => (answer = resolve))
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 5; i++) engine.onEvent(stepEnded())
    expect(engine.unreviewedSteps("s1")).toBe(5)

    const wait = engine.catchUp("s1")
    await tick()
    expect(engine.status("s1").catchingUp).toBe(true)
    answer({ ok: true })
    expect(await within(wait)).toBeUndefined()

    // The review ran immediately rather than after the debounce.
    expect(host.prompts.length).toBeGreaterThanOrEqual(1)
    expect(engine.unreviewedSteps("s1")).toBe(0)
    expect(engine.status("s1").catchingUp).toBe(false)
    expect(host.interrupts).toHaveLength(0)
    engine.dispose()
  })

  test("retries a failing pass 3 times, then stops the primary with a notice", async () => {
    const host = makeCatchUpHost()
    host.generateError = "401 unauthorized"
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 5; i++) engine.onEvent(stepEnded())

    expect(await within(engine.catchUp("s1"))).toBeUndefined()

    // The first pass plus 3 retries, each a single non-transient attempt.
    expect(host.prompts).toHaveLength(4)
    expect(host.interrupts).toEqual(["s1"])
    const notice = host.injections.find((i) => i.text.includes("failed 4 times in a row"))
    expect(notice?.delivery).toBe("queue")
    expect(notice?.text).toContain("401 unauthorized")
    // Failed passes leave their steps unreviewed.
    expect(engine.unreviewedSteps("s1")).toBe(5)
    engine.dispose()
  })

  test("a retry that succeeds releases the wait without stopping, and re-reads the failed slice", async () => {
    const host = makeCatchUpHost()
    let failures = 2
    host.run = async () => {
      if (failures > 0) {
        failures -= 1
        return { ok: false, error: "401 unauthorized" }
      }
      return { ok: true }
    }
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 5; i++) engine.onEvent(stepEnded())

    expect(await within(engine.catchUp("s1"))).toBeUndefined()

    expect(host.prompts).toHaveLength(3)
    // The successful retry still saw the transcript the failed passes could not review.
    expect(host.prompts[2]).toContain("edit")
    expect(host.interrupts).toHaveLength(0)
    expect(engine.unreviewedSteps("s1")).toBe(0)
    engine.dispose()
  })

  test("an interrupted turn releases the wait", async () => {
    const host = makeCatchUpHost()
    let answer!: (result: AdvisorRunResult) => void
    host.run = () => new Promise<AdvisorRunResult>((resolve) => (answer = resolve))
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 5; i++) engine.onEvent(stepEnded())

    const wait = engine.catchUp("s1")
    await tick()
    engine.onEvent({ type: "session.execution.interrupted", data: { sessionID: "s1" } })

    expect(await within(wait)).toBeUndefined()
    expect(engine.status("s1").catchingUp).toBe(false)
    answer?.({ ok: true })
    engine.dispose()
  })

  test("turning the advisor off releases the wait", async () => {
    const host = makeCatchUpHost()
    host.run = () => new Promise<AdvisorRunResult>(() => {})
    const engine = new TestEngine(makeConfig(), host)
    for (let i = 0; i < 5; i++) engine.onEvent(stepEnded())

    const wait = engine.catchUp("s1")
    await tick()
    engine.setSessionEnabled("s1", false, false)

    expect(await within(wait)).toBeUndefined()
    engine.dispose()
  })

  test("a disabled advisor never holds the primary", async () => {
    const host = makeCatchUpHost()
    const engine = new TestEngine(makeConfig({ enabled: false }), host)
    for (let i = 0; i < 6; i++) engine.onEvent(stepEnded())

    expect(await within(engine.catchUp("s1"), 50)).toBeUndefined()
    expect(host.prompts).toHaveLength(0)
    engine.dispose()
  })
})

describe("AdvisorEngine advisor agent", () => {
  test("advise delivers live on a settled turn, and is truthful once the review ends", async () => {
    const host = makeHost("/repo", [user, terminal])
    host.responses.push('{"notes":[{"severity":"concern","note":"The retry loop never backs off"}]}')
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", false)

    expect(host.acks).toEqual(["Delivered."])
    expect(host.injections).toHaveLength(1)
    // Outside a live review the tool says so instead of pretending.
    expect(await engine.advise("adv_s1_advisor", { note: "late" })).toBe("Dropped: no review is in progress for this session.")
    engine.dispose()
  })

  test("mid-turn, a non-blocker waits for the run to end while a blocker interrupts now", async () => {
    const host = makeHost("/repo", [user, midwork])
    const engine = new TestEngine(makeConfig(), host)
    const seenDuringRun: number[] = []
    host.run = async ({ advisorSessionID }) => {
      host.acks.push(await engine.advise(advisorSessionID, { note: "Consider a smaller diff", severity: "concern" }))
      seenDuringRun.push(host.injections.length)
      host.acks.push(await engine.advise(advisorSessionID, { note: "This deletes the user's data dir", severity: "blocker" }))
      seenDuringRun.push(host.injections.length)
      return { ok: true }
    }

    await engine.review("s1", true)

    expect(host.acks[0]).toBe("Queued for the end of the turn. Do not re-raise.")
    expect(host.acks[1]).toBe("Delivered.")
    // Nothing during the concern; the blocker landed while the run was live.
    expect(seenDuringRun).toEqual([0, 1])
    expect(host.injections[0]!.text).toContain("deletes the user's data dir")
    // The deferred concern arrives after the run settles.
    expect(host.injections).toHaveLength(2)
    expect(host.injections[1]!.text).toContain("smaller diff")
    engine.dispose()
  })

  test("one persistent advisor session per primary session, fed only what is new", async () => {
    const messages: SessionMessage[] = [user, midwork]
    const host = makeHost("/repo", messages)
    const engine = new TestEngine(makeConfig(), host)

    await engine.review("s1", true)
    messages.push({ id: "m9", type: "assistant", finish: "stop", content: [{ type: "text", text: "all wired up" }] })
    await engine.review("s1", false)

    expect(host.advisorSessions).toEqual(["s1/advisor"])
    expect(host.prompts).toHaveLength(2)
    expect(host.prompts[0]).toContain("### Session update")
    expect(host.prompts[0]).toContain("[in progress — more steps follow]")
    expect(host.prompts[1]).toContain("all wired up")
    expect(host.prompts[1]).not.toContain("do the thing")
    engine.dispose()
  })

  test("the advisor's own session is never reviewed or held", async () => {
    const host = makeHost("/repo", [user, midwork])
    const engine = new TestEngine(makeConfig(), host)
    await engine.review("s1", true)
    expect(engine.isAdvisorSession("adv_s1_advisor")).toBe(true)

    for (let i = 0; i < 6; i++) engine.onEvent({ type: "session.step.ended", data: { sessionID: "adv_s1_advisor", finish: "tool-calls" } })

    expect(engine.unreviewedSteps("adv_s1_advisor")).toBe(0)
    expect(await engine.catchUp("adv_s1_advisor")).toBeUndefined()
    engine.dispose()
  })

  test("a run past its deadline is stopped and counts as a failure", async () => {
    const host = makeHost("/repo", [user, terminal])
    let aborted = false
    host.run = ({ signal }) =>
      new Promise<AdvisorRunResult>((resolve) => {
        signal.addEventListener("abort", () => {
          aborted = true
          resolve({ ok: false, error: "aborted" })
        })
      })
    const engine = new TestEngine(makeConfig({ requestTimeoutMs: 20 }), host)

    await engine.review("s1", false)

    expect(aborted).toBe(true)
    expect(engine.status("s1").lastError).toContain("timed out")
    expect(host.injections.some((i) => i.text.includes("timed out"))).toBe(true)
    engine.dispose()
  })

  test("notes after the answer carry omp's boundary guidance the first time only", async () => {
    const messages: SessionMessage[] = [user, terminal]
    const host = makeHost("/repo", messages)
    const engine = new TestEngine(makeConfig(), host)
    host.responses.push('{"notes":[{"severity":"concern","note":"first late concern"}]}')
    await engine.review("s1", false)
    messages.push({ id: "m10", type: "assistant", finish: "stop", content: [{ type: "text", text: "again" }] })
    host.responses.push('{"notes":[{"severity":"concern","note":"second late concern"}]}')
    await engine.review("s1", false)

    const notes = host.injections.filter((i) => i.text.includes("late concern"))
    expect(notes).toHaveLength(2)
    expect(notes[0]!.text).toContain("aggregated review from other models")
    expect(notes[1]!.text).not.toContain("aggregated review from other models")
    engine.dispose()
  })
})

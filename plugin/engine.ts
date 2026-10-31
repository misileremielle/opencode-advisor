/**
 * The advisor engine: observes a session, runs the reviewer model(s), and
 * routes admitted notes back into the session.
 *
 * Observation is pull-based. The engine subscribes to the event stream only as
 * a trigger (`session.idle`, `session.step.ended`, ...); when triggered it reads
 * the authoritative transcript with `ctx.session.context` and reviews the current
 * update since the last review. The reviewer is a real OpenCode agent with its
 * own persistent session (a child of the primary), omp's prompt, and only
 * `read`/`grep`/`glob`/`advise`, so it verifies with real tool calls and keeps
 * its own context across reviews, as omp's advisor does.
 */
import {
  EmissionGuard,
  resolveChannel,
  severityRank,
  type DeliveryChannel,
  type Note,
  type Severity,
} from "./guard.ts"
import { formatAdvisoryBatch } from "./prompts.ts"
import { StaleNoteFilter } from "./stale.ts"
import type { AdvisorConfig, AdvisorSpec } from "./config.ts"
import { renderDelta, type SessionMessage } from "./transcript.ts"
import { appendNote, NOTES_BASE } from "./notes-file.ts"
import { liveInstances } from "./instances.ts"
import { parseSelector, type ModelRef } from "./model.ts"

export { parseSelector, type ModelRef } from "./model.ts"

export interface AdvisorEvent {
  type?: string
  location?: { directory?: string }
  data?: Record<string, unknown>
}

export interface InjectInput {
  sessionID: string
  text: string
  description?: string
  metadata?: Record<string, unknown>
  delivery: "steer" | "queue"
  resume: boolean
}

/** Everything the engine needs from the OpenCode plugin context. */
export interface EngineHost {
  directory: string
  /** Where delivered notes are appended for the OpenChamber panel. */
  notesBase?: string
  /** Identifies this plugin instance in status output and injected metadata. */
  instance?: string
  listMessages(sessionID: string): Promise<SessionMessage[]>
  /** Open the advisor's persistent session as a child of the primary session. */
  createAdvisorSession(input: { parentID: string; advisor: AdvisorSpec; model?: ModelRef }): Promise<string>
  /**
   * Send one transcript update to an advisor session and wait for its run to
   * settle. `ok: false` means the run errored; aborting `signal` must stop it.
   */
  runAdvisor(input: { advisorSessionID: string; text: string; signal: AbortSignal }): Promise<AdvisorRunResult>
  inject(input: InjectInput): Promise<string | undefined>
  /** Resolve a `provider/model#variant` selector (or the default) to a usable model. */
  resolveModel(selector: string | undefined): Promise<ModelRef | undefined>
  /** The session's own agent and location; absent means "review everything". */
  getSession?(sessionID: string): Promise<{ agent?: string; location?: { directory?: string } } | undefined>
  /** The agent roster, so auxiliary agents can be skipped. */
  listAgents?(): Promise<{ id: string; mode?: string; hidden?: boolean }[]>
  /** Persist a per-session enable override so it survives a plugin reload. */
  onSessionOverride?(sessionID: string, enabled: boolean | undefined): Promise<void> | void
  /** Persist review counters so /advisor status survives a plugin reload. */
  persistCounters?(sessionID: string, counters: PersistedCounters): void
  /** Raise a user-facing notification (OpenChamber only; a no-op elsewhere). */
  notify?(input: NotifyInput): void
  /** Stop the session's running turn (`ctx.session.interrupt`). */
  interrupt?(sessionID: string): Promise<void>

  /** Whether this instance still owns reviewing for its directory. */
  isClaimOwner?: () => boolean
  log(level: "debug" | "warn", message: string, data?: Record<string, unknown>): void
}

export type AdvisorRunResult = { ok: true } | { ok: false; error: string }

/** A review in flight, so the `advise` tool can route its notes. */
interface ActivePass {
  advisor: AdvisorSpec
  sessionID: string
  streaming: boolean
  state: SessionState
  guard: EmissionGuard
  /** Non-blocker notes raised mid-turn, delivered when the run settles. */
  deferred: Note[]
  raised: number
  routed: number
}

export interface NotifyInput {
  title: string
  body: string
  sessionID: string
  directory: string
  /** Ask to show even when the user is looking at OpenChamber. */
  showWhenFocused: boolean
}

/**
 * Review counters that outlive a plugin instance.
 *
 * Every reload builds a new instance with blank counters, which made status read
 * "Reviews 0 · no review yet" on sessions with dozens of reviews, and made
 * successive cards disagree with each other. Persisting them is what makes the
 * numbers mean "this session" rather than "this instance since it loaded".
 */
export interface PersistedCounters {
  reviews: number
  notesDelivered: number
  lastNoteCount: number
  lastReviewAt?: number
  lastOutcome?: string
  lastError?: string
}

interface SessionState {
  sessionID: string
  seeded: boolean
  reviewedCount: number
  streaming: boolean
  completedTurns: number
  immuneTurnStart?: number
  guards: Map<string, EmissionGuard>
  priorNotes: Map<string, string[]>
  reviewInProgress: boolean
  queuedReview: { streaming: boolean } | null
  timer?: ReturnType<typeof setTimeout>
  enabled?: boolean
  notesDelivered: number
  lastReviewAt?: number
  lastError?: string
  /** Completed review passes, even those that raised no notes. */
  reviews: number
  /** Notes raised (not necessarily delivered) by the most recent pass. */
  lastNoteCount: number
  /** Human-readable outcome of the most recent pass, e.g. "no notes". */
  lastOutcome?: string
  backlog: number
  userMessageCount: number
  steersSinceUser: number
  /** Blockers delivered inside the current user turn (see the per-turn cap). */
  blockersThisUserTurn: number
  /** Why the configured reviewer model could not be used, if it could not. */
  lastModelWarning?: string
  /** Failure notices already delivered, so a broken reviewer is loud once. */
  noticesSent: Set<string>
  /** Primary steps that have ended (`session.step.ended`). */
  stepsEnded: number
  /** Steps covered by a review pass that completed without failing. */
  reviewedSteps: number
  /** Whether the current review pass failed (model error, unparseable reply, or a throw). */
  passFailed: boolean
  /** Set while the primary is held in the context hook until the reviewer catches up. */
  catchUp?: CatchUp
  /** Completed turn at which the boundary guidance was last shown. */
  boundaryGuidanceTurn?: number
}

interface CatchUp {
  promise: Promise<void>
  release: () => void
  /** Failed passes since the wait began. */
  failures: number
}

const DEBOUNCE_MS = 350
/** Floor between mid-turn reviews: advice about a turn that is still moving is
 *  worth less the more of it there is, and a fast agent outruns the reviewer. */
const MIDTURN_MIN_INTERVAL_MS = 30_000
/** Ceiling on delivered blockers per user turn: past this a blocker is far more
 *  likely to be churn than signal, and the turn-end pass can re-raise it. */
const MAX_BLOCKERS_PER_USER_TURN = 2
/** How long a session's reviewability verdict is trusted. */
const SESSION_FILTER_TTL_MS = 5 * 60_000
const PRIOR_NOTE_LIMIT = 40
/** Safety net: at most this many steering wake-ups per user turn, so a
 *  confused reviewer cannot loop the primary indefinitely. */
const MAX_STEERS_PER_USER_TURN = 4
/**
 * Reviews run in the background until this many primary steps are unreviewed.
 * From then on the primary's next model request waits — with no time cap —
 * until every step has been reviewed.
 */
const CATCH_UP_THRESHOLD = 5
/** Failed passes retried during a catch-up wait before the primary is stopped. */
const CATCH_UP_RETRIES = 3
/** omp's cadence for repeating the boundary guidance header. */
const BOUNDARY_GUIDANCE_TURNS = 50

function resolvedDirectory(directory: string | undefined): string | undefined {
  return directory ? directory.replace(/\/+$/, "") : undefined
}

export class AdvisorEngine {
  #config: AdvisorConfig
  #host: EngineHost
  #sessions = new Map<string, SessionState>()
  #abort = new AbortController()
  #sessionOrder: string[] = []
  #reviewable = new Map<string, { ok: boolean; at: number }>()
  #agents?: { at: number; reviewable?: Set<string> }
  /** Advisor runs in flight, by advisor session, so `advise` knows where a note belongs. */
  #activePasses = new Map<string, ActivePass>()
  /** Advisor session per primary session and advisor slug. */
  #advisorSessions = new Map<string, string>()
  /** Every advisor session this engine opened. */
  #advisorSessionIDs = new Set<string>()
  /** Queued notes that are hidden from the model if they reach it late. */
  readonly staleNotes = new StaleNoteFilter()

  constructor(config: AdvisorConfig, host: EngineHost) {
    this.#config = config
    this.#host = host
  }

  get enabled(): boolean {
    return this.#config.enabled
  }

  get config(): AdvisorConfig {
    return this.#config
  }

  #state(sessionID: string): SessionState {
    let state = this.#sessions.get(sessionID)
    if (!state) {
      state = {
        sessionID,
        seeded: false,
        reviewedCount: 0,
        streaming: false,
        completedTurns: 0,
        guards: new Map(),
        priorNotes: new Map(),
        reviewInProgress: false,
        queuedReview: null,
        notesDelivered: 0,
        reviews: 0,
        lastNoteCount: 0,
        backlog: 0,
        userMessageCount: 0,
        steersSinceUser: 0,
        blockersThisUserTurn: 0,
        noticesSent: new Set<string>(),
        stepsEnded: 0,
        reviewedSteps: 0,
        passFailed: false,
      }
      this.#sessions.set(sessionID, state)
      this.#sessionOrder.push(sessionID)
      if (this.#sessionOrder.length > 64) {
        const oldest = this.#sessionOrder.shift()
        if (oldest) this.#sessions.delete(oldest)
      }
    }
    return state
  }

  #isEnabled(sessionID: string): boolean {
    const state = this.#sessions.get(sessionID)
    return state?.enabled ?? this.#config.enabled
  }

  /**
   * Whether this session's advisor is on. The prompt-time context line follows
   * this, so a session with the advisor off still gets an untouched prompt.
   */
  advisoriesActive(sessionID: string): boolean {
    return this.#isEnabled(sessionID)
  }

  /** Whether a turn is running, which is what routing keys off. */
  isStreaming(sessionID: string): boolean {
    return this.#sessions.get(sessionID)?.streaming ?? false
  }

  /**
   * Deliver a command reply. Steering gets an answer into a running turn
   * immediately instead of parking it until the next boundary, but a turn can
   * end between that decision and the post, and a steer into a finished turn is
   * not a path worth trusting to deliver — so a failed steer is resent queued
   * rather than dropped. A dropped reply reads as "the command did nothing".
   */
  async reply(sessionID: string, text: string): Promise<void> {
    const metadata = { advisor: { kind: "command", instance: this.#host.instance, raisedAt: Date.now() } }
    try {
      // Always queued, never steered: OpenChamber does not render synthetics, so a
      // steered reply is consumed by the running turn invisibly — and this session
      // is usually running, which made /advisor status vanish exactly when asked.
      // A queued reply surfaces at the next turn boundary and in the inbox.
      await this.#host.inject({ sessionID, text, description: "advisor", metadata, delivery: "queue", resume: false })
    } catch (err) {
      this.#host.log("warn", "advisor reply was not delivered", { sessionID, error: (err as Error).message })
    }
  }

  /**
   * Restore review counters persisted before a reload. Deliberately excludes
   * `reviewedCount`: which transcript slice has been seen is per instance, and
   * a fresh instance should re-read the current turn rather than skip it.
   */
  seedCounters(sessionID: string, counters: Partial<PersistedCounters>): void {
    const state = this.#state(sessionID)
    if (typeof counters.reviews === "number") state.reviews = counters.reviews
    if (typeof counters.notesDelivered === "number") state.notesDelivered = counters.notesDelivered
    if (typeof counters.lastNoteCount === "number") state.lastNoteCount = counters.lastNoteCount
    if (typeof counters.lastReviewAt === "number") state.lastReviewAt = counters.lastReviewAt
    if (typeof counters.lastOutcome === "string") state.lastOutcome = counters.lastOutcome
    if (typeof counters.lastError === "string") state.lastError = counters.lastError
  }

  #countersOf(state: SessionState): PersistedCounters {
    return {
      reviews: state.reviews,
      notesDelivered: state.notesDelivered,
      lastNoteCount: state.lastNoteCount,
      lastReviewAt: state.lastReviewAt,
      lastOutcome: state.lastOutcome,
      lastError: state.lastError,
    }
  }

  setSessionEnabled(sessionID: string, enabled: boolean | undefined, persist = true): boolean {
    const state = this.#state(sessionID)
    state.enabled = enabled
    if (persist) void this.#host.onSessionOverride?.(sessionID, enabled)
    const active = this.#isEnabled(sessionID)
    // Turning the advisor off is also the user's way out of a catch-up wait.
    if (!active) this.#releaseCatchUp(state, "advisor disabled")
    return active
  }

  toggleSession(sessionID: string): boolean {
    return this.setSessionEnabled(sessionID, !this.#isEnabled(sessionID))
  }

  /* ---------------------------------------------------------------- *
   * Event intake
   * ---------------------------------------------------------------- */

  onEvent(event: AdvisorEvent): void {
    const type = event.type
    const sessionID = typeof event.data?.sessionID === "string" ? (event.data.sessionID as string) : undefined
    if (!type || !sessionID) return
    // The advisor's own runs are not work to review.
    if (this.#advisorSessionIDs.has(sessionID)) return

    const directory = resolvedDirectory(event.location?.directory)
    if (directory && resolvedDirectory(this.#host.directory) && directory !== resolvedDirectory(this.#host.directory)) return

    const state = this.#state(sessionID)

    switch (type) {
      case "session.step.started": {
        state.streaming = true
        return
      }
      case "session.step.ended": {
        state.stepsEnded += 1
        const finish = typeof event.data?.finish === "string" ? event.data.finish : undefined
        // A step that ended with tool calls is mid-run; anything else yields.
        state.streaming = finish === "tool-calls"
        this.#schedule(sessionID, state.streaming)
        return
      }
      case "session.execution.succeeded": {
        state.streaming = false
        state.completedTurns += 1
        this.#schedule(sessionID, false)
        return
      }
      case "session.execution.failed": {
        state.streaming = false
        this.#schedule(sessionID, false)
        return
      }
      case "session.execution.interrupted": {
        state.streaming = false
        this.#releaseCatchUp(state, "turn interrupted")
        return
      }
      case "session.idle": {
        // Belt and braces: some paths settle without an execution event.
        state.streaming = false
        this.#schedule(sessionID, false)
        return
      }
      default:
        return
    }
  }

  /**
   * Whether this session is worth reviewing at all.
   *
   * Auxiliary work is not: Magic Context and friends run historian, dreamer,
   * compaction and title sessions continuously — hidden agents, plus subagents
   * like `explore` — so reviewing them produces notes while the user's own
   * session sits idle, which is exactly how the advisor came to look like it
   * would not stop. Sessions outside this instance's location are skipped for
   * the same reason: their reviewer would inspect the wrong repository.
   *
   * Fails open: a host without the lookups, or an empty agent roster, reviews
   * everything rather than silently reviewing nothing.
   */
  async #isReviewable(sessionID: string): Promise<boolean> {
    if (!this.#host.getSession) return true
    const now = Date.now()
    const cached = this.#reviewable.get(sessionID)
    if (cached && now - cached.at < SESSION_FILTER_TTL_MS) return cached.ok
    let ok = true
    try {
      const session = await this.#host.getSession(sessionID)
      // An unknown session is not evidence of auxiliary work: skip only what
      // this lookup positively identifies as someone else's or not worth it.
      if (session) {
        const theirs = resolvedDirectory(session.location?.directory)
        const ours = resolvedDirectory(this.#host.directory)
        ok = (theirs === undefined || ours === undefined || theirs === ours) && (await this.#isReviewableAgent(session.agent))
      }
    } catch (err) {
      this.#host.log("warn", "advisor session lookup failed", { error: (err as Error).message })
    }
    this.#reviewable.set(sessionID, { ok, at: now })
    return ok
  }

  async #isReviewableAgent(agentID: string | undefined): Promise<boolean> {
    if (!this.#host.listAgents || agentID === undefined) return true
    const now = Date.now()
    if (!this.#agents || now - this.#agents.at > SESSION_FILTER_TTL_MS) {
      const list = await this.#host.listAgents()
      const reviewable = new Set(
        list.filter((agent) => agent.mode === "primary" && agent.hidden !== true).map((agent) => agent.id),
      )
      // An empty roster means "could not tell", not "nothing is reviewable".
      this.#agents = { at: now, reviewable: reviewable.size > 0 ? reviewable : undefined }
    }
    return this.#agents.reviewable === undefined || this.#agents.reviewable.has(agentID)
  }

  #schedule(sessionID: string, streaming: boolean): void {
    const state = this.#state(sessionID)
    if (!state.timer && !state.reviewInProgress) state.backlog += 1
    if (state.timer) clearTimeout(state.timer)
    // Mid-turn work is reviewed at most once per interval; later triggers ride
    // along with the pending one instead of stacking a note every step.
    const since = state.lastReviewAt === undefined ? Number.POSITIVE_INFINITY : Date.now() - state.lastReviewAt
    const gap = streaming ? MIDTURN_MIN_INTERVAL_MS - since : 0
    const delay = Math.max(DEBOUNCE_MS, gap)
    state.timer = setTimeout(() => {
      state.timer = undefined
      void this.review(sessionID, streaming)
    }, delay)
    // Keep the process from being held open by a pending review.
    state.timer.unref?.()
  }

  /* ---------------------------------------------------------------- *
   * Review
   * ---------------------------------------------------------------- */

  async review(sessionID: string, streaming: boolean): Promise<void> {
    const state = this.#state(sessionID)
    // Losing the review claim is not a failure: another instance for this
    // directory is newer, and two reviewers means two sets of state, two
    // budgets and duplicate notes. Stay quiet and let the owner work.
    if (this.#host.isClaimOwner && !this.#host.isClaimOwner()) {
      this.#skipReview(state, "not the review owner")
      return
    }
    if (!(await this.#isReviewable(sessionID))) {
      this.#skipReview(state, "session not reviewable")
      return
    }
    if (!this.#isEnabled(sessionID)) {
      this.#skipReview(state, "advisor disabled")
      return
    }
    if (state.reviewInProgress) {
      state.queuedReview = { streaming }
      return
    }
    state.reviewInProgress = true
    state.passFailed = false
    // Steps that ended before this pass read the transcript are the ones it covers.
    const stepsAtStart = state.stepsEnded
    // Where the reviewed cursor stood before this pass, so a failed pass leaves
    // its slice unreviewed and a retry reads it again.
    let rollback: number | undefined
    try {
      const messages = await this.#host.listMessages(sessionID)
      state.streaming = streaming

      // A new user message resets the per-turn steering cap.
      const userMessages = messages.reduce((n, m) => (m.type === "user" ? n + 1 : n), 0)
      if (userMessages > state.userMessageCount) {
        state.userMessageCount = userMessages
        state.steersSinceUser = 0
        state.blockersThisUserTurn = 0
      }

      if (!state.seeded) {
        // Seed to the last user turn so enabling mid-session does not replay the
        // entire conversation on the first review.
        const lastUser = findLastIndex(messages, (m) => m.type === "user")
        state.reviewedCount = lastUser >= 0 ? lastUser : messages.length
        state.seeded = true
        if (lastUser < 0) return
      }

      // Incremental, as omp's is: the advisor keeps its own session, so it
      // already holds everything before the last reviewed message.
      const transcript = renderDelta(messages.slice(state.reviewedCount), {
        includeThinking: this.#config.includeThinking,
        maxChars: this.#config.maxTranscriptChars,
        wip: streaming,
      })
      rollback = state.reviewedCount
      state.reviewedCount = messages.length
      if (!transcript) return

      const advisors = this.#config.advisors.filter((a) => a.enabled)
      if (advisors.length === 0) return

      let notes = 0
      // Cleared before the pass, not after: a failed model call sets it from
      // inside the advisor, and clearing afterwards would erase exactly the
      // error /advisor status exists to report.
      state.lastError = undefined
      for (const advisor of advisors) {
        if (this.#abort.signal.aborted) return
        notes += await this.#reviewWith(advisor, sessionID, transcript, streaming, state)
      }
      state.reviews += 1
      state.lastNoteCount = notes
      state.lastReviewAt = Date.now()
    } catch (err) {
      if (this.#abort.signal.aborted) {
        state.lastOutcome = "aborted"
        return
      }
      state.passFailed = true
      state.lastError = (err as Error).message
      this.#host.log("warn", "advisor review failed", { sessionID, error: (err as Error).message })
    } finally {
      state.reviewInProgress = false
      state.backlog = Math.max(0, state.backlog - 1)
      if (state.passFailed) {
        if (rollback !== undefined) state.reviewedCount = rollback
      } else if (!this.#abort.signal.aborted) {
        state.reviewedSteps = Math.max(state.reviewedSteps, stepsAtStart)
      }
      // Persisted here rather than on the success path so a failed pass is
      // remembered too: /advisor status should survive a reload with the same
      // numbers, and a failure that vanishes on reload reads as "never ran".
      // Skipped when our own shutdown cancelled the pass: that snapshot is
      // half-finished, and writing it would persist a count this instance never
      // actually reached.
      if (!this.#abort.signal.aborted) this.#host.persistCounters?.(sessionID, this.#countersOf(state))
      const queued = state.queuedReview
      state.queuedReview = null
      if (queued) {
        scheduleMicrotask(() => void this.review(sessionID, queued.streaming))
      }
      this.#afterPass(sessionID, state, queued !== null)
    }
  }

  /**
   * One review by one advisor: the transcript update goes to the advisor's own
   * persistent session as its next user message, and the advisor works it like
   * any OpenCode agent — `read`/`grep`/`glob` to verify, `advise` to speak.
   * Notes arrive through {@link advise} while the run is live; this returns the
   * number delivered once the run settles.
   */
  async #reviewWith(
    advisor: AdvisorSpec,
    sessionID: string,
    transcript: string,
    streaming: boolean,
    state: SessionState,
  ): Promise<number> {
    const model = await this.#host.resolveModel(advisor.model ?? this.#config.model)
    state.lastModelWarning = model?.warning
    if (model?.warning) {
      const resolved = model.providerID + " at " + model.id
      const text = "Advisor: " + model.warning + ". Reviews are using " + resolved + " instead; run the advisor status command for details."
      await this.#noticeOnce(state, sessionID, model.warning, text)
    }

    const guard = state.guards.get(advisor.slug) ?? new EmissionGuard({ budgetPerUpdate: advisor.maxNotesPerUpdate ?? this.#config.sharedMaxNotesPerUpdate })
    state.guards.set(advisor.slug, guard)
    guard.beginUpdate()

    const fail = async (message: string): Promise<number> => {
      state.lastError = message
      state.passFailed = true
      state.lastOutcome = "advisor error"
      this.#host.log("warn", "advisor run failed", { advisor: advisor.name, error: message })
      await this.#noticeOnce(
        state,
        sessionID,
        `error:${message}`,
        `Advisor: the reviewer's run failed (${message}), so this session is getting no advice. /advisor status has the details.`,
      )
      return 0
    }

    let advisorSessionID: string
    try {
      advisorSessionID = await this.#advisorSession(sessionID, advisor, model)
    } catch (err) {
      if (this.#abort.signal.aborted) return 0
      return fail(`could not open the advisor session: ${(err as Error).message}`)
    }

    const pass: ActivePass = { advisor, sessionID, streaming, state, guard, deferred: [], raised: 0, routed: 0 }
    this.#activePasses.set(advisorSessionID, pass)
    let result: AdvisorRunResult
    try {
      result = await this.#runAdvisor(advisorSessionID, transcript)
    } finally {
      this.#activePasses.delete(advisorSessionID)
    }
    if (this.#abort.signal.aborted) {
      state.lastOutcome = "aborted"
      return pass.routed
    }
    // Withheld mid-turn notes are delivered once the advisor's run settles,
    // as omp flushes its deferred notes at the end of the update.
    if (pass.deferred.length > 0) {
      pass.routed += await this.#routeNotes(advisor, pass.deferred, sessionID, streaming, state, guard)
    }
    if (!result.ok) {
      await fail(result.error)
      return pass.routed
    }
    const count = pass.raised === 0 ? "no notes" : `${pass.raised} notes`
    // Mid-turn the reviewer is instructed to withhold non-blocking critique,
    // so "no notes (in progress)" is expected, not a clean bill of health.
    state.lastOutcome = streaming ? `${count} (in progress)` : count
    return pass.routed
  }

  /** The advisor's persistent session for this primary session, created on first use. */
  async #advisorSession(sessionID: string, advisor: AdvisorSpec, model: ModelRef | undefined): Promise<string> {
    const key = `${sessionID}\u0000${advisor.slug}`
    const known = this.#advisorSessions.get(key)
    if (known) return known
    const id = await this.#host.createAdvisorSession({ parentID: sessionID, advisor, model })
    this.#advisorSessions.set(key, id)
    this.#advisorSessionIDs.add(id)
    return id
  }

  /** Run one advisor update under the per-run deadline. */
  async #runAdvisor(advisorSessionID: string, text: string): Promise<AdvisorRunResult> {
    const controller = new AbortController()
    const onAbort = () => controller.abort()
    this.#abort.signal.addEventListener("abort", onAbort, { once: true })
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, this.#config.requestTimeoutMs)
    try {
      const result = await this.#host.runAdvisor({ advisorSessionID, text, signal: controller.signal })
      if (timedOut) return { ok: false, error: `timed out after ${Math.round(this.#config.requestTimeoutMs / 1000)}s` }
      return result
    } catch (err) {
      if (timedOut) return { ok: false, error: `timed out after ${Math.round(this.#config.requestTimeoutMs / 1000)}s` }
      return { ok: false, error: (err as Error).message }
    } finally {
      clearTimeout(timer)
      this.#abort.signal.removeEventListener("abort", onAbort)
    }
  }

  /**
   * The `advise` tool. Called from the advisor's own session while its review
   * runs; the answer is the tool result the advisor sees, so it is truthful
   * about what happened to the note, as omp's is.
   */
  async advise(advisorSessionID: string, input: { note?: unknown; severity?: unknown }): Promise<string> {
    const pass = this.#activePasses.get(advisorSessionID)
    if (!pass) return "Dropped: no review is in progress for this session."
    const note = typeof input.note === "string" ? input.note.trim() : ""
    if (!note) return "Dropped: empty note."
    const severity: Severity | undefined =
      input.severity === "nit" || input.severity === "concern" || input.severity === "blocker" ? input.severity : undefined
    pass.raised += 1
    // Mid-turn, only a blocker interrupts now; the rest wait for the run to end.
    if (pass.streaming && severity !== "blocker") {
      pass.deferred.push({ note, severity })
      return "Queued for the end of the turn. Do not re-raise."
    }
    const routed = await this.#routeNotes(pass.advisor, [{ note, severity }], pass.sessionID, pass.streaming, pass.state, pass.guard)
    pass.routed += routed
    return routed > 0 ? "Delivered." : "Dropped: already raised, over this update's budget, or not deliverable now."
  }

  /** Whether a session is one of this engine's advisor sessions (never reviewed or held). */
  isAdvisorSession(sessionID: string): boolean {
    return this.#advisorSessionIDs.has(sessionID)
  }

  /* ---------------------------------------------------------------- *
   * Catch-up
   * ---------------------------------------------------------------- */

  /**
   * Hold the primary until the reviewer has caught up, when it has fallen
   * `CATCH_UP_THRESHOLD` steps behind. Called from the `"context"` hook, which
   * OpenCode awaits before every primary model request, so the returned promise
   * is what pauses the agent. There is deliberately no time cap: the wait ends
   * when every step is reviewed, the turn is interrupted, the advisor is turned
   * off, or `CATCH_UP_RETRIES` retries of a failing pass have run out — at
   * which point the primary is stopped.
   */
  async catchUp(sessionID: string): Promise<void> {
    const state = this.#sessions.get(sessionID)
    if (!state || this.#abort.signal.aborted || this.#advisorSessionIDs.has(sessionID)) return
    if (state.catchUp) return state.catchUp.promise
    if (state.stepsEnded - state.reviewedSteps < CATCH_UP_THRESHOLD) return
    if (!this.#isEnabled(sessionID)) return
    if (this.#host.isClaimOwner && !this.#host.isClaimOwner()) return
    if (!(await this.#isReviewable(sessionID))) return
    // A concurrent hook for the same session may have started the wait meanwhile.
    if (state.catchUp) return (state.catchUp as CatchUp).promise

    let release!: () => void
    const promise = new Promise<void>((resolve) => {
      release = resolve
    })
    state.catchUp = { promise, release, failures: 0 }
    this.#host.log("debug", "advisor catch-up wait started", {
      sessionID,
      unreviewedSteps: state.stepsEnded - state.reviewedSteps,
    })
    this.#reviewNow(sessionID, state)
    return promise
  }

  /** Steps the reviewer has not yet covered. */
  unreviewedSteps(sessionID: string): number {
    const state = this.#sessions.get(sessionID)
    return state ? state.stepsEnded - state.reviewedSteps : 0
  }

  /** A pass that will not run counts its steps as handled, so nothing waits on it. */
  #skipReview(state: SessionState, reason: string): void {
    state.backlog = 0
    state.reviewedSteps = state.stepsEnded
    this.#releaseCatchUp(state, reason)
  }

  /** Review now instead of after the debounce or the mid-turn interval. */
  #reviewNow(sessionID: string, state: SessionState): void {
    if (state.timer) {
      clearTimeout(state.timer)
      state.timer = undefined
    }
    if (state.reviewInProgress) {
      state.queuedReview = { streaming: true }
      return
    }
    void this.review(sessionID, true)
  }

  /** Decide, after a pass, whether a catch-up wait ends, retries, or stops the primary. */
  #afterPass(sessionID: string, state: SessionState, queued: boolean): void {
    const wait = state.catchUp
    if (!wait) return
    if (this.#abort.signal.aborted) {
      this.#releaseCatchUp(state, "plugin shutting down")
      return
    }
    if (state.passFailed) {
      wait.failures += 1
      if (wait.failures > CATCH_UP_RETRIES) {
        void this.#stopForFailedReviews(sessionID, state, wait.failures)
        return
      }
      this.#host.log("debug", "advisor catch-up retrying a failed pass", { sessionID, failures: wait.failures })
      if (!queued) this.#reviewNow(sessionID, state)
      return
    }
    if (state.stepsEnded - state.reviewedSteps <= 0) {
      this.#releaseCatchUp(state, "caught up")
      return
    }
    if (!queued) this.#reviewNow(sessionID, state)
  }

  #releaseCatchUp(state: SessionState, reason: string): void {
    const wait = state.catchUp
    if (!wait) return
    state.catchUp = undefined
    this.#host.log("debug", "advisor catch-up wait ended", { sessionID: state.sessionID, reason })
    wait.release()
  }

  /**
   * The reviewer kept failing while the primary waited for it. Waiting longer
   * only spends the user's time, and continuing unreviewed is what the wait
   * exists to prevent, so the turn is stopped and the user decides.
   */
  async #stopForFailedReviews(sessionID: string, state: SessionState, failures: number): Promise<void> {
    const error = state.lastError ? ` Last error: ${state.lastError}.` : ""
    try {
      await this.#host.inject({
        sessionID,
        text:
          `Advisor: the reviewer failed ${failures} times in a row while the agent waited for it to catch up, ` +
          `so this run was stopped.${error} Fix the reviewer, or run /advisor off to continue without it.`,
        description: "advisor notice",
        metadata: { advisor: { kind: "notice", instance: this.#host.instance, raisedAt: Date.now() } },
        delivery: "queue",
        resume: false,
      })
    } catch (err) {
      this.#host.log("warn", "advisor stop notice failed", { error: (err as Error).message })
    }
    if (this.#host.interrupt) {
      try {
        await this.#host.interrupt(sessionID)
      } catch (err) {
        this.#host.log("warn", "advisor could not stop the session", { sessionID, error: (err as Error).message })
      }
    }
    this.#releaseCatchUp(state, "reviewer kept failing")
  }

  /* ---------------------------------------------------------------- *
   * Delivery
   * ---------------------------------------------------------------- */

  async #routeNotes(
    advisor: AdvisorSpec,
    notes: Note[],
    sessionID: string,
    streaming: boolean,
    state: SessionState,
    guard: EmissionGuard,
  ): Promise<number> {
    if (notes.length === 0) return 0
    const immuneActive = isImmuneActive(state, this.#config.immuneTurns)

    const admitted: { note: string; severity?: Severity; channel: DeliveryChannel }[] = []
    for (const note of notes) {
      const severity = note.severity
      // A nit is "a non-interrupting aside at the next step boundary". With no
      // running turn there is no next step boundary, so the aside can only
      // arrive stale: it would sit in the inbox until the next turn and then be
      // replayed as context about work already settled.
      if (!streaming && (severity === "nit" || severity === undefined)) {
        this.#host.log("debug", "advisor nit dropped on a settled turn", { advisor: advisor.name })
        continue
      }
      let channel = resolveChannel({
        severity,
        streaming,
        interruptImmuneTurnActive: immuneActive,
      })
      // Safety net: past the per-turn cap, a would-be steer becomes a queued
      // note, so the primary is never woken in an unbounded loop.
      if (channel === "steer" && state.steersSinceUser >= MAX_STEERS_PER_USER_TURN) channel = "queue"
      // A third blocker inside one user turn is churn far more often than signal;
      // dropping it here keeps it out of the guard's history, so the turn-end
      // pass can still raise it if it survives.
      if (severity === "blocker" && state.blockersThisUserTurn >= MAX_BLOCKERS_PER_USER_TURN) {
        this.#host.log("debug", "advisor blocker suppressed by per-turn cap", { advisor: advisor.name })
        continue
      }
      const decision = guard.admit(note.note, { rank: severityRank(severity), pending: channel !== "steer" })
      if (!decision.accepted) {
        this.#host.log("debug", "advisor note suppressed", { advisor: advisor.name, reason: decision.reason })
        continue
      }
      if (decision.displacedKey !== undefined) {
        const index = admitted.findIndex((a) => normalizeKey(a.note) === decision.displacedKey)
        if (index !== -1) admitted.splice(index, 1)
      }
      admitted.push({ note: note.note, severity, channel })
    }
    if (admitted.length === 0) return 0

    const groups: Record<DeliveryChannel, Note[]> = { steer: [], queue: [], preserve: [] }
    for (const entry of admitted) groups[entry.channel].push({ note: entry.note, severity: entry.severity })

    let steered = false
    const injectGroup = async (channel: DeliveryChannel, group: Note[]) => {
      if (group.length === 0) return
      // The host promotes only `steer` items at a step boundary; a `queue` item
      // waits for the next run, i.e. the end of the turn. So a mid-turn queued
      // note goes out as a steer without `resume`: it lands after the current
      // model call or tool round and never wakes an idle agent. Only a steering
      // note resumes; a preserve note waits for the next turn.
      const delivery: "steer" | "queue" = channel === "steer" || (channel === "queue" && streaming) ? "steer" : "queue"
      // After the primary has answered, omp prefaces the batch with its
      // boundary guidance: first time, then at most once every 50 turns.
      const boundary =
        !streaming &&
        (state.boundaryGuidanceTurn === undefined || state.completedTurns - state.boundaryGuidanceTurn >= BOUNDARY_GUIDANCE_TURNS)
      if (boundary) state.boundaryGuidanceTurn = state.completedTurns
      const send = (to: "steer" | "queue") => {
        const text = formatAdvisoryBatch(group, advisor.name, { boundary })
        return this.#host.inject({
          sessionID,
          text,
          description: channel === "preserve" ? "advisor note" : "advisor",
          metadata: { advisor: { slug: advisor.slug, name: advisor.name, severities: group.map((n) => n.severity ?? "nit"), instance: this.#host.instance, raisedAt: Date.now() } },
          delivery: to,
          resume: to === "steer" && channel === "steer",
        })
      }
      let id: string | undefined
      let sentAs: "steer" | "queue" = delivery
      try {
        id = await send(delivery)
      } catch (err) {
        // One flaky injection must not drop a whole pass's notes: a failed steer
        // is retried queued (nothing interrupted the agent, so `steered` stays
        // false and the immune window is untouched), a failed queue/preserve is
        // logged and left unrouted so a later pass can raise it again.
        if (channel !== "steer") {
          this.#host.log("warn", "advisor note was not delivered", { advisor: advisor.name, error: (err as Error).message })
          return
        }
        this.#host.log("warn", "advisor steer failed; retrying queued", { advisor: advisor.name, error: (err as Error).message })
        try {
          id = await send("queue")
          sentAs = "queue"
        } catch (retryError) {
          this.#host.log("warn", "advisor note was not delivered, even queued", { advisor: advisor.name, error: (retryError as Error).message })
          return
        }
      }
      if (id) {
        // A steered note lands in the running turn; a queued or preserved one
        // can wait minutes in the inbox. Track those, except blockers, so a
        // note that reaches the model late is filtered out of the request.
        if (sentAs === "queue" && !group.some((n) => n.severity === "blocker")) {
          this.staleNotes.track(id, formatAdvisoryBatch(group, advisor.name, { boundary }))
        }
        state.notesDelivered += group.length
        // Bridge to the OpenChamber panel: one JSONL line per delivered note,
        // keyed by the project directory like the claims. Fail-open inside.
        for (const entry of group) {
          appendNote(this.#host.notesBase ?? NOTES_BASE, this.#host.directory, {
            t: Date.now(),
            sessionID,
            advisor: advisor.name,
            severity: entry.severity ?? "nit",
            note: entry.note,
            instance: this.#host.instance,
          })
        }
      }
      for (const note of group) guard.markRouted(note.note)
      if (channel === "steer" && delivery === "steer" && id) steered = true
    }

    const deliveredBefore = state.notesDelivered
    await injectGroup("steer", groups.steer)
    await injectGroup("queue", groups.queue)
    await injectGroup("preserve", groups.preserve)

    // Page the user, because a note lands in the agent's context either way but
    // is invisible in OpenChamber's timeline until it renders advisor notices.
    if (state.notesDelivered > deliveredBefore && this.#config.notify !== "off" && this.#host.notify) {
      const top = admitted.reduce((best, entry) =>
        severityRank(entry.severity) > severityRank(best.severity) ? entry : best,
      )
      this.#host.notify({
        title: `Advisor · ${top.severity ?? "note"}`,
        body: admitted.map((entry) => entry.note).join(" · ").slice(0, 500),
        sessionID,
        directory: this.#host.directory,
        // A blocker pages even while the user is looking at OpenChamber. The
        // agent is still not woken: this only skips the away-only gate.
        showWhenFocused: this.#config.notify === "always" || top.severity === "blocker",
      })
    }

    const prior = state.priorNotes.get(advisor.slug) ?? []
    for (const entry of admitted) prior.push(entry.note)
    state.priorNotes.set(advisor.slug, prior.slice(-PRIOR_NOTE_LIMIT))

    if (steered) state.immuneTurnStart = state.completedTurns
    if (groups.steer.length > 0) state.steersSinceUser += groups.steer.length
    state.blockersThisUserTurn += admitted.filter((entry) => entry.severity === "blocker").length
    return admitted.length
  }

  /**
   * Tell the session once about a failure that otherwise leaves no trace. A
   * reviewer that cannot call its model is indistinguishable from one with
   * nothing to say, so "nothing happened" must never mean "it has been failing
   * all along". Keyed by message, so a *different* failure still speaks up.
   */
  async #noticeOnce(state: SessionState, sessionID: string, key: string, text: string): Promise<void> {
    if (state.noticesSent.has(key)) return
    state.noticesSent.add(key)
    try {
      await this.#host.inject({
        sessionID,
        text,
        description: "advisor notice",
        metadata: { advisor: { kind: "notice", instance: this.#host.instance, raisedAt: Date.now() } },
        delivery: "queue",
        resume: false,
      })
    } catch (err) {
      this.#host.log("warn", "advisor notice failed", { error: (err as Error).message })
    }
  }

  /* ---------------------------------------------------------------- *
   * Status and lifecycle
   * ---------------------------------------------------------------- */

  status(sessionID: string): {
    enabled: boolean
    /** The plugin-default state (`options.enabled`). */
    defaultEnabled: boolean
    /** The per-session override, when one was set; undefined means "use the default". */
    override?: boolean
    advisors: {
      name: string
      slug: string
      model?: string
      /** The roster entry's own switch. */
      rosterEnabled: boolean
      /** Whether this advisor will actually review (session on AND roster on). */
      active: boolean
      notes: number
      items: string[]
    }[]
    notesDelivered: number
    backlog: number
    /** Primary steps not yet covered by a successful review pass. */
    unreviewedSteps: number
    /** Whether the primary is currently held until the reviewer catches up. */
    catchingUp: boolean
    reviews: number
    lastReviewAt?: number
    lastNoteCount: number
    lastOutcome?: string
    lastError?: string
    /** Why the configured reviewer model could not be used, if it could not. */
    modelWarning?: string
    /** Live plugin instances in this process; more than one means duplicated reviewers. */
    instances: number
    /** This engine's instance id. */
    instance?: string
    /** False when another live instance owns reviewing for this directory. */
    owner: boolean
  } {
    const state = this.#state(sessionID)
    const enabled = this.#isEnabled(sessionID)
    return {
      enabled,
      defaultEnabled: this.#config.enabled,
      override: state.enabled,
      advisors: this.#config.advisors.map((a) => {
        const items = state.priorNotes.get(a.slug) ?? []
        return {
          name: a.name,
          slug: a.slug,
          model: a.model,
          rosterEnabled: a.enabled,
          active: enabled && a.enabled,
          notes: items.length,
          items,
        }
      }),
      notesDelivered: state.notesDelivered,
      backlog: state.backlog,
      unreviewedSteps: Math.max(0, state.stepsEnded - state.reviewedSteps),
      catchingUp: state.catchUp !== undefined,
      reviews: state.reviews,
      lastReviewAt: state.lastReviewAt,
      lastNoteCount: state.lastNoteCount,
      lastOutcome: state.lastOutcome,
      lastError: state.lastError,
      modelWarning: state.lastModelWarning,
      instances: liveInstances(),
      instance: this.#host.instance,
      owner: this.#host.isClaimOwner ? this.#host.isClaimOwner() : true,
    }
  }

  dispose(): void {
    this.#abort.abort()
    for (const state of this.#sessions.values()) {
      if (state.timer) clearTimeout(state.timer)
      this.#releaseCatchUp(state, "plugin shutting down")
    }
    this.#sessions.clear()
  }
}

/* ------------------------------------------------------------------ *
 * helpers
 * ------------------------------------------------------------------ */

function findLastIndex<T>(items: readonly T[], predicate: (item: T) => boolean): number {
  for (let i = items.length - 1; i >= 0; i--) if (predicate(items[i]!)) return i
  return -1
}

function isImmuneActive(state: SessionState, immuneTurns: number): boolean {
  if (state.immuneTurnStart === undefined || immuneTurns <= 0) return false
  return state.completedTurns < state.immuneTurnStart + immuneTurns
}

function normalizeKey(note: string): string {
  return note
    .toLowerCase()
    .normalize("NFKC")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
}

function scheduleMicrotask(fn: () => void): void {
  queueMicrotask(fn)
}

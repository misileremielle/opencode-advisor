import { describe, expect, test } from "bun:test"
import { advisorPermissions, advisorReadDirs } from "../plugin/index.ts"

// OpenCode's permission matcher: `*` matches anything, including `/`; the last matching rule wins.
const matches = (input: string, pattern: string) =>
  new RegExp("^" + pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$", "s").test(input)
const effect = (rules: ReturnType<typeof advisorPermissions>, action: string, resource: string) =>
  rules.findLast((rule) => matches(action, rule.action) && matches(resource, rule.resource))?.effect

describe("advisorPermissions", () => {
  const dirs = advisorReadDirs("/Users/u", "/var/folders/x/T")
  const rules = advisorPermissions(["read", "grep", "glob"], dirs)

  test("allows reads in skills and OpenCode's temp folder", () => {
    expect(effect(rules, "external_directory", "/Users/u/.config/opencode/skills/gog-readonly/*")).toBe("allow")
    expect(effect(rules, "external_directory", "/var/folders/x/T/opencode/*")).toBe("allow")
  })

  test("keeps the OpenCode config folder and everything else outside closed", () => {
    expect(effect(rules, "external_directory", "/Users/u/.config/opencode/*")).toBe("deny")
    expect(effect(rules, "external_directory", "/Users/u/Documents/*")).toBe("deny")
  })

  test("still denies tools it was not granted", () => {
    expect(effect(rules, "edit", "*")).toBe("deny")
    expect(effect(rules, "shell", "*")).toBe("deny")
    expect(effect(rules, "read", "*")).toBe("allow")
    expect(effect(rules, "advise", "*")).toBe("allow")
  })
})

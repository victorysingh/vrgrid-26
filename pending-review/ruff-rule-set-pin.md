# pending-review: pin the ruff RULE SET in `pyproject.toml`, not just the ruff version

**Status: PROPOSAL ONLY. `pyproject.toml` is not edited.** It is shared config that governs every
contributor's idea of "clean", so it follows the same rule as the `ci.yml` version pin: staged, not
applied unilaterally. `.github/CODEOWNERS` does not list `pyproject.toml`, so it has **no explicit
owner** — which is part of why this drifted unnoticed.

**Raised 2026-09-24**, as the root-cause item behind the R-d confusion. Separate from **LINT-DEBT**,
which is about the 69 errors on `jp/p99-alloc-fixes`; this is about why two people can run the same
command and disagree about the answer.

---

## The problem, precisely

`pyproject.toml` currently says:

```toml
[tool.ruff]
line-length = 100
extend-exclude = ["src/perception/frnet"]
```

There is **no `[tool.ruff.lint] select`**, so ruff falls back to its **built-in default rule set** —
and that default is a moving target across releases. Measured, not assumed:

| tree | ruff 0.12.0 default | ruff 0.16.8 default |
|---|---|---|
| `vrgrid26/main@8acbfe9` | **10 errors** | **All checks passed** |
| `jp/p99-alloc-fixes` | 3 errors | **69 errors** |

Both runs are correct. Both are `ruff check .`. **The rule set moved in both directions**: 0.16.8
drops the `E7xx` family from its defaults (which is why main's `E731`/`E741`/`E702` vanish) while
adding `I001`, `RUF100`, `PLW1510`, `RUF059`.

**This produced a real, costly error.** R-d was flagged as a false closure on the strength of "10
errors on main" measured with 0.12.0. The closure was correct; CI installs 0.16.8, under which main
genuinely is clean. The flag reached a drafted message to Shrestha and a public PR comment before it
was caught and retracted. **Nobody was careless — the config permits two honest people to reach
opposite conclusions and both be right.**

A version pin in `ci.yml` fixes CI's determinism. It does **not** fix this: a contributor on a
different ruff still sees a different answer locally, and the next version bump moves the goalposts
again. Only an explicit rule set makes "clean" mean one thing.

## ⚑ The part that makes this a decision, not a chore

Pinning the rule set is **not a no-op**, and whatever is chosen will very likely turn `main` red on
day one. Measured with ruff 0.16.8 across both trees:

| `select = …` | `main@8acbfe9` | `jp/p99-alloc-fixes` |
|---|---|---|
| *(0.16.8 default — today)* | **clean** | 69 |
| `["E", "F"]` — the classic baseline | **80 errors** | 107 |
| `["E", "F", "I"]` | **80 errors** | 130 |
| `["E", "F", "I", "RUF", "PLW"]` | **132 errors** | 228 |

So today's green `main` is an artifact of a **narrow default**, not evidence that the code satisfies
the conventional baseline. Selecting `E,F` — arguably the least opinionated choice available —
surfaces 80 pre-existing errors immediately.

**This audit deliberately does not recommend a set.** The cost differs by an order of magnitude
across the options and the trade-off is the team's.

## Two shapes the decision can take

1. **Fix-first.** Choose the set, fix everything it surfaces, then pin. CI stays green throughout;
   the fixing is front-loaded (80–132 on main, plus JP's branch).
2. **Baseline-then-ratchet.** Pin the set *and* record the current violations as a baseline — either
   `per-file-ignores` or a checked-in `# noqa` sweep — so CI goes green immediately and the backlog
   is burned down deliberately. Risk: a baseline nobody prunes becomes permanent, which is the same
   failure mode as an unpruned allowlist.

Either way, three things should land together or not at all: the **`select` list**, the **ruff
version pin in `ci.yml`**, and a line in `CLAUDE.md` saying the pinned pair is what "clean" means.
Pinning the version without the rule set leaves the trap half-open.

## Suggested shape, for the room to accept or replace

```toml
[tool.ruff.lint]
# Pinned explicitly so "clean" means one thing across contributors and across ruff
# releases. Without this, ruff uses its built-in defaults, which changed between
# 0.12 and 0.16 IN BOTH DIRECTIONS and let two people disagree about `ruff check .`
# while both were right (see pending-review/ruff-rule-set-pin.md).
select = [...]        # <- the decision
```

With the matching version pin in `.github/workflows/ci.yml` — **Shrestha's file, so that half stays
staged too.**

## What is NOT being claimed here

- That any particular rule set is correct. The table above is evidence for choosing, not a choice.
- That `main` has a quality problem. It passes the gate as configured; the point is that the gate's
  meaning is version-dependent.
- That this is urgent. It is cheap to leave alone and cheap to fix later — but it will keep producing
  the R-d-shaped disagreement until it is fixed, and that one cost real time to unwind.

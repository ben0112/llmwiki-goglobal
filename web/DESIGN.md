# DESIGN

The system serves a focused reading-and-progress tool. Restraint is the governing instinct: this is a surface a user stares at for an hour, so it must stay quiet and let the content carry. Subtle changes over bold ones.

## Theme

Dark, cool, near-neutral. Scene: a builder reading a dense ML course at their desk in the evening, wanting to focus and feel steady progress. The dark is calm, not "tool dark for cool's sake"; the content surface is a hair lighter than the sidebar so the reading column reads as the foreground.

Never pure `#000` / `#fff`. All neutrals carry a faint cool tint (hue ~265, chroma ≤0.01).

## Color

Strategy: **Near-monochrome.** Tinted zinc neutrals carry the entire surface. The only deliberate color is a green completion check. Primary action is ink (white), not a colored accent.

OKLCH tokens:

| Token | Value | Role |
|---|---|---|
| `bg` | `oklch(0.180 0.005 265)` | content surface |
| `panel` | `oklch(0.160 0.005 265)` | sidebar / second neutral layer (darker than content) |
| `raised` | `oklch(0.205 0.006 265)` | cards, popovers, active nav, inputs-at-rest contrast |
| `hair` | `oklch(0.265 0.006 265)` | hairline dividers, rails, progress tracks |
| `border` | `oklch(0.310 0.006 265)` | input borders, card borders |
| `ink` | `oklch(0.960 0.003 265)` | primary text |
| `muted` | `oklch(0.670 0.010 265)` | secondary text |
| `faint` | `oklch(0.510 0.010 265)` | metadata, labels, placeholders |
| `ok` | `oklch(0.700 0.090 155)` | the only color: a green check on a completed lesson / part |

This product is **near-monochrome**. The surface is zinc; the single deliberate color is the green completion check. Treat any other hue as a mistake until proven otherwise.

- **Primary actions are ink, not colored.** Buttons are `bg-ink text-bg` (white-on-dark), exactly like the original wiki's "Upload Sources". No colored primary button.
- **Progress and active state use neutral tints, not color.** Progress fills are `ink` at low opacity (~25%); the active lesson is `raised` background plus a small `ink` dot; "now" / "up next" are muted *text*, not colored chips.
- **`ok` (green) is reserved for completion only** — a done check on a lesson or part. It never appears on in-progress, active, or primary surfaces.
- **Lesson states differentiate by glyph, not hue:** completed (green check), in-progress (ink dot + raised bg), not-started (muted text), locked (lock icon).
- **Highlights are a faint neutral wash** (`ink` ~7%), no ring, no icon, no border; the note surfaces on hover and in the right-rail list. Marking text must never look like a button.
- **In-prose links** keep the original app's subtle blue (that predates course mode and stays consistent with the wiki). Course mode adds no new color beyond the green check.
- No streak counter, no flame, no gamification color.

## Typography

One family: **Inter** (system-ui stack fallback). No serif/display pairing; product UI does not need it, and KaTeX provides its own math face. Fixed rem/px scale, not fluid.

| Use | Size / weight |
|---|---|
| Page title (lesson / course) | 30–32px, bold, tracking-tight |
| Subtitle | 14–15px, muted |
| Body (reading column) | 15px, line-height 1.72, max ~68ch |
| UI labels / metadata | 10–11px, uppercase, tracking ~0.08–0.1em, faint |
| Nav / controls | 13px |
| Data (counts, %) | tabular-nums |

## Elevation

Flat. Structure comes from the two neutral layers (`panel` vs `bg`) and hairline borders, not shadows. Shadow is reserved for true overlays (hover note cards, popovers, command palette): one soft large shadow, nothing layered.

## Motion

150–250ms, ease-out (`cubic-bezier(.22,1,.36,1)`). Motion conveys state only: progress fills, panel swaps, hover reveals. No page-load choreography, no bounce, never animate layout properties.

## Components

- **Sidebar nav item** — tri-state status glyph (completed: accent check on `accSoft`; in-progress: partial accent ring; not-started: empty `border` circle) + label. States: default, hover (`raised/60`), active (`raised` bg + ink + medium weight). Quiz nodes carry a small bordered `quiz` chip.
- **Module group** — uppercase label, count (`2/4`), and a 1px progress hairline beneath.
- **Course header** — overall progress ring + title; clickable, routes to Overview.
- **Reading column** — breadcrumb + read-time + status row, title, subtitle, body, right-rail (`On this page` ToC with accent current-item, then `Your notes`).
- **Completion zone** — a single quiet region under a divider: primary "Mark complete", "Up next" target, `n / total`. Not a heavy card; arms on scroll-to-end.
- **Quiz card** — numbered, question, answer textarea, "Reveal answer key" gated until typed, answered counter + progress bar, and an MCP grade-handoff strip (neutral border, accent icon only).
- **Module list (Overview)** — a divided list with inline progress bars, never an identical card grid.
- **Sidebar timeline** — a single rail connects the units of a Part. Each unit is a node (completed/in-progress/locked glyph); the current unit expands to its sub-steps (Lesson, Checkpoint). Part labels are faint uppercase dividers on the rail.
- **Reading progress** — one quiet `ink`-low-opacity hairline fixed at the very top of the content area, driven by scroll position. No label, no section count.
- **Checkpoint** — rendered inline from an embedded ` ```checkpoint ` fenced block (same mechanism as ` ```mermaid `), one question at a time, with answer textarea, gated "Reveal key", and an MCP grade-handoff line. The block is the definition; answers and scores live in frontmatter/index, never in the block.
- **Gating** — locks are *soft*: a locked unit is dimmed with a lock icon and a "pass the checkpoint to unlock" hint, but remains clickable. Guidance, not a cage.

## Bans (in addition to the shared absolute bans)

- No top tabs layered over the sidebar (second nav axis).
- No serif/display fonts in this product.
- No second saturated accent; no gradient accents; no hero-metric cards; no identical card grids.
- No loud highlights (rings, washes with icons, colored side-stripes). Annotation stays subtle.

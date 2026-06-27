# Course mode — V0 spec

A course is a **mode of the existing wiki engine**, not a new app. Same renderer, same sidebar, same files on disk; it differs only in affordances (ordered hierarchy, progress, locks). See [PRODUCT.md](./PRODUCT.md) for purpose/users/principles and [DESIGN.md](./DESIGN.md) for the visual system. Visual reference: `docs/course-mode-mockup.html` (git-ignored scratch).

## V0 job

Render a course as markdown and **make progress legible.** That is the whole of V0. Checkpoints/quizzes and highlights/notes are specified but deliberately deferred (see Scope in PRODUCT.md).

## Structure on disk = course structure

The directory tree *is* the outline, mapped by depth. Today's `buildTreeFromDocs` nests one level and leans on `index.json`; replace it with a recursive builder over full paths.

```
/wiki/
  01-learning-primitives/        → Part I        (folder; prefix → order + "I")
    01-what-learning-means.md     → Lesson 1
    02-every-training-method.md   → Lesson 2
    03-value-based-learning.md    → Lesson 3
  02-deep-rl-lineage/            → Part II
    advanced/                    → a Module       (deeper folder → sub-group)
      01-ppo.md
```

- **Folder depth → role:** depth-0 folder = Part, deeper folder = Module, files = lessons. Renderer styles by depth, so 2-level (Part → lessons) and 3-level (Part → Module → lessons) both work.
- **Numeric prefixes (`01-`) own order** — natural sort. Strip the prefix and title-case for display.
- **Folders own structure; frontmatter owns state.** Never encode progress in folder/file names.

## Data model

- **KB-level:** a `type: course` flag switches the renderer into course mode. Overall progress is *derived* (completed lessons / total), not stored.
- **Per-lesson frontmatter:** `status: not_started | in_progress | complete`, `completed_at`, optional `est_minutes`. `type` defaults to lesson.
- **Reading progress** (sections read within a lesson) is derived from scroll position past H2s; persist only the coarse lesson `status`.
- **Locks are derived, soft:** a lesson is locked until the previous lesson is `complete`. Locked items are dimmed with a lock icon but remain clickable (guidance, not a cage).

## UI

**Sidebar (the only navigation — no top tabs):**
The wiki and course share **one sidebar renderer** with three zones; the course is the wiki base plus a progress layer.

- **Top — identity + tools.** Header (clickable → overview). Tools zone: course = Search; wiki = Search + Graph + Upload (Graph/Upload are wiki-only). The course header also carries the overall progress bar + `N/M` count.
- **Middle — content only.** Folder-driven outline, no meta pages (Overview is the header; Log/Settings live behind the header chevron). Wiki: Part/section labels + grouped pages threaded by a faint group guide. Course: the same, plus a **rail that fills to the current position** and per-lesson status nodes (done = green check, current = ink dot on `raised`, not-started = empty circle, locked = lock icon), plus per-Part counts.
- **Bottom.** **Sources flows as the end of the nav** (no divider above it, so it is not buried), then a single divider, then the **account** in its own zone (identity is separate from the wiki/course; never wall off Sources).

**Lesson page:**
- Breadcrumb (`Part I · The learning primitives`), title (the body's H1), subtitle.
- Body rendered by the existing `WikiContent` (remark-math + rehype-katex; the `\\`-over-escape fix already landed). Math, tables, code, lists come free.
- Right rail = "On this page" **reading tracker**: each section gets a green check as you scroll past it; current section highlighted; this is the in-lesson progress signal.
- A quiet reading-progress hairline fixed at the top of the content.
- Completion zone at the end: **Mark complete** (writes `status: complete`), "Up next", running `n / total`.

**Overview:**
- Just the rendered `overview.md` (prose) plus one understated **resume** strip ("Continue where you left off · %"). The sidebar is the syllabus; do not duplicate it here.

## Visual system

Near-monochrome zinc. The **only color is the green completion check**; primary actions are ink (white) buttons. No streak, no gamification color. Full tokens and rules in DESIGN.md.

## Build order (against real components)

1. **`type: course` flag** on the KB — the mode switch.
2. **Recursive `buildTreeFromDocs`** in `web/src/components/kb/KBDetail.tsx` — folders → Part/Lesson, rail + status glyphs from frontmatter, soft locks. Highest payoff, most unknowns; derisk first.
3. **Reading tracker + completion** in `WikiContent` — section checkmarks on scroll, Mark-complete writes frontmatter, sidebar/overall progress derive from it.

"Update the formatting" mostly means *use the existing `WikiContent` renderer* and tune prose styles, not rebuild markup. Out of scope for V0: checkpoint block renderer, notes/highlights, grading loop.

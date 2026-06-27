# PRODUCT

register: product

## Product purpose

A local-first knowledge environment where Claude (over MCP) generates and maintains long-form content the user reads, annotates, and progresses through. Two content modes share one engine:

- **Wiki** — a reference graph compiled from the user's own sources. You dip in and out.
- **Course** — an ordered journey with checkpoints and progress. You work through it.

Same renderer, same sidebar, same files on disk. "Course" is a *mode*, not a separate app or a forked UI.

## Users

Technical, self-directed builders and learners (the primary user builds AI applications). Fluent in Linear, Notion, Obsidian, Raycast. They expect density, keyboard speed, and no hand-holding, but they also want reading to feel calm and progress to feel real. They are skeptical of anything that looks like consumer edtech.

## Tone

Calm, precise, scholarly. The interface earns trust by disappearing into the task. A quiet study, not a gamified app. Progress is *felt* (rings fill, checkpoints clear) but never *celebrated* (no confetti, XP, mascots, streaks-as-pressure).

## Anti-references

- Duolingo / consumer edtech gamification: XP bars, confetti, badges, mascots, loud streaks.
- Generic SaaS dashboards: hero-metric cards, gradient accents, identical icon-card grids.
- Coursera / Udemy course chrome.
- Sterile admin-panel zinc with no point of view.

We are closer to a well-made reader (Readwise, Bear, Stripe Docs) than to any of these.

## Scope — V0

V0 does exactly one thing: **render a course as markdown and make progress legible.** Nothing else.

The single most important payoff (the thing chat cannot do): **come back the next day, see what you've read, and resume.** Progress is persisted to lesson frontmatter (`status`) on disk, so it survives across days/sessions/machines — not browser state. That cross-session resume is the core value; everything else serves it.

- In: folder-driven Part/Lesson hierarchy, the rail-as-progress-meter, the reading tracker that checks off sections on scroll, lesson completion, overall + per-Part counts, soft locks, the markdown overview with a resume action, and **code blocks with syntax highlighting** (it's just markdown — for reading PyTorch).
- Out (phase 2, deliberately deferred): **checkpoints/quizzes**, **highlights/notes**, **executable/graded coding exercises**, and **video embeds**. All valuable; none needed to prove the core. Do not build them in V0.

## Strategic principles

1. **Content is the hero; chrome recedes.** The lesson is the loudest thing on screen. Everything else is quiet.
2. **One engine, two modes.** Wiki and course differ in affordances (progress, checkpoints, completion), not in their visual language. Switching a KB's type must not feel like switching apps.
3. **Progress is felt, not gamified.** A filling ring and a cleared checkpoint, not points and prizes.
4. **Annotation is first-class.** Highlighting and noting what you read is core, not a bolt-on. It stays subtle: it marks the text, it does not decorate it.
5. **Claude drives, the UI renders.** State lives in files (frontmatter, `.quiz`, KB metadata); Claude mutates it over MCP; the UI reflects it. The user starting a turn is the trigger; there is no reverse-trigger magic.
6. **Familiar patterns are features.** Sidebar nav, breadcrumbs, a reading column with a right-rail ToC. Do not invent a second navigation axis (no top tabs over a sidebar) for flavor.

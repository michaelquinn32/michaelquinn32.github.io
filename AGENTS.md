# Agent Instructions for michaelquinn32.github.io

This file provides context for AI agents working on this Jekyll blog.

## Repository workflow

This is a personal, non-SOC repository. Commits do not require a Linear issue.
Use a feature branch and signed commits. The Delphos-configured `gw` does not
manage this repository; use `git worktree` with an explicit repository path.
Commit only the requested work and preserve unrelated changes.

## Site Overview

- **Platform**: Jekyll static site hosted on GitHub Pages
- **Theme**: Minimal Mistakes (remote theme)
- **Domain**: msquinn.com
- **Author**: Michael Quinn
- **Focus**: Engineering leadership, AI-native development, agentic workflows, and knowledge work

## Blog Post Style Guide

### Structure

Posts follow a consistent structure:

1. **Frontmatter** with title, excerpt, tags, image, header, and feature flags
2. **Introduction** section (`## Introduction`) that frames the problem or topic
3. **Body sections** with descriptive headings
4. **Conclusion** summarizing takeaways
5. **Tools Used** section listing AI tools, editors, and resources used

### Frontmatter Template

```yaml
---
title: "Title in sentence case with quotes"
excerpt: "One-line description of the post"
tags: [Tag1, Tag2, Tag3]
comments: true
modified: YYYY-MM-DD
show_newsletter_signup: true
use_math: false
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/image-name.jpg
header:
  image: /images/image-name.jpg
  caption: "Photo Credit: **Source**"
---
```

### Tone and Voice

The following analysis is drawn from close reading of the posts and the author's own prose. Use it to calibrate drafts, not turn the voice into a checklist. Current author feedback is evidence of that voice; a reviewer must question its interpretation of the guide before editing that voice away.

#### Core stance

Michael writes as a practitioner talking to a reader about something he cares about. He has a position, expresses enthusiasm and uses vivid language. Ground capability claims in experience or attributable evidence, and state the limitations that affect the argument. Do not replace this personal investment with a neutral report or make every observation carry a defensive disclaimer.

#### Sentence-level patterns

- **Sentence length follows the thought.** Short declarations supply emphasis; longer sentences carry explanation, qualification and conversational movement. Do not split every thought into terse assertions or impose a manuscript-wide sentence-length limit.
- **Rhetorical questions used sparingly, to pivot.** "Could you run a Ralph loop to write a design doc? A grant proposal? A regulatory filing? Not yet."
- **Concrete before abstract.** Michael almost always opens a section with a specific incident or number, then extracts the principle. He earns abstractions; he does not lead with them.
- **Semicolons over em dashes.** Where most writers reach for an em dash to insert an aside, Michael uses a semicolon or splits into two sentences.
- **Speak to the reader.** Signposting, direct address and explaining why an example matters are part of the voice. Mentions of the chapter, the writing process or what comes next are not defects in themselves. Judge what the passage does for the reader.
- **Natural rhythm.** Let a thought run when it needs room, then use a short sentence to land it. Do not manufacture alternation, fragments or a target distribution to demonstrate variety.

#### Paragraph-level patterns

- **Give each paragraph a focus.** Let its length follow the thought; do not make every paragraph the same size or a self-contained claim and caveat.
- **The pivot paragraph.** Michael often has a paragraph that begins with "But" or "And yet" that complicates the previous point. This is where his honesty about limitations lives. Example: "But the image is also misleading if you take it at face value."
- **Attribution is woven in, not dropped.** He names people and their ideas inline: "Ethan Mollick at Wharton recently made the same observation" rather than bare citations.
- **Lists are available when they help the reader.** Use bullets for steps, comparisons or points that benefit from being seen together, including parts of an argument. Keep the prose that develops and connects those points; do not replace it with a list by default.

#### Section-level patterns

- **Introductions give the reader a way in.** Personal experience often provides the opening. Broader framing can come first when it helps the reader understand why the experience matters. Do not force an anecdote onto an argument or invent one to satisfy a structure rule.
- **Body sections follow a pattern: experience, then framework, then implication.** He describes what happened, names the pattern or framework it illustrates, then says what it means for the reader.
- **Conclusions are forward-looking, not summary.** They acknowledge uncertainty ("I don't know where this ends"), identify tensions, and point toward what comes next. They never mechanically restate section headings.

#### Recurring rhetorical moves

- **The "irony" turn.** Michael frequently identifies an irony or paradox in the situation. "And here's the irony: the skills that are so often dismissed as 'soft' turned out to be the hard ones." This is a signature move.
- **The honest caveat.** Put a qualification where it changes the claim or the reader's decision. Do not append a counterweight to every positive statement or repeatedly announce that an example is not universal proof.
- **Quoting others to extend, not to defer.** He quotes Mollick, Karpathy, Cowen, etc. to build on their ideas, never as argument-from-authority. The quote opens a door; Michael walks through it with his own analysis.
- **Experience and evidence.** Firsthand accounts give the argument a concrete basis. Research can also introduce, challenge or support a point. Do not demand a personal anecdote before every research claim or turn an anecdote into comparative proof.

#### Narrative and prose structure

Essays and book chapters develop an argument for someone reading through them.
They are not engineering documents optimized for looking up isolated facts. Give
the reader a reason to keep following: a question, an experience, a tension or an
idea whose implications emerge as the prose develops.

A paragraph should have a focus without having to stand alone. Let one paragraph
prepare the next. Transitions, delayed explanations and returns to earlier ideas
can carry the argument. Repetition can provide a reminder or a payoff; identify
what a return contributes before calling it redundant.

Do not flatten this movement into a sequence of claims, qualifications and verdicts,
or make every section a miniature specification. Headings orient the reader; they
do not replace the prose that connects the ideas. A personal story is one way to
build this movement, not a required opening or permission to invent experience.

#### What Michael does NOT do

- No AI slop phrases: no "delve," "tapestry," "navigate the landscape," "in today's digital age"
- No excessive hedging: claims are stated directly, then bounded with specific qualifications
- No emoji in prose (occasional emoji in informal contexts like tweet embeds)
- No corporate jargon: no "leverage," "synergize," "drive innovation"
- No forced analogies or extended metaphors that aren't load-bearing
- Does not open posts with "time since I last posted" or similar weak hooks

#### Voice summary in one sentence

Michael talks to the reader as an engaged practitioner: personal, expressive and willing to argue a position, with evidence where it matters and room for a thought to develop.

### Code Snippets

- Use fenced code blocks with language specifiers (```python, ```yaml, etc.)
- Include relevant snippets that illustrate key concepts
- Keep snippets focused: show the important parts, not entire files
- Add brief explanations before or after code blocks

### Common Patterns

- **Transcript-style quotes**: For AI conversations, use code blocks with `@me:` and `@Gemini:` or similar
- **Embedded tweets**: Use Twitter/X embed code for social references
- **Internal links**: Reference other posts with relative URLs (`/blog/YYYY/MM/DD/slug/`)
- **Images**: Store in `/images/` with descriptive names

### Topics and Themes

Recent posts focus on:

- Agentic development workflows (Claude Code, Gemini, OpenCode)
- Context engineering and prompt design
- Practical AI tool usage and comparisons
- The "vibe coding" phenomenon with nuanced perspective
- Knowledge work and how AI is changing it

### Style

-  Avoid em dashes, as they can almost always be replaced by a semicolon
-  No Oxford comma
-  Prefer complete sentences
-  Use short sentences for emphasis, not as the default shape of every thought
-  Keep conversational asides when they carry the thought; split them when they obscure it
-  No bold taxonomy labels (e.g., "**Cost type:**", "**Benefit scaling:**") as section scaffolding; weave framework vocabulary into prose naturally
-  Avoid internal subsection headings (`###`) within short sections; continuous prose reads better unless the section is long enough to genuinely need navigation
-  Never fabricate anecdotes or lived experience; if the author hasn't provided a specific story, don't invent one
-  When referencing a person previously introduced by full title (e.g., "Our CTO Caleb Fenton"), subsequent references use first name only
-  Use "Delphos Labs" not "Delphos" when referring to the company
-  Attributed references over un-attributed statements; cite sources inline
-  Be precise about economic and technical terminology; don't use terms like "marginal analysis" unless the framework actually performs marginal analysis in the economic sense

## Project Structure

```
_posts/           # Blog posts in YYYY-MM-DD-slug.md format
images/           # Post images and assets
_layouts/         # Custom layouts (extends Minimal Mistakes)
assets/css/       # Custom CSS overrides
.github/
  scripts/        # Python scripts for automation
  workflows/      # GitHub Actions workflows
post-elements/    # Supporting docs (excluded from build)
scripts/          # Local dev/QA scripts (e.g. writing-qa/writing_stats.py)
mise.toml         # Tool versions + tasks (writing-QA pipeline)
.vale.ini         # Vale prose-linter config
styles/           # Vale styles: Blog/ + config/vocabularies/ committed; packages gitignored
```

## The book

The book *AI Engineering for Everyday Knowledge Work* lives in its own private repo,
`michaelquinn32/everyday-ai-book`, checked out locally at `~/personal/everyday-ai-book`.
It used to sit in a gitignored `book/` directory here. Do not recreate it in this repo.

The relationship runs one way. The book repo pins this repo as a `blog/` submodule and
harvests from `_posts/`; nothing here depends on the book. Two consequences worth
knowing when editing this repo:

- **Renaming or deleting a file in `_posts/` or `_pages/` breaks book citations.** The
  book cites posts by path, and its `mise run check-links` will catch the break, but only
  after someone runs it. If you rename a post, say so, or run the book's link check.
- **The book's submodule pointer is pinned to a commit.** New posts do not appear in the
  book until someone runs `git submodule update --remote blog` over there. That is
  deliberate; the pin records which version of a post a chapter harvested from.

The book's voice guide is this file. The "Blog Post Style Guide" and "Style" sections
below are canonical for both the blog and the book, so changes here propagate to the
book's drafting standard. That is intended; keep them in one place.

## Consulting offers

Client-facing offer materials live in their own private repo,
`michaelquinn32/consulting-offers`, checked out locally at `~/personal/consulting-offers`.
The Trucker Huss offer used to sit in an untracked `post-elements/trucker-huss-offer/`
directory here. Do not recreate it in this repo. **This repo is public; client-derived
material must never be committed to it.** `.gitignore` carries a guard entry.

The relationship is the same shape as the book's. The offers repo pins this repo as a
`blog/` submodule for the voice guide, and nothing here depends on the offers repo. The
"Blog Post Style Guide" and "Style" sections below are therefore canonical for three
consumers: the blog, the book, and the offers.

`_pages/services.md` is the public-facing version of this work and stays here. It is a
productized summary; the offers repo holds the client-specific artifacts.

## Newsletter System

The site has a MailerLite-powered newsletter with:

- **Sign-up form**: Embedded at the end of posts (controlled by `show_newsletter_signup: true`)
- **Automated draft creation**: GitHub Action creates MailerLite campaign drafts when new posts are pushed
- **Manual send**: Content is copied into MailerLite's block editor for final review

### Newsletter Workflow Files

- `.github/workflows/newsletter.yml` - Triggers on new posts in `_posts/`
- `.github/scripts/create_campaign.py` - Parses markdown and creates MailerLite draft

## Development

### Local Build

```bash
bundle exec jekyll serve
```

### Key Configuration

- `_config.yml` - Main Jekyll config
- `post-elements/` is excluded from the build
- `vendor/` is excluded from the build

### Tooling (mise)

Tools for this repo are managed with [mise](https://mise.jdx.dev/) via `mise.toml`. mise
manages standalone CLIs and language runtimes and defines the writing-QA tasks. This repo is
outside the auto-trusted Delphos paths, so mise needs a one-time trust:

```bash
mise trust      # one-time, in the repo root
mise install    # installs the pinned tools (e.g. Vale)
```

Prefer `mise run <task>` and `mise exec -- <tool>` over globally installed copies.

### Writing QA pipeline

Vale audits prose quality and AI-writing tells, deterministically and offline. It is the primary
signal. An optional commercial API can be used as a pre-publish sanity check. Both are
CLI/scriptable so an agent can shell out to them.

**Vale + style packages (the primary check).** Deterministic prose linting. Vale is pinned in
`mise.toml`; the style packages are declared in `.vale.ini` and fetched by `vale sync`.

```bash
mise run vale-sync                                   # one-time / after editing .vale.ini
mise run lint-prose _posts/2026-01-01-slug.md        # human-readable
mise run lint-prose-json _posts/2026-01-01-slug.md   # JSON for an agent to parse
```

Style packages: `write-good`, `proselint`, `alex`, `Readability` (Vale Hub), plus
[`ai-tells`](https://github.com/tbhb/vale-ai-tells) (pinned by release-zip URL), which detects
AI-writing tells (structure announcements, rhetorical self-answers, verb tricolons, sycophancy,
buzzwords). The AI-tells style is the closest thing to a "GPTisms" ruleset.

Some rules are disabled in `.vale.ini` on purpose, and should stay disabled:

- **`ai-tells.SemicolonUsage`** flags a semicolon joining a comma-free clause as an AI tell (at
  error level). This repo's voice guide does the opposite; it *prefers* a semicolon over an em
  dash. Leaving this on would flag the author's signature construction on nearly every paragraph.
- **`write-good.E-Prime`** flags every use of "to be." That is a fringe stylistic exercise, not a
  quality signal, and it is pure noise for this author.

Note on spelling: `Vale.Spelling` flags technical terms and names (Bazel, Terraform, Clippy,
Starlark, Dafny, Delphos, and so on). Maintain the accept-list at `styles/config/vocabularies/Blog/accept.txt`
rather than silencing the check wholesale.

Treat `ai-tells` findings as a prompt to reread, not an automatic rewrite: the ruleset assumes
a docs register, and this blog is more personal, so some flags (a deliberate tricolon, an
"of course") are fine on a second look. Fix the real tells; keep the voice.

**Statistical signal (deterministic; useful, not a gate).** There are two kinds of "statistical"
signal, and they are not the same. Deterministic readability and sentence-distribution metrics
are genuinely useful for finding places to improve, and carry no false-positive problem. Use
them. (The ML-based AI *detectors* are the ones to treat cautiously; see the commercial API note
below.)

Vale reports the standard readability grades (Flesch-Kincaid, Gunning-Fog, SMOG, LIX, and so on).
Their built-in thresholds are tuned for docs/marketing (grade <= 8, reading-ease >= 70) and do
not fit long-form analytical prose, so `.vale.ini` demotes them to `suggestion`: read the score
for trend, not compliance. A grade of ~10 is normal and fine here.

For the signal Vale cannot compute (it only has aggregate counts), use the stats script:

```bash
mise run stats _posts/2026-01-01-slug.md
```

It prints the sentence-length distribution (mean, standard deviation, min/max, and counts of
short and long sentences) plus reading grades, all offline stdlib. Read the distribution as
a prompt to listen to the prose. It does not establish voice quality or AI authorship.
There is no required standard deviation, proportion of short sentences or quota of sentences
over 40 words. Do not pad, split or combine sentences to hit a metric. The custom
`styles/Blog/AvgSentenceLength.yml` metric also flags a runaway average inside Vale; inspect
the actual passages before deciding whether anything needs to change.

**Commercial API (optional, pre-publish only).** No install; an HTTP call with an API key in an
env var (e.g. GPTZero at `api.gptzero.me/v2/predict/text`, or Originality.ai). These are
statistical AI-detectors with meaningful false-positive rates, so treat any score as advisory,
never a gate. Reserve it for a final pre-publish check given cost and latency; do not call it
per-revision. Verify the current endpoint/auth against the vendor's docs before relying on it.

(A local statistical detector, Binoculars, was evaluated and dropped: it is GPU-bound and has a
high real-world false-positive rate. Do not reintroduce it without a good reason.)

**Tools that do NOT exist (do not try to install them).** Some AI-generated install guides
reference `agent-style` and `anywhere-agents` as pip packages, and `vale-ai-writing` /
"ammil-industries" as the AI-tells Vale style. None of these are real. The AI-tells style is
`tbhb/vale-ai-tells`. If a new writing tool is proposed, verify it resolves to a real
repo/package before adding it to `mise.toml` or `.vale.ini`.

## Working with This Repo

When drafting blog posts:

1. Follow the frontmatter template exactly
2. Start with `## Introduction` 
3. Use descriptive section headings
4. Include code snippets where relevant
5. End with `## Conclusion` and `## Tools Used`
6. Be honest about what worked and what didn't

When automating:

1. Prefer parsing markdown directly over building Jekyll
2. Use environment variables for secrets (MAILERLITE_API_KEY, etc.)
3. Keep Python scripts self-contained with minimal dependencies
4. Output clear, actionable information in workflow logs

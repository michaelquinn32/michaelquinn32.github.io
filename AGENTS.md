# Agent Instructions for michaelquinn32.github.io

This file provides context for AI agents working on this Jekyll blog.

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

The following analysis is drawn from close reading of all posts from 2025-2026. Use it to calibrate drafts.

#### Core stance

Michael writes as a practitioner sharing what he learned, not as an authority declaring truths. The implicit contract with the reader is: "I tried this, here's what happened, here's what I think it means." He is genuinely excited about AI but never uncritical. Every capability claim is grounded in a specific experience, and every limitation is stated plainly.

#### Sentence-level patterns

- **Short declarative sentences for emphasis, followed by longer explanatory ones.** "The bottleneck shifts. It's no longer 'how fast can I write code?' It's 'how many well-scoped tasks can I run simultaneously, and how quickly can I evaluate results?'"
- **Rhetorical questions used sparingly, to pivot.** "Could you run a Ralph loop to write a design doc? A grant proposal? A regulatory filing? Not yet."
- **Concrete before abstract.** Michael almost always opens a section with a specific incident or number, then extracts the principle. He earns abstractions; he does not lead with them.
- **Semicolons over em dashes.** Where most writers reach for an em dash to insert an aside, Michael uses a semicolon or splits into two sentences.
- **No throat-clearing.** Introductions open with action or an observation, never with meta-commentary about what the post will cover. "Last week I produced a 100+ page technical design doc in about three days." Not: "In this post, I want to explore..."
- **Natural burstiness.** Sentence length varies widely; 5-word punches alternate with 30-word explanatory sentences. This is genuine, not performed.

#### Paragraph-level patterns

- **One idea per paragraph, rigorously.** Paragraphs rarely exceed 4-5 sentences.
- **The pivot paragraph.** Michael often has a paragraph that begins with "But" or "And yet" that complicates the previous point. This is where his honesty about limitations lives. Example: "But the image is also misleading if you take it at face value."
- **Attribution is woven in, not dropped.** He names people and their ideas inline: "Ethan Mollick at Wharton recently made the same observation" rather than bare citations.
- **Lists used for concrete items (tools, steps), never for arguments.** Arguments are always in prose.

#### Section-level patterns

- **Introductions ground in personal experience.** Every post opens with a specific moment: sitting with his daughter, finishing a design doc, reflecting on a month at a new job. The personal anchor comes first; the thesis emerges from it.
- **Body sections follow a pattern: experience, then framework, then implication.** He describes what happened, names the pattern or framework it illustrates, then says what it means for the reader.
- **Conclusions are forward-looking, not summary.** They acknowledge uncertainty ("I don't know where this ends"), identify tensions, and point toward what comes next. They never mechanically restate section headings.

#### Recurring rhetorical moves

- **The "irony" turn.** Michael frequently identifies an irony or paradox in the situation. "And here's the irony: the skills that are so often dismissed as 'soft' turned out to be the hard ones." This is a signature move.
- **The honest caveat.** After describing something that worked well, he immediately names the thing that didn't. "Even though this remains my top complaint, there's a caveat here."
- **Quoting others to extend, not to defer.** He quotes Mollick, Karpathy, Cowen, etc. to build on their ideas, never as argument-from-authority. The quote opens a door; Michael walks through it with his own analysis.
- **The lived-experience proof.** Claims are backed by "I did X, and Y happened" rather than "studies show." When he does cite studies, they support a point he has already established through experience.

#### What Michael does NOT do

- No AI slop phrases: no "delve," "tapestry," "navigate the landscape," "in today's digital age"
- No excessive hedging: claims are stated directly, then bounded with specific qualifications
- No meta-commentary: no "it's worth noting," "it's important to mention," "let's discuss"
- No emoji in prose (occasional emoji in informal contexts like tweet embeds)
- No bullet-point arguments; bullets are reserved for tool lists, steps, and specifications
- No corporate jargon: no "leverage," "synergize," "drive innovation"
- No forced analogies or extended metaphors that aren't load-bearing
- Does not open posts with "time since I last posted" or similar weak hooks

#### Voice summary in one sentence

Michael writes like an engineer who reads economics and management theory, grounds everything in what he built last week, and trusts the reader to handle complexity without being talked down to.

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
-  Keep sentences punchy
-  Avoid parenthetic asides or similar tangents within sentences; use two sentences instead
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
short and long sentences) plus reading grades, all offline stdlib. The key number is the
standard deviation. This voice prizes "burstiness," short punches alternating with long
explanatory sentences, so a healthy draft shows a double-digit stdev and a real supply of both
short (<= 8 words) and long (> 40 words) sentences. A high mean with a low stdev reads as
uniformly dense and is a mild AI tell. The custom `styles/Blog/AvgSentenceLength.yml` metric also
flags a runaway average inside Vale.

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

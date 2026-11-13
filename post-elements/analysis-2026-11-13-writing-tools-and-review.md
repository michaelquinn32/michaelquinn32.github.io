# Writing tools and review recommendations

**Date:** 13 November 2026  
**Status:** Recommendations for author review; not an installed workflow.

## Recommendation

Improve the existing writing loop rather than add another prose detector.
Teach drafting agents to develop the reader's understanding. Require reviewers
to explain what a passage does before recommending its removal. Keep factual
verification independent from judgments about readability and voice.

The first implementation should update guidance and pilot the review procedure.
Only then should we automate the mechanical work that proves useful. Do not
build a numerical score for voice, narrative quality or supposed AI authorship.

## Evidence and limits

The research used close readings of eleven complete primary texts by Scott
Alexander and Dean W. Ball, listed below. The sample spans technical explanation,
argument, personal reflection and operational policy. Search summaries located
sources; the analysis rests on the essays themselves.

This was not a representative sample or a reader-comprehension experiment.
The proposed mechanisms are editorial hypotheses. Popularity does not establish
that a particular technique caused readers to understand or remember a claim.
Neither writer's technical or political conclusions were independently audited.

The portable lesson is that variety serves a continuous argument. Examples can
establish a distinction before it receives an abstract name. A recurring image
can acquire new meaning. An objection can change the proposal instead of
appearing as a defensive sentence after every positive claim.

Do not imitate either writer's distinctive voice, import their political
positions or make their mannerisms mandatory. Alexander's explicit warning
against forced application belongs alongside his advice. Ball's description of
his usual writing as romantic and discursive is not an instruction to make
every practical explanation lyrical.

## Current repository findings

- `AGENTS.md` is the canonical voice guide. It already protects signposting,
  narrative development, expressive language and intentional repetition.
- `mise.toml` exposes Vale lint and writing statistics. Those tasks already
  provide enough mechanical infrastructure for a first editorial pilot.
- The output of `scripts/writing-qa/writing_stats.py` correctly rejects voice
  scores and sentence-length quotas. Its module docstring still calls low
  sentence-length variation a mild AI tell. Remove that conflicting claim;
  retain the measurements as descriptive information.
- At the initial check, this worktree's configured Vale packages were absent;
  lint stopped because `write-good` was missing. During the companion draft
  revision, `mise run vale-sync` successfully downloaded the five configured
  packages. Sync success does not establish that the prose passed lint.
- There is no blog-local reviewer configuration in this checkout. Do not claim
  that writing this recommendation has registered a command or installed agents.

## Changes to make

| Priority | Location | Concrete change | Evidence that it helps |
| --- | --- | --- | --- |
| First | `AGENTS.md` | Add the drafting and review guidance proposed below. Keep this as the voice authority. | Author review finds fewer unnecessary cuts and fewer paragraphs that merely restate a claim. |
| First | `scripts/writing-qa/writing_stats.py` | Correct the docstring's unsupported AI-authorship inference. Do not change scores to make drafts pass. | Script documentation and output describe the same limits. |
| First | Existing `mise` tasks | Sync configured Vale styles, then retain stdout, stderr and exit status during reviews. | The report distinguishes lint findings from a checker that never ran. |
| Next | Proposed `docs/writing-workflow.md` | Document the small drafting and review sequence below; route agents to it from `AGENTS.md`. Reference the voice guide rather than copy it. | A new session can follow the process without the drafting conversation. |
| After a pilot | Proposed `scripts/writing-qa/repetition_report.py` and a `mise` task | Locate repeated spans and nearby repeated sentence openings. Report locations and surrounding text; never rewrite or fail a draft for repetition alone. | Intentional motifs and quotations remain intact; the report helps find real accidental repeats. |
| After a pilot | Reviewer prompts | Add the reader-journey and edit-fidelity questions to existing reviewer roles. Do not create a fourth general-purpose critic by default. | Findings have distinct responsibilities and do not duplicate each other. |

The proposed repetition tool should exclude frontmatter and fenced code. Mark
quotations, captions and headings separately so they do not masquerade as prose
defects. Use configurable reporting limits, not a universal repetition threshold.
Exact matching cannot determine whether an idea is redundant; that remains a
reading task. Do not add an embedding service before a simple local report has
demonstrated value.

## Proposed guidance for drafting agents

This wording can be added to `AGENTS.md` after author approval:

> Before drafting, identify the reader, the question the piece explores and the
> change in understanding it should produce. Sketch the important turns in that
> progression; do not impose a fixed number of sections or an identical pattern
> on every paragraph. Let examples, evidence and objections create the need for
> the next idea. A personal anecdote is optional and must never be invented.
>
> Keep Michael's perspective present. Useful signposting, direct address, humor,
> enthusiasm and reflective asides are available tools, not defects by default.
> Vary what the prose does as well as how its sentences sound. Do not manufacture
> a sentence-length distribution or replace a recurring metaphor merely for variety.
>
> When returning to an idea, supply orientation, a new application, a consequence
> or a changed understanding. If a return contributes nothing, merge or remove it;
> synonyms alone do not solve conceptual duplication. Keep stable names for stable
> concepts. Put qualifications where they change the claim, not after every claim.
>
> Separate firsthand experience, author opinion, hypothetical examples and
> externally sourced facts. Check dates, scope and causal claims. Mark unanswered
> research questions explicitly. Fluent transitions are not evidence for the
> connection they imply. Borrow explanatory techniques from other writers, not
> their identity, content or distinctive phrasing.

## A small writing loop

1. **Agree on the job.** Record the intended reader, central question, author
   position and relevant constraints. Distinguish agreed requirements from ideas
   still being explored. A short note is enough.
2. **Prepare evidence.** Record substantive claims, their sources, what was read
   and what remains unverified. Author feedback can substantiate personal
   experience; it cannot verify an external research result.
3. **Draft the progression.** Build connected prose around the important turns.
   Do not optimize isolated paragraphs before the argument works as a whole.
4. **Review the whole piece.** Read from beginning to end before proposing local
   cuts. Then inspect argument, sources and reader experience under distinct
   assignments. Run mechanical checks as inputs, not verdicts.
5. **Validate proposed edits.** Test the criticism against the actual passage and
   its function. Preserve facts, scope, uncertainty and the author's position.
6. **Get author decisions.** Open the complete draft in tuicr. Read saved comments
   before editing, and retain the distinction between an unreviewed suggestion
   and an accepted change.
7. **Recheck the revision.** Confirm that requested changes are present and have
   not introduced contradictions elsewhere. Re-read the ending against the
   opening. Do not report a full new source audit if only changed claims were checked.

Store temporary review records outside public source material, or in a confirmed
ignored location. The blog is public. Do not copy book research, client material
or raw conversations into publishable notes merely because an agent can read them.

## Improve the reviews we already use

The book's requirements, evidence and style assignments provide a useful
division of responsibility. A blog workflow can use the same division without
copying private files or importing chapter-specific requirements.

### Argument and reader progression

Read the brief and full draft. Describe the actual sequence of reasoning before
criticizing it. Identify where a reader gains a distinction, encounters a
problem, tests an explanation or sees a new implication. Flag a missing link
only when you can name what the reader must supply to cross it.

An objection counts as addressed only if the text answers it, limits the claim
or changes the proposal. Mentioning the objection does not settle it. Conversely,
do not demand a counterargument in every section. The ending need not summarize;
it should make productive use of what the piece has established.

### Evidence and inference

Check the strength of the connection between evidence and claim, including
connections hidden in transitions and analogies. Distinguish a memorable example
from evidence about prevalence. Separate a proof of stated properties from a
claim of complete system correctness. Report inaccessible sources and failed
searches rather than replace them with confident secondary summaries.

### Voice, continuity and accessibility

Read the draft with the canonical voice and audience guidance, without prep notes
that explain away omissions. For a repetition concern, classify the function:
orientation, development, payoff or redundant restatement. Then distinguish
conceptual repetition from repetitive sentence shapes.

Check whether an example helps the intended reader make an inference. Assess a
metaphor by what it explains and where it stops fitting. Preserve a recurring
image when its later use adds meaning. Treat humor as a possible source of
clarity and companionship, not a joke quota. Do not neutralize personal investment
to produce an apparently more professional register.

### Required content of a finding

Use the existing result format when one exists. Add these elements to the
reasoning, rather than introduce an incompatible schema:

- Exact passage and source location.
- The passage's apparent job in the whole piece.
- The specific reader cost, unsupported inference or violated requirement.
- Whether the finding is a demonstrated defect or editorial judgment.
- The smallest useful change, including a no-change option when justified.
- What the proposed change could lose: evidence, scope, voice, orientation or payoff.

Reject findings that say only "AI-sounding," "too much meta-commentary," "repetitive"
or "too long." Reject clean rewrites that alter a fact or silently increase certainty.
Resolve overlapping findings once. Author decisions should inform later review,
without making previously accepted prose immune to new evidence.

## Pilot and validation

Pilot the guidance on a fixed draft and a small, separately labeled set of
editorial cases. Include accidental repetition, a useful recurring image, needed
signposting, unexplained terminology and an elegant but unsupported inference.
Include clean passages so that a reviewer can demonstrate restraint.

Ask the author to judge usefulness without labeling which version used the new
guidance when practical. Record accepted, rejected and revised findings, the
reasons for rejection and the cost of review. Track meaning-changing edits
separately from harmless preferences. Do not equate more findings, fewer lint
alerts or lower reading grade with better writing.

Test preservation of the author's argument separately from improvements in
readability. If learning or independent judgment is a goal, ask what the author
can explain and defend; a polished artifact alone cannot answer that question.

Stop adding rules when they produce formulaic prose or repetitive review demands.
Revise the guidance from actual author decisions before turning it into a reusable
tool. Installing packages or discovering reviewer definitions is not validation
of editorial quality.

## Source readings

All eleven essays below were read in full for the style analysis. Dates identify
the essay versions encountered; web texts may change. These are craft examples,
not authorities for every technical claim they contain.

| Writer | Primary text | Editorial observation |
| --- | --- | --- |
| Scott Alexander | [Nonfiction Writing Advice](https://slatestarcodex.com/2016/02/20/writing-advice/), 20 February 2016 | Explicit guidance on continuity, variety, repetition, concrete examples and the danger of forced rules. |
| Scott Alexander | [The Obligatory GPT-3 Post](https://slatestarcodex.com/2020/06/10/the-obligatory-gpt-3-post/), 10 June 2020 | Outputs create questions about capabilities before the essay widens to scaling. |
| Scott Alexander | [God Help Us, Let's Try To Understand AI Monosemanticity](https://www.astralcodexten.com/p/god-help-us-lets-try-to-understand), 27 November 2023 | Small examples and diagrams build an intuition before introducing difficult abstractions. |
| Scott Alexander | [In Search Of AI Psychosis](https://www.astralcodexten.com/p/in-search-of-ai-psychosis), 26 August 2025 | Different sections test different explanations, rather than repeatedly announce uncertainty. |
| Scott Alexander | [Shameless Guesses, Not Hallucinations](https://www.astralcodexten.com/p/shameless-guesses-not-hallucinations), 16 March 2026 | Escalating school-test examples carry both explanation and humor. |
| Scott Alexander | [An Open Letter To Steven Pinker On AI](https://www.astralcodexten.com/p/an-open-letter-to-steven-pinker-on), 6 October 2026 | Explicit counterarguments and recurring jokes organize a polemic; ridicule is not a portable requirement. |
| Dean W. Ball | [On AI "Black Boxes"](https://www.hyperdimensional.co/p/on-ai-black-boxes), 29 August 2024 | Technical explanation connects to an institutional demand for explainability. |
| Dean W. Ball | [Out of Thin Air](https://www.hyperdimensional.co/p/out-of-thin-air), 14 August 2025 | A changed assumption creates a practical recommendation; the author distinguishes this register from his usual discursive style. |
| Dean W. Ball | [How I Approach AI Policy](https://www.hyperdimensional.co/p/how-i-approach-ai-policy), 18 September 2025 | Forecast confidence and grounds for action are distinguished; vivid scenarios can still outrun their qualifications. |
| Dean W. Ball | [2023](https://www.hyperdimensional.co/p/2023), 25 March 2026 | Personal history explains intellectual commitments and subsequent decisions. |
| Dean W. Ball | [The Beauty of Something Grown](https://www.hyperdimensional.co/p/the-beauty-of-something-grown), 16 October 2026 | A recurring image develops from technical framing to political and emotional implication; eloquence does not establish the technical premise. |

## Scope of this delivery

This document records proposed changes. It does not change `AGENTS.md`, install
new reviewers, add scripts or alter the book repository. The companion blog
revision is separate work and should not be mistaken for a completed pilot of
these recommendations.

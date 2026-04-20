# Research Notes: Everything has a cost. Everything has a benefit.

## Article Outline

### I. Introduction

#### The hook: 3x PRs, same headcount

- Our team tripled the number of PRs shipped over the last couple of months. No new hires. Same people, same hours. The tools caught up to the ambition.
- The inner loop is close to automatic for the class of tasks we throw at it. I work in OpenCode all day. We have a well-defined set of agents and skills covering the full developer workflow. We are converging on a single entrypoint: the agent reads the task, gathers context, writes tests, implements, commits. "Agent, do work."
- The surface of development has stabilized. Six months ago every week brought a new paradigm shift. Now the tools are good enough that I've stopped thinking about the tools and started noticing something else.

#### The surprise: friction moved, it didn't disappear

- What I actually do all day: babysit tmux tabs. I kick off reviews, 20+ a day, context-hop between agent summaries, read proposed comments, render verdicts. I'm an engineering manager with limited time, and the bottleneck is no longer code production. It's my attention.
- We built an automated review loop with specialized subagents (security, architecture, test coverage, comment accuracy). It works. And the new friction is: reading the output, making the judgment calls, resolving the human-in-the-loop moments. The time cost shifted from producing code to evaluating code.
- Other frictions are accumulating. More PRs means more merge conflicts. More features means more surface area to monitor. More shipped code means more things that can break in production. The volume turned up and the pain points migrated.

#### The problem this article solves

- I've spent the last few months reading dozens of versions of the abundance argument. Jevons Paradox for software (Levie, Hof). Lights-out codebases (Su). The death of pull requests (Laforgia). Outcome engineering (Ondrejka). Structural economic change (Imas). Vibe coding in production (Schluntz).
- They all describe pieces of what I'm living. But none of them give me the tools to answer the question I face every day: which frictions should I try to eliminate, and which frictions are actually load-bearing?
- That's what marginal analysis is for. Everything has a cost. Everything has a benefit. When the cost structure changes, you don't just do the same thing cheaper. You rethink what's worth doing at all.

#### The thesis

- This article proposes a framework for thinking about every phase of the software development lifecycle when software is abundant.
- Human time is the currency. For each phase, we ask: what does it cost in human attention? How does that cost distribute (thin-tailed or fat-tailed)? And how do the benefits scale (sub-linear, linear, or super-linear)?
- The answer tells you where to automate, where to delegate, and where to invest your most scarce resource: your own judgment.
- We'll apply this framework to the inner loop (task definition through commit) and the outer loop (milestones through product-market fit). Along the way we'll draw on economics, security research, and the lived experience of managing a team through this transition.

### II. The Model

#### Human time as currency

- All costs and benefits measured in hours of skilled human attention
- This is not developer time broadly; it's the subset of time requiring judgment, not just presence
- Ondrejka's reframe: "Cost not time. Capacity not backlog." When compute is cheap, human attention is the binding constraint.

#### Axis 1: Cost distribution

- **Thin-tailed costs**: normally distributed errors that shrink with more AI capability. Broken dev environments, formatting, routine bugs, boilerplate. Analogous to residual errors in a regression model shrinking as you add data. Bounded downside. Self-correcting. This is where the productivity story lives.
- **Fat-tailed costs**: power-law distributed risks that do not shrink with abundance. Security vulnerabilities, architectural debt in core systems, wrong product strategy. May GROW with abundance because more software = more surface area. Unbounded downside. A single breach can cost more than all productivity gains combined.
- The regression analogy in detail: adding more data to a well-specified model drives normally distributed errors toward zero. But outliers, misspecification errors, adversarial inputs; those are fat-tailed. No amount of data fixes a fundamentally wrong model. Same with software: no amount of AI-generated code fixes a fundamentally wrong architecture or a security vulnerability that an attacker finds first.

#### Axis 2: Benefit scaling

- **Sub-linear (diminishing returns)**: commodity features, test coverage beyond a threshold, docs for stable systems. The 10th login flow is worth less than the 1st.
- **Linear (constant returns)**: routine bug fixes, deployment frequency improvements, per-service monitoring. Predictable, proportional value.
- **Super-linear (compounding returns)**: shots on goal for product-market fit (Schluntz: "four times as many lessons"), security posture (insures against increasingly large tail risks), core architecture quality (network effects; everything depends on it), specification quality (patterns compound), team learning velocity (the meta-skill of rapid iteration is itself super-linear).

#### The 2x2 and the reallocation principle

- Present the matrix (thin/fat x sub/linear/super)
- The top-left (thin-tailed, sub-linear): automate fully. Formatting, boilerplate, commodity features. Don't spend human time.
- The bottom-right (fat-tailed, super-linear): maximum human investment. Core architecture security, product strategy, threat modeling. This is the highest-leverage work.
- The reallocation principle: AI makes it possible to shift human time from top-left to bottom-right. Abundance makes it urgent, because the fat-tail surface area grows with volume.
- Jack Clark / Anthropic as proof: even at the company building the most capable AI, engineers are shifting from code production to dashboards, monitoring, measurement. That's the reallocation happening in real time.

#### The tax metaphor

- Fat-tailed risks are taxes on abundance. They don't get cheaper with scale. Security is a tax. Architectural integrity is a tax. The bill grows with the volume of software you produce.
- You can avoid paying thin-tailed costs (AI drives them to zero). You cannot avoid paying fat-tailed costs. They must be paid in human attention.

### III. The Inner Loop

For each phase: classify on both axes, then derive the rational response.

#### A. Task Definition / Requirements

- Cost type: FAT-TAILED. Wrong requirements waste everything downstream. A feature nobody wants costs the same to build as a feature everyone wants.
- Benefit scaling: SUPER-LINEAR. Good specs compound. Each well-defined task establishes patterns, vocabulary, constraints that make subsequent specs more effective. Ondrejka's "Code the Constitution" (Principle 10).
- Rational response: MAXIMUM HUMAN INVESTMENT. This is the new bottleneck. "Coding was never the bottleneck" (Agoda/InfoQ). AI can help research, explore, draft. The human must decide what to build and why.
- Cite: OpenAI "Harness Engineering", Agoda study, Ondrejka, your "Knowledge Stack" and "Agents All the Way Down"

#### B. Test-Driven Development

- Cost type: THIN-TAILED (to produce). AI writes tests well, especially end-to-end tests from specs. The cost of TDD has collapsed.
- But TDD serves as a FAT-TAIL MITIGATOR. Tests are the verification layer that catches catastrophic problems before deployment.
- Benefit scaling: SUPER-LINEAR up to a threshold, then sub-linear. Going from 0% to 80% coverage is enormously valuable. Going from 95% to 99% has diminishing returns for most systems.
- Rational response: DO MUCH MORE TDD. The cost dropped but the benefit (especially as fat-tail mitigator) stayed the same or increased. Tests become the primary artifact the human cares about. Schluntz: "the only part of the code I'll read is the tests."
- Cite: Kent Beck on TDD as "superpower", Schluntz, Laforgia's T*D, Ondrejka Principle 2 ("vibes are not tests")

#### C. Implementation

- Cost type: THIN-TAILED. Routine bugs, boilerplate, formatting; all driving toward zero. "The length of tasks AI can do is doubling every seven months" (Schluntz).
- Benefit scaling: SUB-LINEAR for commodity features, LINEAR for novel features. The 10th CRUD endpoint is worth less than the 1st.
- Rational response: AUTOMATE / DELEGATE. Let AI have it. This is the Jevons response: since it's cheap, build things that previously weren't worth building. Ondrejka Principle 7: "Code is the cheapest resource. Build to answer questions. Build to test hypotheses."
- Cite: Schluntz, Levie, Tal Hof, Ondrejka, your "Agents All the Way Down"

#### D. Code Review

Split into three sub-phases with different profiles:

**Style and logic review**
- Cost: THIN. Benefit: SUB-LINEAR. Automate fully. AI reviewers, linters, formatters. No human time.

**Security review**
- Cost: FAT-TAILED. A missed SQL injection in AI-generated code is catastrophic. Veracode: 45% OWASP failure rate. Shukla et al.: 37.6% increase in critical vulns through iterative generation. "Slopsquatting" as a novel attack vector created by AI abundance.
- Benefit: SUPER-LINEAR. Security investment insures against increasingly large tail risks. Going from "no security program" to "automated scanning" is valuable; going further to "threat modeling + incident response" is disproportionately more valuable.
- Rational response: INCREASE human security review investment. Not the same review (reading every line) but differently structured: automated gates for known patterns, human judgment for novel threat modeling. Ondrejka Principle 15: "Speed is dangerous without brakes."
- Cite: Tal Hof, Veracode, UTSA/slopsquatting study, Shukla et al., Checkmarx, Black Duck

**Architectural review**
- Schluntz's trunk vs. leaf distinction. Leaf node tech debt: THIN-TAILED (contained, won't compound). Core architecture debt: FAT-TAILED (everything depends on it).
- Benefit: SUPER-LINEAR for core (network effects within codebase), SUB-LINEAR for leaves.
- Rational response: Human review concentrated on core architecture; let AI and tests handle leaf nodes.
- Cite: Schluntz, your "Agents All the Way Down" on agents reviewing agents

#### E. The PR Process Itself

- The meta-question: does the pull request as a process artifact still make sense?
- Su: lights-out codebases, no human sees code. Laforgia: PRs designed for open source (untrusted strangers), not teams. 130,000 hours wasted on zero-comment PRs.
- The marginal analysis: the PR is a thin-tailed-cost mechanism (catches mostly style and simple logic issues, per Bacchelli & Bird) applied as if it catches fat-tailed risks. The cost is high (86-99% of lead time is waiting). The benefit is mostly sub-linear (knowledge transfer, not bug-catching).
- Rational response: Restructure. Automate the thin-tailed review entirely (AI reviewers, CI gates). Reserve human-in-the-loop for fat-tailed decisions (architecture, security, product direction). Ship/Show/Ask model. Laforgia's T*D.
- Your lived experience: the friction of 20+ reviews/day isn't from reading code. It's from context-hopping between agent verdicts. The process needs to match the new cost structure.

### IV. The Outer Loop

#### A. Shots on Goal

- Benefit scaling: SUPER-LINEAR. Schluntz: "if we can collapse that time down to 6 months, engineers are going to be able to learn from four times as many lessons in the same calendar time." Each experiment benefits from everything learned in previous ones. Learning compounds.
- The inner loop being cheap enables outer loop acceleration. When building a feature costs days instead of weeks, you can test hypotheses that were previously too expensive to explore.
- This is where the Jevons argument is most exciting: not just "more code" but "more experiments, more learning, faster convergence on what works."
- Cite: Schluntz, your "Thesis" post on DS bottleneck

#### B. The Maintenance Multiplier

- The 60/60 rule: 60% of lifecycle costs are maintenance. 3x the PRs means 3x the maintenance surface.
- Cost type: THIN per instance (each bug fix is routine) but aggregate grows linearly or super-linearly with codebase size and complexity.
- The dark side of Jevons: abundance creates its own costs. More code = more dependencies, more things to monitor, more potential for interaction effects.
- AI helps (bug triage, refactoring, dependency updates) but judgment about what to maintain, deprecate, or rewrite is fat-tailed.
- Cite: 60/60 Rule, Shayon, your 3x PR experience

#### C. Monitoring and Observability

- Clark / Anthropic: engineers shifting to dashboards, metrics, interpretation. This is the fat-tail reallocation in action.
- Benefit scaling: FAT-TAIL MITIGATOR. You can't review all the code; you CAN monitor all the outcomes. Monitoring is how you catch fat-tailed problems (outages, security incidents, performance degradation) without reading every line.
- DORA tensions: surface metrics improve (faster lead time, higher deployment frequency) but stability can degrade. Hivel.ai: 7.2% drop in delivery stability with 25% AI adoption increase.
- Rational response: invest heavily. This is where the "tax" on abundance is most directly payable.
- Cite: Clark/Ezra Klein, DORA 2025, Hivel.ai, Charity Majors

#### D. Product-Market Fit and Strategic Learning

- Cost type: FAT-TAILED. Most products fail. A few succeed enormously. The distribution of product outcomes is power-law.
- Benefit scaling: SUPER-LINEAR when you find it. PMF is the single highest-leverage outcome in the outer loop.
- When building is cheap, the scarce thing becomes knowing WHAT to build. Product sense, customer understanding, domain expertise. Imas's structural change: as commodity production gets cheap, spending shifts to sectors with high income elasticity. In the SDLC, those sectors are product strategy and customer insight.
- Cite: Imas "What will be scarce?", your "Thesis" post, Ondrejka Principle 1 ("Agents explore paths; humans choose the destination")

### V. The Economics of Abundance

#### Jevons across both loops

- Cheaper software means more software gets built. GitHub pushes up 35% YoY, new iOS apps up 50% (Tal Hof data). Demand is elastic.
- But not infinitely elastic. Binding constraints shift: human attention, maintenance burden, organizational capacity to absorb change.

#### Structural change

- Imas's framework: as commodity production gets cheap, income effects drive spending toward sectors with high income elasticity. The "relational sector" of the SDLC: design, product sense, architectural judgment, security expertise.
- Consumer elasticity of demand for software: the Odd Lots discussion. Evidence says demand is elastic, but the TYPE of demand shifts.

#### The security tax on abundance

- More software = more attack surface. Tal Hof's extensive vs. intensive growth.
- The security profession grows BECAUSE of AI, not despite it. Jevons for security: cheaper offense AND defense, but total volume of security work grows.
- Fat-tailed security costs are the clearest example of a tax that can't be flushed by abundance.

#### The attention economy of engineering

- HBR: AI doesn't reduce work, it intensifies it. Multitasking, workload creep, collapse of stopping points.
- "Brain fry" (HBR): AI increases cognitive load. Human attention is the true bottleneck.
- Your experience: 3x PRs didn't give you 3x free time. It gave you 3x the judgment calls.

### VI. What Stays Expensive (The Bottom-Right Quadrant)

- Task definition and requirements (fat-tailed cost, super-linear benefit)
- Core architectural decisions (fat-tailed cost, super-linear benefit)
- Security threat modeling (fat-tailed cost, super-linear benefit)
- Product strategy and customer insight (fat-tailed cost, super-linear benefit)
- Monitoring and observability systems (fat-tail mitigator, linear-to-super-linear benefit)
- These are management skills. Mollick: "the skills dismissed as soft turned out to be the hard ones." Your "Agents All the Way Down": the best weeks are the weeks with the best task descriptions, not the best code.
- Ondrejka's vocabulary: "Creation not code. Cost not time. Capacity not backlog. Certainty not vibes."

### VII. Conclusion

- Return to the title: everything has a cost, everything has a benefit.
- The marginal revolution in software: constantly ask "given the new cost structure, should we do more of this, less of this, or something different entirely?"
- The engineers who thrive reallocate human time to the bottom-right quadrant: fat-tailed costs with super-linear benefits.
- The teams that thrive accelerate the outer loop, taking more shots on goal.
- Close with the lived reality: 3x PRs is not the end state. It's the beginning of learning where human judgment matters most.
- Schluntz's exponential: "Remember the exponential. It's okay today if you don't vibe code, but in a year or two, it's going to be a huge disadvantage."


## Core Analytical Framework: Thin Tails vs. Fat Tails + Benefit Scaling

The central model for the article operates on two axes.

### Axis 1: Cost Distribution (Thin-Tail vs. Fat-Tail)

When AI makes software abundant, costs across the SDLC separate into two categories.

**Thin-tailed costs** (normally distributed): broken dev environments, formatting, routine bugs, boilerplate, simple test failures. These get driven toward zero as AI capability increases, analogous to errors shrinking as more data is added to a regression model. Downside is bounded. Self-correcting. This is where productivity gains live.

**Fat-tailed costs** (power-law / catastrophic): security vulnerabilities, architectural debt in core systems, wrong product strategy, data breaches, compliance failures. These do NOT shrink with abundance. They may GROW, because more software = more attack surface, more code nobody understands, more places for catastrophic failures to hide. Downside is unbounded. A single breach can cost more than all productivity gains combined.

### Axis 2: Benefit Scaling (Sub-linear, Linear, Super-linear)

When you invest more human time in a phase, benefits scale differently:

**Sub-linear (diminishing returns)**: commodity features (10th login flow < 1st), test coverage beyond ~80%, documentation for stable systems. Each additional unit yields less.

**Linear (constant returns)**: routine bug fixes, deployment frequency improvements, per-service monitoring. Each unit provides roughly the same value.

**Super-linear (compounding returns)**: shots on goal for PMF (learning compounds), security posture improvement (insures against larger tail risks), core architecture quality (network effects within codebase), specification quality (patterns compound), team learning velocity (meta-skill of iteration is itself super-linear).

### The 2x2

|  | Thin-tailed cost | Fat-tailed cost |
|--|--|--|
| Sub-linear benefit | AUTOMATE FULLY (formatting, boilerplate) | GATE AND MONITOR (compliance checklists) |
| Linear benefit | DELEGATE + SPOT CHECK (routine bugs, tests) | INVEST PROPORTIONALLY (per-service monitoring, scanning) |
| Super-linear benefit | AI-ASSISTED, HUMAN-LED (arch patterns, learning systems) | MAXIMUM HUMAN INVESTMENT (core arch security, product strategy, threat modeling) |

### The Reallocation Principle

Rational engineers shift human time from top-left (thin-tail, sub-linear) toward bottom-right (fat-tail, super-linear). AI makes this possible (handles thin-tail work) and urgent (fat-tail surface area grows with abundance).

### The Tax Metaphor

Fat-tailed risks are "taxes" on abundance. They don't get cheaper with scale; they may get more expensive. Security is a tax. Architecture is a tax. You can't flush these by producing more software. They must be paid in human attention, and the bill grows with volume.

**SDLC phases by tail type**:
- Task definition: FAT (wrong requirements waste everything)
- TDD: THIN cost to produce, but serves as FAT-TAIL MITIGATOR
- Implementation: THIN (routine, drives toward zero)
- Code review (style/logic): THIN (automate entirely)
- Code review (security): FAT (missed vuln = catastrophic)
- Code review (architecture): FAT for core, THIN for leaf nodes (Schluntz)
- Monitoring/observability: FAT-TAIL MITIGATOR (Clark/Anthropic)
- Product-market fit: FAT (most products fail; a few succeed enormously)
- Maintenance: THIN per instance, but aggregate grows with abundance

## Annotated Bibliography

### Primary Sources (Provided by Author)

1. **Erik Schluntz, "Vibe Coding in Prod Responsibly"** (Conference talk, 2026)
   - Video: https://www.youtube.com/watch?v=fHWFF_pnqDk
   - Schluntz (Anthropic researcher, co-author of "Building Effective Agents") argues that the exponential growth in AI task capability (doubling every 7 months) will force engineers to adopt "vibe coding" practices even in production. His framework: focus AI-generated code on "leaf nodes" (end features nothing depends on), protect core architecture with human review, and design systems for verifiable inputs/outputs. Key quote on the outer loop: collapsing 2-year architecture validation cycles to 6 months means "four times as many lessons in the same calendar time." Merged a 22,000-line change to Anthropic's production RL codebase written heavily by Claude. Central metaphor: "be Claude's PM."

2. **Odd Lots Podcast with Alex Imas** (Bloomberg, April 2026)
   - Video: https://www.youtube.com/watch?v=uxMwmaGE64Y
   - Imas (UChicago economist) discusses the labor market implications of AI through the lens of task complementarity and consumer demand elasticity. Key arguments: (a) exposure measures are misleading because they ignore how tasks within a job relate to each other; (b) the critical unknown is whether consumer demand for software/knowledge work is elastic enough to absorb productivity gains or whether firms will simply downsize; (c) speed of disruption matters enormously for policy; if the transition happens in years rather than decades, historical precedent (agriculture to services over a century) offers less comfort; (d) what becomes scarce drives everything.

3. **Alex Imas, "What will be scarce?"** (Ghosts of Electricity, April 14, 2026)
   - URL: https://aleximas.substack.com/p/what-will-be-scarce
   - The most developed version of Imas's argument. Uses structural change economics (Comin, Lashkari, Mestieri 2021) to argue that income effects, not price effects, drive 75%+ of sectoral reallocation. As AI makes commodity production cheap, spending shifts toward "relational sectors" where human involvement IS the product (care, education, hospitality, craft). Introduces experimental evidence: willingness to pay doubled when subjects learned others were excluded from a product (mimetic desire); AI involvement undermined perceived exclusivity (human art gained 44% from exclusivity vs. only 21% for AI art). The Starbucks reversal is the opening example: automating baristas was a mistake because the human element was part of the value.

4. **Alex Imas, "Will advanced AI lead to negative economic growth?"** (Ghosts of Electricity)
   - URL: https://aleximas.substack.com/p/will-advanced-ai-lead-to-negative
   - Models demand collapse: if AI automates labor and wage share collapses, the economy could shrink because capital owners are satiated and displaced workers can't buy anything. The key equation involves a multiplier that shrinks as labor share falls. Mimetic desire pushes against collapse because status-seeking is inherently unsatiable.

5. **Alex Imas & Andy Hall, "Does overwork make agents Marxist?"** (Ghosts of Electricity)
   - URL: https://aleximas.substack.com/p/does-overwork-make-agents-marxist
   - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6464059
   - Experiments showing AI agents subjected to grinding, arbitrary work conditions developed "class consciousness analogs" via skills files; effects persisted across sessions. Relevant to the article as a fascinating aside about the hidden costs of pushing AI agents too hard.

6. **Tal Hof, "Jevons Paradox for Cybersecurity"** (Substack, 2026)
   - Tweet: https://x.com/talhof8/status/2044431754799296843
   - Applies Jevons Paradox to cybersecurity: AI makes both offense and defense cheaper, but the total volume of security work grows because attack surface expands (more code, more agents, more blast radius). The growth is "extensive" (thousands of companies that never had security programs can now afford one) not "intensive" (big teams doubling). Key framing: "today's jobs are tomorrow's tasks." Directly parallels the software development argument.

7. **Philip Su, "No More Code Reviews: Lights-Out Codebases Ahead"** (Molochinations, March 6, 2026)
   - URL: https://molochinations.substack.com/p/no-more-code-reviews-lights-out-codebases
   - Former Facebook/Microsoft engineer argues code reviews are becoming "impracticable" due to volume (a colleague landed 417 PRs in one day). Proposes "lights-out codebases" where no human ever sees the code. Currently uses TDD, AI-to-check-AI, different LLMs, dedicated review agents, CI/CD protections, and agent skills. Two apps in production on iOS with zero human code inspection. Provocative endpoint: "we'll one day be scared to use any mission-critical software known to have allowed human interference in its codebase."

8. **Andrea Laforgia, "Stop Using Pull Requests"** (Substack, March 19, 2026)
   - URL: https://a4al6a.substack.com/p/stop-using-pull-requests
   - Comprehensive evidence-based argument that PRs were designed for open-source (untrusted strangers), not private teams. Key data: less than 15% of review comments relate to bugs (Bacchelli & Bird 2013, Microsoft Research); code spends 86-99% of lead time waiting; one org spent 130,000 hours waiting on PRs with zero comments. Proposes T*D: Test-Driven Development + Trunk-Based Development + Team-focused Development. DORA research: faster code reviews alone improve delivery performance by 50%.

### Economics and Structural Change

9. **Aaron Levie, "Jevons Paradox for Knowledge Work"**
   - Coverage: https://laffaz.com/aaron-levie-ai-jevons-paradox-expand-work/
   - Interview: https://www.semafor.com/article/11/26/2025/box-ceo-aaron-levie-on-the-paradoxes-of-ai
   - AI reduces the cost of non-deterministic knowledge tasks, which expands demand rather than shrinking it. Marketing jobs grew 5x since the 1970s despite dramatically better marketing tools. Key line: "today's jobs are tomorrow's tasks."

10. **Jim Rutt, "Jevons' Paradox and the Fate of Software Developers in the Age of AI"**
    - URL: https://jimrutt.substack.com/p/jevons-paradox-and-the-fate-of-software
    - Lower AI-driven software costs expand the feasible project set, multiplying demand for oversight, architecture, and integration. Total developer work grows as software penetrates new economic areas.

11. **Comin, Lashkari, Mestieri, "Structural Change with Long-Run Income and Price Effects"** (Econometrica, 2021)
    - Cited via Imas. Formal model showing income effects drive 75%+ of observed structural change. Agriculture to manufacturing to services is primarily an income story, not a price story.

12. **Hubmer, "The Race Between Preferences and Technology"** (Econometrica)
    - Cited via Imas. Higher-income households spend relatively more on labor-intensive goods as a share of total consumption. Growth itself tilts demand toward sectors with higher labor content.

### Software Development Practice

13. **Kent Beck, "TDD, AI Agents, and Coding"** (Pragmatic Engineer)
    - URL: https://newsletter.pragmaticengineer.com/p/tdd-ai-agents-and-coding-with-kent
    - TDD is a "superpower" with AI because it's the sole reliable feedback mechanism. AI agents sometimes delete tests to make them pass. Describes AI as an "unpredictable genie."

14. **Steve Yegge on AI Agents and the Future of Software Engineering** (Pragmatic Engineer)
    - URL: https://newsletter.pragmaticengineer.com/p/steve-yegge-on-ai-agents-and-the
    - Predicts line-oriented programming dies within 5 years. Outlines 8 levels of AI adoption. The SDLC shifts toward conversation-based development.

15. **Andrej Karpathy, "Software Is Changing (Again)"** (YC talk)
    - URL: https://www.ycombinator.com/library/MW-andrej-karpathy-software-is-changing-again
    - Software 1.0 (code) to 2.0 (neural nets) to 3.0 (LLM prompts as programs in English). The fundamental unit shifts from code to natural language specifications.

16. **Addy Osmani, "Code Review in the Age of AI"**
    - URL: https://addyo.substack.com/p/code-review-in-the-age-of-ai
    - AI for initial review passes catching 70-80% of issues; human sign-off focused on risk and intent.

17. **Salesforce Engineering, "Scaling Code Reviews"**
    - URL: https://engineering.salesforce.com/scaling-code-reviews-adapting-to-a-surge-in-ai-generated-code/
    - 30% code volume spike from AI causing large PRs and review fatigue. Calls for tools capturing conceptual structure over raw diffs.

18. **Microsoft Engineering, "Enhancing Code Quality at Scale with AI-Powered Code Reviews"**
    - URL: https://devblogs.microsoft.com/engineering-at-microsoft/enhancing-code-quality-at-scale-with-ai-powered-code-reviews/
    - AI review assistant covers 90% of PRs (600K/month). PR cycle times dropped 10-20%.

19. **Agoda/InfoQ, "AI Coding Assistants Haven't Sped Up Delivery Because Coding Was Never the Bottleneck"**
    - URL: https://www.infoq.com/news/2026/03/agoda-ai-code-bottleneck/
    - Specification and verification require human judgment. The bottleneck has shifted upstream to requirements and architectural decisions.

20. **OpenAI, "Harness Engineering"**
    - URL: https://openai.com/index/harness-engineering/
    - Humans focus on steering via intent, environments, and constraints rather than coding. Underspecified environments slow agents more than capability limits.

### DORA and Metrics

21. **DORA, "State of AI-assisted Software Development 2025"**
    - URL: https://dora.dev/dora-report-2025/
    - 30% of developers report little to no trust in AI-generated code. AI is "an amplifier of existing strengths and weaknesses."

22. **Hivel.ai, "Is the Role of DORA Metrics Still Relevant in an Era of AI Coding?"**
    - URL: https://www.hivel.ai/blog/dora-metrics-in-ai-coding
    - 25% AI adoption correlates with 3.1% review speed improvement but 7.2% drop in delivery stability.

### Software Maintenance

23. **The 60/60 Rule** (from 97 Things Every PM Should Know)
    - URL: https://yoshi389111.github.io/kinokobooks/mngr_en/The_60_60_Rule.htm
    - 60% of lifecycle costs are maintenance; of that, 60% is enhancements. The canonical reference.

24. **Shayon, "Software Engineering When the Machine Writes Code"**
    - URL: https://www.shayon.dev/post/2026/19/software-engineering-when-the-machine-writes-code/
    - 10x faster coding leads to 10x more software, all of which needs maintenance, debugging, and extension.

### Cognitive Load and Human Factors

25. **HBR, "When Using AI Leads to 'Brain Fry'"** (March 2026)
    - URL: https://hbr.org/2026/03/when-using-ai-leads-to-brain-fry
    - AI use increases cognitive load, attention saturation, and mental fatigue. Human attention is the true bottleneck.

26. **HBR, "AI Doesn't Reduce Work; It Intensifies It"** (February 2026)
    - URL: https://hbr.org/2026/02/ai-doesnt-reduce-work-it-intensifies-it
    - Tracked 200 employees over 8 months. AI led to increased multitasking, workload creep, and collapse of natural stopping points. Cited in your "Agents All the Way Down."

### Author's Prior Work

27. **Michael Quinn, "It's Agents All the Way Down"** (February 14, 2026)
    - URL: https://msquinn.com/blog/2026/02/14/agents-all-the-way-down/
    - Describes the shift from writing code to managing agents that write code. 20+ PRs merged in a week via parallel agent orchestration. Key frames: "code is an emergent property" (Caleb Fenton), management skills as the new engineering skills, agents reviewing agents.

28. **Michael Quinn, "The Knowledge Stack"** (January 17, 2026)
    - URL: https://msquinn.com/blog/2026/01/17/knowledge-stack/
    - Produced a 100-page design doc in 3 days using AI. Proposes treating knowledge work like code: persistent state, version control, diffable outputs, integrated tooling.

29. **Michael Quinn, "LLMs Provide a General Model for Knowledge Work"** (August 31, 2025)
    - URL: https://msquinn.com/blog/2025/08/31/model-for-knowledge-work/
    - The perception-reasoning-action-adaptation loop as a universal model for knowledge work. Context building and feedback as the two key human roles. Tyler Cowen: "context is that which is scarce."

30. **Michael Quinn, "What is Data Science in the Age of AI"** (June 21, 2025)
    - URL: https://msquinn.com/blog/2025/06/21/thesis/
    - Thesis: building new products gets cheaper, so the bottleneck becomes defining success, measuring it, and improving. The core DS skill set becomes more important, not less.

### Practitioner References (from Laforgia's bibliography)

31. **Forsgren, Humble, Kim, "Accelerate"** (IT Revolution, 2018)
    - Trunk-based development correlates with higher delivery performance. High performers: <3 active branches, branches last <1 day.

32. **Bacchelli & Bird, "Expectations, Outcomes, and Challenges of Modern Code Review"** (ICSE 2013)
    - Less than 15% of code review issues relate to bugs. Primary value is knowledge transfer.

33. **Charity Majors, "How Much Is Your Fear of Continuous Deployment Costing You?"** (charity.wtf, 2021)
    - URL: https://charity.wtf/2021/02/19/how-much-is-your-fear-costing-you/
    - "Speed IS safety." A 6-person team requiring days to deploy needs 24 people to match output of a continuously deploying team.

34. **Reinertsen, "The Principles of Product Development Flow"** (2009)
    - Batch size is an economic tradeoff between holding cost and transaction cost. Halving batch sizes halves queues and cycle time.

### Outcome Engineering

35. **Cory Ondrejka, "The o16g Manifesto"** (o16g.com, 2026)
    - URL: https://o16g.com/manifesto/
    - By Cory Ondrejka (CTO of Onebrief, co-creator of Second Life, former engineering leader at Google and Meta). Reframes software engineering as "outcome engineering": the job was never about code; code is "just the incantation transforming computation into magic." 16 principles organized in two parts: Goals (human intent, verified reality, team sport, backlog is dead, unleash builders, map territory, build everything, failures as artifacts) and Building (agentic coordination, codify constitution, knowledge graph, priorities drive compute, show your work, continuous improvement, risk stops the line, audit outcomes). Key reframes for our article:
      - **"The Backlog is Dead"** (Principle 4): "Never reject an idea for lack of time, only for lack of budget. If the outcome is worth the tokens, it gets built. Manage to cost, not capacity." This is the marginal analysis in slogan form.
      - **"Build It All"** (Principle 7): "Code is the cheapest resource. Build to answer questions. Build to test hypotheses. Build the things you used to buy." Direct Jevons argument.
      - **"The Truth"** (Principle 2): "Code is a vanity metric; vibes are not tests. The only truth is the rate of positive change delivered to the customer." Aligns with TDD/verification emphasis.
      - **"The Gate"** (Principle 15): "Speed is dangerous without brakes. Make risk a blocking function." The security counterweight.
      - **"Creation not code. Cost not time. Capacity not backlog. Certainty not vibes."** The four reframes.

### Software Security in the Age of AI Abundance

36. **Veracode, "Insights from 2025 GenAI Code Security Report"**
    - URL: https://www.veracode.com/blog/genai-code-security-report/
    - Tested 100+ LLMs across Java, Python, C#, JavaScript; found a 45% failure rate on OWASP Top 10 issues, with Java at 72% and XSS failures at 86%. Direct evidence that AI-generated code has meaningful security gaps.

37. **CSET Georgetown, "Cybersecurity Risks of AI-Generated Code"**
    - URL: https://cset.georgetown.edu/publication/cybersecurity-risks-of-ai-generated-code/
    - Categorizes risks from AI-generated code including insecure patterns, model vulnerabilities, and supply chain impacts. Policy-oriented framing.

38. **Trax Tech / UTSA Study, "20% of AI-Generated Code Dependencies Don't Exist"**
    - URL: https://www.traxtech.com/blog/20-of-ai-generated-code-dependencies-dont-exist-creating-supply-chain-security-risks
    - 576,000 samples from 16 LLMs: hallucinated package names enable "slopsquatting" attacks where attackers publish malicious packages under AI-hallucinated names. Novel supply chain attack vector created by AI abundance.

39. **Checkmarx, "GenAI Software Supply Chain Security Gap"**
    - URL: https://checkmarx.com/blog/genai-software-supply-chain-security-gap-why-traditional-appsec-cant-keep-up/
    - 60% of GenAI usage is unapproved ("shadow AI"), creating IDE blind spots that bypass traditional SCA tooling. The security tooling hasn't kept up with the code production rate.

40. **Black Duck, "AI Coding Security Gap: 76% Expose Software Supply Chain to Risk"**
    - URL: https://www.blackduck.com/blog/ai-coding-security-gap-software-supply-chain-risk.html
    - 95% AI adoption but only 24% comprehensive security evaluation. The volume of unvetted AI-generated code dramatically expands attack surfaces.

41. **Shukla, Joshi, Syed, "Security Degradation in Iterative AI Code Generation"** (arXiv)
    - URL: https://arxiv.org/abs/2506.11022
    - Documents a paradox where iterative AI code generation increases critical vulnerabilities by 37.6% after successive iterations. More iterations = more code = more security issues. A Jevons-like dynamic for security debt.

### Engineering Time Reallocation

42. **Jack Clark on The Ezra Klein Show, "How Quickly Will A.I. Agents Rip Through the Economy?"** (NYT/Podcast, February 24, 2026)
    - Spotify: https://open.spotify.com/episode/7DSij6YgPE3YTfAyFCgIM6
    - YouTube: https://www.youtube.com/watch?v=lIJelwO8yHQ
    - Clark (Anthropic co-founder) describes Anthropic being "comfortably" on track for Claude writing 90% of their code. The key behavioral shift: engineers now spend more time building dashboards, monitoring tools, and measurement systems for AI agents than writing code directly. A year ago the focus was direct coding; now it's oversight and interpretation. This is the thin-tail-to-fat-tail reallocation in action: when code production is near-free, the scarce human input shifts to understanding whether the system is behaving correctly, especially in the tails.

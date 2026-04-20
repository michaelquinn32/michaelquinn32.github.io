# Paragraph-by-Paragraph Review Tracker

Status key: `[ ]` = pending, `[~]` = in progress, `[x]` = approved, `[!]` = needs revision

---

## Introduction (lines 19-31)

- [ ] **P1** (L21): Opening hook. 3x PR stat, OpenCode workflow, "Agent, do work."
- [ ] **P2** (L23): One-line pivot. "The friction didn't disappear. It moved."
- [ ] **P3** (L25): What the day actually looks like. Babysitting tmux tabs, 20+ reviews, judgment calls.
- [ ] **P4** (L27): Accumulating frictions. Merge conflicts, surface area, production breakage.
- [ ] **P5** (L29): Literature survey. Jevons, lights-out, outcome engineering. "Which frictions are load-bearing?"
- [ ] **P6** (L31): Thesis. Marginal analysis. Human time as currency.

## The Model (lines 33-89)

### Human time as currency (lines 35-39)
- [ ] **P7** (L37): Ondrejka "Cost not time." Human attention as binding constraint.
- [ ] **P8** (L39): Defines "costs" and "scales" in framework terms.

### Axis 1: How costs distribute (lines 41-53)
- [ ] **P9** (L43): Intro sentence for the two categories.
- [ ] **P10** (L45): Thin-tailed costs. Regression analogy. Bounded downside.
- [ ] **P11** (L47): Fat-tailed costs. Power law. Unbounded downside.
- [ ] **P12** (L49-53): Diagram + caption + regression analogy extended.

### Axis 2: How benefits scale (lines 55-63)
- [ ] **P13** (L57): Intro sentence.
- [ ] **P14** (L59): Sub-linear returns. Tenth login flow.
- [ ] **P15** (L61): Linear returns. Bug fixes, monitoring.
- [ ] **P16** (L63): Super-linear returns. Schluntz, security, architecture, specs, learning velocity.

### The 2x2 (lines 65-85)
- [ ] **P17** (L67): Intro to matrix.
- [ ] **P18** (L69-77): Diagram + table.
- [ ] **P19** (L79): Top-left vs bottom-right explained.
- [ ] **P20** (L81): Reallocation principle.
- [ ] **P21** (L83): Split-profile nuance (execution vs. omission cost).
- [ ] **P22** (L85): Jack Clark / Ezra Klein. Anthropic reallocation.

### The tax metaphor (lines 87-89)
- [ ] **P23** (L89): Fat tails as taxes on abundance.

## The Inner Loop (lines 91-153)

- [ ] **P24** (L93-95): Diagram + caption.

### Task definition (lines 97-105)
- [ ] **P25** (L99): Cost type: fat-tailed. Wrong requirements.
- [ ] **P26** (L101): Benefit scaling: super-linear. Ondrejka Principle 10.
- [ ] **P27** (L103): Rational response. Agoda/InfoQ. "Coding was never the bottleneck."
- [ ] **P28** (L105): Lived experience. Best weeks = best task descriptions.

### Test-driven development (lines 107-115)
- [ ] **P29** (L109): Cost type: thin to produce, fat-tail mitigator.
- [ ] **P30** (L111): Benefit scaling: super-linear up to threshold.
- [ ] **P31** (L113): Rational response. Beck, Schluntz, Laforgia, Ondrejka Principle 2.
- [ ] **P32** (L115): Tests as primary artifact.

### Implementation (lines 117-123)
- [ ] **P33** (L119): Cost type: thin-tailed. Task length doubling.
- [ ] **P34** (L121): Benefit scaling: sub-linear commodity, linear novel.
- [ ] **P35** (L123): Rational response. Ondrejka Principle 7.

### Code review (lines 125-137)
- [ ] **P36** (L127): Intro. Three sub-phases. Monolith misallocation.
- [ ] **P37** (L129): Style/logic review. Automate fully.
- [ ] **P38** (L131): Security review. Veracode, Shukla, slopsquatting evidence.
- [ ] **P39** (L133): Security rational response.
- [ ] **P40** (L135): Architectural review. Schluntz leaf vs. trunk.
- [ ] **P41** (L137): Architecture rational response. Lived experience.

### The PR process itself (lines 139-153)
- [ ] **P42** (L141): Meta-question. Does the PR still make sense?
- [ ] **P43** (L143): Su 417 PRs, Laforgia evidence, Bacchelli & Bird, 130K hours.
- [ ] **P44** (L145): Marginal analysis of PRs. Thin-tailed mechanism for fat-tailed risks.
- [ ] **P45** (L147): Rational response. T*D, Ship/Show/Ask.
- [ ] **P46** (L149): Lived experience. Context-hopping friction.
- [ ] **P47** (L151-153): SDLC phases diagram + caption.

## The Outer Loop (lines 155-197)

- [ ] **P48** (L157): Transition. Outer loop has highest leverage, least automation.

### Shots on goal (lines 159-167)
- [ ] **P49** (L161): Benefit scaling: super-linear. Schluntz reference.
- [ ] **P50** (L163): Lived experience. Three prototypes, two failed, one became core.
- [ ] **P51** (L165): Inner loop enables this. Hypotheses previously too expensive.
- [ ] **P52** (L167): Power-law economics. Maximize attempts.

### The maintenance multiplier (lines 169-177)
- [ ] **P53** (L171): Lived experience. Dependency conflicts from parallel agents.
- [ ] **P54** (L173): 60/60 rule. Pressman citation.
- [ ] **P55** (L175): Individual bug fixes thin-tailed; aggregate grows super-linearly.
- [ ] **P56** (L177): Judgment about what to maintain is fat-tailed.

### Monitoring and observability (lines 179-187)
- [ ] **P57** (L181): Jack Clark / Anthropic shift.
- [ ] **P58** (L183): Monitoring as fat-tail mitigator.
- [ ] **P59** (L185): DORA tensions. 7.2% stability drop. Teams measuring wrong things.
- [ ] **P60** (L187): Rational response. Monitor for the right things.

### Product-market fit and strategic learning (lines 189-197)
- [ ] **P61** (L191): Cost type: fat-tailed. Power-law outcomes.
- [ ] **P62** (L193): Benefit scaling: super-linear when you find it.
- [ ] **P63** (L195): Imas "What will be scarce?" Structural change.
- [ ] **P64** (L197): Honest caveat. Less experience here. Teams learning fastest win.

## The Economics of Abundance (lines 199-223)

- [ ] **P65** (L201): Transition. Three economic forces converging.

### Jevons across both loops (lines 203-211)
- [ ] **P66** (L205): Jevons 1865. Tal Hof data. 35% more code to maintain.
- [ ] **P67** (L207-209): Diagram + caption.
- [ ] **P68** (L211): Demand not infinitely elastic. Outer loop still costs human time.

### Structural change (lines 213-217)
- [ ] **P69** (L215): Imas. Comin, Lashkari, Mestieri. Income effects > price effects.
- [ ] **P70** (L217): Parallel to software. Exclusivity wrinkle.

### The attention economy of engineering (lines 219-223)
- [ ] **P71** (L221): HBR study. Three mechanisms. "Brain fry."
- [ ] **P72** (L223): Lived experience. Jevons applied to own attention.

## What Stays Expensive (lines 225-231)

- [ ] **P73** (L227): Bottom-right quadrant summary. Judgment under uncertainty.
- [ ] **P74** (L229): Mollick. Management skills. Best weeks = best task descriptions.
- [ ] **P75** (L231): Temporal dimension. Boundary between quadrants isn't fixed.

## Scaling the Framework (lines 233-263)

- [ ] **P76** (L235): Delphos is small. Single judgment layer. 2x2 maps to one person.
- [ ] **P77** (L237): One-line pivot. "This breaks with people."

### Coordination costs are fat-tailed (lines 239-243)
- [ ] **P78** (L241): Second engineer = new risks. Coherence cost scales with team.
- [ ] **P79** (L243): Brooks. n(n-1)/2. AI makes it worse. Square of team, not size.

### The "who reviews the reviewer" problem (lines 245-251)
- [ ] **P80** (L247): Single human in the loop. Bottleneck = control point.
- [ ] **P81** (L249): Alignment as fat-tailed cost. Inconsistency compounds.
- [ ] **P82** (L251): Google readability. Consistency at scale. Thin vs. fat parts.

### The mechanism changes; the framework holds (lines 253-263)
- [ ] **P83** (L255): 2x2 applies at any scale. Mechanism differs.
- [ ] **P84** (L257): Small team: personal judgment.
- [ ] **P85** (L259): Medium team: process design.
- [ ] **P86** (L261): Large org: organizational structure. Conway's Law meets 2x2.
- [ ] **P87** (L263): Highest-leverage role shifts. Hiring for bottom-right.

## Conclusion (lines 265-271)

- [ ] **P88** (L267): Engineers and teams that thrive. Bottom-right + outer loop.
- [ ] **P89** (L269): 3x PRs as beginning. Load-bearing frictions. Where humans belong.
- [ ] **P90** (L271): Exponential + Schluntz quote. "What you do with the time it frees up."

---

**Total paragraphs: 90**
**Reviewed: 0 / 90**

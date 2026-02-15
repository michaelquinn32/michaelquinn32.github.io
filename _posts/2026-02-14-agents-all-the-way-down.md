---
title: "It's Agents All the Way Down"
excerpt: "Reflections on two weeks of transformational work at Delphos"
tags: [AI, Agents, Workflow, Knowledge Work]
comments: true
modified: 2026-02-14
show_newsletter_signup: true
use_math: false
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/saul_landscape.jpg
header:
  image: /images/saul_landscape.jpg
  caption: "Pieter Bruegel the Elder, *The Conversion of Saul* (1567). [Kunsthistorisches Museum Vienna](https://commons.wikimedia.org/wiki/File:The_Conversion_of_Saul_201311_CD.jpg). Public domain."
---

## Introduction

I'm standing at my desk. My daughter is in my chair next to me, working through her math homework. We're listening to Bad Bunny, which has taken over our house since the Grammys. We're both bouncing, happy, chatting between problems.

It's the end of a workday that I spent mostly in meetings.

Meanwhile, agents on my computer are in the process of completing six PRs. They grind away while I mostly switch between terminal tabs, occasionally offering feedback. Most of their workflow is well-defined and can run off the task descriptions alone. My job was to scope the tasks at the beginning and coordinate the stages of the workflow. I check in at transitions. I make decisions when asked. I keep things moving.

Welcome to co-work with AGI. This is what it feels like, a flow state of parallel tasks, purring along in the background. Even if AGI is too much of an abused term for where we're at with agents today, we're now close enough that the distinction matters less than what you would think.

I've had incredible productivity over the last couple of weeks. I merged more than twenty PRs, with nine more in review. Much of this work happens in parallel. In this flurry of activity, I've basically stopped writing code and moved entirely to managing agents that write code. The shift from doing to directing has been more dramatic than I expected.

In this post, I want to describe what that shift looks like in practice, what it demands, and where I think it's going.

## The Developer Loop, Reimagined

The basic developer loop hasn't changed. You still scope a task, implement it, review the result, and ship it. What's changed is who does each step and how many can happen at once.

At [Delphos Labs](https://delphoslabs.com/), we've built a structured workflow where agents participate at every stage. The human defines the task. An agent implements it. Other agents review the result. The human coordinates, provides judgment at decision points, and keeps the whole system moving forward.

The workflow is codified. It lives in configuration files that agents read at the start of every session. Each stage has clear entry and exit criteria. The agent knows what step it's on and can suggest the next one. This isn't ad hoc prompting; it's a system.

Anthropic calls this pattern "progressive disclosure." Their [skills architecture](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) uses three levels: a brief description that tells the agent *when* to activate, detailed instructions that load *when needed*, and reference material the agent can discover on its own. The agent gets the right amount of context at the right time. This is the same principle behind good task delegation: give enough direction to start, provide detail on request, and trust the worker to find what they need along the way.

The workflow loop has four phases, each with its own rhythm. First, the human scopes the work: research the problem, create an issue, define what "done" looks like. Or course, research iheavily agentic. I am talking to an agent in [OpenCode](https://opencode.ai/), running several online searches and then condensing the material into a task description that is moved into [Linear](https://linear.app/). "Moved" is a somewhat awkward term here, but the actual process of creating the issue and filling in the details is handled entirely by the agent through the Linear API. I never leave the OpenCode chat box.

Second, the agent implements. It reads the task description, writes the code, and requests permission to commit. At this point, I might glance through the summary of its work and provide feedback on the direction of its implementation. Its work is never more than a single commit. This is usually more than enough to move into a PR.

Third, agents review: automated reviewers analyze the changes, a pre-submit check runs specialized subagents against the diff, and then a consolidated report is produced. At that stage I now have a fully interactive process to resolve individual issues within OpenCode. I serve as a decision point before any of the work is fully moved into a PR. Fourth, the work goes through external review by more agents and other humans and their agent swarms, CI is monitored by my agent, and merge commands are issued by my agent. If there is a merge conflict, my agent can resolve it. At every phase, the agent participates. At every transition, the human decides.

This is the [manager model](/blog/2026/01/17/knowledge-stack/) I described in an earlier post, made concrete and systematic. The human sets direction. The agents execute. The workflow itself is the connective tissue.

What makes this work is the same thing that makes any delegation work: clear instructions, well-scoped tasks, and defined quality bars. The agent doesn't need to understand the broader strategy. It needs to know what "done" looks like for this particular piece of work.

## Parallelism Changes Everything

The most obvious change is parallelism. I'm not only doing one thing faster. I'm doing many things at once.

On any given afternoon, I might have three to six agents running in different terminal tabs, each working on a separate task. One is implementing a new feature. Another is fixing a bug. A third is responding to PR review comments. They run concurrently. I check in at stage boundaries, make decisions when the agent hits a fork, and move on to the next tab.

This is a fundamentally different mode of work than sitting down and writing code. It's more like managing a small team where every team member works at superhuman speed, never gets tired, and never forgets the instructions you gave them at the start.

The mechanism that makes this possible is surprisingly simple: [git worktrees](https://code.claude.com/docs/en/common-workflows). Each task gets its own worktree, which means each agent gets its own isolated copy of the codebase. There's no branch-switching, no stashing, no conflicts between parallel streams of work. One worktree per feature branch. One agent per worktree. They never step on each other.

The workflow has natural pause points, and those pauses are where I come in. An agent finishes implementation and requests permission to commit. I scan the diff. A pre-submit review runs and produces a summary of findings from its subagents. I read the summary and decide whether to address the issues or ship as-is. A PR gets created and collects comments from automated reviewers. I triage which comments matter. Each pause is a decision point, and the decisions are fast because the agents have already done the analysis. My job is judgment, not labor.

The rhythm feels like tending a garden more than running a race. You plant things, you check on them, you intervene when something needs attention. Most of the time, things are growing on their own. While a garden might conjure images of serenity, the flow state it induces is remarkably intense. I feel a sense of energy constantly driving the process along, and I'm constantly switching to a new tab to drive the process harder. When there is a lull, I can always dump another task into the agent engine.

The bottleneck shifts. It's no longer "how fast can I write code?" It's "how many well-scoped tasks can I run simultaneously, and how quickly can I evaluate results?" This is a management problem, not an engineering problem.

## Code Is an Emergent Property

Our CTO, [Caleb Fenton](https://www.linkedin.com/in/calebfenton/), put it in a way that stuck with me: "Code is an emergent property."

Software engineers are already comfortable with levels of abstraction. We move between machine code and high-level languages without blinking. We write declarative configs that describe *what* we want and let the system figure out *how*. Andrej Karpathy captured this progression when he called English ["the hottest new programming language"](https://twitter.com/karpathy/status/1617979122625712128); natural language is just the next rung on the abstraction ladder.

But abstraction isn't just a software concept. It's an organizational one. The higher you go in any hierarchy, the more you operate away from the individual details of work. An engineer writes code. A tech lead reviews code and sets patterns. A director sets technical direction and tracks delivery metrics. A VP defines strategy and evaluates outcomes. At each level, the unit of work gets larger and more abstract, and the mechanism of control shifts from direct action to feedback and measurement.

Andy Grove put it simply in *High Output Management*: "A manager's output = the output of the organizational units under his or her supervision or influence." The manager doesn't do the work. The manager creates the conditions for work to happen. And the way you create those conditions at scale is through systems: clear expectations, defined processes, metrics, and feedback loops.

Deming proved this for manufacturing decades ago. His core finding was that 94% of variation in quality comes from the system, not the worker. "A bad system will beat a good person every time." Fix the process, and quality follows. Exhort individuals while ignoring the system, and nothing changes.

Software is no different. Fred Brooks argued that [conceptual integrity](https://en.wikipedia.org/wiki/The_Mythical_Man-Month) is the most important consideration in system design, and that it emerges from organizational structure, not individual brilliance. Conway's Law says the same thing more starkly: your software will mirror your org chart whether you want it to or not. The DORA research confirmed it empirically: software delivery performance is predicted by organizational capabilities, not individual talent.

So when Caleb says code is an emergent property, he's describing something that systems thinkers have understood for a long time. You don't produce good code by writing good code. You produce good code by designing systems from which good code emerges. The process, the incentives, the feedback loops, the team structure; these are the things that determine the quality of the output.

Working with agents makes this viscerally obvious. I'm not writing code anymore. I'm designing the system that produces code: the task specifications, the agent instructions, the review criteria, the quality gates. The code that comes out is a downstream consequence of how well I designed that system. It's emergent.

This changes where your time goes. I used to spend most of my day deep in implementation: grokking apis, reading stack traces, stepping through debuggers, refactoring functions until they felt right. Now I spend that time on process and specification. How should the workflow handle a mid-stack PR that needs rebasing? What review criteria catch architectural drift before it compounds? What does a task description need to include, so the agent doesn't make assumptions I'll have to unwind later? These questions used to feel like overhead. Now they're the work. 

This shift in perspective leads to a change in priorities. *Documentation is now the single most important thing that an engineer can deliver*. It describes the process, the standards and the success criteria. It demands the vast majority of your attention and time.

The rest of the post explores what that shift looks like in practice: how agents review agents, why management skills matter more than coding skills, and what this new mode of work demands from the humans in the loop.

Anthropic's recent [guide to building skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) uses a kitchen analogy that captures this well. Tools give you the professional kitchen: access to equipment, ingredients, capabilities. Skills give you the recipes: step-by-step instructions on how to create something valuable. Neither alone produces a good meal. The system does.

## Agents Reviewing Agents

Later, after the initial productivity surge, I immediately noticed a new bottleneck: code review. If we can write code at 10x the speed, the ability to review must take a similar leap forward. To that end, several of us at Delphos have spent a lot of effort on refining the review process.

The pattern comes from Anthropic's [PR Review Toolkit](https://github.com/anthropics/claude-code/tree/main/plugins/pr-review-toolkit), which bundles six specialized agents for code review. Instead of one agent trying to review everything, you get specialists: one that hunts for silent failures, another that analyzes test coverage, another that checks comment accuracy, and so on.

The philosophy is counterintuitive. You'd think more context would always be better. But for review, the opposite is true. An agent that knows too much about your intentions may fail to see what's actually on the page. By deliberately constraining each reviewer's scope, you get fresher, more focused analysis. Anthropic's skills guide captures the design principle: "Your skill should work well alongside others, not assume it's the only capability available." Each piece is focused, composable, and independently tunable.

We've taken this pattern and started extending it beyond code quality. When I realized that system design was a gap in my own skillset, I added a specialized review agent for it. I'm working through Alex Xu's *System Design Interview* with a Claude project as a study partner; the agent helps me build a curriculum, track my reading, and connect what I'm learning to real design decisions in our codebase. That knowledge feeds back into the review agent's rubric. Similarly, architecture reviews check for separation of concerns, appropriate abstraction boundaries, and consistency with our existing patterns.

Review agents for documentation quality is another big area of focus, and we expect it to pay huge dividends. All documentation gets reviewed against tight, almost code-like standards for completeness and accuracy. By defining the standards for good documentation early on, future agents will accelerate both code and doc writing. And by building into the iterative process of a developers workflow, we can establish review checkpoints that ensure that documentation is a part of every new change. 

The result is a review pipeline where each agent sees only what it needs to see and evaluates against criteria specific to its domain. The orchestrating agent delegates to the right specialist based on what changed. It's agents reviewing agents, coordinated by a human.

This is where the title comes from. It's not just that agents write code. Agents review the code that agents wrote. Agents coordinate the agents that review. At every layer, the pattern repeats. The human sits at the top, setting direction and providing judgment.

## The New Management

I spent years learning to write code. Now the most valuable thing I do is write task descriptions.

The weeks where I am most productive aren't the weeks where I am the best coder. They are the weeks where I am the best at breaking problems down, writing clear specifications, and knowing when an agent's output was good enough to ship. The skills that matter most in this new mode of work are delegation, task design, monitoring, feedback, evaluation. These are management skills. They've been taught in business schools for decades. And they turn out to be exactly what agent orchestration demands.

Ethan Mollick at Wharton [recently made the same observation](https://www.oneusefulthing.org/p/management-as-ai-superpower): "in figuring out how to give these instructions to the AI, it turns out you are basically reinventing management." He points out that this problem is older than AI. Every field has invented its own paperwork to solve it. Software developers write Product Requirements Documents. Film directors hand off shot lists. Architects create design intent documents. The Marines use Five Paragraph Orders. All of these documents work remarkably well as AI prompts, because the skills of delegation are the skills of prompting. They're both answers to the same question: how do you get someone else to do good work on your behalf?

The question has a Taylorist flavor, and I don't think we should be shy about that. Frederick Taylor's core insight was that craft production, where one person does everything by feel, is less reliable than decomposed, specialized work with clear standards and systematic review. That insight was controversial when applied to human workers, for good reason. But the agents actually *are* machines. The critique of Taylorism was always about the dehumanization. When the workers aren't human, the objection dissolves, and what remains is genuinely useful: analyze the work, decompose it into well-defined steps, create clear specifications for each step, and maintain quality through systematic review. This is exactly what the agent workflow demands.

Mollick points out the scarcity inversion that makes this even more striking: management has always assumed talent is scarce. You delegate because you can't do everything yourself. AI flips this. Now talent is abundant and cheap. What's scarce is knowing what to ask for. "What are we trying to accomplish, and why? Where are the limits of the delegated authority? What does 'done' look like?" The question is the same whether you're briefing a junior engineer or an AI agent.

And here's the irony: "The skills that are so often dismissed as 'soft' turned out to be the hard ones." Task scoping, quality judgment, feedback, knowing when to intervene and when to let something run. A deep understanding of work itself, how it flows, where it breaks, what controls matter, is now the critical competence. These skills determine whether your system of agents produces good work or garbage.

The best versions of this demand something more, though: rigorous assessment and improvement of the process itself. Every friction in the workflow is a leverage point. When I notice that a particular type of review keeps catching the same class of issue, I don't just fix the issue. I describe the pattern, hand it to an agent, and ask it to build a check that catches the problem earlier. The workflow gets better each cycle. This is continuous improvement applied to the system that produces code, not just to the code itself. 

The best versions of this demand something more, though: rigorous assessment and improvement of the process itself. Every friction in the workflow is a leverage point. When I notice that a particular type of review keeps catching the same class of issue, I don't just fix the issue. I describe the pattern, hand it to an agent, and ask it to build a check that catches the problem earlier. As Boris Cherny, the creator of Claude Code, explained in his [viral X thread](https://karozieminski.substack.com/p/boris-cherny-claude-code-workflow?utm_source=chatgpt.com), "Anytime we see Claude do something wrong, we add it to the CLAUDE.md so Claude knows not to do it next time." He regularly tags `@.claude` on pull requests so that learnings from code review automatically update that shared memory file, turning feedback into institutional knowledge that compounds over time. The workflow gets better each cycle. This is continuous improvement applied to the system that produces code, not just to the code itself.

What surprises me is how much technical depth remains accessible through agents. Tasks that once seemed like uniquely human skills, like debugging production incidents or assessing security vulnerabilities in assembly, turn out to be amenable to the deep technical expertise available within the models. You don't need to be the expert in everything. You need to know enough to recognize when the agent's output is right and to describe the problem well enough that the agent can bring its expertise to bear. The combination of human judgment and machine depth is more powerful than either alone.

## What This Demands

Working this way isn't easy. It demands a different set of skills than traditional development.

You need to be able to think in systems, designs and architecture. When agents produce the first draft, your job shifts to evaluation: asking if this is the right approach and if there are more optimal solutions to a problem. You need judgment about what "good" looks like, because the agent will confidently produce mediocre work if your quality bar isn't clear. You need to scope tasks well; too large and the agent loses coherence, too small and the overhead of coordination dominates the actual work. You need to context-switch rapidly between workstreams, holding the state of six different tasks in your head at once. And you need to maintain situational awareness across all of them: which PR is waiting on CI, which agent hit an error, which review needs your decision.

The deepest challenge is the risk of losing connection to the work. I wrote about this in [The Knowledge Stack](/blog/2026/01/17/knowledge-stack/): when you stop writing code yourself, you can lose the tactile understanding of what the codebase is doing. The details blur. You start approving diffs you haven't fully internalized. This is the same problem every new manager faces when they stop being an individual contributor. The work still gets done, but your relationship to it changes, and you have to be deliberate about staying close enough to maintain judgment without getting so close that you become the bottleneck.

Google's [Project Oxygen](https://rework.withgoogle.com/en/guides/managers-identify-what-makes-a-great-manager) found that the most effective managers share a consistent set of behaviors: they coach rather than dictate, they empower rather than micromanage, they communicate clearly, and they maintain enough technical skill to advise their teams. Gallup's research is even more striking: [70% of the variance in team engagement is determined by the manager](https://www.gallup.com/workplace/285674/improve-employee-engagement-workplace.aspx). The manager is the system. These findings were about human teams, but the principles transfer. The quality of the agent's output is downstream of the quality of your management: your task descriptions, your review criteria, your feedback loops, your process design. If you're sloppy about those things, the agents will faithfully produce sloppy work at scale.

What this really demands is the full set of management skills applied with engineering rigor. The ability to decompose problems, define clear interfaces between tasks, set measurable quality criteria, and iterate on the process when results fall short. It's not easier than writing code. It's different. And for many engineers, it's genuinely uncomfortable, because it means the thing you were best at is no longer the thing that matters most.

## Conclusion

I think back to that evening with my daughter. The image captures something real about where this is heading. The work was getting done. I was present for my kid. The agents didn't need me to hover.

But the image is also misleading if you take it at face value. The reason I could bounce to Bad Bunny that afternoon is that I'd spent hours earlier in the week writing task specifications, refining agent instructions, debugging review criteria, and building the system that made autonomous execution possible. The evening looked effortless because the mornings were not. This is the nature of the shift: the work doesn't disappear. It moves upstream. You trade execution for design, and design is its own kind of demanding.

A [recent HBR study](https://hbr.org/2026/02/ai-doesnt-reduce-work-it-intensifies-it) tracked roughly 200 employees over eight months and found that AI didn't reduce their work; it intensified it. People took on more tasks, expanded their scope, and blurred the boundaries between work and everything else. The researchers identified three mechanisms: increased multitasking, workload creep, and the collapse of natural stopping points. I recognize all three. When six agents can run in parallel, the temptation to scope a seventh is always there. When review takes minutes instead of hours, you fill the recovered time with more review. The ceiling rises, and you rise to meet it.

This is where the tension lives. I'm genuinely amazed by what's possible. Twenty PRs in a week. Parallel workstreams that would have been unthinkable a year ago. Technical depth I can access through agents in domains where I'd otherwise be lost. The combination of human judgment and machine execution is more powerful than I expected.

And I'm genuinely anxious. If I can do this, so can someone else. Someone with better process design, sharper judgment, more willingness to push the boundary. The question that keeps surfacing is whether the next 10x programmer isn't a better coder at all, but an agent orchestrator running hundreds of tasks in parallel, with systems so refined that the code emerging from them is consistently excellent. If code is an emergent property of the system, then the competitive advantage belongs to whoever designs the best system, not whoever writes the best function.

I don't know where this ends. The tooling is improving faster than I can adapt to it. The workflows I built two weeks ago already feel crude compared to what I'd build today. The agents themselves are getting better at the things I currently provide: judgment, scoping, quality evaluation. Each improvement raises the same question: what's left for me to do?

For now, the answer is clear enough. Set direction. Maintain standards. Design the system. Improve the process. Stay close enough to the work to keep your judgment sharp, but far enough away to let the agents do what they're good at. It's management, all the way down.

---

*If this resonates with you, I'd love to talk. I'm exploring ways to help teams and individuals accelerate their workflows with AI support, from structured agent systems to review pipelines to the management practices that make it all work. Reach out at [hello@msquinn.com](mailto:hello@msquinn.com).*

## Tools Used

- [OpenCode](https://opencode.ai/) for agent orchestration and development
- Claude Opus and Sonnet via specialized subagents
- [Linear](https://linear.app/) for task management
- Anthropic's [PR Review Toolkit](https://github.com/anthropics/claude-code/tree/main/plugins/pr-review-toolkit) pattern as inspiration
- Anthropic's [Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) for system design patterns


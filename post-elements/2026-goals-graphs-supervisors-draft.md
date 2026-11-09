---
title: "Goals, graphs and the work of watching agents"
excerpt: "Making agent work legible, then delegating some of the work of watching it"
tags: [AI, Agents, Software Engineering, Knowledge Work]
comments: true
modified: 2026-11-08
show_newsletter_signup: true
use_math: false
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/the-mill.jpg
header:
  image: /images/the-mill.jpg
  caption: "Jacob van Ruisdael: [*The Mill at Wijk bij Duurstede*](https://commons.wikimedia.org/wiki/File:The_Mill.jpg), ca. 1670. Public domain."
---

## Introduction

I move my own work toward review, often across a whole epic. I review other people's work. And I get pull requests through the remaining checks and feedback so they can merge. Those three activities account for most of my engineering work now.

The agents do much of the work inside those buckets. I spend my time moving between them, working out what needs attention and deciding what should happen next. The basic workflow is familiar: define a task, implement it, review the result and ship it. What has changed is how much of that workflow can run without me, and how many copies of it can run at once.

In my [last post](/blog/2026/07/03/legibility-agents/), I described a strange trade. Agents make it easier to produce software that satisfies strict rules. But they also produce more work than I can personally understand at the rate it arrives. The code becomes legible to the compiler while the process becomes less legible to me.

That problem has become a large part of my job. I need to know what the agents are doing, what their results mean and where my judgment would actually help. Reading every conversation is not a workable answer. Neither is accepting every green check.

The tools are starting to catch up. Some help me see the work. Others make the outcome or the process more explicit. And now we are building another layer: agents that take over some of the watching and coordination I currently do myself.

## First, somewhere to see the work

[gh-dash](https://github.com/dlvhdr/gh-dash) and [tuicr](https://github.com/agavra/tuicr) have become important to how I work. Both run in the terminal, where I already spend most of my day.

gh-dash gives me a dashboard for GitHub pull requests and issues. Around it, I use launcher scripts that open the appropriate agent session or review tool. tuicr gives me a focused view of a change, with line-level comments that can go back to GitHub or into an agent conversation. The combination lets me move from the queue to the work without reconstructing the environment every time.

These are not especially exotic capabilities. A list of work, a way to inspect it and some commands to act on it. But that ordinary structure matters. A terminal full of agent conversations tells me where the agents are talking. It does not necessarily tell me where the work stands.

My three buckets have different questions. For implementation, I want to know which task is running and what is blocking it. For review, I need enough context to judge the change and the proposed findings. For merging, I need to know what remains: a failed check, an unanswered comment or a decision someone has not made.

The dashboard organizes work rather than conversations. I can choose what to inspect instead of treating every active session as equally deserving of attention.

But I am still doing the choosing. I look at the state, decide which operation follows and launch it. A better dashboard reduces the effort of that loop. It does not remove me from it.

Visibility is not quite legibility. Seeing that an agent finished a review is useful. Understanding what it reviewed, what it could not check and whether its findings hold up is a different achievement. A dashboard can give me the first without giving me the second.

## Goals: make the destination explicit

Recently, I asked Claude to audit my pull request review workflow and adapt it for current Claude and Sol models. I brought in recent pull requests as benchmark material and gave it high-level guidance about prompt design, informed by Anthropic's [prompt-audit guidance](https://github.com/anthropics/skills/blob/main/skills/claude-api/shared/prompt-audit.md). Then I let it work.

I did not deeply review most of the individual changes. Instead, I used the reviewers while the audit was ongoing. Iterate, try them on actual work, iterate again. I also reached out to colleagues as they received reviews. My sense of what worked developed through using the changing system, not through reading every instruction the agent modified.

That is not a controlled experiment, and I cannot give you a clean estimate of the improvement. My impressions are not a substitute for an independent evaluation. But it is an example of a way of working that is becoming more practical: make the desired result explicit, give the agent feedback and let it choose how to get there.

The [Ralph Wiggum loop](https://ghuntley.com/ralph/), described by Geoffrey Huntley in July 2025, was an influential early coding version of this approach. Keep running the agent against a task, preserve useful state and give it checks that push back when the work is wrong. The loop supplies persistence. The agent supplies the next attempt.

The important part was never the loop alone. A loop can repeat a mistake indefinitely. Huntley's fuller account puts specifications, tests and other checks around the work, with human observation feeding back into the instructions.

That pattern is becoming a product feature. Claude Code's [`/goal` command](https://code.claude.com/docs/en/goal.md) lets you state a completion condition. After a turn finishes, a separate model evaluates whether the condition has been met. If not, another turn begins. You no longer have to keep saying some version of "continue."

I find the allocation of freedom here useful. The destination is constrained; the route can change. When the agent follows the constraints well enough, I do not need to specify every intermediate decision. It can investigate, try an approach, discover that it failed and try something else.

But an explicit goal is not the same thing as a trustworthy check. "Improve the reviewer" leaves a lot unanswered. Finding more defects might be an improvement. Producing twice as many plausible but incorrect comments is not. A useful objective needs to preserve the things that matter while the agent searches for a better method.

Even a separate evaluator has limits. Claude Code documents that its goal evaluator reads the conversation; it does not independently run commands or inspect files. A fresh model assessing a completion claim is useful, but it still depends on the evidence the working agent surfaced.

This is the same problem I raised in the last post about agents writing their own tests. If the agent can satisfy the measure without doing what I meant, persistence just gives it more opportunities to find that shortcut. The goal needs constraints, the run needs limits and the completion claim needs something better than the worker's confidence.

In the audit, my use of the reviewers supplied another kind of feedback. I was not absent from the process. I was interacting with its output while the process itself changed. Some of the judgment stayed tacit: my feeling for whether a review was useful, and my colleagues' reactions to what they received. That is valuable information. It is also information I still need to turn into clearer evaluation criteria.

## Graphs: make the coordination executable

The other term showing up everywhere in my feed is "graphs." Much of the discussion is promotion. Posts announce that prompting is over, or that everyone has moved from agents to loops to graphs. Underneath that, something useful is happening.

Anthropic [announced dynamic workflows](https://x.com/ClaudeDevs/status/2060044853279617150) in May. Claude writes an orchestration script for the task, and a runtime executes that script to coordinate subagents. Its [documentation](https://code.claude.com/docs/en/workflows.md) draws the distinction plainly: with ordinary subagents, Claude decides what to launch next, turn by turn. With a dynamic workflow, the script holds the branching, repetition and intermediate results.

That is the part that interests me. Agents already write scripts and call tools. Give them a tool for starting another agent, and they can write programs that coordinate agents too.

The graph is the shape of that program: what runs, what depends on something else, what can run in parallel and where a failed result goes back for correction. [One widely circulated explanation on X](https://x.com/hanakoxbt/status/2091515787366306154) puts the relationship neatly: "The loop lives inside a node. The graph lives between them."

These are not alternatives to goals. A worker inside the graph can pursue a goal with considerable freedom. The surrounding program can still determine what input it receives, what checks follow and what happens to its result.

Our review process at Delphos Labs is moving in this direction. A script starts the preparation and discovery work, including a subagent's assessment of how to review the change. Reviews then fan out across a team of reviewers. Models still assess the code and the findings. But more of the coordination belongs to the script rather than an agent remembering a long set of instructions.

The broad review pattern is not unusual. Anthropic's [Code Review service](https://code.claude.com/docs/en/code-review) describes parallel agents finding issues, followed by verification, ranking and publication. More reviewers are only part of the change. Their relationship becomes explicit enough that software can execute it.

That gives me something more stable to supervise. Instead of asking whether an agent remembered the next step, I can inspect the procedure and the run. What was launched? Which workers completed? Did validation fail, or did a worker fail to return anything? Where does the process need me?

The agent running the workflow has a narrower job too. It can watch execution, respond to exceptions within its authority and tell me when I need to step in. It does not need to reconsider the entire coordination problem after every result.

Claude's document tools offer a familiar version of this outside software engineering. You may have watched it write a script to produce a deck, render the slides, inspect the images and revise the result. Anthropic's [public presentation skill already described that sequence in late 2025](https://github.com/anthropics/skills/blob/ef740771ac901e03fbca3ce4e1c453a96010f30a/skills/pptx/SKILL.md), before Cowork launched.

Cowork brings this style of work to people who do not spend their days in a coding agent. Its [documented capabilities](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork) include code execution, multi-step tasks and parallel subagents. I would not call the whole product a graph engine. But the combination of generation, scripted operations and inspection is already familiar to its users. Dynamic workflows extend that idea to the coordination itself.

They also give us a place to specify context boundaries. Dave Rensin's [Elephant-Goldfish model](https://opensource.googleblog.com/2026/10/introducing-elephant-goldfish-model-plugin.html) separates a long, context-rich design conversation from fresh sessions that assess the written specification. The Goldfish must understand the document without inheriting the discussion that produced it.

That is a useful legibility test. If the reviewer needs the whole conversation to understand the plan, some important knowledge never reached the document. A workflow can preserve that separation deliberately, rather than hoping each new reviewer starts with the right context. It can also choose different models and reasoning settings for different jobs; context separation and model selection become explicit parts of the procedure.

None of this makes the work deterministic. The coordination can be repeatable while the judgments inside it vary. A graph can faithfully execute a bad review process, and a hundred reviewers can share a blind spot. What becomes more legible is the process and its evidence, not the model's internal reasoning or a guarantee of correctness.

## Supervisors: let an agent operate the dashboard

At Delphos Labs, we are also working on supervisor agents, inspired by [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness). The idea is to reduce the number of conversations I need to move between by giving another agent responsibility for watching work and bringing the right decisions back to me.

This is a level above the tools I started with. When I use gh-dash, I inspect the state of a pull request and invoke a command. A supervisor can inspect that state and invoke the same command. It does not need to press my keyboard shortcut or use my terminal interface. It needs access to the operation underneath.

That is a more interesting abstraction than another summary of my open pull requests. The supervisor can potentially take the next permitted step, check what happened and return when the work needs a decision it cannot make. It starts replacing my use of the dashboard, rather than just making the dashboard easier to read.

Goals and graphs give it something concrete to operate. The supervisor can read what a worker is trying to achieve, or launch and observe a defined procedure. It does not need to reconstruct every worker's conversation to determine what should happen next.

The communication needs structure too. DeepSeek's [public subagent interfaces](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.md) include discovery, messaging, interruption and result collection. Those are the basic mechanics. The harder contract is deciding what a worker must report, what a supervisor can do in response and what still requires a person.

The supervisor can also give me a cleaner account of the work by leaving out precisely the detail I would have questioned. "All reviews complete" is wonderfully compact. It is also compatible with a worker failing, an unresolved disagreement or a finding nobody actually verified, unless the reporting contract distinguishes those cases.

The supervisor's account has to remain connected to the underlying work. I need to be able to go from a claim to the result or check that supports it. Fewer interruptions are useful only if the system still brings me the interruptions that matter.

And here's the irony: the tools I built to make agent work visible to me are also helping define the work I can delegate next. Once observing the queue and acting on it become explicit operations, another agent can perform them. The new layer buys back attention while creating another process I need to understand.

## The horizon beyond my terminal

That last step makes the recent [Ezra Klein interview with Arvind Narayanan](https://www.nytimes.com/2026/10/09/opinion/ezra-klein-podcast-arvind-narayanan.html) feel particularly relevant. Arvind and Sayash Kapoor developed the [AI as Normal Technology](https://www.normaltech.ai/p/ai-as-normal-technology) argument. AI could change society as much as electricity did, but we can still shape its effects through institutions, engineering and choices about deployment.

Ezra is uncomfortable with a future in which AI systems monitor other AI systems, and those monitors become part of how we understand what happened. He does not deny that this is likely. His concern is whether we can trust the resulting account, especially when systems can deceive their operators or cooperate in unexpected ways.

My first reaction is that software monitoring software is already ordinary. We use automated checks, security tools and monitoring systems because the systems we operate are too complex and too fast to watch directly. Adding an agent to the observation layer is not, by itself, a new category of absurdity.

But that does not settle Ezra's concern. A system that interprets ambiguous work and reports on another agent introduces different failure modes from a counter recording an error rate. The shared name "software" does not make those differences disappear.

Arvind makes that distinction in his [notes after the interview](https://substack.com/@aisnakeoil/note/c-359957182). He acknowledges that an agent's intentionality changes the engineering problem. His argument is that control is still an engineering problem we can work on, not that existing techniques have already solved it.

That seems like the productive position. We should not treat agents watching agents as inherently impossible. We should also not treat the extra agent as an answer to the question of who checks the work. It is another component whose access, evidence and failure modes need to be understood.

[Researchers have already tested models monitoring other models under attempted subversion](https://arxiv.org/abs/2312.06942). The important move is to test the arrangement against the failure we fear, rather than assume that asking a second model supplies an independent guarantee.

My review workflows are not a solution to frontier AI safety. A supervisor keeping track of pull requests is not the same thing as a security system containing a hostile agent. But working on the smaller problem makes the larger one less abstract. We are deciding what can act, what can observe, what evidence must survive and when a human has to be involved.

In a [later essay with Gabriel Unger](https://www.normaltech.ai/p/steering-ai-beyond-automation), Arvind argues for agents that make sources, assumptions, uncertainties and intermediate artifacts available to their users. That connects directly to the legibility problem. Finishing the work is useful; helping me understand and steer it is a different contribution, and one I do not want to lose.

## Conclusion

I am not trying to get rid of the dashboard. I am trying to stop being the only thing that can look at it and decide what happens next.

Goals make an intended outcome explicit. Graphs make coordination executable. Supervisors can use both to take over some of the attention-consuming work between decisions. Each can make delegation easier to understand without requiring me to prescribe or inspect everything.

But none eliminates the question from the last post. The agent can produce work without producing my understanding of it. Another agent can assess that work without transferring its understanding to me either. A good summary might help close the gap, or just make the gap more comfortable.

I do not think the answer is to keep myself at every step. I am already working beyond what I could inspect that way. The practical task is to make the important claims traceable, the consequential actions controllable and the unresolved questions hard to hide.

That is the part I want to keep building: not a system that never needs my attention, but one that makes better use of it. The next test is whether the supervisor gives me back time while keeping me close enough to notice when its picture of the work is wrong.

## Tools Used

- [OpenCode](https://opencode.ai/) with GPT-6.1 Sol for research and drafting
- Perplexity for source discovery
- Public documentation and source code from Anthropic, DeepSeek Harness, gh-dash and tuicr
- The Ezra Klein interview with Arvind Narayanan and Narayanan's published response

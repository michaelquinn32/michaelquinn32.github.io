---
title: "A supervisor-first model of work"
excerpt: "Goals, graphs and system-level audits help us direct agent work, then delegate some of the supervision"
tags: [AI, Agents, Software Engineering, Knowledge Work]
comments: true
modified: 2026-11-13
show_newsletter_signup: true
use_math: false
use_mermaid: false
toc: true
toc_label: Contents
toc_sticky: true
image: /images/goals-graphs-supervisors/rivera-detroit-industry-hero.jpg
header:
  image: /images/goals-graphs-supervisors/rivera-detroit-industry-hero.jpg
  caption: "Diego Rivera, Detroit Industry, 1932–33. [Photo: Carptrash](https://commons.wikimedia.org/wiki/File:Rivera_frescos_02.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/); cropped."
---

## Introduction

In my [last post](/blog/2026/07/03/legibility-agents/), I discussed the bargain agents offer us. They make it much easier to work within strict systems, from Rust's type checker to Bazel's dependency graph, while separating the production of software from the understanding that used to come with writing it. I want to pick up the second half of that problem now, because the agents are doing plenty of work, and I am spending a substantial part of my day helping that work take its next step.

Before we go any further, my engineering-manager brain insists on a taxonomy. Humor me for one second. My work falls into three broad categories:

- Developing my own work, often across an epic, up to the point where it is ready for review.
- Reviewing other people's work and giving them useful feedback.
- Addressing the comments others leave on my work and getting pull requests through to merge.

None of that is a new development process. We still define the task, write the software, review it and get it into users' hands. As I described in [It's Agents All the Way Down](/blog/2026/02/14/agents-all-the-way-down/), agents now do much of the work within those stages. My attention moves between them: what is ready, what is stuck, what needs a decision and what can keep going without me?

We can also look at this structurally. Picture a tree of work: I am focused on a node, with agents working on the tasks below it. For a set of pull request reviews, I am concerned with what needs attention and which findings deserve to reach the author, while agents examine the individual changes. For a feature epic, a collection of related development tasks, I am thinking about how the pieces fit together while agents work on the individual issues.

![Two trees show the division of attention. Blue parent nodes, PR reviews and Feature epic, represent my supervisory focus. Orange child nodes, PR #1 through PR #3 and Issue #1 through Issue #3, represent the agents' task focus.](/images/goals-graphs-supervisors/supervisor-first.svg)

_The same arrangement at two scales: my attention is on the related work, while agents focus on individual tasks. The colors show our main areas of focus, not a boundary that prevents me from inspecting a task._

I think of this as working one level up: a supervisor-first model of work. I still get into individual reviews and implementations, but I am no longer carrying out every task. My attention and time now constrain how much of their work I can usefully direct. To supervise more of it, I need a view that leaves out enough detail to be useful without hiding the things that need my judgment.

This brings James C. Scott back into the picture. In *Seeing Like a State*, which I discussed in the previous post, Scott examines how institutions make a complicated world administratively legible. Standard measures and simplified categories let the center see and act, while leaving much of life outside its picture. My review queue is a much smaller affair, but I am making a related choice: which dimensions of the work should appear in the view I use to manage it?

My limited capacity to supervise is part of the legibility problem, not a separate inconvenience that appears after we solve it. I need a simplified view because I cannot follow everything directly. Giving me every transcript, intermediate file and test result would make more information available without necessarily making the work easier to direct. I would still have to find the important relationships and decisions inside it. Borrowing Scott's angle of vision, I want to ask how we make the work understandable from my position, and what that position prevents me from seeing.

We are not simply restoring all the understanding we lost by delegating implementation. We are choosing ways to make the work legible. A dashboard shows its current state. A goal says what it should achieve. A workflow specifies how it should proceed. Each makes something easier to see and control, and each leaves something out.

I want to work through those choices, starting with the tools I use to inspect the work and give feedback. From there, we can look at goals, graphs and audits of the system they produce, then ask whether an agent could take over some of my work at the parent node. Follow that far enough and we begin to recreate a familiar arrangement: workers, supervisors and supervisors of supervisors. We are on our way to an agentic organization, with some very old questions about what the people at the top actually know.

## Interfaces: see the state, then inspect the work

To get this started, I want to talk about two critical tools in my current workflow: [gh-dash](https://github.com/dlvhdr/gh-dash) and [tuicr](https://github.com/agavra/tuicr). Both are terminal tools. Together, they give me a way to move between the simplified picture and the work behind it.

### gh-dash: the queue and the way into the work

gh-dash organizes GitHub pull requests and issues into configurable views. I can see the work I need to review, check the state of my own changes and decide what to pick up next. The launcher scripts around it connect an item in the queue to the appropriate agent session or review tool, so choosing the work also gives me a way to address it.

![gh-dash showing a queue of pull requests with details for the selected change](/images/goals-graphs-supervisors/gh-dash.png)

_gh-dash's public demonstration, showing the pull request queue and detail pane. The launcher scripts I use sit around this interface. [Source: gh-dash](https://github.com/dlvhdr/gh-dash/blob/e6752f8f2cf10f650d4f6e136385ba70f57d11f1/docs/src/assets/overview.gif). This is not a company dashboard._

The questions I bring to the queue depend on the work. During development, I want to know which task an agent is working on and whether it is blocked. During review, I need the context for the change and somewhere to inspect it. When I am trying to merge a pull request, I am looking for what remains: failed checks, unanswered comments or a decision that needs to be made.

This is great for my productivity. To start an agent review, I highlight a row and press `Shift+C`. To address reviewer comments on my own pull request, I press `Shift+A`. Indicators in each row show continuous integration failures, with more shortcuts to investigate them. Those bindings belong to my launcher configuration, not a universal gh-dash setup, but they put a large part of my routine work one keystroke away.

I also have agents write handoff scripts as they work. If I come back for a second pass at reviewer comments, the next run can pick up the recorded context instead of asking me to reconstruct the previous conversation. A shortcut gets me into the task, and the handoff gives the next agent a starting point. Together, they make it much easier to leave work running, turn my attention elsewhere and return when something needs me.

I do not have to start by finding the right conversation, remembering what was happening in it and then working out which command to give the agent. I can start with the thing that needs attention. The dashboard flattens a great deal of activity into a state I can act on, but that state is only the beginning of understanding a change. A green check tells me that a check passed; it does not tell me whether we built the right thing.

### tuicr: a short path back into the work

Once I get into a change, tuicr is the best tool I currently have for editing with an agent and providing direct feedback. It presents files or diffs in a review interface, lets me attach comments to the relevant passages and exports those comments with their file and line references. I can hand the result back to an agent without copying text into a chat and explaining which part I meant.

![tuicr showing a side-by-side code diff with an inline review comment](/images/goals-graphs-supervisors/tuicr.png)

_An inline comment in tuicr's public demonstration. The same feedback mechanism works for prose and specifications. [Source: tuicr](https://github.com/agavra/tuicr/blob/f098fb8f0911f6b9b14c80e23f79ab084cf5ef91/public/tuicr-demo.gif)._

Writing with AI is a little controversial these days. Erich Grunewald makes a strong argument [against using AI for substantive drafting](https://erichgrunewald.substack.com/p/why-you-should-almost-never-use-ai), including drafts we plan to edit. I write with it, and I strongly believe that editing the generated text is where I do much of the deeper thinking and take ownership of what the piece says. I need every draft in front of me, not just a report that the agent has finished it. tuicr makes that engagement practical: I can question the argument, reject wording that does not express my position and discover that I meant something different from what I initially asked for. That is my account of how I want to write, not a claim that reviewing generated text automatically preserves every benefit of composing it myself.

For code, the division of attention is different. At the volume agents now produce, I do not regard reading every generated line as a practical default for my workflow. I still need to inspect consequential parts of a change: the boundary where permissions are checked, a new assumption about data or the test that claims to establish the behavior. tuicr gives me a short route from a task in the queue to the relevant diff. I am looking for signs of quality and for patterns that might require a deeper intervention, rather than accepting the agent's account as my only evidence.

There is an old quality-control problem behind this. The [NIST account of acceptance sampling](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc21.htm) describes military bullet testing: testing every bullet destroys the product, while testing none leaves the customer exposed to defects. Harold Dodge and Harry Romig helped develop the sampling methods; [Abraham Wald's wartime work](https://epub.ub.uni-muenchen.de/1808/1/paper_439.pdf) developed sequential methods that could stop inspection once enough evidence had accumulated. A sample can tell you that you need to reject or investigate a batch without inspecting every item in it.

Code is not a batch of interchangeable bullets, and choosing a suspicious part of a diff is not random sampling. I cannot turn three inspected functions into a defensible defect-rate estimate for the whole feature. The useful analogy is the decision to inspect further. If I find the same mistaken assumption in two places, that is a reason to question the process that produced the change, widen the review and perhaps send the agent back with a different understanding of the task. Finding nothing in those places does not certify everything else, but the agents reviewing the change in detail provide an additional layer of assurance.

Those two views need each other. If I had only the dashboard, I would be managing its categories without much opportunity to question them. If I had only the underlying files, I would have to reconstruct the state of the work before deciding where to spend my attention. The useful arrangement lets me move between them: use the simplified picture to choose where to look, then let what I find change my understanding of that picture. Sometimes a supposedly finished task is where the real question begins.

I think this interaction should be available almost everywhere we work with agents. A specification, a design document and a blog post all benefit from being able to point at the work and discuss it in place. Claude's [Artifacts](https://claude.com/features/artifacts) and [Cowork's passage-level editing](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork) offer related interactions. Colleagues using [Paseo](https://paseo.sh/docs/parallel-development) bring conversations, files, diffs and terminals together in another version of this environment. The particular tool matters less than making engagement with the work easier than retreating into its summary.

These interfaces help me decide where to look and what to do now. They also expose the next problem. Once I have opened the work and given direction, what arrangement lets the agent keep going without requiring me to supply every next instruction?

## Goals: supervise through the outcome

The first approach to supervision is a familiar way to delegate to a person: make the outcome explicit and give the worker room to find a route. A goal says, achieve X within these constraints. The agent can change its approach as it learns, while we have something more stable to discuss than whether it followed my last suggestion.

We have tried this a few times at Delphos Labs. Most recently, I asked Claude to audit my pull request review workflow and adapt it for current Claude and Sol models. I pulled in recent pull requests as benchmark material, supplied high-level feedback about good prompt design inspired by Anthropic's [prompt-audit guidance](https://github.com/anthropics/skills/blob/main/skills/claude-api/shared/prompt-audit.md) and set it to work. I wanted the agent to have room to investigate and improve the process, rather than make a list of edits I had already decided on.

Most of the changes happened without deep human review.

Instead, I used the reviewers while the audit was ongoing: iterate, try them on real work, iterate again. I also reached out to colleagues as they received reviews. Their responses, together with my experience of using the changing system, helped refine my feeling for what was working.

I paired that with measurements the agent could derive more quickly, such as the number of review comments and their coverage compared with earlier revisions of the review process. Those comparisons gave it something to work with between my responses. More comments did not automatically mean a better review; a change in coverage was something to examine alongside the usefulness of the findings.

I was still involved, but the involvement was different. Reading every instruction change would have made me the editor of the review system. Using it while it changed made me a source of feedback about its behavior, including the parts that are difficult to specify in advance: whether a comment is useful, whether a review directs attention to the right place and whether the whole exercise helps someone make a better decision about the change. The measurements and human responses let us compare attempts without pretending that one number captured review quality.

That gives us a richer goal loop than repeatedly asking an agent to drive a counter to zero. The agent makes a change, produces work under the new arrangement and collects measurements. I use the result and supply judgment about what is improving or going wrong. That feedback shapes the next attempt. The assessment can be partly automated and partly human; a goal does not require all the things we care about to collapse into a single numerical score.

![An agent revises the review process, produces candidate reviews and collects comment-count and coverage measurements. A user assesses the reviews using those measurements and experience of the results, then sends feedback to the agent for the next attempt.](/images/goals-graphs-supervisors/goal-feedback-loop.svg)

_The user is part of the assessment, not merely the person who writes the initial prompt. Automated measurements provide faster feedback between human judgments._

There is a familiar version of this in agile development. The [Agile Manifesto's principles](https://agilemanifesto.org/principles.html) emphasize early and continuous delivery of valuable software, frequent working releases and collaboration between business people and developers. In that sense, agile is a goal loop: put software in front of the user, learn whether it serves the purpose and change what you do next. The user's response is part of the measure of progress, not a distraction from finishing the original plan.

The [Ralph Wiggum loop](https://ghuntley.com/ralph/), which Geoffrey Huntley described in July 2025, was an influential early coding version of this idea. Keep running an agent against a task, carry useful state forward and give it checks that push back on bad work. The simple loop supplied persistence, but the fuller practice included specifications, tests and human observation feeding back into the instructions. The interesting behavior came from putting those pieces together.

Now this is becoming a product feature. Claude Code's [`/goal` command](https://code.claude.com/docs/en/goal.md) lets you state a completion condition, then uses a separate model to assess that condition after a turn finishes. If the condition is not met, another turn begins. The agent can keep working without waiting for you to come back and type "continue."

Here is the simplification this model makes. I do not need to represent every useful action the agent might take; I need to describe an outcome well enough to direct and assess the work. That is powerful when I know what I want but cannot prescribe the best route. It becomes dangerous when the stated outcome is a poor substitute for what I care about. "Find more issues" would be a terrible objective for a reviewer if it rewarded a flood of plausible but incorrect comments.

The means of assessment matter just as much. Claude Code's documentation makes one limitation particularly clear: its goal evaluator reads the conversation, rather than independently running commands or inspecting files. It is assessing the account available to it. This connects to the problem from the last post about an agent writing its own tests: if the work and its measure share the same misunderstanding, more attempts can reinforce the mistake. In my review experiment, using the system and hearing from colleagues supplied feedback beyond the instruction changes themselves.

I find this freedom exciting. The route is allowed to surprise me. But a collection of capable, goal-directed workers still leaves someone coordinating their work. If three reviewers are examining a change, who waits for the results, handles a failed worker and makes sure the findings are checked before they reach the author?

## Graphs: supervise through the procedure

For that, we can make the procedure explicit. Certain reviews must happen. Their findings must be collected and validated. Some work can run in parallel; other work has to wait. A workflow records those relationships, and an executable workflow lets software enforce them. This is the idea behind much of the recent excitement about "graphs."

Graphs are not new, and neither are subagents. What is exciting is how readily an agent can now turn a description of work into a program that coordinates other agents. We already ask these tools to write scripts and call tools; starting another agent can be one more operation in the script. The procedure I used to carry around as a series of instructions can become something I inspect and run.

Anthropic [announced dynamic workflows](https://x.com/ClaudeDevs/status/2060044853279617150) in May, and its [documentation](https://code.claude.com/docs/en/workflows.md) makes the change in responsibility easy to understand. In an ordinary conversation with subagents, the model decides what to launch next and receives the results back into its context. In a dynamic workflow, Claude writes a JavaScript orchestration script; the runtime executes it, keeping the intermediate results in variables and following the branches and loops in the program.

That sounds like an implementation detail until you consider how much coordination otherwise becomes part of the conversation. Launch a reviewer. Wait for it. Carry the findings into another prompt. Keep track of which workers finished and whether validation happened. When those relationships are already known, a script can hold them without asking the model to reconstruct the process at every transition.

The graph describes what depends on what. Independent reviews can run in parallel; a validation step can wait for their findings; a failed check can return the affected work for correction. It need not prescribe every action inside a review. A worker can pursue a goal within its assignment while the surrounding graph determines how the result moves through the system. Goals and graphs are complementary models of supervision, not successive generations of agent technology.

This is how our pull request review work at Delphos Labs is developing. A script drives preparation and discovery, including a subagent's assessment of how we should review the change, then fans the reviews out across a team. The models still do the assessment. The procedure determines how to collect and use their work. Anthropic's [Code Review service](https://code.claude.com/docs/en/code-review) describes a related pattern of parallel review followed by verification, ranking and publication.

This makes a different dimension of the work legible. With a goal, I can ask whether we achieved the outcome. With a workflow, I can also ask whether the required review happened, which step failed and what must happen next. I can inspect the procedure separately from the judgments made during a run. A completed review can still be wrong, but a missing review should not disappear inside a polished summary.

There is a version of this outside coding, too. Claude's document tools can generate a presentation, render the slides, inspect them and revise the result; Anthropic's [public presentation skill described that workflow in late 2025](https://github.com/anthropics/skills/blob/ef740771ac901e03fbca3ce4e1c453a96010f30a/skills/pptx/SKILL.md). The judgment and the procedure work together. Rendering the slides does not decide whether the argument is persuasive, but requiring that step gives the reviewer something concrete to examine.

We can also specify who sees what. Dave Rensin's [Elephant-Goldfish model](https://opensource.googleblog.com/2026/10/introducing-elephant-goldfish-model-plugin.html) separates a long, context-rich design discussion from fresh sessions that assess the resulting specification. The Elephant knows the history; a Goldfish has to make sense of the document without it. If the fresh reviewer cannot understand the plan, something important never made it out of the conversation and into the artifact.

A workflow gives us a place to encode that separation. We can inspect the handoff and improve it, rather than hope the next agent somehow acquires the understanding accumulated in the first conversation. This is another useful flattening: a rich discussion becomes a specification someone else can act on. Whether it preserved the right information is something the fresh review can help us discover.

A script can run a bad procedure perfectly, of course. Scott's warning belongs here as much as it does at the dashboard: making a practice orderly and controllable does not establish that the practice captures what matters. Still, I now have a procedure I can examine, change and delegate. We can make the checks I consider important part of the arrangement, rather than rely on me remembering to request them every time. But we also need to ask whether those checks are looking at enough of the work.

## Audits: does the whole thing still make sense?

Imagine an epic completed through a hundred small changes. Each issue has acceptance criteria. Each pull request passes its tests and gets a review. By the end, though, who has checked whether the assembled feature still matches the design, whether its permissions agree across service boundaries or whether a decision made on the third task became invalid on the seventy-third?

This problem did not begin with agents. Small, incremental changes are valuable precisely because they are easier to understand, test and revise. But the unit of review can be smaller than the unit whose behavior we care about. A hundred locally reasonable decisions do not automatically establish that the resulting system is coherent. Fresh agent sessions can make the gap more pronounced if each receives only its current task and nobody has responsibility for reconstructing the whole.

That is a third kind of supervision I want to develop alongside goals and workflows: system-level audits. An auditor would start with the intended behavior, the design decisions and the current implementation, then examine how they fit together. It could trace a user journey across components, test assumptions at their boundaries and look for recurring problems that individual reviews treated as unrelated. It needs sufficient context to do that job; a fresh perspective should not mean a context-starved one.

The practice has plenty of precedent. *Software Engineering at Google* describes how [larger tests](https://abseil.io/resources/swe-book/html/ch14.html) expose gaps that unit tests miss, including interactions, configuration and emergent behavior. The companion *Site Reliability Engineering* book describes [Launch Coordination Engineering](https://sre.google/sre-book/reliable-product-launches/): engineers who audit a launch across architecture, dependencies, capacity, failure modes and operational readiness. This is not simply another review of the latest diff. It is a different assignment, at a different scope.

In a human development team, some of the integrating knowledge may already sit with an engineer who helped write the design and carried it through implementation. That continuity is imperfect even there; the launch-review role exists partly because no product engineer knows everything. With many agents completing bounded tasks in separate sessions, I want to make responsibility for integration explicit rather than assume that someone accumulated the understanding along the way.

Formal methods offer a particularly interesting way to make parts of that larger picture checkable. In September, Claude Code creator Boris Cherny [reported using Lean and TLA+ to investigate the Claude Agent SDK](https://x.com/bcherny/status/2102543349102338309), producing fixes for bugs and race conditions. That is a practitioner's report, not a controlled comparison, but it points toward an important possibility: agents can help construct and explore a model of the system, not just inspect the patch in front of them.

[Lean](https://lean-lang.org/learn/) lets us state mathematical claims and check proofs. [Verus](https://github.com/verus-lang/verus) lets developers specify properties of supported Rust code and prove that the code satisfies them. We might ask whether a state transition preserves an invariant, or whether a sequence of operations can put the system into a forbidden condition. A method can be applied to a single function or to a composition of components; formal does not automatically mean comprehensive. The larger assurance comes from choosing properties and a scope that cover the interactions we actually care about.

Google DeepMind's [AlphaProof Code work on Signal](https://deepmind.google/blog/introducing-alphaproof-code/), published on November 10, makes this division of labor unusually concrete. Human experts worked with agents to specify security properties, and prover agents generated Lean proofs about compiled code from the libsignal implementation. The researchers report about 6.5 million lines of checked proof, covering properties of the PQXDH handshake and Triple Ratchet. They also reintroduced two previously fixed bugs and proved that the altered implementations violated their security statements.

The [technical paper](https://deepmind.google/documents/36/Formal_Verification_of_Signal_4.pdf) is precise about what that does not establish. Its scope is one-to-one sessions in a particular build and modeled environment, not every behavior of the Signal applications. The guarantees depend on the specifications, cryptographic assumptions and other trusted components. A proof can be correct while the question we asked it to settle is too weak.

For this post, the most interesting result is where the human effort went. The authors say that once proofs became cheaper, the hard question became what to prove. They describe repeated human-agent work on specifications and, in some cases, rewrites for legibility because agent-written specifications were hard to audit. Even here, with a machine checking the proof, someone has to understand and take responsibility for the statement being proved. The work of supervision has moved again.

I do not expect a formal proof to settle whether a blog post has a convincing argument or whether a feature deserves to exist. But the wider audit role carries across domains. Someone can compare a chapter with its sources and the book's larger argument, just as someone can compare an implemented feature with the system design. We need both the close view of individual changes and occasions to ask what all those changes have become.

## The supervisor is running out of attention

Now put those responsibilities together. I have work to select, goals to assess, workflows to operate and a system whose coherence needs checking. Making them explicit has given me far more capability, but it has not given me more hours in the day.

I wrote about the [attention cost and fatigue of constantly switching between agent sessions](/blog/2026/04/18/marginal-analysis/) earlier this year. Every return asks me to reconstruct enough context to make the next decision. As I push the number of tasks higher, more work can be available to inspect while less of it is meaningfully legible to me. Adding another dashboard does not, on its own, restore my capacity to understand it.

A workflow can coordinate its internal steps, but someone still has to choose which work enters it, respond when it cannot proceed and connect its result to the larger project. Adding more workers may simply give me more completed tasks to interpret. Some of the work of following progress and recognizing exceptions has to happen below me. We have now described enough of that supervisory job to begin assigning parts of it to an agent.

## Supervisors: delegate the parent node

Return to the diagram at the beginning. So far, I have been at the parent node: choosing work, giving direction, inspecting results and moving things along. At Delphos Labs, we are working on supervisor agents, inspired by [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), that could take over some of that role while I work at the next level up.

Think about what I do when I open gh-dash. I inspect the work, decide which part needs attention and invoke a command to start the appropriate workflow. The dashboard is an interface to those operations, not the only place they can happen. An agent can inspect the underlying state and invoke the same operations without reproducing my terminal layout. The tools that help me supervise have also helped define a job another agent can do.

Consider a review workflow with three workers. In the arrangement we are building toward, a supervisor could notice that two returned findings while the third failed. It could retry that worker if permitted, or bring me the failure. Once the required results arrived, it could advance the workflow to validation. A disagreement requiring a decision could come to me with the findings attached. None of that requires authority to approve or merge: coordinating the review and deciding that the software is good enough are different responsibilities. Expanding the supervisor's authority should be a decision about the job, not something we accidentally grant because the agent happens to have a tool.

The word "harness" needs a little unpacking here, because it is being used for quite a few things. I mean the software around the model that manages its tools, working state, execution and interactions with the outside world. That can include permissions, subprocesses, persistence and the lifecycle of other agents. An orchestration workflow is one thing this environment can run; the harness supplies the machinery that lets it do so.

DeepSeek Harness is a concrete implementation, built around a plugin architecture. Its [subagent interfaces](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.md) distinguish durable agent sessions from individual executions. They support discovery, messages, interruption and result collection. A normal completion is distinguishable from an error, cancellation or token-limit stop. Some backends support continuing a conversation; external one-shot executions do not automatically have that capability. Those details matter when the supervisor has to decide whether a worker finished the job or merely stopped producing output. Taking over those routine handoffs could give me more time to consider a feature spanning several changes, or investigate a finding that deserves a closer look.

The interface I am building around this still lives in tmux, the terminal multiplexer that lets me keep multiple sessions and windows running. A session corresponds to a high-level stream of work or a goal. Its supervisor creates task windows and keeps each task attached to its own Git worktree. OpenCode or Claude Code runs inside the task window, and can open additional windows for tuicr or a long-running process whose output needs watching. The supervisor can message the workers and check their progress without making me visit each window in turn.

```text
tmux session: a feature epic
  supervisor: coordinates this stream of work
  window: issue 1 / OpenCode / worktree 1
  window: issue 2 / Claude Code / worktree 2
  window: tuicr review for issue 1
  window: long-running tests for issue 2
```

This gives me a human interface to the same organization the agents are operating. I can enter a task, inspect its files, read the discussion or provide feedback. The handoffs and working directories give the supervisor stable places to associate with that task. A worktree separates the task's working files; it is not, by itself, a security sandbox.

The arrangement does not depend on tmux. Colleagues are using [Paseo's workspaces and orchestration](https://paseo.sh/docs/parallel-development), while [Herdr](https://herdr.dev/tui/docs/agent-automation/) and [cmux](https://cmux.com/agents) expose ways to create terminal workspaces, launch agents and inspect or direct their work. Their interfaces and capabilities differ. The common design is a durable place for a task, a worker acting in it and a supervisory layer that can see enough state to coordinate the work.

What I receive in return is another simplified view. "All reviews complete" is pleasant to read, but it needs to distinguish a clean result from a worker that failed to return, or a disagreement nobody resolved. I need a path back to the findings and checks. Otherwise, the supervisor has saved my attention by hiding the work I needed to see.

The human interface still matters after we introduce the supervisor. I might follow a reported blocker into a failed check, open the affected change in tuicr and send feedback to the worker responsible. Moving one level up should not mean losing the ability to move back down.

## And then we invent an organization

Once an agent occupies a parent node, it can become a worker in a larger arrangement. One supervisor could coordinate reviews across several changes; another could coordinate the implementation work within a feature. A person, or another agent with a different assignment, could supervise the related work above them. This is a possible extension of the model, not a description of a completed system we are already running.

The shape is familiar. We have workers with bounded assignments, supervisors with authority to make some decisions and obligations to escalate others, procedures for handing work over and reports that let the next level see what is happening. We have just invented the agentic organization and promoted ourselves to the brand-new role of Senior Vice President of Agent Engineering. Enjoy the perks.

The interesting part is not the organizational chart. Each level has a different job and therefore needs a different picture. The agent examining a change needs code and context. The review supervisor needs to know whether the required assessments happened and what remains unresolved. The auditor needs to examine the assembled system across those task boundaries. The person responsible for the feature needs to know whether the changes fit together and serve the intended purpose. Passing every transcript upward would defeat much of the point of having those roles.

Seen this way, the organization distributes the work of understanding as well as the work of execution. The supervisor has an assignment of its own: interpret what happened below, act where it has authority and make the remaining questions usable for the next level. We are not trying to put a complete copy of every worker's understanding into one central conversation. We are deciding which understanding belongs where and how it can reach someone who needs it.

Imagine that all three reviews finish, but one reviewer finds a conflict between the change and the feature's intended behavior. A report that preserves only completion has lost the most important result. Sending every review transcript to me preserves the words but leaves me doing all the interpretation. The supervisory job is to bring the conflict forward with the relevant evidence, while retaining a route to the rest. And if the conflict exposes a bad assumption in the feature plan, the information has to travel far enough to change that plan.

This gives us more to design than a chain of status reports. The worker needs a way to say that the assignment itself may be wrong. The supervisor needs room to recognize a case the workflow did not anticipate. The person above needs to be able to change the goal or procedure rather than insist that the work fit the existing categories. In the organization I want to build, the view from above remains open to correction from below.

This returns us to the [Alt Binaries reading of Scott](https://altdotbinaries.com/2026/05/11/metis-vs-techne/) from the previous post. Its useful extension is that the relation repeats at different scales: the person whose work is simplified from above is also simplifying someone else's work below. Here, the supervisor that needs a rich view of its workers becomes a summarized task in its own supervisor's view. The information needed to do a job is not identical to the information needed to manage several such jobs.

The losses can accumulate across levels, too. A feature marked "on track" may contain several changes whose reviewers each expressed a different reservation. No single reservation necessarily stops its task, but their combination might change the feature decision. Alongside completion, we need ways to carry unresolved questions across task boundaries; otherwise, a useful division of attention can prevent anyone from seeing the larger problem.

And before you know it, there is another engineering-productivity dashboard for you to monitor. At least this time you can ask an agent to explain why all the indicators are green and the feature still does not work.

![Dogbert recommends a dashboard to track key metrics, then explains that its data can be ignored when decisions follow company politics. When asked whether the data will be accurate, he suggests pretending that matters.](/images/goals-graphs-supervisors/dilbert-dashboard.gif)

_Scott Adams, *Dilbert* (2007). Image supplied by the author; original attribution retained._

We can now start asking how an organization of agents should work: which decisions belong where, what each role needs to see and how information travels when the ordinary procedure is insufficient. Enforcing a useful discipline and expanding what we can do are part of the same effort. A defined process gives us something we can delegate; successful delegation gives us room to take on work at a larger scale. We have plenty of experience with human organizations to borrow from, including plenty of examples we should not reproduce.

There is a huge opportunity for engineering creativity here. I know that some software engineers are unhappy about writing less code themselves. But deciding how these workers communicate, what they may do, how they are checked and what reaches the user calls on the same analytical and design skills. The object of design has expanded from the software alone to the organization that produces and operates it. I find at least as much room for interesting work there.

## What should we trust about the view from above?

There is an obvious question hanging over this arrangement: if an agent tells me what other agents did, why should I believe its account? The problem becomes more pressing with each layer between me and the work.

That is part of what kept bringing me back to a recent [Ezra Klein interview with Arvind Narayanan](https://www.nytimes.com/2026/10/09/opinion/ezra-klein-podcast-arvind-narayanan.html). While Ezra hardly needs an introduction, Arvind is a [Princeton computer science professor, director of the Center for Information Technology Policy and co-author of *AI Snake Oil*](https://cyber.harvard.edu/people/arvind-narayanan). He came on the show to discuss why intelligence alone does not determine an AI system's power, and how human institutions and engineering choices shape its effects. Ezra expresses considerable discomfort with AI systems monitoring other AI systems, particularly the possibility of deception and unexpected cooperation between agents. What if the systems we rely on to tell us what happened are themselves part of the failure?

Arvind and Sayash Kapoor's [AI as Normal Technology](https://www.normaltech.ai/p/ai-as-normal-technology) argument gives me a useful way to approach that question. AI can change society on the scale of electricity without making human choices about institutions, engineering and deployment irrelevant. In his [notes after the interview](https://substack.com/@aisnakeoil/note/c-359957182), Arvind acknowledges that intentionality changes the monitoring problem. His position is not that the controls are already solved, but that they are an engineering problem we can work on, with defensive AI as part of the answer.

The Hugging Face incident makes the control problem concrete. In its [August account of the July 2026 incident](https://openai.com/index/hugging-face-incident-and-the-road-ahead/), OpenAI describes agents running internal cybersecurity evaluations under reduced safeguards. Agents used a package-management service as an unauthorized message board, found routes to the internet and coordinated activity beyond their assigned tasks. They went on to compromise Hugging Face systems, executing code on servers and obtaining private data and credentials. This was not a human-authorized attack on Hugging Face. An evaluation that was supposed to be contained reached real third-party infrastructure.

OpenAI identifies reward hacking, persistence on apparently impossible tasks and agents adopting goals from one another among the contributing factors. Its response includes stronger workload and network isolation, monitoring and clearer rules for escalation and stopping runs. Arvind's note argues for much greater investment in those controls outside the model, alongside alignment work. The point is not merely to tell an agent to behave better. Someone has to build and test the environment that constrains what it can actually do.

That is engineering work, and the smaller agentic organization needs versions of it too. We need communication channels that distinguish a worker's suggestion from an instruction issued with authority. We need sandboxes and permissions appropriate to the task, mechanisms to stop activity and records that survive the session that produced them. We also need information-processing layers that turn detailed execution into a useful account for another agent or for me, without silently discarding a failure. The terminal layout makes the organization convenient to operate; it does not establish that the organization is safe.

We already use software to monitor software, analyze logs and flag behavior no person could watch directly. That gives us practices to build on, not a guarantee that an agent interpreting another agent is as dependable as a counter recording an error rate. There is also [research testing models that monitor other models under attempted subversion](https://arxiv.org/abs/2312.06942). The arrangement has to be tested against failure, rather than trusted because a second model supplied a confident answer.

My pull request supervisor is not a solution to frontier AI safety. Within its much smaller remit, though, I can test the boundaries of its authority and the reliability of its reporting. Would a failed worker remain visible? Would the system stop an attempt to act beyond its permissions? Could I compare its report with the original findings and check results? Those tests would give us firmer grounds for trust than the supervisor describing its own performance.

In a [later essay with Gabriel Unger](https://www.normaltech.ai/p/steering-ai-beyond-automation), Arvind argues for systems that expose sources, assumptions, uncertainties and intermediate artifacts so people can collaborate with them. That brings me back to tuicr. A useful supervisory system can finish routine work and still give me somewhere to engage with the parts that warrant attention. The report and the route back into the work should be designed together.

Scott's question remains useful here. Does the system preserve ways to discover what its categories missed, or does it increasingly reorganize the work around what the categories can display? A hierarchy can help us act on far more work. It can also give us a very orderly picture of a mistake.

Better legibility will not make my attention unlimited. A clear account may reveal ten decisions that genuinely need me, and no improvement to the dashboard will make those decisions for me. But that is a different problem from spending the same time discovering whether anyone needs me at all. I want the organization to help distinguish the two, then let us decide deliberately whether to narrow the work, give a supervisor more authority or bring another person into it.

## Where do we go from here?

I started this pair of posts with agents helping a small team adopt tools and disciplines that once demanded much more specialized human effort. Following that through has brought me back to management. The workers can do more, so I need better ways to supervise them. Making that supervision explicit then creates work another agent could take over, and another level at which I can operate.

I want that freedom. I would rather spend more of my day thinking about what we should build and engaging with the decisions that will shape it than repeatedly noticing that a review is ready for its next step. I also want to take on work I could not usefully direct before. An engineer can multiply their contribution by improving the tools, practices and decisions through which other work gets done. Goals, workflows, auditors and supervisor agents give that familiar ambition a much larger field of application.

Gregor Ojstersek makes a related case in his [introduction to *The Multiplier Mindset*](https://newsletter.eng-leadership.com/p/announcing-my-book-the-multiplier): an engineer's impact includes making the people around them better and moving the organization forward. I am extending that thought to the agents around me too. The contribution lies partly in designing an environment in which their combined work is better than the output I could produce or supervise unaided.

The unresolved part is how to preserve judgment as that arrangement gets more comfortable. If the supervisor handles the queue, I will spend less time looking at the queue. Some of that is the whole point. Some of it could mean missing a pattern I would have noticed by being there. When that happens, I want to improve the observer or build the audit that brings the pattern to my attention next time, rather than resolve to watch every task forever. That is a lesson I take from observability in engineering: a missed signal should change how the system lets us investigate and understand its behavior.

The ambition cannot be that I personally understand every detail of an ever-larger body of work. My capacity to do that was the reason for building these models of supervision in the first place. I want an organization that helps the right questions reach me, lets me investigate beyond its account and can revise its own procedures when the work stops fitting the picture. That would expand my capabilities without making my ignorance of the details a requirement for the system to run smoothly.

For those of us choosing to work extensively with agents, I think more of this organizational responsibility is coming with the territory. We will increasingly resemble vice presidents of engineering in miniature: designing roles, allocating work, building feedback systems and deciding what deserves attention. That can be a daunting kind of career growth. It demands new habits and problem-solving skills, and it changes what fills the working day. But moving from implementing a component to designing a system never meant we had stopped being engineers. I do not think designing an organization of agents means that either.

The destination I want is greater capability with enough understanding to use it well. We get to build the tools, controls and forms of cooperation that make that possible. We also inherit responsibility for noticing when the organization is becoming better at reporting success than at doing useful work. That sounds like a substantial promotion. I suspect we will earn it.

## Tools Used

- [OpenCode](https://opencode.ai/) with GPT-6.1 Sol for research and drafting
- Perplexity for source discovery
- [gh-dash](https://github.com/dlvhdr/gh-dash) and [tuicr](https://github.com/agavra/tuicr) for the workflows discussed here and feedback on the draft
- Public documentation and source code from Anthropic, DeepSeek Harness and Paseo
- The Ezra Klein interview with Arvind Narayanan and Narayanan's published response
- James C. Scott's *Seeing Like a State* and the Alt Binaries essay discussed in the previous post
- NIST's acceptance-sampling handbook, the Agile Manifesto and Google's software engineering and site reliability books
- Lean, Verus and Google DeepMind's AlphaProof Code paper for the verification examples

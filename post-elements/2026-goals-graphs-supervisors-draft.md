---
title: "Goals, graphs and the work of watching agents"
excerpt: "Making agent work legible, then delegating some of the work of watching it"
tags: [AI, Agents, Software Engineering, Knowledge Work]
comments: true
modified: 2026-11-09
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

In my [last post](/blog/2026/07/03/legibility-agents/), I discussed legibility and agents, and I am going to finish the second half of that work now. The first half was about the bargain these tools offer: agents make it much easier to work within strict systems, from Rust's type checker to Bazel's dependency graph, while quietly separating the production of software from the human understanding that used to come with writing it. We gain a great deal of capability. How do we keep enough visibility into the work to use that capability well?

You could group a lot of my engineering work these days into one of three broad categories:

- Developing my own work, often across an epic, up to the point where it is ready for review.
- Reviewing other people's work and giving them useful feedback.
- Addressing the comments others leave on my work and getting pull requests through to merge.

None of that is a new development process. We still define the task, write the software, review it and get it into users' hands. As I described in [It's Agents All the Way Down](/blog/2026/02/14/agents-all-the-way-down/), agents now do much of the work within those stages, while I move between them, provide direction and help at the decision points. The process has become more explicit as we have found ways to delegate more of it.

In this post, I want to work through what that looks like, starting with the tools I use to see and act on the work. From there, I will describe goals and graphs, two complementary ways of making agent work more legible, before moving up another level to supervisor agents. That last step leads into a broader question raised in a recent Ezra Klein interview: what does it mean to rely on agents to watch other agents?

## First, somewhere to see the work

To get this started, I want to talk about two critical tools in my current workflow: [gh-dash](https://github.com/dlvhdr/gh-dash) and [tuicr](https://github.com/agavra/tuicr). Both are terminal tools, but they help with different parts of the job. gh-dash gives me a surface for grabbing and addressing work; tuicr lets me get into the work itself and give an agent precise feedback.

### gh-dash: the queue and the way into the work

gh-dash organizes GitHub pull requests and issues into configurable views. I can see the work I need to review, check the state of my own changes and decide what to pick up next. That would be useful on its own, but the launcher scripts around it are just as important as the dashboard. They connect an item in the queue to the appropriate agent session or review tool, so choosing the work also gives me a way to start addressing it.

![gh-dash showing a queue of pull requests with details for the selected change](/images/goals-graphs-supervisors/gh-dash.png)

_gh-dash's public demonstration, showing the pull request queue and detail pane. The launcher scripts I use sit around this interface. [Source: gh-dash](https://github.com/dlvhdr/gh-dash/blob/e6752f8f2cf10f650d4f6e136385ba70f57d11f1/docs/src/assets/overview.gif). This is not a company dashboard._

The questions I bring to the queue depend on the work. When an agent is developing something, I want to know which task it is working on and whether it is blocked. When I am reviewing, I need the context for the change and somewhere to inspect it. When I am trying to merge a pull request, I am looking for what is left: failed checks, unanswered comments or a decision that needs to be made. gh-dash and the launchers tie those questions to the workflows I already use.

I like this because it puts the work at the center.

I do not have to start by finding the right conversation, remembering what was happening in it and then working out which command to give the agent. I can start with the thing that needs attention and enter the conversation from there.

### tuicr: feedback where the work lives

Once I get into a change, tuicr is the best tool I currently have for editing with an agent and providing direct feedback. It presents files or diffs in a review interface, lets me attach comments to the relevant passages and exports those comments with their file and line references. I can hand the result back to an agent without copying text into a chat and explaining which part I meant.

![tuicr showing a side-by-side code diff with an inline review comment](/images/goals-graphs-supervisors/tuicr.png)

_An inline comment in tuicr's public demonstration. The same feedback mechanism works for prose and specifications. [Source: tuicr](https://github.com/agavra/tuicr/blob/f098fb8f0911f6b9b14c80e23f79ab084cf5ef91/public/tuicr-demo.gif)._

The feedback on an earlier draft of this post came back through that interface. Comments attached to particular paragraphs explained where the writing felt abrupt, where I needed to introduce an idea and where the structure was getting repetitive. That is exactly the kind of conversation I want to have with an agent: look at this passage, here is what is wrong with it, carry that understanding into the next draft.

Although tuicr is primarily a code review tool, I think this interaction should be available almost everywhere we work with agents. A specification, a design document and a blog post all benefit from being able to point at the work and discuss it in place. The file is not just the output at the end of the conversation; it becomes part of the conversation.

Claude's [Artifacts](https://claude.com/features/artifacts) supports this kind of feedback, with comments and editing alongside the work. [Cowork also supports selecting text in a Markdown draft and asking Claude to edit that passage](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork). These are important improvements in the user experience, not just conveniences. If we want people to engage with what an agent produced rather than skim its summary, we should make that engagement easier.

Colleagues who use [Paseo](https://paseo.sh/docs/parallel-development) are working with another version of this emerging environment. Its documented workflow brings agent conversations, files, diffs and terminals together, with separate workspaces for parallel tasks and split panes for reviewing the result. The details differ from my terminal setup, but the need is similar: somewhere to organize the agents, somewhere to inspect their work and a short path from inspection to useful feedback.

I expect these capabilities to become foundational parts of agent platforms. Better review interfaces will not automatically restore the understanding we used to gain from writing code, but they give us a much better place to start than asking people to supervise everything through a chat transcript. And once those interfaces are connected to explicit workflows, we can begin delegating more than the individual task.

## Goals: make the destination explicit

The first approach I want to discuss is goals. Rather than tell an agent exactly how to proceed, you make the outcome explicit and let it work toward that outcome, changing its approach as it learns. This is a familiar way to delegate to a person, but it becomes especially interesting when the worker can run through many attempts without you prompting each one.

We have tried this a few times at Delphos Labs. Most recently, I asked Claude to audit my pull request review workflow and adapt it for current Claude and Sol models. I pulled in recent pull requests as benchmark material, supplied high-level feedback about good prompt design inspired by Anthropic's [prompt-audit guidance](https://github.com/anthropics/skills/blob/main/skills/claude-api/shared/prompt-audit.md) and set it to work. I wanted the agent to have room to investigate and improve the process, rather than make a list of edits I had already decided on.

Most of the changes happened without deep human review.

Instead, I used the reviewers while the audit was ongoing: iterate, try them on real work, iterate again. I also reached out to colleagues as they received reviews. Their responses, together with my experience of using the changing system, helped refine my feeling for what was working.

I was still involved, but the involvement was different. Reading every instruction change would have made me the editor of the review system. Using it while it changed made me a source of feedback about its behavior, including the parts that are difficult to specify in advance: whether a comment is useful, whether a review directs attention to the right place and whether the whole exercise helps someone make a better decision about the change.

I cannot put a percentage on the improvement. What I can describe is the freedom this gave the agent: with the outcome and constraints explicit, it could investigate and try changes I had not prescribed, while I assessed the behavior those changes produced. That is why I find goal-directed work promising.

The [Ralph Wiggum loop](https://ghuntley.com/ralph/), which Geoffrey Huntley described in July 2025, was an influential early coding version of this idea. Keep running an agent against a task, carry useful state forward and give it checks that push back on bad work. The simple loop supplied persistence, but the fuller practice included specifications, tests and human observation feeding back into the instructions. The interesting behavior came from putting those pieces together.

Now this is becoming a product feature. Claude Code's [`/goal` command](https://code.claude.com/docs/en/goal.md) lets you state a completion condition, then uses a separate model to assess that condition after a turn finishes. If the condition is not met, another turn begins. The agent can keep working without waiting for you to come back and type "continue."

Of course, the quality of the goal matters enormously. "Find more issues" would be a terrible objective for a reviewer if it rewarded a flood of plausible but incorrect comments. The work needs constraints as well as a destination, and completion needs evidence that does not disappear into the agent's confidence about itself. Claude Code's documentation makes one limitation particularly clear: its goal evaluator reads the conversation, rather than independently running commands or inspecting files. A separate evaluator is useful, but it still has to see the right evidence.

This connects to the problem from the last post about an agent writing its own tests. If the work and its measure share the same misunderstanding, green checks can reinforce the mistake. Giving the agent more attempts does not fix the measure. That is where independent checks, bounded runs and feedback from actual use become important.

For me, the attraction is not that a goal makes the work completely predictable. It gives me a stable question to ask while the agent explores: are we getting closer to the thing I actually want? The route is allowed to surprise me.

## Graphs: make the coordination executable

Goals make the destination explicit. There is another way to make the work easier to follow: write down the procedure in a form that software can execute. This is the idea behind much of the recent excitement about "graphs," a term you may have seen all over your feed if you follow people building with coding agents.

I tend to think of these as structured workflows. Graphs are not new, and neither are subagents. What is exciting is how readily an agent can now turn a description of work into a program that coordinates other agents. We already trust these tools to write scripts and call tools; starting another agent can be one more operation in the script.

Anthropic [announced dynamic workflows](https://x.com/ClaudeDevs/status/2060044853279617150) in May, and its [documentation](https://code.claude.com/docs/en/workflows.md) makes the change in responsibility easy to understand. In an ordinary conversation with subagents, the model decides what to launch next and receives the results back into its context. In a dynamic workflow, Claude writes a JavaScript orchestration script; the runtime executes it, keeping the intermediate results in variables and following the branches and loops in the program.

That sounds like an implementation detail until you consider how much coordination can otherwise become part of the conversation. Launch a reviewer. Wait for it. Decide which reviewer comes next. Carry the findings into another prompt. Keep track of which workers finished and whether the validation happened. When those relationships are already known, a script can hold them without asking the model to reconstruct the process at every transition.

The graph describes what depends on what. Independent reviews can run in parallel; a validation step can wait for their findings; a failed check can return the affected work for correction. [One widely circulated explanation on X](https://x.com/hanakoxbt/status/2091515787366306154) puts the relationship to loops neatly: "The loop lives inside a node. The graph lives between them." A worker may have considerable freedom to achieve its goal while the surrounding procedure determines how its result moves through the system.

This is how our pull request review work at Delphos Labs is developing. A script drives preparation and discovery, including a subagent's assessment of how we should review the change, then fans the reviews out across a team. The models still do the assessment. The procedure determines how to collect and use their work. Anthropic's [Code Review service](https://code.claude.com/docs/en/code-review) describes a related pattern of parallel review followed by verification, ranking and publication.

The agent overseeing the run can then spend its effort on exceptions: watching for a failure, responding within its authority and telling me when I need to step in. It does not have to improvise the whole review process. And I can inspect the procedure separately from the judgments made during a run, which is a useful distinction when I am trying to understand why something went wrong.

There is a version of this that will be familiar even if you do not use coding agents. Claude's document tools can write a script to generate a deck, render its slides, inspect the result and revise it. Anthropic's [public presentation skill described that workflow in late 2025](https://github.com/anthropics/skills/blob/ef740771ac901e03fbca3ce4e1c453a96010f30a/skills/pptx/SKILL.md), before Cowork launched. Cowork's [documented capabilities](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork) bring code execution, multi-step work and parallel subagents into a much more accessible interface. You ask for a finished piece of work and, underneath, Claude combines judgment with a series of scripted operations and checks.

I would not describe all of Cowork as a graph engine, but it is a useful way into the idea. The work is already more than a model producing one answer. Dynamic workflows take another step by making the coordination itself executable, rather than leaving it entirely to the agent's conversation.

We can be deliberate about what each worker sees, too. Dave Rensin's [Elephant-Goldfish model](https://opensource.googleblog.com/2026/10/introducing-elephant-goldfish-model-plugin.html) separates a long, context-rich design discussion from fresh sessions that assess the resulting specification. The Elephant knows the history; a Goldfish has to make sense of the document without it. If the fresh reviewer cannot understand the plan, something important never made it out of the conversation and into the artifact.

A workflow gives us a place to encode that separation, along with the model, reasoning settings and other parameters for each job. The fresh reviewer may still share the designer's model weaknesses, but it can expose assumptions that the original conversation took for granted. We can examine the handoff itself and improve it, instead of discovering much later that the next agent never understood what we wanted.

The enthusiasm on X sometimes turns this into a progression where graphs replace loops and prompting is suddenly obsolete. I do not think that is the useful interpretation. Goals and graphs can sit inside each other, and neither removes the need to describe good work. One gives an agent room to search toward an outcome; the other makes the relationships between pieces of work explicit enough to run and inspect. Together, they give us much more to work with than a collection of conversations.

Of course, a script can run a bad procedure perfectly. The gain is that the procedure is now available for inspection and revision. We have moved one more part of the work out of an agent's moment-to-moment decisions and into something both people and software can operate.

## Supervisors: let an agent operate the dashboard

Now we can return to the dashboard, because there is a natural next step. At Delphos Labs, we are working on supervisor agents, inspired by [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), that take over some of the watching and coordination I currently do myself.

Think about what I do when I open gh-dash. I look at the work, decide which part needs attention and invoke a command to start the appropriate workflow. The dashboard is an interface to those operations, not the only place they can happen. A supervisor can inspect the same state and execute the same commands without pressing my keyboard shortcuts or reproducing my terminal layout.

This is the abstraction above the dashboard: an agent starts doing some of what I do with the tools, rather than providing another screen for me to look at. Where I used to switch into a conversation to ask what happened, it can collect the result. Where the next step is already defined and permitted, it can launch that step. And where the work needs my judgment, it can bring me the question with the relevant context.

We are building toward this, and the explicit workflows give us a useful starting point. The supervisor can ask whether a goal has been reached, observe a workflow's progress or send a worker back to address a failed check, without having to invent the process for getting a pull request from one state to another.

DeepSeek's [public subagent interfaces](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.md) include the mechanics you would expect: finding agents, messaging them, interrupting them and collecting results. The same discipline we apply to workflows can extend to those interactions. A worker should have a clear way to report completion or a blocker; the supervisor should know which responses it can make and when it must ask a person.

The benefit I am after is fewer conversations competing for my attention. I still want to inspect a change in tuicr or go back to the dashboard when that is useful. But I do not want every piece of work to wait for me to notice that it is ready for the next ordinary operation. We have made the workers faster; now we can do something about the human repeatedly moving work between them.

This also puts pressure on the quality of the supervisor's account. "All reviews complete" is a pleasant thing to read, but it needs to distinguish a clean result from a worker that failed to return, or a disagreement nobody resolved. I need a path back to the findings and checks, not just a reassuring sentence. Otherwise, the supervisor has saved my attention by hiding the work I needed to see.

And here is the part I find especially interesting. The tools that make the work easier for me to see also help define the work another agent can take over. The launchers I use from the dashboard become operations a supervisor can invoke. The review feedback becomes something a worker can address. We are not throwing away the human interface; we are building on the structure it helped us create.

## The horizon beyond my terminal

This brings me to the broader horizon, and to a conversation I kept thinking about while working through these ideas. On a recent [Ezra Klein episode with Arvind Narayanan](https://www.nytimes.com/2026/10/09/opinion/ezra-klein-podcast-arvind-narayanan.html), the discussion turns to AI systems monitoring other AI systems. Arvind and Sayash Kapoor developed the [AI as Normal Technology](https://www.normaltech.ai/p/ai-as-normal-technology) argument: AI can change society on the scale of electricity without making human choices about institutions, engineering and deployment irrelevant.

Ezra expresses considerable discomfort with the monitoring picture. If one AI tells us what another AI did, and we increasingly depend on that account because we cannot follow the work ourselves, what exactly are we trusting? He is particularly concerned about deception and unexpected cooperation between agents, not merely the existence of complicated software.

I understand the discomfort. I also think that focusing on the word "agent" can make the arrangement sound stranger than it is. We already use software to monitor software, analyze logs and flag behavior no person could watch directly at the speed or scale it occurs. When an agent helps inspect another agent's work, we are extending that observation layer, not inventing the idea that machines can help us see what other machines are doing.

That does not make every such arrangement trustworthy. A system interpreting another agent's behavior has different failure modes from a counter recording an error rate. But it does suggest a more useful question than whether agents watching agents feels too strange to accept: what would make this particular arrangement worthy of our trust, and how would we find out when it is not?

Arvind's [notes after the interview](https://substack.com/@aisnakeoil/note/c-359957182) are useful here. He explicitly acknowledges that an agent's intentionality changes the problem compared with traditional engineering systems. His position is not that the controls are already solved, but that they are an engineering problem we can work on, with defensive AI as part of the answer. There is already [research testing models that monitor other models under attempted subversion](https://arxiv.org/abs/2312.06942), rather than assuming that another model provides an independent guarantee.

I find that position much closer to the work in front of me. My pull request supervisor is not a solution to frontier AI safety, but it does require choices about what an agent can do, what it must report and where a human decision belongs. We are designing the execution and observation layers together. The goals, workflows and communication contracts are some of the means available to us, and their quality matters as much as the number of agents we can run.

There is also more than one kind of success to pursue. In a [later essay with Gabriel Unger](https://www.normaltech.ai/p/steering-ai-beyond-automation), Arvind argues for systems that expose sources, assumptions, uncertainties and intermediate artifacts so people can collaborate with them. That connects back to tuicr and the review of this draft. The agent does not have to choose between finishing work for me and giving me somewhere to engage with it; a better interface can help it do both.

This is why I do not see legibility as a temporary problem we will solve by adding one more reviewer. Each new layer changes what we can see and what we have to understand. The practical response is to keep improving the tools and practices at those boundaries, rather than insist that a human must personally perform every act of observation for any of it to count.

## Conclusion

I started this pair of posts with agents helping a small team use tools and disciplines that once demanded much more specialized human effort. Following that through brings us to a different question: how much of the surrounding work can we delegate without losing our ability to understand and direct it?

I am excited by how much room there is to improve, especially in the work between tasks. Getting the agents to write code faster has brought us to the point where I can spend a substantial part of my day helping finished work take its next step. I would rather spend more of that time thinking about what we should build and engaging with the decisions that will shape it, and I want the supervisor to help make that possible.

The unresolved part is how to preserve judgment as that arrangement gets more comfortable. If the supervisor handles the queue, I will spend less time looking at the queue. Some of that is the whole point. Some of it could mean missing the pattern I would have noticed by being there, and I do not yet know how well we can distinguish the two.

That is what I want to test next. Not whether I can run one more agent, but whether the next layer gives me a better relationship to the work: less time moving it along, more useful opportunities to question it and enough understanding to recognize when I need to step back in.

## Tools Used

- [OpenCode](https://opencode.ai/) with GPT-6.1 Sol for research and drafting
- Perplexity for source discovery
- [gh-dash](https://github.com/dlvhdr/gh-dash) and [tuicr](https://github.com/agavra/tuicr) for the workflows discussed here and feedback on the draft
- Public documentation and source code from Anthropic, DeepSeek Harness and Paseo
- The Ezra Klein interview with Arvind Narayanan and Narayanan's published response

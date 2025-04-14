---
layout: post
title: "Some thoughts after a month of working on AI"
excerpt: "It's been one month since I left Google for an AI startup, which makes it a good time for some reflection."
tags: [AI, LLMs, MCP]
comments: true
modified: 2025-04-13
use_math: true
image: 
    feature: island.jpg
    credit: ImageFx
    creditlink: https://labs.google/fx/tools/image-fx/5pugg7pnm0000
---

{% include _toc.html %}

# Introduction

It's been a month since I left Google and started working at [Delphos Labs](https://delphoslabs.com/), an AI-focused startup. A lot happened in that time.

*  A lot of new models (or access to previously announced models): [GPT-4.5](https://openai.com/index/introducing-gpt-4-5/), [Gemini 2.5](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/), [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), etc. etc.
*  Huge amounts of new funding coming in, topped by Anthropic's [$3.5 billion](https://www.anthropic.com/news/anthropic-raises-series-e-at-usd61-5b-post-money-valuation) funding round
*  The continued rise of agents as the primary abstraction for LLM development, and the accompanying burst of 

And of course, [GPT 4o Image Generation](https://openai.com/index/introducing-4o-image-generation/) came out, which definitely made a splash on X. And while my little test of the new model wasn't especially Ghibli-inspired, I was quite happy with the results.

![cartoon of a skier crashing](https://michaelquinn32.github.io/images/crash.png "That looks fun")

While this post won't be necessarily focused so much on the news, I wanted to use the milestone as an opportunity to collect my thoughts on my experience thus far. These aren't necessarily that fleshed out, but I think they will be common themes as Delphos and the rest of the industry continue to grow.

# The rise of small projects

Over the last month, many of the AI projects that I've been working on can be easily described as *small*. Instead of building a complete, integrated system, the project focuses on completing one or more tasks. These task might chain together to eventually form a larger system, but each component comes with its own set of requirements and goals. Think of projects like

*  Providing a [GPT](https://openai.com/index/introducing-gpts/) to another team member to help them with their work
*  Or building a GPT for myself to speed up searching for information during code review
*  Building a specific agent to accomplish a data collection task, often with search
*  Doing extensive research with an agent to eventually configure other tools

These type of quick, small projects, often geared towards a specific goal, have really focused on my thoughts on the value of AI, LLMs and the forthcoming wave of agents that we will all interact with. Often, we are thinking about a task that we would like a critically-minded, detailed-orient person to take on, but there isn't capacity to fully hand this off. Instead, AI steps in as the next best, and often better, option.
This is a huge shift from the past, where we would often think of AI as a way to automate a task that was already being done by a human. Now, we are thinking about how to augment our own work with AI.

On a day-to-day basis, I'm also seeing my own behavior shift towards an AI first approach and away from traditional tools. Search is the most obvious of these. While I don't necessarily have a single preferred option, I find myself using either Perplexity, ChatGPT or Gemini before looking for an answer in a search engine. Moreover, the way I search seems to be changing, as I am becoming more prone to ask questions than to try and track down a reference that I already have in mind.

# Text is data; data is text

All of these small projects revolve around text data in various forms.

*  I need something that can reference pages in a book so that I don't have to do the search
*  I need a reviewer for a dozen or so papers
*  I need to quickly search through a whole repository of code

So much work over the last couple of decades has been focused on applying structure to this type of data. A favorite example of mine is [Kythe](https://kythe.io/), which creates a language-agnostic graph schema for a repository of code. It powers one of the best tools within Google: the ability to [search the Google mono-repo](https://abseil.io/resources/swe-book/html/ch22.html).

I'm seriously questioning about the long-term future of such tools. Why invest all the engineering hours into a new tool like Kythe when almost all questions can be answered by an LLM that fits the codebase into context? What other similar body of knowledge problems will we knock off by simply getting more powerful models over the next couple of years? And as data scientists, we should be heartened to learn how good LLMs are becoming at [structured data extraction](https://docs.llamaindex.ai/en/stable/use_cases/extraction/). It opens up all new realms for applying measurement and statistics to new types of data.

# The burden of choice

I swear, one of the hardest problems with switching from Google to a new startup is the sheer number of choices that you have to make. At Google, we had a lot of tools and systems that were already in place, and most of the developer environment was essentially "solved." You don't pick your own editor, build system, linter, formatter, etc. You just use the tools that are already there.

Now, even though most of my startup uses VS Code, Cursor or similar tools built on the same foundation, I still need to figure out

*  How we configure [ruff](https://docs.astral.sh/ruff/)
*  If we are using [mypy](https://mypy-lang.org/) or if [pyright](https://github.com/microsoft/pyright) is sufficient for most of our needs
*  Which LSP in VS Code is supposed to tie this all together
*  What to do about other code health tools, like [dead code monitoring](https://github.com/jendrikseipp/vulture), [deps checks](https://github.com/fredrikaverpil/creosote), etc. 

Progress in the field of AI means that this getting started problem applies to almost all of the projects that I've taken on so far. While I don't care too much about development environments, I need to think about

*  Models used
*  Frameworks for calling the models
*  If we need to add tool use
*  Should we go through an existing provider that already has a UI in place
*  etc. etc.

I would like to think that we've largely settled on [PydanticAI](https://ai.pydantic.dev/) for most use cases, but a project that I was working on just tonight revealed a gap in what I could do with that framework. I ended up using [LiteLLM](https://www.litellm.ai/) instead. At the current moment, I think the standard set of features includes:

*  Structuring outputs and providing validation; Pydantic's models are good at this but the exact LLM support varies
*  Connecting to different tools and especially search, which seems ever more critical
*  Easy approaches to swapping out models and providers
*  Chaining together different models and tools
*  And probably a lot more

Swapping models is especially important, as the field is moving so quickly. The LLM arena is a good reference point for the current landscape today, but no "winning" model seems to stay at the top for more than a couple of weeks.

<iframe width="560" height="315" src="https://storage.googleapis.com/public-arena-no-cors/scatterplot.html" width="560" title="lmarena.ai/price
"></iframe>

*From [lmarena.ai/price](https://lmarena.ai/?price).*

<br/>

# Even ICs are now managers

A lot has been said about the future of software engineering in the age of AI. While there is more to the debate than what I can get to here, one clear trend is that the role of the software engineer is changing. In the past, we were often seen as the "doers" of the team, responsible for writing code and implementing features. Now, with the rise of AI and LLMs, our role is shifting towards that of a manager or overseer. I often apply the model of the Google career ladder to better frame this trend. As a great Google joke goes:

> L4s code but don't talk; L5s talk and code; L6s talk and don't code

LLMs are quickly becoming the "L4" of the software engineering world. They can write code, but they don't have the same level of understanding or context that a human engineer does. As a result, we are seeing a shift towards a more managerial role for software engineers, where we are responsible for overseeing the work of LLMs and ensuring that they are producing high-quality code.

One big part of that is in the importance of system design, architecture and the overall flow of information. We are still critical in understanding and describing both the nature of the problem and the steps needed to solve it. This is a big highlight from the recent episode of Cognitive Revolution on [AMIE](https://research.google/blog/amie-a-research-ai-system-for-diagnostic-medical-reasoning-and-conversations/). The developers of the system spent a lot of time talking to practitioners about asking them "how do you think about" diagnosing disease, suggesting treatments, etc.

<iframe width="560" height="315" src="https://www.youtube.com/embed/Jo31LEns2zY?si=vz08v-Yn-D9h1-kq&amp;start=2473" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<br/>

That said, while I think that writing is one of the most important skills for a good people leader, I'm still not sure on what the right approach to communication is for when we're working with models. Most guides to [Prompt Engineering](https://drive.google.com/file/d/1AbaBYbEa_EbPelsT40-vj64L-2IwUJHy/view) suggest that we are entering a paradigm where we communicate with models in a very particular style. The exact nature of this style of communicate is still being worked out. For example, how critical is formatting a prompt? With reasoning models, do we need to follow chain of thought? As context windows grow, is simplicity still critical? I expect a lot of this to get worked out as models continue to gain more capabilities.

# You have agency; you are king 👑

I want to wrap this up by diving into one of the more controversial topics in the AI world: the rise of agents. So much of the conversation focuses on things like job replacement, automation and the like. But I think that the most important thing to remember is that we are still in control, and the collective we will remain in control for a very long time. AI systems are already incredibly smart, but they are ultimately guided by humans. It is up to us to steer them, enforce align and ultimately derive value from them. It seems a little unfair to tap Andrej Karpathy twice in the same post, but I'll do it anyway. Here's what we had to say on all of this.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Agency &gt; Intelligence<br><br>I had this intuitively wrong for decades, I think due to a pervasive cultural veneration of intelligence, various entertainment/media, obsession with IQ etc. Agency is significantly more powerful and significantly more scarce. Are you hiring for agency? Are… <a href="https://t.co/8yvECKi7GU">https://t.co/8yvECKi7GU</a></p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/1894099637218545984?ref_src=twsrc%5Etfw">February 24, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Another way: *You can just do stuff*. And the new tools make you all the better at doing that stuff. What a time to be alive.

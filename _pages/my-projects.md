---
title: My Projects
modified: 2026-04-20
excerpt: "Michael's ongoing and occassionally updated projects."
permalink: /my-projects/
author_profile: true
classes: wide
header:
    image: /images/projects_main.jpg
    caption: "Photo credit: Michael Quinn"
---

## AI Matrix Live

[AI Matrix Live](https://aimatrixlive.com/) is a public dashboard tracking global AI adoption using the AI Matrix framework from [Simpson (2026)](https://doi.org/10.5281/zenodo.18181372). It plots countries on two dimensions, access and agency, placing each in one of four quadrants: Full Empowerment, Elite Empowerment, Passive Dependency, and Full Dependency. The data comes from the Anthropic Economic Index on HuggingFace, and the pipeline detects new releases, applies the paper's methodology, and deploys automatically.

[You can view the source on GitHub](https://github.com/michaelquinn32/ai-matrix-live).

## Measuring Validity and Reliability of Human Ratings

In this blog post for the *Unofficial Google Data Science Blog*, Jeremy Miles, Ka Wong and I take on the challenge of human-labeled data. While we often treat human judgements as ground truth, there is a wide variety of theoretical and practical challenges to using this data. We discuss some of the challenges around measurement, provide concepts and definitions for the type of work we do at Google, and explore a case study of measuring the quality of human-provide labels across different platforms.

[You can read the whole article here](https://www.unofficialgoogledatascience.com/2023/07/measuring-validity-and-reliability-of.html).

## Patrick: Parameterized testing in R is kind of cool!

I wrote and maintain the R package [`patrick`](https://github.com/google/patrick), which is an adaptation of Python parameterized testing libraries like [`parameterized`](https://github.com/wolever/parameterized). With `patrick`, you can R tests in the common `testthat` framework and then add parameters so that cases are more reusable. This is especially useful for testing functions that take a variety of inputs, like statistical models or data transformations.


## Large scale machine learning using TensorFlow, BigQuery and CloudML Engine within RStudio

I spoke on behalf of Google Cloud at [rstudio::conf 2018](https://posit.co/resources/videos/large-scale-machine-learning-using-tensorflow-bigquery-and-cloudml-engine-within-rstudio/). The talk showcased how someone can quickly start an instance of RStudio Server in Google Cloud and then use already-connected Google products like TensorFlow and BigQuery to solve Data Science problems. This came together in a demo I prepared that built on model on Google Analytics data. 

## Bayesian Portfolio Optimization in R

The project was originally a paper that I published in the Central Asia Business Journal, which is [edited by former colleagues at KIMEP university](https://kimep.kz/academics/en/central-asia-business-journal/). I'm sure that the digital edition of the most recent version will eventually go up. Or at least I hope.

Since the code was originally structured to accompany a paper, it's a bit of a mess right now. But I'm coercing it into a package, so that it might be useful to someone intrepid researcher wishing to reproduce the results published by the leading English-lange business journal in Central Asia. I'm sure it will be a great honor!

The package implements several Bayesian estimators for the parameters in a classic Markowitz portfolio optimization algorithm. A Gibbs sampler is used to find the Bayesian estimates.

[You can view it here on Github](https://github.com/michaelquinn32/bpoR).

## AdventureR

This package contains all of the functions written for posts on this blog. Obviously, the package will always be "in development," since this blog is an ongoing project. Nonetheless, this should make it a lot easier to organize, document and share the code produced here.

[You can view it here on Github](https://github.com/michaelquinn32/adventureR).

## LambdaList

I created this package to accompany my [R translation of *List Out of Lambda*](https://michaelquinn32.github.io/list-out-of-lambda-in-R/). Motivation for the project comes from Steve Losh, [whose article lays the foundation for purely functional lists](https://stevelosh.com/blog/2013/03/list-out-of-lambda/). I also relied on [Hadley Wickham and Lionel Henry's package `purrr`](https://github.com/hadley/purrr), which provided many of the functional programming examples in the article.

[You can view the package here on Github](https://github.com/michaelquinn32/lambdaList).

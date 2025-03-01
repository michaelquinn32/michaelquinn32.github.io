---
layout: page
title: My Projects
modified: 2015-10-24
excerpt: "Michael's ongoing and occassionally updated projects."
image:
  feature: projects_main.jpg
  credit: Michael Quinn
---

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

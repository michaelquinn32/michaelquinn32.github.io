---
layout: post
title: "Going Old School with the Spam Email Dataset"
excerpt: "It's time to break out some oldschool classification algorithms!"
tags: [R, Machine Learning, Classification, Spam Data]
comments: true
modified: 2015-12-06
use_math: true
image: 
    feature: clouds.jpg
    credit: Superfamous
    creditlink: https://images.superfamous.com/Cloud-Mountain-II
---




{% highlight text %}
## Error in library(dplyr): there is no package called 'dplyr'
{% endhighlight %}



{% highlight text %}
## Error in library(purrr): there is no package called 'purrr'
{% endhighlight %}



{% highlight text %}
## Error in library(readr): there is no package called 'readr'
{% endhighlight %}



{% highlight text %}
## Error in library(ggplot2): there is no package called 'ggplot2'
{% endhighlight %}



{% highlight text %}
## Error in library(caret): there is no package called 'caret'
{% endhighlight %}



{% highlight text %}
## Error in library(adventureR): there is no package called 'adventureR'
{% endhighlight %}

{% include _toc.html %}

## Introduction

John Marden was a Professor Emeritus when I attended UIUC. While I didn't have the chance to attend any of his classes, he had a huge influence on my academic career. He is the author of an incredible series of "Old School" courses notes on a variety of topics in statistical analysis.

* [Mathematical Statistics: Old School](https://istics.net/pdfs/mathstat.pdf) was my text for PhD-level math stats
* [Analysis of Variance: Old School](https://istics.net/pdfs/anova.pdf) was my theory linear models text
* [Multivariate Analysis: Old School](https://istics.net/pdfs/multivariate.pdf) was the textbook for my multivariate class

In honor of Dr. Marden, we're doing fitting some Old School classification models to an Old School Machine learning dataset. I'll be looking at two models today: linear discriminant analysis (LDA) and Naive Bayes (NB). I'll touch a bit on the theory behind each model and the classification problem itself. Most importantly, I've developed code to implement each model, as I think both are excellent learning opportunities. 

In that regard, this blog now has a package. [You can find it on Github](https://github.com/michaelquinn32/adventureR), and you can install the most recent version with `devtools`.


{% highlight r %}
# install.packages("devtools")
devtools::install_github("michaelquinn32/adventureR")
{% endhighlight %}

[Code is a means of communication](https://r4ds.had.co.nz/expressing-yourself.html), and good code is well-documented code. The easiest way to do that was just put the package together and throw in [Roxygen headers](https://r-pkgs.had.co.nz/man.html) for each function. That said, the blog's sister package is only meant to illustrate the exercises on the blog. It is permanently a work in progress, and nothing was written for perfect application in every context. *Caveat emptor*.

In today's post, I'll be working with the [Hewlett-Packard Spam Data](https://archive.ics.uci.edu/ml/datasets/Spambase). It's available in the UCI Repository Of Machine Learning Databases. While it's not that old as far as standard statistical datasets are concerned (the data were published in 1999), it's already considered a classic classification dataset. The familiarity of the data should help us focus on the real issue at hand, the classification algorithms.

## The Problem

### A Formal Description of Classification

Consider a multivariate dataset that consists of a $n \times p$ set of predictors $\mathbf{X}$, with $p$ columns corresponding to different variables and $n$ rows for each observation. We assume that each row is independently distributed. For classification problems, each observation belongs to one of $K$ groups described by the length-$n$ vector $y$.

$$
y \in \left\{ 0, \dots, K - 1 \right\}
$$

Terminology is always a bit of a problem, but I'll do my best to adhere to the terminology outlined by Hastie, Tibishirani and Friedman in [The Elements of Statistical Learning](https://statweb.stanford.edu/~tibs/ElemStatLearn/). The vector $y$ is our *outcome* or *target*, and our problem is one of *supervised learning*. In the spam dataset, an observation can belong to either the "spam" or "not spam" group, meaning that our problem is *binary*.

A prediction is ultimately a function. Given a single observation of $\mathbf{X}$, let's call is $x_j$ we want a function that produces our best estimate of the target $\hat{y}_j$. Formally, we want $\mathrm{G}()$ such that

$$
\hat{y}_j = \mathrm{G} (x_j)
$$

What do we mean by "best" estimate of the target? Well, that's a bit of a tricky question. Informally, we could define "best" as the estimator that gets us closest to the target. For whatever reason, statisticians prefer to formalize this statement through a *loss function* ($L$), which is a statement about our error. Here, we'll use $0-1$ loss:

$$
L(y_i, \hat{y}_i) = \mathrm{I} (y_i \neq \hat{y}_i)
$$

Our *risk* is the expected value of our loss function,

$$
R(y, \hat{y}) = \mathrm{E} \left[L(y, \hat{y}) \right]
$$

Expanding for a finite number of observations $n$, we get

$$
R(y, \hat{y}) = \frac{1}{n} \sum_{i = 1}^{n} \mathrm{E} \left[ \mathrm{I} (y_i \neq \hat{y}_i | \mathbf{X} =  x_i) \right]
$$

If you dig through a Math Stats book for awhile, you'll come across the property that the expectation of an indicator function is just the probability of the event. Let's plug that in.

$$
R(y, \hat{y}) = \frac{1}{n} \sum_{i = 1}^{n} P \left( y_i \neq \hat{y}_i | \mathbf{X} =  x_i \right)
$$

Minimizing our risk is equivalent to minimizing the probability statement in the left-hand side. Moreover, this minimization problem is equivalent to maximizing a slightly different probability statement. Thanks to the complement property of probabilities, 

$$
P \left( y_i \neq \hat{y}_i  \middle| \mathbf{X} =  x_i \right) = 1 - P \left( y_i = \hat{y}_i | \mathbf{X} =  x_i \right)
$$

This gives us what we're ultimately looking for. We want a function that maximizes the probability that our prediction is the same as the target, given the data that we have. Formally, that's 

$$
P \left( y_i = \hat{y}_i \middle|  \mathbf{X} =  x_i \right)
$$ 

In a multi-class problem, we would need to calculate the probability for each class separately and then pick accordingly. In the binary case, finding the max is much easier. Let $\mathrm{G} (x_j) = 1$ if

$$
P \left( y_i = 0 | \mathbf{X} =  x_i \right) < P \left( y_i = 1 | \mathbf{X} =  x_i \right)
$$

We can also just treat this as a ratio. This has some nice mathematical properties once we implement our different classifiers.

$$
1 < \frac{P \left( y_i = 1 | \mathbf{X} =  x_i \right)}{P \left( y_i = 0 | \mathbf{X} =  x_i \right)}
$$

The fraction on the right-hand-side is known as an expression of *odds*. For a variety of reasons,
it is often much easier to work with the *log odds*. This implies that $\mathrm{G} (x_j) = 1$ if

$$
0 < \log \left( \frac{P \left( y_i = 1 | \mathbf{X} =  x_i \right)}{P \left( y_i = 0 | \mathbf{X} =  x_i \right)} \right)
$$

### The Naive Bayes Classifier

A lot of the differences in our models come from how we evaluate the probability statements that form the log odds. For example, we can apply Bayes Rule. By that, I mean

$$
P \left( y_i = 1 | \mathbf{X} =  x_i \right) = \frac{P(y_i = 1) P(\mathbf{X} =  x_i | y_i = 1)}{P (\mathbf{X} =  x_i)}
$$

In the numerator of the expression above, $P(y_i = 1)$ is called the *prior probability*. It is a statement about the distribution of the groups within our data. The other numerator component, $P(\mathbf{X} =  x_i \| y_i = 1)$, is called the likelihood. In Bayesian statistics, we often get to ignore the denominator, so we often say that the posterior is proportional to the prior times the likelihood. 

Since we're basing our classification rule on a ratio, we don't even need to worry about normalizing constants. This is a big advantage of evaluating the odds instead of comparing the individual probability statements. Our problems simplifies to $\mathrm{G} (x_i) = 1$ if

$$
0 < \log \left( \frac{P(y_i = 1) P(\mathbf{X} =  x_i | y_i = 1)}{P(y_i = 0) P(\mathbf{X} =  x_i | y_i = 0)} \right)
$$

For the Naive Bayes classifier, we assume that each column of $\mathbf{X}$ is normally distributed and independent. This is often too general of an assumption to make, but Naive Bayes tends to perform well regardless. This means that the $j$-th column of the $\mathbf{X}$ matrix now has the following pdf:

$$
f_{x_j} (x_j) = \frac{1}{\sqrt{2 \pi \sigma_j^2}} \exp \left\{- \frac{(x_j - \mu_j)^2}{2 \sigma_j^2} \right\}
$$

Assuming that the columns are independently distributed gives us an easy shortcut to calculating the likelihood in the definition of Bayes Rule above: the likelihood is now just the product of the pdf of each column. Formally,

$$
P(\mathbf{X} =  x_i | y_i = 1) = \prod_{j = 1}^p P(\mathbf{X}_j =  x_{ij} | y_i = 1)
$$

Big products like this are often hard to work with, but applying the log takes the product and turns it into a sum. Plugging everything back into the odds ratio and simplifying, we get $\mathrm{G} (x_i) = 1$ if

$$
0 < \log \frac{p_1}{p_0} + \sum_{j = 1}^p \left[ \frac{1}{2} \frac{\sigma_{0j}^2}{\sigma_{1j}^2} + \frac{(x_j - \mu_{0j})^2}{2 \sigma_{0j}^2} - \frac{(x_j - \mu_{1j})^2}{2 \sigma_{1j}^2} \right]
$$

where $p_k$ are the group proportions, $\mu_{kj}$ are the group means for each column and $\sigma_{kj}^2$ are the column variances. This is a pretty easy classification rule to implement.

### Linear Discriminant Analysis

So that's Naive Bayes. Remember, the form the of the classifier is based almost entirely on our assumptions about how we'll evaluate the log odds. This isn't the only viable set of assumptions we could apply. Often, Naive Bayes is too general. Instead, we could assume that $\mathbf{X}$ follows a multivariate (MV) normal distribution. That is,

$$
\mathbf{X} \sim \mathcal{N}_p (\mu, \Sigma)
$$

The pdf for the MV normal distribution is similar to the univariate normal, with some additional matrix algebra.

$$
f_{x} (x_1, \dots, x_k) = \frac{1}{\sqrt{(2 \pi)^k | \Sigma |}} \exp \left \{ -\frac{1}{2} (x - \mu)' \Sigma^{-1} (x - \mu) \right\}
$$

Where $\| \Sigma \|$ is the determinant of the covariance matrix. Keep in mind that in the above equation we're dealing with vectors and matrices, not scalars. 

With this assumption, we can implement Linear or Quadratic Discriminant Analysis (LDA or QDA). The former has been around for almost 100 years now. Like most of Statistics, it was created by RA Fisher. [You can read the original article online](https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x/abstract), along with everything else published in the very unfortunately named *Annals of Eugenics*. The name's so bad that Wiley has to throw up a disclaimer. Ouch.

The methods differ over assumptions about the covariance matrix used in the problem. In LDA, we assume that different groups have the same covariance. We do not make this assumption in QDA. Grouped covariance makes the math in LDA a lot simpler, it also saves us from estimating a bunch of extra parameters. For that reason, QDA tends to overfit and you need a lot of data ( even more than the spam data ) to get improved performance. I'll show that in this post, but I won't actually implement QDA (*the remaining proof is left as an exercise for the student...*).

To estimate the covariance matrix in LDA, we use a *pooled covariance*:

$$
\hat{\Sigma} = \frac{1}{n - 2} \left( \sum_{i = 1}^{n_0} (\mathbf{X}_{0i} - \mu_0)(\mathbf{X}_{0i} - \mu_0)' + \sum_{i = 1}^{n_1}(\mathbf{X}_{1i} - \mu_1)(\mathbf{X}_{1i} - \mu_1)' \right)
$$

Where $\mathbf{X}_{ki}$ is the $i$-th observation within the $k$-th group, and $\mu_k$ is the group mean.

Returning to the log odds and plugging in the MV normal pdf, we get the following classification rule. In LDA, this is usually referred as the *discriminant function* $d_1(x)$.

$$
d_1(x) = \log \left( \frac{p_1 \frac{1}{\sqrt{(2 \pi)^k | \hat{\Sigma} |}} \exp \left \{ -\frac{1}{2} (x - \mu_1)' \hat{\Sigma}^{-1} (x - \mu_1) \right\} }{p_0 \frac{1}{\sqrt{(2 \pi)^k | \hat{\Sigma} |}} \exp \left \{ -\frac{1}{2} (x - \mu_0)' \hat{\Sigma}^{-1} (x - \mu_0) \right\}}\right)
$$

That's still pretty ugly. Luckily, there are plenty of opportunities for simplification. Dropping out like terms and carrying the logarithm through the division, we get

$$
d_1(x) = \log(p_1) -\frac{1}{2} (x - \mu_1)' \hat{\Sigma}^{-1} (x - \mu_1) - \log(p_0) + \frac{1}{2} (x - \mu_0)' \hat{\Sigma}^{-1} (x - \mu_0)
$$

Separating the terms that depend on $x$ from those that don't, we get

$$
d_1(x) = (\mu_1 - \mu_0)' \hat{\Sigma}^{-1} x - \frac{1}{2} \left( \mu_1 ' \hat{\Sigma}^{-1} \mu_1 - \mu_0 ' \hat{\Sigma}^{-1} \mu_0 \right) + \log \left( \frac{p_1}{p_0} \right)
$$

Or more simply,

$$
d_1(x) = a' x + b
$$

where

$$
a = \hat{\Sigma}^{-1}(\mu_1 - \mu_0)
$$

and

$$
b = - \frac{1}{2} \left( \mu_1 ' \hat{\Sigma}^{-1} \mu_1 - \mu_0 ' \hat{\Sigma}^{-1} \mu_0 \right) + \log \left( \frac{p_1}{p_0} \right)
$$

In multivariate statistics, a *quadratic form* is the matrix operation $a' X a$ where $a$ is a length $n$ vector and $X$ is an $n \times n$ matrix. The calculation of $b$ above uses two of them. This expression will come in handy later.

If you haven't noticed already, the expression $d_1(x) = a' x + b$ follows the canonical definition of a hyperplane. In discriminant analysis it is often referred to as the *separating plane*. If none of those words mean anything to you, don't worry, it's just the multivariate definition of the classical linear equation $y = ax + b$. Formally speaking, the separating plane maximizes the ratio of two sums of squares, $d_R$,

$$
d_R = \frac{a' B a}{a' W a}
$$

Where $B$ is the *between-group* sum of squares

$$
B = (\mu_1 - \mu_0) (\mu_1 - \mu_0)'
$$

And $W$ is the *within-group* sum of squares, which is essentially the pooled covariance estimate that we described above.

$$
W = \left( \Sigma_1 + \Sigma_0 \right)
$$

In the two dimensional binary case, it looks something like this. For now, ignore the function `LDA` and just focus on the image. I'll talk about it a lot more soon.


{% highlight r %}
# LDA example -------------------------------------------------------------

mus <- list(group1 = c(0,0), group2 = c(2,2))
Sigma <- matrix(c(2, -1, -1, 2), nrow = 2)

example_data <- map(mus, ~ MASS::mvrnorm(n = 50, mu = .x, Sigma = Sigma)) %>% 
    map(as.data.frame) %>%
    bind_rows(.id = "group")
{% endhighlight %}



{% highlight text %}
## Error in bind_rows(., .id = "group"): could not find function "bind_rows"
{% endhighlight %}



{% highlight r %}
example_lda <- LDA(group ~ . , example_data)
{% endhighlight %}



{% highlight text %}
## Error in LDA(group ~ ., example_data): could not find function "LDA"
{% endhighlight %}



{% highlight r %}
slope <- -1 * reduce(example_lda$a, `/`)
{% endhighlight %}



{% highlight text %}
## Error in reduce(example_lda$a, `/`): could not find function "reduce"
{% endhighlight %}



{% highlight r %}
intercept <- -1 * example_lda$b %>% as.numeric %>% `/`(example_lda$a[2])
{% endhighlight %}



{% highlight text %}
## Error: object 'example_lda' not found
{% endhighlight %}



{% highlight r %}
ggplot(example_data, aes(V1, V2)) +
    geom_point(aes(color = group)) +
    geom_abline(slope = slope, intercept = intercept, lty = 2) +
    scale_color_brewer(palette = "Set1") +
    ggtitle("LDA Example")
{% endhighlight %}



{% highlight text %}
## Error in ggplot(example_data, aes(V1, V2)): could not find function "ggplot"
{% endhighlight %}

We have two color-coded classes with a discriminating line running between them. Observations on one side of the line would be classified as one group and vice versa. Obviously, this is not a perfect classifier, but it is nonetheless optimal for a linear split. 

## Implementing Each Classifier

You can view the complete code for each classifier on Github. Since they're a little on the long side, I'll save you all from posting the complete code here. [Here's LDA](https://github.com/michaelquinn32/adventureR/blob/master/R/LDA.R), and here's [Naive Bayes](https://github.com/michaelquinn32/adventureR/blob/master/R/naive_bayes.R). Both functions depend on the `purrr` package (thanks Hadley!) for some functional programming tools. [You can learn about it here](https://github.com/hadley/purrr). If I had to pick a favorite package, it would probably be it.

The implementation of each classifier follows a similar strategy:

* Use `model.frame` to convert the formula input into a standard data frame. The target is always the first column and the remainder is always the set of predictors.
* Perform some assertions and clean up factors. I created a simple function `to_number` to convert a factor to numeric. It uses `purrr::compose` to chain `as.character` and `as.numeric`.


{% highlight r %}
to_number <- function(.factor) {
    compose(as.numeric, as.character)(.factor)
}
{% endhighlight %}

* Get the components of each equation: prior probabilities, means and variances.
* Last map the components to the data matrix and collect the results.

The mapping function does most of the work for each classification method. In `LDA`, the mapper calculates the equation of the discriminating plane.


{% highlight r %}
preds <- apply(x, 1, function(.x) crossprod(a, .x) + b)
{% endhighlight %}

In Naive Bayes, the mapper calculates the log odds with the column means and variances for each group.


{% highlight r %}
preds <- log(p_rat) + apply(x, 1, log_odds, mus, sigmas)
{% endhighlight %}

The function `log_odds` function calcates the kernel function for each class `class_dist` and adds the log of the variance ratio.


{% highlight r %}
log_odds <- function(x, mus, sigmas) {
    var_rat <- reduce(sigmas, `/`)
    class_dist <- map2(mus, sigmas, ~ (x - .x)^2 / .y)
    class_mat <- do.call(cbind, class_dist)
    sum(.5 * log(var_rat) + class_mat %*% c(.5, -.5))
}
{% endhighlight %}

## A Test

So, which function is a better fit for our spam problem? Well, we're statisticians. Let's create a test.

* Randomly split `spam_data` into test and training components
* Fit each model to the former and score on the latter
* Repeat many, many times (or 100 times. That's many, many. Right?)

Since this is a classification problem, I used [the F1 score](https://en.wikipedia.org/wiki/F1_score) as a performance metric. This score is the harmonic mean of two separate metrics:

* `precision` the proportion of cases predicted to be true that are actually true
* `recall` the proportion of true cases classified as true.

In code, we implement this with three simple equations.


{% highlight r %}
precision <- function(fit, actual) {
    sum((fit == 1) & (fit == actual)) / sum(fit)
}

recall <- function(fit, actual) {
    sum((fit == 1) & (fit == actual)) / sum(actual)
}

f1_score <- function(prec, rec) {
    2 * (prec * rec) / (prec + rec)
}
{% endhighlight %}

Before we get to the test, we should check out the spam dataset in detail. Let's pull it in from the web and coerce the target into a factor. We'll be comparing our methods against implementations within `caret`, and that package requires factors for classification.


{% highlight r %}
# Links -------------------------------------------------------------------

spam_http <- "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
spam_names <- "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"


# Get Data ----------------------------------------------------------------

col_names <- read_lines(spam_names, skip = 33) %>% gsub(":.*", "", .) %>% c("target")
{% endhighlight %}



{% highlight text %}
## Error in read_lines(spam_names, skip = 33): could not find function "read_lines"
{% endhighlight %}



{% highlight r %}
col_types <- c(replicate(55, col_double()), replicate(3, col_integer()))
{% endhighlight %}



{% highlight text %}
## Error in col_double(): could not find function "col_double"
{% endhighlight %}



{% highlight r %}
spam_data <- read_csv(spam_http, col_names = col_names, col_types = col_types)
{% endhighlight %}



{% highlight text %}
## Error in read_csv(spam_http, col_names = col_names, col_types = col_types): could not find function "read_csv"
{% endhighlight %}



{% highlight r %}
# Fix target --------------------------------------------------------------

spam_data <- mutate(spam_data, target = as.factor(target))
{% endhighlight %}



{% highlight text %}
## Error in mutate(spam_data, target = as.factor(target)): could not find function "mutate"
{% endhighlight %}

An EDA should show some potential pitfalls in the test.


{% highlight r %}
# Basic EDA ---------------------------------------------------------------

summary_funs <- list(mean = partial(mean, na.rm = TRUE),
                     sd = partial(sd, na.rm = TRUE),
                     min = partial(min, na.rm = TRUE),
                     p25 = partial(quantile, probs = .25, names = FALSE, na.rm = TRUE),
                     median = partial(median, na.rm = TRUE),
                     p75 = partial(quantile, probs = .75, names = FALSE, na.rm = TRUE),
                     max = partial(max, na.rm = TRUE))
{% endhighlight %}



{% highlight text %}
## Error in partial(mean, na.rm = TRUE): could not find function "partial"
{% endhighlight %}



{% highlight r %}
spam_data[-58] %>% map(~ invoke_map(summary_funs, x = .x)) %>%
    invoke(rbind.data.frame, .) %>%
    kable
{% endhighlight %}



{% highlight text %}
## Error in invoke(rbind.data.frame, .): could not find function "invoke"
{% endhighlight %}

The spam dataset is pretty sparse. As it turns out, this will cause a few problems for our Naive Bayes classifier. If we randomly split the dataset for cross validation, there is a nontrivial chance that one or more of our columns with be constant and have zero variance. You can't divide by zero, and we'll get an `NaN` returned.

I added a couple functions to check for zero variance columns and remove them from a data frame. This was added to the function `naive_bayes()`.


{% highlight r %}
# Zero variance predicate
zero_variance <- function(.col, .thresh) {
    var(.col) < .thresh
}

# Apply to data frame and remove constant columns
nzv <- function(.data, .target = "target", .thresh = 1e-4) {
    y <- .data[,.target, drop = FALSE]
    .data[.target] <- NULL
    id <- map_lgl(.data, Negate(zero_variance), .thresh)
    cbind(.data[id], y)
}
{% endhighlight %}

Our target is close to balanced. This is very helpful in classification, and we shouldn't have to worry about a skewed target resulting from random sampling.


{% highlight r %}
spam_data %>% count(target) 
{% endhighlight %}



{% highlight text %}
## Error in count(., target): could not find function "count"
{% endhighlight %}

As a check for our implementation for each method, I'll pull out the same classification methods in the `caret` packaged. For all intents and purposes, this is the gold standard for predictive modeling in R. [It has a wonderul site as well](https://topepo.github.io/caret/), which is a thorough tutorial on predictive modeling as much as it is package documentation. If we compete well with `caret`, we've done a good job. `caret` has LDA, QDA and Naive Bayes. We'll use all three, along with the option of removing zero variance columns from our training data.


{% highlight r %}
# Implementing the Experiment ---------------------------------------------

# Store models to be tested
TC <- trainControl(method = 'none')
{% endhighlight %}



{% highlight text %}
## Error in trainControl(method = "none"): could not find function "trainControl"
{% endhighlight %}



{% highlight r %}
NB_pars <- data.frame(fL = 0, usekernel = FALSE)
proc <- c("nzv")

models <- list(my_lda = partial(LDA, target ~ .),
               caret_lda = partial(train, target ~ ., method = 'lda',
                                   trControl = TC),
               caret_qda = partial(train, target ~ ., method = 'qda',
                                   trControl = TC, preProcess = proc),
               my_nb = partial(naive_bayes, target ~ .),
               caret_nb = partial(train, target ~ ., method = 'nb',
                                  trControl = TC, tuneGrid = NB_pars,
                                  preProcess = proc))
{% endhighlight %}



{% highlight text %}
## Error in partial(LDA, target ~ .): could not find function "partial"
{% endhighlight %}

Since we'll be calling all of these models against the same datasets, it's easiest to just store these as a list and `invoke` them. We'll add another function called `score_model()` with methods for our methods and `caret`'s default `train` class. This will standardize all of our outputs and make them easier to compare.

We need a simple helper function to split the dataset.


{% highlight r %}
split_data <- function(.data, .test_rate = 0.1) {
    n <- nrow(.data)
    svec <- sample(c("train", "test"), n, replace = TRUE, prob = c(1 - .test_rate, .test_rate))
    split(.data, svec)
}
{% endhighlight %}

And the function `rerun` in `purrr` will generate all 100 splits for us.


{% highlight r %}
# Create list of random training and test dataframes

dfs <- rerun(100, split_data(spam_data))
{% endhighlight %}



{% highlight text %}
## Error in rerun(100, split_data(spam_data)): could not find function "rerun"
{% endhighlight %}



The actual test is a relatively simple pipeline. We take our list of data frames. We call each of the models with the combination of `map` and `invoke_map`. The results in a list of lists, where each item in the top level of the list contains a list of five fit models. With that, we map again. We takes the test data out of `dfs` and use the models in the list of lists to get our scores.

The standardized outputs of `score_model`, which has methods for all of the classes of models involved, makes sure that everything is eventually consistent. It's the grease that makes the whole machine run.


{% highlight r %}
results <- dfs %>%
    map( ~ invoke_map(models, data = .x$train)) %>%
    map2(dfs, ~ map(.x, score_model, newdata = .y$test,
                    actual = .y$test$target))
{% endhighlight %}

So, what did we get? Right now, we have a pretty messy nested list. To clean this up, we need to make some data frames and bind them together. After that, we can put together a little plot. Since we're comparing results across distinct categories (the models), a boxplot is best.


{% highlight r %}
# Lets make a nice plot

results %>% at_depth(2, data.frame) %>%
    at_depth(1, bind_rows, .id = "model") %>%
    bind_rows(.id = "iteration") %>%
    ggplot(aes(model, f1)) +
    geom_boxplot() +
    ggtitle("Comparison of F1 in classification models")
{% endhighlight %}



{% highlight text %}
## Error in ggplot(., aes(model, f1)): could not find function "ggplot"
{% endhighlight %}

Our version of LDA performed exactly the same as `caret`'s. YAY! Either one is the best, but I'm pretty sure you know which one I love more. 

Our version of Naive Bayes lagged a little, but it's way ahead of `caret`'s QDA and Naive Bayes. Why's that? It took awhile to diagnose, but it turns out that `caret`'s default correction for zero variance is pretty aggressive. So much so that it adversely affects the model's performance. We have much tighter standards for zero variance columns, and end up rejecting a much smaller number of them.

Moreover, the version of Naive Bayes used by `caret` comes from the package `klaR`, which does not use log odds in its classification rule. There's probably a good reason for this, but the result is something that doesn't do that well with very sparse data frames. We get a long stream of warnings about probabilities close to 0 or 1 is the odds. It ultimately has to round, which introduces a host of problems. 

We could expand on this test by throwing in some other methods, but that's probably enough for today. Please, take the time to check out the implementation of each method in the blog's package. It's surprisingly simple and straightforward. And please don't hesitate to let me know your thoughts.

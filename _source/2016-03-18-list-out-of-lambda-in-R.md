---
layout: post
title: "An R translation of List out of Lambda"
excerpt: "We'll use a modern classic in literate programming to illustrate some of the more interesting sides of functional programming in R."
tags: [R, Functional Programming, Lists, Thought Experiments]
comments: true
modified: 2016-03-19
use_math: true
image: 
    feature: harlem.jpg
    credit: Olds, Elizabeth, via the NY Public Library
    creditlink: https://digitalcollections.nypl.org/items/913dd5f0-d56d-0131-4387-58d385a7bbd0
---

{% include _toc.html %}

## Introduction

We're going to do something a little different today. Instead of [explaining classic modeling methods](https://michaelquinn32.github.io/analyzing-spam-data/), [exploring some research](https://michaelquinn32.github.io/bootstrap/) or [solving a problem](https://michaelquinn32.github.io/geysers/), I'm going to translate an article from JavaScript into R. "That's it," you say? Well, well, well, dear reader have some faith in your translator. As anyone familiar with [Pevear and Volokhonsky's incredible translations of Russian literature knows](https://www.theparisreview.org/interviews/6385/the-art-of-translation-no-4-richard-pevear-and-larissa-volokhonsky), the work of a translator is much, much more than just a verbatim transcription. The translator brings a foreign world alive in a new and exciting context, adding as much of his or her authoritative voice as the original.

Now *that's* what we're going to do. Today's article is Steve Losh's [List Out of Lambda](https://stevelosh.com/blog/2013/03/list-out-of-lambda/), and I would dare to say that it is a modern classic in literate programming. I was lucky enough to stumble across it while working through [Hadley Wickham's Advanced R](https://adv-r.had.co.nz/).[^1] If you haven't read that article yet, boy you're in for a treat. Please, give it a shot before reading any further, and I'll do my best to show you how R can interpret Steve's ideas.

[^1]: And thanks Hadley for all of the amazing functional programming ideas in [`purrr`](https://github.com/hadley/purrr)

Once you're done with that. Please take a peak at [`lambdaList`, the package that accompanies this post](https://github.com/michaelquinn32/lambdaList). Even better, install it so that you can code along with the article.

## A tale of four functions

Losh's article is an attempt at a challenge: how much of a modern programming language do you actually need to program? We're going to build off of that by taking a slightly different approach: what, exactly, is a list? If you've spent a little time in the R pool, I'm sure that you could define it by rote: *a list is a recursive vector*. And you wouldn't be wrong. 

But that's just one implementation of a list. Why don't we try a different definition? Say, one that only uses functions. If we tie our hands that way, then we get at a much more interesting idea. To quote Losh,

> A "list" is "a function that knows how to return its own head or tail when asked."

OK, that probably doesn't make a whole lot of sense. And it probably not a complete definition either, since a list is a function that can also tell you if its empty. A better illustration of these ideas comes from the following four functions (plus one definition).


{% highlight r %}
empty_list <- function(selector) {
    selector(NULL, NULL, TRUE)
}

empty_list <- structure(empty_list, class = c('fl', 'function'))

head.fl <- function(x, ...) {
    x(function(h, t, e) h)
}

tail.fl <- function(x, ...) {
    x(function(h, t, e) t)
}

is_empty <- function(ls) {
    ls(function(h, t, e) e)
}
{% endhighlight %}

Take a second to mull these over. And all you pedants over there, cool it with the "you're abusing the class system nonsense." I know I'm abusing the class system, but if we're going to create "objects" out of functions and preserve even a modicum of beauty in our code, we're going to need to take some short cuts. If we can all take it easy for just one second, I promise that everyone will have a lot of fun.

For those still wondering what all the `.fl` stuff is, sorry. That last comment was probably confusing. [You might want to check out Hadley Wickham's explanation of S3](https://adv-r.had.co.nz/OO-essentials.html#s3). A lot of the functions that we are creating overlap with functions in base R and the standard packages. In some cases, there will be a generic function already defined. This allows us to call `tail.fl` as `tail` on our functional lists and get the reaction that we want. The other cases are handled by the [`lambdaList` package](https://github.com/michaelquinn32/lambdaList).

Alright, let's get back on track. What have we done? Believe it or not, we've got almost everything we need (save one function) for Losh's implementation of a list: "a function that returns its head or tail." While R doesn't use inheritance like C++, Java or Python, it's still useful to think of the preceding functions as a "base class" that we'll build off of later.

1. `empty_list` returns a closure that takes arguments for its head, tail and empty status. Keep these arguments in your head going forward, even though they aren't ever named. Right now, only the latter has a non-`NULL` value. That will change.
2. `head` uses an anonymous function to call the head argument from `empty_list` or any other functional list object that we build with it.
3. `tail` does the same, but with a different argument.
4. The same can be said for `is_empty`.

All but the final function take advantage of the S3 class system in one form or another. `empty_list` is a function with a class (yes, R actually lets you do this), which means that `head.fl` and `tail.fl` are just methods for that class. This saves us the hassle of overwriting any base functions, and even better, it gives us much nicer code once we start writing scripts.

It's extremely helpful to actually work through some of these calls step by step. For example, when we call `head` on `empty_list`,


{% highlight r %}
# 1. Calling head against empty_list
head(empty_list)

# 2. Since head is generic, it dispatches head.fl
head.fl(empty_list)

# 3. empty_list replaces the returned closure
head.fl(empty_list) {
    empty_list(function(h, t, e) h)
}

# 4. The anonymous function replaces selector and returns NULL
empty_list(function(h, t, e) h) {
    function(h = NULL, t = NULL, e = TRUE) h
}
{% endhighlight %}

It might help to write this down as well. In fact, it wouldn't hurt to do a similar exercise with `tail.fl` and `is_empty`. In the end, these four functions show us some of the incredible power of anonymous functions, closures and lazy evaluation.

* These functions were written to call other functions that weren't defined yet
* We can supply arguments to functions with other functions
* We don't need to bind these functions either and can swap them out as needed

I understand that it doesn't see like we've accomplished much yet. "What good is an empty list any way?" I hear ya. Have patience. The cool stuff starts in the next section.

## Where's the data?

These principals come together in our list constructor functions. The package accompanying this post has three "builders", but the first has more back-end functionality while the other two are more user-facing. Either way, all three will use `empty_list` as a base function to mark the final value of the list, nesting values in additional functional calls. What in the world does that mean? Well, let's start by having a look at `prepend`.


{% highlight r %}
prepend <- function(ls, el) {
    structure(function(selector) selector(el, ls, FALSE), class = c('fl', 'function'))
}
{% endhighlight %}

While the call to `structure` makes everything a little cluttered, the heart of `prepend` is the same style closure that we saw in `empty_list`. But now, the anonymous function replaces the closure's arguments with the prepended element, the list and `FALSE` (a prepended list is no longer empty!).

In practice, this is what that looks like.


{% highlight r %}
library(lambdaList)
{% endhighlight %}



{% highlight text %}
## Error in library(lambdaList): there is no package called 'lambdaList'
{% endhighlight %}



{% highlight r %}
prepend(prepend(empty_list, 2), 1)
{% endhighlight %}



{% highlight text %}
## Error in prepend(prepend(empty_list, 2), 1): could not find function "prepend"
{% endhighlight %}

`empty_list` serves an important purpose in this whole process. On the one hand, it begins the chain of `prepend` by serving as the first list argument. On the other, it marks the "end" of the list, being the first list passed to our chain of function calls. This is key for all of the methods we want to call on our object, like `print.fl`.


{% highlight r %}
print.fl <- function(x, ...) {
    if (is_empty(x)) {
        cat("")

    } else {
        cat(head(x), "")
        print(tail(x))
    }
}
{% endhighlight %}

The preceding function unwinds and prints the functional list `x` recursively. First, it checks if we're at the end. If not, then we grab the first element in the list (its head), print it and then call print again with the tail. This recursive pattern will come up again and again in our methods. And in almost all cases, checking for the `empty_list` provides the baseline case.

In case you haven't noticed, `prepend` isn't all that user friendly, considering how it builds list in reverse order. This brings us to our other list building functions: `append` and `concat`. The former builds lists in the order that a user might expect, while the latter lets us combine lists, like `c` in base R. Surprise, surprise are also recursive. 

Here's `append`.


{% highlight r %}
append.fl <- function(ls, el) {
    if (is_empty(ls)) prepend(ls, el)
    else prepend(append(tail(ls), el), head(ls))
}
{% endhighlight %}

Even though append constructs lists in the opposite direction of `prepend`, we're still using our base constructor. Although it is more difficult to reason over, this construction ensures that `empty_list` still marks then end. Remember, this is the key to all of the methods we're writing.

Let's work through it:

* In the base case, we are checking if the functional list is empty. If so, we start building.
* Otherwise, we "work backwards," prepending the oldest elements to our list and working to the newest

Although recursion is costly, we've bought ourselves a much cleaner syntax for generating lists. Even better, it plays nicely with `magrittr` and looks fully at home in modern R. Notice that our lists are pretty agnostic to the type of data that we put in. As long as it can be used in the argument of a function,[^2] our list can store it.

[^2]: We face other implementation methods, like printing and working with more esoteric data types. This isn't exactly an issue with the implementation of functional lists, so we can skip this issue for now. Just know, for now, that if you throw a data frame in these lists, nothing will print out. Maybe I'll fix that later.


{% highlight r %}
library(magrittr)

my_stuff <- empty_list %>%
    append(21) %>%
    append('a') %>%
    append(8 + 3i)
{% endhighlight %}



{% highlight text %}
## Error: object 'empty_list' not found
{% endhighlight %}

From all appearances, we have a data object. We don't. This chain above is still just a function.


{% highlight r %}
is.function(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}

But we can work with the data in the arguments, just like typical data objects in R.


{% highlight r %}
print(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}

And our accessor functions give all of the same functionality as object properties. They work just like methods should.


{% highlight r %}
# Using accessor functions: head
head(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}



{% highlight r %}
# Using accessor functions: empty
is_empty(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error in is_empty(my_stuff): could not find function "is_empty"
{% endhighlight %}



{% highlight r %}
# Using accessor functions: tail
tail(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}

But again, the "object" they return, when it's not a scalar, is still just a function.


{% highlight r %}
# The tail is still a function!
tail(my_stuff) %>% is.function
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}

Wild, right?

The same principals give us an even more general constructor function.


{% highlight r %}
concat <- function(lsa, lsb) {
    if (is_empty(lsa)) lsb
    else prepend(concat(tail(lsa), lsb), head(lsa))
}
{% endhighlight %}

The recursion in `concat` works almost exactly the same as the recursion in `append`:

* We check to see if we're at the end of the first list. If so, we can safely return the second list.
* To this, we recursively attach the tail elements of the first list until we've exhausted them all.

Here's what we can do with it.


{% highlight r %}
x <- 2

other_stuff <- empty_list %>%
    append(x) %>%
    append(seq_len(3)) 
{% endhighlight %}



{% highlight text %}
## Error: object 'empty_list' not found
{% endhighlight %}



{% highlight r %}
concat(my_stuff, other_stuff)
{% endhighlight %}



{% highlight text %}
## Error in concat(my_stuff, other_stuff): could not find function "concat"
{% endhighlight %}

The combination of a generic functional list and relatively bare printing method can lead to some confusion. The final "element" in the list is the sequence. The list treats this as a single element of the list, even though it's hard to tell by just printing.

## What else do we need?

We have a variety of tools to build lists, but we can't do much with them yet. It's time to change that. R's lists have a length property. Our functional lists should have this property too.


{% highlight r %}
length.fl <- function(x) {
    if (is_empty(x)) 0
    else length(tail(x)) + 1
}
{% endhighlight %}

We're now counting recursively. Empty lists have no length. They're our base case. Otherwise, we call the function again on the tail of the list and add one. This strips off one element from the list, and we add one element to our count. And just like that, we have the list's length.

Here's the function in practice.


{% highlight r %}
length(empty_list)
{% endhighlight %}



{% highlight text %}
## Error: object 'empty_list' not found
{% endhighlight %}



{% highlight r %}
length(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}

Lists in R also has some simple indexing functions. For better or worse, these functions use ordinal numbers, counting from one, while most other programming languages count from zero. For the sake of variety, why not implement the latter? It's surprisingly similar to finding the length of a list.


{% highlight r %}
nth <- function(ls, n) {
    if (n <= 0) head(ls)
    else nth(tail(ls), n - 1)
}
{% endhighlight %}

In stead of counting up, we count down with each recursive call. Just as before, we strip off an element of the list with each step. The first element of remaining list is the element we're looking for. Again, for R users, this implementation has us indexing from 0.[^3]

[^3]: Indexing from one would just require changing the base case in the recursion.

Getting a range of elements is a little more complicated. We'll build it up from two simpler functions. The first, `take`, returns all of the elements before an index. The second, `drop`, function returns all of the elements after the index, including the index itself. This gives a function that indexes just like lists in Python, for those familiar with that language.


{% highlight r %}
take <- function(ls, n) {
    if (n <= 0) empty_list
    else prepend(take(tail(ls), n - 1), head(ls))
}

drop.fl <- function(ls, n) {
    if (n <= 0) ls
    else drop(tail(ls), n - 1)
}

slice <- function(ls, start, end) {
    take(drop(ls, start), end - start)
}
{% endhighlight %}

In all three cases, we use the arguments to count and mark our location in the list. In the case of `take` and `drop`, we're counting down from the index. Both have the same recursive base case, to check where the index has been reduced to zero. For all intents and purposes, `slice` is just a combination of those two functions.

Here are all of the accessor of the functions in practice. All are pretty straightforward.


{% highlight r %}
print(my_stuff)
{% endhighlight %}



{% highlight text %}
## Error: object 'my_stuff' not found
{% endhighlight %}



{% highlight r %}
nth(my_stuff, 2)
{% endhighlight %}



{% highlight text %}
## Error in nth(my_stuff, 2): could not find function "nth"
{% endhighlight %}



{% highlight r %}
take(my_stuff, 2)
{% endhighlight %}



{% highlight text %}
## Error in take(my_stuff, 2): could not find function "take"
{% endhighlight %}



{% highlight r %}
drop(my_stuff, 2)
{% endhighlight %}



{% highlight text %}
## Error in drop(my_stuff, 2): unused argument (2)
{% endhighlight %}



{% highlight r %}
slice(my_stuff, 1, 3)
{% endhighlight %}



{% highlight text %}
## Error in slice(my_stuff, 1, 3): could not find function "slice"
{% endhighlight %}

## Generating lists

We can construct lists element-wise, and we can access elements of those lists. These are all very useful, but we might want some more high-powered options for quickly generating functional lists. This is especially true if we want to test the limits of our objects.

First, we'll implement sequences. They seem like a logical addition in this area.


{% highlight r %}
seq.fl <- function(empty, start, end) {
    if (end <= start) empty
    else prepend(seq(empty, start + 1, end), start)
}
{% endhighlight %}

Look! We have more recursive, and even more counting with arguments. We check to see if start has reached the value of end. If not, we add start to the list and move it up one. In order to take advantage of S3 methods dispatch and avoid overwriting default R behavior, we add an `empty` argument to the beginning of `seq`. That way, it knows that it's building a functional list.

R uses a lot of sequences, but one of its most powerful tools for generating lists comes from the function `replicate`. It takes advantage of R's ability to store and evaluate expressions, i.e. literal bits of code. With `replicate`, you are simply asking R to take this bit of code, run it a bunch of times and give it back as a list.[^4]

[^4]: At least that's how it's supposed to work when you're doing functional programming. You need to set `simplify = FALSE` in the arguments to get back a list.

I don't want to get to deep into R's non-standard evaluation and meta-computing tools. [Hadley Wickham explains these well](https://adv-r.had.co.nz/Computing-on-the-language.html). Instead, let's just implement a version of `rerun` where we require the user to specify that they have an expression in advance.


{% highlight r %}
rerun <- function(n, expr) {
    if (n <= 0) empty_list
    else prepend(rerun(n - 1, expr), eval(expr))
}
{% endhighlight %}

As before, we count with the arguments. Now, at each recursive step, we evaluate our expression. Again, remember, the user needs to `quote` the expression so R knows what it's supposed to do with it.

Here they are in practice.


{% highlight r %}
seq(empty_list, 1, 11)
{% endhighlight %}



{% highlight text %}
## Error: object 'empty_list' not found
{% endhighlight %}



{% highlight r %}
rerun(5, quote(rnorm(1)))
{% endhighlight %}



{% highlight text %}
## Error in rerun(5, quote(rnorm(1))): could not find function "rerun"
{% endhighlight %}

Now that it's much easier to generate functional lists, we can move to the grand finale: functional programming.

## What else can functions do?

The field of functional programming is dominated by three families of functions:

* **Maps**: apply a function to every element in the list
* **Filters**: take a predicate function that returns true or false and returns the elements of the list that fulfill that condition (or don't)
* **Reductions**: take a list and recursively fold it down to a scalar

Don't believe me? [Check out R's help file on common higher-order functions](https://stat.ethz.ch/R-manual/R-devel/library/base/html/funprog.html). I promise you'll see lots of filters, maps and reductions. For me, this is the most exciting part of the entire project, so I plan on showing a few options. 

### Maps

The most basic higher-order function is the map, which takes a function and applies it to every element in the list. For a single-list map, we work with a unary function: one that takes a single argument and returns a single value.

This is a unary function.


{% highlight r %}
plus_5 <- function(x) x + 5
{% endhighlight %}

In a functional paradigm, iterators like `map` replace `for` loops. While it wouldn't be too difficult to implement a `for` loop with the functional list that we've created, the syntax wouldn't be nearly as terse and expressive. We'd have to create some sort of intermediate variable (like a running total in a sum), among other things. This is one of the primary reasons that `lapply` is preferred in R. 

Plus, mapping is very easy to do recursively.


{% highlight r %}
map <- function(ls, fn, ...) {
    if (is_empty(ls)) empty_list
    else prepend(map(tail(ls), fn, ...), fn(head(ls), ...))
}
{% endhighlight %}

As always, since we are building lists, our base case returns an empty list. Otherwise, we evaluate the function on the first element of the list and prepend it a new list. Most of the other details above give us the ability to add more arguments to our function.

Unary functions are great, but they're not everything. Why not work with binary functions? In that case, we now have something to combine two functional lists together. Although it is a little verbose, a binary mapper isn't all that different from the unary version above.


{% highlight r %}
map2 <- function(lsa, lsb, fn, ...) {
    if (is_empty(lsa)) empty_list
    else prepend(map2(tail(lsa), tail(lsb), fn, ...), fn(head(lsa), head(lsb), ...))
}
{% endhighlight %}

The recursive process is essentially the same as before. The only difference is the need to track both functional lists at the same time.

We can do a whole lot with these two functions. Here are some examples.


{% highlight r %}
# Let's make some integers
some_numbers <- rerun(20, quote(sample.int(10, 1)))
{% endhighlight %}



{% highlight text %}
## Error in rerun(20, quote(sample.int(10, 1))): could not find function "rerun"
{% endhighlight %}



{% highlight r %}
some_numbers
{% endhighlight %}



{% highlight text %}
## Error: object 'some_numbers' not found
{% endhighlight %}



{% highlight r %}
# A simple unary function
square <- function(x)  x * x

map(some_numbers, square)
{% endhighlight %}



{% highlight text %}
## Error in map(some_numbers, square): could not find function "map"
{% endhighlight %}



{% highlight r %}
map2(some_numbers, some_numbers, `+`)
{% endhighlight %}



{% highlight text %}
## Error in map2(some_numbers, some_numbers, `+`): could not find function "map2"
{% endhighlight %}

The most important aspect of maps is one we've repeatedly touched on already. We are using first-class functions, i.e. functions that are accepting other functions as arguments. The variety of options is only limited by the variety of functions that we want to pass to our mapper. 

### Filters

Filters are a lot like maps in that they apply a particular function to every element in our functional list. But unlike maps, they do not return every element. As I previously mentioned, filters return the elements of a list that fulfill a predicate function. What's a predicate function? That's one that tests an element and returns either `TRUE` or `FALSE`. 

For example, this is a predicate function.


{% highlight r %}
greater_than_10 <- function(x) x > 10
{% endhighlight %}

Today, we'll look at four of filtering functions. First, here's `filter` and `remove`. Both rely on a predicate function (obviously), but `filter` keeps the `TRUE` cases, while `remove` keeps the `FALSE` cases.

Here they are.


{% highlight r %}
filter.fl <- function(ls, fn, ...) {
    if (is_empty(ls)) empty_list
    else if (fn(head(ls), ...)) prepend(filter(tail(ls), fn, ...), head(ls))
    else filter(tail(ls), fn, ...)
}

remove.fl <- function(ls, fn, ...) {
    if (is_empty(ls)) empty_list
    else if (fn(head(ls), ...)) remove(tail(ls), fn)
    else prepend(remove(tail(ls), fn, ...), head(ls))
}
{% endhighlight %}

Both functions build lists, which means their base recursion tests for and returns empty lists. In the case of `filter`, we proceed by iteratively checking the function on the first element of the remaining list. The `TRUE` cases are sent to `prepend`, while the `FALSE` cases just call the function again without that element. `remove` isn't all that different, except it prepends with `FALSE` and calls the function when `TRUE`.

Here's what they look like.


{% highlight r %}
# Here's a predicate
is_even <- function(x) x %% 2 == 0

# Filtering and removing
filter(some_numbers, is_even)
{% endhighlight %}



{% highlight text %}
## Error: object 'some_numbers' not found
{% endhighlight %}



{% highlight r %}
remove(some_numbers, is_even)
{% endhighlight %}



{% highlight text %}
## Warning in remove(some_numbers, is_even): object 'some_numbers' not found
{% endhighlight %}

When filtering lists, we often only want the first object that fulfills the predicate or its position. For that, we implement `find` and `position`. This functions are quite similar to our first filters, but they have a few more cases to check.


{% highlight r %}
find.fl <- function(ls, fn, ...) {
    if (is_empty(ls)) FALSE
    else if (fn(head(ls), ...)) head(ls)
    else find(tail(ls), fn)
}

position <- function(ls, fn, start = 0) {
    if (start >= length(ls)) FALSE
    else if (fn(nth(ls, start))) start
    else position(ls, fn, start + 1)
}
{% endhighlight %}

For both functions, we want to return `FALSE` if we didn't get a match. Otherwise, we next test to see if the predicate is true. When it is, `find` returns the object and `position` gives us the index. In the case of the latter, we need an extra argument `start` to keep track of our position in the list. This gives the added functionality of being able to start our search at any point we want in the list.

Here's a call of each function against the previously generated numbers. We'll stick with the same predicate.


{% highlight r %}
find(some_numbers, is_even)
{% endhighlight %}



{% highlight text %}
## Error: object 'some_numbers' not found
{% endhighlight %}



{% highlight r %}
position(some_numbers, is_even)
{% endhighlight %}



{% highlight text %}
## Error: object 'some_numbers' not found
{% endhighlight %}

### Reductions

We wrap up our little tour of functional programming by looking at reductions. Generally speaking, this class of functions takes a binary function and *folds* it along a list, eventually resulting in a single scalar. What does that mean? Consider the following case, reducing a sequence of numbers with addition.


{% highlight r %}
reduce([0, 1, 2, 3], +) => 
    (((0 + 1) + 2) + 3) =>
    ((1 + 2) + 3) =>
    (3 + 3) =>
    6
{% endhighlight %}

This is another obvious candidate for recursion, and a good implementation keeps track of the intermediate calculations through a value in the arguments. Fortunately, most reductions allow for initialization with a specific value, and we can use that. With that, we can implement two different reductions, one for each "direction" of the list.


{% highlight r %}
reduce <- function(ls, fn, init = 0, ...) {
    if (is_empty(ls)) init
    else reduce(tail(ls), fn, fn(init, head(ls)))
}

reduce_right <- function(ls, fn, init = 0, ...) {
    if (is_empty(ls)) init
    else reduce_right(tail(ls), fn, fn(head(ls), init), ...)
}
{% endhighlight %}

Keeping track of the computations is an especially nice property, making it relatively easy to reason about the computation. On the first recursive step, we apply the function on `init` and the first value of the functional list. This becomes the new init, and we call `reduce` again on all but the first value of the list. This continues until the list is empty, causing the base case to be true. Now, we return the final computation.

Combining `reduce` and `rerun` leads to  some nice computational power. For example,


{% highlight r %}
mean.fl <- function(x)  reduce(x, `+`) / length(x)
rerun(500, quote(rnorm(1))) %>% mean
{% endhighlight %}



{% highlight text %}
## Error in rerun(500, quote(rnorm(1))): could not find function "rerun"
{% endhighlight %}

`reduce_right` is a bit more complicated to use, as it leads to many counterintuitive results. Don't believe me? Try running a right reduction with subtraction. It's usually helpful to work out the problem on paper before going that route.

## Some limitations

Over the course of the article, we've created a data structure using only functions, extended it with many other methods and demonstrated a whole suite a functional programming tools. I hope you found this exercise illuminating and surprising. Few things have changed my approach to programming the way that *List out of Lambda* has, and I hope seeing an implementation in R makes you more excited for functional programming.

That said, this approach to programming does not play nice with the current R interpreter. In particular, R limits the number of recursive calls you can execute in a single call. This is meant to protect you from a runaway recursive call, but it ends up tying your hands as well. This threshold is pretty low for a lot of modern problems. As you can see, you can only build vectors that are less than 1500 calls long using rerun.


{% highlight r %}
test <- rerun(1500, quote(rnorm(1)))
{% endhighlight %}



{% highlight text %}
## Error in rerun(1500, quote(rnorm(1))): could not find function "rerun"
{% endhighlight %}

Even if you reset `options(expressions=)`, you'll quickly hit another hard cap: the size of the stack in C. For those that didn't know, each new function call creates its own environment (or frame in C) to contain all of the local variables and expressions within the function. For your own sake, C limits the number of calls you make at once. Within R, there's not much you can do about it.

Not surprisingly, creating a new frame on the stack with each recursive step is both memory intensive and somewhat slow. This is a particular feature of the C family of languages. Functional programming languages like `scala` and `haskell` are optimized for long recursive calls. [This is known as tail recursion optimization in certain languages](https://stackoverflow.com/questions/310974/what-is-tail-call-optimization).[^5] Despite all of R's functional programming strengths, this is not a feature in the current implementation of the language. Considering the fact that R is built in C, it might not ever be either.

[^5]: Haskell is slightly special in this regard, using a method called guarded recursion that uses a special kind lazy evaluation to avoid having to create more frames. [See more here](https://wiki.haskell.org/Tail_recursion).

A quick benchmark of the final functions shows how costly recursion can be.


{% highlight r %}
library(microbenchmark)
{% endhighlight %}



{% highlight text %}
## Error in library(microbenchmark): there is no package called 'microbenchmark'
{% endhighlight %}



{% highlight r %}
microbenchmark(
    'Functional' = mean(rerun(500, quote(rnorm(1)))),
    'Classic' = Reduce(`+`, replicate(500, rnorm(1), simplify = FALSE)) / 500,
    'Vectors' = mean(rnorm(500))
)
{% endhighlight %}



{% highlight text %}
## Error in microbenchmark(Functional = mean(rerun(500, quote(rnorm(1)))), : could not find function "microbenchmark"
{% endhighlight %}

The results above speak for themselves. R is already optimized to use vectors and vectorized implementations of most algorithms are much more efficient. 

But that doesn't mean that we can't do lots of interesting things at the edges of the language's design, and taking on these sorts of projects can illustrate all sorts of ideas about how computing in general works. Functions can be used to hold data. Iterators don't need loops. Smart uses of recursion lead to much cleaner and transparent programs. These are true even if the objects we create are slow. And they are lessons we should keep in mind any time we tackle a new project in R.

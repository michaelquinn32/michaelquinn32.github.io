#' This is a fun hashing experiment

# Prelim
library(magrittr)
library(dplyr)

# Separate a vector into equal-value bins, cut reallocates overflow
hash <- function(x, nbins) {
    seq_along(x) %>% cut(nbins, label = FALSE) %>% split(sort(x), .) 
}

# Check to see if a value is within the values given by an array
is_within <- function(vec, x) {
    (min(vec) <= x) && (x <= max(vec))
}

# Find list entry that contains a value x
lookup <- function(y, list) {
    vapply(list, is_within, x = y, logical(1)) %>% which(useNames = FALSE) 
}

# Combine the previous two functions
a_hash_lookup <- function(y, x, nbins) {
    if (!is.list(x)) x <- hash(x, nbins)
    y %>% lookup(x)
}

# Create a vectorized version
v_hash_lookup <- function(y, x, nbins) {
    if (!is.list(x)) x <- hash(x, nbins)
    y %>% lapply(a_hash_lookup, x = x, nbins) %>% setNames(y)
}

# Self lookup
self_lookup <- function(x, nbins, id = seq_along(x)) {
    if (length(id) == 1) a_hash_lookup(x[id], x, nbins) 
    else v_hash_lookup(x[id], x, nbins)
    
}

# Test Vector
x <- rnorm(100)

# Check hash output
hash(x, 10) %>% sapply(length)

# Test atomic version
self_lookup(x, nbins = 10, id = 2)
which(x[2] == sort(x))

# Test vector version
self_lookup(x, nbins = 10, id = 1:10) %>% unlist
1:10 %>% sapply(function(id) which(x[id] == sort(x)))

# Trim the results
self_lookup_t <- function(x, nbins, id = seq_along(x), trimFUN = head) {
    self_lookup(x, nbins, id) %>% lapply(trimFUN, 1) %>% unlist
}

# A binned example
iris[-5] %>% lapply(self_lookup_t, 10) %>% data.frame(iris, .) %>% head

# More practical
iris[-5] %>% lapply(cut, 10) %>% data.frame(iris, .) %>% head

# What is cut?
methods("cut")
getAnywhere("cut.default")

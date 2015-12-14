f1 <- function(u) {
    if (u <= .5 ) 1 - sqrt(1 - 2 * u)
    else 1 + sqrt(2 * u - 1)
}

summary_funs <- list(mean = mean,
                     variance = function(x) var(x) / length(x),
                     sd = function(x) sqrt(var(x) / length(x)))

set.seed(1312014)

runif(1000) %>% 
    map_dbl(f1) %>% 
    compose(list,list)(x = .) %>%
    invoke_map(summary_funs, .x = .)

pbinom(8, 10, .5, lower.tail = FALSE)
pbinom(80, 100, .5, lower.tail = FALSE)

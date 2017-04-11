rm(list=ls())

library(ggplot2)

roc <- function(prob, label) {
    label <- label[order(prob, decreasing = TRUE)]
    dat <- data.frame(n = seq(length(label)),
                      tp = cumsum(label),
                      tn = cumsum(1 - label))
    dat$fpr <- dat$tn / sum(1-label)
    dat$tpr <- dat$tp / sum(label)
    dat$label <- label
    dat$prob <- sort(prob, decreasing = TRUE)
    return(dat)
}

plot.roc <- function(dat, illustrate = FALSE) {
    return(ggplot(data = dat, aes(x = fpr, y = tpr)) + geom_line() + labs(x = 'False Positive Rate', y = 'True Positive Rate') + geom_abline(intercept = 0, slope = 1, linetype = 2))
}

plot.gamma <- function(prob, label) {
    dat <- gamma.func(prob, label)
    i <- which.min(abs(dat$gamma - 1))[1]
    return(ggplot(data = dat) + geom_line(aes(x = fpr, y = tpr), color = 1) + geom_line(aes(x = x, y = y), color = 'red', linetype = 2) + geom_abline(intercept = dat$y[i] - dat$gamma[i] * dat$x[i], slope = dat$gamma[i], color = 'blue') + labs(x = 'False Positive Rate', y = 'True Positive Rate') + geom_abline(intercept = 0, slope = 1, linetype = 2))
}

gamma.func <- function(prob, label) {
    dat <- roc(prob, label)
    fpr <- c(dat$fpr)
    tpr <- c(dat$tpr)
    smoothed <- lowess(fpr, tpr, f = 0.1)
    rr <- rank(smoothed$x)
    uniq.id <- c()
    uniq.rank <- c()
    for(i in seq_along(rr)) {
        if(!rr[i] %in% uniq.rank) {
            uniq.id <- c(uniq.id, i)
            uniq.rank <- c(uniq.rank, rr[i])
        }
    }
    g.score <- diff(c(0, smoothed$y[uniq.id])) / diff(c(0, smoothed$x[uniq.id]))
    names(g.score) <- uniq.rank
    dat$gamma <- g.score[as.character(rr)]
    for(i in seq_along(dat$gamma)) {
        dat$gamma[i] <- min(dat$gamma[seq(i)])
    }
    dat$gamma[dat$gamma == Inf] <- max(dat$gamma[dat$gamma != Inf])
    dat$x <- smoothed$x
    dat$y <- smoothed$y
    return(dat)
}

trans1 <- function(x) {
    ret <- x
    ret <- 0.3 + (ret - 0.3) / 10
    ret[x < 0.3] <- 0.3 - (0.3 - x[x < 0.3]) / 10
    return(ret)
}

trans2 <- function(x) {
    ret <- x
    ret <- 1 - (1 - ret) / 2
    ret[x < 0.7] <- x[x < 0.7] / 2
    return(ret)
}

set.seed(111)

n <- 1000
p <- rbeta(n, 0.8, 0.8)

p1 <- 1 / (1 + exp(- p / 10))
p2 <- 1 / (1 + exp(-5 * p))

p1 <- trans1(p)
p2 <- trans2(p)

l1 <- rbinom(n, prob = p, size = 1)
l2 <- rbinom(n, prob = p, size = 1)

dat1 <- gamma.func(p1, l1)
dat2 <- gamma.func(p2, l2)
roc.opt <- roc(c(dat1$gamma, dat2$gamma), c(dat1$label, dat2$label))

roc.naive <- roc(c(p1, p2), c(l1, l2))
roc.scaled <- roc(c(p1 / mean(p1), p2 / mean(p2)), c(l1, l2))

roc.opt$Method <- 'Optimal Ranking'
roc.naive$Method <- 'Raw Probability Score'
roc.scaled$Method <- 'Scale-adjusted'

plot.dat <- rbind(roc.opt, roc.naive, roc.scaled)

jpeg('~/Dropbox/public/blog/rocpool/roc_pool_1.jpg', width = 300, height = 300)
plot.roc(roc(p1, l1))
dev.off()

jpeg('~/Dropbox/public/blog/rocpool/roc_pool_2.jpg', width = 300, height = 300)
plot.roc(roc(p2, l2))
dev.off()

jpeg('~/Dropbox/public/blog/rocpool/roc_pool_compare.jpg', width = 300, height = 350)
print(ggplot(data = plot.dat, aes(x = fpr, y = tpr, color = Method)) + geom_line() + labs(x = 'False Positive Rate', y = 'True Positive Rate') + theme(legend.position = 'bottom') + geom_abline(intercept = 0, slope = 1, linetype = 2) + guides(color = guide_legend(nrow = 2)))
dev.off()

jpeg('~/Dropbox/public/blog/rocpool/roc_gamma.jpg', width = 300, height = 300)
plot.gamma(p1, l1)
dev.off()

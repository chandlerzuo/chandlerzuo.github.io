## weighted logistic regression
rm(list = ls())
library(ggplot2)
set.seed(1)
generateData <- function(n = 1000, p = 4, pi1 = 0.1) {
  X <- matrix(rnorm(n * p), nrow = n)
  X.test <- matrix(rnorm(n * p), nrow = n)
  b <- rep(c(5, -5), each = as.integer(p / 2))
  Z <- X %*% b
  Z.test <- X.test %*% b

  id1 <- seq(as.integer(pi1 * n))
  X[id1[Z[id1] < 0], ] <- -X[id1[Z[id1] < 0], ]
  id2 <- setdiff(seq(n), seq(as.integer(pi1 * n)))
  X[id2[Z[id2] > 0], ] <- -X[id2[Z[id2] > 0], ]

  id1 <- seq(as.integer(pi1 * n))
  X.test[id1[Z.test[id1] < 0], ] <- -X.test[id1[Z.test[id1] < 0], ]
  id2 <- setdiff(seq(n), seq(as.integer(pi1 * n)))
  X.test[id2[Z.test[id2] > 0], ] <- -X.test[id2[Z.test[id2] > 0], ]

  Z <- X %*% b
  Y <- rbinom(n, size = 1, prob = exp(Z) / (1 + exp(Z)))
  Z.test <- X.test %*% b
  Y.test <- rbinom(n, size = 1, prob = exp(Z.test) / (1 + exp(Z.test)))

  dat.fit <- data.frame(cbind(Y, X))
  dat.test <- data.frame(cbind(Y.test, X.test))
  names(dat.fit) <- names(dat.test) <- c("y", "v1", "v2", "v3")

  assign("dat.fit", dat.fit, envir = parent.frame())
  assign("dat.test", dat.test, envir = parent.frame())
}


myglm <- function(dat,  weight = c(1, 1), subsample = FALSE) {
  Y <- dat$y
  X <- dat[, 2:4]
  id1 <- which(Y == 1)
  id0 <- which(Y == 0)
  if(subsample) {
    if(length(id1) > length(id0)) {
      id1 <- sample(id1, length(id0))
    } else {
      id0 <- sample(id0, length(id1))
    }
  }
  Y <- Y[c(id0, id1)]
  X <- X[c(id0, id1), ]
  weights <- rep(weight, c(length(id0), length(id1)))
  dat.fit <- data.frame(cbind(Y, X))
  names(dat.fit) <- c("y", "v1", "v2", "v3")
  glm(y ~ ., data = dat.fit, weights = weights)
}

##summary(myglm(dat.fit))

plotROC <- function(dat, dat.test) {

  fit <- list(seq(3))
  fit[[1]] <- myglm(dat, subsample = TRUE)
  fit[[2]] <- myglm(dat, weight = c(mean(dat$y), 1 - mean(dat$y)))
  fit[[3]] <- myglm(dat)
  dat.roc <- NULL
  dat.point <- NULL
  methods <- c("Subsample", "Weighted", "Naive")
  for(i in seq(3)) {
    p1 <- predict(fit[[i]], newdata = dat.test[, -1], type = "link")
    ord <- order(1 - p1)
    y.true <- dat.test$y[ord]
    p1 <- p1[ord]
    names(p1) <- NULL
    fpr <- cumsum(1 - y.true) / sum(1 - y.true)
    power <- cumsum(y.true) / sum(y.true)
    dat1 <- data.frame(Method = methods[i],
                       Power = power,
                       fpr = fpr)
    dat2 <- data.frame(Method = methods[i],
                       Power = mean((p1 > 0.5)[y.true == 1]),
                       fpr = mean((p1 > 0.5)[y.true == 0])
                       )
    dat.roc <- rbind(dat.roc, dat1)
    dat.point <- rbind(dat.point, dat2)
    print(table(y.true, p1 > 0.5))
  }

  dat.point <<- dat.point

  ggplot(data = dat.roc, aes(x = fpr, y = Power, col = Method, linetype = Method)) + geom_path() + labs(x = "False Positive Rate", y = 'True Positive Rate') + theme(legend.position = "top") + geom_point(data = dat.point, aes(x = fpr, y = Power, color = Method), cex = 8, pch = 10)
}

for(n in c(10000)) {
  for(pi1 in c(0.05, 0.1, 0.5)) {
    generateData(n, 4, pi1)
    jpeg(file = paste("roc_", n, "_", pi1, ".jpg", sep = ""), width = 500, height = 500)
    print(plotROC(dat.fit, dat.test) + theme(axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), legend.text = element_text(size = 15), legend.title = element_text(size = 15), axis.title.x = element_text(size = 20), axis.title.y = element_text(size = 20)))
    dev.off()
  }
}

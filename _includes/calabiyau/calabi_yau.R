rm(list = ls())
library(plot3D)

triple <- function(z, x, y, n) {
    alpha <- pi / 3
    I <- complex(1, 0, 1)
    if(z == 0) {
        z1 <- exp(2 * pi * I * x)
        z2 <- 0
    } else {
        z1 = exp(2 * pi * I * x) * exp(log(cos(I * z)) * 2 / n)
        z2 = exp(2 * pi * I * y ) * exp(log(-I * sin(I * z)) * 2 / n)
    }
  return(c(Re(z2), cos(alpha) * Im(z1) + sin(alpha) * Im(z2), Re(z1), Arg(z1), Arg(z2)))
}

oneGrid <- function(x, y, m = 20, n) {
    M <- mesh(seq(-1, 1, length.out = m),
              seq(0, pi / 2, length.out = m))

    dat <- apply(cbind(c(M$x), c(M$y)),
                 1, function(w)
                     triple(complex(1, w[1], w[2]), x, y, n))
    x.mesh <- matrix(dat[1, ], nrow = m)
    y.mesh <- matrix(dat[2, ], nrow = m)
    z.mesh <- matrix(dat[3, ], nrow = m)
    w.mesh <- matrix(dat[4, ], nrow = m)
    v.mesh <- matrix(dat[5, ], nrow = m)
    return(list(x = x.mesh, y = y.mesh, z = z.mesh, w = w.mesh, v = v.mesh))
}

myColorRamp <- function(v1, v2) {
    v1 <- v1 + pi
    v2 <- v2 + pi
    x <- colorRamp(c("black", "red"))(v1 / 2 / pi) + 
        colorRamp(c("black", "green"))(v2 / 2 / pi)
    rgb(x[,1], x[,2], x[,3], maxColorValue = 255)
}

plotCalabiYau <- function(n, phi = 0, theta = 0) {
    for(k in seq(n)) {
        for(l in seq(n)) {
            grid.dat <- oneGrid(k / n, (l + 0.5) / n, n = n)
            scatter3D(x = c(grid.dat$x), y = c(grid.dat$y), z = c(grid.dat$z), col = NULL, cex = 0, surf = list(x = grid.dat$x, y = grid.dat$y, z = grid.dat$z, col = myColorRamp(grid.dat$w, grid.dat$v), alpha = 1), xlim = c(-1.8, 1.8), ylim = c(-1.8, 1.8), zlim = c(-1.8, 1.8), add = !(k == 1 & l == 1), phi = phi, theta = theta, alpha = 1, colkey = FALSE, axes = FALSE, bty = "n")
        }
    }
}

jpeg("calabi_yau_5.jpg")
plotCalabiYau(5)
dev.off()
jpeg("calabi_yau_5_side1.jpg")
plotCalabiYau(5, 60, 60)
dev.off()
jpeg("calabi_yau_5_side2.jpg")
plotCalabiYau(5, -60, 60)
dev.off()
jpeg("calabi_yau_5_side3.jpg")
plotCalabiYau(5, -60, -60)
dev.off()

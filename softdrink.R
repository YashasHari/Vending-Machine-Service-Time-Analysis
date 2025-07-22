softdrink_path <- "softdrink.txt"
if (!file.exists(softdrink_path)) {
  cat("softdrink.txt not found. Please select manually.\n")
  softdrink_path <- file.choose()
}
data <- read.table(softdrink_path, header = TRUE)

# Fit linear regression model
model <- lm(Y ~ x1 + x2, data = data)
summary(model)

# Basic plots
par(mfrow = c(2, 2))
plot(data$x1, type = "b", main = "x1 vs Observation Index", xlab = "Index", ylab = "x1")
plot(data$x2, type = "b", main = "x2 vs Observation Index", xlab = "Index", ylab = "x2")
plot(resid(model), type = "b", main = "Residuals vs Observation Index", xlab = "Index", ylab = "Residuals")
plot(fitted(model), resid(model), main = "Residuals vs Fitted Values", xlab = "Fitted", ylab = "Residuals")

# Hat matrix and leverage
X <- model.matrix(model)
H <- X %*% solve(t(X) %*% X) %*% t(X)
hii <- diag(H)
p <- ncol(X)
n <- nrow(X)
thresh <- 2 * p / n
high_leverage <- which(hii > thresh)
cat("High leverage points:", high_leverage, "\n")

# Scaled and internally studentized residuals
sigma_hat <- summary(model)$sigma
ei <- resid(model)
ei_scaled <- ei / sigma_hat
ri <- ei / (sigma_hat * sqrt(1 - hii))
outlier_scaled <- which(abs(ei_scaled) > 2)
cat("Scaled residuals outside ±2:", outlier_scaled, "\n")

# Ratio of internally studentized to scaled
plot(ri / ei_scaled, type = "b", main = "r_i / e_i_scaled", xlab = "Index", ylab = "Ratio")

# Externally studentized residuals
t_i <- rstudent(model)
ratio_t_r <- t_i / ri
plot(ratio_t_r, type = "b", main = "t_i / r_i", xlab = "Index", ylab = "Ratio")

# EXACT PRESS residuals and test statistic (Exercise 2 compliant)
X_full <- model.matrix(~ x1 + x2, data = data)
Y <- data$Y
exact_ti <- numeric(n)

for (i in 1:n) {
  data_minus_i <- data[-i, ]
  X_minus_i <- model.matrix(~ x1 + x2, data = data_minus_i)
  Y_minus_i <- data_minus_i$Y
  model_minus_i <- lm(Y_minus_i ~ x1 + x2, data = data_minus_i)
  beta_hat_minus_i <- coef(model_minus_i)
  x_i <- X_full[i, , drop = FALSE]
  Y_tilde_i <- as.numeric(x_i %*% beta_hat_minus_i)
  num <- Y[i] - Y_tilde_i
  sigma_hat_sq_i <- summary(model_minus_i)$sigma^2
  XtX_inv_minus_i <- solve(t(X_minus_i) %*% X_minus_i)
  leverage_term <- as.numeric(x_i %*% XtX_inv_minus_i %*% t(x_i))
  denom <- sqrt(sigma_hat_sq_i * (1 + leverage_term))
  exact_ti[i] <- num / denom
}

t_crit <- qt(0.995, df = n - p - 1)
outliers_exact <- which(abs(exact_ti) > t_crit)

cat("Exact t_i values:\n")
print(round(exact_ti, 3))
cat("\nOutliers at 99% level:", outliers_exact, "\n")

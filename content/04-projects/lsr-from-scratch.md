---
title: "LSR from Scratch"
date: "2026-02-05"
type: "essay"
status: "Experimental"
tags: ["Statistics", "Data Science", "Python", "Math"]
summary: "An LSR implentation from scratch, using no libraries, only pure statistics."
reading_time: "10 min"
tech_stack: ["Python", "Statistics"]
github_link: "https://github.com/matjsz/lsr"
math: true
---

# Introduction

This project trains an LSR statistical model based on a given dataset made out of data points on a 1D array. No library is used other than matplotlib for the graph plots.

## What is LSR?

A Least Squares Regression model is capable of optimizing a linear regression on a given dataset by minimizing its residuals.

### What is a residual?

A residual is basically the distance between a real data point and a predicted y via a linear model. Basically, in ML language, it's the loss function value.

### How do we calculate residuals?

It's simple, basically $(e=y-{\^y})$, which in turn, gives the distance between the predicted value (${\^y}$) and the real value ($y$).

## Finding the optimal regression line

First, we find the mean:

$$
\frac{\sum{x_i}}{n}
$$

```python
def _calculate_mean(self):
    x_sum = 0
    x_n = len(self.xs)
    for k in self.xs:
        x_sum += k
    x_mean = x_sum / x_n

    y_sum = 0
    y_n = len(self.ys)
    for j in self.ys:
        y_sum += j
    y_mean = y_sum / y_n

    return (x_mean, y_mean)
```

This is for the $x$ and for the $y$, so: $\bar{x}$ and $\bar{y}$.

After finding the mean for each feature, it's time to find the variance, which will allow us to find the standard deviation:

$$
\sigma^2 = \frac{\sum(x_i - \bar{x})^2}{n-1}
$$

```python
def _calculate_variance(self):
    # x
    x_n = len(self.xs)
    x_mean_dist_sum = 0
    for x in self.xs:
        x_mean_dist_sum += (x - self.x_mean) ** 2
    x_variance = x_mean_dist_sum / (x_n - 1)  # applies bessel's correction

    # y
    y_n = len(self.ys)
    y_mean_dist_sum = 0
    for y in self.ys:
        y_mean_dist_sum += (y - self.y_mean) ** 2
    y_variance = y_mean_dist_sum / (y_n - 1)  # applies bessel's correction

    return (x_variance, y_variance)
```

The $n-1$ is important because we are dealing with a sample, not with the population of the data. This is called **Bessel's Correction** and **Towards Data Science** has a very clever explanation to why it matters to this context, you can check it out here: https://towardsdatascience.com/bessels-correction-why-do-we-divide-by-n-1-instead-of-n-in-sample-variance-30b074503bd9/

In this case, we still find the variance ($\sigma^2$) for each feature ($x$ and $y$) so it's actually:

$$
s^2_x = \frac{\sum(x_i - \bar{x})^2}{n-1}
$$

$$
s^2_y = \frac{\sum(y_i - \bar{y})^2}{n-1}
$$

After finding the variance, it's time for the *standard deviation*. This is crucial to understand how the data shifts against the mean, but using standard units based on the original data. Since variance has already been found, it's just a matter of:

$$
s_x = \sqrt{s^2_x}
$$

$$
s_y = \sqrt{s^2_y}
$$

```python
def _calculate_std_deviation(self):
    # x
    x_std_deviation = math.sqrt(self.x_variance)
    y_std_deviation = math.sqrt(self.y_variance)

    return (x_std_deviation, y_std_deviation)
```

As simple as that.

In LSR, the variance and the standard deviation are our powerful weapons to find the optimal line, because they are needed on the next calculation: the *correlation coefficient*. It's a value between -1 and 1 that indicates how correlated the data truly is. If it's negative, it's negatively correlated, which means that it's on a downward slope, otherwise, it's on a upward slope, just that.

It's crucial to have the correlation coefficient to actually find the $m$ and $b$ of our linear regression. This value is denoted as $r$.

$$
r = \frac{1}{n-1} \sum{(\frac{x_i-\bar{x}}{s_x})(\frac{y_i-\bar{y}}{s_y})}
$$

If you already know some statistics you may notice that the products on the sum are actually the z-scores!

$$
z = \frac{x_i-\bar{x}}{s_x}
$$

```python
def _calculate_zscore(
    self, i: float | int, mean: float | int, std_dev: float | int
):
    """
    Calculates the z-score for a single point

    i: Data point
    mean: Mean
    std_dev: Standard deviation
    """
    return (i - mean) / std_dev

def _calculate_r(self):
    """
    Calculates correlation coefficient for the dataset
    """

    n = len(self.data_points)

    correlation_sum = 0
    for i in self.data_points:
        x, y = i[0], i[1]

        x_zscore = self._calculate_zscore(x, self.x_mean, self.x_std_deviation)
        y_zscore = self._calculate_zscore(y, self.y_mean, self.y_std_deviation)

        correlation_sum += x_zscore * y_zscore

    return correlation_sum / (n - 1)
```

That's why standard deviation matters here. We sum the products of the z-scores for each feature to actually find how correlated they are!

Now to actually find the $m$ and $b$:

$$
m = r\frac{s_y}{s_x}
$$

$$
b = \bar{y} - (m\bar{x})
$$

Which in turn gives us:

$$
y = mx+b
$$

```python
def fit(self):
    """
    Fits the Least Squares Regression model by calculating the means, variances, standard deviations, the correlation coefficient and finally, the slope (m) and intercept (b). All of this of course, based on the data points given upon the object's instantiation.
    """
    self.mean = self._calculate_mean()
    self.x_mean = self.mean[0]
    self.y_mean = self.mean[1]
    print(f"Succesfully found mean - x: {self.x_mean} | y: {self.y_mean}")

    self.variance = self._calculate_variance()
    self.x_variance = self.variance[0]
    self.y_variance = self.variance[1]
    print(
        f"Succesfully found variance - x: {self.x_variance} | y: {self.y_variance}"
    )

    self.std_deviation = self._calculate_std_deviation()
    self.x_std_deviation = self.std_deviation[0]
    self.y_std_deviation = self.std_deviation[1]
    print(
        f"Succesfully found standard deviation - x: {self.x_std_deviation} | y: {self.y_std_deviation}"
    )

    self.r = self._calculate_r()
    print(f"Succesfully found correlation coefficient: {self.r}")

    self.m = self.r * (self.y_std_deviation / self.x_std_deviation)
    self.b = self.y_mean - (self.m * self.x_mean)

    print(f"Succesfully found m and b - m: {self.m} | b: {self.b}")

    self._calculate_residuals()

    return (self.m, self.b)
```

And... done!

```python
def predict(self, x: float | int):
    return self.m * x + self.b

def plot(
    self, with_original_points: bool = True, predict: int | float | None = None
):
    plt.figure(figsize=(10, 6))

    line_xs = [min(self.xs), max(self.xs)]
    line_ys = [self.predict(x) for x in line_xs]

    plt.plot(
        line_xs,
        line_ys,
        color="orange",
        label=f"LSR: y={self.m:.2f}x + {self.b:.2f}",
    )

    plt.scatter(self.xs, self.ys, label="Original Data")

    for x, y in zip(self.xs, self.ys):
        pred_y = self.predict(x)
        plt.plot([x, x], [y, pred_y], color="red", linestyle="--", alpha=0.5)

    plt.title("Least Squares Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

By testing that out with real data:

```python
from lsr import LSR
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/house_prices_practice.csv").sample(n=50)

X = 'LotArea'
Y = 'SalePrice'

formatted_data = list(zip(df[X].to_list(), df[Y].to_list()))

model = LSR(data_points=formatted_data)
model.fit()

model.plot()
```

Check out the project at: https://github.com/matjsz/lsr

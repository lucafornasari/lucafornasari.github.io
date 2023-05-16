---
title: 'Linear Regression'
date: 2023-04-20
permalink: /posts/2023/04/linear-regression/
tags:
  - machine learning
  - linear regression
---

In this post we will talk about **Linear Regression** models, one of the fundamental techniques in statistical modeling and machine learning. We will dive into different approaches trying to understand the mathematical and statistical basis behind. 

**Disclaimer:** *all of the informations you will find in this post are what i learned during the Machine Learning course at Politecnico di Milano (academic year 2022/2023). I will use this blog to better elaborate what I'm learning and, hopefully, leave something useful to the internet people.* 

Linear Regression models provide a powerful way to understand the relationship between a dependent variable and one or more independent variables, and to make predictions based on this relationship.
What Linear Regression will produce is a model **linear in parameters** that maps inputs into continuous targets.

Let's start analyzing a linear model:

$$
y(X,w)=Xw
$$

- y(X,w): The dependent variable y is the response variable that we want to predict. y is a continuous variable, and we want to model the relationship between y and one or more independent variables. **Nx1** dimensional.
- X: The independent variables X are the predictor variables that we use to model the relationship with the dependent variable y. X is a matrix of features that describe each observation in the data set. **Nxn** dimensional (*N* represents the samples, *n* the features).
- w: The parameters w are the coefficients that we estimate to fit the model to the data. Each element of the parameter vector w corresponds to the weight of the corresponding feature in the matrix X. **nx1** dimensional.

But what can we approximate using a linear model described above?
The answer is: only lines. But Linear Regression is mostly used to approximate curves, right? How is it possible to keep a *linear* model that approximates *curves*?
We can extend the previous model by using **basis functions**

$$
y(\Phi,w)=\Phi w
$$

- $\Phi$ is a **NxM** vector of non-linear basis functions of the input vector **x**. In this way we will be able to have a model that is linear in parameters **w**.

$$
X=[x_1, x_2, ..., x_N]^T \\
\phi=[\phi_1(X), \phi_2(X), ..., \phi_M(X)]^T \\
\boldsymbol{\Phi}=[\phi(x_1), \phi(x_2), ..., \phi(x_N)]^T 
$$

![](https://ik.imagekit.io/frnz98/LR_1.png?updatedAt=1683710631203)
*Figure 1.1: Linear model in input space (top left) and in feature space (bottom right)*

Some possible basis functions we can use:

- Polynomial $\rightarrow \phi_j(x) = x^j$
  
- Gaussian $\rightarrow \phi_j(x) = e^{-\frac{(x-\mu_j)^2}{2\sigma^2}}$
  
- Sigmoidal $\rightarrow \phi_j(x) = \frac{1}{1+e^{(\mu_j-x)/\sigma}}$


Once the model has been defined, we need to define the **Loss Function**. Generally, the
common choice is the *square loss*, i.e. (considering 1 input vector and 1 target value):

$$
L(t, y(X))=(t − y(X))^2
$$

This is an empirical loss computed from the target and the inputs, which are random variables. This implies that the loss has a distribution, i.e. the joint distribution p(t, x). In order to evaluate how bad the model performs on a sample < x, t >, we consider the expected loss:

$$
L(t, y(X))=\int \int (t − y(X))^2p(X, t)dXdt
$$


# Direct Approach

Let's now dive into the first technique we will discover in this post: the Ordinary **Least Squares** method (OLS).
It is classified as a *Direct Approach* since it finds the optimal model y(X) without passing from the definition above with probabilities, but it estimates it directly from the data.

### Model definition
The LS model is the one desribed in function **(2)**: NxM matrix $\Phi$, Mx1 parameter vector w and Nx1 result y.

### Loss definition
The Loss Function is typically the residual sum of squares **RSS**, which is the L2-norm of the residual error $\epsilon$:

$$
L(X,w)=\frac{1}{2}\sum(t-y(X,w))^2 \\
\epsilon=(t-y(X,w))
L(X,w)=\frac{1}{2}||\epsilon||_2^2=\epsilon^T\epsilon
$$

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/1043px-Linear_least_squares_example2.svg.png" style="display: block; margin-left: auto; margin-right: auto;width: 400px;height: 400px;">

## Optimization
Now that we have our loss function we have to minimize it, hence choose $\boldsymbol{w}$ so that $E(\boldsymbol{w})$ is as small as possible. Since the error function is a quadratic function of the coefficients $\boldsymbol{w}$, its derivative with respect to them will be linear, and so the minimization has a unique solution, denoted by $\boldsymbol{w^*}$.

Putting the gradient of the Loss = 0 and checking that the eigenvalues of the Hessian are >=0, we find that the solution found by the LS method is:

$$
\boldsymbol{w}^{LS} = (\Phi^T\Phi)^{-1}\Phi^Tt
$$

The method **cannot** be applied if:
- The matrix $\Phi^T\Phi$ is singular, hence some features are dependent
- The number of data is smaller than the number of the features (N<M), i.e. the system $y(X,w)=\Phi w$ is not solvable wrt **w+*.
- There are too many data, thus the computation is too expensive. So, in this case a good idea could be using an online algorithm, in which the gradient is computed one sample at a time. The most famous and used algorithm is the **stochastic gradient descent**, which updates the parameters **w** at each iteration in the following way:

$$
\boldsymbol{w}^{k+1} = \boldsymbol{w}^k - \eta\nabla E_n
$$



------

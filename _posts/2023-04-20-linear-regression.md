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


## Direct Approach

Let's now dive into the first technique we will discover in this post: the Ordinary **Least Squares** method (OLS).
It is classified as a *Direct Approach* since it finds the optimal model y(X) without passing from the definition above with probabilities, but it estimates it directly from the data.

### Model definition
The LS model is the one desribed in function **(2)**: NxM matrix $\Phi$, Mx1 parameter vector w and Nx1 result y.

### Loss definition
The Loss Function is typically the residual sum of squares **RSS**, which is the L2-norm of the residual error $\epsilon$:

$$
L(X,w)=\frac{1}{2}\sum(t-y(X,w))^2 \\
\epsilon=(t-y(X,w)) \\
L(X,w)=\frac{1}{2}||\epsilon||_2^2=\frac{1}{2}\epsilon^T\epsilon
$$

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/1043px-Linear_least_squares_example2.svg.png" style="display: block; margin-left: auto; margin-right: auto;width: 400px;height: 400px;">

### Optimization
Now that we have our loss function we have to minimize it, hence choose $\boldsymbol{w}$ so that $E(\boldsymbol{w})$ is as small as possible. Since the error function is a quadratic function of the coefficients $\boldsymbol{w}$, its derivative with respect to them will be linear, and so the minimization has a unique solution, denoted by $\boldsymbol{w^*}$.

Putting the gradient of the Loss = 0 and checking that the eigenvalues of the Hessian are >=0, we find that the solution found by the LS method is:

$$
\boldsymbol{w}^{LS} = (\Phi^T\Phi)^{-1}\Phi^Tt
$$

The method **cannot** be applied if:
- The matrix $\Phi^T\Phi$ is singular, hence some features are dependent
- The number of data is smaller than the number of the features (N<M), i.e. the system $y(X,w)=\Phi w$ is not solvable wrt **w**.
- There are too many data, thus the computation is too expensive. So, in this case a good idea could be using an online algorithm, in which the gradient is computed one sample at a time. The most famous and used algorithm is the **stochastic gradient descent**, which updates the parameters $\boldsymbol{w}$ at each iteration in the following way:

$$
\boldsymbol{w}^{k+1} = \boldsymbol{w}^k - \eta\nabla L(X,w)
$$


## Discriminative Approach

In linear regression, a discriminative approach refers to modeling the conditional probability distribution of the target variable *t* given the input variables *X*. In other words, it focuses on estimating the probability $P(t|X)$, which represents the **Likelyhood** of observing a particular target value *t* given an input value *X*. The method we are going to see is called **Maximum Likelyhood** estimation.

### Model definition
Differently from the direct approach, we assume that the estimated target *t* is given by a deterministic function $y(x,w)$ with additive Gaussian noise:

$$
y(X,w)=\Phi w + \epsilon
$$

where $\epsilon \sim \mathcal{N}(0,\sigma^2)$. Hence $t \sim \mathcal{N}(y(X,w),\sigma^2)$.

### Loss definition
Given that the entries are all **i.i.d.** (Independent Identically Distributed), the global likelyhood for N samples is the product of the likelyhood of the single entries:

$$
p(\boldsymbol{t}|\Phi\boldsymbol{w},\sigma^2) = \prod_{n=1}^{N}\mathcal{N}(t_n|\boldsymbol{w}^T\phi(x_n), \sigma^2)
$$

By exploiting the *ln* properties, we can consider the **log-likelyhood** in order to handle a sum and not a product:

$$
ln(p(\boldsymbol{t}|\Phi\boldsymbol{w},\sigma^2)) = \sum_{n=1}^{N}ln(\mathcal{N}(t_n|\boldsymbol{w}^T\phi(x_n), \sigma^2))
$$

### Optimization
Since the zeros of a function *f* are the same of a function *ln(f)*, we can put the gradient of the log-likelyhood = 0 and find:

$$
\boldsymbol{w^{ML}} = (\Phi^T\Phi)^{-1}\Phi^T\boldsymbol{t}
$$

We notice that the result of the **ML** estimation, in case of **i.i.d.** variables, is the same of the **LS** estimation.
Moreover, the **ML** estimation has the minimum variance among all the unbiased estimators:

$$
VAR[\boldsymbol{w^{ML}}] = (\Phi^T\Phi)^{-1}\Phi^T\sigma^2
$$


## Regularization

When we design and train a ML model is important to avoid **overfitting** and **underfitting**. In particular, the first one usually depends on the high values of the parameters. 
Resularization techniques help to avoid the problem by using a loss that is also function of the parameters' modules.

$$
L(w)=L_D(w)+\lambda L_w(w)
$$

Where $L_D$ is the error on data, and $L_w$ is the error on parameters.
The two most used solutions are called **Ridge* and **Lasso**:

### Ridge Regression

$$
L_w(w)=\frac{1}{2}w^Tw=\frac{1}{2}\sum|w_j|^2=\frac{1}{2}||\epsilon||_2^2 \\
w^{ridge}=(\lambda I+\Phi^T\Phi)^{-1}\Phi^Tt
$$

Where $\lambda>0$. This forces the eigenvalues of the inverted matrix to be > 0 and thus there are no big values of parameters (smoother function). The advantage of Ridge regression is that the loss function remains quadratic in $w$, so its exact minimizer can be found in closed form.

### Lasso Regression

$$
L_w(w)=\frac{1}{2}\sum|w_j|=\frac{1}{2}||\epsilon||_1
$$

where $\lambda>0$. It is nonlinear, there is no close-form solution but the non-important features are set to zero thus it leads to a sparse model.

<img src="https://i.ibb.co/373pzxc/ridge-lasso.png" style="display: block; margin-left: auto; margin-right: auto;width: 650px;height: 400px;">

In the picture above we can see where the origin of sparsity in Lasso comes from: the optimum value $\boldsymbol{w}^*$ will probably be on one of the vertices, hence on the axes, where some features are zero (the features in the image are indicated as $\beta_j$).
In cases of multi-correlation, i.e. many features are correlated with each other, this can be useful as the Lasso regression will set some of them to zero as said before.


## Bayesian Linear regression

Till now we have adopted a frequentist approach, namely we've seen the probabilities in terms of frequencies of random, repeatable events. Sometimes, however, it's not possible to repeat multiple times an event to obtain a notion of probability. Moreover, we've also seen that using maximum likelihood for setting the parameters we have the problem of the model complexity. Regularization is a good answer, but still, the choice of the basis functions remains important and also the value of $\lambda$ is an incognita.

Bayesian approach aims to estimate the parameters by incorporating prior information or assumptions about them. It is called **Maximum A Posteriori** (MAP) since it finds the parameter values that maximize the posterior probability of the parameters given the observed data. The posterior probability is computed using Bayes' theorem, which states:

$$
P(A|B)=\frac{p(B|A)P(A)}{P(B)}
$$

An important consideration is that this approach is not affected by the problem of overfitting and it also leads to automatic methods of determining model complexity using the training data alone.

But let's now apply the Bayes theorem to our case:

$$
P(w|D)=\frac{P(D|w)P(w)}{P(D)}
$$

- $P(w|D)$: posterior distribution, the probability of the parameters $w$ given the data $D$.
- $P(D|w$: likelyhood.
- $P(w)$: prior distribution, it represents the prior knowledge, assumptions, or beliefs about the parameters before observing the data. If we have any domain knowledge, or a guess for what the model parameters should be, we can include them in our model, unlike in the frequentist approach which assumes everything there is to know about the parameters comes from the data. If we don’t have any estimates ahead of time, we can use non-informative priors for the parameters such as a normal distribution.

Let's now jump into our model definition.
Since we assume a Gaussian likelyhood, it is convenient to consider a Gaussian distribution for parameters too:

$$
P(w) \sim N(w|w_0, S_0)
$$

Where $w_0$ is a **Mx1** mean and $S_0$ is the **MxM** covariance matrix.
For the likelyhood the definition is the same as the MLE approach.

Thus, according to Bayes theorem:

$$
p(\boldsymbol{w}|\Phi\boldsymbol{t},\Phi,\sigma^2) \propto \mathcal{N}(\boldsymbol{w}|\boldsymbol{w_0},\boldsymbol{S_0})\mathcal{N}(\boldsymbol{t}|\Phi\boldsymbol{w},\sigma^2\boldsymbol{I_N}) = \mathcal{N}(\boldsymbol{w}|\boldsymbol{w_N},\boldsymbol{S_N})
$$

The result is a **multivariate Gaussian** distribution $\mathcal{N}(\boldsymbol{w}|\boldsymbol{w_N},\boldsymbol{S_N})$.

Maximizing the posterior we get:

$$
\boldsymbol{w_N} = \boldsymbol{S_N}(\boldsymbol{S_0}^{-1}\boldsymbol{w_0}+\frac{\Phi^T\boldsymbol{t}}{\sigma^2}) \\
\boldsymbol{S_N}^{-1} = \boldsymbol{S_0}^{-1} + \frac{\Phi^T\Phi}{\sigma^2}
$$

- If $\boldsymbol{S_0} \rightarrow + \infty$ we have no prior knowledge, thus the MAP estimation is the same of the MLE: $\boldsymbol{w^{MAP}} = (\Phi^T\Phi)^{-1}\Phi^T\boldsymbol{t}$.
- If $\boldsymbol{w_0}=0, \boldsymbol{S_0}=\tau^2 I$, MAP estimation coincides with a Ridge regression with $\lambda=\frac{\sigma^2}{\tau^2}: \boldsymbol{w^{MAP}} = (\frac{\sigma^2}{\tau^2} I+\Phi^T\Phi)^{-1}\Phi^T\boldsymbol{t}$.





------

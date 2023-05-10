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
\boldsymbol{\Phi}=[\Phi_1(X), \Phi_2(X), ..., \Phi_M(X)]^T \\
\boldsymbol{\Phi}=[\Phi(x_1), \Phi(x_2), ..., \Phi(x_N)]^T 
$$

![](https://ik.imagekit.io/frnz98/LR_1.png?updatedAt=1683710631203)
*Figure 2.1: Linear model in input space (top left) and in feature space (bottom right)*

------

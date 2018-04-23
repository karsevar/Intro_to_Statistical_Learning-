### chapter 7 Moving Beyond Linearity:
#standard linear regression can have significant limitations in terms of predictive power. This is because the linearity assumption is almost always an approximation, and sometimes a poor one. In chapter 6 we see that we can improve upon least squares using ridge regression, the lasso, principal components regression, and other techniques. In that setting, the improvement is obtained by reducing the complexity of the linear model, and hence the variance of the estimates. 

#In this chapter we relax the linearity assumption while still attempting to maintain as much interpretability as possible. We do this by examining very simple extensions of linear models like polynomial regression and step functions, as well as more sophisticated approaches such as splines, local regression, and generalized additive models. 
	#Polynomial regression extends the linear model by adding extra predictors, obtained by raising each of the original predictors to a power. This approach provides a simple way to provide a non-linear fit to data. 
	#Step functions cut the range of a variable into K distinct regions in order to produce a qualitative variable. this has the effect of fitting a piecewise constant function. 
	#Regression splines are more fexible than polynomials and step functions, and in fact are an extension of the two. They involve dividing the range of X into K distinct regions. Within each region, a polynomial function is fit to the data. However, these polynomials are contrained so that they join smoothly at the region boundaries, or knots. Provided that the interval is divided into enough regions, this can produce an extremely flexible fit. 
	#Smoothing splines are similar to regression splines, but arise in a slightly different situation. Smoothing splines result from minimizing a residual sum of squares criterion subject to a smoothness penality. 
	#Local regression is similar to splines, but differs in an important way. The regions are allowed to overlap, and indeed they do so in a very smooth way. 
	#Generalized additive models allow us to extend the methods above to deal with multiple predictors. 
	
##7.1 Polynomial regression:
#A polynomial regression allows us to produce an extremely non-linear curve. Notice that the coefficients in (7.1) can be easily estimated using least squares linear regression because this is just a standard linear model with predictors x_i, x_i^2, x_i^3, ..., x_i^d. Generally speaking, it is unusual to use d greater than 3 or 4 because for large values of d, the polynomial cruve can become overly flexible and can take on some very strange shapes. This is especially true near the boundary of the X variable. (Meaning that the model is highly biased and can't be used effectively to map predicted Y variables).

library(ISLR)
# Figure 7.1 representation:
#logistical regression with a quadratic transformation: (Two variables in all):
age.fit <- lm(wage ~ poly(age, 4), data = Wage)
plot(Wage$age, Wage$wage)
conf.pred <- predict(age.fit, interval = "confidence", level = 0.95, newdata = data.frame(age = c(1:110))) 
lines(x = c(1:110), y = conf.pred[,1], col = "blue", lty = 1)
lines(x = c(1:110), y = conf.pred[,2], col = "blue", lty = 2)
lines(x = c(1:110), y = conf.pred[,3], col = "blue", lty = 2)
#just like in the representation in figure 7.1 left plot. Will need to see what operation the author used in the second plot. 

#It seems that the author made the data set into a logistical model through the converstion of the wage data into a categorical variable with two levels (Wage > 250 and Wage <250)
Wage$wage10 <- rep(NA, times = nrow(Wage))
for( i in 1:nrow(Wage)){
	if(Wage$wage[i] > 250){
		Wage$wage10[i] <- TRUE
	} else {
		Wage$wage10[i] <- FALSE
	}
}
table(Wage$wage10)

glm.fit <- glm(wage10 ~ poly(age, 4), data = Wage, family = binomial)
summary(glm.fit)
glm.probs <- predict(glm.fit, newdata = data.frame(age = c(1:100)),type = "response", level = 0.95)
plot(y = glm.probs, x = c(1:100))# I will come back to this command later. Interesting little experiment on how to graph binomial prediction data. I can't seem to find the confidence interval argument for the prediction equivalent of glm().

#In figure 7.1, a pair of dotted curves accompanies the fit, these are (2 times) standard error curves. Let's see how these arise. Suppose we have computed the fit at a particular value of age, x_0: To see the equation look at page 267. 

#What is the variance of the fit? Least squares returns variance estimates for each of the fitted coefficients beta_hat_j, as well as the covariances between the pairs of the coefficient estimates. We can use these to compute the estimated variance of f_hat(x_0). The estimated pointwise standard error of f_hat(x_0) is the square root of this variance. This computation is repeated at each reference point x_0, and we plot the fitted curve , as well as the twice standard error because, for normally distributed error terms, this quantity corresponds to an approximate 95 percent confidence interval. 

#It seems like the wages in figure 7.1 are from two distinct populations there appears to be a high earners group earning more than 250,000 as well as a low earners group. We can treat wage as a binary variable by splitting it into these two groups. logistic regression can then be used to predict this binary response, using polynomial functions of age as predictors.

#The result is shown in the right-hand panel of figure 7.1. The grey marks on the top and bottom of the panel indicate the ages of the high earners and the low earners. The solid blue curve indicates the fitted probabilities of being a high earner, as a function of age. The estimated 95 percent confidence interval is shown as well. We see that here the confidence intervals are fairly wide, especially on the right-hand side. Although the sample size for this data set is substantial (n = 3000), there are only 79 high earners, which results in a high variance in the estimate dcoefficients and consequently wide confidence intervals. 

##7.2 Step functions:
#Using polynomial functions of the features as predictors in a linear model imposes a global structure on the non-linear function of X. We can instead use step functions in order to avoid imposing such a global structure. Here we breaks the range of X into bins, and fit a different constant in each bin. This amount to converting a continuous variable into an ordered categorical variable. 

#In greater detail, we create cutpoints c_1, c_2, ..., c_K in the range of X, and then construct K + 1 new variables. To see figure 7.4 look at page 268. where I(.) is an indicator function that returns a 1 if the condition is true, and returns a 0 otherwise. For example, I(c_k <= X) equals 1 if c_k <= X, and equals 0 otherwise. These are sometimes called dummy variables. Notice that for any value of X, C_0(X) + C_1(X) + ... + C_k(X) = 1, since X must be in exactly one of the K + 1 intervals. We then use least squares to fit a linear model using C_1(X), C_2(X), ..., C_K(X) as predictors:
		#y_i = beta_0 + beta_1C_1(x_i) + .... + beta_kC_k(x_i) + epsilon.
		
#for a given value of X, at most one of C_1, C_2, ..., C_K can be non-zero> Note that when X < c_1, all of the predictors in (7.5) are zero, so beta_0 can be interpreted as the mean value of Y for X < c_1. By comparison, (7.5) predicts a response for X in c_j <= X < c_j + 1 relative to X < c_1.

#An example of fitting step functions to the Wage data from figure 7.1 is shown in the left hand panel of figure 7.2. We also fit the logistic regression model ( see page 270) in order to predict the probability that an individual is a high earner on the basis of age. The right-hand panel of figure 7.2 displays the fitted posterior probabilities obtained using this approach. 

#Unfortunately, unless there are natural breakpoints in the predictors, piecewise-constant functions can miss the action. For example, in the left hand panel of Figure 7.2, the first bin clearly misses the increasing trend of wage with age. 

##7.3 Basis Functions:
#Polynomial and piecewise-constant regression models are in fact special cases of a basis function approach. The idea is to have at hand a family of functions or transformations that can be applied to a variable X: b_1(X), b_2(X), ..., b_k(X). 

#Note that the basis functions b_1(.), b_2(.), ..., b_k(.) are fixed and known. (In other words, we choose the functions ahead of time.) For polynomial regression, the basis functions are b_j(x_i) = x^j_i, and for piecewise constant functions they are b_j(x_i) = I(c_j <= x_i < c_j + 1). We can think of this method's equation as a standard linear model with predictors b_1(x_i),b_2(x_i), ..., b_k(x_i). hence, we can use least squares to estimate the unknown regression coefficients in (7.7). Importantly, this means that all of the inference tools for linear models that are discussed in chapter 3 are available in this setting. 

#Thus far we have considered the use of polynomial functions and piecewise constant functions for our basis functions; however, many alternatives are possible. For instance, we can use wavelets or Fourier series to construct basis functions. 

##7.4 Regression Splines:
##7.4.1 Piecewise Polynomials:
#Instead of fitting a high-degree polynomial over the entire range of X, piecewise polynomial regression involves fitting separate low-degree polynomials over different regions of X. For example, a piecewise cubic polynomial works by fitting a cubic regression model of the form:
		# y_i = beta_0 + beta_1x_i + beta_2x_^2_i + beta_3x^3_i + epsilon
		
#Where the coefficients beta_0, beta_1, and beta_2 differ in different parts of the range of X. The points where the coefficients change are called knots. For example, piecewise cubic with no knots is just a standard cubic polynomial, as in (7.1) with d = 3.

#Using more knots leads to a more flexible piecewise polynomial. In general, if we place K different knots throughout the range of X, then we will end up fitting K+1 different cubic polynomials. Note that we do not need to use a cubic polynomial. For example, we can instead fit piecewise linear functions. 

##7.4.2 Constraints and splines:
#The top left panel of Figure 7.3 looks wrong because the fitted curve is just too flexible. To remedy this problem, we can fit a piecewise polynomial under the constraint that the fitted curve must be continuous. In other words, there cannot be a jump when age = 50. The top right plot in Figure 7.3 shows the resulting fit. This looks better than the top left plot, but the V-shaped join looks unnatural. 

#In the lower left plot, we have added two additional constraints: now both the first and second derivatives of the piecewise polynomials are continuous at age = 50. In other words, we are requiring that the piecewise polynomial be not only continuous when age = 50, but also very smooth. Each constraint that we impose on the piecewise cubic polynomials effectively frees up on degree of freedom, by reducing the complexity of the resulting piecewise polynomial fit. so in the top left plot, we are using eight degrees of freedom, but in the bottom left plot we imposed three constraints (continuity, continuity of the first derivative, and continuity of the second derivative) and so are left with five degrees of freedom. The curve in the bottom left plot is called a cubic spline. In general, cubic spline with K knots uses a total of 4 + K degrees of freedom. 

#In figure 7.3, the lower right plot is a linear spline, which is continuous at age = 50. The general definition of a degree-d spline is that it is a piecewise degree-d polynomial, with continuity in derivatives up to degree d - 1 at each knot. Therefore, a linear spline is obtained by fitting a line in each region of the predictor space defined by the knots, requiring continuity at each knot. 

#Just as there were several ways to represent polynomials, there are also many equivalent ways to represent cubic splines using different choices of basis functions in (7.9). The most direct way to represent a cubic spline using (7.9) is to start off with a basis for a cubic polynomial --- namely x, x^2, x^3 -- and then add one truncated power basis function per knot. Consult page 273 for the knot equation and weird symbol that goes with knot. 

#Unfortunately, splines can have high variance at the outer range of the predictors --- that is, when X takes on either a very small or very large value. Figure 7.4 shows a fit to the Wage data with three knots. We see that the confidence bands in the boundary region appear fairly wild. A natural spline is a regression spline with additional boundary contraints;the function is required to be linear at the boundary (in the region where X is smaller than the smallest knot, or larger than the largest knot). This additional constraint means that natural splines generally produce more stable estimates at the boundaries. 

##7.4.4 Choosing the number of locations of the knots:
#When we fit a spline, where should we place the knots? The regression spline is most flexible in regions that contain a lot of knots, because in those regions the polynomial coefficients can change rapidly. Hence, one option is to place more knots in places where we feel the function might vary most rapidly, and to place fewer knots where it seems more stable. While this option can work well, in practice it is common to place knots in a uniform fashion. One way to do this is to specify the desired degrees of freedom, and then have the software automatically place the corresponding number of knots at unform quantiles of the data. 

#How many knots should we use, or equivalently how many degrees of freedom should our spline contain? One option is to try out different numbers of knots and see which produces the best looking curve. A somewhat more objective approach is to use cross-validation. With this method, we remove a portion of the data (say 10 percent), fit a spline with a certain number of knots to the remaining data, and then use the spline to make predictions for the held out portion. We repeat this process multiple times until each observation has been left out once, and then compute the overall cross-validation RSS. This procdure can be repeated for different numbers of knots K. Then the value of K giving the smallest RSS is chosen. 

#Figure 7.6 shows ten-fold cross validation mean squared errors for splines with various degrees of freedom fit to the Wage data. The left-hand panel corresponds to a natural spline and the right-hand panel to a cubic spline. The two methods produce almost identical results, with clear evidence that a one-degree fit (a linear regression) is not adequate. Both curves flatten out quickly, and it seems that three degrees of freedom for the natural spline and four degrees of freedom for the cubic spline are quite adequate. 

#In Section 7.7 we fit additive spline models simultaneously on several variables at a time. This could potentially require the selection of degrees of freedom for each variable. In cases like this we typically adopt a more pragmatic approach and set the degrees of freedom to a fixed number, say four, for all terms. 

##7.4.5 comparison to polynomial regression:
#Regression splines often give superior results to polynomial regression. This is because unlike polynomials, which must use a high degree to produce flexible fits, plines introduce flexibility to increasing the number of knots by keeping the degree fixed. Generally, this approach produces more stable estimates. Splines also allow us to place more knots, and hence flexibility, over regions where the function f seems to be changing rapidly, and fewer knots where f appears more stable. 

##7.5 Smoothing splines:
##7.5.1 An overview of smoothing splines:
#In the last section we discussed regression splines, which we create by specifying a set of knots, producing a sequence of basis functions, and then using least squares to estimate the spline coefficients. We now introduce a somewhat different approach that also produces a spline. 

#In fitting a smooth curve to a set of data, what we really want to do is find some function, say g(x), that fits the observed data well: that is, we want RSS to be small. However, there is a problem with this approach. If we don't put any constraints on g(x_i), then we can always make RSS zero simply by choosing g such that it interpolates all of the y_i. Such a function would woefully overfit the data. What we really want is a function g that makes RSS small, but that is also smooth. 

#A natural approach is to find the function g that minimizes (look at 277 to see equation 7.11) where lambda is a nonnegative tuning parameter. the function g that minimizes (7.11) is known as a smoothing spline. 

#Equation 7.11 takes the Loss + penalty formulation that we encounter in the context of ridge regression and the lasso in chapter 6. Where the equation describes a loss function that encourages g to fit the data well and a penalty term that penalizes the variability in g. The notation g''(t) indicates the second derivative of the function g. The first derivative g'(t) measures the slope of a function at t, and the second derivative corresponds to the amoutn by which the slope is changing. Hence, broadly speaking, the second derivative of a function is a measure of its roughness: it is large in absolute value if g(t) is very wiggly near t, and it is close to zero otherwise. The larger the value of lambda, the smoother g will be.

#When lambda = 0, then the penalty term in (7.11) has no effect, and so the function g will be very jumpy and will exactly interpolate the training observations. When lambda -> infinity, g will be perfectly smooth -- it will just be a straight line that passes as closely as possible to the training points. In fact, in this case, g will be the linear least squares line, since the loss function in (7.11) amounts to minimizing the residual sum of squares. for an intermediate value of lambda, g will approximate the training observations but will be somewhat smooth. We see that lambda controls the bias-variance trade off of the smoothing spline. 

#The function g(x) that minimizes (7.11) can be shown to have some special properties: it is a piecewise cubic polynomial with knots at the unique values of x_1, x_2, ..., x_n, and continuous first and second derivatives at each knot. Furthermore, it is linear in the region outside of the extreme knots. However, it is not the same natural cubic spline that one would get if one applied the basis function approach described in section 7.4.3 with knots at x_i, ..., x_n --- rather, it is a shrunken version of such a natural cubic spline, where the value of the tuning parameter lambda in (7.11) controls the level of shrinkage. 

## 7.5.2 Choosing the smoothing parameter lambda:
#We have seen that a smoothing spline is simply a natural cubic spline with knots at every unique value of x_i. It might seem that a smoothing spline will have far too many degrees of freedom, since a knot at each data point allows a great deal of flexibility. But the tuning parameter lambda controls the roughness of the smoothing spline, and hence the effective degrees of freedom. It is possible to show that as lambda increases from 0 to infinity, the effective degrees of freedom, which we write df_lambda, decrease from n to 2. 

#In the context of smoothin splines, why do we discuss effective degrees of freedom instead of degrees of freedom? Usually degrees of freedom reger to the number of free parameters, such as the number of coefficients fit in a polynomial or cubic spline. Although as smoothing spline has n parameters and hence n nominal degrees of freedom, these n parameters are heavily constrained or shrunk down. Hence df_lambda is a measure of the flexibility of the smoothing spline -- the higher it is, the more flexible (and the lower bias but higher variance) the smoothing spline. The definition of effective degrees of freedom is somewhat technical. We can write
			#g_hat = S_lambda*y,
#Where g_hat is the solution to (7.11) for a particular choice of lambda -- that is, it is a n-vector containing the fitted values of the smoothing spline at the training points x_1, ..., x_n. Equation 7.12 indicates that the vector of fitting values when applying a smoothing spline to the data can be written as a n * n matrix S_lambda times the response vector y.

#In fitting a smoothing spline, we do not need to select the number or location of the knots --- there will be a knot at each training observation, x_1, ..., x_n. Instead, we have another problem: we need to choose the value of lambda through LOOCV method as a means to calculate which lambda has the smallest RSS value. Look at page 279 to see the equation used for this problem. 

## 7.6 Local regression:
#Local regression is a different approach for fitting flexible non-linear functions, which involves computing the fit at a target point x_0 using only the nearby training observations. 

#In order to obtain the local regression fit at the new point, we need to fit a new weighted least squares regression model by minimizing (7.14) for a new set of weights. Locat regression is sometimes referred to as a memory based procedure, because like nearest neightbors, we need all the training data each time we wish to compute a prediction. 

#algorithm 7.1 (local regression At X = x_0)
	#1. Gather the fraction s = k/n of training points whose x_i are closest to x_0.
	#2. Assign a weight K_i0 = K(x_i, x_0) to each point in this neightborhood, so that the point furthest from x_0 has weight zero, and the closest has the highest weight. All but these k nearest neighbors get weight zero.
	#3. Fit a weighted least squares regression of the y_i on the x_i using the aforementioned weights, by finding beta_hat_0 and beta_hat_1 that minimize equation 7.14 found on page 282.
	#4. The fitted value at x_0 is given by f_hat(x_0) = beta_Hat_0 + beta_hat_1*x_0
	
#In order to perform local regression, there are a number of choices to be made, such as how to define the weighted function K, and whether to fit a linear, constant, or quadratic regression in Step 3 above. While all of these choices mkae some difference, the most important choice is the span s, defined in Step 1 above. The span plays a role like that of the tuning parameter lambda in smoothing splines: it controls the flexibility of the non-linear fit. The smaller the value of s, the more local and wiggly will be our fit; alternatively, a very large value of s will lead to a global fit to the data using all of the training observations. We can again use cross-validation to choose s, or we can specify it directly. 

#The idea of local regression can be generalized in many different ways. In a setting with multiple features X_1, X_2, ..., X_p, one very useful generalization involves fitting a multiple linear regression model that is global in some variables, but local in another, such as time, Such varying coefficient models are a useful way of adapting a model to the most recently gathered data. Local regression also generalizes very naturally when we want to fit models that are local in a pair of variables X_1 and X_2, rather than one. We can simply use two-dimensional neightborhoods, and fit bivariate linear regression models using the observations that are near each target point in two-dimensional space. Theoretically the same approach can be implemented in higher dimensions, using linear regressions fit to p-dimensional neighborhoods. However, local regression can perform poorly if p is much larger than about 3 or 4 because there will generally be very few training observations close to x_0. Nearest neighbors regression suffers from a similar problem in high dimensions.

##7.7 Generalized Additive models:
#Generalized additive models (GAMs) provide a general framework for extending a standard linear model by allowing non-linear functions of each of the variables, while maintaining additivity. Just like linear models, GAMs can be applied with both quantitative and qualitative responses. 

##7.7.1 GAMs for Regression Problems:
#Look at page 283 for the additive model formula. It is called an additive model because we calculate a separate f_j for each X_j, and then add together all of their contributions. 
#In sections 7.1-7.6, we discuss many methods for fitting funcitons to a single variable. The beauty of GAMs is that we can use these methods as building blocks for fitting an additive model. In fact, for most of the methods that we have seen so far in this chapter, this can be done fairly trivially. Take, for example, natural splines, and consider the task of fitting the model:
		#wage = beta_0 + f_1(year) + f_2(age) + f_3(education) + epsilon

#on the Wage data. Here year and age are quantitative variables, and education is a qualitative variable with five levels. We fit the first two functions using natural splines. We fit the third function using a separate constant for each level, via the usual dummy variable approach of section 3.3.1. Hence the entire model is just a big regression onto spline basis variables and dummy variables, all packed into one big regression matrix. 

#Standard software such as the gam() function can be used to fit GAMs using smoothing splines, via an approach known as backfitting. This method fits a model involving multiple predictors by repeatedly updating the fit for each predictor in turn, holding the others fixed. The beauty of this approach is that each time we update a function, we simply apply the fitting method for that variable to a partial residual.

#We do not have to use splines as the building blocks for GAMs: we can just as well use local regression, polynomial regression, or any combination of the approaches seen earlier in this chapter in order to create a GAM.

##7.7.2 GAMs for classification problems:
#A natural way to extend (7.17) to allow for non-linear relationships is to use the model
#log(p(X)/1 - p(X)) = beta_0 + f_1(X_1) + f_2(X_2)+ ... + f_p(X_p).

## 7.8 Lab: Non-linear Modeling:
library(ISLR)
attach(Wage)

##7.8.1 Polynomial Regression and Step Functions 
fit <- lm(wage ~ poly(age, 4), data = Wage)
coef(summary(fit))
#The function returns a matrix whose columns are a basis of orthogonal polynomials, which essentially means that each column is a linear combination of the variables age, age^2, age^3, age^4. 

#Howeverm we can also use poly() to obtain age, age^2, age^3, and age^4 directly, if we prefer. We can do this by using the raw = TRUE argument to the poly() function. Later we see that this does not affect the model in a meaningfulway -- though the choice of basis clearly affects the coefficient estimates, it does not affect the fitted values obtained. 

fit2 <- lm(wage ~ poly(age, 4, raw = TRUE), data = Wage)
coef(summary(fit2))

#There are several other equivalent ways of fitting this model, which showcase the flexibility of the formula language in R. For example:
fit2a <- lm(wage ~ age + I(age^2) + I(age^3) + I(age^4), data = Wage)
coef(fit2a)
#this simply creates the polynomial basis functions on the fly, taking care to protect terms like age^2 via the wrapper function I().
fit2b <- lm(wage ~ cbind(age, age^2, age^3, age^4), data = Wage)
coef(fit2b)
#this does the same more compactly, using the cbind() function for building a matrix from a collection of vectors; any function call such as cbind() inside a formula also serves as a wrapper.

agelims <- range(age)
age.grid <- seq(from = agelims[1], to = agelims[2])
preds <- predict(fit, newdata = list(age = age.grid), se= TRUE)#Will need to see if I can streamline this process through just simply using predict() with the arguments level = 0.95 and interval = "confidence"
se.bands <- cbind(preds$fit+2*preds$se.fit, preds$fits-2 * preds$se.fit) 

Finally, we plot the data and add the fit from the degree-4 polynomial:
par(mfrow = c(1,2), mar = c(4.5,4.5,1,1), oma = c(0,0,4,0))
plot(age, wage, xlim = agelims, cex = 0.5, col = "darkgrey")
title("Degree-4 Polynomial", outer = TRUE)
lines(age.grid, preds$fit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty = 3)

#Will need to see if I can come up with the same plot through using Tilman Davie's methods:
preds2 <- predict(fit, newdata = data.frame(age = c(age.grid)), interval = "confidence", level = 0.95)
preds2
plot(age, wage, xlim = agelims, cex = 0.5, col ="darkgrey")
lines(age.grid, preds2[,1], col = "blue", lwd = 2)
lines(age.grid, preds2[,2], col = "blue", lwd = 1, lty = 3)
lines(age.grid, preds2[,3], col = "blue", lwd = 1, lty = 3)
#Yeah that settles it the author made this operation more complicated than it should be. Both commands give rise to the same graphics. 

#We mentioned earlier that whether or not an othogonal set of basis functions is produced in the poly() function will not affect the model obtained in a meaningful way. What do we mean by this? The fitted values obtained in either case are identical.
preds3 <- predict(fit2, newdata = list(age = age.grid), se = TRUE)
max(abs(preds$fit - preds3$fit))

#In performing a polynomial regression we must decide on the degree of the polynomial to use. One way to do this is by using hypothesis tests. We now fit models ranging from linear to a degree-5 polynomial and seek to determine the simplest model which is sufficient to explain the relationship between wage and age. We use the anova() function, which performs an analysis of variance (ANOVA , using an F-test) in order to test the null hypothesis that a model M_1 is sufficient to explain the data against the alternative hypothesis that a more complex model M_2 is required. In order to use the anova() function, M_1 and M_2 must be nested models: the predictors in M_1 must be a subset of the predictors in M_2. In this case, we fit five different models and sequentially compare the simpler model to the more complex model. 
fit.1 <- lm(wage ~ age, data = Wage)
fit.2 <- lm(wage ~ poly(age,2), data = Wage)
fit.3 <- lm(wage ~ poly(age, 3), data = Wage)
fit.4 <- lm(wage ~ poly(age, 4), data = Wage)
fit.5 <- lm(wage ~ poly(age, 5), data = Wage)
anova(fit.1, fit.2, fit.3, fit.4, fit.5)

#the p-value comparing the linear model 1 to the quadratic model 2 is essentially zero, indicating that a linear fit is not sufficient. Similarly the p-value comparing the quadratic Model 2 to the cubic Model 3 is very low, so the quadratic fit is also insufficient. The p-value comparing the cubic and degree-4 polynomials, Model 3 and model 4, is approximately 5 percent while the degree-5 polynomial Model 5 seems unnecessary because its p-value is 0.37. Hence, either a cubic or a quartic polynomial appear to provide a reasonable fit to the data, but lower or higher order models are not justified. 

#In this case, instead of using the anova() function, we could have obtained these p-values more succinctly by exploiting the fact that poly() creates orthogonal polynomials. 
coef(summary(fit.5))
#notice that the p-values are the same, and in fact the square of the t-statistics are equal to the F-statistics from the anova() function; for example:
(-11.983)^2

#However, the ANOVA method works whether or not we used orthogonal polynomials; it also works when we have other terms in the model as well. For example, we can use anova() to compare these three models:
fit.1 <- lm(wage ~ education + age, data = Wage)
fit.2 <- lm(wage ~ education + poly(age, 2), data = Wage)
fit.3 <- lm(wage ~ education + poly(age,3), data = Wage)
anova(fit.1, fit.2, fit.3)

#Next we consider the task of predicting whether an individual earns more than 250,000 per year. We proceed much as before, except that first we create the appropriate response vector, and then apply the glm() function using family = "binomial" in order to fit a polynomial logistic regression model.
fit <- glm(I(wage > 250) ~ poly(age, 4), data = Wage, family = binomial)

#Note that we again use the wrapper I() to create this binary response variable on the fly. The expression wage>250 evaluates to a logical variable containing TRUEs and FALSEs, which glm() coerces to binary by setting the TRUEs to 1 and the FALSEs to 0. 

preds <- predict(fit, newdata = list(age = age.grid), se = TRUE)

#However, calculating the confidence intervals is slightly more involved than in the linear regression case. The default prediction type for a glm() model is type = "link".
pfit <- exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit <- cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands <- exp(se.bands.logit)/(1+exp(se.bands.logit))

#Note that we could have directly computed the probabilities by selecting the type = "response" option in the predict() function.
preds <- predict(fit, newdata = list(age = age.grid), type = "response", se = TRUE)

#However, the corresponding confidence intervals would not have been sensible because we would end up with negative probabilities.

plot(age, I(wage>250), xlim = agelims, type = "n", ylim = c(0,0.2))
points(jitter(age), I((wage>250)/5), cex = 0.5, pch = "l", col = "darkgrey")# I found out that you need to /5 command for the upper points to be superimposed onto the graphic assembly. 
lines(age.grid, pfit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty =3)

#In order to fit a step function we use the cut() function.
table(cut(age, 4))
fit <- lm(wage ~ cut(age, 4), data = Wage)
coef(summary(fit))

#Here cut() automatically picked the cutpoints at 33.5, and 64.5 years of age. We could also have specified our own cutpoints directly using the breaks option. The function cut() returns an ordered categorical variable; the lm() function then creates a set of dummy variables for use in the regression. The age<33.5 category is left out, so the intercept coefficient of 94,160 can be interpreted as the average salary for those under 33.5 years of age, and the other coefficients can be interpreted as the average additional salary for those in the other age groups. 

preds <- predict(fit, newdata = list(age = age.grid), interval = "confidence", level = 0.95)
plot(x = age, y = wage, col = "darkgrey")
lines(x = age.grid, y = preds[,1], lty = 1, lwd = 2, col = "blue")
lines(x = age.grid, y = preds[,2], lty = 2, col = "blue")
lines(x = age.grid, y = preds[,3], lty = 2, col = "blue")

glm.fit <- glm(I(wage > 250) ~ cut(age, 4), data = Wage, family = binomial)
preds.glm <- predict(glm.fit, newdata = list(age = age.grid), se =TRUE)
pfit <- exp(preds.glm$fit)/(1+exp(preds.glm$fit))
se.bands.logit <- cbind(preds.glm$fit+2*preds.glm$se.fit, preds.glm$fit-2*preds.glm$se.fit)
se.bands <- exp(se.bands.logit)/(1+exp(se.bands.logit))

plot(age, I(wage>250), xlim = agelims, type = "n", ylim = c(0,0.2))
points(jitter(age), I((wage>250)/5), cex = 0.5, pch = "l", col = "darkgrey")
lines(x = age.grid, y = pfit, lty = 1, lwd = 2, col = "blue")
matlines(x = age.grid, y = se.bands, col ="blue", lwd = 1, lty = 3)# pretty cool. The author was right the operations are the same between the step function and polynomial function commands. Will need to remember this lesson later on through my studies. Also I can use this same techniques with the genetics datasets that Alexa gave the R group study material on. 

## 7.8.2 Splines:
#In order to fit regression splines in R, we use the splines library. In Section 7.4, we say that regression splines can be fit by contructing an appropriate matrix of basis functions. The bs() function generates the entire matrix of basis functions for splines with the specified set of knots. By default, cubic splines are produced. Fitting wage and age using a regression spline is simple:
library(splines)# Cool according to the concole the splines package is installed within the R concole by default. 
fit <- lm(wage ~ bs(age, knots= c(25,40,60)), data = Wage)# Cool the knot argument and the bs() function worked. 
pred <- predict(fit, newdata = list(age = age.grid), se = TRUE)
plot(age, wage, col = "gray")
lines(age.grid, pred$fit, lwd = 2)
lines(age.grid, pred$fit - 2*pred$se, lty = "dashed")
lines(age.grid, pred$fit + 2*pred$se, lty = "dashed")

#Here we have prespecified knots at ages 25, 40, and 60. this produces a spline with six basis functions. (Recall that a cubic spline with three knots has seven degrees of freedom are used up by an intercept, plus six basis functions.) We could also use the df option to produce a spline with knots at uniform qunatiles of the data. 

dim(bs(age, knots = c(25, 40, 60)))
dim(bs(age, df = 6))
attr(bs(age, df = 6), "knots")

#In the case R chooses knots at ages 33.8, 42.0, and 51.0, which correspond to the 20the, 50th, and 75th percentiles of age. The function bs() also has a degree argument, so we can fit splines of any degree, rather than the default of 3 (which yields a cubic spline).
#In order to instead fit a natural spline, we use the ns() function. Here we fit a natural spline with four degrees of freedom. 
plot(age, wage, col = "gray")
fit2 <- lm(wage ~ ns(age, df = 4), data = Wage)
pred2 <- predict(fit2, newdata = list(age = age.grid), se = TRUE)
lines(age.grid, pred2$fit, col = "red", lwd = 2)# This line depicts the natural spline (remember that the natural spline has less variance due to the fact that the formula assembly has boundary constraints). 
lines(age.grid, pred$fit, col = "blue", lwd = 2)# this line illustrates the normal cubic spline.

#As with the bs() function, we could instead specify the knots directly using the knots option.
#In order to fit a smoothing spline, we use the smooth.spline() function. 
plot(age, wage, xlim = agelims, cex = 0.5, col = "darkgrey")
title("Smooth.spline(age, wage, df = 16)")
fit <- smooth.spline(age, wage, df = 16)
fit2 <- smooth.spline(age, wage, cv = TRUE)
fit2$df
lines(fit, col = "red", lwd = 2)
lines(fit2, col = "blue", lwd = 2)

#Notice that in the first call to smooth.spline(), we specified df = 16. The function then determines which value of lambda leads to 16 degrees of freedom. In the second call to smooth.spline(), we select the smoothness level by cross-validation; this results in a value of lambda that yields 6.8 degrees of freedom. 

#In order to perform local regression, we use the loess() function.
plot(age, wage, xlim = agelims, cex = 0.5, col = "darkgrey")
title("Local Regression")
fit <- loess(wage ~ age, span = 0.2, data = Wage)
fit2 <- loess(wage ~ age, span = 0.5, data = Wage)
lines(age.grid, predict(fit, data.frame(age = age.grid)), col = "red", lwd = 2)
pred2 <- predict(fit2, newdata = data.frame(age = age.grid))
lines(x = age.grid[-c(63,62)], y =pred2$y, col = "blue", lwd = 2)# weirdly enough the loess() span = 0.5 model eliminated the 63 and 62 index position in the age.grid vector. Will need to see what cased this. 
legend("topright", legend = c("span = 0.2", "span = 0.5"), col = c("red", "blue"), lty = 1, lwd = 2, cex = 0.8)

#Here we have performed local linear regression using spans of 0.2 and 0.5: that is, each neightborhood consists of 20 percent or 50 percent of the observations. The larger the span, the smoother the fit. The locfit library can also be used for fitting local regression models in R. 

##7.8.2 GAMs:
#We now fit a GAM to predict wage using natural spline functions of year and age, treating education as a qualitative predictor. Since this is just a big linear regression model using an appropriate choice of basis functions, we can simply do this using the lm() function.
gam1 <- lm(wage ~ ns(year, 4) + ns(age, 5) + education, data = Wage) 

#We now fit the model using smoothing splines rather than natural splines. In order to fit more general sorts of GAMs, using smoothing splines or other components that cannot be expressed in terms of basis functions and then fit using least squares regression, we will need to use the gam library in R.

#the s() function, which is part of the gam library, is used to indicate that we sould like to use a smoothing spline. We specify that the function of year should have 4 degrees of freedom, and that the function of age will have 5 degrees of freedom. Since education is qualitative, we leave it as is, and it is converted into four dummy variables. We use the gam() function in order to fit a GAM using these components. All of the terms in (7.16) are fit simultaneously, taking each other into account to explain the response. 
library(gam)
gam.m3 <- gam(wage ~ s(year, 4)+ s(age, 5) + education, data = Wage)
par(mfrow = c(1,3))
plot(gam.m3, se = TRUE, col = "blue")
#the generic plot() function recognizes that gam2 is an object of class gam and invokes the appropriate plot.gam() method. Conveniently, even though gam1 is not of class gam but rather of class lm, we can still use plot.gam() on it. 

plot.Gam(gam1, se = TRUE)# the plot.gam() function in the book was changed to plot.Gam() in the current R iteration.

#In these plots, the function of year looks rather linear. We can perform a series of ANOVA tests in order to determine which of these three models is best: a GAM that excludes year (M_1), a GAM that uses a linear function of year (M_2), or a GAM that uses a spline function of year (M_3).

gam.m1 <- gam(wage ~ s(age,5) + education, data = Wage)
gam.m2 <- gam(wage ~ year + s(age,5) + education, data = Wage)
anova(gam.m1, gam.m2, gam.m3)

#We find that there is compelling evidence that a GAM with a linear function of year is better than a GAM that does not include year at all (p-value = 0.00014). However, there is no evidence that a non-linear function of year is needed (p-value = 0.349). In other words, based on the results of this ANOVA, M_2 is preferred. 

summary(gam.m3)

#The p-vaue for year and age correspond to a null hypothesis of a linear relationship versus the alternative of a non-linear relationship. The large p-value for year reinforces our conclusion from the ANOVA test that a linear function is adequate for this term. However, there is very clear evidence that a non-linear term is required for age. 

#We can make predictions from gam objects, just like from lm objections using the predict() method for the class gam. Here we make predictions on the training set. 
preds <- predict(gam.m2, newdata = Wage)

#We can also use local regression fits as building blocks in a GAM, using the lo() function. 
gam.lo <- gam(wage ~ s(year, df = 4) + lo(age,span = 0.7) + education, data = Wage)
plot.Gam(gam.lo, se = TRUE, col = "green")

#Here we have used local regression for the age term, with the span of 0.7. We can also use the lo() function to create interactions before calling the gam() function. 
gam.lo.i <- gam(wage ~ lo(year, age, span = 0.5) + education, data = Wage)

#the following line fits a two-term model, in which the first term is an interaction between year and age, fit by a local regression surface. We can plot the resulting two-dimensional surface if we first install the akima package.
library(akima)
plot(gam.lo.i)

#In order to fit a logistic regression GAM, we once again use the I() function in constructing the binary response variable, and set family = binomial.
gam.lr <- gam(I(wage > 250) ~ year + s(age, df = 5) + education, family = binomial, data = Wage)
par(mfrow = c(1,3))
plot(gam.lr, se = T, col = "green")
#It is easy to see that there are no high earners in the <HS category:
table(education, I(wage>250))

#Hence, we fit a logistic regression GAM using all but this category. This provides more sensible results:
gam.lr.s <- gam(I(wage>250) ~ year + s(age, df = 5)+education, family = binomial, data = Wage, subset = (education != "1. < HS Grad"))
plot(gam.lr.s, se = T, col = "green")

##7.9 Exercises:
##conceptual:
#1. Again just like in the conceptual exercises in chapter 6 this question is way too advanced for they rusty mathematical skills. I simply have never study single variable calculus in my life. 
#the github account of anaboughi has all the answers for this question (the only problem is that it is written in a different computer language.)

#2.) The following answers where obtained from asaboughi. I some what understand the theory involved but sadly I still have a long way to go before I feel comfortable answering these questions.

#(a) g(x) = k because RSS term is ignroed and g(x) = k would minimize the area under the curve of g^(0).

#(b) g(x) \ alpha x^2. g(x) would be quadratic to minimize the area under the curve of its first derivative.

#(c) g(x) \ alpha x^3. g(x) would be cubic to minimize the area under the curve of its second derivative. 

#(d) g(x) \ alpha x^4. g(x) would be quartic to minimize the area under the curve of its third derivative.

#(e) the penalty term no longer matters. This is the formula for linear regression, to choose g based on minimizing RSS.

#3.) 
#Asadoughi's solution:
x = -2:2 
y = 1 + x + -2 * (x-1)^2 * I(x>1)# so in other words, for this function to work in R don't include >= or <= within the formula assembly.
plot(x, y)

#4.)
x <- -2:2 
y <- 1 + 1 * (I(0 < x < 2) - (x - 1) * I(1 < x < 2)) + 3 * ((x-3)*I(3 < x < 4)+I(4 < x < 5))
#this seems to not be the correct syntax for this problem.

#Asadoughi's solution:
x <- -2:2 
y <- c(1 + 0 + 0,
		1 + 0 + 0,
		1 + 1 + 0,
		1 + (1-0) + 0,
		1 + (1-1) + 0
		)
plot(x,y)

#5.) Again sadly this question is too far above my understanding of mathematics:
#Asadoughi's solution:
#(a) We'd expect g_hat_2 to have the smaller traing RSS because it will be a higher order polynomial due to the order of the derivative penalty function.

#(b) We expect g_hat_1 to have the smaller test RSS because h_hat_2 could overfit with the extra degree of freedom.

#(c) Asadoughi says that it's a trick question since when lambda = 0 the model will simply over fit the training data. And hence g_hat_1 and g_hat_2 will equal each other.

##Applied:
#6.)
#(a)
#first cross validation technique set cross validation:
library(leaps)
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Wage), rep = TRUE)
test <- (!train)
wage.fit <- regsubsets(wage ~ poly(age, 5), data = Wage[train,], nvmax = 5)
test.mat <- model.matrix(wage~poly(age, 5), data = Wage[test,])
val.errors <- rep(NA, 5)
for(i in 1:5){
	coefi <- coef(wage.fit, id = i)
	pred <- test.mat[,names(coefi)]%*%coefi
	val.errors[i] <- mean((Wage$wage[test]-pred)^2)
}
val.errors
which.min(val.errors)
coef(wage.fit, 2)
#The best subset cross validation technique picked poly(age, 5)2 as the best variable, as it has the lowest sum squared error rate of the other iterations. 

#LOOCV method:
library(boot)
lcv.fit <- glm(wage ~ poly(age, 5), data = Wage)
cv.err <- cv.glm(Wage, lcv.fit)
cv.err$call
names(cv.err)
# I really don't know how I can interpret this result. I guess that I will have to do this manually or just simply use k-fold cross validation.

#K-fold cross validation:
predict.regsubsets <- function(object, newdata, id, ...){
	form <- as.formula(object$call[[2]])
	mat <- model.matrix(form, newdata)
	coefi <- coef(object, id = id)
	xvars <- names(coefi)
	mat[,xvars]%*%coefi
}
#Since resubsets() doesn't have a predict function the author was forced to create one himself. the function can be found on page 250.
k <- 5
set.seed(1)
folds <- sample(1:k, nrow(Wage), replace = TRUE)
cv.errors <- matrix(NA, k, 5, dimnames = list(NULL, paste(1:5)))
for (j in 1:k){
	wage.fit <- regsubsets(wage ~ poly(age, 5), data = Wage[folds!=j,], nvmax = 5)
	for ( i in 1:5){
		pred <- predict(wage.fit, Wage[folds == j,], id = i)
		cv.errors[j, i] = mean((Wage$wage[folds == j]-pred)^2)
	}
}
cv.errors

mean.cv.errors <- apply(cv.errors, 2, mean)
which.min(mean.cv.errors) #this cross validation picked poly(age, 5)2 as well. Hence a degree two transformation of variable age is more than sufficient. 

fit1 <- lm(wage ~ poly(age, 1), data = Wage)
fit2 <- lm(wage ~ poly(age, 2), data = Wage)
fit3 <- lm(wage ~ poly(age, 3), data = Wage)
fit4 <- lm(wage ~ poly(age,4), data = Wage)
fit5 <- lm(wage ~ poly(age, 5), data = Wage)
anova(fit1, fit2, fit3, fit4,fit5)# Interestingly the anova method picked the three degree method instead of the second degree method. The reason for this is that the cross validation methods find the model with the least sum squared error rate regardless of the p-value while the anova method focuses on the p-value only. 

plot(age, wage, col = "darkgrey")
age.grid <- seq(18, 80)
pred <- predict(fit2, newdata = data.frame(age = age.grid), interval = "confidence", level = 0.95)
pred2 <- predict(fit3, newdata = data.frame(age = age.grid), interval = "confidence", level = 0.95)
lines(x = age.grid, y = pred[,1], lty = 1, col = "blue")
lines(x = age.grid, y = pred2[,1], lty = 1, col = "red")
legend("topright",legend = c("CV degree 2", "ANOVA degree 3"), col = c("blue", "red"), lty = c(1,1))
#Most likely the problem is that set subset cross validation doesn't have the right number of observations to create a model that can correctly predict future response variables. And in addition the k-folds method most likely needs more folds to create a better model. 

#(b) Step function method 
fit <- lm(wage ~ cut(age, 4), data = Wage)

#set cross validation:
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Wage), rep = TRUE)
test <- (!train)
wage.fit <- regsubsets(wage ~ cut(age, 15), data = Wage[train,], nvmax = 14)
test.mat <- model.matrix(wage~cut(age, 15), data = Wage[test,])
val.errors <- rep(NA, 14)
for(i in 1:14){
	coefi <- coef(wage.fit, id = i)
	pred <- test.mat[,names(coefi)]%*%coefi
	val.errors[i] <- mean((Wage$wage[test]-pred)^2)
}
val.errors
which.min(val.errors)
coef(wage.fit, 10)

# Will need to see if this answer is correct. 

#I was wrong for both questions:
#asadoughi's solutions:
set.seed(1) 
library(boot)
all.deltas <- rep(NA, 10)
for(i in 1:10){
	glm.fit <- glm(wage~poly(age, i), data = Wage)
	all.deltas[i] <- cv.glm(Wage, glm.fit, K = 10)$delta[2]
}
plot(1:10, all.deltas, xlab = "Degree", ylab = "CV error", type = "l", pch = 20, lwd = 2, ylim = c(1590, 1700))
min.point <- min(all.deltas)
sd.points <- sd(all.deltas)
abline(h = min.point+ 0.2 * sd.points, col = "red", lty = "dashed")
abline(h = min.point - 0.2 * sd.points, col = "red", lty = "dashed")

#(b)
set.seed(1)
all.deltas <- rep(NA, 10)
for(i in 2:10){
	Wage$age.cut <- cut(Wage$age, i)
	glm.fit <- glm(wage~age.cut, data = Wage)
	all.deltas[i] <- cv.glm(Wage, glm.fit, K = 10)$delta[2]
}
plot(2:10, all.deltas[-1])
agelims <- range(Wage$age)
lm.fit <- glm(wage ~ cut(age, 8), data = Wage)
agelims <- range(Wages$age)
age.grid <- seq(from = agelims[1], to = agelims[2])
lm.pred <- predict(lm.fit, data.frame(age = age.grid))
plot(wage~age, data = Wage, col = "darkgrey")
lines(age.grid, lm.pred, col = "red", lwd = 2)

#7.) 
names(Wage)
str(Wage)
levels(Wage$jobclass)
levels(Wage$maritl)
pairs(Wage)
ggplot(Wage, aes(x = maritl, y = wage)) + geom_boxplot()# married people have a higher median income than the other categories. 
ggplot(Wage, aes(x = jobclass, y = wage)) + geom_boxplot()# And of course information workers make a higher median income than industrial (but this difference is only a couple thousand dollars a year). 
ggplot(Wage, aes(x = race, y = wage)) + geom_boxplot()

model.fit <- lm(wage ~ jobclass + race + maritl, data = Wage)
summary(model.fit)# Will need to look into these components further.
model.fit2 <- lm(wage ~ jobclass, data = Wage)
summary(model.fit2)# There seems to be a high statistical significance for these variables. Will need to mess around with this variable further.

model.fit3 <- lm(wage ~ maritl, data = Wage)
summary(model.fit3)
#Just as I suspected without other numeric variables I can't use splines and GAMs. I'm only stuck using boxplots and creating categorical regression models. 

#8.)
str(Auto)
pairs(Auto)

model.fit1 <- lm(mpg ~ displacement + acceleration + weight, data = Auto)
summary(model.fit1)
# Even though the p-value seems very high with regards to the correlation between mpg and displacement, let's play around with the line of the model a little bit through quadratic transformation. 
plot(x = Auto$displacement, y = Auto$mpg)
model.dis <- lm(mpg ~ displacement + I(displacement^2), data = Auto)
summary(model.dis)# the significance of displacement increased through squaring the variable. Let's see if the variable continues to grow in significance through increased degrees.
model.dis2 <- lm(mpg ~ poly(displacement, 5), data = Auto)
summary(model.dis2)# Displacement decreases in significance with increased degrees above 2.
preds <- predict(model.dis2, newdata = data.frame(displacement = seq(range(Auto$displacement)[1], range(Auto$displacement)[2])), interval = "confidence")
lines(x = seq(range(Auto$displacement)[1], range(Auto$displacement)[2]), y = preds[,1], col = "blue", lty = 1)
lines(x = seq(range(Auto$displacement)[1], range(Auto$displacement)[2]), y = preds[,2], lty = 2)
lines(x = seq(range(Auto$displacement)[1], range(Auto$displacement)[2]), y = preds[,3], lty = 2)

model.fit2 <- lm(mpg ~ poly(displacement, 2) + acceleration + weight, data = Auto)
summary(model.fit2)# let's see if we can increase the statistical significance of acceleration.

plot(x = Auto$acceleration, y =Auto$mpg)
# I wonder if step function transformation might be a good fit for this variable.
summary(lm(mpg ~ cut(acceleration, 5), data = Auto))# Awesome step transformation really does agree with this variable according to the p-values of each of the 5 levels. Let's see what the optimum number of cuts is through k-fold cross validation.
set.seed(1)
all.deltas <- rep(NA, 10)
for(i in 2:10){
	Auto$acceleration.cut <- cut(Auto$acceleration, i)
	glm.fit <- glm(mpg ~ acceleration.cut, data = Auto)
	all.deltas[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[2]
}
which.min(all.deltas)# It seems like three breaks is the best method in decreasing the training MSE rate. 
summary(lm(mpg ~ cut(acceleration, 3), data = Auto))# This model has better p-values than the 5 alternative. 

model.fit3 <- lm(mpg ~ poly(displacement, 2) + cut(acceleration, 3) + weight, data = Auto)
summary(model.fit3)

#To finish up my exploration with this dataset I will play around with the connection with mpg and year (the data in which the car was manufactured).
summary(lm(mpg ~ year, data = Auto))
plot(x = Auto$year, y = Auto$mpg)
set.seed(1)
all.deltas <- rep(NA, 10)
for(i in 1:10){
	glm.fit <- glm(mpg~ns(year, df = i), data = Auto)
	all.deltas[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[2]
}
which.min(all.deltas)#the best lambda value to use in this case is 9.
model.year <- lm(mpg~ns(year, df = 4), data = Auto)
model.pre <- predict(model.year, newdata = data.frame(year = seq(min(Auto$year), max(Auto$year))), interval = "confidence", level = 0.95)
lines(x = seq(min(Auto$year), max(Auto$year)), y = model.pre[,1])
summary(model.year)
model.gam <- gam(mpg ~ poly(displacement, 2) + cut(acceleration, 3) + ns(year, df = 9), data = Auto)
par(mfrow = c(1,3))
plot(model.gam, se = TRUE, col = "blue")

#9.) 
library(MASS)
str(Boston)
??Boston
#formula: nox ~ dis (which means that distance away from employment centers is correlated to nitrogen oxide concentration)
		# Null states that there is no correlation 
		# Alternative hypothesis states that there is a correlation.
		
summary(Boston$nox)
summary(Boston$dis)

#(a) 
boston.fit1 <- lm(nox ~ poly(dis, 3), data = Boston)
summary(boston.fit1)# P-value seems to be very low with the cubic transformation. Meaning that the cubic transformation is indeed statistically significant.
detach(Wage)
attach(Boston)
plot(x = dis, y = nox)
dis.grid <- seq(min(dis), max(dis), by = 0.25)
preds <- predict(boston.fit1, newdata = data.frame(dis = dis.grid), interval = "confidence", level = 0.95)
lines(x = dis.grid, y = preds[,1], col = "blue", lty = 1, lwd = 2)# I have to say that this is a pretty good fit. 
lines(x= dis.grid, y = preds[,2], col = "blue", lty = 2, lwd = 1)
lines(x= dis.grid, y = preds[,3], col ="blue", lty = 2, lwd = 1)# The confidence intervals seem to be widening with increased distance away from the employment centers. this must be because the model might be overfitting the data a little bit (but again there is no evidence to support this thought).

#(b) 
plot(dis, nox)
all.rss <- rep(NA, 10)
for(i in 1:10){
	boston.fit1 <- lm(nox ~ poly(dis, i), data = Boston)
	all.rss[i] <- sum(boston.fit1$residuals^2)
	preds <- predict(boston.fit1, newdata = data.frame(dis = dis.grid), interval = "confidence", level = 0.95)[,1]
	lines(x = dis.grid, y = preds, lty = 1, col = i)
	
}
legend("topright", legend = c(1:10), lty = rep(1,times = 10), col = c(1:10))
all.rss
#As expected, train RSS monotonically decreases with degree of polynomial, since the model is overfitting the training dataset.

#(c)
set.seed(1)
all.deltas <- rep(NA, 10)
for(i in 1:10){
	glm.fit <- glm(nox~poly(dis, i), data = Boston)
	all.deltas[i] <- cv.glm(Boston, glm.fit, K = 10)$delta[2]
}
which.min(all.deltas)
#According to the k fold cross validation model the best degree transformation was calculated at four. 
plot(x = 1:10, y = all.deltas, type = "l")# As you can see through this graphic representation, the four degree mark has the lowest RSS value.
plot(dis, nox)

#(d) (misunderstanding of the question presented. Will use this answer for question e):
#(e)
set.seed(1)
for(i in 1:25){
model.spline <- lm(nox ~ bs(dis, df = i), data = Boston)
all.deltas[i] <- sum(model.spline$residuals^2)
}
which.min(all.deltas)# According to this cross validation call the best degrees of freedom value for this problem is 10. With that said, I will need to see how I can convert this value into knots. 
attr(bs(dis, df = 10), "knots")# The following command prints the number of knots that were used to answer this question. The reason why I didn't code for the knot intervals individually is because the df argument is the most straight forward way to calculate the number of knots needed and where they should be located. Will need to see if asaboughi has the same sentiment. 
plot(x = 1:25, y = all.deltas, type = "l")# As you can see the RSS values are quite volitial for the entirety of the degrees of freedom range, but just as the which.min() function result said, the best degree of freedom value is 10 (of course when one sets set.seed(1)).
dis.grid <- seq(from = min(dis), to = max(dis), by = 0.1)
pred <- rep(NA, times = length(dis.grid))
plot(x = dis, y = nox, pch = 16, col = "darkgrey")
for(i in 1:25){
	model.spline <- lm(nox ~ bs(dis, df = i), data = Boston)
	pred <- predict(model.spline, newdata = data.frame(dis = dis.grid), interval = "confidence", level = 0.95)[,1]
	lines(dis.grid, pred, col = i, lty = 1, lwd = 1)
}
legend("topright", legend = c(1:25), col = c(1:25), lwd = 1, lty = 1)
#According to the function bs() the degree of freedom was too small. Hence the for loop only graphed three variations. The problem is that I don't know which three variations that the console used.
#One can see that with increased degrees of freedom the model line because more jumpy. This means that, just like what the theory about splines states, that will increased df the number of splines increases as well thus making the model less and less linear. (remember that a regular spline withdf = 0 is a linear line).
summary(model.spline)

#(f) the following answer is from problem (e) 
set.seed(1)
all.deltas <- rep(NA, 10)
for(i in 1:10){
	glm.fit <- glm(nox~bs(dis, i), data = Boston)
	all.deltas[i] <- cv.glm(Boston, glm.fit, K = 10)$delta[2]
}
which.min(all.deltas)# the Answer again is 10 degrees of freedom for this problem. 

#For fun let's test this method out on natural splines and smooth splines. 
#natural splines:
set.seed(1)
for(i in 1:10){
	glm.fit <- glm(nox~ns(dis, df = i), data = Boston)
	all.deltas[i] <- cv.glm(Boston, glm.fit, K = 10)$delta[2]
}
which.min(all.deltas)# It has the same optimal degrees of freedom value 10 as the normal spline function (bs()). Will need to see if the confidence intervals are the same as well. 
lm.fit1 <- lm(nox~ns(dis, df = 10), data = Boston)# It seems that the df argument is so large that any argument that is used to assess confidence intervals and se values is degraded. Will need to see how this affects plotting ns() and bs() function later on in my studies.

#smooth splines:
smooth.fit <- smooth.spline(Boston$dis, Boston$nox, cv = TRUE)
smooth.fit$df# the degrees of freedom value that the smooth.spline() function proposes was 15.43. 

#10.) 
#(a) 
dim(College)
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(College), rep = TRUE)
test <- (!train)
regfit.fwd <- regsubsets(Outstate ~ .,nvmax = 17, method = "forward", data = College[train,])
summary(regfit.fwd)
# According to this model, the best variables are Private, Expend, Room.board, perc.alumni, PhD, Grad.Rate, Personal, and Top25perc.
#this is 8 variables in all. I can decrease the variables a little more but I believe that such a step will decrease the accuracy of the fit. 
college.fit <- gam(Outstate ~ Private + Expend + Room.Board + perc.alumni + PhD + Grad.Rate + Personal + Top10perc, data = College[train,])
college.var <- College[,c("Outstate", "Private", "Expend","Room.Board","perc.alumni","PhD","Grad.Rate","Personal","Top10perc")]
pairs(college.var)
summary(college.fit)
detach(Boston)
attach(College)
plot(Outstate, Expend)# this relationship looks linear, but I believe that a small spline transformation might fix the fit a little more. the questio is should I use a natural spline or a smooth spline.
ggplot(College, aes(x = Private, y = Outstate)) + geom_boxplot()
plot(x = PhD, y = Outstate)# Spline function might work with this variable as well.
plot(x = Top10perc, y = Outstate)# This looks like a step function might be the best option for this variable 
plot(x = Personal, y = Outstate)# Regardless of the fact that the variable's p-value is still well under the 0.05 significance level cut off. The points seem to be overly concentrated in a particular region of the plot. This might bring about volatility (regaring the confidence intervals) in the lower values.
plot(x = perc.alumni, y = Outstate)# Step function might be the right choice for this variable 
plot(Grad.Rate, Outstate)# step function as well since a college can't have over 100 percent graduation rate. And any graduation rate that goes over 100 percent should have the same Out of State tuition level as that of colleges with a 100 percent graduation rate.
plot(x = Room.Board, y = Outstate)# I'll say that I will use a polynomial transformation for this variable just because I haven't used one yet in this exercise.

#embarrassingly I forgot to include the c_p, bic, and adjusted R^2 statistics within my answer (which are used to determine the best number of variables that I should include within my model).
regfit.sum <- summary(regfit.fwd)
par(mfrow = c(1,3))
plot(regfit.sum$cp, xlab = "number of variables", ylab = "C_p value", type = "l")
plot(regfit.sum$adjr2, xlab = "number of variables", ylab = "Adjusted R squared", type = "l")
plot(regfit.sum$bic, xlab = "number of variables", ylab = "BIC value")
which.min(regfit.sum$cp)# 14 variables
which.max(regfit.sum$adjr2)# 14 variables
which.min(regfit.sum$bic)# 9 variables

#Asadoughi's solution:
train <- sample(length(Outstate), length(Outstate)/2)
test <- -train
College.train <- College[train,]
library(gam)
gam.fit <- gam(Outstate ~ Private + s(Room.Board, df = 2) + s(PhD, df = 2) + s(perc.alumni, df = 2) + s(Expend, df = 5) + s(Grad.Rate, df = 2), data = College.train)
par(mfrow = c(2,3))
plot(gam.fit, se = T, col = "blue")

#(c) 
gam.pred <- predict(gam.fit, College[test,])
gam.err <- mean((College$Outstate[test] - gam.pred)^2)
gam.err
gam.tss <- mean((College$Outstate - mean(College$Outstate[test]))^2)
test.rss <- 1 - gam.err / gam.tss
test.rss

#(d)
summary(gam.fit)

#11.) Asadoughi's solution 
#(a)
set.seed(1)
X1 <- rnorm(100)
X2 <- rnorm(100)
eps <- rnorm(100, sd = 0.1)
Y <- -2.1 + 1.3 * X1 + 0.54 * X2 + eps

#(b)
beta0 <- rep(NA, 1000)
beta1 <- rep(NA, 1000)
beta2 <- rep(NA, 1000)
beta1[1] <- 10

#(c,d,e)
for(i in 1:1000){
	a = Y - beta1[i] * X1
	beta2[i] <- lm(a~X2)$coef[2]
	a <- Y - beta2[i] * X2
	lm.fit <- lm(a~X1)
	if (i < 1000){
		beta1[i+1] <- lm.fit$coef[2]
	}
	beta0[i] <- lm.fit$coef[1]
}
plot(1:1000, beta0, type = "l", xlab = "iteration", ylab = "betas", ylim = c(-2.2,1.6), col ="green")
lines(1:1000, beta1, col = "red")
lines(1:1000, beta2, col = "blue")

#(f)
lm.fit <- lm(Y ~ X1 + X2)
plot(1:1000, beta0, type = "l", xlab = "iteration", ylab = "betas", ylim = c(-2.2, 1.6), col = "green")
lines(1:1000, beta1, col = "red")
lines(1:1000, beta2, col = "blue")
abline(h = lm.fit$coef[1], lty = "dashed", lwd = 3, col = rgb(0,0,0, alpha = 0.4))
abline(h = lm.fit$coef[2], lty = "dashed", lwd = 3, col = rgb(0,0,0, alpha = 0.4))
abline(h = lm.fit$coef[3], lty = "dashed", lwd = 3, col = rgb(0,0,0, alpha = 0.4))
#Will need to research a little bit more about backfitting.

#12.) Asadoughi's solution:
set.seed(1)
p <- 100
n <- 1000
x <- matrix(ncol = p, nrow = n)
coefi <- rep(NA, p)
for(i in 1:p){
	x[,i] <- rnorm(n)
	coefi[i] <- rnorm(1) * 100
}
y <- x %*% coefi + rnorm(n)

beta <- rep(0, p)
max_iterations <- 1000
errors <- rep(NA, max_iterations + 1)
iter <- 2
errors[1] <- Inf 
errors[2] <- sum((y - x %*% beta)^2)
threshold <- 0.0001
while(iter < max_iterations && errors[iter-1] - errors[iter] > threshold){
	for(i in 1:p){
		a <- y - x %*% beta + beta[i] * x[,i]
		beta[i] <- lm(a~x[,i])$coef[2]
	}
	iter <- iter + 1 
	errors[iter] <- sum((y - x %*% beta)^2)
	print(c(iter-2, errors[iter-1], errors[iter]))
}
plot(1:11, errors[3:13])


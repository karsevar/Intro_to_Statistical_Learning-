### Chapter 5 Resampling Methods:
#Resampling methods are an indispensable tool in modern statistics. They involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model. For example, in order to estimate the variability of a linear regression fit, we can repeatedly draw different samples from the training data, fit a linear regression to each new sample, and then examine the extent to which the resulting fits differ. Such an approach may allow us to obtain informaiton that would not be available from fitting the model only once using the original training sample. 

#This chapter will most likely be about cross-validation and bootstrapping. Cross-validation can be used to estimate the test error associated with a given statistical learning method in order to evaluate its performance, or to select the appropriate level of flexibility. The process of evaluating a model's performance is known as model assessment, whereas the process of selecting the proper level of flexibility for a model is known as model selection. The bootstrap is used in several contexts, most commonly to provide a measure of accuracy of a parameter estimate or of a given statistical learning method. 

##5.1 Cross-validation:
#In this section, we instead consider a class of methods that estimate the test error rate by holding out a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations. (Wait a second, is that exactly what we just did with the exercises in the chapter 4 lab and problem sets?)

## 5.1.1 The validation set approach:
#Suppose that we would like to estimate the test error associated with fitting a particular statistical learning method on a set of observations. The validation set approach is a very simple strategy for this task. It involves randomly dividing the available set of observations into two parts, a training set and a validation set. The resulting validation set error rate --- typically assessed using MSE in the case of a quantitative response --- provides an estimate of the test error rate. 

##Answering whether the best model to predict for mpg is a linear interpretation of horsepower or a quadratic interpretation using the validation set approach. 
#We randomly split the 392 observations into two sets, a training set containing 196 of the data points, and a validation set containing the remaining 196 observations. The result of this experiment can be see in figure 5.2. As you can see the author created 10 different fitting lines for the quadratic transformation model. This technique is very much like those described in simulation for data science with r. All ten curves indicate that the model with a quadratic term has dramatically smaller validation set MSE than the model with only a linear term. Furthermore, all ten curves indicate that there is not much benefit in including cubic or higher-order polynomial terms in the model. But it is worth noting that each of the ten curves results in a different test MSE estimate for each of the ten regression models considered. And there is no consensus among the curves as to which model results in the smallest validation set MSE. Based on the variability among these curves, all that we can conclude with any confidence is that the linear fit is not adequate for this data. 
library(ISLR)
summary(lm(mpg ~ horsepower, Auto))# Will need to review how the author obtained the MSE from each model.
summary(lm(mpg ~ horsepower + I(horsepower^2), Auto))

#The validation set approach is conceptually simple and is easy to implement. But it has two potential drawbacks:
	#As is shown in the right hand panel of figure 5.2 (located on page 193), the validation estimate of the test error rate can be highly variabl, depending on precisely which observations are included in the training set and which observations are included in the validation set. (In other words, a training set with an outlier will tend to have completely different results from the other sampling sets. Thus this is where imputation comes into play).
	#In the validation approach, only a subset of the observations -- those that are included in the training set rather than in the validation set --- are used to fit the model. Since statistical methods tend to perform worse when trained on fewer observations, this suggests that the validation set error rate may tend to overestimate the test error rate for the model fit on the entire data set.
	
##5.1.2 Leave one out cross validation:
#Like the validation set approach, LOOCV involves splitting the set of observations into two parts. However, instead of creating two subsets of comparable size, a single observation (x_1, y_1) is used for the validation set, and the remaining observations {(x_2, y_2), ..., (x_n, y_n)} make up the training set. The statistical learning method is fit on the n - 1 training observations, and a prediction y_hat_1 is made for the excluded observation, using its value x_1. Since (x_1, y_1) was not used in the fitting process, MSE_1 = (y_1 - y_hat_1)^2 provides an approximately unbiased estimate for the test error. But even though MSE_1 is unbiased for the test error, it is a poor estimate because it is highly variable, since it is based upon a single observation (x_1, y_1). 

#We can repeat the procedure by selecting (x_2, y_2) for the validation data, training the statistical learning procedure on the n -1 observations {(x_1, y_1), (x_3, y_3), ..., (x_n, y_n)}, and computing MSE_1, ..., MSE_n. The LOOCV estimate for the test MSE is the average of these n test error estimates. The formula can be found on page 194.

#LOOCV has a couple of major advantages over the validation set approach. First, it has far less bias. In LOOCV, we repeatedly fit the statistical learning method using training sets that contain n - 1 observations, almost as many as are in the entire data set. This is in contrast to the validation set approach, in which the training set is typically around half the size of the original data set. Consequently, the LOOCV approach tends not to overestimate the test error rate as much as the validation set approach does. Second, in contrast to the validation approach which will yield different results when applied repeatedly due to randomness in the training / validation set splits, performing LOOCV multiple times will always yield the same results: there is no randomess in the training/validation set splits (due to only one observation per sampling is being taken away for validation).

#On the subject of LOOCV approach optimization, with least squares linear or polynomial regression, an amazing shortcut makes the cost of LOOCV the same as that of a single model fit. To see the least squares linear or polynomial equation look at page 195. Where y_hat_i is the ith fitted value from the original least squares fit, and h_i is the leverage defined on page 98. This is like the ordinary MSE, except the ith residual is divided by 1-h_i. The leverage lies between 1/n and 1, and reflects the amount that an observation influences its own fit. Hence the residuals for high-leverage points are inflated in this formula by exactly the right amount for this equality to hold. 

##5.1.3 k-fold cross-validation:
#This approach involves randomly dividing the set of observations into k groups, or folds, of approzimately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k - 1 folds. The mean squared error, MSE_1, is then computed on the observations in the held-out fold. This procedure is repeated k times; each time, a different group of observations is treated as a validation set. This process results in k estimates of the test error, MSE_1, MSE_2, ..., MSE_k. To see the k-fold CV estimate equation look at page 196.

#In practice, one typically performs k-fold CV in which k is set to equal n. In practice, one typically performs k-fold CV using k = 5 or k = 10. What is the advantage of using k = 5 or k = 10 rather than k = n? The most obvious advantage is computational. LOOCV requires fitting the statistical learning method n times. This has the potential to be computationally expensive (except for linear models fit by least squares, in which case the least squares linear or polynomial formula can be used). But cross validation is a very general approach that can be applied to almost any statistical learning method. The practicality of LOOCV is proportionate to the amount of n values a partcular dataset has. If n is too large the task of averaging all the MSE_n values will become too computationally expensive. But for the k-fold method this same situation will become very much an advantage and, in contrast, computationally inexpensive due to the computer only having to calculate k number of MSE values.

#Interesting figure 5.6 (on page 197 illustrates) that the k-fold method mirrors the LOOCV method in flexibility when ran on the Auto dataset.

#When we perform cross-validation, our goal might be to determine how well a given statistical learning procedure can be expected to perform on independent data; in this case, the actual estimate of the test MSE is of interest. But at other times we are interested only in the location of the minimum point in the estimated test MSE curve. This is because we might be performing cross-validation on a number of statistical learning methods, or on a single method using different levels of flexibility, in order to identify the method that results in the lowest test error. For this purpose, the location of the minimum point in the estimated test MSE is important, but the actual value of the estimated test MSE is not. 

##5.1.4 Bias-variance trade-off for k-folds cross-validation:
#Putting aside the computational advantages, the k-fold CV gives more accurate estimates of the test error rate than does LOOCV due to the bias-variance trade-off. 

#It is mentioned in Section 5.1.1 that the validation set approach can lead to overestimates of the test error rate, since in this approach the training set used to fit the statistical learning method contains only half the observations of the entire data set. Using this logic, it is not hard to see that LOOCV will give approximately unbiased estimates, which is almost as many as the number of observations in the full data set. And performing k-fold CV for, say, k = 5 or k = 10 will lead to an intermediate level of bias, since each training set contas (k - 1)n/k observations -- fewer than in the LOOCV approach, but substantially more than in the validation set approach. Therefore, from the perspective of bias reduction, it is clear that LOOCV is to be preferred to k-fold CV. 

#However, we know that bias is not the only source of concern in an estimating procedure; we must also consider the procedure's variance. It turns out that LOOCV has higher variance than does k-fold CV with k < n. Why is this the case? When we perform LOOCV, we are in effect averaging the outputs of n fitted models, each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other. In contrast, when we perform k-fold CV with k < n, we are averaging the outputs of k fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller. Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from k-fold CV.

#To summarize, there is a bias-variance trade-off associated with the choice of k in k-folds cross validation. Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10, as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance. 

##5.1.5 Cross-validation on classification problems:
#In this setting, cross-validation works just as described earlier in this chapter, except that rather than using MSE to quantify test error, we instead use the number of misclassified observations. For instance, in the classification setting, the LOOCV error rate takes the form: To see the LOOCV classification regression equation look at page 199.

#Err_i = I(y_i != y_hat_i). The k-fold CV error rate and validation set error rates are defined analogously. 

## 5.2 The bootstrap:
#The bootstrap is a widely applicable and extremely powerful statistical tool that can be used to quantify the uncertainty associated with a given estimator or statistical learning method. As a simple example, the bootstrap can be used to estimate the standard errors of the coefficients from a linear regression fit. The bootstrap approach allows us to use a computer to emulate the process of obtaining new sample sets, so that we can estimate the variability of sigma_hat without generating additional samples. Rather than repeatly obtaining independent data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set. 

#This apprach is illustrated in Figure 5.11 on a simple data set, which we call Z, that contains only n = 3 observations. We randomly select n observations from the data set in order to produce a bootstrap data set, Z*1. The sampling is performed with replacement, which means that the same observation can occur more than once in the boostrap data set. In this example, Z*1 contains the third observation. Note that if an observation is contained in Z*1, then both its X and Y values are included. We can use Z*1 to produce a new bootstrap estimate for sigma, which we call sigma_hat*1. This procedure is repeated B times for some large value of B, in order to produce B different bootstrap data sets, Z*1, Z*2, ..., Z*B , and B corresponding sigma estimates, sigma_hat*1, sigma_hat*2, ..., sigma_hat*B. We can compute the standard error of these bootstrap estimates using the formula: to see the formula look at page 204. 

#this serves as an estimate of the standard error of sigma_hat estimated from the original dataset. 

##5.3. Lab: Cross-validation and the bootstrap:
##5.3.1 The Validation set approach:
#We begin by using the sample() function to split the set of observations into two halves, by selectin a random subset fo 196 observations out of the original 362 observations. We refer to these observations as the training set. This operation is very much the same as the other operations used in the preceding chapters of this book and chapters 1 through 5 of the Machine Learning with R text. Notice how the sample() function was used in this case since, most likely the author isn't sure if the data set is randomly ordered. If random order is known then one can just simply split the data set through basic subsetting. 

library(ISLR)
set.seed(1)
train <- sample(392, 196)

#(Here we use a shortcut in the sample command). We then use th subset option in lm() fot fit a linear regression using only the observations corresponding to the training set. 
lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)# rmember that the subset argument is used primarily as a indexing argument. Which tells the function which index positions to pull the data from.

#We now use the predict() function to estimate the response for all 392 observations, and se use the mean() function to calculate the MSE of the 196 observations in the validation set. Note that the -train index below selects only the observations that are not in the training set. 
attach(Auto)
mean((mpg - predict(lm.fit,Auto))[-train]^2)# So this is how you find the Mean squared Error. Will need to remember this operation.
		#Most likely the formula looks like:
			#mean((Y_pop - Y_hat))[test_dataset]^2)
			
#Therefore, the estimated test MSE for the linear regression fit is 26.14. We can use the poly() function to estimate the test error for the polynomical and cubic regressions.
lm.fit2 <- lm(mpg ~ poly(horsepower,2), data = Auto, subset = train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)# The poly() function tells the lm() function to interpret the regression line as a squared polynomial. poly(dataset, degree of quadratic transformation).

lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)

#These error rates are 19.82 and 9.78, respectively. If we choose a different training set instead, then we will obtain somewhat different errors on the validation set. 
set.seed(2)
train <- sample(392, 196)
lm.fit <- lm(mpg~horsepower, subset = train)
mean((mpg-predict(lm.fit, Auto))[-train]^2)
#The response is 23.30 MSE.
lm.fit2 <- lm(mpg~poly(horsepower, 2), data = Auto, subset = train)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
#The response is 18.90
lm.fit3 <- lm(mpg ~ poly(horsepower, 3), data =Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
#The response is 19.25 MSE. So in other words, this model is a little worse than the quadratic transformation. 

#These results are consistent with our previous findings: a model that predicts mpg using a quadratic function of horsepower performs better than a model that involves only a linear function of horsepower, and there is little evidence in favor of a modle that uses a cubic function of horsepower. 

##5.3.2 Leave - one -out Cross-validation:
#In this lab, we will perform linear regression using the glm() function rather than the lm() function because the latter can be used together with cv.glm(). The cv.glm() function is part of the boot library. 
#It's important to remember that the glm() function is the same as the lm() function if you don't describe the family argument in the glm() function assembly. 
library(boot)
glm.fit <- glm(mpg ~ horsepower, data = Auto)
cv.err <- cv.glm(Auto, glm.fit)
cv.err$delta

#the cv.glm() function produces a list with several components. The two numbers in the delta vector contain the cross validation results. In this case the numbers are identical (up to two decimal places) and correspond to the LOOCV statistic given in (5.1). Below, we discuss a situation in which the two numbers differ. Our cross-validation estimate for the test error is approximately 24.23.

#We can repeat this procedure for increasingly complex polynomial fits. To automate the process, we use the for() function to initiate a for loop which iteratively fits polynomial regressions for polynomials of order i = 1 to i = 5, computes the associated cross-validation error, and stores it in the ith element of the vector cv.error. 
cv.error <- rep(0,5)
for (i in 1:5){
	glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
	cv.error[i] <- cv.glm(Auto,glm.fit)$delta[1]
}
cv.error 
#We see a sharp drop in the estimated test MSE between the linear and quadratic fits, but then no clear improvement from using the higher order polynomials. 

##5.3.3 k-Fold Cross-Validation
#The cv.glm() function can also be used to implement k-fold CV. Below we use k = 10, a common choice for k, on the Auto data set. We once again set a random seed and initialize a vector in which we will store the CV errors corresponding to the polynomial fits of orders one to ten. 
set.seed(17)
cv.error.10 <- rep(0,10)
for (i in 1:10){
	glm.fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
	cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1]
}
cv.error.10

#Notice that the computation time is much shorter than that of LOOCV (In principle, the computation time for LOOCV for a least squares linear model should be faster than for k-fold CV, due to the availability of the formula (5.2) for LOOCV; however, unfortunately the cv.glm() function does not make use of this formula.) We still see little evidence that using cubic or higher order polynomial terms leads to lower test error than simply using a quadratic fit. 

#We saw in Section 5.3.2 that the two numbers associated with delta are eseentially the same when LOOCV is performed. When we instead perform k-fold CV, then the two numbers associated with delta differ slightly. The first is the standard k-fold CV estimate, as in (5.3). The second is a bias-corrected version. On this dataset, the two estimates are very similar to each other. 

## 5.3.4 The bootstrap:
##Estimating the Accuracy of a Statistic of interest:
#One of the great advantages of the bootstrap approach is that it can be applied in almost all situations. No complicated mathematical calculations are required. Performing a bootstrap analysis in R entails only two steps. First, we must create a function that computes the statistic of interest. Second, we use the boot() function, which is part of the boot library, to perform the bootstrap by repeatedly sampling observations from the data set with replacement.

#To illustrate the use of the bootstrap on this data, we must first create a function, alpha.fn(), which takes as input the (X,Y) data as well as a vector indicating which observations should be used to estimate (I believe the symbol is actually alpha). The function then outputs the estimate for alpha based on the selected observations.
alpha.fn <- function(data, index){
	X <- data$X[index]
	Y <- data$Y[index]
	return((var(Y) - cov(X,Y))/(var(X) + var(Y)-2*cov(X,Y)))
} 

#this function returns, or outputs, an estimate for alpha based on applying (5.7) to the observations indexed by the argument index. For instance, the following command tells R to estimate alpha using all 100 observations.
#The auther uses the Porfolio data set in the ISLR package for this example.
alpha.fn(Portfolio, 1:100)
#The next command uses the sample() function to randomly select 100 observations from the range 1 to 100, with replacement. this is equivalent to constructing a new bootstrap data set and recomputing alpha_hat based on the new dataset.
dim(Portfolio)#It's important to keep in mind that Portfolio only has 100 total observations. Hence explaining why the first argument in sample is set to 100 only.
set.seed(1)
alpha.fn(Portfolio, sample(100,100, replace = T))
#The response is 0.596

#We can implement a bootstrap analysis by performing this command many times, recording all of the corresponding estimates for alpha, and computing the resulting standard deviation. However, the boot() function automates this approach. Below we produce R = 1,000 bootstrap estimates for alpha. 
boot(Portfolio, alpha.fn, R = 1000)
#The final output shows that using the original data, alpha_hat = 0.5758, and that the bootstrap estimate for SE(alpha_hat) is 0.0886. 

##Estimating the Accuracy of a linear Regression model:
#The bootstrap approach can be used to assess the variability of the coefficient estimates and predictions from a statistical learning method. Here we use the bootstrap approach in order to assess the variability of the estimates for beta_0 and beta_1, the intercept and slope terms for the linear regression model that uses horsepower to predict mpg in the Auto data set. We will compare the estimates obtained using the bootstrap to those obtained using the formulas for SE(beta_hat_o) and SE(beta_hat_1) described in Section 3.1.2. 

#We first create a simple function, boot.fn(), which takes in the Auto data set as well as a set of indices for the observations, and returns the intercept and slope estimates for the linear regression model. We then apply this function to the full set of 392 observations in order to compute the estimates of beta_0 and beta_1 on the entire data set using the usual linear regression coefficient estimate formulas from chapter 3. Note that we do not need the { and } at the beginning and end of the funciton because it is only one line long. 
boot.fn<- function(data, index)
return(coef(lm(mpg~horsepower, data = data, subset = index)))# No way this function actually worked with this syntax. I should remember this rule.
boot.fn(Auto, 1:392)

#The boot.fn() function can also be used in order to create boostrap estimates for the intercept and slope terms by randomly sampling from among the observations with replacement. Here we give two examples:
set.seed(1)
boot.fn(Auto, sample(392, 392, replace = T))
boot.fn(Auto, sample(392, 392, replace = T))

#Next, we use the boot() function to compute the standard errors of 1000 bootstrap estimates for the intercept and slope terms. 
boot(Auto, boot.fn, 1000)
#this indicates that the bootstrap estimate for SE(beta_hat_0) is 0.86, and that the bootstrap estimate for SE(beta_hat_1) is 0.0074. As discussed in Section 3.1.2, standard formulas can be used to compute the standard errors for the regression coefficients in a linear model. These can be obtained using the summary() function. 
summary(lm(mpg~horsepower, data =Auto))$coef
#the results for the standard error is 0.717 for the intercept and 0.0064 for the slope. Recall that the standard formulas given in Equation 3.8 on page 66 rely on certain assumptions. For example, they depend on the unknown parameter sigma^2, the noise variance. We then estimate sigma^2 using the RSS. Now although the formula for the standard errors do not rely on the linear model being correct, the estimate for sigma^2 does. We see in figure 3.8 on page 91 that there is a non-linear relationship in the data, and so the residuals from a linear fit will be inflated, so will sigma_hat^2. Secondly, the standard formulas assumed (somewhat unrealistically) that the x_i are fixed, and all the variability comes from the variation in the errors E_i. The bootstrap approach does not rely on any of these assumptions, and so it is likely giving a more accurate estimate of the standard errors of beta_hat_0 and beta_hat_1 than the summary() function. 

#Below we compute the bootstrap standard error estimates and the standard linear regression estimates that result from fitting the quadratic model to the data. Since this model provides a good fit to the data there is now a better correspondence between the bootstrap estimates and the standard estimates of SE(beta_hat_0), SE(beta_hat_1), and SE(beta_hat_2).
boot.fn <- function(data, index)
coefficients(lm(mpg~horsepower + I(horsepower^2), data = data, subset = index))
set.seed(1)
boot(Auto, boot.fn, 1000)

##5.4 Exercises:
##Conceptual:
#1.) This exercise is a little too much for my current ability. Will need to go back to this question once I obtain enough calculus knowledge in this particular area. The answer is perfectly written in the github repository of Anaboughi and yahwes. 

#2.) We will now derive the probability that a given observation is part of a bootstrap sample. Suppose that we obtain a bootstrap sample from a set of n observations.

#(a). I sadly don't know how to answer this conceptual question at the moment. Will need to go back to this question later on in my studies. 
	#Yahwes solution: Probability is equal to not selecting that one observation out of all observations: frac{n - 1}{n} (I sadly don't know how to read this final equation. Will need to look into what syntax he is using).
	
#(b). Yahwes solution: Because bootstrap uses replacement, the probability is the same as part a: frac{n-1}{n}

#(c) Yahwes solution: Probability of not selecting the jth observation is the same for each selection. After n selections, the probability selecting the jth observation is: frac{n-1}{n}^n = (n-\frac{1}{n})^n

#(d). Yahwes solution: 
1 - (1 - 1/5)^5 #Interesting will need to look into how he obtained such a simplistic answer.

#(e). Yahwes solution:
	#through looking at the preceding solution I can guess that the equation for finding the probability of not finding the jth observation in a particular bootstrap regime is:
		#1 - (1 - (1 / (out of the number of bootstrap iterations)))^ number of bootstrap iterations
1 - (1 - 1/100)^100

#(f) Yahwes solution:
1 - (1 - 1/10000) ^ 10000

#(g). 
n <- c(1:100000) 
jth <- c(1 - (1 - 1/n)^n)
head(jth)
plot(x = n, y = jth, ylab = "probability of jth observation", xlab = "n")# Will need to look into how to make this probability chart more readable. We can see though that the probability of jth observation being in the bootstrap decrease rapidly between 0 and 1 n values. Will need to rescale this chart to see the actual trend of the probability. 

#Yahwes solution: This is embarrassing I made this problem more complicated then it should have been.
plot(1:100000, 1 - (1-1/1:100000)^(1:100000))# cool though, Yahwes and I have the same plot solution.

#(h) 
store <- rep(NA, 10000)
for(i in 1:10000){
	store[i] <- sum(sample(1:100, rep = TRUE) == 4)>0
} 
mean(store)# the result is 0.6326 and sadly I don't know how to interpret this value at the moment. 
#Yahwes solution: The resulting fraction of 10000 bootstrap samples that have the 4th observation is clase to our predicted probability of 1 - (1-1/100)^100 = 63.4 percent.

#3.) K fold validation questions:
#(a). k-fold validation is implemented much in the same way as the k nearest neighbors algorithm and the naive bayes method in that the data set is slight into a number of random groups (that are in no why illustrative of the inherent trends of the underlying values, unlike k-NN and naive bayes) and resulting estimator values of each of the groups are collected and averaged into a single value. The perpose of this method is to make cross validation less variable than the average method of simply splitting the dataset into a training set and a test set (since such a method is detrimental if the amount of observations are limited) and less computationally expensive than the Leave out one Cross validation technique (LOOCV) method. 

#(b). (i). for the validation set approach, the k-fold method is more advantagous because the sampling is randomized into k number of groups, hence the analyst has full control on how many groups he wants to create in order to validate whether the values obtained from an estimator is true or false. In addition, the validation set approach (as stated in the comment before) suffers from the need to split the dataset into two equally sized groups (the test set and the training set). This split orientation can give rise to an overly optimistic training error rate with the use of the validation set approach. Though k-fold still has its set of biases, it's estimations are still more accurate than that of the validation set approach. 

#(ii) For the LOOCV approach, bias is greatly reduced through its implimentation at the expense of increased variance due to the fact that the method is ran n = i times {(x_2, y_2), ..., (x_n,y_n)} {(x_1, y_1), ..., (x_n, y_n)} meaning that the test set is equal to n. This is perfect for limiting bias but the increased variance of n testing dataset will only exagerate the estimator of the underlying dataset. Hence k-fold is the perfect middle ground between too much variance and too much bias "since each training set contains (k-1)n/k observations -- fewer than in the LOOCV approach, but substantially more than in the validation set approach. Therefore, from the perspective of bias reduction, it is clear that LOOCV is to be preferred to k-fold CV" (page 183). But from a variance reduction perspective k-fold CV is more advantagous than LOOCV.

# 4.) Yahwes solution: We can use the bootstrap method to sample with replacement from our dataset and estimate Y"s from each sample. With the results of different predicted Y values, we can then estimate the standard deviation of our prediction.

##Applied:
#5.) (a) 
library(ISLR)
default.glm <- glm(default ~ income + balance, family = binomial, data = Default)
summary(default.glm)# All of the p-values look respectable for the two variables in the logistical regression model. 

#(b) (i) 
dim(Default)# It seems that the number of observations is 10000 and the number of variables is 4. The dataset will be split fifty percent training and fifty percent test. Make sure to remember that the sampling should be random.
set.seed(1)
index <- sample(10000, 5000)
default.train <- Default[index,]
default.test <- Default[-index,]

#(ii) 
default.glm <- glm(default ~ balance + income, data = default.train, family = binomial)

#(iii)
default.probs <- predict(default.glm, default.test, type = "response")
default.pred <- rep("No", 5000)
default.pred[default.probs > 0.5] <- "Yes"
table(default.pred, default.test$default)
mean(default.pred == default.test$default)# The model has a correct prediction rate of 97.14 percent which means that the error rate is:
1 - mean(default.pred == default.test$default)# 2.86 percent (which is very good). 

#(iv) 
mean(default.pred != default.test$default)# Very good the test error rate for this command is very much the same as the value in the operation above. It seems that the validation set error is actually this operation. Will need to remember this. 

#(c) 
#Attempt 1:
val.set <- function(data, var1, var2, resp, sam){
	index <- sample(nrow(data), sample)
	train <- rep(NA, times = index)
	test <- rep(NA, times = index)
	default.probs <- rep(NA, times = index)
	default.pred <- rep("No", times = index)
	error <- rep(NA, times = 3)
	for (i in 1:3){
		set.seed(i)
		train <- data[index,]
		test <- data[-train,]
		glm.fit <- glm(resp ~ var1 + var2, data, family = binomial) 
		default.probs <- predict(glm.fit, test, type = "response")
		default.pred[default.probs > 0.5] <- "Yes"
		error[i] <- mean(default.pred != test$resp)
	}
	return(error)
}

val.set(data = Default, var1 = income, var2 = balance, resp = default, sam = 5000)# I can't seem to get this function to work will need to see what the problem is later on in my studies on this matter. 

#attempt 2:
set.seed(2)
index <- sample(10000, 5000)
default.train <- Default[index,]
default.test <- Default[-index,] 
default.glm <- glm(default ~ balance + income, data = default.train, family = binomial)
default.probs <- predict(default.glm, default.test, type = "response")
default.pred <- rep("No", 5000)
default.pred[default.probs > 0.5] <- "Yes"
table(default.pred, default.test$default)
error1 <- mean(default.pred != default.test$default)

set.seed(3)
index <- sample(10000, 5000)
default.train <- Default[index,]
default.test <- Default[-index,] 
default.glm <- glm(default ~ balance + income, data = default.train, family = binomial)
default.probs <- predict(default.glm, default.test, type = "response")
default.pred <- rep("No", 5000)
default.pred[default.probs > 0.5] <- "Yes"
table(default.pred, default.test$default)
error2 <- mean(default.pred != default.test$default)

set.seed(4)
index <- sample(10000, 5000)
default.train <- Default[index,]
default.test <- Default[-index,] 
default.glm <- glm(default ~ balance + income, data = default.train, family = binomial)
default.probs <- predict(default.glm, default.test, type = "response")
default.pred <- rep("No", 5000)
default.pred[default.probs > 0.5] <- "Yes"
table(default.pred, default.test$default)
error3 <- mean(default.pred != default.test$default)
error.list1 <- list(error1, error2, error3)
error.list1
#According to this output the error rates are 2.76 percent, 2.48 percent, and 2.62 percent respectively. Which means that there is only a little bit of variance, but this variance is all accounted for in the model. 

#(d)
set.seed(2)
index <- sample(10000, 5000)
train <- Default[index, ]
test <- Default[-index,]
glm.fit2 <- glm(default ~ income + balance + student, data = train, family = binomial)
glm.probs <- predict(glm.fit2, test, type = "response")
glm.pred <- rep("No", 5000)
glm.pred[glm.probs > 0.5] <- "Yes" # The posterior probability wasn't disclosed in the questions description hence I'm just assuming that the threshold is set to greater than 0.5. 
error <- mean(glm.pred != test$default)

set.seed(3)
index <- sample(10000, 5000)
train <- Default[index, ]
test <- Default[-index,]
glm.fit2 <- glm(default ~ income + balance + student, data = Default, family = binomial)
glm.probs <- predict(glm.fit2, test, type = "response")
glm.pred <- rep("No", 5000)
glm.pred[glm.probs > 0.5] <- "Yes"
error2 <- mean(glm.pred != test$default)

set.seed(4)
index <- sample(10000, 5000)
train <- Default[index, ]
test <- Default[-index,]
glm.fit2 <- glm(default ~ income + balance + student, data = Default, family = binomial)
glm.probs <- predict(glm.fit2, test, type = "response")
glm.pred <- rep("No", 5000)
glm.pred[glm.probs > 0.5] <- "Yes"
error3 <- mean(glm.pred != test$default)
error.list2 <- list(error, error2, error3)
error.list2# Interestingly the error rate only increased by 0.001 with set.seed(2), stayed the same (0.0248) with set.seed(3), and for set.seed(4) the error rate increased by 0.0008. You can say that the error rate only slightly increased through the inclusion of the student dummy code variable. 

#6.) (a)
set.seed(1)
default.fit <- glm(default ~ income + balance, data = Default, family = binomial)
summary(default.fit)

#(b). Solution obtained from Yahwes:
set.seed(1)
boot.fn <- function(df, trainid){
	return(coef(glm(default~income + balance, data = df, family = binomial, subset = trainid)))
}
boot.fn(Default, 1:nrow(Default))# Now I understand the trainid is for the subset argument in the glm() function assembly. The subset argument can only take in index vectors (in other words, numeric sequences). 

#Little experiment with the MSE equation used in the bootstrap training exercises:
		#Keep in mind that the MSE can be calculated through: mean((response - predict(dataset equation))[-train]^2)
		
MSE.fn <- function(df, trainid){
	glm.fit <- glm(default ~ income + balance, data = df, family = binomial, subset = trainid)
	return(mean((as.numeric(Default$default) - predict(glm.fit, Default[-1]))[-trainid]^2))
}
MSE.fn(df = Default, sample(10000, 5000))
# Now I understand this function doesn't work because I was calculating the mean squared error with a response variable that is a factor (or rather a categorical variable). Most likely the MSE formula is different between qualitative and quantitative response variables. 
#Changing the qualitative response variable (default) into a numeric vector resulted in a MSE value of 56.70854 (Which is very high). Keep in mind though that the MSE equation for categorical variables may use a different formula than qualitative variables (which means that this value could be wrong). 
boot(Default, MSE.fn, R = 1000)# Seems that there's something wrong with my function. Will need to look into this. Or perhaps one needs to think a little more before creating operations that can be ran with the boot function. 

#(c) 
library(boot)
boot(Default, boot.fn, R = 1000)

#(d) 
summary(default.fit)
#From what I can see the standard errors are very much the same within 0.029 for the intercept, 0.015 for the beta_1 value, and 0.05 for the beta_2 value. The following values obtained from the bootstrap function are as follows:
	#beta_0 4.377e-01, beta_1 4.97e-06, and beta_2 2.322e-04
#As for the logistical regression estimate:
	#beta_0 4.348e-01, beta_1 4.985e-06, beta_2 2.274e-04
	
#7.) (a)
head(Weekly)
dim(Weekly)
colnames(Weekly)
glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly, family = binomial)
summary(glm.fit)

#(b) 
dim(Weekly[-1,])
index <- -1
glm.fit2 <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-1,], family = binomial)

#(c) 
index.prob <- predict.glm(glm.fit2, Weekly[1,], type = "response")
if (index.prob > 0.5){
	index.preb <- "Up"
} else {
	index.preb <- "Down"
}
index.preb
Weekly[1,]
index.prob# From what I can see the observation was indeed correctly classified since it carried a posterior probability of 57.1 percent which is a little over the 50 percent cut off. 

# Yahwes seemed to have obtained the opposite result. In that the classification for observation 1 was false. Most likely my code is correct and the only thing wrong with my conclusion was that I forgot to look at the first observation's direction value or used the set.seed() function. 

#Sweet the former was true. I forgot to look at the actual direction value in the data set for observation 1. The prediction was actually false. The Direction for observation 1 was actually Down (instead of up). 

#(d) (i)
set.seed(1)
direct.prob <- rep(NA, times = nrow(Weekly)) 
for (i in 1:nrow(Weekly)){
	glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-i,], family = binomial)
	direct.prob[i] <- predict(glm.fit, Weekly[i,], type = "response")
}
head(direct.prob)# Note to self, for loops don't like the subset argument within the glm() function. 

#(ii)
direct.pred <- rep(NA, times = nrow(Weekly))
for (i in 1:length(direct.pred)){ 
if (direct.prob[i] > 0.5){
	direct.pred[i] <- "Up"
} else {
	direct.pred[i] <- "Down"
}
}
head(direct.pred)
library(tidyverse)
CrossTable(Weekly$Direction, direct.pred)# According to Asadoughi, the false postive and false negative error rate is 490 (or rather 450 total false postives and 40 false negatives). which means:
mean(direct.pred != Weekly$Direction)# an error rate of 44.995 percent.

#(iii) 
set.seed(1)
direct.pred <- 0
Up_pred <- rep(0, times = nrow(Weekly))
for (i in 1:nrow(Weekly)){
	glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-i,], family = binomial)
	direct.prob <- predict(glm.fit, Weekly[i,], type = "response")
	if (direct.prob > 0.5){
		Up_pred[i] <- 1
	} else {
		Up_pred[i] <- 0
	}
}
sum(Up_pred)
sum(Up_pred) - table(Weekly$Direction)[2]# this function works perfectly, but the solution still differs with asadoughi result with 490 total errors. I believe that the CrossTable() function from the gmodels packages was the closest I got to the correct answer. 

#Asadoughi's solution 
count <- rep(0, dim(Weekly)[1])
for(i in 1:(dim(Weekly)[1])){
	glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-i,], family = binomial)
	is_up <- predict.glm(glm.fit, Weekly[i,], type = "response") > 0.5
	is_true_up <- Weekly[i,]$Direction == "Up"
	if(is_up != is_true_up){
		count[i] = 1
		}
}
sum(count)# I have to say that this function was ingenious. I hope that I reach this same state of understanding by the end of this year. 

#(e) 
mean(count)# this is hilarious. I obtained the same error rate with my for loop. which means that my for loops can very much be considered a correct answer. 

#8.) (a) Anadoughi's solution:
set.seed(1)
y <- rnorm(100)
x <- rnorm(100)
y <- x - 2 * x^2+rnorm(100) 
# n = 100, p = 2 (I believe that the p variable is the number of polynomials, or rather the polynomial transformation degree, of the model). 
		#Y = X - 2(X)^2 + epsilon (which is the irreducible error)
		
# (b).
plot(x,y)# this looks very much like a perfect parabla. Pretty cool, will need to remember this method when modeling mathematical equations from the pre-calculus book. 

#(c) 
#(i)
Data <- data.frame(x,y)# Now I understand. The cv.glm() functions better when a data.frame data structure is used for the data argument.
set.seed(1) 
glm.fit1 <- glm(y ~ x)
glm.delta1 <- cv.glm(Data,glm.fit1)$delta
glm.delta1

#(ii) 
set.seed(1)
glm.fit2 <- glm(y ~ poly(x, 2))
glm.delta2 <- cv.glm(Data, glm.fit2)$delta
glm.delta2

#(iii) 
glm.fit3 <- glm(y ~ poly(x, 3))
glm.delta3 <- cv.glm(Data, glm.fit3)$delta
glm.delta3

#(iv) 
glm.fit4 <- glm(y ~ poly(x, 4))
glm.delta4 <- cv.glm(Data, glm.fit4)$delta
glm.delta4
# Just like what the author said about the advantages of using quadratic transformations. The second degree transformation was by far the best method at reducing the LOOCV errors to an exceptible range, but at the 3rd and 4th degrees the advantages are hard to see. 

#(d) 
set.seed(123) 
glm.fit1 <- glm(y ~ x)
glm.delta1 <- cv.glm(Data,glm.fit1)$delta
glm.delta1

#(ii) 
glm.fit2 <- glm(y ~ poly(x, 2))
glm.delta2 <- cv.glm(Data, glm.fit2)$delta
glm.delta2

#(iii) 
glm.fit3 <- glm(y ~ poly(x, 3))
glm.delta3 <- cv.glm(Data, glm.fit3)$delta
glm.delta3

#(iv) 
glm.fit4 <- glm(y ~ poly(x, 4))
glm.delta4 <- cv.glm(Data, glm.fit4)$delta
glm.delta4
# I don't see any difference between the set.seed(1) examples and the set.seed(123) examples in terms of LOOCV error. 
#Anadoughi's interpretation: Exact same, because LOOCV will be the same since it evaluates in n folds of a singl observation.

#e.) the model with the lowest LOOCV error was the second degree polynomial transformation since the trend of the data suggests that it follows a quadratic trend. 

#f.) Anaboughi's solution 
summary(glm.fit)
summary(glm.fit2)
summary(glm.fit3)# P-values show statistical significance of linear and quadratic terms, which agrees with the CV results.

#9.) (a)
??Boston 
library(MASS)
mu_hat <- mean(Boston$medv)
mu_hat# the mu_hat variable was calculated at 22.533 

#(b) 
st_error <- sd(Boston$medv)/sqrt(nrow(Boston))
st_error #The standard error is 0.4089

#c.) 
set.seed(1)
boot.fn <- function(data, index) return(mean(data[index]))# function obtained from asadoughi.
bstrap <- boot(Boston$medv, boot.fn, 1000)# The standard error between both of these methods are very much similar. 

#d.) Anaboughi's solution 
t.test(Boston$medv)
c(bstrap$t0 - 2*0.4119, bstrap$t0 + 2 * 0.4119)# Boostrap estimate only 0.02 away for t.test estimate.

#(e)
set.seed(1)
median_medv <- median(Boston$medv)

#f.)
boot.fn <- function(data, index) return(median(data[index]))
bstrap2 <- boot(Boston$medv, boot.fn, 1000)
bstrap2# The bootstrap function calculated the standard error of the median is 0.3801.

#g.)
mu_hat_0.1 <- quantile(Boston$medv, seq(0.1,1, by = 0.1))[1]
mu_hat_0.1

#h.)
set.seed(1)
boot.fn <- function(data, index) return(quantile(data[index], seq(0.1, 1))[1])
boot(Boston$medv, boot.fn, 1000)# The bootstrap calculated a standard error of 0.505 and in comparison to the 10th quantile value of 12.75 this value is very small. 

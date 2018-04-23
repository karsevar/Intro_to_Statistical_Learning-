### Chapter 2 Statistical Learning:
##2.1 What is Statistical Learning?
#Advertising data set and example problem:
#Our goal is to develop an accurate model that can be used to predict sales on the basis of the three media budgets (TV, Radio, and newspaper). 
#In this setting, the advertising budgets are input variables while sales is an output variable. The input variables are typically denoted using the symbol X, with a subscript to distinguish them. So X_1 might be the TV budget, X_2 the radio budget, and X_3 the newspaper budget. The inputs go by different names, such as predictors, independent variables, features, or sometimes just variables. The output variable -- in this case is sales --- is often called the response or dependent variable, and is typically denoted using the symbol Y. 
par(mfrow = c(1,3))# lets see if I can use the par() command with a ggplot graphic geomtric layer. 
library(ggplot2)
library(ISLR)
??Advertising
dim(Advertising)# It seems that I will have to carryout this experiment later on since I can't seem to find the Advertising dataset that the author used for his data represention graphics. 

par(mfrow = c(1,2), mar = c(3,4,2,2))
??plot()
str(Wage)
wage_flea <- Wage
wage_flea$education <- as.numeric(wage_flea$education)
plot(x = wage_flea$education, y = Wage$wage, pch = 16, col = "red")# this isn't the right data set will need to most likely spend a day just finding these datasets that the author used for his explainations. 

#More generally, suppose that we observe a quantitative response R and p different predictors. We assume that there is some relationship between Y and X = (X_1, X_2, X_3, ..., X_p), which can be written in the very general form Y = f(X) + E.

#Here F is some fixed but unknown function of X_1, ..., X_p, and E is a random error term, which is independent of X and has mean zero. In this formulation, F represents the systematic information that X provides about Y. As another example, consider the left-hand panel of Figure 2.2, a plot of income suggests that one might be able to predict income using years of education. However, the function F that connects the input variable to the output variable is in general unknown. In this situation one must estimate F based on the observed points. Since Income is a simulated data set, F is known and is shown by the blue curve in the right hand panel of Figure 2.2. The vertical lines represent the error terms E. We note that some of the 30 observations lie above the blue curve and some lie below it; overall, the errors have approximately mean zero. 

#For the following graphic we plot income as a function of years of education and seniority. Here F is a two dimensional surface that must be estimated based on the observed data. 

par(mfrow = c(1,2), mar = c(3,4,2,2))
??Wage
??Income
# After looking through the help file I found that the dataset named uswages is the faraway package might be the dataset used in the author's examples. Will need to look into this. 
library(faraway)
??uswages 
plot(x = uswages$educ, y = uswages$wage)# This plot doesn't look like the perfect representation of the data in the book. Again will really need to find out where the author obtained this data. Or perhaps the author is simulating the data from scratch through simulation methods. Will need to look into his variables so that I can recreate his data. 

#In essence, statistical learning refers to a set of approaches for estimating F. In this chapter we outline some of the key theoretical concepts that arise in estimating F, as well as tools for evaluating the estimates obtained. 

##2.1.1 Why Estimate F?
#There are two main reasons that we may wish to estimate F: prediction and interence.

##Prediction:
#In many situations, a set of inputs X are readily available, but the output Y cannot be easily obtained. In this setting, since the error therm averages to zero, we can predict Y using Y_hat = f_hat(X), where f_hat represents our estimate for f, and Y_hat represents the resulting prediction for Y. In this setting, F_hat is often treated as the black box, in the sense that one is not typically concerned with the exact form of f_hat, provided that it yields accurate predictions for Y. 

## three dimensional regression plot experiment using The book of R:
# Since I can't use the Income data within the book, I will have to make do with the uswages data in the faraway package. 
plot(x = uswages$educ, y = uswages$wage)
summary(lm(wage ~ educ + exper, data = uswages))# The p-values illustrate that the variables of age and education are stistically significant but the R-squared value is still under 20 percent meaning that there are more variables that influence the prediction variable. But even with that said, I will still create a 3d plot of this formula.
Wage.fit <- lm(wage ~ educ + exper, data = uswages)
len <- 20 
educ.seq <- seq(min(uswages$educ), max(uswages$exper), length = len)
exper.seq <- seq(min(uswages$exper), max(uswages$exper), length = len)
educ.exper <- expand.grid(educ = educ.seq, exper= exper.seq)
wage.pred <- predict(Wage.fit, newdata= educ.exper, interval = "prediction", level = 0.99)
wage.pred.mat <- matrix(wage.pred[,1], nrow = len, ncol = len)
library(rgl)
persp3d(x = educ.seq, y = exper.seq, z = wage.pred.mat, col = "green")
points3d(uswages$educ, uswages$exper, uswages$wage, col = "gray", size = 5)
#this graphical representation really does illustrate that the formula did not capture the entirety of the wage movement with concerns to the uswages dataset. Most likely I will need to find a better set of variables and use some logistical or quadratic transformations. 

#The accuracy of Y_hat as a prediction for Y depends on two quantities, which we will call the reducible error and the irreducible error. In general, f_hat will not be a perfect estimate for f, and this inaccuracy will introduce some error. This error is reducible because we can potentially improve the accuracy of f_hat by using the most appropriate statistical learning technique to estimate f. However, even if it were possible to form a perfect estimate for f, so that our estimate response took the form of Y_hat = f(X), our prediction would still have some error in it. This is because Y is also a function of E, which, by definition, cannot be predicted using x. Therefore, variability associated with e also affects the accuracy of our predictions. This is known as the irreduciable error. 

#The focus of this book is on techniques for estimating f with the aim of minimizing the reducible error. It is important to keep in mind that irreducible error will always provide an upper bound on the accuracy of our prediction for Y. this bound is almost always unknown in practice. 

##Inference:
#We are often interested in understanding the way that Y is affected as X_1, ..., X_p change. In this situation we wish to estimate f, but our goal is not necessarily to make predictions for Y. We instead want to understand the relationship between X and Y, more specifically, to understand how Y changes as a function of X_1, ..., X_p. In this setting, one may be interested in answering the following questions:
	#Which predictors are associated with the response?
	#What is the relationship between the response and each predictor?
	#Can the relationship between Y and each predictor be adequately summarized using a linear equation, or is the relationship more complicated?
	
#this situation falls into the inference paradigm. Another example involves modeling the brand of a product that a customer might purchase based on variables such as price, store location, discount levels, competition price, and so forth. In this situation one might really be most interested in how each of the individual variables affects the probability of purchase. For this instance, what effect will changing the price of a product have on sales? this is an example of modeling for inference.

#Depending on whether our ultimate goal is prediction, inference, or a combination of the two, different methods for estimating f may be appropriate. 

##2.1.2 How do we estimate f?
#Our goal is to apply a statistical learning method to the training data in order to estimate the unknown function f. In other words, we want to find a function f_hat such that Y ~ f_hat(X) for any observation (X,Y). Broadly speaking, most statistical learning methods for this task can be characterized as either parametric or non-parametric. 

##Parametric methods involve a two-step model-based approach.
	#First, we make an assumption about the functional form, or shape of f. For example, one very simple assumption is that f is linear. Once we have assumed taht f is linear, the problem of estimating f is greatly simplified. Instead of having to estimate an entirely arbitrary p-dimensional function f(X), one only needs to estimate the p + 1 coefficients beta_0, beta_1, ..., beta_p.
	
	#Second, after a model has been selected, we need a procedure that uses the training data to fit or train the model. In the case of the linear model, we need to estimate the parameters. The most common approach to fitting the model is referred to as least squares, which we discuss in chapter 3. 
	
#The model-based approach just described is referred to as parametric it reduces the problem of estimating f down to one of estimating a set of parameters. Assuming a parametric form for f simplifies the problem of estimating f because it is generally much easier to estimate a set of parameters, such as beta_0, beta_1, ..., beta_p in the linear model, than it is to fit an entirely arbitrary function f. The potential disadvantages to this method are; the utilization of a highly flexible model that promotes overfitting and the event that the model does not follow the trends of the data points at all.

#the following graphic shows an example of the paramtrix approach applied to the uswages data from the last 3d plot. We have fit a linear model of the form:
	#wage ~ beta_0, beta_1 * education + beta_2 * experience
	# remember that this model is completely inadequate at following the actual movement of the prediction variable (wage). Hopefully the author explains transformations.
	
##Non-parametric methods:
#Non-parametric methods do not make explicit assumptions about the functional form of f. instead they seek an estimate of f that gets as clase to the data points as possible without being too rough or wiggly. such approaches can have a major advantage over parametric approaches: by avoiding the assumption of a particular function form for f, they have the potential to accurately fit a wider range of possible sahpes for f. Non-parametric methods do suffer from a major disadvantage: since they do not reduce the problem of estimating f to a small number of parameters, a very large number of observations is required in order to obtain an accurate estimate for f. 

##2.1.3 The trade-off between prediction accuracy and model interpretability:
#Or the many methods that we examine in this book, some are less flexible or more restrictive, in the sense that they can produce just a relatively small range of shapes to estimate f. For example, linear regression is a relatively inflexible approach, because it can only generate linear functions such as the lines shown in figure 2.1 or the plane shown in figure 2.3.
#For the remainder of page 40 the author talks about the tradeoffs of interpretability and flexibility (and vice versa). non-parametric spline methods might create a very flexible model at the expense of interpretability while the linear regression model has the exact opposite characteristic.

#We have established that when inference is the goal, there are clear advantages to using simple and relatively inflexible statistical learning methods. In some settings, however, we are only interested in prediction, and the interpretability of the predictive model is simply not of interest. For instance, if we seek to develop an algorithm to predict the price of a stock, our sole requirement for the algorithm is that it predict accurately --- interpretability is not a concern.

##2.1.4 Supervised Versus Unsupervised Learning:
#Many classical statistical learning methods such as linear regression and logistic regression, as well as more modern approaches such as GAM, boosting, and support vector machines, operate in the supervised learning domain.
#Unsupervised learning describes the somewhat more challenging situation in which for every observation i = 1, ..., n, we observe a vector of measurements x_i, but no associated response y_i. It is not possible to fit a linear regression model, since there is no response variable to predict. the situation is referred to as unsupervised because we lack a response variable that can supervise our analysis. 

##2.1.5 Regression Versus classification Problems:
#Variables can be characterized as either quantitative or qualitative (also known as categorical). Quantitative variables take on numerical values while qualitative variables take on values in one of K different classes or categories.
#We tend to refer to problesm with a quantitative response as regression problems, while those involving a qualitative response are often referred to as classificiation problems. Least squares linear regression is used with a quantitative respons, whereas logistic regression is typically used with a qualitative (two-class binary) response. 

##2.2 Assessing Model Accuracy:
##2.2.1 Measuring the quality of fit:
#In order to evaluate the performance of a statistical learning method on a given data set, we need some way to measure how well its predictions actually match the observed data. That is, we need to quantify the extent to which the predicted response value for a given observation. In this regression setting, the most commonly used measure is the mean squared error (To see the equation look at page 45).
#Where f_hat(x_i) is the prediction that f_hat gives for the ith observation. The MSE will be small if the predicted responses are very close to the true responses, and will be large if for some of the observations, the predicted and true responses differ substantially. 

#The MSE is computed using the training data that was used to fit the model, and so should more accurately be referred to as the traing MSE. But in general, we do not really care how well the method works on the training data. Rather, we are interested in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data. 

#To state it more mathematically, suppose that we fit our statistical learning method on our training observations {(X_1, y_1),(x_2, y_2), ..., (x_n, y_n)}, and we obtain the estimate f_hat. We can compute f_hat(x_1), (x_2), ..., (x_n). If these are approximately equal to y_1, y_2, ..., y_n, then the training MSE given by the following equation is small. However we want to known whether f_hat(x_0) is approximately equal to y_0, where (x_0, y_0) is a previously unseen test observation not used to train the statistical learning method. We want to choose the method that gives the lowest test MSE, as opposed to the lowest training MSE. In other words, if we had a large number of test observations, we could compute Ave(f_hat(x_0) - y_0)^2, the average squared prediction error for these test observations (x_0, y_0). We'd like to select the model for which the average of this quantity --- the test MSE --- is as small as possible. 

#How can we go about trying to select a method that minimizes the test MSE? In some settings, we may have a test data set available --- that is, we may have access to a set of observations that were not used to train the statistical learning method. We can then simply evaluate the following equation on the test observations, and select the learning method for which the test MSE is smallest.

#In the right-hand panel of figure 2.9, as the flexibility of the statistical learning method increases, we observe a monotone decrease in the training MSE and a U-shape in the test MSE. This si a fundamental property of statistical learning that holds regardless of the particular data set at hand and regardless of the statistical method being used. As model flexibility increases, training MSE will decrease, but the test MSE may not. When a given method yields a small training MSE but a large test MSE, we are said to be overfitting the data. This happens because our statistical learning procedure is working too hard to find patterns in the training data and may be picking up some pattersn that are just caused by random chance rather than by true properties of the unknown function f. This problem gives rise to a small training MSE value and a large test MSE value. 

#Interesting phrase that I heard being passed around in the cyber security space (cross validation which is a method for estimating test MSE using the training data.)

##2.2.2 The Bias Variance Trade off:
#The U-shape observed in the test MSE curves turns out to be the result of two competing properties of statistical learning methods. Though the mathematical proof is beyond the scope of this book, it is possible to show that the expected test MSE, for a given value x_0, can always be decomposed into the sum of three fundamental quantities: the variance of f_hat(x_0), the squared bias of f_hat(x_0) and the variance of the error terms E. That is:
	#E(y_0 - f_hat(x_0))^2 = Var(f_hat(x_0)) + [Bias(f_hat(x_0))]^2 + Var(E).
	
#Here the notation E(y_0 - f_hat(x_0))^2 defines the expected test MSE, and refers to the average test MSE that we would obtain if we repreatedly estimated f using a large number of training sets, and tested each at x_0. The overall exprected test MSE can be computed by averaging E(y_0 - f_hat(x_0))^2 over all possible values of x_0 in the test set. 

#In addition the equation tells us that in order to minimize the expected test error, we need to select a statistical learning method that simultaneously achieves low variance and low bias. Note that variance is inherently a nonnegative quantity, and squared bias is also nonnegative. Hence, we wee that the expected test MSE can never lie below Var(E), the irreducible error for (2.3).

#variance (for statistical learning models) refers to the amount by which f_hat would change if we estimated it using a different training data set. Since the training data are used to fit the statistical learning method, different training data sets will result in a different f_hat. But ideally the estimator for f should not vary too much between training sets. however, if a method has high variance the small changes in the training data can result in large changes in f_hat. In general, more flexible statistical methods have higher variance. 

#Bias refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. As for example linear regression models have a high amount of bias while (alternatively) non-parametric boosting methods do not. Meaning that more flexible methods result in less bias. 

#As a general rule, as we used more flexible methods, the variance will increase and the bias will decrease. The relative rate of change of these two quantities determines whether the test MSE increases or decreases. As we increase the flexibility of a class of methods, the bias tends to initially decrease fasten than the variance increases. Consequently, the expected test MSE declines. However, at some point increasing flexibility has little impact on the bias but starts to significanntly increase the variance. When this happens the test MSE increases.

#The relationship between bias, variance, and test set MSE given in equation 2.7 and displayed in figure 2.12 is referred to as the bias variance trade off. Good test set performance of a statistical learning method requires low variance as well as low squared bias. this is referred to as a trade-off because it is easy to obtain a method with extremely low bias but high variance (for instance, by drawing a curve that passes through every single training observation) or a method with very low variance but high bias (by fitting a horizontal line in the data). The challenge lies in finding a method for which both the variance and squared bias are low. this trade-off is one of the most important recurring themes in this book. 

## 2.2.3 The Classification Setting:
#thus far, our discussion of model accuracy has been focused on the regression setting. But many of the concepts that we have encountered, such as the bias-variance trade-off, transfer over to the classification setting with only some modifications due to the fact that y_i is no longer numerical. Suppose that we seek to estimate f on the basis of training obersvations {(y_1, y_1), ..., (x_n,y_n)}, where now y_1, ..., y_n car qualitative. The most common approach for quantifying the accuracy of our estimate f_hat is the training error rate, the proportion of mistakes that are made if we apply our estimate f_hat to the training observations:
	#look at page 52 for the training error rate equation. 
	
#the training error rate is computed based on the data that was used to train our classifier. As in the regression setting, we are most interested in the error rates that result from applying our classifier to test observations that were not used in training. The test error rate associated with a set of test observations of the form (x_0, y_0) is given by:
		#Ave(I(y_0 != y_hat_0)), 
#where y_hat_0 is the predicted class label that results from applying the classifier to the test observation with predictor x_0. A good classifier is one for which the test error is smallest. 

##The Bayes Classifier:
#It is possible to show (though the proof is outside of the scope of this book) that the test error given in (2.9) is minimized, on average, by a very simple classifier that assigns each observation to the most likely class, given its predictor values. In other words, we should simply assign a test observation with predictor vector x_0 to the class j for which Pr(Y = j| X=x_0) is largest. Note that the following formula is a conditional probability: it is the probability that Y = j, given the observed predictor vector x_0. this very simple classifier is called the Bayes classifier. In a two-class problem where there are only two possible response values, say class 1 or class 2, the Bayes classifier corresponds to predicting class one if Pr(Y = 1|X = x_0) > 0.5, and class two otherwise. 

#The Bayes classifier produces the lowest possible test error rate, called the Bayes error rate. Since the Bayes classifier will always choose the class for which (2.10) is largest, the error rate at X = x_0 will be 1 - max_j Pr(Y = j|X = x_0). 
	
##K-Nearest Neighbors:
# pages 54 through 57 give a very good description of k-nearest neighbors training error. I should really go back to this section later on in my studies. 

## 2.3 Lab: Introduction to R:
#The rnorm() function generates a vector of random normal variables, with first argument n the sample size. Each time we call this function, we will get a different answer. Here we creae two correlated sets of numbers x and y, and use the cor() function to compute the correlation between them.
x <- rnorm(50)
y <- x + rnorm(50, mean = 50, sd = .1)
cov(x, y)

#by default, rnorm() creates standard normal random variables with a mean of 0 and a standard deviation of 1. However, the mean and standard deviation can be altered using the mean and sd arguments, as illustrated above. Sometimes we sant our code to reproduce the exact same set of random numbers; we can use the set.seed() function to do this. The set.seed() function takes an arbitrary integer argument.
set.seed(1303)
rnorm(50)

#The mean() and var() functions can be used to compute the mean and variance of a vector of numbers. Applying sqrt() to the output of var() will give the standard deviation. Or we can simply use the sd() function.
set.seed(3)
y <- rnorm(100)
mean(y)
var(y)
sqrt(var(y))
sd(y)

x <- rnorm(100)
y <- rnorm(100)
plot(x, y)
plot(x, y, xlab = "This is the x-axis", ylab = "This is the y-axis", main = "plot of X vs Y")

#We will now create some more sophisticated plots. The contour() function produces a contour plot in order to represent the three dimensional data; it is like a topographical map. It takes three arguments:
	#1. A vector of the x values (the first dimension)
	#2. A vector of the y values (the second dimension)
	#3. A matrix whose elements correspond to the z value (the third dimension) for each pair of (x,y) coordinates.

x <- seq(1, 10)	
y <- x
f = outer(x, y, function(x,y) cos(y)/(1+x^2))
contour(x, y, f)
contour(x, y, f, nlevels = 45, add = TRUE)
fa <- (f - t(f))/2
contour(x, y, fa, nlevels = 15)# I really need to brush up on my contour function parameters I saddly don't remember how to place a palette into the contour() function assembly. Also the preceeding contour plots seem a little weird. I really need to experiment with outer() I little more to see how it differs with expand.grid(). 

#The image() function works the same way as contour(), except that it produces a color-coded plot whose colors depend on the z value. This is known as a heatmap, and is sometimes used to plot temperature in weather forecasts. Alternatively, persp() can be used to produce a three-dimensional plot. The arguments theta and phi control the angles at which the plot is viewed.
image(x, y, fa)# Cool so this is the function that I was thinking about. Man I really need need a refresher on R graphical functions.
persp(x, y, fa)# cools so persp() is kind of like a low tech persp3d() function. I remember that I used this same function on chapter 21 in the Tilman Davies text book.
persp(x, y, fa, theta = 30)# and the theta and phi change the perspective of the plot. This is different from the rgl funciton persp3d because the user can manipulate the graphic without the need for phi and theta input values. 
library(rgl)
persp3d(x, y, fa)# Illustration but the default color is a little too dark will need to change the fill a little for dramatic effect.
persp3d(x, y, fa, col = "light green")# That looks a lot better. 
persp(x, y, fa, theta = 30, phi = 20)
persp(x, y, fa, theta = 30, phi = 70)
persp(x, y, fa, theta= 30, phi = 40)

##2.3.4 Loading Data:
getwd()
Auto <- read.csv("Auto.csv", header = TRUE, na.strings = "?")
dim(Auto)
str(Auto)
dimnames(Auto)
plot(Auto$cylinders, Auto$mpg)
Auto$cylinders <- as.factor(Auto$cylinders)

#If the variable plotted on the x-axis is categorical, then boxplots will automatically be produced by the plot() function:
attach(Auto)
plot(cylinders, mpg)
plot(cylinders, mpg, col = "red", varwidth = TRUE, horizontal = TRUE, xlab = "cylinders", ylab = "MPG")

#The hist() function can be used to plot a histogram. Note that col = 2, has the same effect as col = "red".
hist(mpg, col = 2, breaks = 15)

#the pairs() function creates a scatterplot matrix, scatterplot for every pair of variables for any given data set. We can also produce scatterplots for just a subset of the variables. 
pairs(Auto)
pairs(~ mpg + displacement + horsepower + weight + acceleration, Auto)

#In conjuntion with the plot() function, identify() provides a useful interactive method for identifying the value for a particular variable for points on a plot. We pass in three arguments to identify. Then clicking on a given point in the plot will cause r to print the value of the variable of interest. Right clicking on the plot will exit the identify() function (control click on a Mac).

plot(horsepower, mpg)
identify(horsepower, mpg, name)# really cool function. I'm glad that I'm reading this section of the chapter despite being well versed in base R functions and subsetting. 

#The summary() function:
summary(Auto)
summary(displacement)
summary(mpg)
detach(Auto)
#Once we have finished using R, we type q() in order to shut it down, or quit. When exiting R, we have the option to save the current workspace so that all objects that we have created in this R session will be available next time. Before exiting R, we may want to save a record of all the commands that we typed in the most recent session;this can be accomplished using the savehistory() function. Next time we enter R, we can load that history using the loadhistory() function. 

##2.4 Exercises 
##Conceptual:
#1. (a) Oh now I understand. What they mean by the sample size n is extremely largem and the number of predictors p is small they are pointing out the number of observations and prediction variables (or rather in this case columns). Due to the number of observations being extremely large, I will pick the inflexible statistical learning method (particularly regression). Since, just as the question described the scenerio of there being a large number of observations and a small number of proposed predictors, an inflexible linear regression method will be well equipped in creating a prediction model that is not effected by the large number of observations. Conversely if a flexible method was used the model will be extremely complicated and the prediction values will be greatly influenced by the random noise in the dataset.

#(b) In this case, I would opted for the flexible non-parametric statistical learning method because we have less observations to work with (this translating into less random noise within the prediction model) and more prediction variables (thus making the task of finding a parametric statistical learning method more difficult).

#(c) In this situation, the inflexible statistical learning method will be the better choice since the relationship between the predictor and response are non-linear and (because of that fact) most non-parametric methods are flexible in nature. The only problem with this alternative is that the smoothing value should be set at a reasonable value as a means to avoid over fitting.

# (d) The extremely high variance values are characteristic of flexible statistical models that have been overfitted to the training dataset. The best remedy for this problem is to use a parametrix flexible statistical learning model (but even with that said, make sure to keep an eye on the bias value as these methods have characteristically high bias).

#2. (a) Most likely the best method for this scenario is to use a regression model as conceptualization of how the predictor values are affecting the response variable is the purpose of this survey and not prediction accuracy.

# (b) Since all we care about (in this scenario) is whether the product was a success or a failure and not how prediction variables interact with the response variable, the black box non-parametric statistical learning method tool kit will be the best match. 

# (c) I would say that this is a prediction and regression problem.

#3. (a) I have no idea how to even begin answering this question:
#solution found in https://github.com/ilyakava/ISL/blob/master/ISL_chap02_Ex.md 

#4. (a) for this problem I will only describe two real-life applications in which classification might be useful due to time constraints:
	# The first scenario I can think of is stock market increase and decrease records where the general public oddly believes that stock market performance should be conceptualized according to benchmarks (like the percent return yesterday, last week, last month, or last year, the standard and poor 500, etc.). Hence that categories in this scenario are conceptualized as whether the stock market or stock is going up or down. The predictors for national economies is the interest rate, GDP, employment rate, etc. while for publicly traded corporations government regulation (that can effect their business product), national interest rate, profit, etc. The prediction variable is the price of the stock or the government bond (whether the value went down or up according to some arbitrary benchmark).
	# Another classification problem is the dropout rate (where the predictor variable can be conceptualized as the probability whether a student will drop out (1) or will continue on and obtain their high school diploma (0)). Prediction variables can be region, school attendence, grade point average, income of the families being surveyed, etc. This scenerio will primarily be used for inference as the causes that lead to public school dropouts are very much important for goverment agencies.
	
#(b) Two regression scenarios:
	# The mpg of 1000 samples of cars and how this bodes for reaching national hydrocarbon consumption figures for the future. The predictor variables will be displacement, weight, horsepower, cylinder count, etc. While the response variable will be the mpg value of each vehicle. Primarily this will be mostly grounded in prediction.
	# Food quality in a school cafeteria. The predictor variable will be the number of students bringing packaged lunch to school, how much of the school lunch the students consume, complaints, etc. This scenerio will be regarded as inferencial. 
	
#(c) Three real life applications of cluster analysis. I honestly can't think of any examples.

#5. For regression methods the advantages of the flexible approach are a lower bias value, the model has the ability to match the movement of the data (with consideration to preset smoothing values), and (for complicated dataset with large amounts of observations) these methods can create a very accurate predictions. As for the inflexible approach; inflexible methods have lower variance (hence meaning that the prediction values will be more accurate) and are easier to explain 

#6. For the answer look at the following link: https://raw.githubusercontent.com/asadoughi/stat-learning/master/ch2/answers

#7. (a) Finding the Euclidean distance:
me_flea <- data.frame(Obs = c(1,2,3,4,5,6), X_1 = c(0,2,0,0,-1,1), X_2 = c(3,0,1,1,0,1), X_3 = c(0,0,3,2,1,1), Y = c("Red","Red","Red","Green","Green","Red"))
me_flea
test.point <- c(0,0,0,NULL)
#Compute euclidean distance between the test point and all the other rows:
sqrt.sum_squares <- function(x) {sqrt(sum(x**2))}
distance <- apply(me_flea[,2:4], 1, sqrt.sum_squares)
distance
#our prediction with k = 1?
sorted <- me_flea[,2:4]
sorted[,4] <- distance
names(sorted)[4] <- "DIST"
sorted <- sorted[order(sorted$DIST),]
K1 <- sorted[1,4]
#our prediction with K =3?
K3 <- sorted[1:3,4]
most_common <- sort(table(K3), decreasing = TRUE)[1]
#If the Bayes decision boundary is non-linear, the best value for K would be low
#Lower K corresponds to a more flexible model. As K grows, bias grows and variance shinks 

#The following solution was created by the author ilyakava. I really need to brush up on my math. I don't undersand how he obtained this answer. 

#8.
#a. 
college <- read.csv("College.csv", header = TRUE)

#b. Look at the data using the fix() function. You should notice that the first column is just the name of each university. We don't really want R to treat this as data. However, it may be handy to have these names for later. Try the following commands:
rownames(college) <- college[,1]
fix(college)

#You should see that there is now a row.names column with the name of each university recorded. This means that R has given each row a name corresponding to the appropriate university. R will not try to perform calculations on the row names. However, we still need to eliminate the first column in the data where the names are stored.
college <- college[,-1]
fix(college)# Cool this is almost like the view() function for Rstudio. Will need to remember this function when dealing with large datasets.

# (c) i.
summary(college)

#ii.
pairs(college[,1:10])

#iii.
library(ggplot2)
ggplot(college, aes(x = Private, y = Outstate)) + geom_boxplot()

#iv. Create a new qualitative variable, called Elite, by binning the Top10perc variable. We are going to divide universities into two groups based on whether or not the proportion of students coming from the top 10 percent of their high school classes exceeds 50 percent 
Elite <- rep("No", nrow(college))
Elite[college$Top10perc > 50] <- "Yes"
Elite <- as.factor(Elite)
college <- data.frame(college, Elite)

library(tidyverse)
college <- college %>%
	mutate(Elite = Top10perc > 50)
head(college, n = 10)# this creates the same column as the base R command above. Knew that tidyverse would make this command a lot easier to carry out.

college %>% 
	group_by(Elite) %>%
	count()
college %>%
	filter(Elite == TRUE)
	
ggplot(college, aes(as.factor(Elite), Outstate)) + geom_boxplot()
summary(college)

#v. 
par(mfrow = c(2,2))
hist(college$Accept)
hist(college$Outstate)
hist(college$Room.Board)
hist(college$Enroll)

#vi.
colnames(college)
??geom_hist
ggplot(college, aes(Accept)) + geom_histogram(bins = 50)
ggplot(college, aes(Room.Board)) + geom_histogram(aes(fill = Elite))
# Looking at this chart you can see that Room.Board tuition is more for Elite schools than none elite schools on average but this statistic can be ignored since there are only a total of 72 elite schools in the dataset. 
college %>%
	filter(Elite == TRUE) %>%
	filter(Room.Board <= 4000)
	
college %>%
	filter(Elite == FALSE) %>%
	filter(Room.Board <= 4000)
	
college %>%
	group_by(Private) %>%
	count()
# Interesting from this line of code you can see that there are 212 public colleges and 565 private colleges in the dataset. I will need to find out how exhaustive this list is with regards to colleges within the US higher eduction system. Interesting statistic none the less. 

pairs(college[,c(3,4,5,6,7,10,12)])
ggplot(college, aes(y = Personal, x = Private, fill = Elite)) + geom_boxplot()# interestingly enough personal spending is the highest for public unelite schools. 

ggplot(college, aes(x = Private, y = Grad.Rate)) + geom_boxplot(aes(fill = Elite))# Elite Colleges have a higher average graduation rate for both the public and private categories. Will need to see if this is true for Acceptence and enrollment rates as well.
ggplot(college, aes(x = Private, y = Accept, fill = Elite)) + geom_boxplot()# Weird acceptance rates are higher for Elite schools as well.
ggplot(college, aes(x = Private, y = Enroll, fill = Elite)) + geom_boxplot()# They even have higher enrollment this is pretty weird will need to see way this is the case.

#9. For this exercise we will use the Auto data set from the lab:
# (a).
str(Auto)
# From what I can see that mpg, displacement, horsepower, weight, and acceleration are quantitative variables. While the cylinders, year, origin, and name are qualitative.
head(Auto$origin, n = 50)# that's strange I really don't know what is described by the origin variable. Will need to look into this. Most likely I'll change this variable into a factor.
Auto$origin <- as.factor(Auto$origin)

#Lets see if displacement can be considered a continuous variable:
ggplot(Auto, aes(x = displacement)) + geom_histogram(bins = 50)# there seems to be repeating values but perhaps it might be intelligent to just leave this variable as a numeric continuous data type. 
sum_auto <- function(x) {
	auto_sum <- rep(NA, ncol(Auto))
	if(is.integer(x) == TRUE | is.numeric(x) == TRUE){
		auto_sum <- sum(x, na.rm = TRUE)
	}
	if(is.factor(x) == TRUE){
		warning("factor detected")
	}
	return(auto_sum)
}
sum_auto(Auto$displacement)
sapply(Auto, FUN = sum_auto)# Cool this function actually works.

range_auto <- function(x) {
	auto_sum <- rep(NA, ncol(Auto))
	if(is.integer(x) == TRUE | is.numeric(x) == TRUE){
		auto_sum <- range(x, na.rm = TRUE)
	}
	if(is.factor(x) == TRUE){
		warning("factor detected")
	}
	return(auto_sum)
}
sapply(Auto, FUN = sum_auto)# Cool this function worked splendidly!

mean_auto <- function(x) {
	auto_sum <- rep(NA, ncol(Auto))
	if(is.integer(x) == TRUE | is.numeric(x) == TRUE){
		auto_sum <- c(mean(x, na.rm= TRUE),sd(x, na.rm = TRUE))
	}
	if(is.factor(x) == TRUE){
		warning("factor detected")
	}
	return(auto_sum)
}# The following function will calculate the standard deviation and the mean for each quantitative predictor while a warning message will be printed for qualitative variable data structures.
sapply(Auto, FUN = mean_auto)

Auto_remove <- function(x){
	Auto <- rep(NA, length = 397)
	Auto <- x[c(-10,-85),]
	return(Auto) 
}
Auto_remove(Auto$displacement)
sapply(Auto, FUN = Auto_remove)# I think a look will do a better job than a function and sapply command call. Perhaps I should test out a for() loop.

for(i in 1:ncol(Auto)){
	Auto[,i] <- Auto[c(-10,-85), i]
}# I haven't the slightest clue of how to make this function work. Will need to look at the answer key. 

newAuto <- Auto[-(10:85), ]
dim(newAuto) == dim(Auto) - c(76,0)
newAuto[9,] == Auto[9,]
newAuto[10,] == Auto[86,]
newAuto$cylinders <- as.numeric(newAuto$cylinders)
sapply(newAuto[,1:7], range)
sapply(newAuto[,1:7], mean)
sapply(newAuto[,1:7], sd)

#(e). 
pairs(Auto)
head(Auto$cylinders, n = 10)
Auto$name[Auto$mpg <=30 & Auto$cylinders == 3]
Auto$name[Auto$mpg <= 20 & Auto$cylinders == 3]# Interesting according to this subsetting line of code I can see that in the three cylinder category the mazda rx2 and maxda rx3 are both rated at below 20 miles per gallon. This is quite curious since the characteristics of three cylinder engines are their fuel efficiency. Perhaps these vehicles most have been labeled as three cylinders by mistake and that their actual engine configuration is rotary. Will need to do some research on this. 
#Knew it according to motortrend the Mazda rx2 and rx3 are both rotary engine vehicles. Will need to create another category for these vehicles.
Auto$name[Auto$cylinders==3]# From what I can see all the three cylinder engines are actually rotary engines and there is no geo metro model insight. 
levels(Auto$cylinders) <- c("Rotary","4","5","6","8")
Auto[Auto$cylinders=="Rotary",]

# From what I can see with the pairs() function command; the variables that should be explored further are displacement, cylinder configuration, weight, acceleration, horsepower, and mpg. 
	#cylinder count and displacement seem to have a positive correlation 
ggplot(Auto, aes(x = cylinders, y = displacement)) + geom_boxplot()
# which is a given since larger engines (more displacement) does correspond with more cylinders. 

	#Increased cylinder count correletates negatively to mpg but positively to horsepower (except for the rotary engines). And of course higher horsepower correlates negatively to mpg.
ggplot(Auto, aes(x = horsepower, y = mpg)) + geom_point() + facet_wrap(~cylinders)
ggplot(Auto, aes(x = cylinders, y = horsepower)) + geom_boxplot()
#These graphics illustrate that yes vehicles with higher cylinder configurations do in fact have higher fuel consumption. Interestingly thought the rotary engine group are very much exceptions to rule (since their fuel consumption mirrors that of five cylinder vehicles) and there are a number of four cylinder vehicles that have abnormally high fuel consumption figures (but again this is most likely because they are high performance rally configurations with larger displacement engines and big turbos). 

#(f). My pairs() plot suggests that weight and acceleration might have a positive correlation on fuel consumption but still the variables of cylinder count and horsepower are the best predictors for fuel economy.

#10.
#(a).
library(MASS)
Boston
??Boston# the dimensions are 506 rows and 14 columns. 

#(b). Due to time constraints, the following answers are from https://github.com/asadoughi/stat-learning/blob/master/ch2/applied.R
pairs(Boston)
# X correlates with: a, b, c. crim: age, dis, rad, tax, ptratio 
#Zn: indus, nox, age, lstat
#Indus: age, dis 
#nox: age, dis 
#dis: lstat
#lstat: medv

#(c)
attach(Boston)
plot(Boston$age, Boston$crim)
#older homes, more crime 
plot(dis, crim)
#closer to work-area more crime 
plot(rad, crim)
#Higher index of accessibility to radial highways,more crime 
plot(tax, crim)
#Higher tax rate, more crime 
plot(ptratio, crim)
#Higher pupil to teacher ratio, more crime 

#(d).
par(mfrow = c(1,3))
hist(crim[crim>1], breaks = 25)
#Most cities have low crime rates, but there is a long tail: 18 suburbs appear to have a crime rate > 20, reaching to above 80.
hist(tax, breaks = 25)
#There is a large divide between suburbs with low tax rates and a peak at 660-680.
hist(ptratio, breaks = 25)
#A skew towards high ratios, but no particularly high ratios.

#(e).
dim(subset(Boston, chas == 1))
#35 suburbs 

#(f)
median(ptratio)

t(subset(Boston, medv == min(medv)))
summary(Boston)
#Not the best place to live, but certainly not the worst.

#(h)
dim(subset(Boston, rm > 7))
dim(subset(Boston, rm > 8))
summary(subset(Boston, rm > 8))
summary(Boston)
#relatively lower crime (comparing range), lower lstat (comparing range).



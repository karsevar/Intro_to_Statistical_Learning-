### chapter 6 Linear Model Selection and Regularization:
#Why might we want to use another fitting proceducre instead of least squares? As we will see, alternative fitting procedures can yield better prediction accuracy and model interpretability.
	#Prediction Accuracy: Provided that the true relationship between the response and the predictors is approximately linear, the least squares estimates will have low bias. If n is greater than p (the number of variables in the model) than the least squares estiamtes tend to also have low variance, and hence will perform well on test observations. However, it n is not much larger than p, then there can be a lot of variability in the least squares fit, resulting in overfitting and consequently poor predictions on future observations not used in the model training. And if p > n, then there is no longer a unique least squares coefficient estimate: the variance is infinite so the method cannot be used at all. By constraining or shrinking the estimated coefficients, we can often substantially reduce the variance at the cost of a negligible increase in bias. This can lead to substantial improvements in the accuracy with which we can predict the response for observations not used in model training.
	#Model interpretability: It is often the case that some or many of the variables used in a multiple regression model are in fact not associated with the response. Including such irrelevant variables leads to unnecessary complexity in the resulting model. By removing these variables -- that is, by setting the corresponding coefficient estimates to zero --- we can obtain a model that is more easily interpreted. Now least squares is extremely unlikely to yield any coefficient estimates that are exactly zero. In this chapter, we see some approaches for automatically performing feature selection or variable selection --- that is, for excluding irrelevant variables from a multiple regression model. 
	
#There are many alternatives, both classical and modern, to using least squares to fit the regression formula. We discuss three important classes of methods:
	#Subset selection. This approach involves identifying a subset of the p-predictors that we believe to be related to the response. We then fit a model using least squares on the reduced set of variables.
	#Shrinkage. This approach involves fitting a model involving all p predictors . However, the estimated coefficients are shrunken towards zero relative to the least squares estimates. This shrinkage (also known as regularization) has the effect of reducing variance. Depending on what type of shrinkage is performed, some of the coefficients may be estimated to be exactly zero. hence, shrinkage methods can also perform variable selection. 
	#Dimension Reduction. This approach involves projecting the p predictors into a M-dimensional subspace, where M < p. This is achieved by computing M different linear combinations, or projections, of the variables. Then these M projections are used as predictors to fit a linear regression model by least squares. 
	
##6.1 Subset selection;
#In this section we consider some methods for selecting subsets of predictors. These include best subset and stepwise model selection procedures.

##6.1.1 Best Subset selection:
# to perform best subset selection, we fit a separate least squares regression for each possible combination of the p predictors. That is, we fit all p models that contain exactly one predictor, all (p_2) = p(p - 1)/2 models that contain exactly two predictors, and so forth. We then look at all of the resulting models, with the goal of identifying the one that is best. 
#The problem of selecting the best model from among the 2^p possibilities considered by best subset selection is not trivial. This is usually broken up into two stages:
	#1. Let M_0 denote the null model, which contains no predictors. This model simply predicts the sample mean for each observation. 
	#2. For k = 1,2, ..., p:
		#a.) fit all (p_k) models that contain exactly k predictors
		#b.) Pick the best among these (p_k) models, and call it M_k. Here best is defined as having the smallest RSS, or equivalently largest R^2.

	#3. Select a single best model from among M_0, ..., M_p using crossvalidated prediction error, C_p (AIC), BIC, or adjusted R^2.
	
# In algorithm 6.1, step 2 identifies the best model (on the training data) for each subset size, in order to reduce the problem from one of 2_p possible models to one of p+1 possible models. In figure 6.1, these models form the lower frontier depicted in red.
#Now in order to select a single best model, we must simply choose among these p + 1 options. This task must be performed with care, because the RSS of these p + 1 models decreases monotonically, and the R^2 increases monotoically, as the number of features included in the models increases. Therefore, it we use these statistics to select the best mode, then we will always end up with a model involving all of the variables. The problem is that a low RSS or a high R^2 indicates a model with a low training error, whereas we wish to choose a model that has a low test error. therefore, in step 3, we use cross-validation prediction error, C_p, BIC, or abjusted R^2 in order to select among M_0, M_1, ..., M_p. 

#Attempt 1: Figure 6.1 page 206
library(ISLR)
colnames(Credit)
class(Credit$Rating)
summary(lm(Rating ~ Ethnicity + Age + Balance, data = Credit))$RSS
lm(Rating ~ Ethnicity + Age + Balance, data = Credit)
# will need to findout how to call the RSS and R squared values within the lm() function assembly. 

#An application of best subset selection is shown in Figure 6.1. Each plotting point corresponds to a least squares regression model fit using a different subset of the 11 predictors in the Credit data set, discussed in chapter 3. Here the variable ethnicity is a three level qualitative variable, and so is represented by two dummy variables, which are selected separately in this case. We have plotted the RSS and R^2 statistics for each model, as a function of the number of variables. The red curves connect the best models for each model size, according to RSS or R^2. This figure shows that, as expected, these quantities improve as the number of variables increases; however, from the three variable model on, there is little improvement in RSS and R^2 as a result of including additional predictors. 

#In the case of logistic regression, instead of ordering models by RSS in step 2 of algorithm 6.1, we instead use the deviance, a measure that plays the role in RSS for a broader class of models. The deviance is negative two times the maximum log-likelihood; the smaller the deviance, the better the fit.

#While best subset selection is a simple and conceptually appealing approach, it suffers from computational limitations. the number of possible models that most be considered grows rapidly as p increases. In general there are 2^p models that involve subsets of p predictors. So if p = 10, there are approximately 1000 possible models to be considered, and if p = 20, then there are over one million possibilities. There are computational shortcuts --- so called branch and bound techniques -- for eliminating some choices, but these have their limitations as p gets large. They also only work for least squares linear regression. We present computationally efficient alternatives to best subset selection next. 

##6.1.2 Stepwise Selection:
#For computational reasons, best subset selection cannot be applied with very large p. Best subset selection may also suffer from statistical problems when p is large. The larger the search space, the higher the chance of finding models that look good on the training data, even though they might not have any predictive power on future data. Thus an enormous search space can lead to overfitting and high variance of the coefficient estimates. For both of these reasons, stepwise methods, which explore a far more restricted set of modes, are attractive alternatives to best subset selection. 

##Forward Stepwise selection:
#Forward stepwise selection is a computationally efficient alternative to best subset selection. While the best subset selection procedure considers all 2^p possible models containing subsets of the p predictors, forward step-wise considers a much smaller set of models. Forward stepwise selection begins with a model containing no predictors, and then adds predictors to the model, one at a time, until all of the predictors are in the model. In partcular, at each step the variable that gives the greatest additional improvement to the fit is added to the model. More formally, the forward stepwise selection procedure is given in Algorithm 6.2 (located on page 207).
#this sounds very familiar to the regression model choosing technique found in the Book of R by Tilman Davies. Funny enough there is actually a package and function that can help with this process. 

#Algorithm 6.2 Forward stepwise selection:
	#1. Let M_0 denote the null model, which contains no predictors.
	#2. For k = 0, ..., p - 1 
		#(a) consider all p - k models that augment the predictors in M_k with one additional predictor. 
		#(b) choose the best among these p - k models, and call it M_k+1. Here best is defined as having smallest RSS or highest R^2.
	#3. Select a single best model from among M_0, ..., M_p using cross validated prediction error, C_p (AIC), BIC, or adjusted R^2.
	
#forward stepwise selection's computational advantage over vest subset selection is clear. Though forward stepwise tends to do will in practice, it is not guaranteed to find the best possible model out of all 2^p models containing subsets of the p predictors. For instance, suppose that in a given data set with p = 3 predictors, the best possible one-variable model contains X_1, and the best possible two variable model instead contains X_2 and X_3. Then forward stepwise selection will fail to select the best possible two-variable model, because M_1 will contain X_1, so M_2 msut also contain X_1 together with one additional variable. 

#Forward stepwise selection can be applied even in the high-dimensional setting where n < p; however, in this case, it is possible to construct submodels M_0, ..., M_n-1 only, since each submodel is fit using least squares, which will not yield a unique solution if p >= n.

##Backward stepwise selection;
#This seems to be just like the method applied in Tilman Davies text book. Will need to look into the page where the functions for this method were described in detail. 

#Like forward stepwise, backward stepwise selection provides an efficient alternative to best subset selection. however, unlike forward stepwise selection, it begins with the full least squares model containing all p predictors, and then iteratively removes the least useful predictor, one-at-a-time. 

#Algorithm 6.3 Backward stepwise selection:
	#1. Let M_p denote the full model, which contains all p predictors.
	#2. for k = p, p-1, ..., 1:
		#(a) Consider all k models that contain all but one of the predictors in M_k, for a total of k - 1 predictors.
		#(b) Choose the best among these k models and call it M_k - 1. Here best is defined as having smallest RSS and highest R^2.
	#3. Select a single best model from among M_0, ..., M_p using cross-validated prediction error, C_p (AIC), BIC, or adjusted R^2.
	
#Like forward stepwise selection, the backward selection approach searches through only 1 + p(p + 1)/2 models, and so can be applied in settings where p is too large to apply best subset selection. Also like forward stepwise selection, backward stepwise selection is not quaranteed to yield the best model containing a subset of the p predictors.
#Backward selection requires that the number of samples n is larger than the number of variables p (so that the full model can be fit). In contrast, forward stepwise can be used even when n < p, and so is the only viable subset method when p is very large. 

##Hybrid Approaches:
#The best subset, forward stepwisem and backward stepwise selection approaches generally give similar but not identical models. As another alternative, hybrid versions of forward and backward stepwise selection are available, in which variables are added to the model sequentially, in analogy to forward selection. However, after adding each new variable, the method may also remove any variables that no longer provide an improement in the model fit. Such an approach attempts to more closely mimic best subset selection while retaining the computational advantages of forward and backward stepwise selection. 

## 6.1.3 Choosing the optimal model:
#In order to select the best model with respect to test error, we need to estimate this test error. There are two common approaches:
	#1. We can indirectly estimate test error by making an adjustment to the training error to account for the bias due to overfitting.
	#2. We can directly estimate the test error, using either a validation set approach or a cross-validation approach, as discussed in chapter 5. 
	
## C_p, AIC, BIC, and adjusted R^2:
#It's important to keep in mind that the RSS and R^2 values in the model are only the measurements of the training data set and as such has no barrings on the test dataset. 

#However, a number of techniques for adjusting the training error for the model size are available. These approaches can be used to select among a set of models with different numbers of variables. We now consider four such approaches: C_p, Akaiki information criterion (AIC), Bayesian information criterion (BIC), and adjusted R^2. 

#This method very much sounds like the statistical power function that I created in the book of R (Under the source code of Tilman Davies). 
#for a fitted least squares model containing d predictors, the c_p estimate of test MSE is computed using the equation 
		#C_p = 1/n(RSS + (2)(d)(sigma)^2),
#Where sigma^2 is an estimate of the variance of the error epsilon associated with each response measurement in the equation above. Essentially, the C_p statistic adds a penalty of 2(d)(sigma^2) to the training RSS in order to adjust for the fact that the training error tends to underestimate the test error. Clearly, the penalty increases as the number of predictors in the model increases;this is intended to adjust for the corresponding decrease in training RSS. As a consequence, the C_p statistic tends to take on a small value for models with a low test error, so when determining which of a set of models is best, we choose the model with the lowest C_p value. In figure 6.2, C_p selects the six variable model containing the predictors income, limit, rating, cards, age and student.

#the AIC criterion is defined for a large class of models fit by maximum likelihood. In the case of the least squares linear regression model with Gaussian errors, maximum likelihood and least squares are the same thing. In this case AIC is given by 
		# AIC = 1/(n)(sigma_hat_2)*(RSS + 2*d*sigma_hat_2),
#where, for simplicity, we have omitted an additive constant. Hence for least squares models, C_p and AIC are proportional to each other.
#BIC is derived from a Bayesian point of view, but ends up looking similar to C_p (and AIC) as well. For the least squares model with d predictors, the BIC is, up to irrelevant constants, given by:
		#BIC = 1/n(RSS + log(n)(d)(sigma_hat_2)).
		
#Like C_p, the BIC will tend to take on a small value for a model with a low test error, and so generally we select the model that has the lowest BIC value. Notice that BIC replaces the 2*d*sigma_hat_2 used by C_p with a log(n)dsigma_hat_2 term, where n is the number of observations. Since log n > 2 for any n > 7, the BIC statistic generally places a heavier penalty on models with many variables, and hence results in the selection of smaller models than C_p. We see that his is indeed the case for the Credit data set; BIC chooses a model that contains only the four predictors income, limit, cards, and student. In this case the curves are very flat and so there does not appear to be much difference in accuracy between the four variable and six variable models.

#The adjusted R^2 statistic is another popular approach for selecting among a set of models that contain different numbers of variables. Recall from chapter 3 that the usual R^2 is defined as 1 - RSS/TSS, where TSS = sum(y_i - y_hat)^2 is the total sum of squares for the response. Since RSS always decreases as more variables are added to the model, the R^2 always increases as more variables are added. For a least squares model with d variables the adjusted R^2 statistic is calculated as 
		#Adjusted R^2 = 1 - RSS/(n - d - 1)/ TSS/(n - 1)

#Unlike C_p, AIC, and BIC, for which a small value indicates a model with a low test error, a large value of adjusted R^2 indicates a model with a small test error. Maximizing the adjusted R^2 is equivalent to minimizing RSS/n - d - 1. while RSS always decreases as the number of variables in the model increases, RSS/n - d - 1 may increase or decrease, due to the presence of d in the denominator. 

#The intuition behind the adjusted R^2 is that once all of the correct variables have been included in the model, adding additional noise variables will lead to only a very small decrease in RSS. Since adding noise variables leads to an increase in d, such variables will lead to an increase in RSS/ n - d - 1, and consequently a decrease in the adjusted R^2. Therefore, in theory, the model with the largest adjusted R^2 will have only correct variables and no noise variables. Unlike the R^2 statistic, the adjusted R^2 statistic pays a price for the inclusion of unnecessary variables in the model. 

##Validation and Cross validation:
#As an alternative to the approaches just discussed, we can directly estimate the test error using the validation set and cross-validation methods discussed in chapter 5. We can compute the validation set error or the cross validation error for each model under consideration , and then select the model for which the resulting estimated test error is smallest. This procedure has an advantage relative to AIC, BIC, C_p, and adjusted R^2, in that it provides a direct estimate of the test error, and makes fewer assumptions about the true underlying model. It can also be used in a wider range of model selection tasks, even in cases where it is hard to pinpoint the model degrees of freedom (the number of predictors in the model) or hard to estimate the error variance sigma^2. 

#The validation errors on the Credit data, for the best d-variable model. The validation errors were calculated by randomly selecting three-quarters of the observations as the trianing set, and the remainder as the validation set. The cross-validation errors were computed using k = 10 folds. In this case, the validation and cross validation methods both result in a six-variable model. However, all three approaches suggest that the four-, five-, and six-variable models are roughly equivalent in terms of their test errors. 

#In this setting, we can select a model using the on-standard error rule. We first calculate the standard error of the estimated test MSE for each model size, and then select the smallest model for which the estimated test error is within on standard error of the lowest point on the curve. The rationale here is that if a set of models appear to be more or less equally good, then we might as well choose the simplest model --- that is, the model with the smallest number of predictors. In this case, applying the one-standard-error rule to the validation set or cross validation approach leads to selection of the three-variable model. 

##6.2 Shrinkage methods:
#The subset selection methods described in section 6.1 involve using least squares to fit a linear model that contains a subset of the predictors. As an alternative, we can fit a model containing all p predictors using a technique that constrains or regularizes the coefficient estimates, or equivalently, the shrinks the coefficient estimates towards zero. It turns out that shrinking the coefficient estimates can significantly reduce their variance. The two best-known techniques for shrinking the regression coefficients towards zero are ridge regression and the lasso. 

# (due to my limited mathematical knowledge the equation described for the shrinkage method will not be presented in this note entry. The equation and explaination is located on page 215). 

##Why does Ridge Regression improve over least squares?
#Ridge regression's advantage over least squares is rooted in the bias-variance trade-off. As lambda increases, the flexibility of the ridge regression fit decreases, leading to decreased variance but increased bias. The minimum MSE is achieved at approximately lambda = 30. Interestingly, because of its high variance, the MSE associated with the least squares fit, when lambda = 0, is almost as high as that of the null model for which all coefficient estimates are zero, when lambda = inf. However, for an intermediate value of lambda, the MSE is considerably lower. 

#In general, in situations where the relationship between the response and the predictors is close to linear, the least squares estimates will have low bias but may have high variance. This means that a small change in the training data can cause a large change in the least squares coefficient as the number of observations n, as in the example in figure 6.5, the least squares estimates will be extremely variable. And if p > n, then the least squares estimates do not even have a unique solution, whereas ridge regression can still perform well by trading off a small increase in bias for the large decrease in variance. And if p>n, then the least squares estimates do not even have a unique solution, whereas ridge regression can still perform well be trading off a small increase in bias for a large decrease in variance. Hence, ridge regression works best in situations where the least squares estimates have high variance. 

#The compuational cost of Ridge regression can be viewed as equal to fitting a model using least squares. Since for any fixed value of lambda, ridge regression only fits a single model, and the model fitting procedure can be performed quite quickly. 

##6.2.2 The Lasso:
#Ridge regression does have one obviours disabvantage . Unlike best subset, forward stepwise, and backward stepwise selection, which will generally select models that involve just a subset of the variables, ridge regression will include all p predictors in the final model. The penalty lambda(sum(beta^2_j)) in 6.5 will shrink all of the coefficients towards zero, but it will not set any of them exactly to zero (unless lambda = inf). This may not be a problem for prediction accuracy, but it can create a challenge in model interpretation in settings in which the number of variables p is quite large. In other words (when interpretated with the Credit data set) the ridge regression will always generate a model involving all ten predictors. Increasing the value of lambda will tend to reduce the magnitudes of the coefficients, but will not result in exclusion of any of the variables.

#To see the equation of the lasso method look at page 219. As with ridge regression, the lasso shrinks the coefficient estimates towards zero. However, in the case of the lasso, the ell_1 penalty (in which the penalty for the ridge regression method is ell_2, will need to learn what these designations are) has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter lambda is sufficiently large. Hence, much like best subset selection, the lasso performs variable selection. As a result, models generated from the lasso are generally much easier to interpret than those produced by ridge regression. We say that the lasso yields sparse models --- that is, models that involve only a subset of the variables. As in ridge regression, selecting a good value of lambda for the lasso is critical.

#As an example, consider the coefficient plots in figure 6.6 located on page 220, which are generated from applying the lasso to the Credit data set. When lambda = 0, then the lasso simply gives the least squares fit, and when lambda becomes sufficiently large, the lasso gives the null model in which all coefficient estimates equal zero. However, in between these two extremes, the ridge regression and lasso models are quite different from each other. Moving from left to right in the right-hand panel of figure 6.6, we observe that at first the lasso results in a model almost simultaneously, shortly followed by income. Eventually, the remaining variables enter the model. Hence, depending on the value of lambda, the lasso can produce a model involving any number of variables. In contrast, ridge regression will always include all of the variables in the model, although the magnitude of the coefficient estimates will depend on lambda.

##Another formulation for ridge regression and the lasso:
#this section is a little too advanced for me at this point in my studies will need to go back to it later on. In any case this section is important in the realm that it describes the similarities between the best set selection process, ridge regression, and the lasso in the estimation of the s value (which is the default penalty value for the coefficients of each variable). Remember that the s value will need to be learned before I can calculate these methods properly by hand. 

##The variable selection property of the lasso:
#Why is it that the lasso, unlike ridge regression, results in coefficient estimates that are exactly equal to zero? The formulations (6.8) and (6.9) can be used to shed light on the issue. Figure 6.7 illustrates the situation. The least squares solution is marked as beta_hat, while the blue diaomond and circle represent the lasso and ridge regression constraints in (6.8) and (6.9), respectively. If s is sufficiently large, then the constraint regions will contain beta-hat, and so the ridge regression and lasso estimates will be the same as the least squares estimates. (Such a large value of s corresponds to lambda = 0 in (6.5) and (6.7).) However, in figure 6.7 the least squares estimates are not the same as the lasso and ridge regression estimates. 

#The ellipses that are centered around beta-hat represent regions of constant RSS. In other words, all of the points on a given ellipse share a common value of the RSS. As the ellipses expand away from the least squares coefficient estimates, the RSS increases. Equations (6.8) and (6.9) indicate that the lasso and ridge regression coefficient estimates are given by the first point at which an ellipse contacts the constraint region. Since ridge regression whas a circular constraint with no sharp points, this intersection will not generally occur on an axis, and so the ridge regression coefficient estimates will be exclusively non-zero. however, the lasso constraint has corners at each of the axes, and so the ellipse will often intersect the straint region at an exis. When this occurs, one of the coefficients will equal zero. The lasso leads to feature selection when p>2 due to the sharp corners of the polyhedron and polytope. 

##Comparing the Lasso and Ridge Regression:
#It is clear that the lasso has a major advantage over ridge regression, in that it produces simpler and more interpretable models the involve only subset of the predictors. Which method leads to better prediction accuracy?

#these two examples illustrate that neither ridge regression nor the lasso will universally dominate the other. In general, one might expect the lasso to perform better in a setting where a relatively small number of predictors have substantial coeffients, and the remaining predictors have coefficients that are very small or that equal zero. Ridge regression will perform better when the response is a function of many predictors, all with coefficients of roughly equal size. However, the number of predictors that is related to the response is never known a priori for real data sets. A technique such as cross-validation can be used in order to determine which approach is better on a particular data set. 

#As with ridge regression, when the least squares estimates have excessively high variance, the lass solution can yield a reduction in variance at the expense of a small increase in bias, and sonsequently can generate more accurate predictions. Unlike ridge regression, the lasso performs variable selection, and hence results in models that are easier to interpret. 

##A simple special case for ridge regression and the lasso:
#The author is using a thought experiment with the use of a diagonal identity matrix 
diag(10) # where 1 illustrates the diagonal values and zero illustrates the non-diagonal values and a data set where p = n (or rather the number of variables equals the number of observations). In addition it is assumed that there is no intercept for both data sets.

#Look at page 225 to see what form the formulas for least squares regression, lasso, and ridge regression methods take.

#Figure 6.10 displays the situation. We can see that ridge regression and the lasso perform two very different types of shrinkage. In ridge regression, each least squares coefficient estimate is shrunken by the same proportion. In contrast, the lasso shrinks each least squares coefficient towards zero by a constant amount, lambda/2; the least squares coefficients that are less than lambda/2 in absolute value are shrunken entirely to zero. The type of shrinkage performed by the lasso in this simple setting (6.15) is known as soft-thresholding. The fact that some lasso coefficients are shrunken entirely to zero explains why the lasso performs feature selection. 

##Bayesian Interpretation for Ridge Regression and the Lasso:
#this method uses Bayesian probability as an interpretation of the model's distribution. Primarily think logistical regression or bayesian theorem in using this method. Check page 226 to see the mathematical interpretation of this method. 

#It turns out that ridge regression and the lasso follow naturally from two special cases of g:
	#If g is a Gaussian distribution with mean zero and standard deviation a function of lambda, then it follows that the posterior mode for beta --- that is, the most likely value for beta, given the data --- is given by the ridge regression solution. (In fact, the ridge regression solution is also the posterior mean). 
	#If g is a double-exponential (Laplace) distribution with mean zero and scale parameter a function of lambda, then it follows that the posterior model for beta is the lasso solution. (However, the lasso solution is not the posterior mean, and in fact, the posterior mean does not yield a sparse coefficient vector.)
	
#The Gaussian and double-exponential priors are displayed in figure 6.11. Therefore, from a Bayesian viewpoint, ridge regression and the lasso follow directly from assuming the usual linear model with normal errors, together with a simple prior distribution for beta. Notice that the lasso prior is steeply peaked at zero, while the Gaussian is flatter and fatter at zero. Hence, the lasso expects a priori that many of the coefficients are (exactly) zero, while ridge assumes the coefficients are randomly distribution about zero. 

##6.2.3 Selecting the tuning parameter:
#Just as the subset selection approaches considered in Section 6.1 require a method to determine which of the models under consideration is best, implementing ridge regression and the lasso requires a method for selecting a value for the tuning parameter lambda in (6.5) and (6.7), or equivalently, the value of the constraint s in (6.9) and (6.8). Cross-validation provides a simple way to tackle this problem. We choose a grid of lambda values and compute the cross-validation error for each value of lambda, as described in chapter 5. We then select the tuning parameter value for which the cross-validation error is smallest. Finally, the model is re-fit using all of the available observations and the selected value of the tuning parameter. 

#To see his conclusion on this experiment look at 228. such understanding of the subject matter can only be obtained from looking at his graphics on the subject of lambda values and the use of cross validation error rates (which I will need to find out how to calculate myself). 

## 6.3 Dimension Reduction methods:
#We now explore a class of approaches that transform the predictors and then fit a least squares model using the transformed variables. We will refer to these techniques as dimension reduction methods. 

#The term dimension reduction comes from the fact that this approach reduces the problem of estimating the p + 1 coefficients beta-0, beta_1, ..., beta_p to the simpler problem of estimating the M = 1 coefficients theta_0, theta_1, ..., theta_M, where M < p. In other words, the dimension of the problems has been reduced from p + 1 to M + 1. 

#Dimension reduction serves to constrain the estimated beta_j coefficients. This constraint on the form of the coefficients has the potential to bias the coefficient estimates. however, in situations where p is large relative to n, selecting a value of M << p can significantly reduce the variance of the fitted coefficients. If M = p, and all the Z_m are linearly independent, then (6.18) poses no constraints. In this case, no dimension reduction occurs, and so fitting (6.17) is equivalent to performing least squares on the original p predictors. 

#All dimension reduction methods work in two steps. First, teh transformed predictors Z_1, Z_2, ..., Z_M are obtained. Second, the model is fit using these M predictors. However, the choice of Z_1, Z_2, ..., Z_M, or equivalently, the selection of the phi_(jm)'s, can be achieved in different ways. In this chapter, we will consider two approaches for this task: principal components and partial least squares. 

## 6.3.1 Principal components regression:
#Principal components analysis (PCA) is a popular approach for deriving a low-dimensional set of features from a large set of variables. PCA is discussed in greater detail as a tool for unsupervised learning in chapter 10. Here we describe its use as a dimension reduction technique for regression. 

##An overview of principal components analysis:
#PCA is a technique for reducing the dimension of a n * p data matrix X. The first principal component direction of the data is that along which the observations vary the most. For instance, consider figure 6.14, which shows population size (pop) in tens of thousands of people, and ad spending for a particular company (ad) in thoursands of dollars, for 100 citie. 

#to see how this data set was calculated through this method look at page 231.

#there is also another interpretation for PCA: the first principal component vector defines the line that is as close as possible to the data. For instance, in figure 6.14, the first principal component line minimizes the sum of the squared perpendicular distances between each point and line. these distances are plotted as dashed line segments in the left hand panel of figure 6.15, in which the crosses represent the projection of each point onto the first principal component line. the first principal component has been chosen so that the projected observations are as close as possible to the original observations. 

#In the right-hand panel of figure 6.15, the left hand panel has been rotated so that the first principal component direction coincides with the x-axis. It is possible to show that the first principal component score for the ith observation, given in (6.20), is the distance in the x-direction of the ith cross from zero. so for example, the point in the bottom-left corner of the left-hand panel of figure 6.15 has a large negative principal component score, z_i1 = -26.1, while the point in the top-right corner has a large positive score, z_i1 = 18.7. These scores can be computed directly using (6.20).

#We can think of the values of the principal component Z_1 as single-number summaries of the joint pop and ad budgets for each location. How well can a single number represent both pop and ad? In this case, figure 6.14 indicates that pop and ad have approximately a linear relationship, and so we might expect that a single-number summary will work well (Hence most likely look for covariance between two variables when using this method). Figure 6.16 displays z_i1 versus both pop and ad. The plots show a strong relationship between the first principal component and the two features. In other words, the first principal component appears to capture most of the information contained in the pop and ad predictors. 

#So far we have concentrated on the first principal component. In general, one can construct up to p distinct principal components. The second principal component Z_2 is a linear combination of the variables that is uncorrelated with Z_1, and has largest variance subject to this constraint. It turns out that the zero correlation condition of Z_1 and Z_2 is equivalent to the condition that the direction must be perpendicular, or orthogonal, to the first principal component direction. The second principal component is given by the formula:
		#Z_2 = 0.544 * (pop - pop_hat) - 0.839 * (ad - ad_hat).
		
#Since the advertising data has two predictors, the first two principal components contain all of the information that is in pop and ad. However, by construction, the first component will contain teh most information. Consider, for example, the much larger variability of z_i1 (the x-axis) versus z_i2 (the y-axis) in the right-hand panel of figure 6.15. The fact that the second principal component scores are much closer to zero indicates that this component captures far less information. This illustrates that for this formula one only needs the first principal component in order to accurately represent the pop and ad budgets.

#With two-dimensional data, such as in our advertising example, we can construct at most two principal components. however, if we had other predictors, such as population age, income level, education, and so forth, then additional components could be constructed. They would successively maximize variance, subject to the constraint of being uncorrelated with the preceding components. 

##The principal components regression approach:
#the principal components regression (PCR) approach involves constructing the first M principal components, Z_1, ..., Z_M, and then using these components as the predictors in a linear regression model that is fit using least squares. The key idea is that often a small number of principal components suffice to explain most of the variability in the data, as well as the relationship with the response. In other words, we assume that the directions in which X_1, ..., X_M show the most variation are the directions that are associated with Y. While this assumption is not guaranteed to be true, it often turns out to be a reasonable enough approximation to give good results. 

#If the assumption underlying PCR holds, then fitting a least squares model to Z_1, ..., Z_M will lead to better results than fitting a least squares model to X_1, ..., X_p, since most of all of the information in the data that relates to the response is contained in Z_1, ..., Z_M, and by estimating only M<<P coefficients we can mitigate overfitting. In the advertising data, the first principal component explains most of the variance in both pop and ad, so a principal component regression that uses this single variable to predict some response of interest, such as sales, will likely perform quite well. 

#It is important to keep in mind that as more principal components are used in the regression model, the bias decreases, but the variance increases. This results in a typical U-shape for the mean squared error. When M = p = 45 the PCR amounts simply to a least squares fit using all of the original predictors. The figure indicates that performing PCR with an appropriate choice of M can result in a substantial improvement over least squares.

#PCR will tend to do well in cases when the first few principal components are sufficient to capture most of the variation in the predictors as well as the relationship with the response. 

#We note that even though PCR provides a simple way to perform regression using M < p predictors, it is not a feature selection method. This is because each of the M principal components used in the regression is a linear combination of all p of the original features. For instance, in (6.19), Z_1 was a linear combination of both pop and ad. Therefore, while PCR often performs quite well in many practical settings, it does not result in the development of a model that relies upon a small set of the original features. In this sence. PCR is more closely related to ridge regression than to the lasso. 

#In PCR, the number of principal components, M, is typically chosen by cross-validation. 

#When performing PCR, we generally recommend standardizing each predictor, using (6.6), prior to generating the principal components. This standardization ensures that all variables are on the same scale. In the absense of standardization, the high-variance variables will tend to play a larger role in the principal components obtained, and the scale on which the variables are measured will ultimately have an effect on the final PCR model. 

##6.3.2 Partial Least Squares:
#We now present partial least squares (PLS), a supervised alternative to PCR. Like PCR, PLS is a dimension reduction method, which first identifies a new set of features Z_1, ..., Z_M that are linear combinations of the original features, and then fits a linear model via least squares using these M new features. PLS identifies these new features in a supervised way --- that is it makes use of the response Y in order to identify new features that not only approximate the old features well, but also that are related to the response. 

#To find out how the author uses this method to find the direction of the z_1 projected values look at page 237.

#the PLS direction does not fit the predictors as closely as does PCA, but it does a better job explaining the response. 

#As with PCR, the number of M of partial least squares directions used in PLS is a tuning parameter that is typically chosen by cross-validation. We generally standardize the predictors and the response before performing PLS.

##6.4 Considerations in High Dimensions:
##6.4.1 High-dimensional data
# We have defined the high-dimensional setting as the case where the number of features p is larger than the number of observations n. But the considerations that we will now discuss certainly also apply if p is slightly smaller than n, and are best always kept in mind when performing supervised learning. 

##6.4.2 What goes wrong in high dimensions?
#When p > n or p = n, a simple least squares regression line is too flexible and hence overfits the data. 

#Unfortunately, C_p, AIC, and BIC approaches are not appropriate in the high-dimensional setting, because estimating sigma_hat^2 is problematic. (for instance, the formula for sigma_hat^2 from chapter 3 yields an estimate sigma_hat^2 = 0 in this setting). Similarly, problems arise in the application of adjusted R^2 in the high dimensional setting, since one can easily obtain a model with an adjusted R^2 value of 1. 

## 6.5 Lab 1: Subset Selection methods:
##6.5.1 Best subset selection:
#Here we apply the best subset selection approach to the Hitters data. We wish to predict a baseball player's Salary on the basis of various statistics associated with performance in the previous year 
#First of all, we note that the Salary variable is missing for some of the players. The is.na() function can be used to identify the missing observations. It returns a vector of the same length as the input vector, with a true for any elements that are missing, and a false for non-missing elements. The sum() function can then be used to count all of the missing elements. 
library(ISLR)
fix(Hitters)
dim(Hitters)
colnames(Hitters)# cool it seems that the names column is not considered an actual column which really does make things easier for data entry within a particular modeling function. 
sum(is.na(Hitters$Salary))# 59 total entries that are not accounted for in the model, which translates into only 263 entries that we can use for the model (if we don't use imputation). 

#The na.omit() function removes all of the rows that have missing values in any variable. 
Hitters <- na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))

#the regsubsets() function (part of the leaps library) performs best subset selection by identifying the best model that contains a given number of predictors, where best is quantified using RSS. The syntax is the same as for lm(). The summary() command outputs the best set of variables for each model size. 
library(leaps)
regfit.full <- regsubsets(Salary ~., Hitters)
summary(regfit.full)# Interesting output. Will need to read more into this function. In addition, I find it odd that the author is not using the suit of backwards selection and forward selection functions that Tilman Davies proposed in his book "the book of R" (perhaps these function came out well after this book was published). 

#An asterisk indicates that a given variable is included in the corresponding model. For instance, this output indicates that the best two-variable model contains only hits and CRBI. By default, regsubsets() only reports results up to the best eight variable model. But the nvmax option can be used in order to return as many variables as are desired. Here we fit up to 19 variable model. 

regfit.full <- regsubsets(Salary ~., data = Hitters, nvmax = 19)
reg.summary <- summary(regfit.full)

#The summary() function also returns R^2, RSS, adjusted R^2, C_p, and BIC. We can examine these to try to select the best overall model. 
names(reg.summary)

#For instance, we see that the R^2 statistic increases from 32 percent, when only one variable is included in the model, to almost 55 percent, when all variables are included. As expected, the R^2 statistic increases monotonically as more variables are included. 
reg.summary$rsq# It's important to remember that index position 1 is the model that only has one variable while conversely index position length(reg.summary$rsq) has 19 variables in all. 

#Plotting RSS, adjusted R^2, C_p, and BIC for all of the models at once will help us decide which model to select. Note the type = "l" option tells R to connect the plotted points with lines. 
par(mfrow = c(2,2))
plot(reg.summary$rss, xlab = "number of variables", ylab = "RSS", type = "l")
points(y = reg.summary$rss, x = c(1:19), pch = 19)
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted RSq", type = "l")
points(y = reg.summary$adjr2, x = c(1:19), pch = 19)

#The points() command works like the plot() command, except that it puts points on a plot that has already been created, instead of creating a new plot (or rather graphic window). The which.max() function can be used to identify the location of the maximum point of a vector. We will now plot a red dot to indicate the model with the largest adjusted R^2 statistic. 
which.max(reg.summary$adjr2)# This function says that the variable count with the largest r^2 was 11 variables. Just as I suspected after looking at the graphic. 
points(11, reg.summary$adjr2[11], col = "red", pch = 20)# Cool little technique will need to remember this the further I descend down the rabbit hole of computational statistics and machine learning. 

#In a similar fashion we can plot the C_p and BIC statistic, and indicate the models with the smallest statistic using which.min().
plot(reg.summary$bic, xlab = "number of variables", ylab ="BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20) 
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col ="red", cex = 2, pch =20)

#The regsubsets() function has a built-in plot) command which can be used to display the selected variables for the best model with a given number of predictors, ranked according to the BIC, C_p, adjusted R^2, or AIC. 
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")

#the top row of each plot contains a black square for each variable selected according to the optimal model associated with that statistic. For instance, we see that several models share a BIC close to -150. However, the model with the lowest BIC is the six-variable model that contains only AtBat, hits, Walks, CRBI, DivisionW, and PutOuts. We can use the coef() funciton to see the coefficient estimates associated with this model. 
coef(regfit.full, 6)

##6.5.2 Forward and backward stepwise selection:
#We can also use the regsubsets() function to perform forward stepwise or backward stepwise selection, using the argument method = "forward" or method = "backward". I wonder if this function is the same as the ones described in Tilman Davies' book. 

regfit.fwd <- regsubsets(Salary ~., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)
regfit.bwd <- regsubsets(Salary ~., data = Hitters, nvmax = 19, method = "backward")
summary(regfit.bwd)
#for instance, we see that using forward stepwise selection, the best one-variable model contains only CRBI, and the best two-variable model additionally includes Hits. For this data, the best two-variable model additionally includes Hits. For this data, the best one-variable through six variable models are each identical for best subset and forward selection. However, the best seven-variable models identified by forward stepwise selection, backward stepwise selection, and best subset selection are different. 
coef(regfit.full, 7)
coef(regfit.fwd, 7)
coef(regfit.bwd, 7)

##6.5.3 Choosing Among Models Using the Validation Set Approach and Cross-validation:
#We just saw that it is possible to choose among a set of models of different sizes using C_p, BIC, and adjusted R^2. We will now consider how to do this using the validation set and cross-validation approaches. 

#In order for these approaches to yield accurate estimates of the test error, we must use only the trianing observations to perform all aspects of model fitting -- including variable selection. Therefore, the determination of which model of a given size is best must be made using only the training observations. This point is subtle but important. It the full data set is used to perform the best subset selection step, the validation set errors and cross validation errors that we obtain will not be accurate estimates of the test error. 

set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)# I really don't understand why the author is using this method instead of the random(nrow(Hitters)) command. Honestly my command will be a little simpler to code. 
test <- (!train)

#Now, we apply regsubsets() to the training set in order to perform best subset selection. 
regfit.best <- regsubsets(Salary ~., data = Hitters[train,], nvmax = 19)

#Notice that we subset the Hitters data frame directly in the call in order to access only the training subset of the data, using the expression Hitters[train, ]. We now compute the validation set error for the best model of each model size. We first make a model matrix from the test data. 

test.mat <- model.matrix(Salary ~., data = Hitters[test,])

#The model.matrix() function is used in many regression packages for building an "X" matrix from data. Now we run a loop, and for each size i, we extract the coefficients from regfit.best for the best model of that size, multiply them into the appropriate columns of the test model matrix to form the predictions, and compute the test MSE. 

val.errors <- rep(NA, 19)
for(i in 1:19){
	coefi <- coef(regfit.best, id = i)# Now I understand this part of the loop creates a coeficient vector for each model from one variable to 19 variables.
	pred <- test.mat[,names(coefi)]%*%coefi# Sadly I need to learn matrix algebra to understand this part of the loop.
	val.errors[i] <- mean((Hitters$Salary[test]-pred)^2)
}

#We find that the best model is the one that contains ten variables. 
val.errors
which.min(val.errors)
coef(regfit.best, 10)

#this was a little tedious, partly because there is no predict() method for regsubsets(). Since we will be using this functio again, we can capture our steps above and write our own predict method. 

predict.regsubsets <- function(object, newdata, id, ...){
	form <- as.formula(object$call[[2]])
	mat <- model.matrix(form, newdata)
	coefi <- coef(object, id = id)
	xvars <- names(coefi)
	mat[,xvars] %*% coefi
}

#Our function pretty much mimics what we did above. The only complex part is how we extracted the formula used in the call to regsubsets(). We demonstrate how we use this function below, when we do cross-validation. Finally, we perform best subset selection on the full data set, and select the best ten-variable model. It is important that we make use of the full data set in order to obtain more accurate coefficient estimates. Note that we perform best subset selection on the full data set and select the best ten variable model, rather than simply using the variables that were obtained from the training set, because the best ten variable model on the full data set may differ from the corresponding model on the training set. 

regfit.best <- regsubsets(Salary ~., data = Hitters, nvmax = 19)
coef(regfit.best, 10)

#In fact, we see that the best ten-variable model on the full data set has a different set of variables than the best ten variable model on the training set. 

#We now try to choose among the models of different sizes using cross validation. This approach is somewhat involved, as we must perform best subset selection within eahc of the k training sets. Despite this, we see that with its clever subsetting syntax, R makes this job quite easy. First, we create a vector that allocates each observation to one of k = 10 folds, and we create a matrix in which we will store the results. 

k <- 10 
set.seed(1)
folds <- sample(1:k, nrow(Hitters), replace = TRUE)
cv.errors <- matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))

#Now we write a for loop that performs cross-validation. In the jth fold, the elements of folds that equal j are in the test set, and the remainder are in the training set. We make our predictions for each model size (using our new predict() method), compute the test errors on the appropriate subset and store them in the appropriate slot in the matrix cv.errors 
for(j in 1:k){
	best.fit <- regsubsets(Salary ~., data = Hitters[folds != j,], nvmax = 19)
	for(i in 1:19){
		pred <- predict(best.fit, Hitters[folds == j,], id = i)
		cv.errors[j, i] <- mean((Hitters$Salary[folds==j]-pred)^2)
	}
}

#this has given us a 10 by 19 matrix, of which the (i, j)th element corresponds to the test MSE for the ith cross validation fold for the best j-variable model. We use the apply() function to average over the columns of this matrix in order to obtain a vector for which the jth element is the cross validation error for the j-variable model. 
mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors

#We see that cross-validation selects an 11 variable model. We now performs best subset selection on the full data set in order to obtain the 11-variable model.
reg.best <- regsubsets(Salary ~., data = Hitters, nvmax = 19)
coef(reg.best, 11)

##6.6 Lab 2: Ridge Regression and the Lasso:
#We will use the glmnet package in order to perform ridge regression and the lasso. The main function in this package is glmnet(), which can be used to fit ridge regression models, lasso models, and more. This function has slightly different syntax from other model-fitting functions that we have encountered thus far in this book. In particular, we must pass in an x matrix as well as a y vector, and we do not use the y ~ x syntax. We will now perform ridge regression and the lasso in order to predict Salary on the Hitters data. 

x <- model.matrix(Salary ~., Hitters)[,-1]
y <- Hitters$Salary

#The model.matrix() function is particularly useful for creating x; not only does it produce a matrix corresponding to the 19 predictors but it also automatically transforms any qualitative variables into dummy variables. The latter property is important because glmnet() can only take numerical quatitative inputs. 

##6.6.1 Ridge regression: 
library(glmnet)
grid <- 10^seq(10, -2, length = 100)
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)
#By default the glmnet() function performs ridge regression for an automatically selected range of lambda values. However, here we have chosen to implement the funciton over a gird of values ranging from lambda = 10^10 to lambda = 10^-, essentially covering the full range of scenarios from the null model containing only the intercept, to the least squares fit. As we will see, we can also compute model fits for a particular vlaue of lambda that is not one of the original grid values. Note that by default, the glmnet() function standardizes the variables so that they are on the same scale. To turn off this default setting, use the argument standardize = FALSE.

#Associated with each value of lambda is a vector of ridge regression cofficients, stored in a matrix that can be accessed by coef(). In this case, it is a 20 by 100 matrix, with 20 rows (one for each predictor, plus an intercept) and 100 columns (one for each value of lambda). 
dim(coef(ridge.mod))

#We expect the coefficient estimates to be much smaller, in terms of ell 2 norm, when a large value of lambda is used, as compared to when a small value of lambda is used. These are the coefficients when lambda = 11,498, along with their ell 2 norm:
ridge.mod$lambda[50]
coef(ridge.mod)[,50]

#In contrast, here are the coefficients when lambda = 705, along with their ell 2 norm. Note the must larger ell 2 norm of the coefficients associated with this smaller value of lambda. 
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

#We can use the predict() function for a number of purposes. For instance, we can obtain the ridge regression coefficients for a new value of lambda, say 50:
predict(ridge.mod, s = 50, type = "coefficients")[1:20,]

#We now split the samples into a training set and a test set in order to estimate the test error of ridge regression and the lasso. There are two common ways to randomly split a data set. The first is to produce a random vector of TRUE and FALSE elements and select the observations corresponding to TRUE for the training data. The second is to randomly choose a subset of numbers between 1 and n; these can then be used as the indices for the training observations. The two approaches work equally well. We used the former method in Section 6.5.3. Here we demonstrate the latter approach. 

set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

#Next we fit a ridge regression model on the training set, and evaluate its MSE on the test set, using lambda = 4. Note the use of the predict() function again. This time we get predictions for a test set, by replacing type = "coefficients" with the newx argument. 
ridge.mod <- glmnet(x[train,], y[train], alpha = 0, lambda = grid, thresh = 1e-12)
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test,])
mean((ridge.pred-y.test)^2)

#The test MSE is 101037. Note that if we had instead simply fit a model with just an intercept, we would have predicted each test observation using the mean of the training observations. In that case, we could compute the test set MSE like this:
mean((mean(y[train])-y.test)^2)

#We could also get the same result by fitting a ridge regression model with a very large value of lambda. Note that 1e10 means 10^10.
ridge.pred <- predict(ridge.mod, s = 1e10, newx = x[test,])
mean((ridge.pred-y.test)^2)

#So fitting a ridge regression model with lambda = 4 leads to a much lower test MSE than fitting a model with just an intercept. We now check whether there is any benefit to performing ridge regression with lambda = 4 instead of just performing least squares regression. Recall that least squares is simply ridge regression with lambda = 0.

ridge.pred <- predict(ridge.mod, s = 0, newx = x[test,], exact = TRUE)# It seems that I can't run this particular function. Will need to look into this problem's solution later on through my studies. 
#Error: used coef.glmnet() or predict.glmnet() with `exact=TRUE` so must in addition supply original argument(s)  x and y  in order to safely rerun glmnet

lm(y ~ x, subset = train)
predict(ridge.mod, s = 0, exact = T, type = "coefficients")[1:20,]
# Same error as before. Will need to look into what the problem is. 

#In general, if we want to fit a (unpenalized) least squares model, then we should use the lm() function, since that function provides more useful outputs, such as standard errors and p-values for the coefficients. 

#In general, instead of arbitrarily choosing lambda = 4, it would be better to use cross-validation to choose the tuning parameter lambda. We can do this using the built-in cross-validation function, cv.glmnet(). By default, the function performs ten-fold cross-validation, though this can be changed using the argument folds.
set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha = 0)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam

#Therefore, we see that the value of lambda that results in the smallest cross-validation error is 212. What is the test MSE associated with this value of lambda?
ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test,])
mean((ridge.pred-y.test)^2)

#This represents a further improvement over the test MSE that we got using lambda = 4. Finally, we refit our ridge regression model on the full data set, using the value of lambda chosen by cross-validation, and examine the coeffient estimates. 
out <- glmnet(x, y, alpha = 0)
predict(out, type = "coefficients", s = bestlam)[1:20,]
#As expected, none of the coefficients are zero --- ridge regression does not perform variable selection.

##6.6.2 The Lasso:
#We say that ridge regression with a wise choice of lambda can outperform least squares as well as the null model on the Hitters data set. We now ask whether the lasso can yield either a more accurate or a more interpretable model than ridge regression. In order to fit a lasso model, we once again use the glmnet() function; however, this time we use the aregument alpha = 1. Other than that change, we proceed just as we did fitting a ridge model. 
lasso.mod <- glmnet(x[train,], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

#We can see from the coefficient plot that depending on the choice of tuning parameter, some of the coefficients will be exactly equal to zero. We now perform cross-validation and compute the associated test error. 
set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test,])
mean((lasso.pred-y.test)^2)

#This is substantially lower than the test set MSE of the null model and of least squares, and very similar to the test MSE of ridge regression with lambda chosen by cross validation. 

#However, the lasso has a substantial advantage over ridge regression in that the resulting coefficient estimates are sparse. Here we see that 12 of the 19 coefficient estimates are exactly zero. So the lasso model with lambda chosen by cross-validation contains only seven variables. 

out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coefficients", s = bestlam)[1:20,]
lasso.coef

##6.7.1 Principal components Regression:
#Principal components regression (PCR) can be performed using the pcr function, which is part of the pls library. We now apply PCR to the Hitters data, in order to predict Salary. Again, ensure that the missing values have been removed from the data.
library(pls)
set.seed(2)
pcr.fit <- pcr(Salary ~., data = Hitters, scale = TRUE, validation ="CV")

#the syntax for the pcr() function is similar to that for lm(), with a few additional options. Setting scale = TRUE has the effect of standardizing each predictor prior to generating the principal components, so that the scale on which each variable is measured will not have an effect. Setting validation = "CV" causes pcr() to compute the ten fold cross validation error for each possible value of M, the number of principal components used. The resulting fit can be examined using summary()
summary(pcr.fit)

#The CV score is provided for each possible number of components, ranging from M = 0 onwards. (We have printed the CV output only up to M = 4.) Note that pcr() reports the root mean squared error; in order to obtain the usual MSE, we must square this quantity. For instance, the root mean squared error of 352.8 corresponds to an MSE 352.8^2 = 124,468.

#One can also plot the cross-validation scores using the validationplot() function. Using val.type = "MSEP" will cause the cross validation MSE to be plotted. 
validationplot(pcr.fit, val.type = "MSEP")

#We see that the smallest cross-validation error occurs when M = 16 components are used. This is barely fewer than M = 19, which amounts to simply performing least squares, because when all of the components are used in PCR no dimension reduction occurs. However, from the plot we also see that the cross-validation error is roughly the same when only one component is included in the model. This suggests that a model that uses just a small number of components might suffice. 
#The summary() function also provides the percentage of variance explained in the predictors and in the response using different numbers of components. this concept is discussed in greater detail in chapter 10. Briefly, we can think of this as the amount of information about the predictors or the response that is captured using M principal components. For example, setting M = 1  only captures 38.31 percent of all the variance, or information, in the predictors. In contrast, using M = 6 components, this would increase to 100 percent. 
set.seed(1)
pcr.fit <- pcr(Salary ~., data = Hitters, subset = train, scale = TRUE, validation ="CV")
names(pcr.fit)
validationplot(pcr.fit,val.type = "MSEP")

#Now we find that the lowest cross-validation error occurs when M = 7 component are used. We compute the test MSE as follows.
pcr.pred <- predict(pcr.fit, x[test,], ncomp = 7)
mean((pcr.pred-y.test)^2)

#this test set MSE is competitive with the results obtained using ridge regression and the lasso. However, as a result of the way PCR is implemented the final model is more difficult to interpret because it does not perform any kind of variable selection or even directly produce coefficient estimates. 
#Finally, we fit PCR on the full data set, using M = 7, the number of components identified by cross validation. 
pcr.fit <- pcr(y ~ x, scale = TRUE, ncomp = 7)
summary(pcr.fit)

##6.7.2 Partial Least squares:
#We implement partial least squares (PLS) using the plsr() function also in the pls library. The syntax is just like that of the pcr() function.
set.seed(1)
pls.fit <- plsr(Salary ~., data = Hitters, subset = train, scale = TRUE, validation = "CV")
summary(pls.fit)
#The lowest cross-validation error occurs when only M = 2 partial least squares directions are used. We now evaluate the corresponding test set MSE.
pls.pred <- predict(pls.fit, x[test,], ncomp = 2)
mean((pls.pred-y.test)^2)
pls.fit <- plsr(Salary ~., data = Hitters, scale = TRUE, ncomp = 2)
summary(pls.fit)

meansq <- rep(NA, 19)
for (i in 1:19){
	pls.pred <- predict(pls.fit, x[test,], ncomp = i)
	meansq[i] <- mean((pls.pred-y.test)^2)
}
meansq# In actuality the lowest M value is actually 1 not 2. Will need to look into why the author used M = 2 in this problem.

#Notice that the percentage of variance in Salary that the two component PLS fit explains, 46.40 percent is almost as much as that explained using the final seven component model PCR fit, 46.69 percent. this is because PCR only attempts to amximize the amount of variance explained in the predictors, while PLS searches for directions that explain the variance in both the predictors and the response. 

##6.8 Exercises:
##conceptual:
#1.) Asadoughi's solutions (Sadly I will need to review this chapter regarding these conceptual questions). 
#(a) Best subset selection has the smallest RSS because the other two methods determine models with a path dependency on which predictors they pick first as they iterate to kth model.

#(b) Best subset selection mayhave the smallest test RSS because it considers more model then the other methods. However, the other models might have better luck picking a model that fits the test data better. 

#(c) Sadly I really need to review the chapter to have begin to answer these simple true or false questions. It's sad that I don't remember these characteristics at all. 

#2.) 
#(a) I believe that statement i is true since, by default, the lasso method is more flexible than least squares regression and that increased accuracy can be achieved (in other words decreased variance) through increased bias. Will need to look into if Asadoughi has the same solution for this conceptual question.
#Actually the solution was iii since it is less flexible and better predictions because of less variance, more bias.

#(b) Interesting Asadoughi said that ridge regression is also less flexible than least squares and as such the same statement applies to this question as well. (statement iii).

#(c) Now in this case I believe that ii is the right answer since non-parametric methods are naturally more flexible than least squares regression and their increase in flexibility gives rise to increased volatility and decreased bias. 
#sweet I was finally correct for once.

#3. I'm sad to say that I don't have the knowledge to answer these questions appropriately. As such Asadoughi's solutions will be used in this section:
#(a) (iv) steadily decreases: As we increase s from 0, all beta's increase from 0 to their least square estimate values. Training error for 0 beta s is the maximum and it steadily decreases to the Ordinary Least Squares RSS.

#(b) (ii) Decrease initially, and then eventually start increasing in a U shape: When s = 0, all beta s are 0, the model is extremely simple and has a high test RSS. As we increase s, beta s assume non-zero values and model starts fitting well on test data and so test RSS decreases. Eventually, as beta s approach their full blown OLS values, they start overfitting to the training data, increasing test RSS. 

#(c) (iii) Steadily increase: When s = 0, the model effectively predicts a constant and has almost no variance. As we increase s, the models includes more beta s and their values start increasing. At this point, the values of beta s become highly dependent on training data, thus increasing the variance. 

#(d) (iv) Steadily decrease: when s = 0, the model effectively predicts a constant and hence the prediction is far from actual value. thus bias is high. 

#(e) (v) Remains constant: By definition, irreducible error is model independent and hence irrespective of the choice of s remains constant. 

#4.) 
# (a). (ii) Because it is assumed that lambda 0 will show that the model is decreasing in RSS with increased variables which gives rise to increasing R^2 but than the model will reach a point when the RSS value can't descrease anymore.
#(actual answer) (iii) Steadily increase: As we increase lambda from 0, all beta's decrease from their least square estimate values to 0. Training error for full-blown-OLS beta s is the minimum and it steadily increases as beta s are reduced to 0. 

# (b). I will stand by my last answer to (a) since it is assumed that the RSS will descrease until it reaches some optimum lambda value than the test RSS will start to increase once again.

# (c). I think the variance will steadily increase since increased lambda will only make the model more flexible (since If I remember correctly lambda 0 is the same as a least squares regression model) and this translates to more variance in the model's predictions. 

# (actual answer) (iv) steadily decreases: When lambda = 0, the beta s have their least square estimate values. The actual estimates heavily depend on the training data and hence variance is high. As we increase lambda, beta s start decreasing and model becomes simpler. In the limited case of lambda approaching infinity, all beta s reduce to zero and model predicts a constant and has no variance. (Sadly I didn't know this. Anadoughi's solution). 

#Anadoughi's solutions 
# (d). (iii) Steadily increases: When lambda = 0, beta s have their least-square estimate values and hence have the least bias. As lambda increases, beta s start reducing towards zero, the model fits less accurately to training data and hence bias increases. In the limited case of lambda approaching infinity, the model predicts a contstant and hence bias is maximum. 

#(e). (v) Remains constant: By definition, irreducible error is model independent and hence irrespective of the choice of lambda, remains constant.

#5.) Sadly this is too advanced for me at this point. will need to go back to this question once I obtain some mathematical proficiency.

#6.) function obtained from Yahwes:
#(a)
betas <- seq(-10, 10, 0.1)
eq.ridge <- function(beta, y = 7, lambda = 10) (y-beta)^2 + lambda*beta^2
plot(betas, eq.ridge(betas), xlab = "beta", main = "Ridge Regression Optimization", pch = 1)
points(5/(1+10), eq.ridge(7/(1+10)), pch = 16, col = "red", cex = 2)
#for y = 7 and lambda = 10 the equation minimizes the ridge regression equation. 
#this is so cool. I hope that I get to this level. 

#(b) betas <-seq(-10, 10, 0.1)
eq.lasso <- function(beta, y = 7, lambda = 10) (y - beta)^2 + lambda*abs(beta)
plot(betas, eq.lasso(betas), xlab = "beta", main = "lasso Regression optimization", pch = 1)
points(7-10/2, eq.lasso(7-10/2), pch = 16, col = "red", cex = 2)

#7.) (this is way too advanced for me. For the answer check asadoughi). 

##Applied:
#8. (a).
x <- rnorm(100)
epsilon <- x + rnorm(100)
plot(x, epsilon)# this data set looks pretty messy to me. 

#(b). 
beta_0 <- 2
beta_1 <- 3
beta_2 <- 7
beta_3 <- 2
y <- beta_0 + beta_1*x + beta_2 * x^2 + beta_3 * x^3 + epsilon

#(c) 
library(leaps) 
data.full <- data.frame("X" = x, "Y" = y)#The book author was right you do need a data.frame to use the regsubsets function.
regsubsets(Y ~ X, data = data.full, nvmax = 10)#It seems that this line doesn't have enough dimensions for this command to be completed. Will need to look into this. According to Asadoughi, this command should have the quadratic shortcut function poly(X, 10), Will look into this .
#Error message:
#Error in `colnames<-`(`*tmp*`, value = make.names(np)) : 
  #attempt to set 'colnames' on an object with less than two dimensions.
model.fit <- regsubsets(Y ~ poly(X, 10), data.full)
summary(model.fit)# The best variables are poly(X, 10)1, Poly(X,10)2, and poly(X,10)3. I hope that this isn't just a glitch on part of the regsubsets() function. 
summary(model.fit)$rsq
summary(model.fit)$cp
summary(model.fit)$bic
par(mfrow = c(2,2))
plot(x = c(1:8), y = summary(model.fit)$rsq, type = "l")
points(x = which.max(summary(model.fit)$rsq),y = summary(model.fit)$rsq[8], col = "red", pch = 16)# should have known that R^2 measure will of course pick the model with the most variables . Will need to use the adjusted R^2 to see what model is the best one to use. 
plot(x = c(1:8), y = summary(model.fit)$adjr2, type = "l")
points(x = which.max(summary(model.fit)$adjr2), y = summary(model.fit)$adjr2[which.max(summary(model.fit)$adjr2)], col = "red", pch = 16)# That looks a lot better. The adjust R squared variable say that 5 total variables should be the cut off. Will need to see what the bic and c_p statistics say as well. 
plot(x = c(1:8), y = summary(model.fit)$bic, type = "l")
points(x = which.min(summary(model.fit)$bic), y = summary(model.fit)$bic[which.min(summary(model.fit)$bic)], col = "red", pch = 16)
plot(x = c(1:8), y = summary(model.fit)$cp, type = "l")
points(x = 3, y = summary(model.fit)$cp[which.min(summary(model.fit)$cp)], col = "red", pch = 16)

#My conclusion is that the best model (adjusting for high optimism for the training error rate) the best model has a total of three prodictor variables. This very much presurposes the answer obtained from the adjusted R squared value (which interestingly said that the best model would have a total of 6 variables), but it is important to remember that although the adjust R^2 value penalizes models with a high amount of predictor variables it does not adjust for the difference between the training error rate and the test error rate. 
#To end my thoughts, the best variables to use in the model are poly(X,10)1, poly(X,10)2, poly(X,10)3, and poly(X,10)6. 

#(d) 
model.fwd <- regsubsets(Y ~ poly(X, 10), method = "forward", data = data.full)
model.fwd.sum <- summary(model.fwd)
model.fwd.sum
dev.new()
par(mfrow = c(2,2))
plot(x = c(1:8), y = summary(model.fwd)$rsq, type = "l")
points(x = which.max(model.fwd.sum$rsq),y = summary(model.fwd)$rsq[8], col = "red", pch = 16)
plot(x = c(1:8), y = model.fwd.sum$adjr2, type = "l")
points(x = which.max(model.fwd.sum$adjr2), y = model.fwd.sum$adjr2[5], col = "red", pch = 16)
plot(x = c(1:8), y = model.fwd.sum$bic, type = "l")
points(x = which.min(model.fwd.sum$bic), y = model.fwd.sum$bic[3], col = "red", pch = 16)
plot(x = c(1:8), y = model.fwd.sum$cp, type = "l")
points(x = 3, y = model.fwd.sum$cp[3], col = "red", pch = 16)


model.bwd <- regsubsets(Y ~ poly(X, 10), method = "backward", data = data.full)
model.bwd.sum <- summary(model.bwd)
model.bwd.sum
dev.new()
par(mfrow = c(2,2))
plot(x = c(1:8), y = summary(model.bwd)$rsq, type = "l")
points(x = which.max(model.bwd.sum$rsq),y = summary(model.fwd)$rsq[8], col = "red", pch = 16)
plot(x = c(1:8), y = model.bwd.sum$adjr2, type = "l")
points(x = which.max(model.bwd.sum$adjr2), y = model.fwd.sum$adjr2[5], col = "red", pch = 16)
plot(x = c(1:8), y = model.bwd.sum$bic, type = "l")
points(x = which.min(model.bwd.sum$bic), y = model.bwd.sum$bic[3], col = "red", pch = 16)
plot(x = c(1:8), y = model.bwd.sum$cp, type = "l")
points(x = 4, y = model.bwd.sum$cp[4], col = "red", pch = 16)  

#BIC statistic
summary(model.fit)$bic
model.bwd.sum$bic
model.fwd.sum$bic
# All three models seem to have the same values.

#C_p statistic:
summary(model.fit)$cp
model.bwd.sum$cp
model.fwd.sum$cp

#All the three models seem to have the same values.

#Abjusted R squared:
summary(model.fit)$adjr2
model.bwd.sum$adjr2
model.bwd.sum$adjr2
which.max(model.bwd.sum$adjr2)
which.max(model.fwd.sum$adjr2)
which.max(summary(model.fit)$adjr2)
# All the adjusted R squared coefficients picked a maximum variable size of 5 variables. 

summary(model.fit)
model.bwd.sum
model.fwd.sum
# All models picked the variables poly(X, 10)1, poly(X, 10)2, and poly(X,10)3 as statistically significant.
#There is no difference between my conclusion and that of yawhes's. As for Anaboughi, he obtained a maximum adjust r squared value of 3. Will need to look into this.

#(e).
#lasso method:
library(glmnet)
xmat <- model.matrix(Y ~ poly(X, 10, raw = T), data = data.full)[,-1]
grid <- 10^seq(10, -2, length = 100)
lasso.mod <- cv.glmnet(xmat, y, alpha = 1, lambda = grid)
best.lambda <- lasso.mod$lambda.min
best.lambda
plot(lasso.mod)
best.model <- glmnet(xmat, y, alpha = 1)
predict(best.model, s = best.lambda, type = "coefficients")
# Now I understand all of the coefficients that were not statistically sigificant were converted to zero and the remaining coefficient estimates are the ones regarded as significant.
summary(model.fit)
#The variables poly(X, 10)1, poly(X, 10)2, and poly(X,10)3 are included in both models as statistically significant. the only differences between the two models are that despite poly(X, 10)6 being regarded as statistically significant by the best subset method it was not included in the lasso method and the lasso method interestingly didn't included poly(x, 10)9.

#(f) Y = beta_0 + beta_7*X^7 + epsilon
beta_0# for this exercise beta_0 is set to 2
epsilon# epsilon will remain the same as exercise 8 (c).
beta_7 <- 5
x# x will remain the same as exercise 8 (c).
Y_beta <- beta_0 + beta_7*x^7 + epsilon

#best subset:
data.full <- data.frame("X" = x, "Y" = Y_beta)
model.best <- regsubsets(Y ~ poly(X, 7), data = data.full)
model.sum <- summary(model.best)
which.min(model.sum$bic)
which.max(model.sum$adjr2)
which.min(model.sum$cp)
#That weird all of the test statistics point to seven variables as the best model.
#poly(X, 7)3, poly(X, 7)1, and poly(X, 7)5 are the most statistically significant variables in the data set. 

#lasso
xmat <- model.matrix(Y ~ poly(X, 7), data = data.full)[,-1]
grid <- 10^seq(10, -2, length = 100)
lasso.mod <- cv.glmnet(xmat, Y_beta, alpha = 1)
best.lambda <- lasso.mod$lambda.min
best.lambda
plot(lasso.mod)
out <- glmnet(xmat, Y_beta, alpha = 1)
lasso.pred <- predict(out, s = best.lambda, type = "coefficients")
lasso.pred#the lasso method states that all of the variables are equally significant. Will need to check this with Yahwes and anaboughi. 

# Will test out anaboughi's solution using his coefficients and y and x values. 
set.seed(1)
X <- rnorm(100)
eps <- rnorm(100)
beta0 <- 3
beta7 <- 7
Y <- beta0 + beta7 * X^7 + eps 
data.full <- data.frame("y" = Y, "x" = X)
mod.full <- regsubsets(y ~ poly(x, 10, raw = T), data = data.full, nvmax = 10)
mod.summary <- summary(mod.full)
which.min(mod.summary$cp)
which.min(mod.summary$bic)
which.max(mod.summary$adjr2)
coefficients(mod.full, id = 1)
coefficients(mod.full, id = 2)
coefficients(mod.full, id = 4)

xmat <- model.matrix(y~ poly(x, 10, raw = 10), data = data.full)[,-1]
mod.lasso <- cv.glmnet(xmat, Y, alpha = 1)
best.lambda <- mod.lasso$lambda.min
best.lambda
best.model <- glmnet(xmat, Y, alpha = 1)
predict(best.model, s = best.lambda, type = "coefficients")
# I guess my methodology was correct and the only reason why Anaboughi obtained different answers was because I simply used different values for x and beta_0 and beta_7.

#9.)
#(a) Training set and testing set split:
set.seed(1)
train <- sample(1:nrow(College), nrow(College)/2)
test <- (-train)
College.test <- College[test,]
College.train <- College[train,]
names(College)
head(College$Apps)
str(College)# Since the Apps variable is a numeric vector the best option is to use the lm() function. 

#(b) 
College.lm <- lm(Apps ~ ., data = College[train,])
summary(College.lm)# the variables that should be taken out of the model are the Personal, top10perc, and Terminal since they all have p-values over the 0.05 significance level threshold.
College.sum <- summary(College.lm)
College.pre <- predict(College.lm, College)
names(College.sum)
R2_vec <- c() 
R2 <- NA
College_exp <- College[,-1]
for(i in 2:ncol(College_exp[train,])){
	R2_vec[i] <- summary(lm(Apps ~ College_exp[,1], data = College, subset= train))$r.squared
}
R2_vec
#this for loop only tests out each variable individual and records the r.squared value into a vector. Will need to look into how I can make a for loop that can test out all of the variables from 1 through 9 and document the changes of the r.squared value in one vector data structure. 

summary(lm(Apps ~ ., data = College, subset = train))$r.squared

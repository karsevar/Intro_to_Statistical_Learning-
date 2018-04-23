### Chapter 3 Linear Regression:
#In this chapter, we review some of the key ideas underlying the linear regression model, as well as the least squares approach that is most commonly used to fit this model. Will need to learn about the least squares approach again (I Sadly don't remember its definition).

#the author is reusing the advertising dataset again for this example. Again will need to find a way to recreate the advertising dataset through using the second or third chapter of simulations for data science with R. Suppose that in our role as statistical consultants we are asked to suggest on the basis of this data, a marketing plan for next year that will result in high product sales. What information would be useful in order to provide such a recommendation? Here are a few important questions that we might seek to address:
	#1. Is there a relationship between advertising budget and sales? Our first goal should be to determine whether the data provide evidence of an association between advertsing expenditure and sales. If the evidence is weak, then one might argue that no money should be spend on advertising.
	
	#2. How strong is the relationship between advertsing budget and sales? Given a certain advertising budget, can we predict sales with a high level of accuracy? 
	
	#3. Which media contribute to sales? To answer this question of how these three sales strategies differ in terms of sales affect, we must find a way to separate out the individual effects of each medium when we have spent money on all three media (This might mean three different regression models for each advertisement style will need to look into this). 
	
	#How accurately can we estimate the effect of each medium on sales? 
	
	#How accurately can we predict future sales?
	
	#Is the relationship linear?
	
	#Is there synergy among the advertising media (This can be expressed through typing in the * syntax in place of the + within the lm() function call hence the regression function call will hypothetically look like this: lm(sales ~ TV * Radio, data = advertising)). This is called the interaction effect.
	
## 3.1 Simple Linear Regression:
#Simple linear regression lives up to its name: it is a very straightforward approach for predicting a quantitative reponse Y on the basis of a single predictor variable X. It assumes that there is approximately a linear relationship between X and Y. 

#For the advertising dataset the regression model will be written as 
			# Sales ~ beta_0 + beta_1 * TV 
#where beta_0 and beta_1 are two unknown contants that represent the intercept and slope terms in the linear model. Togethe, beta_0 and beta_1 are known as the model coefficients or parameters. Once we have used our training data to produce estimates beta_hat_0 and beta_hat_1 for the model coefficients, we can predict future sales on the basis of a paritcular value of TV advertising by computing.

#(important mathematical notation) Where y_hat indicates a prediction of Y on the basis of X = x. Here we use a hat symbol to denote the estimated value for an unknown parameter or coefficient, or to denote the predicted value of the response.

#In practice, beta_0 and beta_1 are unknown. So before we can use the equation Y ~ beta_0 + beta_1 * x to make predictions, we must use data to estimate the coefficients. Let
			#(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)
#represent n observation pairs, each of which consist of a measurement of X and a measurement of Y. 

#The most common approach to fitting the beta-0 and beta_1 coefficients to n = 200 (for the advertising dataset) is to use the approach that involves minimizing the least squares citerion, and we take that approach in this chapter. 

#Let y_hat_i = beta_hat_o + beta_hat_1 * x_i be the prediction for Y based on the ith value of X. Then e_i = y_i - y_hat_i represents the ith residual -- this is the difference between the ith observed response value and the ith repsonse value that is predicted by our linear model. We define the residual sum of squares (RSS) as 
			#RSS = e^2_1 + e^2_2 + ...+ e^2_n,
# or equivalently as 
			#RSS = (y_1 - beta_hat_0 - beta_hat_1 * x_1)^2 + ... + (y_n - beta_hat_0 - beta_hat_1 * x_n)^2
			
## 3.1.2 Assessing the Accuracy of the coefficient estimates:
#Recall from the linear regression equation that the true relationship between X and Y takes the form Y = f(X) + e for some unknown function f, where e is a mean-zero random error term. If f is to be approximated by a linear function, then we can write this relationship as:
		# Y = beta_0 + beta_1X + e.
#Here beta_0 is the intercept term - that is, the expected value of Y when X = 0 and beta_1 is the slope --- the average increase in Y associated with a one-unit increase in X. Ther error term is a catch all for what we miss with this simple model. We typically assume that the error term is independent of X. 

#The model given by the preceding equation defines the population regression line, which is the best linear approximation to the true relationship between X and Y. 
xnorm <- rnorm(100)
ynorm <- rnorm(100)
plot(x = xnorm, y = ynorm)
norm.Fit <- lm(ynorm ~ xnorm)
abline(norm.Fit, col = "blue")
# this is the population regression line but I don't really know how to calculate the residual lines that the author created in his example on page 79. Will need to look into this.
# Now I understand, this line that I created was actually the least squares line while the line that the author described as Y = 2 + 3X + e is the population line. 

#Notice that different data sets generated from the same true model result in slightly different least square lines, but the unobservaed population regression line does not change. 

#Interesting the author is describing the principle of bootstrapping from the Simulation for data Science with R text book. He describes that least squares (or rather sample means) will over estimate and underestimate the population mean but with enough sample mean models you can obtain a very truthful approximation of the population mean (the population regression line). This principle was explored in the bootstrapping chapter with the use of flipping a coin a 1000 times with a 1000 separate trails. 

#We continue the analogy with the estimation of the population mean mu of a random variable Y. A natural question is as follows: how accurate is the sampel mean Mu_hat as an estimate of mu?How far off will that single estimate of mu_hat be? In general, we answer this question by computing the standard error of mu_hat, written as SE(mu_hat). 
			#Var(mu_hat) = SE(mu_hat)^2 = sigma^2/n
#Where sigma is the standard deviation of each of the realizations y_i of Y^2. Roughly speaking, the standard error tells us the average amount that this estimate mu_hat differs from the actual value of mu.
#Page 66 has the equation for estimating the sample standard errors for the beta_0 and beta_1 constants. 

#In general, sigma^2 is not known, but can be estimated from the data. This estimate RSE = sqrt(RSS/(n - 2)).

#Standard errors can be used to sompute confidence intervals. A 95 percent confidence interval is defined as a range of values such that with 95 percent probability, the range will contain the true unknown value of the parameter.

#In the case of the advertising data, the 95 percent confidence interval for beta_0 is [6.130, 7.935] and the 95 percent confidence interval for beta_1 is [0.042, 0.053]. Therefore, we can conclude that in the absence of any advertising, sales will, on average, fall somewhere between 6,130 and 7,940 units Furthermore, for each 1000 increase in television advertising, there will be an average increase in sales of between 42 and 53 units. 

#standard errors can also be used to perform hypothesis tests on the coefficients. 
	#H_0: there is no relationship between X and Y 
	#H_a: there is some relationship between X and Y
# or rather:
	#H_0: beta_1 = 0: since zero slope means that there is no correlation between predictor and responds 
	#H_a: beta_1 != 0 
	
#To test the null hypothesis, we need to determine whether beta_hat_1, our estimate for beta_1, is sufficiently far from zero that we can be confident that beta_1 is non-zero. How far is far enough? This of course depends on the accuracy of beta_hat_1 --- that is, it epends on SE(beta_hat_1). If SE(beta_hat_1) is small, then even relativelysmall values of beta_hat_1 may provide strong evidence that beta_1 is not equal to 0 , and hence if the opposite is true (if the SE(beta_hat_1) value is large then the absolute value of beta_1 needs to be large as well for the alternative hyporthesis to hold true). In practice, we compute the t-statistic, given by:
			#t = beta_hat_1 - 0/SE(beta_hat_1),
#which measures the number of standard deviations that beta_hat_1 is away from 0, If there really is no relationship between X and Y, then we expect that the t-statistic formula above with have a t-distribution with n - 1 degrees of freedom. The t-distribution has a bell shape and for values of n greater than approximately 30 it is quite similar to the normal distrubiotn. 

# look at page 68 for the t-statistics and p-value as well as the Beta_0 and beta_1 coefficient null hypothesis. 

## 3.1.3 Assessing th accuracy of the model:
#Once we have rejected the null hypothesis in favor of the alternative hypothesis, it is natural to want to quantify the extent to which the model fits the data. The quality of a linear regression fit is typically assessed using two related quantities: the residual standard error (RSE) and the R^2 statistic. 

##Residual Standard Error:
#Recall from the model for the sample regression line that associated with each observation is an error term e. Due to the presence of these error terms, even if we knew the true regression line, we would not be able to perfectly predict Y and X. The RSE is an estimate of the standard deviation of e. Roughly speaking, it is the average amount that the response will deviate from the true regression line.

# for example; the RSE for the sales ~ tv regression line was solved at 3.26 meaning that the actual sales in each market deviate from the true regression line by approximately 3260 units, on average. In the advertising data set, the mean value of sales over all markets is approximately 14,000 units, and so the percentage error is 3260/14000 = 23 percent.

#The RSE is considered a measure of the lack of fit of the model. Meaning that a small RSE value means that the formula for the sample regression line fits the true regression line well while a large RSE value means the exact opposite. 

##R^2:
#Since the RSE illustrates that a particular model deviates from the true regression line by values of the response variable (Y), this method is not always clear what constitutes a good RSE. The R^2 statistics provides an alternative measure of fit. It takes the form of a proportion -- the proportion of variance explained -- and so it always takes on a value between 0 and 1, and is independent of the scale of Y. 

#To calculate the R^2, we use the formula 
			#R^2 = TSS - RSS / TSS = 1 - RSS / TSS
			
#Where TSS is the total sum of squares and can be thought of as the amount of variability inherent in the response before the regression is performed. In contrast, RSS measures the amount of variability that is left unexplained after performing the regression. R^2 measures the proportion of variability in Y that can be explained using X. An R^2 value that is close to 1 indicates that a large proportion of the variability in the response has been explained by the regression while a low R^2 means the exact opposite. 

#The R^2 statistic is a measure of the linear relationship between X and Y. Recall that correlation, defined as (look at page 70 for the formula) is also a measure of the linear relationship between X and Y. This suggests that we might be able to use r = Cor(X,Y) instead of R^2 in order to assess the fit of the linear model. In fact, it can be shown that in the simple linear regression setting, R^2 = r^2 . In other words, the squared correlation and the R^2 statistic are identical. However, in the next section we will discuss the multiple linear regressio problem in that R^2 is a better predictor overall since correlation (r^2) only works with single predictor variable regression formulas.

## 3.2 Multiple Linear regression:
# When working with multiple predictor variables you can assess their correlation to the response variable individually (which in the case of the Advertising dataset's three predictor variables equates into three separate regression models). 

#However, the approach of fitting a separate simple linear regression model for each predictor is not entirely satisfactory. First of all, it is unclear how to make a single prediction of sales given levels of the three advertising media budgets, since each of the budgets is associated with a separate regression equation. Second, each of the three regression equations ignores the other two media in forming estimates for the regression coefficients. 

#Instead of fitting a separate simple linear regression model for each predictor, a better approach is to extend the simple linear regression model so that it can directly accommodate multiple predictors. We can do this by givening each predictor a separate slope coeficient in a single model. 
# supposing that we have p distinct predictors (meaning that the predictors are not correlated in any way) the multiple linear regression model takes the form:
			# Y = beta_0 + beta_1X_1+beta_2X_2 + ... + beta_pX_p + E
#Where X_j represents the jth predictor and beta_j quantifies the association between that variable and the response. We interpret beta_j as the average effect on Y of a one unit increase in X_j, holding all other predictors fixed. The multiple linear regression equation becomes:
				#sales = beta_0 + beta_1 * TV + beta_2 * radio + beta_3 * newspaper + E 
				
##3.2.1 Estimating the Regression coefficients:
#As was the case in the simple linear regression setting, the regression coefficients beta_0, beta_1, ..., beta_p are unknown and must be estimated.
				#y_hat = beta_hat_0 + beta_hat_1(x_1) + beta_hat_2(x_2) + ... + beta_hat_p(x_p)
				
#the parameters are estimated using the same least squares approach as that of its single linear regression counter part. 
#the values beta_hat_0, beta_hat_1, ..., beta_hat_p that minimize the multiple linear regression formula are the multiple least squares regression coefficient estimate. Unlike the simple linear regression estimates, the multiple regression coefficients have somewhat complicated forms that are most easily represented using matrix algebra. For this reaso, we do not provide them here. 

#toy data experiment with multiple linear regression:
x1_norm <- rnorm(100)
x2_norm <- rnorm(100)
y_norm <- x2_norm * x1_norm
x1_seq <- seq(min(x1_norm), max(x1_norm), length = 20)
x2_seq <- seq(min(x2_norm), max(x2_norm), length = 20)
x1_x2 <- expand.grid(x1_seq, x2_seq)
x_fit <- lm(y_norm ~ x1_norm + x2_norm)
x_pred_mat <- matrix(predict(x_fit, newdata = x1_x2), nrow = 20, ncol = 20)
library(rgl)
persp(x = x1_seq, y = x2_seq, z = x_pred_mat)
persp3d(x = x1_seq, y = x2_seq, z = x_pred_mat, col = "light green")
points3d(x = x1_norm, y = x2_norm, z = y_norm, col = "red", highlight.3d = TRUE)# This doesn't look like a very good example. Will play around with this experiment a little more. 

# This (the coefficient table on page 74) illustrates that the simple and multiple regression coefficients can be quite different. This different stems from the fact that in the simple regression case, the slope term represents the average effect of a $1000 increase in newspaper advertising, ignoring other predictor such as Tv and radio. In contrast, in the multiple regression setting, the coefficient for new paper represents the average effect of increasing spending by 1000 while holding TV and radio fixed. 

#Does it make sense for the multiple regression to suggest no realtionship between sales and newpaper while the simple linear regression implies the opposite? In fact if does. The reason for this statement can be seen in the table on page 75. 

# Funny situation where the regression model was made that found a correlation between increased ice cream sales and increased shark attacks. The actual cause for the shark attacks was a combination of temperature and more people visiting the beach on warm days (The correlation between newspaper and radio can be thought of the same way). 

##3.2.2 Some Important Questions:
# When we perform multiple linear regression, we usually are interested in answering a few important questions:
	#1. Is at least one of the predictors X_1, X_2, ..., X_p useful in predicting the response?
	#2. Do all the predictors help in explain Y, or is only a subset of the predictors useful?
	#3. How well does the model fit the data?
	#4. Given a set of predictor values, what response value should we predict and how accurate is our prediction?
	
##One: Is there a Relationship Between the Response and Predictors?
#Recall that in the simple linear regression setting, in order to determine whether there is a relationship between the response and the predictor we can simply check whether beta_1 is equal to 0. In a multiple regression setting with p predictors, we need to ask whether all of the regression coefficients are zero. As in the simple linear regression setting, we use a hypothesis test to answer this question. We test the null hypothesis, 
		#H_0: beta_1 = beta_2 = ... = beta_p = 0
		#H_a: at least one beta_j is non-zero 
		
#this hypothesis is performed by computing the F-statistic,
			# F = (TSS - RSS)/p / RSS/(n-p-1)
#In this situation an F-statistic of 1 illustrates that there is no correlation between the predictors and the response variables while a large F-statistic illustrates there is a large amount of evidence illustrating that the predictor variables are correlated to the response. How large does the F value have to be to reject the null hypothesis? This depends on the values of n and p. When n is large, an F-statistic that is just a little larger than 1 might still provide evidence against the H_0. In contrast, a larger F-statistic is needed to reject H_0 if n is small

# When H_0 is true and the errors e_i have a normal distribution, the F _statistic follows an F-distribution. 

#Sometimes we want to test that a particular subset of q of the coefficients are zero. this corresponds to a null hypothesis:
			#H_0: beta_p-q + 1 = beta_p - q + 2 = ... = beta_p = 0
#where for convenience we have put the variables chosen for omission at the end of the list. In this case we fit a second model that uses all the variables except those last q. Suppose that the residual sum of squares for that model is RSS_0. Then the appropriate F-statistic is 
		#F = (RSS_0 - RSS)/q / RSS/(n - p - 1)
#(interesting warning) If we use the individual t-statistics and associated p-values in order to decide whether or not there is any association between the variables and the response, there is a very high chance that we will incorrectly conclude that there is a relationship. However, the F statistic does not suffer form this problem because it adjusts for the number of predictors. Hence, if H-0 is true, there is only 5 percent chance that the F-statistic will result in a p-value below 0.05, regardless of the number of predictors or the number of observations. 

#the approach of using an F-statistic to test for any association between the predictors and the response works when p is relatively small, and certainly small compared to n. However, sometimes we have a very large number of variables. If p> n then there are more coefficients beta_j to estimate than observations from which to estimate them. In this case we cannot even fit the multiple linear regression modelusing least squares, so the F-statistic cannot be used, and neither can most of the other concepts that we have seen so far in this chapter.

##Two: Deciding on Important Variables:
#The task of determining which predictors are associated with the response, in order to fit a single model involving only those predictors, is referred to as variable selection. 

#Ideally, we would like to perform variable selection by trying out a lot of different models, each containing a different subset of the predictors. We determine which model is best through Mallow's C_p, Akaike information criterion, Bayesian information criterion, and adjusted R^2. 
#Unfortunately, there are a total of 2^p models that contain subsets of p variables. Therefore, unless p is very small, we cannot consider all 2^p models, and instead we need an automated and efficient approach to choose a smaller set of models to consider. There are three classical approaches for this task:
		#forward selection
		#Backward selection 
		#Mixed selection
#These methods sound very much like the statistical hypothesis testing section in Tilman Davies book "the book of R".
#Backward selection cannot be used if p > n, while forward selection can always be used. Forward selection is a greedy approach, and might include variables early that later become redundant. Mixed selection can remedy this. 

##three: model fit:
#Two of the most common numerical measures of model fit are the RSE and the R^2, the fraction of variance explained. 

##Four: predictions:
#Once we have fit the multiple regression model, it is straightforward to apply (3.21) in order to predict the response Y on the basis of a set of values for the predictors X_1, X_2, ..., X_p. However, there are three sorts of uncertainty associated with this prediction:
	#1. The coefficients beta_hat_1, beta_hat_2, ..., beta_hat_p are all estimates of there population counterparts. And the inaccuracy in the coefficient estimates is related to the reducible error from chapter 2. We can compute a confidence interval in order to determine how close Y_hat will be to f(X).
	
	#2. Of course, in practice assuming a linear model for f(x) is almost always an approximation of reality, so there is an additional source of potentially reducible error which we call model bias. So when we use a linear model, we are in fact estimating the best linear approximation to the true surface. However, here we will ignore this discrepancy and operate as if the linear model were correct. 
	
	#Even if we knew f(X) the response value cannot be predicted perfectly because of the random error e in the model. This is called the irreducible error. We use prediction intervales to show the variability of Y and Y_hat. Prediction intervals are always wider than confidence intervals, because they incorporate both the error in the estimate for f(x) and the uncertainty as to how much an individual point will differ from the population regression plane. 
	
## 3.3 Other considerations in the Regression model:
## 3.3.1 Qualitative predictors
library(ISLR)
Credit#swee the credit dataset is actually in the located in the books R package. Will love to follow along with his explainations. 
colnames(Credit)
str(Credit)
#Candidates for categorical variables include Gender, Student, Married, and Ethnicity. 
pairs(Credit[,c("Balance","Age","Cards","Education","Income","Limit","Rating")])
# This may seem elementary to some people but I find it interesting that people with higher limits have higher ratings. But again this is rather elementary since to obtain a higher limit on ones credit cards you need a good credit rating and vice versa. This is much like good grades affecting school graduation levels. 
# The only trends I can see are the interplay of Balance and limit and income and limit as well as income and rating and balance and rating. Will need to see what the author concluded with this graphic. 
#No way I forgot ethnicity in the graphic. 
pairs(Credit[,c("Balance","Age","Cards","Education","Income","Limit","Rating", "Ethnicity","Student")])# On top of ethnicity I'm adding the variable student as well. 
# the number of levels are too few for me to infer anything at my current understanding of statistics and statistical inference. 

##Predictors with Only Two levels:
#For this example the author used a regression model that had gender as the predictor variable and balance as the response. 
		# mock equation: Balance ~ beta_hat_0 + Beta_hat_1 * Gender(most likely we will have to separate the levels with categorical variables) 

#If a qualitative predictor only has two levels, or possible values, then incorporating it into a regression model is very simple. We simply create an indicator or dummy variable that takes on two possible numerical values. For example, based on the gender variable, we can create a new variable that tkaes the form:
			#x_i 1 (if ith person is female)
				#0 (if ith person is male)
#and use this variable as a predictor in the regression equation. This results in the model
			#y_i = beta_0 + beta_1*x_i + E_i = 
			#beta_0 + beta_1 + E_i if ith person is female 
			#beta_0 + E_i if ith person is male (this variable takes the place of the y intercept of the formula)

#Now beta_0 can be interpreted as the average credit card balance among males, beta_0 + beta_1 as the average credit card balance among females, and beta_1 as the average difference in credit card balance between females and males. 
#The average credit card debt for males is estimated to be 509.80, whereas females are estimated to carry 19.73 in additional debt for a total of 509.80 + 19.7 = 529.53. However, we notice that the p-value for the dummy variable is very high. This indicates that there is no statistical evidence of a difference in average credit card balance between the genders. 

#the labeling of males 0 and females 1 is mathematically arbitrary but in a sense does matter with concerns to interpretation of the formula. In the reverse scenario beta_0 will be 509.80 and beta_1 -19.73.

#Inplace of the 0/1 coding scheme, we could create a dummy variable.
		#x_i 1 if ith person is female 
		#	 -1 if ith person is male 
#and use this variable in the regression equation This results in the model:
#y_i = beta_0+ beta_1*x_i + E_i = beta_0 + beta_1 + E_i (female)
								 #beta_0 - beta_1 + E_i (male)
								 
#Now beta_0 can be interpreted as the overall average credit card balance, and beta_1 is the amount that females are above the average and males are below the average. In this example, the estimate for beta_0 would be 519.665, halfway between the male and female averages of 509.80 and 529.53. The estimate for beta_1 would be 9.865, which is half of 19.73, the average difference between females and males. 

##qualitative predictors with more than two levels:
#When a qualitative predictor has more than two levels, a single dummy variable cannot represent all possible values. In this situation, we can create additional dummy variables. For example, for the ethnicity variable we create two dummy variables. The first could be:
		#x_i_1 = 1 (if ith person is Asian)
				#0 (if ith person is not Asian)
#and the second could be 
		#x_i_2 = 1 (if ith person is Caucasian)
				#0 (if ith person is not Caucasian)
#then both of these variables can be used in the regression equation, in order to obtain the model:
#y_i = beta_0 + beta_1*x_i_1 + beta_2*x_i_2 + E_i = 
#beta_0 + beta_1 + E_i (if ith person is Asian)
#beta_0 + beta_2 + E_i (if ith person is Caucasian)
#beta_0 + E_i (if ith person is African American)

#Now beta_0 can be interpreted as the average credit card balance for African Americans, beta_1 can be interpreted as the difference in the average between the Asian and African American categories, and beta_2 can be interpreted as the difference in the average balance between the Cancasian and African American categories. There will always be one fewer dummy variable than the number of levels. The level with no dummy variable --- African American in this example --- is known as the baseline. From the p_values of each ethnicity, the only ethnicity that has any statistical significance is the intercept (African American category) while the other categories have p-values in excess of 60 percent. 
#The Coefficients and their p-values do depend on the choice of dummy variable coding. Rather than rely on the individual coefficients, we can use an F-test to test H_0: beta_1 = beta_2 = 0; this does not depend on the coding. This F-test has a p-value of 0.96, indicating that we cannot reject the null hypothesis that there is no relationship between balance and ethnicity.

#Using this dummy variable approach presents no difficulties when incorporating both quantitative and qualitative predictors. For example, to regress balance on both a quantitative variable such as income and a qualitative variable such as student, we must simply create a dummy variable for student and then fit a multiple regression model using income and the dummy variable as predictors for credit card balance. 

##3.3.2 Extensions of the Linear Model:
#the standard linear regression model makes several highly restrictive assumptions that are often voilated in practice. Two of the most important assumptions state that the relationship between the predictor and response are additive and linear. The additive assumption means that the effect of changes in a predictor X_j on the response Y is independent of the values of the other predictors. The linear assumption states that the change in the response Y due to a one-unit change in X_j is constant, regardless of the value of X_j. 

##Removing the Additive assumption (most likely the author is going to fit a quadratic or logarithmic equation on a particular number of variables. Oh wait this is for the linear assumption.). 

#Now I understand the additive assumption ignores interaction between variables (or rather the phenomena called synergy)
#One way of extending the simple linear regression model into accoounting for the interaction effect is to include a third predictor, called an interaction term =, which is constructed by computing the product of X_1 and X_2. This results in the model:
		#Y = beta_0 + beta_1*X_1 + beta_2*X_2 + beta_3*X_1X_2 + E.
#or rather:
		#Y = beta_0 (beta_1 + beta_3X_2)X_1 + beta_2X_2 + E 
		  #= beta_0 + beta_~_1X_1 + beta_2X_2 +E
		  
#Where beta_~_1 = beta_1 + beta_3X_2. Since beta_~_1 changes with X_2, the effect of X_1 on Y is no longer constant: adjusting X_2 will change the impact of X_1 and Y. 

#(interesting idea that has eluded me when reading the linear regression section in Tilman Davies book "The book of R) If the interaction between X_1 and X_2 seems important, then we should include both X_1 and X_2 in the model even if their coefficient estimates have large p-values. The rationale for this principle is that if X_1 * X_2 is related to the response then whether or not the coefficients of X_1 or X_2 are exactly zero is of little interest. Also X_1 * X_2 is typically correlated with X_1 and X_2, and so leaving them out tends to alter the meain of the interaction.
#In R code this means that instead of writing the regression model lm(sales ~ radio*tv) we need to write this same model as lm(sales~radio*tv + radio + tv) for stylistic purposes.

#Making regression models that have qualitative variables as well as quantitative. Consider the credit data set, and suppose the we wish to predict balance using the income (quantitative) and student (qualitative) variables. In the absence of an interaction term, the model takes the form 
		#balance_i ~ beta_0 + beta_1 * income_i + beta_2 (student)
												 #0 (not a student)
				  #= beta_1 * income_i + beta_0 + beta_2 (student)
				  								 #beta_0 (not a student)

#Notice that this amounts to fitting two parallel lines to the data, one for students and one for non-students. The lines for students and non-students have different intercepts, beta_0 + beta_2 versus beta_0, but the same slope, beta_1. This represents a potentially serious limitation of the model, since in fact a change in income may have a very different effect on the credit card balance of a student versus a non-student. 

#this limitation can be addressed by adding an interaction variable, created by multiplying income with the dummy variable for student. Our model now becomes:
		#balance_i ~ beta_0 + beta_1 * income_i + beta_2 + beta_3 (student)
												 #0 (not student)
				  #= (beta_0 + beta_2) + (beta_1 + beta_3) * income_i (if student)
				  #  beta_0 + beta_1 * income_i (if not student)
				  
#Once again, we have two different regression lines for the students and the non-students. But now those regression lines have different intercepts, beta_0 + beta_2 versus beta_0, as well as different slopes, beta_1 + beta_3 versus beta_1. This allows for the possibility that changes in income may affect the credit card balances of students and non-students differently. 

##Non-linear Relationships:
#(And now the author will talk about log and quadratic transformations within regression models).
# the linear regression model assumes a linear relationship between the response and predictors. But in some cases, the true relationship between the response and the predictors may be non-linear. Here we present a very simple way to directly extend the linear model to accommodate non-linear relationships, using polynomial regression. 
str(Auto)
colnames(Auto)
cylinders_fac <- as.factor(Auto$cylinders)
plot(Auto$horsepower, Auto$mpg, pch = 16, col = c("green","blue","orange","pink","purple")[cylinders_fac])
legend("topright",legend = c("rotary","4","5","6","8"), col = c("green","blue","orange","pink","purple"), pch = c(rep(16, times = 5)))
#Not the best looking graphical illustration but it's still useable. 
Auto_fit1 <- lm(mpg ~ horsepower, Auto)
abline(Auto_fit1, lwd = 2, col = "yellow")
Auto_fit2 <- lm(mpg ~ horsepower + log(horsepower), Auto)
hors.seq <- seq(min(Auto$horsepower)-50, max(Auto$horsepower)+50, length = 30)
Auto.order2 <- predict(Auto_fit2, newdata = data.frame(horsepower = hors.seq))
lines(hors.seq, Auto.order2, col = "purple", lwd = 2)
Auto_fit3 <- lm(mpg ~ horsepower + I(horsepower^2), Auto)
Auto.order3 <- predict(Auto_fit3, newdata = data.frame(horsepower = hors.seq))
lines(hors.seq, Auto.order3, col = "blue", lwd = 2)
Auto_fit4 <- lm(mpg ~ horsepower + I(horsepower^2) + I(horsepower^3) + I(horsepower^4) + I(horsepower^5), Auto)
Auto.order4 <- predict(Auto_fit4, newdata = data.frame(horsepower = hors.seq))
lines(hors.seq, Auto.order4, col = "green", lwd = 2)
#cool the following code is the syntax the author used to create figure 3.8 in page 91. I find it odd that he didn't include logarithmic transformation in this example. 
legend("bottomleft", lwd = c(rep(2, times = 4)), col = c("yellow","purple","blue","green"), legend = c("Linear","Log","Degree 2","Degree 5"))

#As illustrated by the code above, a simple approach to incorportating non-linear associations in a linear model is to include transformed versions of the predictors in the model. For example, the points in the preceding graphic seem to have a quadratic shape, suggesting that the model of the form:
#mpg = beta_0 + beta_1 * horsepower + beta_2 * horsepower^2 + E
#may provide a better fit. The preceding equation involves predicting mpg using a non-linear function of horsepower. but it is still a linear model. that is simply a multiple linear regression model with X_1 = horsepower and X_2 = horsepower^2. The R^2 value for the quadratic fit in comparison to the linear fit illustrates that the quadratic line is more statistically significant.
summary(Auto_fit3)
summary(Auto_fit1)
summary(Auto_fit4)# the fifth degree line illustrates a somewhat better R^2 value but visually the line seems too wiggly to infer any conclusions from it's predictions. 

#the approach that we have just described for extending the linear model to accommodate non-linear relationships is known as polynomial regression, since we have included polynomial functions of the predictors in the regression model. 

## 3.3.3 Potential problems:
#When we fit a linear regression model to a particular data set, many problems may occur. Most common among these are the following:
		#1. Non-linearity of the response predictor relationships 
		#2. Correlation of error terms 
		#3. Non-constant variance of error terms 
		#4. Outliers 
		#5. High leverage points 
		#6. collinearity
		
##1. non-linearity of the data:
#The linear regression model assumes that there is a straight-line relationship between the predictors and the response. If the true relationship is far from linear, then virtually all of the conclusions that we draw from the fit are suspect. In addition, the prediction accuracy of the model can be significantly reduced. 
#Residual plots are a useful graphical tool for identifying non-linearity. Given a simple linear regression model, we can plot the residuals, e_i = y_i - y_hat_i, versus the predictor x_i. In this case of a multiple regression model, since there are mulitple predictors, we instead plot the residuals versus the predicted (or fitted) value y_hat_i. Ideally, the residual plot will show no discernible patterm. The presence of a pattern may indicate a problem with some aspect of the linear model. 
par(mfrow = c(1,2))
plot(Auto_fit4, which = 1)
plot(Auto_fit3, which =1)
#the code of figure 3.9 on page 93

#If the residual plot indicates that there are non-linear associations in the data, then a simple approach is to use non-linear transformations of the predictors, such as logX, sqrt(X), and X^2, in the regression model. 

##2. Correlation of Error terms:
#An important assumption of the linear regression model is that the error terms, E_1, E_2, ..., E_n, are uncorrelated. What does this mean? For instance, if the errors are uncorrelated, then the fact that E_i is positive provides little or no information about the sign of E_i + 1. The standard errors that are computed for the estimated regression coefficients or the fitted values are based on the assumption of uncorrelated error terms. If in fact there is correlation among the error terms, then the estimated standard errors will tend to underestimate the true standard errors. As a result, confidence and prediction intervalues will be narrower than they should be. For example, 95 percent confidence interval may in reality have a much lower proability than 0.95 of containing the true value of the parameter. In addition, p-values associated with the model will be lower than they should be; this could cause us to erroneously conclude that a parameter is statistically significant. In short, if the error terms are correlated, we may have an unwarrented sense of confidence in our model. Look into time series analysis for a better illustrations of this phenomena and how statistical analysts deal with these problems. 

##3. Non-constant Variance of Error terms:
#another important assumption of the linear regression model is that the error terms have a constant variance, Var(E_i) = sigma^2. The standard errors, confidence intervals, and hypothesis tests associated with the linear model rely upon this assumption. 
#Unfortumately, it is often the case that the variances of the error terms are non-constant. For instance, the variances of the error terms may increase with the value of the response. One can identify non-constant variances in the errors, or heteroscedasticity, from the presence of a funnel shape in the residual plot. When faced with this problem, one possible solution is to transform the response Y using a concave function such as logY (most likely for increasing variance with increases in the value of X) or sqrt(Y) (for decreasing variance with decreases in the value of X).Such a transformation results in a greater amount of shrinkage of the larger response, leading to a reduction in heroscedasticity. Sometimes we have a good idea of the variance of each response. For example, the ith response could be an average of n_i raw observations. If each of these raw observations is uncorrelated with variance sigma^2, then their average has variance sigma^2_i = sigma^2 / n_i. In this case a simple remedy is to fit our model by weighted least squares, with weights proportional to the inverse variances --- i.e. w_i = n_i in this case.

##4. Outliers 
#An outlier is a point for which y_i is far from the value predicted by the model. In this case, removing the outlier has little effect on the least squares line: it leads to almost no change in the slope, and a miniscule reduction in the intercept. It is typical for an outlier that does not have an unusual predictor value to have little effect on the least squares fit. However, even if an outlier does not have much effect on the least squares fit, it can cause other problems. For instance, in this example, the RSE is 1.09 when the outlier is included in the regression, but it is only 0.77 when the outlier is removed. Since the RSE is used to compute all confidence intervals and p-values, such a dramatic increase cased by a single data point can have implications for the interpretation of the fit. Similarly, inclusion of the outlier cases the R^2 to decline from 0.892 to 0.805 (In this case you might want to use imputation but then again I believe this method can only be used on NA values and not outliers). 

#Residual plots can be used to identify outliers. But in practice, it can be difficult to decide how large a residual needs to be before we consider the point to be an outlier. To address this problem, instead of plotting the residuals, we can plot the studentized residuals, computed by dividing each residual e_i by its estimated standard error. Observations whose studentized residuals are greater than 3 in absolute value are possible outliers. 
#If we believe that an outlier has occurred due to an error in data collection or recording, then one solution is to simply remove the observation. However, care should be taken, since an outlier may instead indicate deficiency with the model, such as a missing predictor. 

##5. High leverage points:
#Observations with high leverage have an unusual value of x_i. High leverage observations tend to have a sizable impact on the estimated regression line. It is cause for concern if the least squares line is heavily affected by just a couple of observations, because any problems with these points may invalidate the entire fit. 
#In a simple linear regression, high leverage observations are fairly easy to identify, since we can simply look for observations for which the predictor value is outsider of the normal range of the observations. But in a multiple linear regression with many predictors, it is possible to have an observation that is well within the range of each individual predictor's values, but that is unusual in terms of the full set of predictors. 
#For the important leverage statistic formula used for simple linear regression models make sure to check page 98. 

#It is clear from this equation that h_i increases with the distance of x_i from x_-(In other words the sample mean). There is a simple extension of h_i to the case of mulitple predictors, though we do not provide the formula here. The leverage statistic h_i is always between 1/n and 1, and the average leverage for all the observations is always equal to (p+1)/n. So if a given observation has a leverage statistic that greatly exceeds (p+1)/n, then we may suspect that the corresponding point has high leverage. 

##6. collinearity:
#Collinearity refers to the situation in which two or more predictor variables are closely related to one another. The concept of collinearity is illustrated by the following graphic that uses the Credit dataset. 
par(mfrow = c(1,2))
plot(Credit$Limit, Credit$Age, ylab = "Age", xlab = "Limit", xlim =c(0,13000))
plot(Credit$Limit, Credit$Rating, ylab = "Rating", xlab = "Limit", xlim = c(0, 13000))
#In the left-hand panel of the preceding graphic, the two predictors limit and age appear to have no obvious relationship. In contract, in the right hand panel of the graphic, the predictors limit and rating are very highly correlated with each other, we say that they are collinear. The presence of collinearity can pose problems in the regression context, since it can be difficult to separate out the individual effects of collinear variables on the response. In other words, wince limit and rating tend to increase or decrease together, it can be difficult to dataermine how each one separately is associated with the response, balance. 
credit.fit1 <- lm(Balance ~ Age + Limit, Credit)
credit.fit2 <- lm(Balance ~ Rating + Limit, Credit)
summary(credit.fit1)
summary(credit.fit2)
		#table 3.11 from page 101: The results for two multiple regression models involving the Credit data set are shown. Model 1 is a regression of balance on age and limit, and model 2 a regression of balance on rating and Limit. The standard error of beta_hat_limit increases 12 fold in the second regression, due to collinearity. 
		
#since collinearity reduces the accuracy of the estimates of the regression coefficients, it causes the standard error for beta_hat_j to grow. Recall that the t-statistic for each predictor is calculated by dividing beta_hat_j by its standard error. Consequently, collinearity results in a decline in the t-statistic. As a result, in the presence of collinearity, we may fail to reject H_0: beta_j = 0. This means that the power of the hypothesis test -- the probability of correctly detecting a non-zero coefficient -- is reduced by collinearity.
#In the second model (located in the preceding table), the collinearity between limit and rating has caused the standard error for the limit coefficient estimate to increase of a factor of 12 and the p-value to increase to 0.701. In other words, the importance of the limit variable has been masked due to the presence of collinearity. 

#A simple way to detect collinearity is to look at the correlation matrix of the predictors. An element of this matrix that is large in absolute value indicates a pair of highly correlated variables, and therefor e a collinearity problem in the data. Unfortunately, not all collinearity problems can be detected by inspection of the correlation matrix: it is possible for collinearity to exist between three or more variables even if no pair variables has a particularly high correlation. We call this situation multicollinearity. Instead of inspecting the correlation matrix, a better way to assess multicollinearity is to compute the variance inflation factor (VIF). The VIF is the ratio of the variance of beta_hat_j when fitting the full model divided by the variance of beta_hat if fit on its own. the smallest possible value for VIF is 1, which indicates the complete absence of collinearity. Typically in practice there is a small amount of collinearity among the predictors. As a rule of thumb, a VIF value that exceeds 5 to 10 indicates a problematic amount of collinearity. The VIF for each variable can be computed using the formula:
			#VIF(beta_hat_j) = 1 / 1-R^2_X_j|X_-j
#where R^2_X_j|X_-j is the R^2 from a regression of X_j onto all of the other predictors. If R^2_X_j|X_-j is close to one, then collinearity is present, and so the VIF will be large. 
#In the Credit data, a regression of balance on age, rating, and limit  indicates that the predictors have VIF value of 1.01, 160.67, and 160.59. 
#When faced with the problem of collinearity, there are two simple solutions. The first is to drop one of the problematic variables from the regression. This can usually be done without much compromise to the regression fit, since the presence of collinearity implies that the information that this variable provides about the response is redundant in the presence of the other variables. The second solution is to combine the collinear variables together into a single predictor. For instance, we might take the average of standardized versions of limit and rating in order to create a new variable that measures credit worthiness. 

## 3.5 Comparison of Linear Regression with K-Nearest Neighbors:
#Will need to find out how the author created the KNN model graphic in figure 3.16. Very interesting graphic. In addition, it seems that the smoothing value is called k (which means the number of groups the data will be divided into will need to learn more about this. More groups translates into a smoother fit).

#the KNN regression method is closely related to the KNN classifier discussed in chapter 2. Given a value for K and a prediction point x_0, KNN regression first identifies the K training observatiosn that are closest to x_0, represented by N_0. 
			#Look at page 105 to see the formula for KNN fits on a data set with p = 2 predictors. 
			
#the optimal value for K will depend on the bias-variance tradeoff , which we introduced in chapter 2. A small value for K provides the most flexible fit, which will have low bias but high variance (since the model is trying to intersect with every observation in the dataset). A small value for K provides the most flexible fit, which will have low bias but high variance. In contrast, larger values of K provide a smoother and less variable fit; the prediction in a region is an average of several points, and so changing on observation has a smaller effect. However, the smoothing may cause bias by masking some of the structure in f(X).

#In what setting will a parametric approach such as least squares linear regression outperform a non-parametric approach such as KNN regression? The parametrix approach will outperform the non-parametrix approach if the parametrix form that has been selected is close to the true form of f. In the case of data that actually has a linear trend: a non-paramtric approach incurs a cost in variance that is not offset by a reduction in bias. 

#The increase in dimension has only caused a small deterioration in the linear regression test set MSE, but it as caused more than a ten-fold increase in the MSE for KNN. This decrease in performance as the dimension increases is a common problem in KNN, and results from the fact that in higher dimensions there is effectively a reduction in sample size. In this data set there are 100 training observations; when p = 1, this provides enough information to accurately estimate f(X). However, spreading 100 observations over p = 20 dimensions results in a phenomenon in which a given observation has no nearby neighbors --- this is the so-called curse of dimensionality. That is, the K observations that are nearest to a given test observation x_0 may be vary far away from x_0 in p-dimensional space when p is large, leading to a very poor prediction of f(x_0) and hence a poor KNN fit. As a general rule, parametric methods will tend to outperform non-parametrix approaches when there is a small number of observations per predictor. 

## 3.6 Lab: Linear Regression:
library(MASS)
library(ISLR)

## 3.6.2 Simple linear regression
#The MASS library contains the Boston data set, which records medv (median house valu) for 506 neightborhoods around Boston. We will seek to predict medv using 13 predictors such as rm (average number of rooms per house), age (average age of houses), and lstat (percent of households with low socioeconomic status).
fix(Boston)# again this function is just like the view() function within the Rstudio program. Note to self when this function is activated the R console won't be able to process any other functions.
names(Boston)

#We will start by using the lm() function to fit a simple linear regression model, with medv as the response and lstat as the predictor. 
lm.fit <- lm(medv ~ lstat, data = Boston)
lm.fit #through just simply calling the formula object within the R console the program instinctively outputs the coefficients for the predictor variable and the response variable. 
summary(lm.fit)#Through the use of the summary() function you can see other descriptive statistics making up the regression formula. 

#We can use the names() function in order to find out what other pieces of information are stored in lm.fit. Although we can extract these quantities by name (lm.fit$coefficients) it is safer to use the extractor functions like coef() to access them.
names(lm.fit)
coef(lm.fit)

#Inorder to obtain a confidence interval for the coefficient estimates, we can use the confint() command.
confint(lm.fit)

#the predict() function can be used to produce confidence intervals and prediction intervals for the prediction of medv for a given value of lstat. 
pred1 <- predict(lm.fit, data.frame(lstat = (c(1:60))), interval = "confidence")#No way the newdata argument command actually worked I believed that the (c()) syntax will through off the function.

pred2 <- predict(lm.fit, data.frame(lstat = (c(1:60))), interval = "prediction")

par(mfrow = c(1,2))
plot(Boston$medv, Boston$lstat, main = "prediction interval")
lines(c(1:60),pred2[,3] , lty = 2, col = "red")
lines(c(1:60), pred2[,2], lty = 2, col = "red")
lines(c(1:60), pred2[,1], lty = 1, col = "black")
plot(Boston$medv, Boston$lstat, main = "confidence interval")
lines(c(1:60), pred1[,3], lty = 2, col = "red")
lines(c(1:60), pred1[,2], lty = 2, col = "red")
lines(c(1:60), pred1[,1], lty = 1, col = "black")
#That's really weird the confidence interval is narrower than the prediction interval. Will need to see why this is the case for this problem. There must be a large error value some where within the formula.

#For instance, the 95 percent confidence interval associated with a lstat value of 10 is (24.47, 25.63), and the 95 percent prediction interval is (12.828, 37.28). As expected, the confidence and prediction intervals are centered around the same point (a prediction value of 25.05 for medv when lstat equals 10), but the latter are substantially wider. 

#We will now plot medv and lstat along with the least squares regression line using the plot() and abline() functions.
attach(Boston)
plot(lstat, medv)
#the abline() can be used to drawn any line, not just the least squares regression line. to draw a line with intercept a and slope b, we type abline(a,b). Below we experiment with some additional settings for plotting lines and points. The lwd = 3 command causes the width of the regression line to be increased by a factor of 3; this works for the plot() and lines() functions also. We can also use the pch option to create different plotting symbols.

abline(lm.fit, lwd = 3)
abline(lm.fit, lwd = 3, col = "red")
plot(lstat, medv, col = "red")
plot(lstat, medv, pch = 20)
plot(lstat, medv, pch = "+")
plot(1:20, 1:20, pch = 1:20)

#Next we examine some diagnostic plots, several of which were discussed in Section 3.3.3. Four diagnostic plots are automatically produced by applying the plot() function directly to the output from lm(). In general, this command will produce one plot at a time, and hitting Enter will generate the next plot. However, it is often convenient to view all four plots together. We can achieve this by using the par() function, which tells r to split the display screen into separate anels so that multiple plots can be viewed simultaneously. For example, par(mfrow = c(2,2)) divides the plotting region into a 2 by 2 grid of panels. 
par(mfrow = c(2,2))
plot(lm.fit)# No way that's so cool all of the diagnostic plots are printed onto one device through this method. Will need to remember this functionality. 

#Alternatively, we can compute the residuals from a linear regression fit using the residuals() function. The funciton rstudent() will return the studentized residuals, and we can use the function to plot the residuals against the fitted values. 
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

#On the basis of the residual plots, there is some evidence of non-linearity. Leverage statistics can be computed for any number of predictors using the hatvalues() function.
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))
#the which.max() function identifies the index of the largest element of a vector. In this case, it tells us which observation has the largest leverage statistic. 

##3.6.3 Multiple Linear Regression:
lm.fit <- lm(medv~ lstat + age, data = Boston)
summary(lm.fit)# The R squared value still hasn't gone up from the inclusion of the age variable within the model. And compared to the lstat, the age variable has a noticeably larger p-value (but still the age variable p-value is still well under the 0.05 significance level).

#The Boston data set contains 13 variables, and so it would be cumbersome to have to type all of these in order to perform a regression using all of the predictors. Instead, we can use the following short-hand:
lm.fit <- lm(medv~., data = Boston)
summary(lm.fit)

#We can access the individual components of a summary object by name (type summary.lm to see what is available). Hence summary(lm.fit)$r.sq gives us the R squared, and summary(lm.fit)$sigma gives us the RSE. The vif() function, part of the car package, can be used to compute variance inflation factors. Most VIF's are low to moderate for this data. 
library(car)
vif(lm.fit)

#What if we would like to perform a regression using all of the variables but one? For example, in the above regression output, age has a high p-value. So we may wish to run a regression excluding this predictor. The following syntax results in a regression using all predictors except age. 
lm.fit1 <- lm(medv ~ .-age, data = Boston)
summary(lm.fit1)
#Alternatively, the update() function can be used. 

lm.fit1 <- update(lm.fit, ~.-age)
lm.fit2 <- update(lm.fit1, ~.-indus)# Taking out the indus variable since it is the last predictor that displays a large p-value. 
summary(lm.fit2) 

## 3.6.4 Interaction terms:
#It is easy to include interaction terms in a linear model using the lm() function. The syntax lstat:black tells R to include an interaction term between lstat and black. The syntax lstat*age simultaneously includes lstat, age, and the interaction term lstat*age as predictors; it is a shorthand for lstat+age+lstat:age.
summary(lm(medv~lstat*age, data = Boston))
summary(lm(medv~lstat+age,data = Boston))

##3.6.5 Non-linear transformations of the predictors:
#The lm() function can also accommodate non linear transformations of the predictors. For instance, given a predictor X, we can create a predictor X^2 using I(X^2). The function I() is needed since the ^ has a special meaning in a formula; wrapping as we do allows the standard usage in R, which is to raise to the power 2. We now perform a regression of medv onto lstat and lstat^2.
lm.fit2 <- lm(medv ~ lstat + I(lstat^2))
summary(lm.fit2)

#the near-zero p-value associated with the quadratic term suggests that it leads to an improved model. We use the anova() function to further quantify the extent to which the quadratic fit is superior to the linear fit. 
lm.fit <- lm(medv ~ lstat)
anova(lm.fit, lm.fit2)

#Here model 1 represents the linear submodel containing only one predictor, lstat, while Model 2 corresponds to the larger quadratic model that has two predictors, lstat and lstat^2. The anova() function performs a hypothesis test comapring the two models. The null hypothesis is that the two models fit the data equally well, and the alternative hypothesis is that the full model is superior (or rather that the two models are different which points to the second model being the most superior). Here the F-statistic is 135 and the associated p-value is virtually zero. This provides very clear evidence that the model containg the predictors lstat and lstat^2 is far superior to the model that only contains the predictor lstat. This is not surprising, since earlier we saw evidence for non-linearity in the relationship between medv and lstat. If we type:
par(mfrow = c(2,2))
plot(lm.fit2)

#then we see that when the lstat^2 term is included in the model, there is little discernible pattern in the residuals.
#In order to create a cubic fit, we can include a predictor of the form I(X^3). However, this approach can start to get cumbersome for higher order polynomicals. A better approach involves using the poly() function to create the polynomial within lm(). For example, the following command produces a fifth order polynomial fit:
lm.fit5 <- lm(medv~poly(lstat, 5))
summary(lm.fit5)
plot(lstat, medv)
pre.fit5 <- predict(lm.fit5, newdata = data.frame(lstat = (c(1:50))), interval = "prediction")
lines(c(1:50), pre.fit5[,1])#that actually looks like a pretty good fit from the perspective of the actual values with the fitted regression line super imposed on the data points. 
par(mfrow = c(2,2))
plot(lm.fit5)

#this suggests that including additional polynomial terms, up to fifth order leads to an improvement in the model fit. However, further investigation of the data reveals that no polynomial terms beyond fifth order have significant p-values in a regression fit. 
#Of course, we are in no way restricted to using polynomial transformations of the predictors. Here we try a log transformation.
summary(lm(medv~log(rm), data = Boston))

##3.6.6 Qualitative predictors 
#We will now examine the Carseats data, which is part of the ISLR library. We will attempt to predict Sales (shild car seat sales) in 400 locations based on a number of predictors.
fix(Carseats)
??Carseats
names(Carseats)

#Given a qualitative variable such as Shelveloc, r generates dummy variables automatically. Below we fit a multiple regression model that includes some interaction terms. 
lm.fit <- lm(Sales ~ . + Income:Advertising +Price:Age, data = Carseats)
summary(lm.fit)

#the contrasts() function returns the coding that R uses for the dummy variables.
detach(Boston)
attach(Carseats)
contrasts(ShelveLoc)
#R has created a ShelveLocGood dummy variable that takes on a value of 1 if the shelving location is good, and 0 otherwise. It has also created a ShelveLocMedium dummy variable that equals 1 if the shelving location is medium, and 0 otherwise. A bad shelving location corresponds to a zero for each of the two dummy variables. The fact that the coefficient for ShelveLocGood in the regression output is positive indicates that a good shelving location is associated with high sales (relative to a bad location). And shelvLocMedium has a smaller positive coefficient, indicating that a medium shelving location leads to higher sales than a bad shelving location but lower sales than a good shelving location. 

##3.6.7 Writing functions:
#Below we provide a simple function that reads in the ISLR and MASS Libraries, called LoadLibraries(). Before we have created the function, R returns an error if we try to call it. 
LoadLibraries <- function(){
	library(ISLR)
	library(MASS)
	print("The libraries have been loaded")
}
LoadLibraries
LoadLibraries()

## 3.7 Exerices:
##conceptual:
#1. The author (with regards to the regression model illustrated in figure 3.4) table 3.4 displays the multiple regressions coefficient when Tv, radio, and newspaper advertising budgets are used to predict product sales using the Advertising data. We interpret these results as follows:for a given amount of TV and newspaper advertising, spending an additional 1000 on radio advertising leads to an increase in sales by approximately 186 units. 
	#Information from table 3.4:
	#beta_0 = 2.939, beta_1 = 0.046, beta_2 = 0.186, and beta_3 = -0.001 
	#Hence when interpreted as a regression model for the advertising dataset:
		# Sales = 2.939 + 0.049(TV) + 0.186(radio) - 0.001(newspaper) 
#this ultimately means that for every 1,000 dollars spent for tv the units sold will be 49, for radio 186 units, and for newspapers -1 units at a base amount of 2939 units (since the beta_0 value is 2.939). 

#2. Now I understand the K-NN classifier is a simpler off shoot of the beyas classifier in that it uses the K value within the overall equation to group the underlying dataset (setting k to 1 will create k_groups = n_observations while setting k to 5 will equate into n/5). Both the k-NN classifier and the k-NN regression model use this same value. The this is where the similarities cease because the k-NN regression model used to create a regression line (much like the parametric method of linear regression) while the is used to estimate the conditional distribution of Y given X, and then classify a method given observation to the class with highest estimated probability. In addition the k-NN classifier (as well as the Bayes classifier) is used to calculate the irreducible error of the model's distribution (this is called the test error rate).

#3. (a) From looking at the model the only possible answer is explaination iv. since the regression for females can be conceptualized as income = 50 + 20(GPA) + 0.07(IQ) + 35(Female) + 0.01(GPA * IQ) - 10 (GPA * Female) as for males income = 50 + 20(GPA) + 0.07(IQ) + 0.01(GPA * IQ). then again I might have to rethink this answer because I didn't really comprehend that the author used the dummy code technique to illustrate this regression model using the categorical variable gender. Will really need to brush up on my regression model interpretation. 
# According to asadoughi the actual answer is statement iii. Again I really need to look into my regression interpretation.

#(b). According to my calculation using the regression model and my small amount of college alegebra training the value is 137.1 (which is impossible since the scale is thousands of dollars meaning that this value translates into 137100).  
50 + 20*(4.0) + 0.07*(110) + 35 + 0.01*(4.0 * 110) - 10*(4.0 * 1)
#No way asadoughi obtained the same answer. Awesome.

#(c). We can't really make that assessment since small beta_j value has no barring on the actual p-value of the interestion between these two variables. Perhaps the interaction has high statistical significance and the beta coefficient just happens to be small. 

#4. (a) Asadoughi's answer: I would expect the polynomial regression to have a lower training RSS than the linear regression because it could make a tighter fit against data that matched with a wider irreducible error (Var(epsilon)).

#(b) converse to (a), I would expect the polynomial regression to have a higher test RSS as the overfit from training would have more error than the linear regression. 

#(c) Since the actual fit of the model is none linear the cubic transformation would fit the data the best hence this will bring about a higher RSS value. 

#the actual answer according to anasdoughi is there is not enough information to tell which test RSS would be lower for either regression given the problem statement is defined as not knowing "how far it is from linear". If it is closer to linear than cubic, the linear regression test RSS could be lower than the cubic regression test RSS. Or, if it is closer to cubic than linear, the cubic regression test RSS could be lower than the linear regression test RSS. It is due to the bias-variance tradeoff: it is not clear what level of flexibility will fit the data better. 

#6. According to Anadoughi the answer is:
# y = B_0 + B_1x
#from (3.4): B_0 = avg(y) - B_1 avg(x)
#right hand side will equal 0 if (avg(y)) is a point on the line.  
# 0 = B_0 + B_1 avg(x) - avg(y)
# 0 = (avg(y) - b_1 avg(x)) + B_1 avg(x) - avg(y)
# 0 = 0

##applied:
#8. (a) 
library(ISLR)
car.fit <- lm(mpg ~ horsepower, Auto)
summary(car.fit)$coefficient
#the H_0 says that beta_1 = 0 while the H_a say that beta1 != 0. This means that the null hypothesis is that there will be no change in mpg with increased horsepower while the alternative hypothesis say that there will be a change in mpg with increased or decreased horsepower. 
# The regression model is:
			# mpg = 39.936 - 0.1578(Horsepower) 
#This means that for every 1 horsepower increase there will be a 0.1578 decrease in mpg at beta_0 value of 39.936

#i. With the p-value being almost equal to zero, statistically there is a relationship between the predictor and the response variables. But still this is only based on the p-value, I still need to check the significance levels and the RSS to see if this model is a good fit for the data.

#ii. Due to the p-value being almost equal to zero, there is a strong relationship between mpg and horsepower. 

#iii. The relationship between mpg and horsepower is negative.

#iv. 
predict(car.fit, newdata = data.frame(horsepower = (98)), interval = "confidence", level = 0.95)
# the result is 24.46708 with a confidence level of 95 percent. This seems really low. We really need to look at how old the data is and perhaps the cylinder variable is throwing off this model.
summary(car.fit)# this model counts for 0.6059 of the underlying movement of mpg. 
summary(lm(mpg ~ horsepower + cylinders, data = Auto))# horsepower is really a statistically significant variable but the R^2 value only moved up 0.05 in relation to the other model. Will need to look into this. Perhaps a quadratic or logarithmic transformation will be a good move for this model. 
predict(car.fit, newdata = data.frame(horsepower = (98)), interval = "prediction", level = 0.95)
#The result is a fit of 24.467, a lower bounds of 14.8094, and an upper bounds of 34.125. 

#(b) 
plot(Auto$horsepower, Auto$mpg, pch = 16, col = "light blue")
abline(car.fit, col = "black") #It seems that the abline function doesn't want to work with this model. Will need to create a prediction object and fill in the resulting values with the line() function call.
car.pre <- predict(car.fit, newdata = data.frame(horsepower = (c(1:300))), interval = "prediction", level = 0.95)
lines(c(1:300), car.pre[,2], lty = 2, col = "red")
lines(c(1:300), car.pre[,3], lty = 2, col = "red")
#this shows that the linear regression model won't be a good fit for the underlying data. Will try out a logarithmic or quadratic transformation withi the data.
par(mfrow = c(2,2))
plot(car.fit)
#there's a high amount of variance in the higher horsepower values. Will need to see the problem this creates. As for the Q-Q plot the model fits part of the data but most likely a I(x^2) transformation will solve this problem.

#Now I understand what the problem is the variance for the response variable is inconsistant throughout the entirety of the dataset. Will most likely have to use the log(Y) or sqrt(Y) technique to fix this problem. This is called heteroscedasticity.
car.fit2 <- lm(log(mpg) ~ horsepower + I(horsepower^2), data = Auto)
par(mfrow = c(2,2))
plot(car.fit2)# This fixed the fit of the model on all four observation plots. 
summary(car.fit2)$coefficient# that's weird the log(Y) transformation messed up the prediction mpg values. Will need to find a way to fix this problem.

#9. (a) 
pairs(Auto)

#(b)
colnames(Auto)
Auto2 <- Auto[-9]
head(Auto2)
cor(subset(Auto2))

#(c) i.
Auto2$origin <- as.factor(Auto$origin)
car.fit <- lm(mpg ~ ., data = Auto2)
summary(car.fit)$coefficient
plot(Auto2$origin, Auto2$mpg)# Interesting so where the vehicle is from really does affect the fuel economy (Which again makes sense considering the large amount of economical cars being produced by Japan and Europe).
car.fit$coef 
#This is a relationship between mpg and displacement, weight, year, and origin (which I recoded as a factor as a means to see which countries make the most economical cars US, Japan, or Europe).

#ii. Displacement, weight, year, and origin appear to have a statistical significance according to a significance level of 0.05.

#iii.
# the coefficient for the year variable suggests that for every year (the year in which the car was manufactured) the fuel economy goes up by 0.777 mpg.

#(d)
par(mfrow = c(2,2))
plot(car.fit)
#the fit seems fine on the lower bounds (according to the qq plot) but the upper bounds of the plot illustrates a curve pattern (illustrating that a quadratic transformation will need to be used). The residuals plot still displays the same funnel shape hence illustrating heteroscedasticity. From the leverage plot, point 14 appears to have high leverage, although not a high magnitude residual (Will need to see what is mean by this statement and will need to practice my interpretation of the leverage plot and studentized residuals)

#Asadoughi's answer:
plot(predict(car.fit), rstudent(car.fit))
#there are possible outliers as seen in the plot of studentized residuals because there are data with a value greater than 3 

#(e) 
car.fit2 <- lm(mpg ~ cylinders*displacement+origin+cylinders+displacement+year:weight, Auto2)
#cylinders * displacement seems to have high p-values. 
summary(car.fit2)

#(f) 
par(mfrow = c(2,2))
car.fit3 <- lm(mpg ~ log(horsepower) + horsepower, data = Auto)
plot(Auto$horsepower,Auto$mpg)
car.pred <- predict(car.fit3, newdata = data.frame(horsepower = (c(1:300))), interval = "confidence", level = 0.95)
lines(c(1:300), car.pred[,1], lty = 2, col = "red")

car.fit4 <- lm(mpg ~ displacement + I(displacement^2), data =Auto)
plot(Auto$displacement, Auto$mpg)
range(Auto$displacement)
car.pred2 <- predict(car.fit4, newdata = data.frame(displacement = (c(1:500))), interval = "confidence", level = 0.95)
lines(c(1:500), car.pred2[,1], col ="red", lty = 2)

car.fit5 <- lm(mpg ~ displacement + sqrt(displacement), Auto)
plot(Auto$displacement, Auto$mpg)
car.pred3 <- predict(car.fit5, newdata = data.frame(displacement = (c(1:500))), interval = "confidence", level = 0.95)
lines(c(1:500), car.pred3[,1], col ="red", lty = 2)

#Anadoughi's answer:
lm.fit2 <- lm(mpg ~ cylinders*displacement+displacement*weight, Auto)
summary(lm.fit2)

#From the correlation matrix, I obtained the two highest correlated paris and used them in picking my interaction effects. From the p-values, we can see that the interaction between displacement and weight is statistically significant, while the interaction between cylinders and displacement is not.

lm.fit3 <- lm(mpg~log(weight)+sqrt(horsepower)+acceleration + I(acceleration^2), Auto)
summary(lm.fit3)
par(mfrow = c(2,2))
plot(lm.fit3)

lm.fit2 <- lm(log(mpg) ~ cylinders + displacement +horsepower+weight+acceleration+year+origin, data = Auto)
summary(lm.fit2)
par(mfrow = c(2,2))
plot(lm.fit2)
plot(predict(lm.fit2),rstudent(lm.fit2))

#10. (a) 
head(Carseats, n = 20)
??Carseats 
colnames(Carseats)
str(Carseats)
carseat.fit <- lm(Sales ~ Price + Urban + US, data = Carseats)

#(b) What this model says is that starting with a beta_0 value of 13034 units; increased price will decrease carseat sales by 54, if the population lives in an urban environmnet carseat sales will decrease by 22 units, and lastly if the population is within the us market carseat purchases will increase by 1201 units. 

#(c)
summary(carseat.fit)# Interestingly the Urban Yes qualitative variable has a very high p-value while the other three variables have p-values well below the 0.05 significance level. 
#equation:
		#Sales ~ 13.043 - 0.05446(price) - 0.021916(Urban Yes) + 1.2006 (US yes) 
carseat.fit$coef

#(d) I can rejection the null hypothesis of beta_j =0 for the Price and US variables. But on that note I can't reject the null hypothesis for the Urban variable since the p-value was calculated at well over 0.05 or 0.10 significance level.

#(e) 
carseat.fit2 <- lm(Sales ~ Price + US, Carseats)

#(f) 
plot(Carseats$Sales, Carseats$Price, col = c("green","blue")[Carseats$US], pch = 16)
legend("topright", legend = c("No","Yes"), pch = c(16,16), col = c("green","blue"))
# Will think about how to fix this graphic later. I just realized that for all of the regression models that I have been predicting for all of the exercises in this chapter, I have been forgetting to include the other variables in the model through the newdata argument. I hope that this oversight doesn't mess up all of my regression predictions. 
par(mfrow = c(2,2))
plot(carseat.fit)
dev.new()
par(mfrow = c(2,2))
plot(carseat.fit2)

summary(carseat.fit)
summary(carseat.fit2)# despite removing the statistically weak variable (Urban) the regression line fits still look very familiar according to all diagnostic plots and checks. In fact the p-values are all the same for both models. With that said though, the QQ plot weirdly reads that all the data points follow the trend of the regression model line. I hope that there isn't anything wrong with my regression models or the way I'm coding them in the concole.

#(g) Anadoughi's answer:
confint(carseat.fit2)
plot(Carseats$Price, Carseats$Sales, col = c("green","blue")[Carseats$US], pch = 16)
abline(confint(carseat.fit2)[,1], lty = 2, col ="red")# I guess graphing the confidence intervals through this method doesn't work.

#experimentation (will need to see if including one variable will inhibit the predict() function's ability to calculate the confidence interval lines and the regression fit line)
carseat.pred <- predict(carseat.fit2, newdata = data.frame(Price = (c(1:210)), US = (rep(c("No","Yes"), times = 210))), interval = "confidence", level = 0.95)
par(mfrow = c(1,2))
plot(Carseats$Price, Carseats$Sales, col = c("green","blue")[Carseats$US], pch = 16, main = "confidence interval through the predict() function")
legend("topright", legend = c("No","Yes"), col = c("green","blue"), pch = c(16,16))
lines(c(1:420), carseat.pred[,2], lty = 2, col = "red")# I goes that from a number of standpoints this method will not work because I' attempting to graph a three dimensional model into a two dimensional shape. I gues that a more intelligent method would be to graph the linear model onto a three dimensional plot or contour map. The problem with this is that I don't remember how to make persp() plots using data from a qualitative variable. Will need to find a way how to solve this problem. 

#(h.) Anadoughi's solution
plot(predict(carseat.fit2), rstudent(carseat.fit2))
#All studentized residuals appear to be bounded by -3 to 3, so no potential outliers are suggested from the linear regression model.
par(mfrow = c(2,2))
plot(carseat.fit2)#According to Anadoughi's analysis of the studentized residual plot, some data points have high leverage. 

#11.
set.seed(1)
x <- rnorm(100)
y <- 2*x+rnorm(100)

#(a)
lm(y~x+0)# You really can perform a regression without an intercept.
plot(x, y)
abline(lm(y~x+0))
summary(lm(y~x+0))
#the p-value for the beta_hat value is almost valued at zero, meaning that the beta value is indeed statistically significant. 
#The st.error is 0.1065, the t-statistic is 18.73, and the coefficient estimate for beta_hat is 1.9939.
# An interesting addition is that the R^2 value is only calculated at 0.7798. Will need to see if this is caused from not having a y intercept.

#(b) 
lm(x ~ y + 0)
summary(lm(x ~ y + 0))
par(mfrow = c(1,2))
plot(x,y)
abline(lm(y ~ x + 0))
plot(y,x)
abline(lm(x~y + 0)) 
# The t-statistic remained at 18.73, the standard error decreased to 0.02089, the regression line for y still has a p-value of almost zero so the variable is statistically significant, and the R^2 value is still calculated at 0.7798.

#(c) Anadoughi
#Both results in (a) and (b) reflect the same line created in 11a. In other words, both of the regression models are interchangeable.

#(d) Anadoughi solution:
(sqrt(length(x) - 1) * sum(x*y)) / (sqrt(sum(x*x) * sum(y*y) - (sum(x*y))^2))
#No way this equation actually calculated the t-statistic for the exercises above.
#this is the same as the t-statistic shown above.

#(e)
#If you swap t(x,y) as t(y,x), then you will find t(x,y) = t(y,x).

#(f) 
lm.fit <- lm(y ~ x)
lm.fit2 <- lm(x ~y)
summary(lm.fit)
summary(lm.fit2)
# The t-statistic for beta_1 is indeed the same for the two regression models. But what about the difference in beta_0 between the two models. Will need to look into this.

#12. (a) Anadoughi's solution
#When the sum of the squares of the observed y-values are equal to the sum of the squares of the observed x-values.

#(b)
set.seed(1)
x = rnorm(100)
y = 2 * x
lm.fit <- lm(y~x+0)
lm.fit2 <- lm(x~y + 0)
summary(lm.fit)
summary(lm.fit2)
#The regression coefficients are different for each linear regression.

#(c)
set.seed(1)
x <- rnorm(100)
y <- sample(x, 100)
sum(y^2)
sum(x^2)
lm.fit <- lm(y ~ x + 0)
lm.fit2 <- lm(x ~ y + 0)
summary(lm.fit)
summary(lm.fit2)
#the regression coefficients are the same for each linear regression. So long as sum sum(x^2) = sum(y^2) the condition in 12a. will be satisfied. 

#13. (a)
set.seed(1)
x <- rnorm(n = 100)

#(B) 
eps <- rnorm(100, 0, sqrt(0.25))

#(c) 
#I believe that the beta_0 value is -1 and the X value is object x. As for eps I'm not really sure where its place should be in the equation. 
# Now I understand, the eps is actually the irreduciable error value within the equation and dataset.
y <- -1 + 0.5*x + eps
length(y)# the length of y is 100 and actually the beta_1 value is 0.5 since beta-1 is actually the slope of the regression model and the X is only the predictor variables for the dataset.

#(d)
plot(x, y)# from looking at the scatter plot I can say that the data has a positive trend, but describing this trend as linear (or rather parametric) will be too much of a stretch. Will need to see the analysis plots to see for sure.

#(e)
range(x) 
lines(c(seq(-2.215, 2.4016, length.out = 100)), y, lty = 2, col = "red")
#this isn't actually the regression model for this line. It seems to be something else entirely. Will need to see what I'm missing.

lm.fit <- lm(y ~x)
summary(lm.fit)
# the beta_hat_0 and beta_hat_1 values for the regression model comforms pretty well with the actual population beta_0 and beta_1 with only a difference of give or take 0.01. This must be the irreduceable error that the author was talking about. 
plot(x,y)
abline(lm.fit, lty = 2, col ="red")

#(f).
plot(x, y)
abline(lm.fit, lwd = 2, col = "red")
abline(-1, 0.5, lwd = 2, col = "blue")
legend(-1, legend=c("model fit","population fit"), col = c("red","blue"), lwd=2)# cool I didn't know you can customize the placement of the legend box through the legend() function will need to look more into this.

#(g)
lm.fit2 <- lm(y ~ x + I(x^2))
summary(lm.fit)
summary(lm.fit2)# the statistical significance of the I(x^2) or rather the quadratic transformation is above the significance level of 0.10. This means that I(x^2) improving the fit of the regression line for this dataset is statistically weak. The trend is actually more linear in character.
par(mfrow = c(2,2))
dev.new()
plot(lm.fit2)
plot(lm.fit)
# After looking at the qqplot for both of the models I have to say that the quadratic transformation model has the best fit regardless of the low statistical significance.

#(h) anadoughi's solution
set.seed(1)
eps1 <- rnorm(100, 0, 0.125)
x1 <- rnorm(100)
y1 <- -1 + 0.5*x1 + eps1
plot(x1, y1)
lm.fit1 <- lm(y1 ~ x1)
summary(lm.fit1)
abline(lm.fit1, lwd = 3, col = 2)
abline(-1, 0.5, lwd = 3, col = 3)
legend(-1, legend = c("model fit", "population fit"), col = 2:3, lwd = 3)
# As expected, the error observed in R^2 and RSE decreases considerably. And the population line and the regression sample line are looking very much the same.

#(i) 
set.seed(1)
eps2 <- rnorm(100, 0, 0.6)
x2 <- rnorm(100)
y2 <- -1 + 0.5*x2 + eps2
plot(x2, y2)
lm.fit2 <- lm(y2 ~ x2)
summary(lm.fit2)
plot(x2, y2)
abline(-1, 0.5, lwd = 3, col =2)
abline(lm.fit2, lwd = 3, col = 3)
legend("topright", legend = c("population fit", "model fit"), col = 2:3, lwd = 3)
# as expected the the data points are more scattered within the graphic and the population line and the regression model line are further apart due to the extra variance in the model.

#(j) Anadoughi's solution
confint(lm.fit)
confint(lm.fit1)
confint(lm.fit2)
#All intervals seem to be centered on approximately 0.5, with the second fit's interval being narrower than the first fit's interval and the last fit's interval being wider than the first fit's interval.

#14. (a)
set.seed(1)
x1 <- runif(100)
x2 <- 0.5*x1+rnorm(100)/10
y = 2+2*x1+0.3*x2+rnorm(100)

#(b)
cor(x1,x2)
plot(x1, x2)

#(c)
lm.fit <- lm(y ~ x1 + x2)
summary(lm.fit)
#the beta_hat_0 coefficient was calculated at 2.1305, the beta_hat_1 at 1.4396, and beta_hat_2 at 1.0097. The only beta X variable that can reject the null hypothesis just barely was x1 or rather beta_1 with a p-value of 0.0487 (which is .012 under the 0.05 significance level cutoff). While the x2 variable has a statistical significance of 0.3754, meaning that the variable should be dropped from the model. The standard errors are 0.2319, 0.7212, and 1.1337 respectively.

#(d)
lm.fit2 <- lm(y ~ x1)
summary(lm.fit2)
#through not including the x2 variable, the x1 variable has a p-value of 2.66e-06 meaning that this particular variable has a fair amount of statistical significance. Hence meaning that the null hypothesis can be rejected.

#(e)
lm.fit3 <- lm(y ~ x2)
summary(lm.fit3)
# Interestingly the x2 variable when included into a simple linear regression model returned back a p-value of 1.37e-05, meaning that the null hypothesis can be rejected in this model as well. beta-2 != 0

#(f) The two answers in exercise c and e do contradict each other in the realm of p-values for beta-1 and beta_2. The p-values for these two variables were much higher together then apart. Will need to find out why this is the case. Very interesting indeed.

#anaboughi's solution:
#no, because x1 and x2 have collinearity, it is hard to distinguish their effects when regressed upon together. When they are regressed upon separately, the linear relationship between y and each predictor is indicated more clearly.

#(g) anadoughi's solution 
x1 <- c(x1, 0.1)
x2 <- c(x2, 0.8)
y <- c(y, 6)
lm.fit1 <- lm(y ~ x1 + x2)
summary(lm.fit1)
lm.fit2 <- lm(y~x1)
summary(lm.fit2)
lm.fit3 <- lm(y ~ x2)
summary(lm.fit3)
#In the first model, it shifts x1 to statistically insignificance and shifts x2 to statistically significance from thechange in p-values between the two linear regressions.
par(mfrow = c(2,2))
plot(lm.fit1)
par(mfrow = c(2,2))
plot(lm.fit2)
plot(lm.fit3)
#In the first and third models, point 101 becomes a high leverage point.
plot(predict(lm.fit1), rstudent(lm.fit1))
plot(predict(lm.fit2), rstudent(lm.fit2))
plot(predict(lm.fit3), rstudent(lm.fit3))
#Looking at the studentized residuals, we don't observe points too far from the absolute value 3 cutoff, except for the second linear regression: y ~ x1.

#15. (a) anaboughi's solution 
library(MASS)
summary(Boston)
Boston$chas <- factor(Boston$chas, labels = c("N","Y"))
summary(Boston)
attach(Boston)
lm.zn = lm(crim~zn)
summary(lm.zn) # yes
lm.indus = lm(crim~indus)
summary(lm.indus) # yes
lm.chas = lm(crim~chas) 
summary(lm.chas) # no
lm.nox = lm(crim~nox)
summary(lm.nox) # yes
lm.rm = lm(crim~rm)
summary(lm.rm) # yes
lm.age = lm(crim~age)
summary(lm.age) # yes
lm.dis = lm(crim~dis)
summary(lm.dis) # yes
lm.rad = lm(crim~rad)
summary(lm.rad) # yes
lm.tax = lm(crim~tax)
summary(lm.tax) # yes
lm.ptratio = lm(crim~ptratio)
summary(lm.ptratio) # yes
lm.black = lm(crim~black)
summary(lm.black) # yes
lm.lstat = lm(crim~lstat)
summary(lm.lstat) # yes
lm.medv = lm(crim~medv)
summary(lm.medv) # yes

#(b)
Boston.fit <- lm(crim ~ ., data = Boston)
summary(Boston.fit)
#The predictors that we can remove are tax, lstat, ptratio, tax, age, rm, nox, chasY, and Indus.
# Meaning that the only values that can reject the null hypothesis at a significance level of 0.05 are zn, nox, dis, rad, black, and medv.

#(c)
x <- c(coefficients(lm.zn)[2],
      coefficients(lm.indus)[2],
      coefficients(lm.chas)[2],
      coefficients(lm.nox)[2],
      coefficients(lm.rm)[2],
      coefficients(lm.age)[2],
      coefficients(lm.dis)[2],
      coefficients(lm.rad)[2],
      coefficients(lm.tax)[2],
      coefficients(lm.ptratio)[2],
      coefficients(lm.black)[2],
      coefficients(lm.lstat)[2],
      coefficients(lm.medv)[2])
y = coefficients(Boston.fit)[2:14]
plot(x, y)

#Coefficient for nox is approximately -10 in univariate model and 31 in multiple regression model.

#(d)
lm.zn = lm(crim~poly(zn,3))
summary(lm.zn) # 1, 2
lm.indus = lm(crim~poly(indus,3))
summary(lm.indus) # 1, 2, 3
# lm.chas = lm(crim~poly(chas,3)) : qualitative predictor
lm.nox = lm(crim~poly(nox,3))
summary(lm.nox) # 1, 2, 3
lm.rm = lm(crim~poly(rm,3))
summary(lm.rm) # 1, 2
lm.age = lm(crim~poly(age,3))
summary(lm.age) # 1, 2, 3
lm.dis = lm(crim~poly(dis,3))
summary(lm.dis) # 1, 2, 3
lm.rad = lm(crim~poly(rad,3))
summary(lm.rad) # 1, 2
lm.tax = lm(crim~poly(tax,3))
summary(lm.tax) # 1, 2
lm.ptratio = lm(crim~poly(ptratio,3))
summary(lm.ptratio) # 1, 2, 3
lm.black = lm(crim~poly(black,3))
summary(lm.black) # 1
lm.lstat = lm(crim~poly(lstat,3))
summary(lm.lstat) # 1, 2
lm.medv = lm(crim~poly(medv,3))
summary(lm.medv) # 1, 2, 3


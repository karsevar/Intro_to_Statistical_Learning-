### Chapter 4 Classification:
#The linear regression model discussed in chapter 3 assumes that the response variable Y is quantitative. Often qualitative variablses are referred to as categorical; we will use these terms interchangeably. In this chapter, we study approaches for predicting qualitative response for an observation can be referred to as classifying that observation, since it involves assigning the observation to a category, or class. On the other hand, often the methods used for classification first predict the probability of each of the categories of a qualitative variable, as the basis for making the classification. In this sense they also behave like regression methods. 

#In this chapter we discuss three of the most widely used classifiers: logistic regression, linear discriminant analysis, and K-nearest neightbors. 

##4.1 An overview of classification:
#In this chapter, we will illustrate the concept of classification using the simulated Default data set. We are interested in predicting whether an individual will default on his or her credit card payment, on the basis of annual income and monthly credit card balance. 
library(ISLR)
Default# Cool the Default data set is attached to the ISLR package. Will need to see if this is the same data set as the one in the book. 
colnames(Default)# four variables within the data set. 
nrow(Default)# And there are a total of 10000 observations.
levels(Default$default)
par(mfrow = c(1,2)) 
plot(Default$balance, Default$income, ylab = "Balance", xlab = "Income", col = c("blue","red")[Default$default], pch = 16)
legend("topright", legend = c("no","yes"), col = c("blue","red"), pch = 16)# graphic representation on page 129 labeled as figure 4.1 in the book. 
boxplot(x = Default$default, y = Default$balance, col = c("blue","red")[Default$default])# It's funny that I forgot how to use the base boxplot() function. The boxplot below is the ggplot2 boxplot function alternative.
library(ggplot2)
ggplot(Default, aes(x = default, y= balance, col = default)) + geom_boxplot()
ggplot(Default, aes(x = default, y = income, col = default)) + geom_boxplot()
#For the graphic above we have plotted annual income and monthly credit card balance for a subset of 10000 individuals. The left-hand panel displays individuals who defaulted in a given month in red and those who did not in blue. (the overall default rate is about 3 percent, so we have plotted only a fraction of the individuals who did not default). It appears that individuals who defaulted tended to have higher credit card balances than those who did not. In the ggplot() functions above we show the distribution of balance split by the binary default variable;the second is a similar plot for income. In this chapter, we learn how to build a model to predict default (Y) for any given value of balance (X_1) and indome( X_2). Since Y is not quantitative, the simple linear regression model of chapter 3 is not appropriate.

#It is worth noting that the figure above displays a very pronounced relationship between the predictor balnace and the response default. In most real applications, the relationship between the predictor and the response will not be nearly so strong. 

## 4.2 Why not linear regression?
#If the response variable's values did take on a natural ordering, such as mild, moderate, and severe, and we felt the gap between mild and moderate was similar to the gap between moderate and severe, then a 1,2,3 coding would be reasonable. Unfortuately, in general there is no natural way to convert a qualitative response variable with more than two levels into a quantitative response that is ready for linear regression.

#For a binary (two level) qualitative response, the situation is better. For instance, perhaps there are only two possibilities for the patient's medical condition: stroke and drug overdose. We could then potentially use the dummy variable approach from section 3.3.1 to code the response as follows:
		#Y = 0 if stroke:
			#1 if drug overdose 
			
#We could then fit a linear regression to this binary response, and predict drug overdose if Y_hat > 0.5 and stroke otherwise. In the binary case it is not hard to show that even if we flip the above coding, linear regression will produce the same final predictions.

#For a binary response with a 0/1 coding as above, regression by least squares does make sense; it can be shown that the X*beta_hat obtained using linear regression is in fact an estimate of Pr(drug overdose| X) in this special case. However, if we use linear regression, some of our estimates might be outside the [0,1] interval, making them hard to interpret as probabilities. Nevertheless, the predictions provide an ordering and can be interpreted as crude probability estimates. Curiously, it turns out that the classifications that we get if we use linear regression to predict a binary response will be the same as for the linear discriminant analysis procedure.

#However, the dummy variable approach cannot be easily extended to accommodate qualitative responses with more than two levels. For these reasons, it is preferable to use a classification method that is truly suited for qualitative response values, such as the ones presented next.

## 4.3 Logistical Regression:
#Consider again the Default data set, where the response default falls into one of two categories, Yes and No. Rather than modeling this response directly, logistic regression models the probability that Y belongs to a particular category. 
ggplot(Default, aes(x = balance)) + geom_freqpoly()# Most likely the author used the freqpoly() function to calculate the probability of default for customers with particular balances. Will need to remember how I mapped logistical regression plots using different predictor values than the default count method.

#For the Default data, logistic regression models the probability of default. For example, the probability of default given balance can be written as 
			#Pr(default =Yes|balance).
#The values of Pr(default = Yes|balance), which we abbreviate p(balnace), will range between 0 and 1. Then for any given value of balance, a prediction can be made for default. For example, one might predict default = Yes for any individual for whom p(balance) > 0.5. Alternatively, if a company wishes to be conservative in predicting individuals who are at risk for default, then they may choose to use a lower threshold, such as p(balance) > 0.1.

##4.3.1 The logistic Model:
#How should we model the relationship between p(x) = Pr(Y = 1/X) and X? (for convenience we are using the generic 0/1 coding for the response).
levels(Default$default) <- c(0,1)
default.num <- as.numeric(Default$default)
default.num[default.num == 1] <- 0
default.num[default.num == 2] <- 1
default.lm <- lm(default.num~ balance, data = Default)
plot(Default$balance, default.num)
abline(default.lm, lty = 2, col = "red")# Interesting the balance for my least squares regression line is a little off in the Balance axis (since the author have a line that went beyond the zero margin in the balance axis). Will need to look into what the problem is. 
			#p(X) = beta_0 + beta_1*X.
#If weuse this approach to predict default = Yes using balance, then we obtain the model shown in the preceding graphic representation. Here we see the problem with this approach: for balances close to zero we predict a negative probability of default; if we were to predict for very large balances, we would get values bigger than 1. These predictions are not sensible, since of course the true probability of default, regardless of credit card balance must fall between 0 and 1. This problem is not unique to the credit default data. Any time a straight line is fit to a binary response that is coded as 0 or 1, in principle we can always predict p(X) < 0 for some values of X and p(X) > 1 for others (unless the range of X is limited).

#to avoid this problem, we must model p(X) using a function that gives outputs between 0 and 1 for all values of X. Many functions meet this description. In logistic regression, we use the logistic function.
			#p(X) = e^beta_0 + beta_1*X / 1 + e^beta_0 + beta_1 *X
			
#To fit the model, we use a method called maximum likelihood, which we discuss in the next section. Notice that for low balances we now predict the probability of default as close to, but never below, zero. Likewise, for high balances we predict a default probability close to, but never above, one. The logistic function will always produce an S-shaped curve of this form, and so regardless of the value of X, we will obtain a sensible prediction. We also see that the logistic model is better able to capture the range of probabilities than is the linear regression model preceding plot. The average fitted probability in both cases is 0.0333 (averaged over the training data), which is the same as the overall proportion of defaulters in the dataset.

#After a bit of manipulation of the preceding equation:
		#p(X) / 1 - p(X) = e^beta_0 + beta_1*X 
		
#The quantity p(X)/[1 - p(X)] is called the odds, and can take on any value between 0 and infinity. Values of the odds close to 0 and infinity indicate very low and very high probabilities of default, respectively. For example, on average 1 in 5 people with an odds of 1/4 will default, since p(X) = 0.2 implies an odds of 0.2/1-0.2 = 1/4. Likewise on average nine out of every ten people with an odds of 9 will default, since p(x) = 0.9 implies an odds of 0.9/1 - 0.9 = 9. Odds are traditionally used instead of probabilities in horse-racing, since they relate more naturally to the correct betting strategy.

#By taking the logarithm of both sides with the preceding equation,arrive at:
			#log(p(X) / 1-p(X)) = beta_0 + beta_1*X.
#The left-hand side is called the log odds or logit. We see that the logistic regression model has a logit that is linear in X.

#In a logistic regression model, increasing X by one unit changes the log odds by beta_1, or equivalently it multiplies the odds by e^beta_1. However, because the relationship between p(X) and X in the logistic function is not a straight line, beta_1 does not correspond to the change in p(X) associated with a one-unit increase in X. The amount that p(X) changes due to a one-unit change in X will depend on the current value of X. But regardless of the value of X, if beta_1 is positive then increasing X will be associated with increasing p(X), and if beta_1 is negative then increasing X will be associated with decreasing p(X). The fact that there is not a straight line relationship between p(X) and X, and the fact that the rate of change in p(X) per unit change in X depends on the current value of X.

## 4.3.2 Estimating the regression coefficients:
#the coefficients beta_0 and beta_1 in the logistical function are unknown, and must be estimated based on the available training data. Although we could use (non-linear) least squares to fit the model, the more general method of maximum likelihood is preferred, since it has better statistical properties. The basic intuition behind using maximum likelihood to fit a logistic regression model is as follows: we seek estimates for beta_0 and beta_1 such that the predicted probability p_hat(X_i) of default for each individual using the logarithic function, corresponds as closely as possible to the individual's observed default status. In other words, we try to find beta_hat_0 and beta_hat_1 such that plugging these estimates into the model for p(X), given by the logarithmic function, yields a number close to one for all individuals who defaulted, and a number close to zero for all individuals who did not. This intuition can be formalized using the mathematical equation called the likelihood function.

#The estimates beta_hat_0 and beta_hat_1 are chosen to maximize this likelihood function. Maximum likelihood is a very general approach that is used to fit many of the non-linear models that we examine throughout this book. In the linear regression setting, the least squares approach is in fact a special case of maximum likelihood. 

#The table on page 134 shows the coefficient estimates and related information that results from fitting a logistic regression model on the Default data in order to predict the probability of default = yes using balance. We see that beta_hat_1 = 0.0055; this indicates that an increase in balance is associated with an increase in the probability of default. To be precise, a one-unit increase in balance is associated with an increase in the log odds of default by 0.0055 units.

#Many aspects of the logistic regression output shown in table 4.1 (page 134) are similar to the linear regression output of chapter 3. For example, we can measure the accuracy of the coefficient estimates by computing their standard errors. The z-statistic plays the same role as the t-statistic in the linear regression output. For instance, the z-statistic associated with beta_1 is equal to beta_hat_1 / SE(beta_hat_1), and so a large (absolute) value of the z-statistic indicates evidence against the null hypothesis H_0: beta_1 = 0. This null hypothesis implies that p(X) = e^beta_0 / 1+e^beta_0 in other words, that the probability of default does not depend on balance. Since the p-values associated with balance in table 4.1 is tiny, we can reject the null hypothesis for the alternative hypothesis which says that there is indded an association between balance and probability of default. The estimated intercept is typically not of interest; its main purpose is to adjust the average fitted probabilities to the proportion of ones in the data. 

## 4.3.3 Making predictions:
#Once the coefficients have been estimated, it is a simple matter to compute the probability of default for any given credit card balance. 

#One can use qualitative predictors with the logistic regression model using the dummy variable approach. As an example the Default data set contains the qualitative variable student. to fit the model we simply create a dummy variable that takes on a value of 1 for students and 0 for non-students. The logistic regression model that results from predicting probability of default from student status can be seen in table 4.2. The coefficient associated with the dummy variable is positive, and the associated p-value is statistically significant. This indicates that students tend to have higher default probabilities than non-students:
		#Pr_hat(default = Yes|student = Yes) = e^--3.5041 + 0.4049 * 1/1 + e^-3.5041 + 0.4049 * 1 = 0.0431
		#Pr_hat(default = Yes| student = No) = e^-3.5041 + 0.4049 *0/ 1 + e^-3.5041 + 0.4049 * 0 = 0.0292.
		
## 4.3.4 Multiple Logistic Regression:
#We now consider the problem of predicting a binary response using multiple predictors. By analogy with the extension from simple to multiple linear regression in chapter 3, we can generalize the logistic function as follows:
		#log(p(X) / 1 - p(X)) = beta_0 + beta_1*X_1 + ... + beta_p*X_p,
#where X = (X_1, ..., X_p) are p predictors. the following equation can be rewritten as:
		#p(X) = e^beta_0+beta_1*X_1+...+beta_p*X_p/1 + e^beta_0+beta_1*X_1+...+beta_p*X_p.
		
#This simple example illustrates the dangers and subtleties associated with performing regressions involving only a single predictor when other predictors may also be relevant. As in the linear regression setting, the results obtained using one predictor may be quite different from those obtained using multiple predictors, especially when there is correlation among the predictors. In general, the phenomenon is kwnown as confounding.

##4.3.5 Logistic Regression for > 2 Response Classes
#We sometimes wish to classify a response variable that has more than two classes. For example, in Section 4.2 we had three categories of medical condition in the emergency room: stroke, drug overdose, epileptic seizure. In this setting, we wish to model both Pr(Y = stroke|X) and Pr(Y = drug overdose|X), with the remaining Pr(Y = epileptic seizure|X) = 1 - Pr(Y = stroke|X) = Pr(Y = drug overdose|X). The two-calss logistic regression models discussed in the previous sections have multiple class extensions, but in practice they tend not to be used all that often. One of the reasons is that the method we discuss in the next section, discriminant analysis, is popular for multiple class classification.

## 4.4 Linear Discrimination Analysis:
#In this alternative approach, we model the distribution of the predictors X separately in each of the response classes (given Y), and then use Bayes' theorem to flip these around into estimates for Pr(Y = k|X = x). When these distributions are assumed to be normal, it turns out that the model is very similar in form to logistic regression.

#Why do we need another method, when we have logistic regression? There are several reasons:
		#When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable. Linear discriminant analysis does not suffer from this problem.
		#If n is small and the distribution of the predictors X is approximately normal in each of the classes, the linear discriminant model is again more stable than the logistic regression model.
		#As mentioned in Section 4.3.5, linear discriminant analysis is popular when we have more than two response classes. 
		
##4.4.1 Using Bayes' Theorem for classification:
#suppose that we wish to classify an observation into one of K classes, where K>= 2. In other words, the qualitative response variable Y can take on K possible distinct and unordered values. Let pi_k represent the overall or prior probability that a randomly chosen observation comes from the kth class; this is the probability that a given observation is associated with the kth category of the response variable Y. Let f_k(X) = Pr(X = x|Y = k) denote the density function of X for an observation that comes from the kth class. In other words, f_k(x) is relativelylarge if there is a high probability that an observation in the kth class has X ~ x, and f_k(x) is small if it is very unlikely that an observation in the kth class has X ~ x. Then Bayes' theorem states that 
				#Pr(Y = k|X = x) = pi_kf_k(x)/sum(length(x) + K)*pi_l*f_l(x).
				#I'm not really sure if this particular rendering is the proper interpretation of the Bayes' theorem. I really need to brush up on my math.
				
#In accordance with our earlier notation, we will use the abbreviation P_k(X) = Pr(Y = k|X). This suggests that instead of directly computing p_k(X) we can simply plug in estimates of pi_k and f_k(X) into the preceding Bayes' theorem. In general, estimating pi_k is easy if we have a random sample of Ys from the population: we simply compute the fraction of the training observations that belong to the kth class. However, estimating f_k(X) tends to be more challenging, unless we assume some simple forms for these densities. We refer to p_k(x) as the posterior probability that an observation X = x belongs to the kth class. That is, it is the probability that the observation belongs to the kth class, given the predictor value for that observation. 

#We know from chapter 2 that the Bayes classifier, which classifies an observation to the class for which p_k(X) is largest, has the lowest possible error rate out of all classifiers. (This is of course only true if the terms for the Bayes' theorem are all correctly specified). Therefore, if we can find a way to estimate f_k(X), then we can develop a classifier that approximates the Bayes classifer.

##4.4.2 Linear Discriminant Analysis for p = 1
#For now, assume that p = 1 --- that is, we have only one predictor. We would like to obtain an estimate for f_k(x) that we can plug into the Bayes' Theorem in order to estimate p_k(x). We will then classify an observation to the class for which p_k(x) is greatest. In order to estimate f_k(x), we will first make some assumptions about its form.
#suppose we assume that f_k(x) is normal or Gaussian. In the one dimensional setting, the normal density takes the form 
		#f_k(x) = 1/sqrt(2*pi*sigma_k) exp(-1/2*sigma^2_k&(x = mu_k)^2),
#where mu_k and sigma_k^2 are the mean and variance parameters for the kth class. For now, let us further assume that sigma^2_1 = ... = sigma^2_k: that is, there is a shared variance term across all K classes, which for simplicity we can denote by sigma^2. 
		#to see the following equation look at page 139.
		
#Note that in (4.12), pi_k denotes the prior probability that an observation belongs to the kth class, not to be confused with pi ~ 3.14, the mathematical constant.) The Bayes classifier involves assigning an observation X = x to the class for which (4.12) is largest. Taking the log of (4.12) and rearranging the terms, it is not hard to show that this is equivalent to assigning the observation to the class for which
		#To see equation 4.13 look at page 140. My mathematical skills really need to be sharped for me to even comprehend some of these symbols.  
#is largest. For instance, if K = 2 and pi_1 = pi_2, then the Bayes classifier assigns an observation to class 1 if 2x(mu_1 - mu_2) > mu^2_1 - mu^2_2, and to class 2 otherwise. In this case, the bayes decision boundary corresponds to the point where
		#x = mu^2_1 - mu^2_2 / 2(mu_1 - mu_2) = mu_1 + mu_2 / 2.

#An example is shown in the left-hand panel of Figure 4.4. The two normal density functions that are displayed, f_1(x) and f_2(x), represent two distinct classes. The mean and variance parameters for the two density functions are mu_1 = -1.25, mu_2 = 1.25, and sigma_1^2 = sigma_2^2 = 1. The two densities overlap, and so given that X = x, there is some uncertainty about the class to which the observation belongs. If we assume that an observation is equally likely to come from either class -- that is, pi_1 = pi_2 = 0.5 --- then by inspection of (4.14), we see that the Bayes classifier assigns the observation to class 1 if x < 0 and class 2 otherwise. Note that in this case, we can compute the Bayes classifier because we know that X is drawn from a Gaussian distribution within each class, and we know all the parameters involved. In a real-life situation, we are not able to calculate the Bayes classifier. 

#In practice, even if we are quite certain of our assumption that X is drawn from a Gaussian distribution within each class, we still have to estimate the parameters mu_1, ..., mu_k, pi_1 ...., and sigma_2. The linear discriminant analysis (LDA) method approximates the Bayes classifier by plugging estimates for pi_k, mu_k, and sigma^2 into (4.13). In particular, the following estimates are used:
		#To see the equations for the mu_hat_k and sigma_hat^2 approximations look at page 141.
#Where n is the total number of training observations, and n_k is the number of training observations in the kth class. The estimate for mu_k is simply the average of all the training observations from the kth class, while sigma_hat^2 can be seen as a weighted average of the sample variances for each of the K classes. Sometimes we have knowledge of the class membership probabilities pi_1, ..., pi_k, which can be used directly. In the absenece of any additional information, LDA estimates pi_k using the proportion of the training observations that belong to the kth class. In other words,
			#pi_hat_k = n_k/n
			
#The LDa classifier plugs the estimates given in (4.15) and (4.16) into (4.13), and assigns an observation X = x to the class for which:
			#to see the equation look at page 141.
#is largest. The word linear in the classifier's name stems from the fact that the discriminant functions are linear functions of x (as opposed to a more complex function of x). 

#to reiterate, the LDA classifier results from assuming that the observations within each class come from a normal distribution with a class-specific mean vector and a common variance sigma^2, and plugging estimates for these parameters into a Bayes classifier. In section 4.4.4, we will consider a less stringent set of assumptions, by allowing the observations in the kth class to have a class specific variance, sigma_k^2.

## 4.4.3 Linear Discriminant Analysis for p > 1 
#We now extend the LDA classifier to the case of multiple predictors. To do this, we will assume that X = (X_1, X_2, ..., X_p) is drawn from multivariate Gaussian (or multivariate normal) distribution, with a class-specific mean vector and a common covariance matrix. 
# The multivariate Gaussian distribution assumes that each individual predictor follows a one-dimensional normal distribution, as in (4.11), with some correlation between each pair of predictors. Two examples of multivariate Gaussian distrubtions with p = 2 are shown in figure 4.5 (located on page 141). The height of the surface at any particular point represents the probability that both X_1 and X_2 fall in a small region around that point. In either panel, if the surface is cut along the X_1 axis or along the X_2 axis, the resulting cross section will have a shape of a one-dimensional normal distribution. The left-hand panel of figure 4.5 illustrates an example in which Var(X_1) = Var(X_2) and Cor(X_1, X_2) = 0, this surface has a characteristic bell shape. However, the bell shape will be distorted if the predictors are correlated or have unequal variances, as is illustrated in the right hand panel. In this situation, the base of the bell will have an elliptical, rather than circular, shape. 
		
		# To see that multivariate Gaussian density function look at page 143
		# To see the vector/matrix version of equation 4.13 look at page 143
		
#Note that there are three lines representing the Bayes decision boundaries because there are three pairs of classes among the three classes. That is, one Bayes decision boundary separates class 1 from class 2, one separates class 1 form class 3, and one separates class 2 from class 3. These three Bayes decision boundaries divide the predictor space into three regions. The Bayes classifier will classify an observation according to the region in which it is located. 

#We can perform LDA on the Default data in order to predict whether or not an individual will default on the basis of credit card balance and student status. The LDA model fit to the 10000 training data results in a training error rate of 2.75 percent. This sounds like a low error rate, but two caveats must be noted.
	#First of all, training error rates will usually be lower than test error rates, which are the real quantity of interest. In orther words, we might expect this classifier to perform worse if we use it to predict whether or not a new set of individuals will default. The reason is that we specifically adjust the parameters of our model to do well on the training data. The higher the ratio of parameters p to number of samples n, the more we expect this overfitting to play a role. For these data we don't expect this to be a problem, since p = 4 and n = 10000.
	#Second, since only 3.33 percent of the individuals in the training sample defaulted, a simple but useless classifier that always predicts that each individaul will not default, regardless of his or her credit card balance and student status, will result in an error rate of 3.33 percent. In other words, the trivial null classifier will acheive an error rate that is only a bit higher than the LDA training set error rate.
	
#In practice, a binary classifier such as this one can make two types of errors: it can incorrectly assign an individual who defaults to the no default category, or it can incorrectly assign an individual who does not default to the default category. It is often of interest to determine which of these two types of errors are being made. A confusion matrix, shown for the Default data in table 4.4, is a convenient way to display this information. The table reveals that LDA predicted that a total of 104 people would default. Of these people, 81 actually defaulted and 23 did not. Hence only 23 out of 9667 of the individuals who did not default were incorrectly labeled. However, of the 33 individuals who defaulted, 252 (75.7 percent) were missed by LDA. 

#Why does LDA do such a poor job of classifyin the customers who default? In other words, why does it have such a low sensitivity? As we have seen, LDA is trying to approximate the Bayes classifier, which has the lowest total error rateout of all classifiers (if the Gaussian model is correct). That is, the Bayes classifier will yield the smallest possible total number of misclassified observations, irrespective of which class the errors come from. That is, some misclassifications will result from incorrectly assigning a customer who does not default ot the default class, and others will result from incorrectly assigning a customer who defaults to the non-default class. We will now see that it is possible to modify LDA in order to develop a classifier that better meets the credit card company's needs (through creating a system that flips the error that flagged excessive amounts of people as non-default risks into flagging excessive amounts of people into default risks, as this scenario is less problematic for their business model). 

#The Bayes classifier works by assigning an observations to the class for which the posterior probability p)k(X) is greatest. In the two-class case, this amounts to assigning an observation to the default class if 
			#Pr(default = Yes|X = x) > 0.5.

#Thus, the bayes classifier, and by extension LDA, uses a threshold of 50 percent for the posterior probability of default in order to assign an observation to the default class. However, if we are concerned about incorrectly predicting the default status for individuals who default, then we can consider lowering this threshold. For instance, we might label any customer with a posterior probability of default above 20 percent to the default class. In other words, instead of assigning an observation to the default class if (4.21) holds, we could instead assign an observation to this class if
		#P(default = Yes|X = x) > 0.2.

#Now LDA predicts that 430 individuals will default. Of the 333 individuals who default, LDA correctly predicts all but 138, or 41.4 percent. This is a vast improvement over the error rate of 75.7 percent that resulted from using the threshold of 50 percent. However, this improvement comes at a cost: now 235 individuals who do not default are incorrectly classified. As a result, the overall error rate has increased slightly to 3.73 percent. But a credit card company may consider this slight increase in the total error rate to be a small price to pay for more accurate identification of individuals who do indeed default. 

#Various error rates are shown as a function of the threshold value. Using a threshold of 0.5, as in (4.21), minimizes the overall error rate, shown as a black solid line. This is to be expected, since the Bayes classifier uses a threshold of 0.5 and is known to have the lowest overall error rate. But when a threshold of 0.5 is used; the error rate among the individuals who default is quite high. As the threshold is reduced, the error rate among individuals who default decreases steadily, but the error rate among the individuals who do not default increases. How can we decide which threshold value is best? Such a decision must be based on domain knowledge, such as detailed information about the costs associated with default. 

#The ROC curve is a popular graphic for simultaneously displaying the two types of errors for all possible thresholds. The name "ROC" is historic, and comes from cummunications theory. It is an acronym for receiver operating characteristics. The overall performance of a classifier, summarized over all possible thresholds, is given by the area under (ROC) the curve (AUC). An ideal ROC curve will hug the top left corner, so the larger the AuC the better the classifier. For this data the AUC is 0.95, which is close to the maximum of one so would be considered very good. We expect a classifier that performs no better than chance to have an AUC of 0.5 (when evaluated on an independent test set not used in model training). ROC curves are useful for comparing different classifiers, since they take into account all possible thresholds. It turns out that the ROC curve for the logistic regression model of section 4.3.4 fit to these data is virtually indistinguishable from this one for the LDA model, so we do not display it here. 

#As we have seen above, varying the classifier threshold changes its true positive and false positive rate. These are also called the sensitivity and one minus the specificity of our classifier. 

##4.4.4 Quadratic Discriminant analysis
#Quadratic discriminant analysis (QDA) provides an alternative approach. Like LDA, the QDA classifier results from assuming that the observations from each class are drawn from a Gaussian distribution, and plugging estimates for the parameters into Bayes' theorem in order to perform prediction. However, unlike LDA, QDa assumes that each class has its own covariance matrix. 
#Unlike (4.19), the quantity x appears as a quadratic function in (4.23). This is where QDA gets its name. 

#Why does it matter whether or not we assume that the K classes share a common covariance matrix? In other words, why would one prefer LDA to QDA, or vice-versa? The answer lies in the bias-variance trade-off. When there are p predictors, then estimating a covariance matrix requires estimating p(p + 1)/2 parameters. QDA estimates a separate covariance matrix for each class, for a total of K_p(p+1)/2 parameters. With 50 predictors this is some multiple of 1225, which is a lot of parameters. By instead assuming that the K classes share a common covariance matrix, the LDA model becomes linear in x, which means there are K_p linear coefficients to estimate. Consequently, LDA is a much less flexible classifier than QDA, and so has substantially lower variance. This can potentially lead to imporved prediction performance. But there is a trade-off: if LDA's assumption that the K classes share a common covariance matrix is badly off, then LDA can suffer from high bias. Roughly speaking, LDA tends to be a better bet than QDA if there are relatively few training observations and so reducing variance is crucial. In contrast, QDA is recommended if the training set is very large, so that the variance of the classifier is not a major concern, or if the assumption of a common covariance matrix for the K classes is clearly untenable.

##4.5 A comparison of Classification methods
#KNN takes a completely different approach from the classifiers seen in this chapter. In order to make a prediction for an observation X = x, and K training observations that are closest to x are identified. Then X is assigned to the class to which the plurality of these observations belong. Hence KNN is a completely non-parametrix approach: no assumptions are made about the shape of the decision boundary. Therefore, we can expect this approach to dominate LDA and logistic regression when the decision boundary is highly non-linear. On the other hand, KNN does not tell us which predictors are important; we don't get a table of coefficients as in table 4.3.

#Finally, QDA serves as a compromise between the non-parametrix KNN method and the linear LDA and logistic regression approaches. Since QDA assumes a quadratic decision boundary, it can accurately model a wider range of problems than can the linear methods. Though not as flexible as KNN, QDA can perform better in the presence of limited number of training observations because it does make some assumptions about the form of the decision boundary.  
		
## 4.6 Lab: logistic Regression, LDA, QDA, and KNN
## 4.6.1 The stock market data:
#We will being by examining some numberical and graphical summaries of the Smarket data, which is part of the ISLR package library. This data set consists of percentage returns for the Standard and Poor 500 stock index over 1250 days, from the beginning of 2001 until the end of 2005. For each date, we have recorded the percentage returns for each of the five previous trading days, Lag1 through lag5. We have also recorded Volume (the number of shares traded on the previous day, in billions), Today (the percentage return on the date in question) and Direction (whether the market was up or Down on this date). 
library(ISLR)
names(Smarket)
dim(Smarket)
summary(Smarket)
str(Smarket)
pairs(Smarket)# Weirdly enough I can't see any descernable trends within the graphic representation. It will be very interesting to see what the author comes up with through the Logistical statistical learning method. Remember that simple logistical regression method can be used in this case due to there being two levels of the response variable (the market can either go up or down).

#The cor() function produces a matrix that contains all of the pairwise correlations among the predictors in a data set. The first command below givens an error message because the Direction variable is qualitative. 
cor(Smarket)
cor(Smarket[,-9])

#As one would exprect, the correlations between the lag variables and today's returns are close to zero. In other words, there appears to be little correlation between today's returns and previous day's returns. The only substantial correlation is between Year and Volume. By plotting the data we see that Volume is increasing over time. In other words, the average number of shares traded daily increases from 2001 to 2005.
#the author is right about this comment since the correlation value is recorded as 0.539 between Year and Volume. This isn't really that large of a correlation but even with that said all the other variables are almost zero.
attach(Smarket)
plot(y = Volume, x = Year)
plot(Volume)# Interesting through only graphing a variable you can see the variable's increase over time (by default the specified variable will be printed on the y axis and the index count will be printed on the x axis).

##4.6.2 Logistic Regression 
#Next, we will fit a logistic regression model in order to predict Direction using Lag1 through Lag5 and Volume. The glm() function fits generalized linear models, a class of models that includes logistic regression. The syntax of the glm() funciton is similar to that of lm(), except that we must pass the argument family = binomial in order to tell R to run a logistic regression rather than some other type of generalized linear model. 
glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Smarket, family = binomial)
summary(glm.fit)

#The smallest p-value here is associated with Lag1. The negative coefficient for this predictor suggests that if the market had a positive return yesterday, then it is less likely to go up today. However, at a value of 0.15, the p-value is still relatively large, and so there is no clear evidence of a real association between Lag1 and Direction.

#We use the coef() function order to access just the coefficients for this fitted model. We can also use the summary() function to access particular aspects of the fitted model, such as the p-values for the coefficients.
coef(glm.fit)
summary(glm.fit)$coef

#The predict() function can be used to predict the probability that the market will got up, given values of the predictors. The type = "response" option tells R to output probabilities of the form p(Y = 1|X), as opposed to other information such as the logit. If no data set is supplied to the predict() function, then the probabilities are computed for the training data that was used to fit the logistic regression model. Here we have printed only the first ten probabilities. We know that these values correspond to the probability of the market going up, rather than down, because the contrasts() function indicates that R has created a dummy variable with a 1 for Up.
glm.probs <- predict(glm.fit, type = "response")
glm.probs[1:10]
length(glm.probs) # the prediction vector is recorded at length 1250. 
contrasts(Direction)

#In order to make a prediction as to whether the market will go up or down on a particular day, we must convert these predicted probabilities into class labels, Up or down. The following two commands create a vector of class predictions based on whether the predicted probability of a market increase is greater than or less than 0.5.
glm.pred <- rep("Down", 1250)
glm.pred[glm.probs > 0.5] <- "Up"

#the first command creates a vector of 1250 Down elements. The second line transforms to Up all of the elements for which the predicted probability of a market increase exceeds 0.5. Given these predictions, the table() function can be used to produce a confusing matrix in order to determine how many observations were correctly or incorrectly classified. 
table(glm.pred, Direction)
summary(Year)

#the diagonal elements of the confusion matrix indicate correct predictions while the off diagonals represent incorrect predictions. Hence our model correctly predicted that the market would go up on 507 days and that it would go down on 145 days, for a total of 507 + 145 = 652 correct predictions. The mean() function can be used to compute the fraction of the days for which the prediction was correct. In this case, logistic regression correctly predicted the movement of the market 52.2 percent of the time. 
#At first glance, it appears that the logistic regression model is working a little better than random guessing. However, this result is misleading because we trained and tested the model on the same set of 1250 observations. In other words, 100 - 52.2 = 47.8 percent is the training error rate. As we have seen previously, the training error rate is often overly optimistic -- it tends to underestimate the test error rate. In order to better assess the accuracy of the logistic regression model in this setting, we can fit the model using part of the data, and then examine how well it predicts the held out data. this will yield a more realistic error rate, in the sense that in practice we will be interested in our model's performance not on the data that we used to fit the model, but rather on days in the future for which the market's movements are unknown. 

#To implement this strategy, we will first create a vector corresponding to the observations from 2001 through 2004. We will then use this vector ro create a held out data set of observations from 2005. 
train <- (Year < 2005)
head(train, n = 20)# interesting this is logical vector of the years that are below 2005.
Smarket.2005 <- Smarket[!train,]
length(Smarket.2005)
Direction.2005 <- Direction[!train]
length(Direction.2005)# The number of rows mirrors the Smarket.2005 testing vector meaning that there won't be any discrepencies in the length of both of these objects. 

#The object train is a vector of 1250 elements corresponding to the observations in our data set. The elements of the vector that correspond to observations that occured before 2005 are set to TRUE, whereas those that correspond to observations in 2005 are set to FALSE. 

#We can fit a logistic regression model using only the subset of the observations that correspond to dates before 2005, using the subset argument. We then obtain predicted probabilities of the stock market going up for each of the days in our test set --- that is, for the days in 2005. 
glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5, Volume, data = Smarket, family = binomial, subset = train)
glm.probs <- predict(glm.fit, Smarket.2005, type = "response")

#Notice that we have trained and tested our model on two completely separate datasets: training was performed using only the dates before 2005, and testing was performed using only the dates in 2005. Finally, we compute the predictions for 2005 and compare them to the actual movements of the market over that time period. 
glm.pred <- rep("Down", 252)
glm.pred[glm.probs > 0.5] <- "Up"
table(glm.pred, Direction.2005)# Interesting I got different numbers than the author. Will need to check is my computer is doing these operations correctly. I did see a warning message after running the slm.fit command. Will need to look into this. 
mean(glm.pred == Direction.2005) # error rate 0.579
mean(glm.pred!=Direction.2005)# error rate 0.421

#The != notation means not equal to, and so the last command computes the test set error rate. The results are rather disappointing: the test error rate is 42 percent which is worse than random guessing.
#We recall that the logistic regression model had very underwhelming p-values associated with all of the predictors, and that the smallest p-value, though not very small, corresponded to Lag1. Perhaps by removing the variables that appear not to be helpful in predicting Direction, we can obtain a more effective model. After all, using predictors that have no relationship with the response tends to cause a deterioration in the test error rate (since such predictors cause an increase in variance without a corresponding decrease in bias), and so removing such predictors may in turn yield an improvement. Below we have refit the logistic regression using just Lag1 and Lag2, which seemed to have the highest predictive power in the original regression model. 
glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Smarket, family = binomial, subset = train)
plot(glm.fit)
glm.probs <- predict(glm.fit, Smarket.2005, type = "response")
glm.pred <- rep("Down", 252)
glm.pred[glm.probs> 0.5] <- "Up"
table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005)# the error for the training data set is 0.56 
106/(106+76)# the error for the test dataset is 0.582

#Now the results appear to be more promising: 56 percent of the daily movements have been correctly predicted. The confusion matrix suggests that on days when logistic regression predicts that the market will decline, it is only correct 50 percent of the time. However, on days when it predicts an increase in the market, it has a 58 percent accuracy rate.

#Suppose that we want to predict the returns associated with particular values of Lag1 and Lag2. In particular, we want to predict Direction on the day when Lag1 and Lag2 equal 1.2 and 1.1, respectively, and on a day when they equal 1.5 and -0.8. We do this using the predict() function. 
predict(glm.fit, newdata = data.frame(Lag1 = c(1.2,1.5), Lag2 = c(1.1, -0.8)), type = "response")

##4.6.3 Linear Discriminant Analysis
#Now we will perform LDA on the Smarket data. In R, we fit a LDA model using the lda() function, which is part of the MASS library. Notice that the syntax for the lda() function is identical to that of lm(), and to that of glm() except for the absence of the family option. We fit the model using only the observations before 2005 
library(MASS)
lda.fit <- lda(Direction ~ Lag1 + Lag2, data = Smarket, subset = train)
lda.fit
plot(lda.fit)
#The LDA output indicates that pi_hat = 0.492 and pi_hat = 0.508; in other wrds, 49.2 percent of the training observations correspond to days during which the market went down. It also provides the group means; these are the average of each predictor within each class, and are used by LDA as estimates of mu_k. These suggest that there is a tendency for the previous 2 days' returns to be negative on days when the market increases, and a tendency for the previous days' returns to be positive on days when the market declines. The coefficients of linear discriminants output provides the linear combination of Lag1 and Lag2 that are used to form the LDA decision rule. In other words, these are the multipliers of the elements of X = x in (4.19). If -0.642*Lag1 - 0.514*Lag2 is large, then the LDA classifier will predict a market increase, and if it is small, then the LDA classifier will predict a market decline. the plot() function produces plots of the linear discriminants, obtained by computing -0.642*Lag1 - 0.514*Lag2 for each of the training observations. 

#the predict() function returns a list with three elements. The first element, class, contains LDA's predictions about the movement of the market. The second element, posterior, is a matrix whose kth column contains the posterior probability that the corresponding observation belongs to the kth class, computed from 4.10. Finally, x contains the linear discriminants described earlier.
lda.pred <- predict(lda.fit, Smarket.2005)
names(lda.pred)

#As we observed in Section 4.5, the LDA and logistic regression predictions are almost identical.
lda.class <- lda.pred$class
table(lda.class, Direction.2005)
mean(lda.class == Direction.2005)

#Applying a 50 percent threshold to the posterior probabilities allows us to recreate the predictions contained in lda.pred$class.
sum(lda.pred$posterior[,1] >= 0.5)
sum(lda.pred$posterior[,1] < 0.5)

#Notice that the posterior probability output by the model corresponds to the probability that the market will decrease:
lda.pred$posterior[1:20,1]
lda.class[1:20]

#If we wnated to use a posterior probability threshold other than 50 percent in order to make predictions, then we could easily do so. For instance, suppose that we wish to predict a market decrease only if we are very certain that the market will indeed descrease on that day --- say, if the posterior probability is at least 90 percent.
sum(lda.pred$posterior[,1]>0.9)
sum(lda.pred$posterior[,1]>0.52)
#No days in 2005 meet that threshold. In fact, the greatest posterior probability of decrease in all of 2005 was 52.02 percent.

##4.6.4 Quadratic Discriminant Analysis:
#We will now fit a QDA model to the Smarket data. QDA is implemented in R using the qda() function, which is also part of the MASS library. The syntax is identical to that of the lda().
qda.fit <- qda(Direction ~ Lag1 + Lag2, data = Smarket, subset= train)
qda.fit
#The output contains the group means. But it does not contain the coefficients of the linear discriminants, because the QDA classifier involves a quadratic, rather than a linear, function of the predictors. The predict() function works in exactly the same fashion as for LDA. 
qda.class <- predict(qda.fit, Smarket.2005)$class
table(qda.class, Direction.2005)
mean(qda.class == Direction.2005)
#Interestingly, the QDA predictions are accurate almost 60 percent of the time, even though the 2005 data was not used to fit the model. This level of accuracy is quite impressive for stock market data, which is known to be quite hard to model accurately. this suggests that the quadratic form assumed by QDA may capture the true relationship more accurately than the linear forms assumed by LDA and logistic regression. 

##4.6.5 K-Nearest Neightbors 
# We will now perform KNN using the knn() function, which is part of the class library. This function works rather differently from the other model-fitting functions that we have encountered thus far. Rather than a two-step approach in which we first fit the model and then we use the model to make predictions, knn() forms predictions using a single command. The function requires four inputs.
	#1. A matrix containing the predictors associated with the training data, labeled train.X below.
	#2. A matrix containing the predictors associated with the data for which we wish to make predictions, labeled test.X below.
	#3. A vector containing the class labels for the training observations, labeled train.Direction below.
	#4. A value for K, the number of nearest neigh-bors to be used by the classifier.
	
#We use the cbind() function, short for column bind, to bind the Lag2 and Lag2 variables together into two matrices, one for the training set and the other for the test set.
library(class)
train.X <- cbind(Lag1, Lag2)[train,]
test.X <- cbind(Lag1, Lag2)[!train,]
train.Direction <- Direction[train]
#Now the knn() function can be used to predict the market's movement for the dates in 2005. We set a random seed before we apply knn() because if several observations are tied as nearest neighbors, then R will randomly break the tie. Therefore, a seed must be set in order to ensure reproducibility of results. 

set.seed(1)
knn.pred <- knn(train.X, test.X, train.Direction, k = 1)
table(knn.pred, Direction.2005)
(83 + 43)/252# 50 percent of the observations are correctly predicted. 

#Below, we repeat the analysis using K = 3.
knn.pred <- knn(train.X, test.X, train.Direction, k = 3)
table(knn.pred, Direction.2005)

#The results have improved slightly. But increasing K further turns out to provide no further improvements. It appears that for this data, QDA provides the best results of the methods that we have examined so far. 

##4.6.6 An application to Caravan Insurance Data:
#Finally, we will apply the KNN approach to the Caravan data set, which is part of the ISLR library. This data set includes 85 predictors that measure demographic characteristics for 5822 individuals. The response variable is Purchase, which indicates whether or not a given individual purchases a caravan insurance policy. In this data set, only 6 percent of people purchase caravan insurance. 
dim(Caravan)
dimnames(Caravan)
detach(Smarket)
attach(Caravan)
summary(Purchase)
348/5474# Only 0.0636 purchased caravan insurance. This equates into only 6.36 percent.

#salary will drive the KNN classification results, and age will have almost no effect. This is contrary to our intuition that a salary difference of 1000 is quite small compared to an age difference of 50 years. Furthermore, the importance of scale to the KNN classifier leads to another issue: if we measure salary in Japanese yen, or if we measured the age in minutes, then we'd get quite different classification results for what we get if these two variables are measured in dollars and years. 
#A good way to handle this problem is to standardize the data so that all variables are given a mean of zero and a standard deviation of one. Then all variables will be on a comparable scale. The scale() function does just this. In standardizing the data, we exclude column 86, because that is the qualitative Purchase variable.
#In other words the author will convert all of the quantitative variables into z-scores. 

standardized.X <- scale(Caravan[,-86])
var(Caravan[,1])
var(standardized.X[,1])
var(standardized.X[,2])

#We now split the observations into a test set, containing the first 1,000 observations, and a training set, containing the remaining observations. We fit a KNN model on the training data using K = 1, and evaluate its performance on the test data.

test <- 1:1000
train.X <- standardized.X[-test,]
test.X <- standardized.X[test,]
train.Y <- Purchase[-test]
test.Y <- Purchase[test]
set.seed(1)
knn.pred <- knn(train.X, test.X, train.Y, k = 1)
mean(test.Y!=knn.pred)
mean(test.Y!="No")
table(knn.pred, test.Y)
9/(68+9)
#Through the use of k = 1 the KNN algorithm predicts that 11.7 percent of the people will actually purchase the insurance.
#Using k = 3, the success rate increases to 19 percent and with K = 5 the rate is 26.7 percent. This is over fourtimes the rate that results from random guessing. It appears that KNN is finding some real patterns in a difficult data set.

knn.pred <- knn(train.X, test.X, train.Y, k = 3)
table(knn.pred, test.Y)
5/26# 19.2 percent 

knn.pred <- knn(train.X, test.X, train.Y, k = 5)
table(knn.pred, test.Y)
4/15# 26.7 percent 

#Using a logistical linear model:
glm.fit <- glm(Purchase ~., data = Caravan, family = binomial, subset = -test)
glm.probs <- predict(glm.fit, Caravan[test,], type = "response")
glm.pred <- rep("No", 1000)
glm.pred[glm.probs > .5] <- "Yes"
table(glm.pred, test.Y)# zero insurance purchases were predicted through this model. The author fixed this problem through using a different probability cut off.

glm.pred <- rep("No", 1000)
glm.pred[glm.probs > 0.25] <- "Yes"
table(glm.pred, test.Y)
11/(22+11)# 33.3 percent 

## 4.7 Exercises
##conceptual:
#1.) Saddly enough my mathematical skills are so limited that I can't even move a simple logarithmic symbol to the other side of the equation. Will need to brush up on my algebra immediately. 

#wait a second I think I understand the question now. The mathematical thought surroundering equations 4.2 and 4.3 are easier to understand than those apparent between equations 4.1 and 4.2. Will need to look into this analysis. Yeah I still can't carry out the simplistic operation for these equations.
	# P(X) = e^beta_0 + beta_1 * X / 1 + e^beta_0 + beta_1 * X
	# P(X) / 1 - p(X) = e^beta_0 + beta_1 * X 
	
#2.) Sadly I don't really know how I should begin to answer this question. will really need to reread this chapter (most linear and quadratic discriminant analysis) and read a couple more books on this topic.

#3.) Again I'm having the same problems as the last three questions. The good news is that Adadoughi has all the solutions for these problems but they are all written (I believe) in mathematica syntax. 

#4.) Experiment on the curse of dimensionality:Asadoughi's solutions 
#a.) On average, 10 percent. For simplicity, ignoring cases when X < 0.05 and X > 0.95

#b.) On average, 1 percent 

#c.) on average, 0.10^100 * 100 = 10^-98 percent 

#d.) As p increases linearly, observations that are geometrically near decreases exponentially.

#e.) Will need to look into how to code hypercubes. This section is a bit too advanced for me. 

#5.) The differences between LDA and QDA.
#(a) Since the decision boundary is linear the LDA will do the best in the test set. The reason for this decision is simple. Because the Linear Discriminant Analysis method thrives on linear decision boundaries and the Quadratic discriminant analysis counter thrives in slightly non-linear decision boundaries. In addition, linear decision boundaries illustrates the assumption of equal covariance between the p values. Thus again pointing towards the LDA method.

#I believe that the LDA method will perform the best on both the training set and the test set.  

#(b) If the decision boundary is non-linear it is assumed that the QDA method will perform the best on the test and training datasets because non-linear decision boundaries illustrate that the p variables all have their own covariate matrix. 

#(c) I believe that with increased sample size the QDA will become better due to the fact that both the QDA and LDA methods are still parametric methods and that with increased samples their fit usually becomes more accurate to the population trend. Even with that said though, for the QDA method to increase in accuracy you need to assume that p stays the same (or rather does not increase) and that the distribution is normal. 

#Problems that I can see with this rationale though are that much like simple linear regression models using quadratic and logarithmic transformations are susceptable to model bias and this can inhibit the model's viability with increased sample size. 

#Anadoughi's solution:
#(a) If the Bayes decision boundary is linear, we expect QDA to perform better on the training set because its higher flexibility will yield a closer fit. On the test set, we expect LDA to perform better than QDA because QDA could overfit the linearity of the Bayes decision boundary.

#(b) If the Bayes decision boundary is non-linear, we expect QDA to perform better both on the training and test sets.

#(c) We expect the test prediction accuracy of QDA relative to LDA to improve, in general, as the sample size increases because a more flexible method will yield a better fit as more smaples can be fit and variance is offset by the larger sample sizes.

#(d) False. With fewer smaple points, the variance from using a more flexible method, such as QDA, would lead to overfit yielding a higher test rate than LDA.
#So in other words, LDA interestingly works better with fewer p values and n values than its QDA counterpart. Which makes sense when one thinks about the problem of superimposing a simple regression model that has a small amount of n values with a quadratic transformation (such a combination can only give rise to model over fitting or increased bias). 

#6.) Suppose we collect data for a group of students in a statistics class with variables X_1 = hours studied, X_2 = undergrad GPA, and Y = receive an A. We fit a logistic regression and produce estimated coefficient, beta_hat_0 = -6, beta_hat_1 = 0.05, beta_hat_2 = 1 
		#P(X| Y = A) = -6 + hours studied(0.05) + GPA(1)
		#P(X| Y = no A) = -6 + hours studied(0.05) + GPA(1)
#This is most likely not a good mathematical argument for the logistical problem but it does give me a very good idea what my next move should be. 

#the logistical regression equation is 
		#log(p(X) / 1 - p(X)) = beta_0 + beta_1*X_1 + beta_2*X_2

#(a) 
pX <- exp(-6 + 0.05 * 40 + 1 * 3.5) / (1 + exp(-6 + 0.05 * 40 + 1 * 3.5))
pX# cool the solution is 38 percent. The solution was retrieved from yahwes. 

#(b) 
x1 <- seq(40, 80, by = 1)
pX <- exp(-6 + 0.05 * x1 + 1 * 3.5) / (1 + exp(-6 + 0.05 * 40 + 1 * 3.5))
cbind(x1, pX)# According to this line of code about 46 hours. But if you change the x1 sequence to print values in the 0.01 scale you get:
x1 <- seq(44,46, by = 0.1)
pX <- exp(-6 + 0.05 * x1 + 1 * 3.5) / (1 + exp(-6 + 0.05 * 40 + 1 * 3.5))
cbind(x1, pX) # 45.7 hours for a fifty percent chance of obtaining an A for the course. 

#Yahwes solution:
(log(0.5/(1- 0.5)) + 6 - 3.5*1) / 0.05# No way I got the wrong answer for this one. I guess my method was extremely flawed. Again I really need to brush up on my mathematics. 

#7.) 
(0.8*exp(-1 / (2*36)*(4-10)^2))/(0.8*exp(-1/(2*36)*(4-10)^2)+(1-0.8)*exp(-1/(2*36)*(4-0)^2))
#the probability is 75.2 percent. solution from Yahwes.

#8.) Solution from Yahwes:
#It seems that Yahwes could find the right amount of information to answer this question, but interesting anadoughi found an answer. The only problem is that it is in mathematica. 

##Applied
#10. The weekly data set will be used for these questions. This particular dataset is very much similar to the Smarket dataset used in the exercises and the text.
library(ISLR)
colnames(Weekly)
str(Weekly)#The only difference I can see is that the year variable displays 1990 through 
summary(Weekly$Year)# to interestingly 2010. I wonder if Smarket has the same number of years. 
summary(Smarket$Year)#the range is 2001 to 2005 for the Smarket dataset. 
dim(Weekly)

#(a) 
summary(Weekly)
pairs(Weekly)# there seems to be a largely positive trend between Volume and Year. Will need to check this out. 
summary(glm(Direction ~ ., family = binomial, data = Weekly))
summary(glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume + Today, family = binomial, data = Weekly))# the p-value for volume is still very high. Will need to see what I'm doing wrong in this line of code. 
summary(glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, family = binomial, data = Weekly))# the p_values have all been corrected after taking out Today, but still the p-value for Volume is still relatively high. Will need to see what the pairs() graphic says. 

#(b)
plot(Weekly.fit)
Weekly.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, family = binomial, data = Weekly)
summary(Weekly.fit)#the values with the highest statistical significance are Lag1, Lag2, and Lag4. The rest have p-values over 0.50. 
#The most statistically significant value is Lag2 which has a p-value of 0.0296 (which is well below the 0.05 significance level cut off).

#(c) 
Weekly.probs <- predict(Weekly.fit, type = "response")
Weekly.probs[1:10]
Weekly.pred <- rep("Down", nrow(Weekly))
Weekly.pred[Weekly.probs > 0.5] <- "Up"
table(Weekly.pred, Weekly$Direction)
mean(Weekly.pred == Weekly$Direction)# this means that the prediction code predicted 56.1 percent of the market movement correctly. 
100 - 56.1 # As for the training error rate is 43.9 percent. Remember that this is an optimistic prediction. 
# looking at the confusion matrix it seems that the model mispredicts that the market will go down more often then go up. This might be advantages for a more conservative investor, but for someone who likes to enjoy returns during economic rallies this model will be impractical. 

#(d) 
glm.fit <- glm(Direction ~ Lag2, family = binomial, data = Weekly)
summary(glm.fit)
glm.train <- Weekly[Weekly$Year <= 2008,] 
glm.test <- Weekly[Weekly$Year > 2008,]
glm.test.dir <- Weekly$Direction[Weekly$Year > 2008]
glm.train.dir <- Weekly$Direction[Weekly$Year <= 2008]
glm.fit <- glm(Direction ~ Lag2, family = binomial, data = glm.train, subset = train)
glm.probs <- predict(glm.fit, glm.test, type = "response")
glm.pred <- rep("Down", nrow(glm.test))
glm.pred[glm.probs > 0.5] <- "Up"
table(glm.pred, glm.test.dir)
mean(glm.pred == glm.test.dir)
# the model predicted the stock market movement correctly about 62.5 percent of the time. Which equates into a:
100 - 62.5 #a 37.5 percent error rate for the test variable predictions. This is a very good value for this kind of dataset.

#(e) Using the LDA:
library(MASS)
lda.fit <- lda(Direction ~ Lag2, family = binomial, data = glm.train)
lda.pred <- predict(lda.fit, glm.test)
lda.class <- lda.pred$class
table(lda.class, glm.test.dir)
#Interesting the linear discriminant analysis method resulted in the same correct predictions as the logistical regression model. 
# 62.5 percent correct rate with an error rate of 37.5 percent 

#(f) Using the QDA method:
qda.fit <- qda(Direction ~ Lag2, data = glm.train)
qda.fit
qda.class <- predict(qda.fit, glm.test)$class
table(qda.class, glm.test.dir)
mean(qda.class == glm.test.dir)
#the accuracy for the QDA method is 58.7 percent. Will need to see why the model didn't predict Down movements with the test direction data. I hope that is doesn't nullify another statistical check that one needs to carry out.

#(g) Using the KNN method:
set.seed(1)
train.X <- as.matrix(Weekly$Lag2[Weekly$Year <= 2008]) 
test.X <- as.matrix(Weekly$Lag2[Weekly$Year > 2008])
knn.test.dir <- Weekly$Direction[Weekly$Year > 2008]
knn.train.dir <- Weekly$Direction[Weekly$Year <= 2008]
knn.pred <- knn(train.X, test.X, knn.train.dir, k = 1)
table(knn.pred, knn.test.dir)
(21 + 31) / (51+53)# The accuracy rate is 50 percent with the KNN method. 

#(h) The methods with the best accuracy values was the Linear Discriminant Analysis and Logistical Regression methods with an estimated 62.5 percent prediction accuracy rate. 

#11. For this question the author is using the Auto dataset. The variable that the author wants me to predict is high or low fuel mileage. Most likely I will have to convert the mpg variable into a categorical variable by some underlying bench mark.
#(a) 
median(Auto$mpg)# It seems that the median value is 22.75 miles per gallon. 
quantile(Auto$mpg)
range(Auto$mpg)
library(tidyverse)
Auto.new <- Auto %>% 
	mutate(mpg01 = Auto$mpg > median(Auto$mpg))
Auto.new$mpg[Auto.new$mpg01==TRUE]
Auto.new$mpg[Auto.new$mpg01==FALSE]#It seems that this line of code has the right designations for each of the vehicles in the data set. I'm glad that this line of code worked perfectly. It's sad that I'm forgetting the basics of R programming will need to brush up on ggplot2 and tidyverse operations in order to become more fluent in this language. 
Auto$mpg01 <- as.factor(Auto)

#(b) 
pairs(Auto.new)# Since the mpg01 variable is only two levels in variable k, it's very hard to see trends through the normal pairs data visualization technique. Will need to think of another method.
#Weird little phonomenon that I picked up on in the pairs data visualization graphic above is that the values for the mpg01 variable seem to be split evenly between the four different drive train classes. In addition, fuel efficient cars usually weigh less and have less horsepower. Will need to look into all of these trends with a couple of boxplots.
par(mfrow = c(1,4))
Auto.new$mpg01 <- as.factor(Auto.new$mpg01)
ggplot(Auto.new, aes(x = Auto.new$mpg01, y = Auto.new$weight)) + geom_boxplot()
ggplot(Auto.new, aes(x = Auto.new$mpg01, y = Auto.new$horsepower)) + geom_boxplot()
ggplot(Auto.new, aes(x = mpg01, y = displacement)) + geom_boxplot()# Interestingly the displacement variable has a high amount of variability within the False category of the mpg01 variable while the true category remains within a specific narrow displacement threshold. This must have something to do with turbo charges four cylinder cars as well as rotary engine drive trains in the Auto.new dataset. 
quantile(Auto.new$displacement[Auto.new$mpg01 == TRUE])
quantile(Auto.new$displacement[Auto.new$mpg01 == FALSE])
#Interesting it seems that the mpg01 False category goes up from a minimum displacement of 70 to a maximum of 455 while the mpg01 True category from a 68 to a maximum of 350. This seems really interesting. Most likely displacement will need to be left out of the model. 
Auto.new %>%
	group_by(cylinders) %>%
	count() 
#I remember from the couple of exercises that I used this same dataset in the 3 cylinder category is actually rotary drive trains thus 3 cylinders should be change to rotary instead. 
Auto.new$name[Auto.new$cylinder==3]#the four cars in the three cylinder are all different models of the mazda rx series (which were all offered in a rotary engine configuration). 
Auto.new %>% 
	group_by(origin) %>%
	count(mpg01)
??Auto
#As you can see in this table, the origin of the car is a factor on whether a car in the dataset surpasses the 22 mpg threshold. Though United States automakers manufactured a total of 72 fuel efficient cars the proportion is thrown off by the fact that they made a total of 173 fuel inefficient cars. Japan has the best proportion in this category in that they made a total of 70 (fuel efficient) and 9 (fuel inefficient) automobiles. 
Auto_year <- Auto.new %>% 
	group_by(year) %>%
	count(mpg01)
fix(Auto_year)# Looking at this command you can see that the number of cars that reached 22 miles per gallon were few compared with those that did not reach this threshold during the 1970s but then during the 1980s there was a massive increase in vehicles surpassed this threshold and a decrease in fuel inefficient vehicles. 
#And so the year variable will most likely make a good predictor within the regression (or even the LDA, QDA, or KNN model). 

#(c) Splitting the data into a training set and a test set:
nrow(Auto.new)# I think I will split the dataset 300 for the training set and 92 for the testing set. 
Auto.train <- Auto.new[1:300,]
dim(Auto.train)
Auto.test <- Auto.new[301:392,]
dim(Auto.test)
for(i in 1:length(Auto.new$name)){
	Auto.name[i] <- Auto.test$name[i] == Auto.new$name[i]
}
Auto.new$name[93]# now I understand the names are most likely too complicated to use this method. Will need to think of another way to carry out this command. 
# Guess I might have to check on these vectors manually. 
Auto.glm <- glm(mpg01 ~ horsepower + year + weight + origin, family = binomial, data = Auto.new)
summary(Auto.glm)# the only variable that doesn't have a respectible p value is the origin variable (but still most likely the United States vehicle market is the reason for this small p-value). The rest of the variables have p-values that are well bellow the 0.05 sigificance level threshold.
#(d)
Auto.lda <- lda(mpg01 ~ horsepower + year + weight + origin, family = binomial, data = Auto.train)
Auto.lda.pre <- predict(Auto.lda, Auto.test)
Auto.lda.class <- Auto.lda.pre$class
table(Auto.lda.class, Auto.test$mpg01)
mean(Auto.lda.class == Auto.test$mpg01)# this model has a correct prediction rate of 88.04 percent with an error rate of 
100 - 88.04 # 11.96 percent, which is very good. I believe that the glm() function (or rather the logistical regression method) will obtain similar numbers. The quadratic discriminant analysis will most like result in a less accurate model. 

#(e)
Auto.qda <- qda(mpg01 ~ horsepower + year + weight + origin, data = Auto.train)
Auto.qda.pre <- predict(Auto.qda, Auto.test)
Auto.qda.class <- Auto.qda.pre$class
table(Auto.qda.class, Auto.test$mpg01)
mean(Auto.qda.class == Auto.test$mpg01)# Interesting the model has a 89.13 correct prediction rate and a 
100 - 89.13 # 10.87 percent error rate. In other words, the model become more accurate with the QDA method. 

#(f) 
Auto.glm <- glm(mpg01 ~ horsepower + year + weight + origin, family = binomial, data = Auto.train, subset = train)
Auto.probs <- predict(Auto.glm, Auto.test, type = "response")
Auto.glm.pre <- rep("False", nrow(Auto.test))
Auto.glm.pre[Auto.probs > 0.5] <- "True"
table(Auto.glm.pre, Auto.test$mpg01)
mean(Auto.glm.pre != Auto.test$mpg01)# According to this command this model has a 100 percent error rate will need to see what the problem is. Most likely the pitfalls of the Bayes thereom is showing in this command (will need to use the Lapace estimate as a means to fix this problem). 

#(g)
set.seed(1) 
X.train <- Auto.train[,c(4,5,7,8)]
X.test <- Auto.test[,c(4,5,7,8)]
X.train.mpg01 <- Auto.train[,10]
Auto.knn <- knn(X.train, X.test, X.train.mpg01, k = 1)
table(Auto.knn, Auto.test$mpg01)
mean(Auto.knn != Auto.test$mpg01)# The error rate was calculated at 22.8 percent at a k value of 1.
knn.error <- list()
for(i in 1:30){
	Auto.knn <- knn(X.train, X.test, X.train.mpg01, k = i)
	knn.error[i] <- mean(Auto.knn != Auto.test$mpg01)
} 
knn.error# It seems that the knn error rate starts to plateau at around k = 29 Will need to look into this through another for() loop.  
for(i in 1:50){
	Auto.knn <- knn(X.train, X.test, X.train.mpg01, k = i)
	knn.error[i] <- mean(Auto.knn != Auto.test$mpg01)
} 
knn.error# I believe that the best number one can obtain with the least amount of k groups is k = 17 which brings the error rate down to 17.39 percent. In fact, I did see 17.39 percent on multiple k values. will need to look into this.

#(12)
#(a)
Power <- function(x){
	power.force<- x^3
	print(power.force)	
}
Power(2)

#(b)
Power02 <- function(x, a){
	power.force <- x^a
	print(power.force)
}
Power02(2,3)# Same answer as the funciton above.
Power02(3,8)

#(c)
Power02(c(10, 8, 131), c(3,17,3))

#(d)
Power3 <- function(x, a){
	power.force <- x^a
	return(power.force)
}
Power3(2,4)

#(e)
Power3(x = c(1:10), a = 2)
plot(x = c(1:10), y = Power3(x= c(1:10), a = 2), xlab = "x", ylab= "y")
lines(x = c(1:10), y = Power3(x = c(1:10), a =2))

Powerplot <- function(x, a){
	x <- rep(NA, length(x))
	y <- rep(NA, length(x))
	for (i in 1:length(x)){
	y[i] <- x[i]^a
	}
	plot(x, y, xlab = "x", ylab = "y", pch = 16)
	lines(x = x, y = y, lty =2)
}
Powerplot(x = c(1:10), a = 3)# Seem to can't get this function to work correctly. will need to come back to this question later.

Powerplot2 <- function(x, a){
	x <- x
	y <- Power3(x, a)
	plot(x = x, y = y, pch =16, ylab = "y", xlab = "x")
	lines(y = y, x = x)
}
Powerplot2(1:10, 3)# Perfect this method worked. Will need to remember how to combine loops into funcitons later on in my studies. 


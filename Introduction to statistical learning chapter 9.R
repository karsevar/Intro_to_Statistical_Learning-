### chapter 9 Support Vector Machines:
## 9.1 Maximal Margin Classifer:
#In this section, we define a hyperplane and introduce the concept of an optimal separating hyperplane.

##9.1.1 What is a Hyperplane?
#In a p-dimensional space, a hyperplane is a flat affine subspace of dimension p - 1. For instance, in two dimensions, a hyperplane is a flat one-dimensional subspace -- in other words, a line. In three dimensions, a hyperplane is a flat two-dimensional subspace --- that is, a plane. In p > 3 dimensions, it can be hard to visualize a hyperplane, but the notion of a p - 1 dimensional flat subspace still applies.

#(interesting) the hyperplane formula uses the same components as the linear regression line equation. Note that (9.1) is simply the equation of a line, since indeed in two dimensions a hyperplane is a line.

#(important addition for multidimensional hyperplane) The equation can be reinterpreted for higher dimensional data sets in much the same equation as multi-linear regression models (they both have very much the same components). 

#Now, suppose that X does not satisfy (9.2); rather:
	#beta_0 + beta_1X_1 + ... +beta_pX_p > 0.
#then this tells us that X lies to one side of the hyperplane. On the other hand, if
	#beta_0 + beta_1X_1 + ... +beta_pX_p < 0,
#then X lies on the other side of the hyperplane. So we can think of the hyperplane as dividing p-dimensional space into halves (much like algebric inequality). One can easily determine on which side of the hyperplane a point lies by simply calculating the sign of the left hand side of (9.2). 

##9.1.2 Classification Using a separating hyperplane:
#Now suppose that we have a n*p data matrix X that consists of n training observations in p-dimensional space and that these observations fall into two classes. Our goal is to develop a classifier based on the training data that will correctly classify the test observation using its feature measurements. 

#Suppose that it is possible to construct a hyperplane that separates the training observations perfectly according to their class labels. We can label the observations from the blue class as y_i = 1 and those from the purple class as y_i = -1. Then a separating hyperplane has the property that
	#beta_0 + beta_1X_i1 + ... +beta_pX_ip > 0 if y_i = 1,
#and 
	#beta_0 + beta_1X_i1 + ... +beta_pX_ip < 0 if y_i = -1.
	
#Equivalently, a separating hyperplane has the property that
	#y_i(beta_0 + beta_1X_i1 + ... +beta_pX_ip) > 0 
#For all i = 1, ..., n.

#If a separating hyperplane exists, we can use it to construct a very natural classifier: a test observation is assigned a class depending on which side of the hyperplane it is located. The right-hand panel of figure 9.2 shows an example of such a classifier. That is, we classify the test observation x^8 based on the sign of f(x^*) = beta_0 + beta_1X^*_1 + ... +beta_pX^*_p. If f(x^*) is positive, then we assign the test observation to class 1, and it f(x^*) is negative, then we assign it to class -1. We can also make use of the magnitude of f(x^*). If f(x^*) is far from zero, then this means that x^* lies far from the hyperplane and so we can be confident about our class assignment for x^*. On the other hand, if f(x^*) is clase to zero, then x^* is located near the hyperplane, and so we are less certain about the class assignment for x^*. Not surprisingly a classifier that is based on a separating hyperplane leads to a linear decision boundary.

##9.1.3 The Maximal Margin Classifier:
#In general, if our data can be perfectly separated using a hyperplane, then there will in fact exist an infinite number of such hyperplanes. This is because a given separating hyperplane can usually be shifted a tiny bit up or down, or rotated, without coming into contact with any of the observations. 

#there are three possible hyperplanes to use as a means to remedy this problem:
	#The natural choice is the maximal margin hyperplane, which is the separating hyperplane that is farthest from the training observations. That is, we can compute the (perpendicular) distance from each training observations to a given separating hyperplane. The maximal margin hyperplane is the separating hyperplance for which the margin is largest -- that is, it is the hyperplane that has the farthest minimum distance to the training observations. We can then classify a test observation based on which side of the maximal margin hyperplane it lies. This is known as the maximal margin classifier. We hope that a classifier that has a large margin on the training data will also have a large margin on the test data, and hence will classify the test observations correctly. (The down side with this method is that the maximal margin classifier tends to overfit the training data when p is large (the very much sounds just like the k-NN algorithm except that it seems to be parametric)).
#If beta_0, beta_1, ..., beta_p are the coefficients of the maximal margin hyperplane, then the maximal margin classifier classifiers the test observations x^* based on the sign of f(x^*) = beta_0 + beta_1X^*_1 + ... +beta_pX^*_p.

#(important addition) It's important to keep in mind that for a maximal margin hyperplane to be created a number of points are needed as achors to measure the approtriate distance (or rather margin between the two classes). These observations are known as support vectors, wince they are vectors in p-dimensional space (in figure 9.3, p = 2) and they support the maximal margin hyperplane in the sense that if these points were moved slightly then the maximal margin hyperplane plane would move as well. Interestingly, the maximal margin hyperplane depends directly on the support vectors, but not on the other observations: a movement to any of the other observations would not affect the separating hyperplane, provided that the observation's movement does not cause it to cross the boundary set by the margin. 

##9.1.4 Construction of the maximal margin classifier:
#To read more about the optimization problem and mathematics of calculating the hyperplane consult page 343. 
#The constraints (9.10) and (9.11) ensure that each observation is on the correct side of the hyperplane and at least a distance M from the hyperplane. Hence, M represents the margin of our hyperplane, and the optimization problem chooses beta_0, beta_1, ..., beta_p to maximize M. This is exactly the definition of the maximal margin hyperplane.

##9.1.5 The non-separable case:
#If a separating hyperplane doesn't exist for the classes, the optimization problem (9.9) - (9.11) has no solution with M > 0. An example is shown in Figure 9.4. In this case, we cannot exactly separate the two classes. However, as we will see in the next section, we can extend the concept of a separating hyperplane in order to develop a hyperplance that almost separates the classes, using a so-called soft margin. the generalization of the maximal classifier to the non-separable case is known as the support vector classifier.

##9.2 Support vector classifiers
##9.2.1 Overview of the support vector classifier:
#A classifier based on a separating hyperplane will necessarily perfectly classify all of the training observations; this can lead to sensitivity to individual observations. Hence if one was to change one of the support vectors just to a tiny margin the maximal margin hyperplane can change drastically (thus possibly decreasing the margin). This is problematic because the distance of an observation from the hyperplane can be seen as a measure of our confidence that the observation was correctly classified. Moreover, the fact that the maximal margin hyperplane is extremely sensitive to a change in a single observation suggests that it may have overfit the training data. 

#In this case, we might be willing to consider a classifier based on a hyperplane that does not perfectly separate the two classes, in the interest of:
	#Greater robustness to individual observations, and 
	#Better classification of most of the training observations

#The support vector classifier, or rather the soft margin classifier, seeks allow some observations to be on the incorrect side of the margin, or even the incorrect side of the hyperplane. (the margin is soft because it can be violated by some of the training observations). 

##9.2.2 Details of the support vector classifier:
#the support vector classifier classifies a test observation depending on which side of a hyperplane it lies. The hyperplane is chosen to correctly separate most of the training observations into the two classes, but may misclassify a few observations. 
#To see the optimization problem for support vector classifiers look at page 346.

#The problem (9.12) - (9.15) seems complex, but insight into its behavior can be made through a series of simple observations presented below. First of all, the slack variable E_i tells us where the ith observation is located, relative to the hyperplane and relative to the margin. If E_i = 0 then the ith observation is on the correct side of the margin. If E_i > 0 then the ith observation is on the wrong side of the margin, and we say that the ith observation has violated the margin. If E_i > 1 then it is on the wrong side of the hyperplane. 

#We now consider the role of the tuning parameter C. In (9.14), C bounds the sum of the E_i's, and so it determines the number of severity of the voilations to the margin (and to the hyperplane) that we will tolerate. We can think of C as a budget for the amount that the margin can be violated by the n observations. If C = 0 then there is no budget for violations to the margin, and it must be the case that E_1 = ... = E_n = 0, in which case (9.12)-(9.15) simply amounts to the maximal margin hyperplane optimization problem. For C>0 no more than C observations can be on the wrong side of the hyperplane, because if an observation is on the wrong side of the hyperplane then E_i > 1, and (9.14) requires that sum(E_i) <= C. As the budget C increates, we become more tolerant of violations to the margin, and so the margin will widen. Conversely, as C decreases, we become less tolerant of violations to the margin and so the margin narrows. 

#In practice, C is treated as a tuning parameter that is generally chosen via cross-validation. As with the tuning parameters that we have seen throughout this book, C controls the bias-variance trade off of the statistical learning technique. When C is small, we seek narrow margins that are rarely violated; this amounts to a classifier that is highly fit to the data, which may have low bias but but high variance. On the other handm when C is larger, that margin is wider and we allow more violations to it;this amounts to fitting the data less hard and obtaining a classifier that is potentially more biased but may have lower variance. 

#the optimization problem (9.12)-(9.15) has a very interesting property: it turns out that only observations that either lie on the margin or that violate the amrgin will affect the hyperplane, and hence the classifier obtained. Observations that lie directly on the margin, or on the wrong side of the margin for their class, are known as support vectors. 

#(How the different philosophy in support vector classification affects the margin location for the soft margin hyperplane method) The fact that only support vectors affect the classifier is in line with our previous assertion that C controls the bias-variance trade off of the support vector classifier. When the tuning parameter C is large, then the margin is wide, many observations violate the margin, and so there are many support vectors. In this case, many observations are involved in determining the hyperplane. In other words, if C is large the hyperplane will have high bias and low variance and if C is small the model will have low bias and high variance (due to the fact that there are less support vectors to dictate the position of the margins).

#The fact that the support vector classifier's decision rule is based only on a potentially small subset of the training observations means that it is quite robust to the behavior of observations that are far away from the hyperplane. This property is distinct from some of the other classification methods that we have seen in preceding chapters, such as linear discriminant analysis. Recall that the LDA classification rule depends on the mean of all of the observations within each class, as well as the within-class covariance matrix computed using all of the observations. In contrast, logistic regression, unlike LDA, has very low sensitivity to observations far from the decision boundary. 

##9.3 Support Vector Machines:
#We first discuss a general mechanism for converting a linear classifier into one that produces non-linear decision boundaries. We then introduce the support vector machine, which does this in an automatic way. 

##9.3.1 Classification with Non-linear Decision Boundaries:
#In the case of the support vector classifier, we could address the problem of possibly non-linear boundaries between classes in the same way as cubic and quadratic transformations for regression models, by enlarging the feature space using quadratic, cubic, and even higher order polynomial functions of the predictors. For instance, rather than fitting a support vector classifier using p features
		#X_1, X_2, ..., X_p, 
#we could instead fit a support vector classifier using 2p features
		#X_1, X^2_1, X_2, X^2_2, ..., X_p, X^2_p.

#To see the resulting equation look at page 350. 
#Why does this lead to a non-linear decision boundary? In the enlarged feature space, the decision boundary that results from (9.16) is in fact linear. But in the original feature space, the decision boundary is of the form q(x) = 0, where q is a quadratic polynomial, and its olsutions are generally non-linear. One might additionally want to enlarge the feature space with higher order polynomial terms, or with interaction terms of the form X_jX_j'. Alternatively, other functions of the predictors could be considered rather than polynomials. It is not hard to see that there are many possible ways to enlarge the feature space, and that unless we are careful, we could end up with a huge number of features. Then computations would become unmanageable. The support vector machine, which we present next, allows us to enlarge the feature space used by the support vector classifier in a way that leads to efficient computations. 

##9.3.2 The support vector machine:
#The support vector machine (SVM) is an extension of the support vector classifier that results form enlarging the feature space in a specific way, using kernels. We may want to enlarge our feature space in order to accommodate a non-linear boundary between the classes. The kernel approach that we describe here is simply an efficient computational approach for enacting this idea. 

#to see the mathematics involved look at page 351.

#To summarize, in representing the linear classifier f(x), and in computing its coefficients, all we need are inner products. Now suppose that every time the inner product (9.17) appears in the representation (9.18), or in a calculation of the soltution for the support vectors classifier, we replace it with a generalization of the inner product of the form
		#K(x_i, x_i'),
#where K is some function that we will refer to as a kernel. A kernel is a function that quantifies the similarity of two observations. For instance, we could simply take 
		#K(x_1, X_i') = sum(x_ij*x_ij') Don't quote me on this represenation. I'm still learning advanced mathematics.

#which would just give us back the support vector classifier. Equation (9.21) is known as a linear kernel because the support vector classifier is linear in the features;the linear kernel essentially quantifies the similarity of a pair of observations using Pearson (standard) correlation. To see the polynomial kernel equaiton look at page 352. 

#The polynomial kernel of degree d leads to a more flexible decision boundary. It essentially amounts to fitting a support vector classifier in a higher dimensional space involving polynomials of degree d, rather than in the original feature space. When the support vector classifier is combined with a non-linear kernel such as (9.22), the resulting classifier is known as a support vector machine. When d = 1, then the SVM reduces to the support vector classifier seen earlier in this chapter. 

#For an interesting kernel equation that fits radial decision boundaries look at page 352 for the radial kernel equation. 

##9.4 SVMs with more than two classes:
#It turns out that the concept of separating hyperplanes upon which SVMs are based does not lend itself naturally to more than two classes. Though a number of proposals for extending SVMs to the K-class case have been made, the two most popular are the one versus one and one versus all approaches.

##One-versus-one classification:
#A one versus one or all pairs approach constructs (K/2) SVMs, each of which compares a pair of classes. For example, one such SVM might compare the kth class, coded as +1, to the k'th class, coded as -1. We classify a test observation using each of the (k/2) classifiers, and we tally the number of times that the test observation is assigned to each of the K classes. The final classification is performed by assigning the test observation to the class to which it was most frequently assigned in these (K/2) pairwise classifications. 

##9.4.2 One - versus all classification:
#The one versus all approach is an alternative procedure for applying SVMs in the case of K > 2 classes. We fit K SVMs, each time comparing one of the K classes to the remaining K - 1 classes. Let beta_0k, beta_1k, ..., beta_pk denoted the parameters that result from fitting an SVM comparing the kth class (coded as +1) to the others (coded as -1). Let x^* denote a test observation. We assign the observation to the class for which beta_0k + beta_1kx^*_1 + ... + beta_pkx^*_p is largest, as this amounts to a high level of confidence that the test observation belongs to the kth class rather than to any of the other classes. 

## 9.6 Lab: support Vector Machines:
#The two packages that the author will be using in these exercises are e1071 and LiblineaR (though the exercise material will be written in e1071 and LiblineaR is just a suggested R package that he recommends for very large linear problems). 

##9.6.1 Support Vector classifier:
# e1071 functions for SVM methods:
	#the svm() function can be used to fit a support vector classifier when the argument kernel = "linear" is used. This function uses a slightly different formulation from (9.14) and (9.25) for the support vector classifier. A cost argument allows us to specify the cost of a violation to the margin. When the cost argument is small, then the margins will be wide and many support vectors will be on the margin or will violate the margin. When the cost argument is large, then the margins will be narrow and there will be few support vectors on the margin or violating the margin (remember that the cost argument is actually the lambda/C value of the support vector equation and hence the same principles apply to its implementation). 
	
#We now use the svm() function to fit the support vector classifier for a given value of the cost parameter. Here we demonstrate the use of this function on a two-dimensional example so that we can plot the resulting decision boundary. We begin by generating the observations, which belong to two classes. 
set.seed(1)
x <- matrix(rnorm(20*2), ncol = 2)
y <- c(rep(-1, 10), rep(1,10))
x[y ==1,]<- x[y==1,] + 1# will need to look into what this line means. 

#We begin by checking whether the classes are linearly separable. 
plot(x, col=(3-y))#the col argument separates the y values into two classes 4 and 2. Most likely the x[y==1,] <- x[y==1,] are used to label the different classes (needed to make the support vector classifier method work).

#they are not. Next, we fit the support vector classifier. Note that in order for the svm() function to perform classification (as opposed to SVM-based regression), we must encode the response as a function variable. We now create a data frame with the response coded as a factor. 
dat <- data.frame(x=x, y = as.factor(y))
library(e1071)
svmfit <- svm(y~., data = dat, kernel = "linear", cost = 10, scale = FALSE) #Interestingly despite the author saying that the decision boundary is none linear he still set the kernel argument to linear and in addition I find the cost argument being set to 10 to be excessive. 

#The argument scale = FALSE tells the svm() function not to scale each feature to have mean zero or standard deviation one; depending on the application one might prefer to use scale = TRUE.

plot(svmfit, dat)# Now I understand why I set kernel to be linear and the cost argument to 10, he was thinking to side set the need for a quadratic kernel through allowing more violations to be permitted within the decision margin. The problem with this graphic is that all the observations are in violation to the support vector classifier (thus making this model inaccurate at best). 

#Note that the two arguments to the plot.svm() function are the output of the call to svm(), as well as the data used in the call to svm(). The region of feature space that will be assigned to the -1 class as shown in the light blue, and the region that will b assigned to the +1 class is shown in purple. The decision boundary between the two classes is linear (because we used the argument kernel = "linear"), though due to the way in which the plotting function is implemented in this library the decision boundary looks somewhat jagged in the plot. We see that in this case only one observation is misclassified (the is quite interesting perhaps I miss interpreted the graphic will need to look into how miss classified points are labeled). (Note that here the second feature is plotted on the x-axis and the first feature is plotted on the y -axis, in constrast to the behavior of the usual plot() function in R.) The support vectors are plotted as crosses and the remaining observations are plotted as circles; we see here that there are seven support vectors. We can determine their identities as follows:
svmfit$index

#We can obtain some basic information about the support vector classifier fit using the summary() command:
summary(svmfit)

#This tells us, for instance, that a linear kernel was used with cost = 10 and that there were seven support vectors, four in one class and three in the other. 

#What if we instead used a smaller value of the cost parameter?
svmfit <- svm(y~., data = dat, kernel = "linear", cost = 0.1, scale = FALSE)
plot(svmfit, dat)
svmfit$index

#Now that a smaller value of the cost parameter is being used, we obtain a larger number of support vectors, because the margin is now wider. Unfortunately, the svm() function does not explicitly output the coefficients of the linear decision boundary obtained when the support vector classifier is fit, nor does it output the width of the margin. 

#The e1071 library includes a built-in function, tune(), to perform cross validation. By default, tune() performs ten-fold cross-validation on a set of models of interest. In order to use this funciton, we pass in relevant information about the set of models that are under consideration. The following command indicates that we want to compare SVMs with a linear kernel,using a range of values of the cost parameter.
tune.out <- tune(svm, y~.,data = dat, kernel = "linear", ranges = list(cost = c(0.001, 0.01,0.1, 1, 5, 10, 100)))
#We can easily access the cross validation errors for each of these models using the summary() command.
summary(tune.out)

#We see that cost  0.1 results in the lowest cross validation error rate. The tune() function stores the best model obtained, which can be accessed as follows:
bestmod <- tune.out$best.model
summary(bestmod)

#the predict() function can be used to predict the class label on a set of test observtions, at any given value of the cost parameter. We begin by generating a test data set.
xtest <- matrix(rnorm(20*2), ncol = 2)
ytest <- sample(c(-1,1), 20, rep = TRUE)
xtest[ytest ==1,]<- xtest[ytest==1,] +1
testdat <- data.frame(x = xtest, y = as.factor(ytest))
#Now we predict the class labels of these test observations. Here we use the best model obtained through cross validation in order to make predictions. 
ypred <- predict(bestmod, testdat)
table(predict = ypred, truth = testdat$y)
1-((6 + 10)/20)# This model has an error rate of 0.2 or rather 20 percent. 

#thus, with this value of cost, 16 of the test observations are correctly classified. What if we had instead used cost = 0.01?
svmfit <- svm(y~., data = dat, kernel = "linear", cost 0.01, scale = FALSE)
ypred <- predict(svmfit, testdat)
table(predict = ypred, truth = testdat$y)
#In this case no additional observations were misclassified or successfully classified. 

#Now consider a situation in which the two classes are linearly separable. Then, we can find a separating hyperplane using the svm() function. We first furthedr separate the two classes in our simulated data so that they are linearly separable:
x[y==1,] <- x[y==1,]+0.5# I have a feeling that the maximal margin classifier will be unable to split the data into exactly two groups without some violations. I will need to look into a series of fixes for this problem.  
plot(x, col = (y+5)/2, pch = 19)

#Now the observations are just barely linearly separable. We fit the support vector classifier and plot the resulting hyperplane, using a very large value of cost so that no observations are misclassified.
data <-data.frame(x=x, y = as.factor(y))
svmfit <- svm(y~., data = dat, kernel = "linear", cost = 1e5)
summary(svmfit)
plot(svmfit, dat)

#No training errors were made and only three support vectors were used. However, we can see from the figure that the margin is very narrow (because the observations that are not support vectors, indicated as circles, are very close to the decision boundary). It seems likely that this model will perform poorly on test data. We now try a smaller value of cost:
svmfit <- svm(y~., data = dat, kernel = "linear", cost = 1)
summary(svmfit)# now this model has 12 support observations.
plot(svmfit, dat)
#Using cost = 1, we misclassify a training observation, but we also obtain a much wider margin and make use of twelve support vectors. It seems likely that this model will perform better on test data than the model with cost = 1e5.

##9.6.2 support vector machine:
#In order to fit an SVM using a non-linear kernel, we once again use the svm() function. However, now we use a different value of the parameter kernel. To fit an SVM with a polynomial kernel we use kernel = "polynomial", and to fit an SVM with a radial kernel we use kernel ="radial". In the former case we also use the degree argument to specify a degree for the polynomial kernel, and in the latter case we use gamma to specify a value for the radial basis kernel.

#We first generate some data with a non-linear class boundary, as follows: 
set.seed(1)
x <- matrix(rnorm(200*2), ncol = 2)
x[1:100,] <- x[1:100,]+2
x[101:150,] <- x[101:150,]-2
y <- c(rep(1,150), rep(2,50))
dat <- data.frame(x=x, y=as.factor(y))

#Plotting the data makes it clear that the class boundary is indeed non-linear:
plot(x, col =y)#The author was right the data frame created earlier looks like the best kernel fit will be radial instead of polynomial, will love to see how the author plans on solving this problem.

#The data is randomly split into training and testing groups. We then fit the training data using the svm() function with a radial kernel and (radial value) = 1:
train <- sample(200, 100)
svmfit <- svm(y~., data = dat[train,], kernel = "radial", gamma = 1, cost = 1)
plot(svmfit, dat[train,])
#the plot shows that the resulting SVM has a decidedly non-linear boundary. The summary() function can be used to obtain some information about the SVM fit:
summary(svmfit)

#We can see from the figure that there are a fair number of training errors in this SVM fit. If we increase the value of the cost, we can reduce the number of training errors. However, this comes at the price of a more irregular decision boundary that seems to be at risk of overfitting the data. 
svmfit <- svm(y~., data = dat[train,], kernel = "radial", gamma = 1, cost = 1e5)
plot(svmfit, dat[train,])# The author is right this radial decision classifier really does over fit the training data. 

#We can perform cross validation using tune() to select the best choice of gamma and cost for an SVM with a radial kernel:
set.seed(1)
tune.out <- tune(svm, y~.,data = dat[train,], kernel = "radial", ranges = list(cost = c(0.1,1,10,100,1000), gamma = c(05,1,2,3,4)))
summary(tune.out)
#According to this command and my computer the best gamma and cost parameter for this data set is cost = 1 and gamma = 1. We can view the test set predictions for this model by applying the predict() function to the data. Notice that to do this we subset the dataframe dat using -train as an index set. 
table(true = dat[-train,"y"], pred = predict(tune.out$best.model, newx=dat[-train,]))
1-((56+6)/100)# The error rate is 38 percent.

##9.6.3 ROC curves:
#The ROCR package can be used to produce ROC curves such as those in Figures 9.10 and 9.1. We first write a short function to plot an ROC curve given a vector containing a numerical score for each observation, pred, and a vector containing the class label for each observation, truth. 

library(ROCR)
rocplot <- function(pred, truth, ...){
	predob <-prediction(pred, truth)
	perf <- performance(predob, "tpr","fpr")
	plot(perf, ...)
}

#SVMs and support vector classifiers output class labels for each observation. However, it is also possible to obtain fitted values for each observation, which are the numerical scores used to obtain the class labels. For instance, in the case of a support vector classifier, the fitted value for an observation X = (X_1, X_2, ..., X_p)^T takes the form betahat_0 + betahat_1X_1 + betahat_2X_2 + ... + betahat_pX_p. For an SVM with a non-linear kernel, the equation that yields the fitted value is given in (9.23). In essence, the sign of the fitted value determines on which side the decision boundary and observation lies (my sign they mean the f(x^*) symbol). Therefore, the relationship between the fitted value and the class prediction for a given observation is simple: if the fitted value exceeds zero then the observation assigned to one class, and if it is less than zero than it is assigned to the other. In order to obtain the fitted values for a given SVM model fit, we use decision.values = TRUE when fitting svm(). Then the predict() function will output the fitted values. 

svmfit.opt <- svm(y ~., data = dat[train,], kernel = "radial", gamma = 2, cost = 1, decision.values = TRUE)
fitted <- attributes(predict(svmfit.opt, dat[train,], decision.values = TRUE))$decision.values

#Now we can produce the ROC plot:
par(mfrow = c(1,2))
rocplot(fitted, dat[train, "y"], main ="Training Data")
#SVM appears to be producing accurate predictions. By increasing the radial value we can produce a more flexible fit and generate further improvements in accuracy. 

svmfit.flex <- svm(y~., data = dat[train,], kernel = "radial", gamma = 50, cost = 1, decision.values = TRUE)
fitted <- attributes(predict(svmfit.flex, dat[train,], decision.values = TRUE))$decision.values
rocplot(fitted,dat[train,"y"], add = TRUE, col = "red")

#However, these ROC curves are all on the training data. We are really more interested in the level of prediction accuracy on the test data. When we compute the ROC curves on the test data, the model with gamma = 2 appears to provide the most accurate results. 
fitted <- attributes(predict(svmfit.opt, dat[-train,], decision.values = TRUE))$decision.values 
rocplot(fitted,dat[-train,"y"], main = "test data")
fitted <- attributes(predict(svmfit.flex, dat[-train,], decision.values = TRUE))$decision.values 
rocplot(fitted, dat[-train, "y"], add = TRUE, col = "red")

##9.6.4 SVM with Multiple Classes:
#If the response is a factor containing more than two levels, then the svm() function will perform multi-class classification using the one-versus-one approach. We explore that setting here by generating a third class of observations.
set.seed(1)
x <- rbind(x, matrix(rnorm(50*2), ncol = 2))
y <- c(y, rep(0,50))
x[y==0,2]<- x[y==0,2]+2
dat <- data.frame(x=x, y=as.factor(y))
par(mfrow =c(1,1))
plot(x, col =(y+1))#It seems that three class were made through this command call. 

#We now fit an SVM to the data:
svmfit <- svm(y~., data = dat, kernel="radial", cost = 10, gamma = 1)
plot(svmfit, dat)

#The e1071 library can also be used to perform support vector regression if the response vector that is passed in to svm() is numerical rather than a factor. 

##9.6.5 Application to Gene Expression Data:
#We now examine the Khan data set, which consists of a number of tissue samples corresponding to four distinct types of small round blue cell tumors. For each tissue sample, gene expression measurements are available. The data set consists of training data, xtrain and ytrain, and testing data, xtest and ytest. 
library(ISLR)
names(Khan)
head(Khan)# Yeah the Khan data set is huge. 
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)#It's important to note that the ytest and ytrain objects are all vectors of length 20 and 63 repectively. And that the xtrain and xtest are all high dimensional matrixes. 

#this data set consists of expression measurements for 2308 genes. The training and test sets consist of 63 and 20 observations respectively. 
table(Khan$ytrain)
table(Khan$ytest)# The response variable has four different levels. Meaning that the author has one of two options the one versus one approach or one versus all approach. 

#We will use a support vector approach to predict cancer subtype using gene expression measurements. In this data set, there are a very large number of features relative to the number of observations. This suggests that we should use a linear kernel, because the additional flexibility that will result from using a polynomial or radial kernel is unnecessary (interesting so the talk with the R meetup was true. Geneologists don't need to use none parametric methods). 
dat <- data.frame(x = Khan$xtrain, y = as.factor(Khan$ytrain))
out <- svm(y~., data = dat, kernel = "linear", cost = 10)
summary(out)
table(out$fitted, dat$y)
#We see that there are no training errors. In fact, this is not surprising, because the large number of variables relative to the number of observations implies that it is easy to find hyperplanes that fully separate the classes. We are most interested not in the training error rate but rather in the performance on the test observations. 

dat.te <- data.frame(x= Khan$xtest, y = as.factor(Khan$ytest))
pred.te <- predict(out, newdata = dat.te)
table(pred.te, dat.te$y)
#We see that using cost=10 yields two test set errors on this data. 

## Exercises:
##Conceptual:
#1.) This problem involves hyperplanes in two dimensions:
#(a) Asadoughi's solution: Again I really need to fix my mathematical proficiency. My ignorance in this subject is really slowing me down. 
set.seed(1)
x1 <- -10:10 
x2 <- 1 + 3 * x1
plot(x1, x2, type = "l", col = "red")
text(c(0), c(-20), "greater than 0", col = "red")
text(c(0), c(20), "less than 0", col = "red")

#2.) 
#(a) Asadoughi's solution. Since my mathematical skills can only make this equation into a parabola, I will need to yet again lean on asadoughi. According to his logic the equation (1+x1)^2 + (2-x2)^2 = 4 is actually a circle with a radius of 2 and a center (-1,2). 
radius <- 2
plot(NA, NA, type = "n", xlim = c(-4,2), ylim = c(-1,5), asp = 1, xlab = "x1", ylab = "x2")
symbols(c(-1), c(2), circles = c(radius), add = TRUE, inches = FALSE)
#No way Asadoughi is a genius. 

#(b) 
radius <- 2
plot(NA, NA, type = "n", xlim = c(-4,2), ylim = c(-1,5), asp = 1, xlab = "x1", ylab = "x2")
symbols(c(-1), c(2), circles = c(radius), add = TRUE, inches = FALSE)
text(c(-1), c(2), "<4")
text(c(-4), c(2), ">4")

#(c) 
plot(x = c(0,-1,2,3), y = c(0,1,2,8), xlab = "x1", ylab = "x2")
symbols(c(-1), c(2), circles = c(radius), add = TRUE, inches = FALSE)
text(c(-1), c(2), "<4; red")
text(c(3), c(2), ">4: blue")
#Only the points c(2,2) and c(3,8) are located within the blue category. 

#(d) Asadoughi's solution:
#The decision boundary is a sum of quadratic terms when expanded. The equation can be seen in Asadoughi's github. It's not really as complicated as the other entries but sadly I don't understand how he created the solution. Really need to get better at mathematics. 

#3.) 
#(a) 
x1_red <- c(3,2,4,1)
x2_red <- c(4,2,4,4)
x1_blue <- c(2,4,4)
x2_blue <- c(1,3,1)
plot(x = c(x1_red), y = c(x2_red), col = c("red"), ylim = c(0,7), xlim = c(0,6))
points(x = x1_blue, y = x2_blue, col = "blue")
abline(0.25,0.75)# Not perfect but close. 

#(b) Asadoughi's solution:
#(2,2), (4,4)\(2,1), (4,3)\ = > (2,1.5),(4,3.5)\b = (3.5-1.5)/(4-2) = 1\a = x_2 - x_1 = 1.5 - 2 = -0.5 
abline(-0.5, 1, col = "black", lty = 2)

#(c) Asadoughi's solution:
#0.5 - x1 + x2 > 0 

#(d) 
abline(0, 1, lty = 3)
abline(-1,1, lty = 3)

#(e) 
#The Support vectors are red(c(2,1),c(4,3)) and blue(c(2,2), c(4,3)).

#(f) The seventh observation will not affect the maximal margin hyperplane because the maximal margin classifier relies on observations 1, 6, 3, and 2. Observation 7 is too far away from the margin of the hyperplane to have an effect on the overall fit. 

#(g) Asadoughi's solution:
x1_red <- c(3,2,4,1)
x2_red <- c(4,2,4,4)
x1_blue <- c(2,4,4)
x2_blue <- c(1,3,1)
plot(x = c(x1_red), y = c(x2_red), col = c("red"), ylim = c(0,7), xlim = c(0,6))
points(x = x1_blue, y = x2_blue, col = "blue")
abline(-0.8, 1)
#-0.8 - x1 + x2 > 0 

#(h) 
plot(x = c(x1_red), y = c(x2_red), col = c("red"), ylim = c(0,7), xlim = c(0,6))
points(c(5), c(5), col = "blue")
abline(-0.5, 1, col = "black", lty = 2)

##Applied:
#4.) 
set.seed(1)
x <- matrix(rnorm(100*2), ncol = 2)
x[1:50,] <- x[1:50,] + 1.50
x[51:100,] <- x[51:100,] - 1.50
plot(x[,1], x[,2])
y <- c(rep(3,50), rep(2, 50))
plot(x, col = y)# Now I get it this is a linear boundary randomly generated values. And what the author wants me to create are values that are divided through a none parametric decision boundary that has a degree of over 1.

set.seed(131)
x <- matrix(rnorm(200*3), ncol = 2)
y <- c(rep(-1, 50), rep(1,50))
x[y==1,] <- x[y==1,]^2 + 2
plot(x, col = (3-y))
#Asadoughi's solution:
set.seed(131) 
x <- rnorm(100)
y <- 3 * x^2 + 4 + rnorm(100)
train <- sample(100, 50)
y[train] <- y[train] + 3
y[-train] <- y[-train] - 3
plot(x[train], y[train], pch = "+", lwd = 4, col = "red", ylim = c(-4, 20), xlab = "X", ylab = "Y")
points(x[-train], y[-train], pch = "o", lwd = 4, col = "blue")#The following lines create a very interesting data separation boundary. Will need to practice a little more before I can create this same randomly generated points. 

library(e1071)
#the plot clearly shows a non-linear separation. We now create both train and test data frames by taking half of positive and negative classes and creating a new z vector of 0 and 1 for classes.
set.seed(315)
z <- rep(0,100)
z[train] <- 1
#Take 25 observations each from train and -train 
final.train <- c(sample(train,25), sample(setdiff(1:100, train),25))
data.train <- data.frame(x=x[final.train], y = y[final.train], z = as.factor(z[final.train]))
data.test<- data.frame(x=x[-final.train], y=y[-final.train], z = as.factor(z[-final.train]))

#Linear kernel:
svm.linear <- svm(z ~., data = data.train, kernel = "linear", cost = 0.1)
plot(svm.linear, data = data.train)
table(true = data.train[,"z"], pred = predict(svm.linear, newx = data.train))
1-((21+23)/50)
#time to find the best tuning parameter for the linear fit:
tune.linear <- tune(svm, z~., data = data.train, kernel = "linear", ranges = list(cost=c(0.1, 1, 10, 100, 1000)))
summary(tune.linear)# the best cost for the linear kernel model is 0.1. But still this is based on the training error rate. 

#Test error rate for the test dataset using a linear kernel:
table(true = data.test[,"z"], pred = predict(tune.linear$best.model, newx = data.test))
1-(14+14)/50# 44 percent error rate this is very high, hence using a linear kernel isn't really the best course of action. 

#quadratic kernel:
set.seed(1)
tune.quad <- tune(svm, z~., data = data.train, kernel = "polynomial", ranges = list(cost =c(0.1,1,10,100,1000), degree = c(1, 2, 3))) # I already know that the data points were created through a 2 degree function. Hence most likely the degree 2 will give rise the the best degree argument. Not really sure about the cost though.
summary(tune.quad)#That's weird the best degree seting is 1 degree will need to check this. 
#After running the model a couple more times, the function is insistant that the best parameters are degree 1 and cost 1. 
table(true = c(data.test[,"z"]), pred = predict(tune.quad$best.model, newx = data.test))
1-((14+14)/50)# Again this model has a classification error rate of 44 percent just like that of the quadratic model. Which is very much to be expected. 
#I might want to play around with the test data a little bit. Most likely the way the training data is partitioned is causing the functions to believe that a linear model is the best course of action.

#radial kernel:
set.seed(1)
tune.radial <- tune(svm,z~., kernel = "radial", data = data.train, ranges = list(cost =c(0.1,1,10,100,1000), gamma = c(0.01,0.1, 1, 2))) 
summary(tune.radial)#that's weird, the tuning function says that gamma 0.01 and cost 1000 gives rise the best best error rate. Will need to test this out on testing data.
table(true = c(data.test[,"z"]), pred = predict(tune.radial$best.model, newx = data.test))# that's funny the true positive and negative values only increased by one through using the radial model. I hope that asadoughi is coming up with the same values and model arguments.
1-((15+15)/50)# 40 percent classification error rate. This is an improvement over the other models by 5 percent.

#Asadoughi's solution:
#linear fit:
svm.linear <- svm(z~., data = data.train, kernel="linear", cost = 10)
plot(svm.linear, data.train)
table(z[final.train], predict(svm.linear, data.train))
1-((21+23)/50)# this is a training error rate of 12 percent

#polynomial kernel:
set.seed(32545)
svm.poly <- svm(z~., data = data.train, kernel = "polynomial", cost = 10)
plot(svm.poly, data.train)
table(z[final.train], predict(svm.poly, data.train))
1-((15+25)/50)# A training error rate of 20 percent. 

#radial kernel 
set.seed(996)
svm.radial <- svm(z~., data = data.train, kernel = "radial", gamma = 1, cost = 10)
plot(svm.radial, data.train)
table(true = z[final.train], pred = predict(svm.radial, data.train))# this model has a perfect classification rate.

#test error rates:
plot(svm.linear, data.test)
plot(svm.poly, data.test)
plot(svm.radial, data.test)
table(z[-final.train], predict(svm.linear, data.test))
table(z[-final.train], predict(svm.poly, data.test))
table(z[-final.train], predict(svm.radial, data.test))# the radial has the best error rate out of all the model fits. In addition, I might have to watch the tune() function since it gave me the wrong values for all the arguments. 

#5.) 
#(a) 
set.seed(421)
x1 <- runif(500)-0.5
x2 <- runif(500)-0.5
y <- 1 * (x1^2-x2^2 > 0)

#(b)
dat <- data.frame(x1 = x1, x2 = x2, y = as.factor(y))
plot(x = x1, y = x2, col = c(1,2)[dat$y])# I believe that a radial SVM model will work perfectly with this data set. 

#(c)
glm.mod <- glm(y~x1 + x2, data = dat, family = binomial)
summary(glm.mod)

#(d)
train <- sample(1:nrow(dat), nrow(dat)/2)
glm.mod <- glm(y~x1 + x2, data = dat, family = binomial)
glm.prob <- predict(glm.mod, newdata = dat, type = "response")
glm.pred <- ifelse(glm.prob > 0.52, 1,0)
data.pos <- dat[glm.pred == 1,]
data.neg <- dat[glm.pred == 0,]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)# Asadoughi is right there is a linear decision boundary when you set the decision threshold to 0.52.

#(e-f) 
glm.new <- glm(y ~ poly(x1, degree = 2) + poly(x2, degree = 2) + I(x1 * x2), data = dat, family = binomial)
summary(glm.new)
#Funny enough these transformations are all statistically insignficant. 
glm.prob <- predict(glm.new, newdata = dat, type = "response")
glm.pred <- ifelse(glm.prob > 0.52, 1,0)
data.pos <- dat[glm.pred == 1,]
data.neg <- dat[glm.pred == 0,]
plot(data.pos$x1, data.pos$x2, col = "blue", xlab = "X1", ylab = "X2", pch = "+")
points(data.neg$x1, data.neg$x2, col = "red", pch = 4)# this looks like a radial transformation can be used to create the decision boundaries. 

#(g) 
svm.linear <- svm(y~., data = dat, kernel = "linear", cost = 0.1)
svm.pred <- predict(svm.linear, dat)
plot(svm.linear, dat)# Very good. Asadoughi came up with the same answer. The linear kernel method of the SVM function can't find a linear decision boundary. 

#(h)
svm.fit <- svm(y~., data = dat, kernel = "radial", gamma = 3, degree = 30)
plot(svm.fit, dat)
svm.pred <- predict(svm.fit, newx = dat)
plot(x1, x2, col = c(1,2)[svm.pred])
table(dat$y, predict(svm.fit, newx = dat))
1-((231+264)/500)# This model only has a training error rate of 1 percent. Extremely good for just guessing on the gamma and cost argument values. 

#(i)
#It seems that my initial analysis regarding the two classes displaying a radial decision boundary. 

#6.)
#(a)
set.seed(1)
x <- matrix(rnorm(200*2), ncol = 2)
x[1:100,] <- x[1:100,] + 1.5
x[101:200,] <- x[101:200,] - 1.5
plot(x[,1], x[,2])
y <- c(rep(3,100), rep(2, 100))
plot(x, col = y)
x1 <- x[,1]
x2 <- x[,2]
dat <- data.frame(x1 = x1,x2 = x2,y = as.factor(y))

#(b)
lambdas <- seq(from = 1, to = 1000) 
mse <- rep(NA, times = length(lambdas))
false <- rep(NA, times = length(lambdas))
train.table <- matrix()
for(i in 1:length(lambdas)){
	svm.linear <- svm(y~., data = dat, kernel = "linear", cost = lambdas[i])
	train.table <- table(dat[,"y"], predict(svm.linear, newx = dat))
	mse[i] <- 1-((train.table[1,1]+train.table[2,2])/200)
	false[i] <- train.table[1,2]+train.table[2,1]
}
par(mfrow = c(1,2))
plot(x=1:1000, y = mse, type = "l", ylab = "Classification Error", xlab = "Cost value")
plot(x=1:1000, y = false, type = "l", lty = 2, col = "red", ylab = "Number of False Neg. and Pos.", xlab = "Cost value")# training classification and total false positive and negative values decrease with increased cost. Even with that said though the decrease looks more like a set of plateaus, where most of the reduction occurs between C value 0 and 200. Will need to narrow this down.
dev.new()
lambdas <- seq(from = 1, to = 30) 
mse <- rep(NA, times = length(lambdas))
false <- rep(NA, times = length(lambdas))
train.table <- matrix()
for(i in 1:length(lambdas)){
	svm.linear <- svm(y~., data = dat, kernel = "linear", cost = lambdas[i])
	train.table <- table(dat[,"y"], predict(svm.linear, newx = dat))
	mse[i] <- 1-((train.table[1,1]+train.table[2,2])/200)
	false[i] <- train.table[1,2]+train.table[2,1]
}
par(mfrow = c(1,2))
plot(x=1:30, y = mse, type = "l", ylab = "Classification Error", xlab = "Cost value")
plot(x=1:30, y = false, type = "l", lty = 2, col = "red", ylab = "Number of False Neg. and Pos.", xlab = "Cost value")
#From what I can see the main reduction in training classification error occurs at about C value 17 where the improvement stops at 0.020 (or rather 2 percent error rate). Remember that this rate is the training error rate and that increased cost only fits the model closer to the training data. Thus causing over fitting, which translates to higher variance and lower bias. 

#Cross validation error rate:
tune.out <- tune(svm, y~., data = dat, kernel = "linear", range = list(cost = c(seq(1,1000))))
summary(tune.out)#interestingly the error rate for the cross validation k-fold method only differs by 1 percent. 

#(c)
set.seed(123)
x <- matrix(rnorm(200*2), ncol = 2)
x[1:100,] <- x[1:100,] + 1.5
x[101:200,] <- x[101:200,] - 1.5
plot(x[,1], x[,2])
y <- c(rep(3,100), rep(2, 100))
plot(x, col = y)
x1 <- x[,1]
x2 <- x[,2]
dat.test <- data.frame(x = x1,x = x2,y = as.factor(y))
lambdas <- seq(1, 40)
table.test <- matrix()
mse_test <- rep(NA, times = length(lambdas))
false.test <- rep(NA, times = length(lambdas))
for(i in length(lambdas)){
	svm.linear <- svm(y~., data = dat, kernel = "linear", cost = lambdas[i])
	table.test <- table(dat.test[,"y"], predict(svm.linear, newx = dat.test, type = "response"))
	mse_test[i] <- 1-((table.test[1,1]+table.test[2,2])/200) 
	false.test[i] <- table.test[1,2]+table.test[2,1]
}
#I can't seem to get this function to work. Will need to see what Asadoughi's solution is as a means to get an idea for what I should do. 

#Asadoughi's solution:
#(a)
#We randomly generate 1000 points and scatter them across line x = y with wide margin. We also create noisy points along the line 5x - xy - 50 = 0. These points make the classes barely separable and also shift the maximum margin classifier.

set.seed(3154)
#Class one 
x.one <- runif(500, 0, 90)
y.one <- runif(500, x.one + 10, 100)
x.one.noise <- runif(50, 20, 80)
y.one.noise <- 5/4 * (x.one.noise - 10) + 0.1

#Class zero 
x.zero <- runif(500, 10, 100)
y.zero <- runif(500, 0, x.zero - 10)
x.zero.noise <- runif(50, 20, 80)
y.zero.noise <- 5 / 4 *(x.zero.noise-10) -0.1

#combine all 
class.one <- seq(1, 550)
x <- c(x.one, x.one.noise, x.zero, x.zero.noise)
y <- c(y.one, y.one.noise, y.zero, y.zero.noise)
plot(x[class.one], y[class.one], col = "blue", pch = "+", ylim = c(0,100))
points(x[-class.one], y[-class.one], col = "red", pch =4)# I really have no idea how he came up with these data points. Sadly I still have a long way to go.
#The plot shows that classes are barely separable. The noisy points create a fictitious boundary 5x - 4y - 50 =0

#(b)
#We create a z variable according to classes.
library(e1071)
set.seed(555)
z <- rep(0, 1100)
z[class.one] <- 1
data <- data.frame(x=x, y=y, z=z)
tune.out <- tune(svm, as.factor(z)~., data = data, kernel ="linear", ranges = list(cost=c(0.01,0.1,1,5,10,100,1000,10000)))
summary(tune.out)
data.frame(cost=tune.out$performances$cost, misclass = tune.out$performances$error *1100)
#The table above shows train-misclassification error for all costs. A cost of 10000 seems to classify all points correctly. This also corresponds to a cross-validation error of 0. 

#(c)
#We now generate a random test-set of same size. This test-set satisfies the true decision boundary x = y
set.seed(111)
x.test <- runif(1000, 0, 100)
class.one <- sample(1000, 500)
y.test <- rep(NA, 1000)
#Set y > x for class.one 
for(i in class.one){
	y.test[i] <- runif(1, x.test[i], 100)
}
#Set y < x for class.zero 
for(i in setdiff(1:1000, class.one)){
	y.test[i] <- runif(1,0, x.test[i])
}
plot(x.test[class.one], y.test[class.one], col = "blue", pch = "+")
points(x.test[-class.one], y.test[-class.one], col = "red", pch = 4)
#We now make same predictions using all linear svms with all costs used in previous part.

set.seed(30012)
z.test <- rep(0, 1000)
z.test[class.one] <- 1
all.costs <- c(0.01, 0.1, 1,5,10,100,1000, 10000)
test.errors <- rep(NA, 8)
data.test <- data.frame(x=x.test, y = y.test, z = z.test)
for( i in 1:length(all.costs)){
	svm.fit <- svm(as.factor(z)~., data = data, kernel = "linear", cost = all.costs[i])
	svm.predict <- predict(svm.fit, data.test)
	test.errors[i] <- sum(svm.predict !=data.test$z)
}
data.frame(cost=all.costs, "test misclass"=test.errors)
#tt(cost) = 10 seems to be performing better on test data, making the least number of classification errors. This is much smaller than optimal value of 10000 for training data. 

#(d) 
#We again see an overfitting phenomenon for linear kernel. A large cost tries to fit correctly classify noisy-points and hence overfits the train data. A small cost, however, makes a few errors on the noisy test points and performs better on test data. 

#7.)
#(a) 
library(ISLR)
gas_01 <- ifelse(Auto$mpg > median(Auto$mpg), 1, 0)
Auto_new <- cbind(Auto, gas_01)
set.seed(1234)
linear_fit <- tune(svm, gas_01~., data = Auto_new,kernel = "linear", ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 30, 40, 50)))
linear_sum <- summary(linear_fit)
plot(linear_sum$performance$cost, linear_sum$performance$error, type = "l")
#According to the cross validation k-fold method the best model will be cost = 1. But knowing about the volitatility inherent in the tune() function will need to check this with other methods to know for sure.

set.seed(1234)
linear_fit2 <- tune(svm, as.factor(gas_01)~., data = Auto_new, kernel = "linear", ranges = list(cost = seq(1, 100, by = 5)))
linear_sum2 <- summary(linear_fit2)
plot(linear_sum2$performance$cost, linear_sum2$performance$error, type = "b")#Again according to this tuning model the best parameter for cost is still 1. I hope this is what Asadoughi obtained as well.

svm_linear <- svm(as.factor(gas_01)~., data = Auto_new, kernel = "linear", cost = 1)
plot(svm_linear, Auto_new, horsepower~weight)#this seems to be not a very good model since I can't really find the classification boundary in any of these graphical subsets. 
#Oh and on a side note, since the number of p (for predictor variables) are more than 2 you need to use the x1~x4 subset within the plot() function to graph the overall svm model.
plot(svm_linear, Auto_new, mpg~horsepower)#It seems that the author was right. The only variable subset that actually shows the classification boundary is mpg~horsepower.

#(c)
#Radial fit 
set.seed(1234)
radial_fit <- tune(svm, as.factor(gas_01)~., data = Auto_new, kernel = "radial", ranges = list(cost = seq(1, 100, by = 5), gamma = c(0.01, 0.1, 1, 2, 3, 4)))
radial_sum <- summary(radial_fit)
library(tidyverse)
radial_matrix <- data.frame(cost = radial_sum$performance$cost, error = radial_sum$performance$error, gamma = radial_sum$performance$gamma)
ggplot(data = radial_matrix, aes(x = cost, y = error, color = as.factor(gamma)))+geom_line()
#According to this tuning function the best parameters are cost = 36 and gamma = 0.01. With that said, a suspected as much since the plot(svm_linear, Auto_new, mpg~horsepower) function command illustrated a quadratic trend. 

svm_radial <- svm(as.factor(gas_01)~., data = Auto_new, kernel = "radial", gamma = 0.01, cost = 36)
plot(svm_radial, Auto_new, mpg~horsepower)#That's weird, for this model I don't see the classifier boundary. Will need to check through the other variables to make sure of this quirk. 

#polynomial kernel:
set.seed(1234)
poly_fit <- tune(svm, as.factor(gas_01)~.,data = Auto_new, kernel = "polynomial", ranges = list(cost = seq(1, 100, by = 5), degree = c(1,2,3,4)))
poly_sum <- summary(poly_fit)
poly_matrix <- data.frame(cost = poly_sum$performance$cost, error = poly_sum$performance$error, degree = poly_sum$performance$degree)
ggplot(data = poly_matrix, aes(x = cost, y = error, color = as.factor(degree)))+geom_line()
#According to this function call the best model parameters for polynomial SVM models are cost = 76 and degree = 1. 
svm_poly <- svm(as.factor(gas_01)~., data = Auto_new, degree = 1, cost = 76)

#(d)
plot(linear_sum$performance$cost, linear_sum$performance$error, type = "l")
linear_sum# The best model can only achieve a 0.0751 error rate.

ggplot(data = radial_matrix, aes(x = cost, y = error, color = as.factor(gamma)))+geom_line()
radial_sum# The best performance was recorded at 0.0152, which is weirdly better than the linear model.

ggplot(data = poly_matrix, aes(x = cost, y = error, color = as.factor(degree)))+geom_line()
poly_sum# The best performance was an error rate of 0.02295

#So in other words, the best model is the radial SVM. Which very much goes against all logic. 
par(mfrow = c(2,2))
plot(svm_linear, Auto_new, mpg~horsepower)
plot(svm_radial, Auto_new, mpg~horsepower)
plot(svm_poly, Auto_new, mpg~horsepower)

#Asadoughi's solution for (d)
plotpairs <- function(fit){
	for(name in names(Auto_new)[!(names(Auto_new) %in% c("mpg", "gas_01", "name"))]){
		plot(fit, Auto_new, as.formula(paste("mpg~", name, sep = "")))
	}
}
plotpairs(svm_linear)
plotpairs(svm_radial)
plotpairs(svm_poly)

#8.) 
library(ISLR)
dim(OJ)
colnames(OJ)

#(a)
train <- sample(1:nrow(OJ), 800)
test <- -train

#(b-c)
set.seed(1234)
svm.linear <- svm(as.factor(Purchase) ~., cost = 0.01, data = OJ[train,], kernel = "linear")
svm.pred.train <- predict(svm.linear, OJ[train,])
table(svm.pred.train, OJ[train, "Purchase"])
1-((429+241)/800)# the training error rate is 0.1625. 

set.seed(1234)
svm.pred.test <- predict(svm.linear, OJ[test,])
table(svm.pred.test, OJ[test, "Purchase"])
1-((146+83)/nrow(OJ[-train,]))# The test error rate is 0.152 which is interestingly less than the training error rate. 

#(d)
set.seed(1234)
linear.tune <- tune(svm,Purchase ~., data = OJ[train,], kernel = "linear", ranges = list(cost = c(0.01, 0.05, 0.1, 0.5, 1, 2, 3,4,5,6,7,8,9,10))) 
summary(linear.tune)#The best model has a cost parameter of 0.01. This values seems way too low. Will need to look into this. 

#I guess I wasn't too wrong, Asaboughi obtained a optimal cost value of 0.3162. Will need to look into this.  
set.seed(1554)
linear.tune2 <- tune(svm,Purchase ~., data = OJ[train,], kernel = "linear", ranges = list(cost = 10^seq(-2, 1, by =0.25))) 
summary(linear.tune2)# For this command my computer says that the optimal cost is 0.01778, but as you look at the results costs 1.778 and 5.623 have the same error rates. 

#(e)
set.seed(1234)
# Cost 0.01778
svm.linear1 <- svm(Purchase~., data = OJ[train,], kernel = "linear", cost = 0.01778)
svm.pred1.train <- predict(svm.linear1, OJ[train,])
svm.table1.train <- table(svm.pred1, OJ[train, "Purchase"])
1 - ((svm.table1.train[1,1] + svm.table1.train[2,2])/nrow(OJ[train,]))# the training error rate is 0.165 (which is only about 0.002 different from the initial value)

svm.pred1.test <- predict(svm.linear1, OJ[-train,])
svm.table1.test <- table(svm.pred1.test, OJ[-train, "Purchase"])
1 - ((svm.table1.test[1,1] + svm.table1.test[2,2])/nrow(OJ[-train,]))# the test error rate is 0.1778 (which is about 0.02 worse than the initial value)

#(f)
set.seed(1234)
radial.tune <- tune(svm, Purchase ~., data = OJ[train,], kernel = "radial", ranges = list(cost = 10^seq(-2, 1, by =0.25)))
summary(radial.tune)#The best model has a cost parameter of 0.1778.
radial.svm <- svm(Purchase ~., data = OJ[train,], cost = 0.1778, gamma = 1, kernel = "radial")

radial.pred.train <- predict(radial.svm, OJ[train,])
table(radial.pred.train, OJ[train, "Purchase"])
1-((478+134)/nrow(OJ[train,]))# The training error rate is 0.235. 

radial.pred.test <- predict(radial.svm, OJ[test,])
table(radial.pred.test, OJ[-train, "Purchase"])
1-((162 + 33)/nrow(OJ[test,]))# The test error rate for the radial model is 0.2778.

#(g)
set.seed(1234)
poly.tune <- tune(svm, Purchase ~., data = OJ[train,], kernel = "polynomial", ranges = list(cost = 10^seq(-2, 1, by =0.25), degrees = 2))
summary(poly.tune)# The best performing model has a cost parameter of 3.1623

svm.poly <- svm(Purchase ~., data = OJ[train,], kernel = "polynomial", cost = 3.1623, degrees = 2)
poly.pred.train <- predict(svm.poly, OJ[train,])
table(poly.pred.train, OJ[train,"Purchase"])
1-((447+234)/nrow(OJ[train,]))# The training error rate for the polynomial SVM is 0.1488.

poly.pred.test <- predict(svm.poly, OJ[test,])
table(poly.pred.test, OJ[test, "Purchase"])
1-((146+74)/nrow(OJ[test,]))# The test error rate is 0.185, which is an improvement over the radial modle but fails to beat the linear SVM model. 

#(h) 
#In other words, the best model for this dataset is the linear model. 

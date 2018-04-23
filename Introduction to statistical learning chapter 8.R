### Chapter 8 Tree-based Methods:
#These involve stratifying or segmenting the predictor space into a number of simple regions. In orer to make a prediction for a given observation, we typically use the mean or the mode of the training observations in the region to which it belongs. Since the set of splitting rules used to segment the predictor space can be summarized in a tree, these types of approaches are known as decision tree methods. 

##8.1 The basics of Decision trees:

##8.1.1 Regression trees:
##Predicting Baseball Players' Salaries Using Regression trees
#In order to motivate regression trees, we begin with a simple example.

##Predicting Baseball Players' Salaries Using Regression trees:
names(Hitters)
Hitters.fit <- lm(log(Salary) ~ Years + Hits, data = Hitters)
par(mfrow = c(2,2))
plot(Hitters.fit)
summary(Hitters.fit)

#We use the Hitters data set to predict a baseball player's Salary based on Years and Hits. We fits remove observations that are mssing Salary values, and log transform Salary so that its distribution has ore of a typical bell-shape. 
#Figure 8.1 shows a regression tree fit to this data. It consists of a series of splitting rules, starting at the top of the tree. The top split assigns observatiosn having Years<4.5 to the left branch. The predicted salary for these players is given by the mean response value for the players in the data set with Years<4.5. For such players, the mean log salary is 5.107, and so we make a prediction of e^5.105 thousands of dollars for these players. Players with years>=4.5 are assigned to the right branch, and then that group is further subdivided by Hits. Overall, the tree stratifies or segments the players into three regions of prediction space.

#(Important terminaology) The points along the tree where the predictor space is split are referred to as internal nodes.. In figure 8.1, the two internal segments of the trees that connect the nodes as branches.

#We might interpret the regression tree displayed in Figure 8.1 as follows Years is the most important factor in determining Salary, and players with less experience earn lower salaries than more experienced players. Given that a player is less experienced, the number of hits that he made in the previous year seems to play little role in his salary. But among players who have been in the major leagues for five or more yeats, the number of hits made in the previous year does affect salary, and players who made more hits last year tend to have higher salaries. 

##prediction via stratification of the feature space:
#We now discuss the process of building a regression tree. Roughly speaking there are two steps:
	#1. We divide the predictor space --- that is, the set of possible values of X_1, X_2, ..., X_p --- into J distinct and non-overlapping regions, R_1, R_2, ..., R_J. 
	#2. For every observation that falls into the region R_j, we make the same prediction, which is simply the mean of the response values for the training observations in R_j.
	
#Step 1 elaboration, How do we construct the regions R_1, ..., R_j? In theory, the regions could have any shape. However, we choose to divide the predictor space into high dimensional rectangles, or boxes. The goal is to find boxes R_1, ..., R_J that minimize the RSS given by (look at the equation on page 306) where y_hat_r*j is the mean response for the train observations within the jth box. We take a top down, greedy approach that is known as recursive binary splitting. The approach is top down because it begins at the top of the tree (at which point all observations belong to a single region) and then successively splits the predictor space; each split is indicated via two new branches further down on the tree. It is greedy because at each step of the tree building process, the best split is made at theat particular step, rather than looking ahead and picking a split that will lead to a better tree in some future step. 

#In order to preform recursive binary splitting, we first select the predictor X_j and the cutpoint s such that splitting the predictor space into the regions {X|X_j < s} and {X|X_j >= s} leads to the greatest possible reduction of RSS. (The notation {x|X_j < s} means the region of predictor space in which X_j takes on a value less than s.) That is, we consider all predictors X_1, ..., X_p, and all possible values of the cutpoint s for each of the predictors, and then choose the predictor and cutpoint such that the resulting tree has the lowest RSS. To see the equationthat is used to find the optimum cut points check page 307. Finding the values of j and s that minimize (8.3) can be done quite quickly, expecially when the number of features p is not too large.

#Next, we repeat the process, looking for the best predictor and best cutpoint in order to split the data further so as to minimize the RSS within each of the resulting regions. However, this time, instead of splitting the entire predictor space, we split on of the two previously identified regions. Again, we look to split on eof these three regions further, so as to minimize the RSS. The process continues until a stopping criterion is reached;for instance, we may continue until no region contains more than five observations.

#Once the regions R_1, ..., R_j have been created, we predict the response for a given test observation using the mean of the training observations in the region in which that test observation belongs. 

##Tree pruning:
#One possible alternative to the process described above is to build the tree only so long as the decrease in the RSS due to each split exceeds some (high) threshold. This strategy will result in smaller trees, but is too short-sighted since a seemingly worthless split early on in the tree might be followed by a very good split --- that is, a split that leads to a large reduction in RSS later on. 

#Therefore, a better strategy is to grow a very large tree T_0, and then prune it back in order to obtain a subtree. How do we dteremine the best way to prune the tree? Intuitively, our goal is to select a subtree that leads to the lowest test error rate. Given a subtree, we can estimate its test error using cross-validation or the validation set approach. However, estimating the cross-validation error for every possible subtree might be too cumbersome to carry out. Instead, we need a way to select a small set of subtrees for consideration. 

#Cost complexity pruning--- also known as weakest link pruning -- gives us a way to do just this. Rather than considering every possible subtree, we consider a sequence of trees indexed by a nonnegative tuning parameter alpha.

#(important algorithm) Algorithm 8.1 Building a regression tree:
	#1. Use recursive binary splitting to grow a large tree on the training data, stopping only when each terminal node has fewer than some minimum number of observations.
	#2. Apply cost complexity pruning to the large tree in order to obtain a sequence of best subtrees, as a function of alpha.
	#3. Use k-fold cross validation to choose alpha. That is, divide the training observations into K folds. For each K = 1, ..., K:
		#a. Repeat steps 1 and 2 on all but the kth fold of the training data.
		#b. Evaluate the mean squared prediction error on the data in the left out kth fold, as a function of alpha.
		
	#4. Return the subtree from step 2 that corresponds to the chosen value of alpha.
	
#The tuning parameter alpha controls a trade-off between the subtree's complexity and its fit to the training data. When alpha = 0, then the subtree T will simply equal T_0, because then (8.4) just measures the training error. However, as alpha increases, there is a price to pay for having a tree with many terminal nodes, and so the quantity will tend to be minimized for a smaller subtree. Equation 8.4 is reminiscent of the lasso from chapter 6, in which a similar formulation was used in order to control the complexity of a linear model (In other words the shrinkage variable (lambda) if I remember correctly). 

#It turns out that as we increase alpha from zero in (8.4), branches get pruned from the tree in a nested and predictable fashion, so obtaining the whole sequence of subtrees as a function of alpha is easy. We can select a value of alpha using validation set or cross-validation. We then return to the full data set and obtain the subtree corresponding to alpha. 

##8.1.2 Classification trees:
#A classification tree is very similar to a regression tree, except that it is used to predict a qualitative response rather than a quantitative one. Recall that for a regression tree, the predicted response for an observation is given by the mean response of the training observations that belong to the same terminal node. In contrast, for a classification tree, we predict that each observations in the region to which it belongs. In interpreting the results of a classification tree, we are often interested not only in the class prediction corresponding to a particular terminal node region, but also in the class proportions among the training observations that fall into that region. 

#It's important to remember that RSS cannot be used for categorical response variables. The natural alternative to RSS is the classification error rate. Since we plan to assign an observation in a given region to the most commonly occurring class of training observations in that region, the classification error rate is simply the fraction of the training observations in that region that do not belong to the most common class.

#to see the equations for classification tree classification error rate (make sure to consult page 312).

#However, it turns out that classification error is not sufficiently sensitive for tree-growing, and in practice two other measures ar preferable:
	#The Gini index. Which is a measure of total variance across the K classes. It is not hard to see that the Gini index takes on a small value if all of the p_hatmk's are clase to zero or one. For this reason the Gini index is referred to as a measure of node purity -- a small value indicates that a node contains predominantly observations from a single class. 
	#An alternative is cross-entropy. One can show that the cross-entropy will take on a value near zero if the p_hatmk's are near zero or near one. Therefore, like the Gini index, the cross-entropy will take on a small value if the mth node is pure.
	
#When building a classification tree, either the Gini index or the cross entropy are typically used to evaluate the quality of a particular split, since these two approaches are more sensitive to node purity than is the classification error rate. Any of these three approaches might be used when pruning the tree, but the classification error rate is preferable if prediction accuracy of the final pruned tree is the goal. 

#It's important to state that regression trees and classification trees can be constructed with qualitative variables as well as quantitative variables. The only difference between the two is that the former will assign a particular category to a termination node. 

#(importance of node purity) Figure 8.6 has a surprising characteristic: some of the splits yield two terminal nodes that have the same predicted value. For instance, consider the split RestECG<1 (this classification tree was created using the Heart data set) near the bottom right of the unpruned tree. Regardless of the value of RestECG, a response value of Yes is predicted for those observations. Why, then, is the split performed at all? The split is performed because it leads to increased node purity. Why is node purity important? Suppose that we have a test observation that belongs to the region given by that right hand leaf. Then we can be pretty certainthat its response value is Yes. In contrast, it a test observation belongs to the region given by the left-hand leaf, then its response value is probably Yes, but we are must less certain. Even though the split RestECF<1 does not reduce the classification error, it improves the Gini index and the cross-entropy, which are more sensitive to node purity.

##8.1.3 Trees Versus Linear models:
#If the relationship between the features and the response is well approximated by a linear model, then an approach such as linear regression will likly work well, and will outperform a method such as a regression tree that does not exploit this linear structure. If instead there is a highly non-linear and complex relationship between the features and the response as indicated by model (8.9), then decision trees may outperform classical approaches. 

##8.1.4. Advantages and Disadvantages of Trees:
#Some people believe that decision trees more closely mirror human decision making than do the regression and classification approaches seen in previous chapters.
#Trees can easily handle qualitative predictors without the need to create dummy variables.
#Unfortunately, trees generally do not have the same level of predictive accuracy as some of the other regression and classification approaches seen in this book. 

##8.3 Baggin, random forests, boosting:
##8.3.1 Bagging:
#We we here that the bootstrap can be used in a completely different context, in order to improve statistical learning methods such as decision trees.

#The decision trees discussed in section 8.1 suffer from high variance (which is characteristic of non parametric statistical models). This means that if we split the training data into two parts at random, and fit a decision tree to both halves, the results that we get could be quite different. In contrast, a procedure with low variance will yield similar results if applied repeatedly to distinct data sets; linear regression tends to have low variance, if the ratio of n to p is moderately large. 

#Recall that given a set of n independent observations Z_1, ..., Z_n, each with variance sigma^2, the variance of the mean Z_bar of the observations is given by sigma^2/n. In other words, averaging a set of observations reduces variance. Hence a natural way to reduce the variance and hence increase the prediction accuracy of a statistical learning method is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions. In other words, we could calculate f_hat(x), f_hat^2, ..., f_hat^B(x) using B separate training sets, and average them in order to obtain a single low-variance statistical learning model. (look at page 316 to see the formula used for this bootstrap technique).
#this is not practical because we generally do not have access to multiple training sets. Instead, we can bootstrap, by taking repeated samples from the single training data seet. In this approach we generate B different bootstrapped data sets. We then train our method on bth bootstrapped training set in order to get f_hat^8b(x), and finally average all the predictions, to obtain (lok at page 317 to see the equation for the baggin method).

#This is called bagging.
#To apply baggin to regression trees, we simply construct B regression trees using B bootstrapped training sets, and average the resulting predictions. These trees are grown deep and are not pruned. Hence each individual tree has high variance, but low bias. Averaging these B trees reduces the variance, Bagging has been demonstrated to give impressive improvements in accuracy by combining together hundreds or even thousands of trees into a single procedure. 

#How can baggin be extended to a classification problem where Y is qualitative? In that situation, there are a few possible approaches, but the simplest is as follows. for a given test observation, we can record the class predicted by each of the B trees, and take a majority vote: the overall prediction is the most commonly occurring class among the B predictions. 

#(important note) The number of trees of B will not lead to overfitting. In practicewe use a value of B sufficiently large that the error has settled down. Using B = 100 is sufficient to achieve good performance in the example with the Heart data set. 

##Out-of-bag error estimation:
#It turns out that there is a very straightforward way to estimate the test error of a bagged model, without the need to perform cross-validation or the validation set approach. Recall that the key to baggin is that trees are repeatedly fit to bootstrapped subsets of the observations. One can show that on average, each baggin tree makes use of around two-thirds of the observations. The remaining one-third of the observations not used to fit a given bagged tree are referred to as the out-of-bag observations. We can predict the response for the ith observation using each of the trees in which that observation was OOB. This will yield around B/3 predictions for the ith observation. In order to obtain a single predictions for the ith observation, we can average these predicted responses (for regression trees) or can take a majority vote (for classification trees). An OOB prediction can be obtained in this way for each of the n observations, from which the overall OOB MSE (for regression) or classification error (for classification problems) can be computed. 

##Variable Importance Measures:
#Baggin improves prediction accuracy at the expense of interpretability. 
#Although the collection of bagged trees is much more difficult to interpret than a single tree, one can obtain an overall summary of the importance of each predictor using the RSS (for baggin regression trees) or the Gini index (for baggin classification trees). In the case of baggin regression trees, we can record the total amount that the RSS is decreased due to splits over a given predictor, averaged over all B trees. A large value indicates an important predictor. Similarly, in the context of baggin classification trees, we can add up the total amount that the Gini index is decreased by splits over a given predictor, average over all B trees. 

##8.2.2 Random Forests:
#Random forests provide an improvement over baggin trees by way of a small tweak that decorrelates the trees. As in baggin, we build a number of decision trees on bootstrapped training samples. But when building these decision trees, each time a plit in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is allowed to use only one of those m predictors. A fresh sample of m predictors is taken at each split, and typically we choose m ~ sqrt(p) -- that is, the number of predictors considered at each split is approximately equal to the square root of the total number of predictors.
#In other words, in building a random forest, at each split in the tree the algorithm is not even allowed to consider a majority of the available predictors. 

#(important constraint with bagging trees) All of the bagged trees will look quite similar to each other. Hence the predictions from the bagged trees will be highly correlated. Unfortuatel, averaging many highly correlated quantities does not lead to as large a reduction in variance as averaging many uncorrelated quantities. In particular, this means that bagging will not lead to a substantial reduction in variance over a single tree in this setting. 

#Random forests overcome this problem by forcing each split to consider only a subset of the predictors. Therefore, on average (p-m)/p of the splits will not even consider the strong predictor, and so other predictors will have more of a chance. We can think of this process as decorrelating and hence more reliable. The main difference between baggin and random forests is the choice of predictor subset size m . For instance, it a random forest is built using m = p, then this amounts simply to bagging.

#(important note) As with bagging, random forests will not overfit if we increase B, so in practice we use a value of B sufficiently large for the error rate to have settled down. 

##8.2.3 Boosting:
#Boosting works in very much the same way as bagging, except that the trees are grown sequentially: each tree is grown using information from previously grown trees. Boosting does not involve bootstrap sampling; instead each tree is fit on a modified version of the original data set. (to see the steps used to create a boosted regression tree see algorithm 8.2 located on page 322).

#Boosting has three tuning parameters:
	#1. The number of trees B. Unlike bagging and random forests, boosting can overfit if B is too large, although this overfitting tends to occur slowly if at all. We use cross-validation to select B.
	#2. The shrinkage parameter lambda, a small positive number. This controls the rate at which boosting learns. typical values are 0.01 or 0.001, and the right choice can depend on the problem. Very small lambda can require using a very large value of B in order to achieve good performance. 
	#3. The number d of splits in each tree, which controls the complexity of the boosted ensemble. Often d = 1 works well, in which case each tree is a stump, consisting of a single split. In this case, the boosted ensemble is fitting an additive model since each term involves only a single variable. More generally do is the interaction depth, and controls the interaction order of the boosted model, since d splits can involve at most d variables. 
	
#(important closing thought) In boosting, because the growth of a particular tree takes into account the other trees that have already been grown, smaller trees are typically sufficient. Using smaller trees can aid in interpretability as well; for instance, using stumps leads to an additive model. 


## 8.3 Lab: Decision trees:
##8.3.1 Fitting Classification trees:
library(tree)
#We first use classification trees to analyze the Carseats data set. In these data, Sales is a continuous variable, and so we begin by recoding it as the binary variable. We use the ifelse() function to create a variable, called High, which takes on a value of Yes if the Sales variable exceeds 8, and takes on a value of No otherwise.
library(ISLR)
attach(Carseats)
High <- ifelse(Sales <=8, "No", "Yes")
Carseats <- data.frame(Carseats, High)
head(Carseats)

#We now use the tree() function to fit a classification tree in order to predict High using all variables but Sales. The syntax of the tree() function is quite similar to that of the lm() function.
tree.carseats <- tree(High ~. -Sales, Carseats)
summary(tree.carseats)
#The residual mean deviance reported is simply the deviance divided by n - |T_0|, which in this case is 400 - 27 = 373. Look at page 325 to see the deviance formula for this problem. 

#One of the most attractive properties of trees is that they can be graphically displayed. We use the plot() function to display the tree structure, and the text() function to display the node labels. The arguement pretty = 0 instructs R to include the category names for any qualitative predictors, rather than simply displaying a letter for each category.
plot(tree.carseats)
text(tree.carseats, pretty = 0)

#If we just type the name of the tree object, R prints output corresponding to each branch of the tree. R displays the split criterion (e.g. Price<92.5), the number of observations in that branch, the deviance, the overall prediction for the branch (Yes or No), and the fraction of observations in that branch that take on values of Yes and No. Branches that lead to terminal nodes are indicated using asterisks.
tree.carseats

#In order to properly evaluate the performance of a classification tree on these data, we must estimate the test error rather than simply computing the training error. For this example the author uses the set cross validation method (split the data set into a training set and a test set). The predict() function can be used for this purpose. Since the tree is calculating a classification response variable the argument type = "class" instructs R to return the actual class prediction. 
set.seed(2)
train <- sample(1:nrow(Carseats), 200)
Carseats.test <- Carseats[-train,]
High.test <- High[-train]
tree.carseats <- tree(High ~. -Sales, Carseats, subset = train)
tree.pred <- predict(tree.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
1-((86+57)/200)#This model has an error rate of 0.285. Which is very low considering how this is just a simple classification tree without any bagging or boosting methods to decrease volatility.

#Next, we consider whether pruning the tree migh lead to improved results. The function cv.tree() performs cross-validation in order to determine the optimal level of tree complexity;cost complexity pruning is used in order to select a sequence of trees for consideration. We use the argument FUN = prune.misclass in order to indicate that we want the classification error rate to guide the cross validation and pruning process (by default the cv.tree() function uses the deviance to guide the process). The cv.tree() function reports the number of terminal nodes of each tree considered (size) as well as the corresponding error rate and the value of the cost-complexity parameter used (k, which corresponds to alpha in equation 8.4).
set.seed(3)
cv.carseats <- cv.tree(tree.carseats, FUN = prune.misclass)
names(cv.carseats)
cv.carseats
par(mfrow = c(1,2))
plot(cv.carseats$size, cv.carseats$dev, type ="b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

#Note that, despite the name, dev corresponds to the cross-validation error rate in this instance. The tree with 9 terminal nodes results in the lowest cross-validation error rate, with 50 cross-validation errors. 

#We now apply the prune.misclass() function in order to prune the tree to obtain the nine-node tree. 
prune.carseats <- prune.misclass(tree.carseats, best = 9)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
#Checking on the predictive accuracy of this model:
tree.pred <- predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
1-((94 + 60)/200)# The error rate is calculated at 0.23 which is only a 0.055 reduction from the previous error rate with the unpruned classification tree. The most advantageous thing about this newly pruned model is the tree is more interpretable than the original. 

#If we increase the value of best, we obtain a larger pruned tree with lower classification accuracy:
prune.carseats <- prune.misclass(tree.carseats, best = 15)
plot(prune.carseats)
text(prune.carseats)
tree.pred <- predict(prune.carseats, Carseats.test, type = "class")
table(tree.pred, High.test)
1-((86 + 62)/200)# 0.26 which is 0.03 more than the previous pruned model. 

##8.3.2 Fitting Regression trees:
#here we fit a regression tree to the Boston data set. First, we create a training set, and fit the tree to the training data. 
library(MASS)
set.seed(2)
train <- sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston <- tree(medv ~., Boston, subset = train)
summary(tree.boston)
#In the context of the regression tree, the deviance is simply the sum of squared errors for the tree. 
plot(tree.boston)
text(tree.boston, pretty = 0)

#the variable lstat measures the percentage of indivduals with lower socioeconomic status. the tree indicates that lower value of lstat correspond to more expensive houses. The tree predicts a median house prive of 46,400 for larger homes in suburbs in which residents have high socioeconomic status (rm >= 7.437 and lstat<9.715).
#Now we use the cv.tree() function to see whether pruning the tree will improve performance.
cv.boston <- cv.tree(tree.boston)#Interesting in place of using the cv error rate to dictate the pruning process the author is allowing the function to use the default deviance pruning process. This is very curious. 
plot(cv.boston$size, cv.boston$dev, type = "b")# so it seems that nine nodes is actually sufficient.

#this is only to show how to prune regression trees:
prune.boston <- prune.tree(tree.boston,best = 5)
plot(prune.boston)
text(prune.boston, pretty = 0)

#In keeping with the cross-validation results, we use the unpruned tree to make the predictions on the test set.
yhat <- predict(tree.boston, newdata = Boston[-train,])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0,1)
mean((yhat-boston.test)^2)# The MSE is 23.93114.
sqrt(mean((yhat-boston.test)^2))# 4.891947
#In other words, the test set MSE associated with the regression tree is 23.93114. The square root of the MSE is therefore around 4.891947, indicating that this model leads to test predictions that are within around 4,891 of the tree median home value for the suburb.

##8.3.3 Bagging and random Forests:
#Recall that baggin is simply a special case of a random forest with m = p. Therefore, the randomForest() function can be used to perform both random forests and bagging. We perform bagging as follows:
library(randomForest)
set.seed(1)
dim(Boston)
bag.boston <- randomForest(medv ~., data = Boston, subset = train, mtry = 13, importance = TRUE)# Now I get it the author set the mtry argument to the same number of predictor variables in the model (13) as a means to create a psuedo baggin model. mtry is actually the random forest M parameter (much like lambda for the lasso).

yhat.bag <- predict(bag.boston, newdata = Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)# The MSE was calculated at 10.9427 using the baggin method which is a more than half reduction in relation to the regression tree. 

#We could change the number of trees grown by randomForest() using the ntree argument:
bag.boston <- randomForest(medv ~., data = Boston, subset = train, mtry = 13, ntree = 25)
yhat.bag <- predict(bag.boston, newdata = Boston[-train,])
mean((yhat.bag-boston.test)^2)# The MSE was calculated at 10.60464.

#Growing a random forest proceeds in exactly the same way, except that we use a smaller value of the mtry argument. By default, randomForest() uses p/3 variables when building a random forest of regression trees, and sqrt(p) variables when building a random forest of classification trees. Here we use mtry = 6.
set.seed(1)
rf.boston <- randomForest(medv ~., data = Boston, subset =train, mtry = 6, importance = TRUE)
yhat.rf <- predict(rf.boston, newdata = Boston[-train,])
mean((yhat.rf-boston.test)^2)# the MSE for random Forests was calculated at only 11.3461 which isn't an improvement of the model that used bagging.

#Using the importance() function, we can view the importance of each variable.
importance(rf.boston)
#Two measures of variable importance are reported. The former is based upon the mean decrease of accuracy in predictions on the out of bag samples when a given variable is excluded from the model. The latter is a measure of the total decrease in node impurity that results from splits over that variable. averaged over all trees. In the case of regression trees, the node impurity is measured by the training RSS, and for classification trees by the deviance. Plots of these importance measures can be produced using the varImpPlot() function.
varImpPlot(rf.boston)

##8.3.4 Boosting:
#Here we use the gbm function within the gbm package. The distribution argument will be set to "gaussian" since this is a regression problem;if it were a binary classification problem, we would use distribution = "bernoulli". The argument n.trees = 5000 indicates that we want 5000 trees, and the option interaction.depth = 4 limits the depth of each tree. 
library(gbm)
set.seed(1)
boost.boston <- gbm(medv ~., data = Boston[train,], distribution ="gaussian", n.trees = 5000, interaction.depth = 4)

#The summary() function produces a relative influence plot and also outputs the relative influence statistics.
summary(boost.boston)

#We see that lstat and rm are by far the most important variables. We can also produce partial dependence plots for these two variables. These plots illustrate the marginal effect of the selected variables on the response after integrating out the other variables. In this case, as we might expect, median house prices are increasing with rm and decreasing with lstat.
par(mfrow = c(1,2))
plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")

#We now use the boosted model to predict medv on the test set.
yhat.boost <- predict(boost.boston, newdata = Boston[-train,], n.trees = 5000)
mean((yhat.boost - boston.test)^2)#13.24306 Interesting this MSE is worse than random forests and Bagging.

#If we want to, we can perform boosting with a different value of the shrinkage parameter lambda. The default lambda value is 0.001.
boost.boston <- gbm(medv~., data = Boston[train,], distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = FALSE)
yhat.boost <- predict(boost.boston, newdata = Boston[-train,], n.trees = 5000)
mean((yhat.boost-boston.test)^2)#the MSE was calculated at 15.43536 which is worse than the other methods except for the simple regression tree. 

## 8.4 Exercises:
##conceptual:
#1.) Asadoughi's solution:
par(xpd = NA)
plot(NA, NA, xlim = c(0,100), ylim = c(0,100), xlab = "X", ylab = "Y")
lines(x = c(40,40), y = c(0,100))
text(x = 40, y = 108, labels = c("t1"), col = "red")
lines(x=c(0,40), y = c(75,75))
text(x=-8, y = 75, labels = c("t2"), col = "red")
lines(x =c(75,75), y = c(0,100))
text(x = 20, y = 80, labels =c("t4"), col = "red")
lines(x = c(75,100), y = c(25,25))
text(x = 70, y = 25, labels = c("t5"), col = "red")
text(x=(40+75)/2, y = 50, labels = c("R1"))
text(x=20, y = (100+75)/2, labels = c("R2"))
text(x=(75+100)/2, y = (100+25)/2, labels = c("R3"))
text(x=(75+100)/2, y = 25/2, labels = c("R4"))
text(x= 30, y = 75/2, labels = c("R5"))
text(x = 10, y = 75/2, labels = c("R6"))
lines(x = c(20,20), y = c(0,75))
text(x = 20, y = 80, labels = c("t4"), col = "red")

#Man really asadoughi is truly a genius to be about to create this graphic representation of a decision tree through the use of base R plot() functions. Will need to look more into this.

#2.) I can't really prove mathematically that using trees which are only of depth one in the implementation of bagging trees or even boosting creates a psuedo-additive model (which seems to be much like that of the General Additive models seen in chapter 7), but logically this rational very much checks out. The reason why I say this is that setting the random forest parameter m to p will only create a bagging model the same can be possible with one depth boosting models. Will need to look into the mathematics between the boosting model formula and the General Additive Model from the end of chapter 7.

#Asadoughi's solution:
#Based on Algorithm 8.2, the first stump will consist of a split on a single variable. By induction, the residuals of that first fit will result in a second stump fit to another distinct, single variable 
#To see the full answer look at Asadoughi's github repo. the mathematics is too advanced for me to even comprehend.

#3.) Cross entropy, Gini statistic, and classification error rate question using the plot() function with R. Will again need to lean on Asadoughi.
p <- seq(0, 1, by = 0.01)
Gini <- p*(1 - p)* 2
entropy <- -(p * log(p) + (1-p)*log(1-p))
class.err <- 1 - pmax(p, 1-p)
matplot(p, cbind(Gini, entropy, class.err), col = c("red","green", "blue"), type = "l")
legend("topright",legend = c("Gini","entropy","class error"), col = c("red","green","blue"), lty = c(1:3))

#4.) This question relates to the plots in Figure 8.12.
#(a) Look at Asaboughi's github repo for the answer to this question.
#(b) 
par(xpd = NA)
plot(NA, NA, type = "n", xlim = c(-2,2), ylim = c(-3,3), xlab = "X1", ylab = "X2")
lines(x=c(-2,2), y = c(1,1))
lines(x=c(1,1), y = c(-3,1))
text(x=(-2+1)/2, y = -1, labels = c(-1.80))
text(x=1.5, y =-1, labels =c(0.63))
lines(x = c(-2,2), y = c(2,2))
text(x= 0, y=2.5, labels = c(2.49))
lines(x=c(0,0), y = c(1,2))
text(x=-1, y=1.5, labels = c(-1.06))
text(x=1, y=1.5, labels = c(0.21))

#5.)
class <- c(0.1,0.15, 0.2,0.2,0.55,0.6,0.6,0.65,0.7,0.75)
#Majority approach
sum(class>=0.5) > sum(class<0.5)# the number of red predictions is greater than the number of green predictions based on a 50 percent threshold, thus Red.

#Average approach:
mean(class)#The average of the probabilities is less than the 50 percent threshold, thus Green.

#6.)
#Sad to say but this explaination on the inner workings of a regression and classification tree will take too much time for me to answer correctly (or rather I'm just too lazy to reiterate what was said at the beginning of the chapter).

##Applied:
#7.)
##Warm up exercises:
set.seed(1)
head(Boston)
bag.boston25 <- randomForest(medv ~., data = Boston, ntree = 25, mtry = 6, importance = TRUE)
yhat.bag25 <- predict(bag.boston25, newdata = Boston)
plot(yhat.bag25, Boston$medv)
abline(0,1)
mean((yhat.bag25-Boston$medv)^2)
#With this method there is a MSE value of 1.974 
summary(bag.boston25)
bag.boston500 <- randomForest(medv ~., data = Boston, ntree = 500, mtry = 6, importance = TRUE)
yhat.bag500 <- predict(bag.boston500, newdata = Boston)
mean((yhat.bag500-Boston$medv)^2)
#The MSE for this method is 1.7637. Which means that calculating our randomforest with 200 ntrees creates a more accurate model. The problem with these values though is that these MSE values are only the training MSEs and not the testing MSEs. 

## Actual answer:
train <- sample(1:nrow(Boston), nrow(Boston)/2)
test <- -train
MSE <- c()
tree <- 1:500
m_val <- seq(0, 7, by = 2)
for(i in 1:length(tree)){
	for(j in 1:length(m_val)){
		bag.boston <- randomForest(medv ~., data = Boston[train,], ntree = i, mtry = j, importance = TRUE)
		yhat <- predict(bag.boston, newdata = Boston[test,])
		MSE[i] <- mean((yhat-Boston$medv[test])^2)
	}
}

#mtry argument set to 0 (equivalent to Bagging).
MSE1 <- c()
for(i in 1:length(tree)){
	bag.boston <- randomForest(medv ~., data = Boston[train,], ntree = i, mtry = 0, importance = TRUE)
	yhat <- predict(bag.boston, newdata = Boston[test,])
	MSE1[i] <- mean((yhat-Boston$medv[test])^2)
}

#mtry argument set to 2 (since the number of predictors is 13 (I desided to go up by factors of 2))
MSE2 <- c()
for(i in 1:length(tree)){
	bag.boston <- randomForest(medv~., data = Boston[train,], ntree = i, mtry = 2, importance = TRUE)
	yhat <- predict(bag.boston, newdata = Boston[test,])
	MSE2[i] <- mean((yhat-Boston$medv[test])^2)
}

#mtry argument set to 4
MSE3 <- c()
for(i in 1:length(tree)){
	bag.boston <- randomForest(medv~., data = Boston[train,], ntree = i, mtry = 2, importance = TRUE)
	yhat <- predict(bag.boston, newdata = Boston[test,])
	MSE3[i] <- mean((yhat-Boston$medv[test])^2)
}

#mtry argument set to 6
MSE4 <- c()
for(i in 1:length(tree)){
	bag.boston <- randomForest(medv~., data = Boston[train,], ntree = i, mtry = 2, importance = TRUE)
	yhat <- predict(bag.boston, newdata = Boston[test,])
	MSE4[i] <- mean((yhat-Boston$medv[test])^2)
}

#Bagging method (M = p)
MSE5 <- c()
for(i in 1:length(tree)){
	bag.boston <- randomForest(medv~., data = Boston[train,], ntree = i, mtry = 2, importance = TRUE)
	yhat <- predict(bag.boston, newdata = Boston[test,])
	MSE5[i] <- mean((yhat-Boston$medv[test])^2)
}

plot(x = 1:500, y = MSE1, col = "black", lty = 1, type = "l", ylab = c("MSE"), xlab = c("Number of Trees"), ylim = c(10,30))
lines(x = 1:500, y = MSE2, col = "red", lty = 1)
lines(x = 1:500, y = MSE3, col = "blue", lty = 1)
lines(x = 1:500, y = MSE4, col = "green", lty = 1)
lines(x = 1:500, y = MSE5, col = "orange", lty = 1)
legend("topright", legend = c("M = 0", "M = 2", "M = 4", "M = 6", "M = p"), col = c("black","red","blue","green","orange"), lty = 1)# will need to find a way to make these lines smoother. 

matplot(x = 1:500, y = cbind(MSE1, MSE2, MSE3, MSE4, MSE5), type = "l")

#this graphical representation is a little unorganized, but the main method that has the lowest MSE value over the range of 1 through 500 is a random forest with an M value of 6 or a bagging method call. Will most likely clean this data up a little more to 

#Experiment:
MSE_matrix <- cbind(MSE1, MSE2, MSE3, MSE4, MSE5)[seq(1, 500, by = 20),]
matplot(x = seq(1, 500, by = 20), y = MSE_matrix, type = "l", lty = 1)
legend("topright", legend = c("M = 0","M = 2","M = 4","M = 6","M = p"), lty = 1, col = 1:5)
#Interesting; this line of code worked perfectly. The main problem with my initial conclusion is that the M = 4 line has the lowest MSE score of every M value between the 0 to 100 mark (I hope that this isn't a random occurance). Even with that said, the M = p line (or rather the bagging method) has the least volatile line of all the examples (especially with regards to the M = 6 line). Hence thanks to this graphic representation I can say that the best line is M = 4. 

#8.) According to the instructions on question 8 part b I will be fitting a regression tree in place of the classification tree that was used in the lab exercises. 
str(Carseats)
??Carseats
range(Carseats$Sales)# The range is between 0 and 16.27. Remember that these values are recorded in thousands of dollars according to the documentation. 

#(a) 
train <- sample(1:nrow(Carseats), nrow(Carseats)/2)
test <- -train 

#(b)
set.seed(1)
Carseats <- Carseats[,-length(Carseats)]
carseats.tree <- tree(Sales ~ ., data = Carseats[train,])
summary(carseats.tree)
names(Carseats)
#Interesting so only 5 of the 7 possible predictor variables were used to grow the regression tree. And in addition the resulting tree only has 9 total nodes.
plot(carseats.tree)
text(carseats.tree, pretty = 0)
carseats.tree# Despite what the author says about regression trees being easy to interpret, I can't see to intergret this particular tree that I created (sadly enough). 
carseats.pred <- predict(carseats.tree, newdata = Carseats[test,])
mean((carseats.pred-Carseats[test,"Sales"])^2)
#The MSE for a regression tree with 9 node is 4.065

#(c) 
carseats.cv <- cv.tree(carseats.tree)
plot(carseats.cv$size, carseats.cv$dev, type = "b")
which.min(carseats.cv$dev) 
carseats.cv$dev
carseats.prune <- prune.tree(carseats.tree, best = 4)
plot(carseats.prune)
text(carseats.prune, pretty = 0)
carseats.pred <- predict(carseats.prune, newdata = Carseats[test,])
mean((carseats.pred - Carseats[test, "Sales"])^2)# Funny enough the MSE value is larger with 4 nodes than it was with 9 nodes. Will need to see what the problem is. 
#The MSE was 4.996 for the record. So in other words, pruning did not improve the test error. 

#(d)
carseats.random <- randomForest(Sales ~., data = Carseats[train,], ntree = 50, importance = TRUE, mtry = 8)
yhat <- predict(carseats.random, newdata = Carseats[test,])
mean((yhat - Carseats[test, "Sales"])^2)# the MSE with the random forest method and ntree set to 50 was 2.629. this is a very respectable reduction in the testing error. 
importance(carseats.random)# For reduction in MSE the most important variables are CompPrice, Income, Advertising, Price, Shelveloc, and Age. As for the reduction in RSS is compPrice, Advertising, income, Price, Selveloc, and Age. Just noticed that both of these criterias picked the same variables. 

#Will need to work out if increasing or decreasing ntree will give rise to better MSE results. 
for (i in 1:500){
	carseats.random <- randomForest(Sales ~., data = Carseats[train,], ntree = i, mtry = 8, importance = TRUE)
	yhat <- predict(carseats.random, newdata = Carseats[test,])
	MSE1[i] <- mean((yhat - Carseats[test, "Sales"])^2)
}

plot(x = seq(1, 500, by = 20), y = MSE1[seq(1,500, by = 20)], ylab = "MSE", xlab = "ntree", type = "b", ylim = c(2,3))
which.min(MSE1)# Interesting the ntree value with the lowest MSE was 26. 
MSE1[26]# The MSE value was calculated at 2.4177. Which is an alright reduction but not extreme enough for me to change my model because of it. 

#(e)
set.seed(1) 
mse <- c()
for (i in 1:8){
	carseats.random <- randomForest(Sales ~., data = Carseats[train,], ntree = 50, mtry = i, importance = TRUE)
	yhat <- predict(carseats.random, newdata = Carseats[test,])
	mse[i] <- mean((yhat - Carseats[test, "Sales"])^2)
}
which.min(mse)# The optimum mtry value is 5

mse2 <- c()
for (i in 1:8){
	carseats.random <- randomForest(Sales ~., data = Carseats[train,], ntree = 26, mtry = i, importance = TRUE)
	yhat <- predict(carseats.random, newdata = Carseats[test,])
	mse2[i] <- mean((yhat - Carseats[test, "Sales"])^2)
}
which.min(mse2)# The optimum mtry value is 3

par(mfrow = c(1,2))
plot(x = 1:8, y = mse, ylab = "Mean squared error", xlab = "M value", type = "b")
title("ntree 50")
plot(x = 1:8, y = mse2, ylab = "Mean Squared Error", xlab = "M value", type = "b")
title("ntree 26")
#Will need to keep in mind that the mtree value does have an effect on the MSE. Assessing each individual mtry and ntree will take too much time though so I will set the ntree value to a default of 50 while assessing each individual mtry value. 

#(e) 
set.seed(1)
for (i in 1:8){
	carseats.random <- randomForest(Sales ~., data = Carseats[train,], ntree = 50, mtry = i, importance = TRUE)
	yhat <- predict(carseats.random, newdata = Carseats[test,])
	mse[i] <- mean((yhat - Carseats[test, "Sales"])^2)
}
mse# Again when ntree is set to 50 the best mtry value is 5. 
which.min(mse)
plot(x = 1:8, y = mse, ylab = "MSE", xlab = "M", type = "b")# The best MSE was calculated at 2.493, which is still lower than the bagging method (though I believe that not setting a seed value is the main reason for this).
#The MSE value decreases with every one increase of M until the values rebound at or close to M = p (predictors).

#9.)
#(a)
set.seed(1)
str(OJ)
dim(OJ)
train <- sample(1:nrow(OJ), 800)
test <- -train
 
#(b) Since Purchase is a categorical variable, the classification tree method will be used for this example.
OJ.tree <- tree(Purchase ~., data = OJ[train,])# Neat there seems to be no predictor named Buy in the data frame. 
OJ.train <- predict(OJ.tree, newdata = OJ[train,], type = "class")
table(OJ.train, OJ[train,"Purchase"])
1 - ((441 + 226)/800)# The training error rate is 0.166 or rather 16.6 percent. It's important to remember though that training error rates are more optimistic than test error rates.
summary(OJ.tree)# The model has a total of 8 terminal nodes.

#(c through d)
OJ.tree
plot(OJ.tree)
text(OJ.tree, pretty = 0)
# I really need some more practice with interpreting decision tree data. I still can't interprete these results. Will need to come back to this exercise later.

#(e)
set.seed(1)
test.OJ <- predict(OJ.tree, newdata = OJ[test,], type = "class")
table(test.OJ, OJ[test, "Purchase"])
1 - ((147 + 62)/270)# The test classification error was calculated at 0.226. This is much lower than the training error rate calculated earlier. 

#(f)
set.seed(1)
OJ.cv <- cv.tree(OJ.tree)
plot(OJ.cv$size, OJ.cv$dev, ylab ="deviation", xlab = "Size", type = "b")#At eight nodes the model seems to rebound. The best tree sizes to test out are 5 and 6.
which.min(OJ.cv$dev)#Interesting according to this command the lowest is 4. But still will need to see if 5 or 6 are good alternatives as well.
OJ.prune <- prune.misclass(OJ.tree, best = 4)
OJ.pred <- predict(OJ.prune, newdata = OJ[test,], type = "class")
table(OJ.pred, OJ[test, "Purchase"])
1 - ((147+62)/270)# test classification error rate is 0.226 for 4 nodes.
 
OJ.prune1 <- prune.misclass(OJ.tree, best = 5)
OJ.pred1 <- predict(OJ.prune1, newdata = OJ[test,], type = "class")
table(OJ.pred1, OJ[test, "Purchase"])
1-((147 + 62)/270)# test classification rate is 0.226 again.

OJ.prune2 <- prune.misclass(OJ.tree, best = 6)
OJ.pred2 <- predict(OJ.prune2, newdata = OJ[test,], type = "class")
table(OJ.pred2, OJ[test, "Purchase"])
# Interestingly the classification error rates are the same between the unpruned tree and the trees of sizes 4, 5, and 6. Either the classification error rate can not be decreased or there must be something wrong with my terminal. Will need to look into this. 
par(mfrow = c(2,2))
plot(OJ.prune)
text(OJ.prune, pretty = 0)
plot(OJ.prune1)
text(OJ.prune1, pretty = 0)
plot(OJ.prune2)
text(OJ.prune2, pretty = 0)
#Interestingly OJ.prune1 and OJ.prune have the same number of nodes. 

#g.)
plot(OJ.cv$size, OJ.cv$dev, xlab = "Cross-Validation class. error", ylab = "number of nodes", type = "b")

#h.)
which.min(OJ.cv$dev)# the best number of nodes is 5. 
# The cross validation error rate for a classification tree of size 4 is 684.04.
OJ.cv$dev
OJ.cv$size

#i.)
OJ.prune5 <- prune.misclass(OJ.tree, best = 4)
OJ.pred <- predict(OJ.prune5, newdata = OJ[test,], type = "class")
table(OJ.pred, OJ[test, "Purchase"])
1 - ((147 + 62)/270)# The classification error rate is 0.226. Again the classification error rate is the same between the unpruned tree and the pruned tree with 4 nodes. 

#j.) 
unpruned.train <- predict(OJ.tree, newdata = OJ[train,], type = "class")
table(unpruned.train, OJ[train, "Purchase"])
1 - ((463 + 205)/800)# The training error rate for the unpruned classification tree is 0.165.

pruned.train <- predict(OJ.prune5, newdata = OJ[train,], type = "class")
table(pruned.train, OJ[train, "Purchase"])
1 - ((463 + 205)/800)# The training error rate for the pruned classification tree is 0.165 as well. So no difference between the unpruned or pruned trees with regards to the training error rate. 


#k.)
unpruned.test <- predict(OJ.tree, newdata = OJ[test,], type = "class")
table(unpruned.test, OJ[test, "Purchase"])
1 - ((147 + 62)/270)# The test classification error rate for the unpruned tree is 0.226

pruned.test <- predict(OJ.prune5, newdata = OJ[test,], type = "class")
table(pruned.test, OJ[test, "Purchase"])
1 - ((147 + 62)/270)# Again the pruned tree has the same classification error rate as the unpruned tree.

#10.)
#(a) 
library(ISLR)
sum(is.na(Hitters$Salary))
Hitters <- Hitters[-which(is.na(Hitters$Salary)),]
Hitters$Salary <- log(Hitters$Salary)
Hitters$Salary
dim(Hitters)

#(b) 
train <- sample(1:nrow(Hitters), 200)
test <- -train 

#(c) (The for loop idea is from asadoughi's solution)
set.seed(103)
library(gbm)
pows <- seq(-10, -0.2, by = 0.1)
lambdas <- 10 ^ pows
length.lambdas <- length(lambdas)
train.mse <- rep(NA, times = length.lambdas)
test.mse <- rep(NA, times = length.lambdas)
pred.train <- c()
pred.test <- c()
for(i in 1:length.lambdas){
	hitters.boost <- gbm(Salary ~., data = Hitters[train,], distribution = "gaussian", n.trees = 1000, shrinkage = lambdas[i])
	pred.train <- predict(hitters.boost, n.tree = 1000, newdata = Hitters[train,])
	pred.test <- predict(hitters.boost, n.tree = 1000, newdata = Hitters[test,])
	train.mse[i] <- mean((pred.train - Hitters[train,"Salary"])^2)
	test.mse[i] <- mean((pred.test - Hitters[test, "Salary"])^2)
}
# It's important to remember that the lambda value can be set through the shrinkage argument.
plot(x = lambdas, y = train.mse, ylab = "MSE value", xlab = "Lambda values", type = "l", col = "blue")
lines(x = lambdas, y = test.mse, col = "green", lty = 1)
legend("topright", legend = c("train MSE","test MSE"), col = c("blue","green"), lty = 1)
lambda.matrix <- cbind(lambdas, train.mse, test.mse)

#(d) Asadoughi's solution 
plot(lambdas, test.mse, type = "b", xlab = "Shrinkage", ylab = "Test MSE", col = "red", pch = 20)
min(test.mse)
lambdas[which.min(test.mse)]# The best test mse for this method is 0.2488
#The best test MSE is obtained at lambda 0.08. 

#(e)
#lasso regression:
lambdas
x.hitters <- model.matrix(Salary~ ., Hitters)[,-1]
y.hitters <- Hitters[,"Salary"]
library(glmnet)
lasso.mod <- glmnet(x.hitters[train,], y.hitters[train], alpha = 1, lambda = lambdas)
set.seed(1)
cv.out <- cv.glmnet(x.hitters[train,], y.hitters[train], alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.hitters <- predict(lasso.mod, s = bestlam, newx = x.hitters[test,])
mean((lasso.hitters-y.hitters[test])^2)# The best MSE value for the lasso method is 0.4926

#normal linear regression:
lm.fit <- lm(Salary ~., data = Hitters[train,])
lm.pred <- predict(lm.fit, newdata = Hitters[test,])
mean((lm.pred - Hitters[test, "Salary"])^2)# The normal linear regression model resulted in a MSE value of 0.4712 (Which is curiously better than the lasso method but still not as low as the boosting method).
summary(lm.fit)

#(f)
hitters.boost <- gbm(Salary ~., data = Hitters[train,], distribution = "gaussian", n.trees = 1000, shrinkage = 0.08)
pred.train <- predict(hitters.boost, n.tree = 1000, newdata = Hitters[train,])
pred.test <- predict(hitters.boost, n.tree = 1000, newdata = Hitters[test,])
mean((pred.train - Hitters[train,"Salary"])^2)
mean((pred.test - Hitters[test, "Salary"])^2)
summary(hitters.boost)# The most imfluential variables for this method are CRuns and CHits. 
par(mfrow = c(1,2))
plot(hitters.boost, i = "CRuns")
plot(hitters.boost, i = "CHits")

#(g) The bagging method:
hitters.bag <- randomForest(Salary ~., data = Hitters[train,], ntree = 50, mtry = 19, importance = TRUE)
hitters.pred <- predict(hitters.bag, newdata = Hitters[test,])
mean((hitters.pred-Hitters[test, "Salary"])^2)# The MSE for the bagging method is 0.2145, thus making this method the best fit for this data set. 

#11.)
#(a)
dim(Caravan)
Caravan$Purchase <- ifelse(Caravan$Purchase == "Yes", 1,0)
Caravan.train <- Caravan[1:1000,]
Caravan.test <- Caravan[1001:5822,]

#(b)
set.seed(342)
caravan.boost <- gbm(Purchase ~., data = Caravan.train, n.trees = 1000, shrinkage = 0.01, distribution = "bernoulli")
caravan.boost2 <- gbm(Purchase ~., data = Caravan.train, n.trees = 1000, shrinkage = 0.01, distribution = "bernoulli", )
summary(caravan.boost)
#The Ppersaut and Mkoopkla variables seem to be the most important variables.

#(c)
boost.prob <- predict(caravan.boost, Caravan.test, n.trees = 1000, type = "response")
boost.pred <- ifelse(boost.prob > 0.2, 1,0)
table(Caravan.test$Purchase, boost.pred)
# Guess that I need to move on to the next chapter of this text. I can't seem to get the gbm function call to work properly, which is interesting since this same function worked perfectly yesterday. 

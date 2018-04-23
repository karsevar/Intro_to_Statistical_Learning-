### Chapter 10 Unsupervised Learning:
#this chapter will instead focus on unsurpervised learning, a set of statistical tools intended for the setting in which we have only a set of features X_1, X_2, ..., X_p, measured on n observations. The goal is then to predict Y using X_1, X_2, ..., X_p. Is there an informative way to visualize the data? Can we discover subgroups among the informative way to visualize the data? Can we discover subgroups among the variables or among the observations? Unsupervised learning refers to a diverse set of techniques for answering questions such as these. In this chapter, we will focus on two particular types of unsupervised learning: principal components analysis, a tool used for data visualization or data pre-processing before supervised techniques are applied, and clustering, a broad class of methods for discovering unknown subgroups in data. 

##10.1 The challenge of Unsupervised Learning:
#Unsupervised learning is often much more challenging (with relation to supervised learning). The exercise tends to be more subjective, and there is no simple goal for the analysis, such as prediction of a response. Unsupervised learning is often performed as part of an exploratory data analysis. Furthermore, it can be hard to assess the results obtained from unsupervised learning methods, since there is no universally accepted mechanism for performing cross validation or validating results on an independent data set. The reason for this difference is simple. If we fit a predictive model using a supervised learning technique, then it is possible to check our work by seeing how the well our model predicts the response Y on observations not used in fitting the model. However, in unsupervised learning, there is no way to check our work because we don't know the true answer --- the problem is unsurpervised. 
#Techniques for unsurpervised learning are of growing importance in a number of fields. A cancer researcher might assay gene expression levels in 100 patients with breast cancer. He or she might then look for subgroups among the breast cancer samples, or among the genes, in order to obtain a better understanding of the disease. An online shopping site might try to identify groups of shoppers with siilar browsing and purchase histories, as well as items that are of particular interest to the shoppers within each group. Then an individual shopper can be preferentially shown the items in which he or she is particularly likely to be interested, based on the purchase histories of similar shoppers. A search engine might choose what search results to display to a particular individual based on the click histories of other individuals with similar search patterns. These statistical learning tasks, and many more, can be performed via unsupervised learning techniques. 

##10.2 Principal components analysis:
#Principal component analysis (PCA) refers to the process by which principal components are computed, and the subsequent use of these components in understanding the data. PCA is an unsupervised approach, wince it involves only a set of features X_1, X_2, ..., X_p, and no associated response Y. Apart from producing derived variables for use in supervised learning problems, PCA also serves as a tool for data visualization (visualization of the observations or visualization of the variables). We now discuss PCA in greater detail focusing on the use of PCA as a tool for unsupervised data exploration, in keeping with the topic of this chapter. 

##10.2.1 What are principal components:
#In placing of using the old method of creating scatter plots of (p/2) = p(p-1)/2, a better method is required to visualize the the n observations with p is large. In particular, we would like ot find a low dimensional representation of the data that captures as much of the information as possible. For instance, if we can obtain a two-dimensional representation of the data that captures most of the information, then we can plot the observations in this low-dimensional space. 

#PCA provides a tool to do just this. If finds a low-dimensional represenation of a data set that contains as much as possible of the variation. The idea is that each of the n observations lives in p-dimensional space, but not all of these dimensions are equally interesting. PCA seeks a small number of dimensions that are as interesting as possible, where the concept of interesting is measured by the amount that the observations very along each dimension. Each of the dimensions found by PCA is a linear combination of the p features. We now explain the manner in which these dimensions, or principal components are found. 
#to see the equations of how the author performed the principle component analysis in creating Z_1 and Z_2 look at page 375 and 376. 

dim(USArrests)#Interesting the dataset already has only 4 p (variables). will need to see if these variables are all normalized Principal components. 
head(USArrests)#Actually they are still individual variables. 

#We illustrate the use of PCA on the USArrests data set. For each of the 50 states in the US, the data set contains the number of arrests per 100,000 residents for each of three crimes. We also record UrbanPop (the percent of the population in each state living in urban areas). The principal component loading vectors have length p = 4. PCA was performed after standardiving each variable to have mean zero and standard deviation one. Figure 10.1 plots the first two principal components of these data. The figure represents both the principal component scores and the loading vector in a single biplot display. The loadings are also given in table 10.1. 

#In Figure 10.1, we see that the first loading vector places approximately equal weight on Assault, murder, and rape, with much less weight on UrbanPop. hence this component roughly corresponds to a measure of overall rates of servious crimes. The second loading vector places most of its weight on UrbanPop and much less weight on the other three features. Hence, this component roughly corresponds to the level of urbanization of the state. Overall, we see that the crime-related variables (Murder, Assault, and Rape) are located close to each other, and that the UrbanPop variable is far from the other three. this indicates that the crime-related variables are correlated with each other --- states with high murder rates tend to have high assault and rape rates --- and that the UrbanPop variable is less correlated with the other three. 

#We can examine differences between the states via the two principal component score vectors shown in Figure 10.1. Our discussion of the loading vectors suggests that states with large positive scores on the first component, such as California, Nevada, and Florida, have high crime rates, while states like North Dakota, with negative scores on the first component, have low crime rates. California also has a high score on the second component, indicating the high level of urbanization, while the opposite is true for states like Mississippi. States close to zero on both components, such as Indiana, have approzimately average levels of both crime and urbanization. 

##10.2.2 Another Interpretation of Principal components:
#the first two principal component loading vectors in a simulated three dimensional data set are shown in the left hand panel of figure 10.2;these two loading vectors span a plane along which the observations have the highest variance. 

#In the previous section, we describe the principal component loading vectors as the directions in feature space along which the data vary the most, and the principal component scores as projections along these directions. However, an alternative interpretation for principal components can also be useful: principal components provide low-dimensional linear surfaces that are closest to the observations. We expand upon that interpretation here. 

#The first principal component loading vector has a very special property: it is the line in p-dimensional space that is closest to the n observations (using average squared Euclidean distance as a measure of closeness). This interpretation can be seen in the left-hand panel of Figure 6.15; the dashed indicate the distand between each observation and the first principal component loading vector. The appeal of this interpretation is clear; we seek a single dimension of the data that lies as close as possible to all of the data points, since such a line will likely provide a good summary of the data. 

#The notion of principal components as the dimensions that are closest to the n observations extends beyond just the first principal component. For instance, the first two principal components of a data set span the plane that is closest to the n observations, in terms of average squared Euclidean distance. The first three principal components of a data set span the three-dimensional hyperplane that is closest to the n observations, and so forth. 

#Using this interpretation, together the first M principal component score vectors and the first M principal component loading vectors provide the best M-dimensional approximation (in terms of Euclidean distance) to the ith observation x_ij. This represenation can be written 
		#x_ij ~ sum(z_im*phi_jm)
#(assuming the orginal data matrix X is column-centered). In other words, together the M principal component score vectors and M principal component score vectors and M principal component loading vectors can given a good approximation to the data when M is sufficiently large. When M = min(n-1,p), then the represenation is exact: 
		#x_ij = sum(z_im*phi_jm).
		
##10.2.3 More on PCA:
##Scaling the variables:
#We have already mentioned that before PCA is performed, the variable should be centered to have mean zero. Furthermore, the results obtained when we perform PCA will also depend on whether the variables have been individually scaled (each multiplied by a different constant). This is in contrast to some other supervised and unsupervised learning techniques, such as linear regression, in which scaling the variables has no effect. 

#For instance, Figure 10.1 was obtained after scaling each of the variables to have a standard deviation one. Why does it matter than we scaled the variables? In these data, the variables are measured in different units; Murder, Rape, and Assault are reported as the number of occurrences per 100,000 people, and UrbanPop is the percentage of the state's population that lives in an urban area. These four variables have variance 18.97, 87.73, 6945.16, and 209.5, respectively. Consequently, if we perform PCA on the unscaled variables, then the first principal component loading vector will have a very large loading for Assault, since that variable has by far the highest variance. The right-hand plot in Figure 10.3 displayes the first two principal components for the USArrests data set, without scaling the variables to have standard deviation one. As predicted, the first principal component loading vector places almost all of its weight on Assault, while the second principal component loading vector places almost all of its weight on UrbanPop. Comparing this to the left hand plot, we see that scaling does indeed have a substantial effect on the results obtained. 

#Because it is undesirable for the principal components obtained to depend on an arbitrary choice of scaling, we typically scale each variable to have standard deviation one before we perform PCA.

#In certain settings, however, the variables may be measured in the same units. In this case, we might not wish to scale the variables to have standard deviation one before performing PCA. For instance, suppose that the variables in a given data set correspond to expression levels for p genes. Then since expression is measured in the same units for each gene, we might choose not to scale the genes to each have standard deviation one. 

##Uniqueness of the Principal components:
#Each principal component loading vector is unique, up to a sign flip. This means that two different software packages will yield the same principal component loading vectors, although the signs of those loading vectors may differ. The signs may differ because each principal component loading vector specifies a direction in p-dimensional space: flipping the sign has no effect as the direction does not change. (the principal component loading vector is a line that extends in either direction, and flipping its sign would have no effect.) Similarly, the score vectors are unique up to a sign flip, since the variance of Z is the same as the variance of -Z. It is worth noting that when we use (10.5) to approximate x_ij we multiply z_im by phi_jm. Hence, if the sign is flipped on both the loading and score vectors, the final product of the two quantities is unchanged. 

##the proportion of variance explained
#How much of the information in a given data set is lost by projecting the observations onto the first few principal components? That is, how much of the variance in the data is not contained in the first few principal components? That is, how much of the variance in the data is not contained in the first few principal components? More generally , we are interested in knowing the proportion of variance explained (PVE) by each principal component. The total variance present in a data set (assuming that the variables have been centered to have mean zero) is defined as:
		#sum(Var(X_j)) = sum(1/n)*sum(x^2_ij),
#And the variance explained by the mth principal component is 
		#1/n*sum(z^2_im) = 1/n*sum(sum(phi_jm*x_ij))^2
#therefore, the PVE of the mth principal component is given by
		#sum(sum(phi_jm*x_ij))^2/sum(sum(x^2_ij))

#The PVE of each principal component is a positive quantity. In order to compute the cumulative PVE of the first M principal components, we can simply sum (10.8) over each of the first M PVEs. In total, there are min(n - 1,p) principal components, and their PVEs sum to one. 

##Deciding How Many Principal components to use:
#In general, a n * p data matrix X has min(n - 1, p) distinct principal components. However, we usually are not interested in all of them; rather, we would like to use just the first few principal components in order to visualize or interpret the data. In fact, we would like to use the smallest number of principal components required to get a good understanding of the data. How many principal components required to get a good understanding of the data. How many principal components are needed? Unfortunately, there is no single answer to this question.

#We typically decide on the number of principal components required to visualize the data by examining a scree plot. We choose the smallest number of principal components that are required in order to explain a sizable amount of the variation in the data. This is done by eyeballing the scree plot, and looking for a point at which the proportion of variance explained by each subsequent principal component drops off. 

#In practice, we tend to look at the first few principal components in order to find interesting patterns in the data. If no interesting patterns are found in the first few principal components, then further principal components are unlikely to be of interest. Conversely, if the first few principal components are interesting, then we typically continue to look at subsequent principal components until no further interesting patterns are found. Since this method is subjective at best it is mainly allocated to exploratory data analysis. 

#On the other hand, if we compute principal components for use in a supervised analysis, such as the principal components regression presented in Section 6.3.1, then there is a simple and objective way to determine how many principal components to use: we can treat the number of principal component score vectors to be used in the regression as a tuning parameter to be selected via cross-validation or a related approach. The comparative simplicity of selecting the number of principal components for a supervised analysis is one manifestation of the fact that supervised analyses tend to be more clearly defined and more objectively evaluated than unsupervised analyses. 

##10.3 Clustering Methods:
#Clustering refers to a very broad set of techniques for finding subgroups, or clusters, in a data set. When we cluster the observations of a data set, we seek to partition them into distinct groups so that the observations within each group are quite similar to each other, while observations in different groups are quite different from each other. Of course, to make this concrete, we must define what it means for two or more observations to be similar or different. 

#Both clustering and PCA seek to simplify the data via a small number of summaries, but their mechanisms are different:
	#PCA looks to find a low-dimensional represenation of the observations that explain a good fraction of the variance;
	#Clustering looks to find homogeneous subgroups among the observations.
	
#In this section we focus on perhaps the two best-known clustering approaches: K-means clustering and hierarchal clustering. In K-means clustering, we seek to partition the observations into a pre-specified number of clusters. On the other hand, in hierarchical clustering, we do not know in advance how many clusters we want; in fact, we end up with a tree-like visual represenation of the observations, called a dendrogram, that allows us to view at one the clusterings obtained for each possible number of clusters, from 1 to n. There are advantages and disadvantages to each of these clustering approaches, which we highlight in this chapter. 

#In general, we can cluster observations on the basis of the features in order to identify subgroups among the observations, or we can cluster features on the basis of the observations in order to discover subgroups among the features. In what follows, for simplicity we will discuss clustering observations on the basis of the features, though the converse can be performed by simply transposing the data matrix. 

##10.3.1 K-Means Clustering:
#K-means clustering is a simple and elegant approach for partitioning a data set into k distinct, non-overlapping clusters. To perform K-means clustering, we must first specify the desired number of clusters K; then the K-means algorithm will assign each observation to exactly one of the K clusters. 

#The K-means clustering proceducer results from a simple and intuitive mathematical problem. We begin by defining some notation. Let C_1, ..., C_k denote sets containing the indices of the observations in each cluster. These sets satisfy two properties:
	# 1. In other words, each observation belongs to at least one of the K clusters.
	# 2. In other words, the clusters are non-overlapping: no observation belongs to more than one cluster. 
	
#The idea behind K-means clustering is that a good clustering is one for which the within-cluster variation is as small as possible. The within cluster variation for cluster C_k is a measure W(C_k) of the amount by which the observations within a cluster differ from each other. Hence we ant to solve the problem:
		#minimize{sum(W(C_k))}.
		
#In words, this formula says that we want to partition the observations into K clusters such that the total within cluster variation, summed over all K clusters, is as small as possible. 

#In order to solve for the preceeding equation, you have to first define the within cluster variation. The most common choice involves squared Euclidean distance. That is, we define
		#W(C_k) = 1/|C_k|*sum(sum(x_ij - x_i'j))^2,
#where |C_k| denotes the number of observations in the kth cluster. In other words, the within cluster variation for the kth cluster is the sum of all of the pairwise squared Euclidean distances between the observations in the kth cluster, devided by the total number of observations in the kth cluster. Combining (10.9) and (10.10) gives the optimization problem that defines K-means clustering,
		#minimize/C_1, ..., C_k{sum(1/|C_k|)*sum(sum(x_ij-x_i'j))^2}.
		
#Now , we would like to find an algorithm to solve (10.11) --- that is, a method to partition the observations into K clusters such that the objective of (10.11) is minimized. A very simple algorithm can be shown to provide a local optimum to the K-means optimization problem. 

#to see the algorithm used consult page 388.

#K-means clustering derives its name from the fact that in step 2(a), the cluster centroids are computed as the mean of the observations assigned to each cluster. 

#Because the K-means algorithm finds a local rather than a global optimum, the results obtained will depend on the initial (random) cluster assignment of each observation in step 1 of the algorithm. (The importance of the starting point is some what similar to gradient descent and ascent as stated by Simulation for data science with R). For this reason, it is important to run the algorithm multiple times from different random initial configurations. Then one selects the best solution, i.e. that for which the objective (10.11) is smallest. Figure 10.7 shows the local optima obtained by running k-means clustering six times using six different initial cluster assignments, using the toy data frame Figure 10.5. 

##10.3.2 Hierarchical Clustering:
#On potential disadvantage of k-means clustering is that it requires us to pre-specify the number of clusters K. Hierarchical clustering is an alternative approach which does not require that we commit to a praticular choice of K. Hierarchical clustering has an added advantage over K-means clustering in that it results in an attractive tree-based represenation of the observations, called a dendrogram. 

#In this section, we describe bottom-up or agglomerative clustering. This is the msot common type of hierarchical clustering, and refers to the fact that a dendrogram is built starting from the leaves and combining clusters up to the trunk.

##Interpreting a Dendrogram:
#Each leaf of the dendrogram represents one of the 45 observations in figure 10.8. However, as we move up the tree, some leaves begin to fuse into branches. these correspond to observations that are similar to each other. As we move higher up the tree, branches themselves fuse, either with leaves or other branches. The earlier fusions occur, the more similar the groups of observations are to each other. On the other hand, observations that fuse later can be quite different. In fact, this statement can be made precise: for any two observations, we can look for the point in the tree where branches containing those two observations are first fused. The height of this fusion, as measured on the vertical axis, indicates how different the two observations are. Thus, observations that fuse at the very bottom of thetree are quite similar to each other, whereas observations that fuse close to the top of the tree will tend to be quite different.

#This highlights a very important point in interpreting dendrograms that is often misunderstood. 

#Proximity between branches and leaves mean nothing in dendrogram illustrations. To put it mathematically, there are 2^n-1 possible reorderings of the dendrogram, where fusions occur, the positions of the two fused branches could be swapped without affecting the meaning of the dendrogram. Therefore, we cannot draw conclusions about the similarity of two observations based on their proximity along the horizontal axis. Rather, we draw conclusions about the similarity of two observations based on the location on the vertical axis where branches containing those two observations first are fused. 

#We can move on to the issue of identifying clusters on the basis of a dendrogram. In order to do this, we make a horizontal cut across the dendrogram, as shown in the center and right hand panels of figure 10.9. The distinct sets of observations beneath the cut can be interpreted as clusters. Further cuts can be made as one descends the dendrogram in order to obtain any number of clusters, between 1 (corresponding to no cut) and n (corresponding to a cut at height 0, so that each observation is in its own cluster). In other words, the height of the cut to the dendrogram serves the same role as the K in K-means clustering: it controls the number of clusters obtained. 

#The characteristic that makes hierarchical clustering very attractive is that one single dendrogram can be used to obtain any number of clusters. In practice, people often look at the dendrogram and select by eye a sensible number of clusters, based on the heights of the fusion and the number of clusters desired. 

#The term hierarchical refers to the fact that clusters obtained by cutting the dendrogram at a given height are necessarily nested within the clusters obtained by cutting the dendrogram at any greater height. However hierarchical structure is not really a constant characteristic of the method and should only be assumpted if the data one is using this method on is such a structure. 

#Due to situations, hierarchal clustering can sometimes yield worse results than K-means clustering for a given number of clusters. 

##the Hierarchical Clusterig Algorithm:
#To see how the method is implementated under the hood see page 394. 

#One important detail is that groups, after the initial one observation grouping step, are clustered into other groups through four different methods (complete, single, average, and centroid). Usually researcher use only the complete and average linkage methods because they give rise to interpretable results (unlike the implementation of single linkages). It's important to remember that that linkage style dictates the shape of the dendrogram itself. linkage method description can be found on page 395. 

##Choice of dissimilarity Measure:
#Correlation based distance considers two observations to be similar if their features are highly correlated, even though the observed values may be far apart in terms of Euclidean distance. This is an unusual use of correlation, which is normally computed between variables; here it is computed between the observation profiles for each pair of observations. Correlation-based distance focuses on the shape of observations profiles rather than their magnitudes. 

#the choice of dissimilarity measure is very important, as it has a strong effect on the resulting dendrogram. In general, careful attention should be paid to the type of data being clustered and the scientific question at hand. These considerations should determine what type of dissimilarity measure is used for hierarchical clustering. 

#In addition to carefully selecting the dissimilarity measure used, one must also consider whether or not the variables should be scaled to have standard deviation one before the dissimilarity between the observations is computed. 

##Small Decisions with Big Consequences:
#In order to perform clustering, some decisions must be made:
		#Should the observations or features first be standardized in some way? 
		#In the case of hierarchical clustering,
			#What dissimilarity measure should be used?
			#What type of linkage should be used?
			#Where should we cut the dendrogram in order to obtain clusters?
		#In the case of K-means clustering, how many clusters should we look for in the data?
		
##Validating the clusters obtained:
#There exist a number of techniques for assigning a p-value to a cluster in order to assess whether there is more evidence for the cluster than one would exprect due to chance.

##Other considerations in clustering:
#Misture models are an attractive approach for accommodating the presence of such outliers. These amount to a soft version of K-means clustering.
#In addition, clustering methods generally are not very robust to perturbations. 

##10.4 Lab 1: Principal Components Analysis:
#In this lab,we perform PCA on the USArrests data set, which is part of the base R package the rows of the data set contain the 50 states, in alphabetical order. 
states <- row.names(USArrests)
states
names(USArrests)

#We notice that the variables have vastly different means. this means that we might have to normalize the data to have a 0 mean value and 1 standard deviation.
apply(USArrests, 2, mean)

#Note that the apply() function allows us to apply a function --- in this case the mean() function -- to each row and column of the data set. the second input here denotes whether we wish to compute the mean of the rows, 1, or the columns, 2. We see that there are on average three times as many rapes as murders, and more than eight times as many assaults as rapes. We can alos examine the variances of the four variabless using the apply() function again. 
apply(USArrests, 2, var)

#the variables also have vastly different variances: the UrbanPop variable measures the percentage of the population in each state living in an urban area, which is not a comparable number to the number of rapes in each state per 100,000 individuals. If we failed to scale the variables before performing PCA, then most of the principal components that we observed would be driven by the Assault variable, since it has by far the largest mean and variance. 

#We now perform principal components analysis using the prcomp() function, which is one of several functions in R that perform PCA.
pr.out <- prcomp(USArrests, scale = TRUE)

#By default, the prcomp() function centers the variables to have mean zero. By using the option scale = TRUE, we scale the variables to have standard deviation one. The output from prcomp() contains a number of useful quantities. 
names(pr.out)

#the center and scale components correspond to the means and standard deviations of the variables that were used for scaling prior to implementating PCA. 
pr.out$center
pr.out$scale

#The rotation matrix provides the principal component loadings; each column of pr.out$rotation contains the corresponding principal component loading vector.
pr.out$rotation

#We see that there are four distinct principal components. This is to be expected because there are in general min(n - 1, p) informative principal components in a data set with n observations and p variables. 

#Using the prcomp() function, we do not seed to explicitly multiply the data by the principal component loading vectors in order to obtain the principal component score vectors. Rather the 50 by 4 matrix x has as its columns the principal component score vectors. That is, the kth column is the kth principal component score vector. 
dim(pr.out$x)

#We can plot the first two principal components as follows:
biplot(pr.out, scale = 0)

#The scale = 0 argument to biplot() ensures that the arrows are scaled to represent the loadings; other values for scale give slightly different biplots with different interpretations. 

#Notice that this figure is a mirrow image of figure 10.1. Recall that the principal components are only unique up to a sign change, so we can reproduce Figure 10.1 by making a few small changes:
pr.out$rotation <- -pr.out$rotation
pr.out$x <- -pr.out$x 
biplot(pr.out, scale = 0)

#The prcomp() function also outputs the standard deviation of each principal component. For instance, on the USArrests data set, we can access these standard deviations as follows:
pr.out$sdev
#The variance explained by each principal component is obtained by squaring these:
pr.var <- pr.out$sdev^2
pr.var

#To compute the proportion of variance explained by each principal component, we simply divide the variance explained by each principal component by the total variance explained by all four principal components:
pve <- pr.var/sum(pr.var)
pve

#We see that the first principal component explains 62 percent of the variance in the data, the nex principal component explains 24.7 percent of the variance, and so forth. We can plot the PVE explained by each component, as well as the cumulative PVE, as follows:
par(mfrow = c(1,2))
plot(pve, xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0,1), type = "b")
plot(cumsum(pve), xlab = "Proportion of Variance Explained", ylim = c(0,1), ylab="Cumulative Proportion of Variance Explained", type = "b")

##10.5 Lab 2:Clustering:
##10.5.1 K-means and Clustering:
#the function kmeans() performs K-means clustering in R. We begin with a simple simulated example in which there truly are two clusters in the data: the first 25 observations have a mean shift relative to the next 25 observations. 
set.seed(2)
x <- matrix(rnorm(50*2), ncol = 2)
x[1:25,1] <- x[1:25,1]+3
x[1:25,2] <- x[1:25,2]-4

#we now perform K-means clustering with K = 2
km.out <- kmeans(x, 2, nstart = 20)
#The cluster assignments of the 50 observations are contained in km.out$cluster.

km.out$cluster
#The K-means clustering perfectly separated the observations into two clusters even though we did not supply any group information in kmeans(). We can plot the data, with each observation colored according to its cluster assignment. 
plot(x, col = (km.out$cluster+1), main = "K-means Clustering Results with K=2", xlab = "", ylab = "", pch = 20, cex = 2)
plot(x)

#Here the observations can be easily plotted because they are two-dimensional. If there were more than two variables then we could instead perform PCA and plot the first two principal components score vectors. 

#In this example, we knew that there really were two clusters because we generated the data. However, for real data, in general we do not know the true number of clusters. We could instead have performed K-means clustering on this example with K = 3.
set.seed(4)
km.out <- kmeans(x, 3, nstart = 20)
km.out 

plot(x, col = (km.out$cluster+1), main = "K-means Clustering Results with K = 3", xlab = "", pch = 20, cex = 2)

#When K = 3, K-means clustering splits up the two clusters. To run the kmeans() function in R with multiple initial cluster assignments, we use the nstart argument. If a value of nstart greater than one is used, then K-means clustering will be performed using multiple random assignments in Step 1 of algorithm 10.1, and the kmeans() function will report only the best results. Here we compare using nstart = 1 and nstart = 20.
set.seed(3)
km.out <- kmeans(x, 3, nstart = 1)
km.out$tot.withinss
km.out <- kmeans(x, 3, nstart = 20)
km.out$tot.withinss

#Note that km.out$tot.withinss is the total within-cluster sum of squares, which we seek to minimize by performing K-means clustering (In other words, this is most likely the K-means clustering equivalent of RSS for least squares regression). The individual within cluster sum of squares are contained in the vector km.out$withinss. 

#We strongly recommend always running K-means clustering with large value of nstart, such as 20 or 50, since otherwise an undesirable local optimum may be obtained.

#When performing K-means clustering, in addition to using multiple initial cluster assignments, it is also important to set a random seed using the set.seed() function. This way, the initial cluster assignments in Step 1 can be replicated, and the K-means output will be fully reproducible. 

##10.5.2 Hierarchical clustering:
#The hclust() function implements hierarchical clustering in R. In the following example we use the data from Section 10.5.1 to plot the hierarchical clustering dendrogram using complete, single, and average linkage clustering, with Euclidean distance as the dissimilarity measure. We begin by clustering observations using complete linkage. The dist() function is used to compute the 50 by 50 inter-observation Euclidean distance matrix. 
hc.complete <- hclust(dist(x), method = "complete")

#We could just as easily perform hierarchical clustering with average or single linkage instead:
hc.average <- hclust(dist(x), method = "complete")
hc.single <- hclust(dist(x), method = "single")

#We can now plot the dendrograms obtained using the usual plot() function. The numbers at the bottom of the plot identify each observation. 
par(mfrow = c(1,3))
plot(hc.complete, main = "complete Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.average, main = "average Linkage", xlab = "", sub = "", cex = 0.9)
plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = 0.9)

#To determine the cluster labels for each observation associated with a given cut of the dendrogram, we can use the cutree() function:

cutree(hc.complete, 2)
cutree(hc.average, 2)
cutree(hc.single, 2)

#For this data, complete and average linkage generally separate the observations into their correct groups. However, single linkage identifies one point as belongingin to its own cluster. A more sensible answer is obtained when four clusters are selected, although there are still two singletons. 
cutree(hc.single, 4)

#To scale the variables before performing hierarchical clustering of the observations, we use the scale() function. 
xsc <- scale(x)
plot(hclust(dist(xsc), method = "complete"), main = "Hierarchical Clustering with Scaled Features")

#Correlation based distance can be computed using the as.dist() function, which converts an arbitrary square symmetric matrix into a form that the hclust() function recognizes as a distance matrix. However, this only makes sense for data with at least three features since the absolute correlation between any two observations with measurements on two features is always 1. Hence, we will cluster a three-dimensional data set. 
x <- matrix(rnorm(30*3), ncol = 3)
dd <- as.dist(1-cor(t(x)))
plot(hclust(dd, method = "complete"), main = "complete Linkage with correlation-based distance", xlab = "", sub = "")

##10.6 Lab 3: NCI60 Data Example:
#Unspervised techniques are often used in the analysis of genomic data. In particular, PCA and hierarchical clustering are popular tools. We illustrate these techniques on the NCI60 cancer cell line microarray data, which consists of 6380 gene expression measurements on 64 cancer cell lines.
library(ISLR)
nci.labs <- NCI60$labs 
nci.data <- NCI60$data 

#Each cell ine is labeled with a cancer type. We do not make use of the cancer types in performing PCA and clustering, as these are unsurpervised techniques. But after performing PCA and clustering, we will check to see the extent to which these cancer types agree with the results of these unsupervised techniques. 

#The data has 64 rows and 6,830:
dim(nci.data)

#We begin by examining the cancer types for the cell lines.
nci.labs[1:4]
table(nci.labs)

##10.6.1 PCA on the NCI60 Data:
#We first perform PCA on the data after scaling the variables (genes) to have standard deviation one, although one could reasonably argue that it is better not to scale the genes. 
pr.out <- prcomp(nci.data, scale = TRUE)

#We now plot the first few principal component score vectors, in order to visualize the data. The observations (cell lines) corresponding to a given cancer type will be plotted in the same color, so that we can see to what extent the observations within a cancer type are similar to each other. We first create a simple function that assigns a distinct color to each element of a numeric vector. The function will be used to assign a color to each of the 64 cell lines, based on the cancer type to which it corresponds. 
Cols <- function(vec){
	cols <- rainbow(length(unique(vec)))
	return(cols[as.numeric(as.factor(vec))])
}

#Note that the rainbow() function takes as its argument a positive integer, and returns a vector containing the number of distinct colors. We now can plot the principal component score vectors. 
par(mfrow = c(1,2))
plot(pr.out$x[,1:2], col = Cols(nci.labs), pch =19, xlab = "Z1", ylab = "Z2")
plot(pr.out$x[,c(1,3)], col = Cols(nci.labs), pch =19, xlab = "Z1", ylab = "Z3")

#the resulting plots are shown in the following figure. On the whole, cell lines corresponding to a single cancer type do tend to have similar values on the first few principal component score vectors. This indicates that cell lines from the same cancer type tend to have pretty similar gene expression levels. 

#We can obtain a summary of the proportion of variance explained (PVE) of the first few principal components using the summary() method for a prcomp object (we have truncated the printout):
summary(pr.out)

#Using the plot() function, we can also plot the variance explained by the first few principal components.
plot(pr.out)

#Note that the height of each bar in the bar plot is given by squaring the corresponding element of pr.out$sdev. However, it is more informative to plot the PVE of each principal component (a scree plot) and the cumulative PVE of each principal component. This can be done with just a little work. 
pve <- 100*pr.out$sdev^2/sum(pr.out$sdev^2)
par(mfrow = c(1,2))
plot(pve, type = "o", ylab = "PVE", xlab = "Principal Component", col = "blue")
plot(cumsum(pve), type = "o", ylab = "Cumulative PVE", xlab = "principal component", col = "brown3")

#(note that the elements of pve can also be computed directly from the summary, summary(pr.out)$importance[2,], and the elements of cumsum(pve) are given by summary(pr.out)$importance[3,].) We see that together, the first seven principal components explain around 40 percent of the variance in the data. This is not a huge amount of the variance. However, looking at the scree plot, we see that while each of the first seven principal components explain a substantial amount of variance, there is a marked decrease in the variance explained by further principal components. That is, there is an elbow in the plot after approximately the seventh principal components. this suggests that there may be little benefit to examining more than seven or so principal components.

##10.6.2 Clustering the observations of the NCI60 Data:
#To begin, we standardize the variables to have mean zero and standard deviation one. As mentioned earlier, this step is optional and should be performed only if we want each gene to be on the same scale.
sd.data <- scale(nci.data)

#We now perform hierarchical clustering of the observations using complete, single, and average linkage. Euclidean distance is used as the dissimilarity measure.
par(mfrow = c(3,1))
data.dist <- dist(sd.data)
plot(hclust(data.dist), labels = nci.labs, main = "complete linkage", xlab = "", sub = "", ylab = "")
plot(hclust(data.dist, method = "average"), labels = nci.labs, main = "Average Linkage", xlab = "", sub = "", ylab = "")
plot(hclust(data.dist, method = "single"), labels = nci.labs, main = "Single Linkage", xlab = "", sub = "", ylab = "")  

#The results are shown in the following graphic. We see that the choice of linkage certainly does affect the results obtained. typically, single linkage will tend to yield trailing clusters: very large clusters onto which individual observations attach one by one. On the other hand, complete and average linkage tend to yield more balanced, attractive clusters. For this reason, complete and average linkage are generally preferred to single linkage. Clearly cell lines within a single cancer type do tend to cluster together, although the clustering is not perfect. We will use complete linkage hierarchical clustering for the analysis that follows. 

#We can sut the dendrogram at the height that will yield a particular number of clusters, say four:
hc.out <- hclust(dist(sd.data))
hc.clusters <- cutree(hc.out,4)
table(hc.clusters, nci.labs)

#there are some clear patterns. All the leukemia cell lines fall in cluster 3, while the breast cancer cell lines are spread out over three different clusters. We can plot the cut on the dendrogram that produces these four clusters:
par(mfrow=c(1,1))
plot(hc.out, labels = nci.labs)
abline(h = 139, col = "red")

#The abline() function draws a straight line on top of any existing plot in R. The argument h = 139 plots a horizontal line at height 139 on the dendrogram; this is the height that results in four distinct clusters. It is easy to verify that the resulting clusters are the same as the ones we obtained using cutree(hc.out, 4).

#Printing the output of hclust gives a useful brief summary of the object:
hc.out 

#How do these NCI60 hierarchical clustering results compare to what we get if we perform K-means clustering with K = 4?
set.seed(2)
km.out <- kmeans(sd.data, 4, nstart = 20)
km.clusters <- km.out$cluster
table(km.clusters, hc.clusters)

#We see that the four clusters obtained using hierarchical clustering and K-means clustering are somewhat different. Cluster 2 in K-means clustering is identical to cluster 3 in hierarchical clustering. However, the other clusters differ: for instance, cluster 4 in K-means clustering contains a portion of the observations assigned to cluster 1 by hierarchical clustering, as well as all of the observations assigned to cluster 2 by hierarchical clustering. 

#Rather than performing hierarchical clustering on the entire data matrix we can simply perform hierarchical clustering on the first few principal component score vectors, as follows.
hc.out <-hclust(dist(pr.out$x[,1:5]))
plot(hc.out, labels = nci.labs, main = "hier. Clust. on First five score vectors")
table(cutree(hc.out,4), nci.labs)

#Not surprisingly, these results are different from the ones that we obtained when we performed hierarchical clustering on the full data set. Sometimes performing clustering on the first few principal component score vectors can given better results than performing clustering on the full data. In this situation, we might view the principal component step as one of denoising the data. We could also perform K-means clustering on the first few principal component score vectors rather than the full data set. 

##10.7 Exercises:
##conceptual:
#1.)
#(a) This question is sadly above my mathematical proficiency. Will need to look into what Asadoughi wrote about this problem. 

#Asadoughi's solution: Again sadly the solution to this problem is written in mathematica and hence I can't really understand it. 

#(b) Asadoughi's soltuion: Equation (10.12) shows that minimizing the sum of the squared Euclidean distance for each cluster is the same as minimizing the within cluster variance for each cluster. 

#2.)
d <- matrix(c(0, 0.3,0.4, 0.7,0.3,0,0.5,0.8,0.4,0.5,0,0.45,0.7,0.8,0.45,0), nrow = 4)

#(a)
par(mfrow = c(1,2))
plot(hclust(as.dist(d), method = "complete"))#Interesting so the main thing that I got wrong with my initial function call was that I did not illustrate that the object d should be considered a distance object. Will need to remember this little addition when working on hierarchical clustering structures and dendrograms.

#(b)
plot(hclust(as.dist(d), method = "single"))

#(c-d)
comp <- hclust(as.dist(d), method = "complete")
single <- hclust(as.dist(d), method = "single")
cut.comp <- cutree(comp, 3)
table(cut.comp)

cut.single <- cutree(single, 3)
table(cut.single)
#I'm not really sure if these outputs are correct will need to look into Asadoughi's solution.

#Asadoughi's solution: (Now I get what the author is getting at, he wants me to comment on how skewed the single linkage method is compared to the complete linkage method.)
#the correct answer is c(1,2) and c(3,4) for the first and second cluster respectively for the complete linkage method.
#And for the single linkage method c(4) and c(3,1,2). You can see that the single linkage method clusters the observations one observation at a time. 

#(e).
d <- matrix(c(0.3,0,0.5,0.8,0.7,0.8,0.45,0,0.4,0.5,0,0.45,0,0.3,0.4,0.7), byrow = TRUE, nrow = 4)
plot(hclust(as.dist(d), method = "complete"))
plot(hclust(as.dist(d), method = "single"))
#This solution reorders the cluster components, which is what you don't want to happen. The only thing that you want is that the underlying components within each cluster to be placed in different positions.

#Asadoughi's solution:
plot(hclust(as.dist(d), method = "complete"), labels =c(2,1,4,3))

#3.) Asadoughi's solution:
#(a) 
set.seed(1)
x <- cbind(c(1,1,0,5,6,4), c(4,3,4,1,2,0))
plot(x[,1], x[,2])

#(b) 
labels <- sample(2, nrow(x), replace=T)
labels#This creates the two class predictor for all 6 observations. 

#(c) 
centroid1 <- c(mean(x[labels==1,1]), mean(x[labels==1,2]))
centroid2 <- c(mean(x[labels==2,1]), mean(x[labels==2, 2]))
centroid1 
centroid2
plot(x[,1], x[,2], col = (labels+1), pch = 20, cex=2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)

#(d)
euclid <- function(a,b){
	return(sqrt((a[1] - b[1])^2 + (a[2]-b[2])^2))
}
assign_labels <- function(x, centroid1, cnetroid2){
	labels <- rep(NA, nrow(x))
	for(i in 1:nrow(x)){
		if(euclid(x[i,], centroid1) < euclid(x[i,], centroid2)){
			labels[i] <- 1
		}else {
			labels[i] <- 2
		}
	}
	return(labels)
}
labels <- assign_labels(x, centroid1, centroid2)
labels

#(e) 
last_labels <- rep(-1, 6)
while(!all(last_labels==labels)){
	last_labels <- labels
	centroid1 <- c(mean(x[labels==1,1]), mean(x[labels==1,2]))
	centroid2 <- c(mean(x[labels==2,1]), mean(x[labels==2,2]))
	print(centroid1)
	print(centroid2)
	labels <- assign_labels(x, centroid1, centroid2)
}
labels

#(f) 
plot(x[,1], x[,2], col = (labels+1), pch = 20, cex = 2)
points(centroid1[1], centroid1[2], col = 2, pch = 4)
points(centroid2[1], centroid2[2], col = 3, pch = 4)

#(4)
#(a) theoretically the fusion of {1,2,3} and {4,5} will occure higher on the tree for the single linkage method because single linkage can only carry out one observation cluster at a time thus skewing the size of the dendrogram. While the complete linkage method has the ability to fuse multiple observations at a time thus giving rise to a more balanced dendrogram illustration. 

#Asadoughi's solution:
#Not enough information to tell. The maximal intercluster dissimilarity could be equal or not equal to the minimal intercluster dissimilarity. If the dissimilarities were equal, they would fuse at the same height. If they were not equal, the single linkage dendrogram would fuse at a lower height.

#(b) Most likely Asadoughi will say not enough information do to the fact that we don't know the dissimilarity distance between observations {5} and {6}.

#Asadoughi's solution 
#They would fuse at the same height because linkage does not affect leaf-to-leaf fusion. Again, I really need to practice this method a little more since I'm getting all of these question wrong. 

#5.) Asadoughi's solution: Clusters selected based on two-dimensional distance:

#(a) Least socks and computers (3,4,6,8) versus more socks and computers (1,2,7,8).

#(b) Purchased computer (5,6,7,8) versus no computer purchase (1,2,3,4). The distance on the computer dimension is greater than the distance on the socks dimension.

#(c) Purchased computer (5,6,7,8) versus no computer purchase (1,2,3,4).

#6.) This problem is way too advanced for my meager skills will need to lean on Asadoughi once again. 

#(a) the first principal component explains 10 percent of the variation means 90 percent of the information in the gene data set is lost by projecting the tissue sample observations onto the first principal component. Another way of explaining it is 90 percent of the variance in the data is not contained in the first principal component.

#(b) Given the flaw shown in pre-analysis of a time-wise linear trend amongst the tissue samples' first principal component, I would advise the researcher to include the machine used (A vs B) as a feature of the data set. this should enhance the PVE of the first principal component before applying the two sample t-test. 

#(c) 
set.seed(1)
Control <- matrix(rnorm(50*1000), ncol = 50)
Treatment <- matrix(rnorm(50*1000), ncol = 50)
X <- cbind(Control, Treatment)
X[1,] <- seq(-18,18-0.36, 0.36)#linear trend in one dimension
pr.out <- prcomp(scale(X))
summary(pr.out)$importance[,1]# 9.911 percent variance explained by the first principal component 

#Now, adding in A vs B via 10 vs 0 encoding.
X <- rbind(X, c(rep(10, 50), rep(0, 50)))
pr.out <- prcomp(scale(X))
summary(pr.out)$importance[,1]
#11.54 percent variance explained by the first principal component. That's an improvement of 1.629 percent. 

##Applied:
#7.) 
library(ISLR)
set.seed(1)
dsc <- scale(USArrests)
a <- dist(dsc)^2
b <- as.dist(1-cor(t(dsc)))
summary(b/a)

#8.) 
#(a)
US.data <- prcomp(USArrests, scale = TRUE)
pve <- 100*US.data$sdev^2/sum(US.data$sdev^2)# The first principal component has a PVE value of 62 percent while the second principal component has a PVE vale of 24.7 percent and so on.
plot(pve, x = c(1:4), type = "b", ylab = "Proportion of variance explained", xlab = "Number of Principal component")
plot(cumsum(pve), type = "b", ylab = "Cumulative PVE", xlab = "Principal Component")

#(b) Asadoughi's solution:
loadings <- pr.out$rotation
US.scale <- scale(USArrests)
pve2 <- rep(NA, 4)
dmean <- apply(US.scale, 2, mean)
dsdev <- sqrt(apply(US.scale, 2, var))
dsc <- sweep(US.scale, MARGIN = 2, dmean, "-")
dsc <- sweep(dsc, MARGIN = 2, dsdev, "/")
for(i in 1:4){
	proto_x <- sweep(dsc, MARGIN = 2, loadings[,i],"*")
	pc_x <- apply(proto_x, 1, sum)
	pve2[i] <- sum(pc_x^2)
}
pve2 <- pve2/sum(dsc^2)
pve2
# Really sure if this is the correct answer will need to look into this output later. The PVE for this manual method are still different than the prcomp() function method. Will need to look into what the problem is. 

#9.)
#(a)
US.custer.unscale <- hclust(dist(USArrests), method = "complete")
plot(US.custer.unscale)# there we go this dendrogram looks more like what the author was refering to in the question description.

#(b)
US.unscale.cut <- cutree(US.custer.unscale, 3)
table(US.unscale.cut)#It seems like the best cut parameter is actually 3 I first tried out 2 for this function but the under of clusters was restricted to 2.
plot(US.custer.unscale)
abline(h = 150, lty = 2, col = "red")

#(c)
head(USArrests)
US.scale <- scale(USArrests)
US.cluster <- hclust(dist(US.scale), method = "complete")
plot(US.cluster)

#(Initial mistake tried to use these commands to calculate the first question) this seems weird but I can't seem to split the dendrogram into three clusters through the cutree() function because the overall spread of the illustration is too unifor. this must be because of the scaling step that was taken before plotting the dendrogram.

#(d)
summary(USArrests)
apply(USArrests, MARGIN = 2, mean)
apply(USArrests, MARGIN = 2, median)
apply(USArrests, MARGIN = 2, var)
#After looking at these values, the variables should all be scaled to a certain extent because the high mean value Assault and UrbanPop will only inflate their significance within the overall dendrogram illustration. In other words, the scale for all of the variables need to be consistent in order for this method to work properly. 

#10.)
#(a)Asadoughi's solution 
set.seed(2)
x <- matrix(rnorm(20*3*50, mean = 0, sd = 0.001), ncol = 50) 
x[1:20,2] <- 1
x[21:40, 1] <- 2
x[21:40, 2] <- 2
x[41:60,1] <- 1
#the concept here is to separate the three classes amongst two dimensions

#(b)
x.PCA <- prcomp(x) #since the observations are all ready uniform in scale there is no need to use the scale() function with this problem. 
pve <- 100*x.PCA$sdev^2/ sum(x.PCA$sdev^2)
par(mfrow=c(1,2))
plot(pve, type = "o", col = "red", ylab = "Cumulative PVE value", xlab = "Principal Component Number")
plot(pve[1:11], type = "o", col = "red", ylab = "Cumulative PVE value", xlab = "Principal Component Number")# this says that most of the variance can be explained with the first two principal components.
par(mfrow = c(1,1))
plot(x.PCA$x[,1:2], col = 2:4, xlab = "Z1", ylab = "Z2", pch = 19)

#(c)
set.seed(2)
km.out <- kmeans(x, 3, nstart = 25)
km.out$cluster
table(km.out$cluster, c(rep(1,20), rep(2,20), rep(3, 20)))
#According to Asadoughi this is a perfect match. will need to look into why this is the case.

#(d)
set.seed(2)
km.out <- kmeans(x, 2, nstart = 25)
km.out# Interesting class 2 absorded what used to be class 1 and 3 and class one took the place of class 2.

#(e)
km.out <- kmeans(x, 4, nstart = 25)
km.out$cluster# class four took the place of class 3 and class one is now currently shared between class 1 and class 3.

#(f) 
km.out <- kmeans(x.PCA$x[, 1:2], 3, nstart = 25)
km.out$cluster#This method has the same output as using the entire data set to calculate the three classes using k-means clustering. 

#(g)
x.scale <- scale(x)
km.out2 <- kmeans(x.scale, 3, nstart = 25)
km.out2$cluster#The class assignments are more erratic. will need to see if this might mean that scaling will need to be used with descretion on a case by case basis. This is because now the values are all equally weighted.

#11.)
#(a)
gene <- read.csv("Ch10Ex11.csv", stringsAsFactors = FALSE, header = FALSE)

#(b)
#unscaled:
dd <- as.dist(1-cor(gene))
par(mfrow = c(1,1))
plot(hclust(dd, method = "complete"))# Two clusters were created through this method
plot(hclust(dd, method = "single"))#Of course this method is useable.
plot(hclust(dd, method = "average"))# At cutree(1) this method creates two clusters in all.
#The results really do depend on the linkage method used in each hierarchical cluster model.

#(c)
pr.out <- prcomp(t(gene))
summary(pr.out)
total.load <- apply(pr.out$rotation, 1, sum)
indices <- order(abs(total.load), decreasing = TRUE)
indices[1:10]
total.load[indices[1:10]]



		
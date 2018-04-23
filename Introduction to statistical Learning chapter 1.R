### Introduction to statistical learning chapter 1:
## An overview of statistical Learning:
#Statistical learning refers to a vast set of tools for understanding data. These tools can be classified as supervised or unsupervised. Broadly speaking, supervised statistical learning involves buidling a statistical model for predicting, or estimating, an output based on one or more inputs. With unsupervised statistical learning, there are inputs but no supervising output.

##The author will use the wage dataset throughout this entire books will need to see what he means by this statement. Namely what package, if any, does this dataset come in?
library(ISLR)
dim(Wage)
dimnames(Wage)
??Wage# It seems like this is the dataset that the author is refering to. The variables seem to be the same and the region the survey was carried out seem to be a perfect match. 
library(ggplot2)
ggplot(data = Wage, aes(x = age, y = wage)) + geom_point() + geom_smooth()# this seems to be a perfect match for the first graphic. The default smoothing method was gam in place of loess.
ggplot(data = Wage, aes(x = year, y = wage)) + geom_point(color = "dark gray") + geom_smooth(method = "lm")# will need to see how the author created a smoothing trend for this data. Will most likely need to use base R graphics to create a compariable graphic. 
#Now I understand he used the base R lm() regression method for the geom_smooth() geometric layer.
ggplot(data = Wage, aes(x = education, y = wage)) + geom_boxplot(aes(fill = education))

#Clearly, the most accurate prediction of a given man's wage will be obtained by combining his age, his education, and the year. In Chapter 3, we discuss linear regression, which can be used to predict wage from this data set. Ideally, we should predict wage in a way that account for the non-linear relationship between wage and age. 

##Stock market Data:
#The wage data involves predicting a continuous or quantitative output value. this is often referred to as a regression problem. However, in certain cases we may instead wish to predict a non-numeric value --- that is, a categorical or qualitiative output. 
dimnames(Smarket)
??Smarket 
head(Smarket, n = 10)
market_me <- ggplot(data = Smarket)
market_me + geom_boxplot(aes(x= Direction, y = Lag1, fill = Direction))
market_me + geom_boxplot(aes(x = Direction, y = Lag2, fill = Direction))
market_me + geom_boxplot(aes(x = Direction, y = Lag3, fill = Direction))

#The following graphics contains the daily movements in the Standard and poor 500 index over a 5 year period between 2001 and 2005. The goal is to predict whether the index will increase or decrease on a given day using the past 5 days' percentage changes in the index. Here the statistical learnin problem does not involve predicting a numerical value. Instead it involves predicting whether a given day's stock market performance will fall into the Up bucket or the Down bucket. This is known as a classification problem. 

##Gene Expression Data:
#The previous two application illustrate data sets with both input and output variables. However, another important class of problems involves situations in which we only observe input variables, with no corresponding output. For example, in a marketing setting, we might have demographic information for a number of current or potential customers. We may wish to understand which types of customers are similar to each other by grouping individuals according to their observed characteristics. This is known as a clustering problem. 

#We devote chapter 10 to a discussion of statistical learning methods for problems in which no natural output variable is available. We consider the NC160 data set, which consists of 6830 gene expression measurements for each of 64 cancer cell lines. Instead of predicting a particular output variable, we are interested in determining whether there are groups, or clusters, among the cell lines based on their gene expression measurements. 

??NCI60
dimnames(NCI60)# Interesting there are no dimension names for this data set.
head(NCI60, n = 10)# this dataset is organized as a list() will need to see what techques the author uses to clean this mess up. I can't seem to find where he obtained the Z_1 and Z_2 variables for his two-dimensional plot. 

# In this particular data set, it turns out that the cell lines correspond to 14 different types of cancer. (however, this information was not used to create the left-hand panel for his graphic) the right-hand panel of the graphic is identical to the left hand panel, except that the 14 cancer types as shown using distinct colored symbols. There is clear evidence that cell lines with the same cancer type tend to be located near each other in this two-dimensional representation. In addition, even though the cancer information was not used to produce the left hand panel, the clustering obtained does bear some resemblance to some of the actual cancer type observed in the righ hand panel. 

#Notation and simple Matrix algebra:
A <- matrix(c(1,2,3,4), ncol = 2, nrow = 2, byrow = TRUE)
B <- matrix(c(5,6,7,8), ncol = 2, nrow = 2, byrow = TRUE)

A %*% B #So this is how matrix multiplication is carried out. Very interesting. 
A * B # good the values are completely different from the earlier command and the command in the book. 

#Note that this operation produces an r * s matrix. It is only possible to compute AB if the number of columns of A is the same as the number of rows of B. 
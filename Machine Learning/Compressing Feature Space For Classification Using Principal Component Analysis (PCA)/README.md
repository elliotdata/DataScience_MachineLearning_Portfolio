# Compressing-Feature-Space-For-Classification-Using-PCA

Project_Link: https://prajivinn.github.io/2022/01/21/pca.html

## Context
Our client is looking to promote Ed Sheeran’s new album - and want to be both targeted with their customer communications, and as efficient as possible with their marketing budget.

As a proof-of-concept they would like us to build a classification model for customers who purchased Ed’s last album based upon a small sample of listening data they have acquired for some of their customers at that time.

If we can do this successfully, they will look to purchase up-to-date listening data, apply the model, and use the predicted probabilities to promote to customers who are most likely to purchase.

The sample data is short but wide. It contains only 356 customers, but for each, columns that represent the percentage of historical listening time allocated to each of 100 artists. On top of these, the 100 columns do not contain the artist in question, instead being labelled artist1, artist2 etc.

We will need to compress this data into something more manageable for classification!


## Actions
We firstly needed to bring in the required data, both the historical listening sample, and the flag showing which customers purchased Ed Sheeran’s last album. We ensure we split our data a training set & a test set, for classification purposes. For PCA, we ensure that we scale the data so that all features exist on the same scale.

We then apply PCA without any specified number of components - which allows us to examine & plot the percentage of explained variance for every number of components. Based upon this we make a call to limit our dataset to the number of components that make up 75% of the variance of the initial feature set (rather than limiting to a specific number of components). We apply this rule to both our training set (using fit_transform) and our test set (using transform only)

With this new, compressed dataset, we apply a Random Forest Classifier to predict the sales of the album, and we assess the predictive performance!

## Results
Based upon an analysis of variance vs. components - we made a call to keep 75% of the variance of the initial feature set, which meant we dropped the number of features from 100 down to 24.

Using these 24 components, we trained a Random Forest Classifier which able to predict customers that would purchase Ed Sheeran’s last album with a Classification Accuracy of 93%

## Growth/Next Steps
We only tested one type of classifier here (Random Forest) - it would be worthwhile testing others. We also only used the default classifier hyperparameters - we would want to optimise these.

Here, we selected 24 components based upon the fact this accounted for 75% of the variance of the initial feature set. We would instead look to search for the optimal number of components to use based upon classification accuracy.

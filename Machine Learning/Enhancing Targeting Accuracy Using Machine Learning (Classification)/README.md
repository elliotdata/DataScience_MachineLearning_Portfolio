# Enhancing-Targeting-Accuracy-Using-ML

Project_Link: https://prajivinn.github.io/2022/01/19/enhancing-targeting-accuracy.html

## Context
Our client, a grocery retailer, sent out mailers in a marketing campaign for their new delivery club. This cost customers $100 per year for membership, and offered free grocery deliveries, rather than the normal cost of $10 per delivery.

For this, they sent mailers to their entire customer base (apart from a control group) but this proved expensive. For the next batch of communications they would like to save costs by only mailing customers that were likely to sign up.

Based upon the results of the last campaign and the customer data available, we will look to understand the probability of customers signing up for the delivery club. This would allow the client to mail a more targeted selection of customers, lowering costs, and improving ROI.

Let’s use Machine Learning to take on this task!


## Actions
We firstly needed to compile the necessary data from tables in the database, gathering key customer metrics that may help predict delivery club membership.

Within our historical dataset from the last campaign, we found that 69% of customers did not sign up and 31% did. This tells us that while the data isn’t perfectly balanced at 50:50, it isn’t too imbalanced either. Even so, we make sure to not rely on classification accuracy alone when assessing results - also analysing Precision, Recall, and F1-Score.

As we are predicting a binary output, we tested four classification modelling approaches, namely:

* Logistic Regression
* Decision Tree
* Random Forest
* K Nearest Neighbours (KNN)

For each model, we will import the data in the same way but will need to pre-process the data based up the requirements of each particular algorithm. We will train & test each model, look to refine each to provide optimal performance, and then measure this predictive performance based on several metrics to give a well-rounded overview of which is best.


## Results
The goal for the project was to build a model that would accurately predict the customers that would sign up for the delivery club. This would allow for a much more targeted approach when running the next iteration of the campaign. A secondary goal was to understand what the drivers for this are, so the client can get closer to the customers that need or want this service, and enhance their messaging.

Based upon these, the chosen the model is the Random Forest as it was a) the most consistently performant on the test set across classication accuracy, precision, recall, and f1-score, and b) the feature importance and permutation importance allows the client an understanding of the key drivers behind delivery club signups.


Metric 1: **Classification Accuracy**

KNN = 0.936
Random Forest = 0.935
Decision Tree = 0.929
Logistic Regression = 0.866

Metric 2: **Precision**

KNN = 1.00
Random Forest = 0.887
Decision Tree = 0.885
Logistic Regression = 0.784

Metric 3: **Recall**

Random Forest = 0.904
Decision Tree = 0.885
KNN = 0.762
Logistic Regression = 0.69

Metric 4: **F1 Score**

Random Forest = 0.895
Decision Tree = 0.885
KNN = 0.865
Logistic Regression = 0.734


## Growth/Next Steps
While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty

# Understanding-Alcohol-Product-Relationships-Using-Associate_Rules_Learning

Project_Link: https://prajivinn.github.io/2022/01/22/association-rule-learning.html

## Context
Our client is looking to re-jig the alcohol section within their store. Customers are often complaining that they can’t find the products they want, and are also wanting recommendations about which other products to try. On top of this, their marketing team would like to start running “bundled” promotions as this has worked well in other areas of the store - but need guidance with selecting which products to put together.

They have provided us a sample of 3,500 alcohol transactions - our task is fairly open - to see if we can find solutions or insights that might help the business address the aforementioned problems!

## Actions
Based upon the tasks at hand - we apply Association Rule Learning, specifically Apriori to examine & analyse the strength of relationship between different products within the transactional data.

We firstly installed the apyori package, which contains all of the required functionality for this task.

We then needed to bring in the sample data, and get it into the right format for the Apriori algorithm to deal with.

From there we apply the Apriori algorithm to provide us with several different relationship metrics, namely:

Support
Confidence
Expected Confidence
Lift
These metrics examine product relationships in different ways, so we utilise each to put forward ideas that address each of the tasks at hand. You can read more about these metrics, and the Apriori algorithm in the relevant section below.

## Results
Interestingly, the strongest relationship existed between two products labelled as “gifts” - this is useful information for the category managers as they may want to ensure that gift products are available in one section of the aisle, rather than existing in their respective product types.

We also saw some strong relationships between French wines, and other French wines - which again is extremely useful for category managers who are thinking about the best way to lay out the products - having sections by country rather than necessarily by type might make it easier for customers to find what they are after.

Another interesting association is between products labelled “small”. At this point, we don’t know exactly what that means - but it is certainly something to take back to the client as they may be able to make more sense of it, and turn it into an actionable insight!

We propose to also build a “search engine” for category managers where they can look-up products by keyword in the product association table.

As an example - we search for any products that associate strongly with “New Zealand” products. There appeared to be some relationship between New Zealand wines and other New Zealand wines, but what was also interesting was that New Zealand wines seemed to be more associated with French & South American wines than they were with Australian Wines.

New Zealand & Australia are often grouped together, but in terms of wine this wouldn’t make sense - perhaps because of the difference climates the wines are very different and thus it wouldn’t make sense to group wines by geographical proximity, but by preference instead. This is only a hypothesis for now - we will need to take this back to the client and get their category experts to help us interpret it!

## Growth/Next Steps
As this is first & foremost an exploratory project, we will take back the results to the client Category Managers & discuss the results, our views on how these insights can be actioned best, and any considerations that need to be taken into account when interpreting.

From there we will recommend applying this same logic to all other categories, as well as potentially across the full-product range.

We will also propose the build of the “Keyword Search Engine” which will help Category Managers extract and utilise the insights held within the data.


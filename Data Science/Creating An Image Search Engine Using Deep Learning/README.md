# Image-Search-Engine-Deep-Learning

Project_Link: https://prajivinn.github.io/2022/01/27/image-search-engine.html

## Context
Our client had been analysing their customer feedback, and one thing in particular came up a number of times.

Their customers are aware that they have a great range of competitively priced products in the clothing section - but have said they are struggling to find the products they are looking for on the website.

They are often buying much more expensive products, and then later finding out that we actually stocked a very similar, but lower-priced alternative.

Based upon our work for them using a Convolutional Neural Network, they want to know if we can build out something that could be applied here.

## Actions
Here we implement the pre-trained VGG16 network. Instead of the final MaxPooling layer, we we add in a Global Average Pooling Layer at the end of the VGG16 architecture meaning the output of the network will be a single vector of numeric information rather than many arrays. We use “feature vector” to compare image similarity.

We pre-process our 300 base-set images, and then pass them through the VGG16 network to extract their feature vectors. We store these in an object for use when a search image is fed in.

We pass in a search image, apply the same preprocessing steps and again extract the feature vector.

We use Cosine Similarity to compare the search feature vector with all base-set feature vectors, returned the N smallest values. These represent our “most similar” images - the ones that would be returned to the customer.

## Results
We test two different images, and plot the search results along with the cosine similarity scores. You can see these in the dedicated section below.

## Discussion, Growth & Next Steps
The way we have coded this up is very much for the “proof of concept”. In practice we would definitely have the last section of the code (where we submit a search) isolated, and running from all of the saved objects that we need - we wouldn’t include it in a single script like we have here.

Also, rather than having to fit the Nearest Neighbours to our feature_vector_store each time a search is submitted, we could store that object as well.

When applying this in production, we also may want to code up a script that easily adds or removes images from the feature store. The products that are available in the clients store would be changing all the time, so we’d want a nice easy way to add new feature vectors to the feature_vector_store object - and also potentially a way to remove search results coming back if that product was out of stock, or no longer part of the suite of products that were sold.

Most likely, in production, this would just return a list of filepaths that the client’s website could then pull forward as required - the matplotlib code is just for us to see it in action manually!

This was tested only in one category, we would want to test on a broader array of categories - most likely having a saved network for each to avoid irrelevant predictions.

We only looked at Cosine Similarity here, it would be interesting to investigate other distance metrics.

It would be beneficial to come up with a way to quantify the quality of the search results. This could come from customer feedback, or from click-through rates on the site.

Here we utilised VGG16. It would be worthwhile testing other available pre-trained networks such as ResNet, Inception, and the DenseNet networks.


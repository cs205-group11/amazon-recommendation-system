Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs. 

# Introduction

The goal of this project is to parallelize the process of generating product recommendations to Amazon's users. Specifically, we aim to predict, as accurately as possible, the rating a user gives to a particular product. If we are able to make accurate predictions, we can recommend products to users that they have not bought yet. 

## Problem Description

[Amazon](https://www.amazon.com/), the world's largest e-commerce marketplace, relies on targeted recommendations in order to sell a broad range of products to its users. These recommendations should be based on a user's previous purchase history as well as products that similar users have purchased. Therefore, computing how similar two users are is an essential part of the recommendation process. Good recommendations benefit both customers, who receive products that are better suited to their needs and are able to save shopping time, as well as Amazon itself, as they are able to sell a greater number of products, successfully market new products, and obtain customer loyalty as buying more products increases the quality of recommended products.  

## Existing Solutions to the Problem

There are two broad approaches to generate recommendations: 

* Content-based systems: these systems aim to assess the features of the products being bought. They aim to classify products into different clusters or categories and then recommend other products within this cluster or category. Some examples of this technique include recommending athletic wear to customers who have bought sports equipment or recommending horror movies to customers who have watched other horror movies. 

* Collaborative filtering systems: these systems aim to assess the users purchasing items. Specifically, they provide a metric to compare how similar two users are and then recommend products that users that are similar to the target user rated highly. Our project will follow this approach and we analyze a few different methods and algorithms in the broad domain of collaborative filtering, including Standard Collaborative Filtering Model (SCF) and Matrix Factorization (MF) optimized through Alternative Least Square (ALS). 

### Collaborative Filtering

In order to perform collaborative filtering, one needs to create a **utility matrix** [[1](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)]. This rows of this utility matrix correspond to users, and the columns correspond to products. Each entry in the matrix is the rating (1-5, both inclusive) given to a product by a user. For example, let us say we have 6 products: P1 through P6, and 5 users: U1 through U5. Since all users do not rate all products, the utility matrix ends up being quite sparse. The corresponding utility matrix looks as follows: 

|    | P1 | P2 | P3 | P4 | P5 | P6 |
|:---|:---|:---|:---|:---|:---|:---|
| U1 |  3 |    |  4 |  5 |    |    |
| U2 |  1 |    |    |    |  5 |    |
| U3 |    |    |  4 |    |    |  3 |
| U4 |    |  5 |  2 |    |  5 |    |
| U5 |  3 |    |    |  5 |    |  4 |

Now, in order to recommend products to a new user, U6, we must first find users in our dataset who are similar to U6. In order to do this, we use a metric called **[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)**, where U6 is treated like a vector and compared with other users in the dataset, also treated as vectors. We essentially use the [dot product](https://en.wikipedia.org/wiki/Dot_product) between two vectors to compute the angle between them. The smaller the angle, the closer the two vectors are to each other and the more similar the users. We then recommend U6 products that these similar users have rated highly.

## Need for Big Data and Big Compute

Amazon's dataset is not, unfortunately, neatly organized into a matrix of users and products. We are dealing with a large, unstructured dataset and in order to process it into a matrix of this form, we would need to make use of **big data processing** solutions such as Spark. Since Amazon has over 50 million users and 10 million products, a matrix this size would not fit on a single node, and we can take advantage of a distributed cluster of nodes in order to perform efficient pre-processing of this dataset. 

In order to compute similarity scores and generate predictions, we rely on a lot of matrix or vector products. These matrix operations can be made parallel through **big compute** and we use multi-threading to speed up these computations. Overall, the goal of our project is to increase the speedup of the whole process of generating recommendations, which includes pre-processing the raw dataset as well as computing predictions using the utility matrix, using a hybrid approach involving big-data processing and big-compute.

* * *

# Methodology and Design

## Data

The raw dataset that we use for this project is the "Amazon Product Data" that was collected by Julian McAuley et al. from University of California, San Diego (UCSD) [2]. We came across this dataset because it was used extensively in machine learning applications such as [3].  This dataset contains 142.8 million product reviews, as well as the associated metadata from Amazon spannning May 1996 to July 2014. A sample review of this dataset is as follows:
```
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```

As shown above, each product contains a range of attributes. The most interesting attributes for our application are `reviewerID`,`asin` and `overall`, and they have the following meanings:
* `reviewerID`: a unique string consisting of letters and numbers representing each individual reviewer.
* `asin`: a unique ID for the product.
* `overall`: the rating given by reviewer `reviewerID` to product `asin`, ranging from 1 to 5. 

## Recommendation System Model

In our project, we use two typical recommendation system models to perform benchmarking, powered by the underlying programming model introduced in the next section. These two recommendation system models are:
* Standard Collaborative Filtering Model (SCF)
* Matrix Factorization (MF) optimized through Alternative Least Square (ALS)

### Model Setup
To begin with, we can assume that we have a *n* × *m* matrix, where *n* represents the number of users and *m* represents the number of products. Each entry in this matrix *r<sub>ij</sub>* is the rating given by user *i* to product *j*.
The overall goal, is to predict a rating that has not yet been given from user *i* to product *j* (i.e. calculate the predicted rating *r<sub>ij</sub>*).

### Standard Collaborative Filtering Model (SCF)
In SCF, we predict the rating based on the nearest neighborhood algorithm (kNN). More specifically, we can calculate the **cosine similarity** between the current user *i* to all other users, and select top *k* users based on the similarity score. From these *k* users, we can calculate the weighted average of ratings for product *j* with the cosine similarity as weights. This averaged rating is used as *r<sub>ij</sub>*.

The **advantage** of this model is as follows:
* Easy to understand
* Easy to implement

However, this model suffers from the following **limitations**:
* It is not computationally efficient
* It does not handle sparsity well (i.e. It does not have accurate predictions if there are not enough reviews for a product)

### Matrix Factorization (MF) optimized through Alternative Least Square (ALS)
In light of the above two limitations of SCF, matrix factorization is a more advanced model that decomposes the original sparse matrix to lower-dimensional matrices incorporating latent vectors. These latent vectors may include higher-level attributes which are not captured by ratings for individual products. 

To factorize a matrix, single value decomposition is a common technique, where a matrix *R* can be decomposed of matrices *U, Σ, V*, where *Σ* is a matrix containing singular values of the original matrix. However, given that R is a sparse matrix, we can find matrices *U* and *V* directly, with the goal that the product of *U* and *V* is an approximation of the original matrix *R*. 

Therefore, this problem is turned into an optimization problem to find *U* and *V*, whose product is a good approximation of *R*. One way to numerically compute this is Alternative Least Square (ALS) [3], where either the user factor matrix or item factor matrix is held constant in turn, and update the other matrix. This approach yields a higher accuracy as seen from Performance Evaluation section.  

## Code Profiling

Code profiling goes here

Code example below:

```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

## Model Description and Programming Model

Model description here and example image below:

![Branching](https://guides.github.com/activities/hello-world/branching.png)

## Parallel application

Description of parallel application

## Platform and Infrastructure

Description of the platform and infrastructure

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

* * *

# Usage Instructions

## Software Design

Discuss Software design here

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```

## How to Use our Code

1. AWS EMR instance with 8 c5.9xlarge(36 CPUs and 72 Gib memory) worker nodes.

2. log in to the AWS EMR cluster

3. Download the rating data http://jmcauley.ucsd.edu/data/amazon/links.html
```
wget http://snap.stanford.edu/data/amazon/productGraph/aggressive_dedup.json.gz
```
4. Unzip the rating data.
```
tar xvzf aggressive_dedup.json.gz
```

5. Put the rating data into the hadoop file system.
```
hadoop fs -put aggressive_dedup.json
```

6. Delete the original copy.
```
rm -r aggressive_dedup.json 
```

7. Pull the git repository containing the code.
```
git clone https://github.com/JinZhaoHong/cs205_amazon_recommendation.git 
```

8. Run the code.
```
spark-submit --num-executors 8 --executor-cores 32  als_recommendation.py aggressive_dedup.json
```

9. You should see some newly generated folder in the hdfs.
```
hadoop fs -ls
```
To increase executor memory, add the flag
``` 
--driver-memory 2g --executor-memory 2g
```

10. If you run the code again, don't forget the delete the output generated by the previous run. For example:
```
hadoop fs -rm -r X

```



## How to Run Tests

How to run tests

### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzillaa</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

* * *

# Results

## Performance Evaluation

Speed-up, throughput, weak and strong scaling

## Optimizations and Overheads

Discussion about overheads and optimizations done

* * *

# Advanced Features

Discussion about advanced features here

* * *

# Discussion

Discuss

## Goals Achieved

Goals Achieved here

## Improvements Suggested

Improvements suggested here

## Interesting Insights

Interesting insights here

## Lessons Learnt

Lessons learnt here

## Future Work

Future work here

* * *

# References

References here

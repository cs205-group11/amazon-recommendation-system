Text can be **bold**, _italic_, or ~~strikethrough~~.

[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.

# Introduction

Introduction here

## Problem Description

Problem description here

## Need for Big Data and Big Compute

Describe the need for big data and big compute here

## Existing Solutions to the Problem

Describe existing solutions here

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

## Challenges

Describe challenges here

> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.

* * *

# Methodology and Design

## Data

The raw dataset that we use for this project is the "Amazon Product Data" that was collected by Julian McAuley et al. from University of California, San Diego (UCSD) [1]. We came across this dataset because it was used extensively in machine learning applications such as [2].  This dataset contains 142.8 million product reviews, as well as the associated metadata from Amazon spannning May 1996 to July 2014. A sample review of this dataset is as follows:
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

As shown above, each product contains a range of attributes. The most interesting attributes for our application is `reviewerID`,`asin` and `overall`, and they have the following meanings:
* `reviewerID`: a unique string consisting of letters and numbers representing each individual reviewer.
* `asin`: a unique ID for the product.
* `overall`: the rating given by reviewer `reviewerID` to product `asin`, ranging from 1 to 5. 

## Model

In our project, we use two typical recommendation system models to perform benchmarking, based on the programming model mentioned in the next section. These two recommendation system models are:
* Standard Collaborative Filtering Model (SCF)
* Matrix Factorization using Alternative Least Square (ALS)

### Model Setup
To begin with, we can assume that we have a $n$ × $m$ matrix, where $n$ represents the number of user and $m$ represents the number of products. Each entry in this matrix $r_{ij}$ is the rating given by user $i$ to product $j$.
The overall goal, is to predict a rating that has not yet been given from user $i$ to product $j$ (i.e. calculate the predicted rating $\hat{r}_{ij}$).

### Standard Collaborative Filtering Model (SCF)
In SCF, we predict the rating based on the nearest neighborhood algorithm (kNN). More specifically, we can calculate the **cosine similarity** between the current user $i$ to all other users, and select top $k$ users based on the similarity score. From these $k$ users, we can calculate the weighted avaerage of ratings for product $j$ with the cosine similarity as weights. This averaged rating is used as $\hat{r}_{ij}$.

The **advantage** of this model is as follows:
* Easy to understand
* Easy to implement

However, this model suffers from following **limitations**:
* It is not computationally efficient
* It does not handle sparsity well (i.e. It does not have accurate predictions if there are not enough reviews for a product)

### Matrix Factorization using Alternative Least Square (ALS)
In light of above two limitations of SCF, Matrix Factorization is a more advanced model that decomposes the original sparse matrix to lower-dimensional matrices incorporating latent vectors and are less sparse. The high level idea behind  

## Code Profiling

Code profiling goes here

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

How to use

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

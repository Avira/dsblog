---
layout: post
title:  "Continuous and automatic key driver analysis of customer feedback"
date:   2016-11-18
author: Manuel J. A. Eugster
---

In all of Avira's products, customers are able (and invited!) to provide feedback
whenever they feel the need to tell us about their experiences with the products
at hand. This feedback does not disappear into nirvana---instead it is collected,
automatically analyzed and distributed within the company to relevant
stakeholders. Product managers and developers---to name a few---use the
processed feedback to make decisions for future product releases.

<center>
<br/>
<img src="{{ site.baseurl }}/assets/cfdrivers/ccdd2.png" width="70%" />
<br/>
</center>

At Avira we try to use the terms **data-driven** and **customer centric** not as
buzzwords, but really execute on these two concepts. The **automatic key driver
analysis for customer feedback** is one example where we developed an end-to-end
pipeline to provide a basis for decisions on data collected from customers.
In general, a [key driver analysis](https://www.thebalance.com/key-driver-analysis-and-chart-reports-2275739)
is the study of the relationships among many factors to identify the most important
ones. In our case, we are interested in the relationship between the general
customer satisfaction and specific joy and pain points customers experience during
the usage of our products. Such an analysis results, for example, in an estimate on
how much the decrease of the customer satisfaction
score is, given that a customer complains about a problem with the automatic update
of our Avira Antivirus product. If this decrease is significant to other complains,
we know that this issue needs to be prioritized.

In this post we will present the full end-to-end pipeline for the continuous and
automatic key driver analysis in detail. We will illustrate the infrastructure
and discuss the statistical model used behind the scenes. This is work developed
by the **Customer Insights Research** team[^1] at [Avira](http://www.avira.com).

[^1]: The Customer Insights Team consists of [Michaela Beckenbach](https://www.xing.com/profile/Michaela_Beckenbach), [Jürgen Lass](https://www.xing.com/profile/Juergen_Lass2), and [Manuel Eugster](http://mjae.net).


## Customer feedback

<center>
<img src="{{ site.baseurl }}/assets/cfdrivers/survey.png" height="400px" />
<br/>
</center>

If a customer decides to give feedback, he or she will be forwarded to a
product-specific survey. Every survey starts with a set of questions standardized
throughout all Avira products and touch-points:

1. The first question is always the
[Net Promotor Score (NPS)](https://en.wikipedia.org/wiki/Net_Promoter) question
on how likely it is that you recommend Avira to family, friends or colleagues.
The rating scale is from 0 (Not at all likely) to 10 (Extremely likely).

1. The second question is the
[Customer Satisfaction Score (CSAT)](https://en.wikipedia.org/wiki/Customer_satisfaction)
on how you would rate the product at hand. The rating scale is
from *Excellent* to *Poor* (note that this is an
[unbalanced](http://www.academia.edu/7904172/Likert_scales_Design_issues)
[Likert scale](https://en.wikipedia.org/wiki/Likert_scale)).

1. The third question is an
[open text](https://en.wikipedia.org/wiki/Unstructured_data) question on how we
could improve the product at hand. The customer is invited to write about his or
her pain points as well as joy points while using the product.

Now, in order to gain insights into the provided customer feedback, we assume a
relationship between the customer satisfaction score (question 2) and the content
written in the open text (question 3): If customers are satisfied with the product,
they will write about their joy points; if they are not satisfied they will write
about their pain points. We want to understand how strong the influence of
individual joy and pain points on the customer satisfaction scoring is.


## End-to-end pipeline

<center>
<img src="{{ site.baseurl }}/assets/cfdrivers/infrastructure.png" width="80%" />
<br/>
</center>

Our end-to-end pipeline from the customers to the stakeholders within Avira works
in principle like the following:

1. Customer feedback is collected with the
[SurveyMonkey](https://www.surveymonkey.com/) service. A *Getter* job continuously
collects the latest survey responses via the SurveyMonkey API.

1. We collect customer feedback in 13 different languages (from English and German
to Traditional Chinese). A *Translator* job automatically translates the collected
texts into English via the [Google Translate API](https://translate.google.com/).

1. We use a dictionary approach to tag the open text feedbacks with Avira
specific tags. A *Tagger* job using
[IBM Text Analytics](http://www-03.ibm.com/software/products/en/spss-text-analytics-surveys)
categorizes the joy and pain points written as open texts into tags.

1. A simple statistical model estimates the relationship between
the customer satisfaction and the tags. The *Modeller* job with the implementation of
the statistical model is implemented using the [R language](http://www.r-project.org).

1. Finally, the results are visualized using a [Tableau](http://www.tableau.com/)
 dashboard.

Most of the underlying [extract, transform, load (ETL)](https://en.wikipedia.org/wiki/Extract,_transform,_load)
jobs to access APIs and read/write databases are written in the
[R language](http://www.r-project.org).


## Key driver visualization

In order to make the results---the estimated key drivers for customer
satisfaction---accessible within the company, we use a simple visualization
created as an interactive Tableau dashboard.

<center>
<br/>
<img src="{{ site.baseurl }}/assets/cfdrivers/viz1.png" width="95%" />
<br/>
</center>

The visualization shows the
key drivers for a specific product (the real tags are anonymized to *Tag1*,
*Tag2*, etc). On the $$x$$-axis the relative frequency of the
tags is shown and on the $$y$$-axis the effect of the tags on the customer
satisfaction score (CSAT). A negative effect is shown as an increase in the
CSAT score (towards the top) and a positive effect is shown as a decrease in the
CSAT score (towards the bottom); no effect is at $$0$$.

To provide guidance for interpretation we split the visualization into four
quadrants using the mean for the relative frequency ($$x$$-axis) and $$0$$ for
the effect on the CSAT ($$y$$-axis). The
two top quadrants contain tags with negative impact on the CSAT, the two bottom
quadrants with positive impact on the CSAT. The two right quadrants contain
tags with a high number of customers talked about, the two left
with a low number of customers talked about. Therefore, the
most critical quadrant with tags that really need to be investigated is the
top-right quadrant (followed by the top-left, the bottom-right, and the
bottom-left quadrants).

In the introduction we used as an example the complain about "a problem with the
automatic update of our Avira Antivirus" product. In this visualization, *Tag1*
is the tag "update problem". This tag is in the top-right quadrant, so many
customers complained about it and it also has high negative effect on the
customer satisfaction score.


<center>
<br/>
<img src="{{ site.baseurl }}/assets/cfdrivers/viz2.png" width="95%" />
<br/>
</center>

The visualization is also used to monitor the progress of tags over time. If,
for example, a development team takes on an action to fix a specific tag, over time
less and less customers should talk about this tag. Again, the "update problem"
tag *Tag1* is an example for such a scenario. For other tags, like *Tag19*, no
action was defined, and therefore the tag continuous to be unvaried over time.


## Inside the *Modeller*

<center>
<img src="{{ site.baseurl }}/assets/cfdrivers/modeller.png" width="70%" />
<br/>
</center>

Now, the effect of specific tags on the customer satisfaction score is estimated
with a simple statistical model. The model takes the CSAT scores and the tags as
input and provide an estimate of the relationship between the tags and the CSAT
score---we model
$$
\text{CSAT} \sim \text{Tags}
$$
.


The purpose of this specific model is to be a middle-term key driver analysis of
the most frequent joy and pain points (represented as tags). This is reflected in our feature and row selection process.


### Data

<center>
<img src="{{ site.baseurl }}/assets/cfdrivers/data.png" width="70%" />
<br/>
</center>

The data for the modeling problem is the following.
The tags are represented as the design matrix $$X^{n \times m}$$, a binary
matrix with $$n$$ observations (the number of customer feedback) and $$m$$
features (the number of tags). If on open text is tagged with a specific tag,
the corresponding feature is $$1$$, otherwise $$0$$. The CSAT score is represented
as a vector $$y^{n \times 1}$$, with $$1$$ being "Excellent" and $$5$$ being "Poor"
(following the German school scoring system; note that we ignore the
unbalanced scale).


**Feature selection.** The purpose of this model is to focus on the customers'
most frequent tags. Because of this, a simple feature selection
method to reduce the dimensionality of the problem can be incorporated. We only
take the top-$$k$$ features, i.e., all the features $$x_{\cdot j}$$
($$j = 1, \ldots, m$$) where
$$\sum_{i=1}^{n} x_{ij} \leq T$$, and $$T$$ is the $$k$$th-largest value of the
ordered column sums. Currently, we use $$k = 20$$ and therefore monitor the 20
most frequent customers' joy and pain points.


**Rolling window.** The model should represent a middle-term key-driver analysis.
This means we are interested in customers' joy and pain points, which are
continuously existent for some time. To obtain such a model, we compute a
[rolling window model](http://stats.stackexchange.com/a/145654) over the last
$$w$$ weeks. Currently, we use a window of the last $$w = 6$$ weeks, which works
nicely with many of our product release schedules.


### Linear regression

To estimate the effect of the joy and pain points (represented as the tags), we
compute a [linear regression](https://en.wikipedia.org/wiki/Linear_regression)
(fitted with [OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares)):

<center>
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_m x_{im} + \epsilon_i,$$
</center>

with $$\epsilon_i \sim N(0, \sigma^2)$$ and $$i = 1, \ldots, n$$. To recap,
$$y_i$$ is the CSAT score given by the $$i$$th customer, and $$x_{i1}, \ldots,
x_{im}$$ are the $$i$$th customer's joy and pain points represented as binary tags.

Now, the interesting parts are the coefficients $$\beta_j$$ ($$j = 1, \ldots, m$$).
A specific coefficient $$\beta_j$$ tells us the expected change in the customer
satisfaction score ($$y$$) when the corresponding tag $$x_{\cdot j}$$ is changed
from 0 to 1 and all other features are held fixed. This is the estimated effect
of the key driver---how much does this joy or pain point contribute to the
customer satisfaction score.


**Bootstrapping.** To stabilize the estimated coefficients we apply
[case resampling](http://statweb.stanford.edu/~tibs/sta305files/FoxOnBootingRegInR.pdf) [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)). This
means, we do not compute one linear regression model but $$B$$ different models,
each one based on a random sample (with replacement) of the $$n$$ observations:

<center>
$$y^1_i = \beta^1_0 + \beta^1_1 x_{i1} + \beta^1_2 x_{i2} + \ldots + \beta^1_m x_{im} + \epsilon_i$$
$$\vdots$$
$$y^B_i = \beta^B_0 + \beta^B_1 x_{i1} + \beta^B_2 x_{i2} + \ldots + \beta^B_m x_{im} + \epsilon_i$$
</center>


The final coefficient $$\beta_j$$ is then defined as the mean value of all $$B$$
estimated coefficients: $$\beta_j = 1/n * \sum_{b=1}^B \beta_j^b$$.


**Visualization.** The coefficients $$\beta_j$$ are the estimated effect of the
key drivers on the customer satisfaction score and are visualized on the
$$y$$-axis of the Tableau dashboard presented above.



### Model evaluation

<center>
<img src="{{ site.baseurl }}/assets/cfdrivers/quality.png" width="100%" />
<br/>
</center>

To evaluate the quality of the estimated key drivers, we monitor the [adjusted
$$R^2$$](https://en.wikipedia.org/wiki/Coefficient_of_determination) value of the
linear regression model (or more exact, the mean adjusted $$R^2$$ of the
  bootstrapped models). The figure above shows a Tableau dashboard with the
quality of models for different Avira products over time. As you can see, the
adjusted $$R^2$$ ranges between $$0.05$$ and $$0.40$$. In a first evaluation phase
we observed that an $$R^2$$ value higher than $$0.10$$ is acceptable (it is a
real-world problem!), therefore we defined a color coding accordingly to this
rule of thumb. We also have different data quality rules in place, if these rules
are not met, no model is computed.

However, the **real evaluation** will be visible after a longer period. The models
provide reasonable estimations of key drivers if **our customers are happier**
because our products got better due to their feedback, and consequently, **our
revenue increases**.

<center>
<br/>
<img src="{{ site.baseurl }}/assets/cfdrivers/eval.png" width="70%" />
<br/>
</center>

***

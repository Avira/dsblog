---
layout: post
title:  "University @Avira: Predicting error-related user behavior in Avira Antivirus"
date:   2017-03-29
author: Manuel J. A. Eugster and Michaela Beckenbach
---

*The best way to learn Data Science is to do data science.* Following this motto,
Avira collaborated with the [University of Liechtenstein](https://www.uni.li)
and participated at the winter semester 2016/2017 seminar "Data Science" with a
real-world data science problem. **Liene Blija**, **Christian Holder**, **Jan Plojhar**,
and **Martin Lukšík**---four brave students---accepted the challenge to tackle
one of our prediction problems we face at Customer Insights Research[^1] at
[Avira](http://www.avira.com/).

[^1]: The students were supervised by [Manuel Eugster](http://mjae.net) and [Michaela Beckenbach](https://www.xing.com/profile/Michaela_Beckenbach).


# Challenge

The challenge at hand was to predict error-related user behavior in Avira Antivirus.
In more detail, the students got a sample data set of device specific error,
installation, and uninstallation events.

<center>
<br/>
<img src="{{ site.baseurl }}/assets/unili2016/events.png" width="70%" />
<br/>
</center>

The goal was to find patterns of error codes that lead to an uninstallation. The
figure above illustrates the idea: A series of error events with error code `5`
lead to an uninstallation of the product and the
churn of the user. On the other hand, error events with the code `1` lead to a
re-installation (if there is no leading error code `4`), whereas error
codes `2` and `3` do not lead to any user action. This is of course a very
simplified illustration and---as the students were about to find out---data
science reality is a bit more complex...

<center>
<br/>
<img src="{{ site.baseurl }}/assets/unili2016/da.png" width="100%" />
<br/>
</center>

The project covered the full [Data Science Lifecycle](https://blogs.technet.microsoft.com/machinelearning/2016/10/11/introducing-the-team-data-science-process-from-microsoft/).
Some of the challenges the student faced, were: First, event data is not ready for
classical predictive modeling and (clever) feature engineering plays a significant
role. Second, the data is highly imbalanced, meaning that only a small number of
cases really lead to an uninstallation. Third, their solutions should be fully
[reproducible](https://en.wikipedia.org/wiki/Reproducibility)
and, for example, easily applicable to a new sample data set spanning a different
time frame.

<center>
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">Students from <a href="https://twitter.com/uni_li">@uni_li</a> here at <a href="https://twitter.com/Avira">@Avira</a> talking and discussing about their data science project. Awesome! <a href="https://t.co/f0euEoGdNo">pic.twitter.com/f0euEoGdNo</a></p>&mdash; Manuel J. A. Eugster (@mjaeugster) <a href="https://twitter.com/mjaeugster/status/798492646142459906">November 15, 2016</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
</center>


# Results

<center>
<br/>
<img src="{{ site.baseurl }}/assets/unili2016/results2.png" width="100%" />
<br/>
</center>

The team used various methods to analyse the data from different angles. For a
first [exploratory analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
they used common statistical visualizations like time series plots and the
more sophisticated technique of
[process mining](https://en.wikipedia.org/wiki/Process_mining). Process mining
gave them a first idea if there are any combinations of events building some
common patterns (not necessarily leading to an uninstallation). Based on this
information and together with some domain knowledge,
potentially useful [features](https://en.wikipedia.org/wiki/Feature_engineering)
were engineered. To build a [predictive model](https://en.wikipedia.org/wiki/Predictive_modelling)
the team then benchmarked various linear and non-linear algorithms
([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression),
[support vector machines](https://en.wikipedia.org/wiki/Support_vector_machine),
[classification trees](https://en.wikipedia.org/wiki/Decision_tree_learning), and
[random forests](https://en.wikipedia.org/wiki/Random_forest)) on a training set
to find the best one. Their analyses showed that the random forest algorithm was
the best performing algorithm with an accuracy of about $$75\%$$ on a balanced
data set.


# Presentation

The students presented their final results at the
[second](https://www.meetup.com/4laendereck-Data-Science-Meetup/events/236241706/)
[4ländereck Data Science Meetup](https://www.meetup.com/4laendereck-Data-Science-Meetup/)
hosted by the [University of Liechtenstein](https://www.uni.li). The students'
work was a first proof-of-concept that showed us that it is possible to detect
patterns of error codes with a relationship to uninstallation. This is very
valuable information to us---supporting our idea of improving our products with,
for example, a proactive support system.

**We congratulate the students for mastering this data science project
successfully!**

<center>
<blockquote class="twitter-tweet" data-lang="en">
<p lang="en" dir="ltr">
<a href="https://twitter.com/hashtag/4leds?src=hash">#4leds</a> meetup with
<a href="https://twitter.com/hashtag/DataScience?src=hash">#DataScience</a> students
<a href="https://twitter.com/uni_li">@uni_li</a> presenting results based on
<a href="https://twitter.com/hashtag/Avira?src=hash">#Avira</a> data. Great job done! Congrats to the team!
<a href="https://t.co/ddKZGNK7bz">pic.twitter.com/ddKZGNK7bz</a></p>&mdash; Michaela Beckenbach (@MichiBeckenbach)
<a href="https://twitter.com/MichiBeckenbach/status/824719238178992130">January 26, 2017</a>
</blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
</center>

***

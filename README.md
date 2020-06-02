
# Statistical Distributions

![](images/distributions.png)

# Order of Business:
    
>    1. Describe the difference between discrete vs continuous variables
>    2. Describe the difference between PMFs, PDFs, CDFs
>    3. Introduce the bernouli and binomial distributions
>    4. Introduce the normal distribution and empirical rule

## What is a statistical distribution?

- A statistical distribution is a representation of the frequencies of potential events or the percentage of time each event occurs.

# Activation

![king_county](images/king_county.png)


<p> A probability distribution describes the probability of an event in a sample space.  We recently finished a project investigating opportunity youth in Seattle's King County.  Considering the idea that a distribution can be the probability of any variable, what interesting distributions did you come across?  For example, you most likely looked at the fraction of opportunity youth in South King County vs. youth that fall outside of that category.  The probability that a youth is an opportunity youth, and the corresponding probability that a youth is not an opportunity you, is probability distribution. </p>


#### Group Discussion Answers

\>  



# 1. Discrete vs Continuous

We will learn about a variety of different probability distributions, but before we do so, we need to establish the difference between **discrete** and **continuous** variables.

## Discrete
>  With discrete distributions, the values can only take a finite set of values.  Take, for example, a roll of a single six-sided die. 

![](images/uniform.png)

> - There are 6 possible outcomes of the roll.  In other words, 4.5 cannot be an outcome. As you see on the PMF plot, the bars which represent probability do not touch, suggesting non-integer numbers between 1 and 6 are not possible results.

#### Examples of discrete distributions:

> 1. The Uniform Distribution:- occurs when all possible outcomes are equally likely.
> 2. The Bernoulli Distribution: - represents the probability of success for a certain experiment (binary outcome).
> 3. The Binomial Distribution - represents the probability of observing a specific number of successes (Bernoulli trials) in a specific number of trials.
> 4. The Poisson Distribution:- represents the probability of 𝑛 events in a given time period when the overall rate of occurrence is constant.


## Continuous

With a continous distribution, the set of possible results is an infinite set of values within a range. One way to think about continuous variables are variables that have been measured.  Measurement can always be more precise.

> - A common example is height.  Although we think of height often in values such as 5 feet 7 inches, the exact height of a person can be any value within the range of possible heights.  In other words, a person could be 5 foot 7.000001 inches tall. 
> - Another example is temperature, as shown below:

![](images/pdf.png)

#### Examples of continuous distributions
> 1. Continuous uniform
> 2. The Normal or Gaussian distribution.
> 3. Exponential


The distinction between descrete and continuous is very important to have in your mind, and can easily be seen in plots. 

Let's do a quick exercise. There are two tasks.  

1. First, simply change the color of the plots representing descrete data to orange and the plots represent continous data to blue.
2. Attach the titles to the distributions you think reflect the data set described.


```python
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
```


```python

title_1 = "height_of_us_women in inches"
title_2 = 'result of flipping a coin 100 times'
title_3 = 'result of rolling a 20 sided dice 1000 times'
title_4 = 'the length of time from today a computer part lasts'
title_5 = 'probability that a picture is a chihauhua\n, a muffin, a bird, or a piece of pizza\n as would guess a neural network'
title_6 = 'probability of rolling a value equal to or below\n a certain number on a 20 sided dice'
no_title = 'no_title'

fig, ax = plt.subplots(2,3, figsize=(15,10))

sns.kdeplot(np.random.exponential(10, size=1000), ax=ax[0][0], color='purple')
ax[0][0].set_xlim(0,80)
ax[0][0].set_title(no_title)

sns.barplot(['outcome_1', 'outcome_2', 'outcome_3', 'outcome_4'], [.4,.5,.08,.02], ax=ax[1][0], color='yellow')
ax[1][0].tick_params(labelrotation=45)
ax[1][0].set_title(no_title)

sns.kdeplot(np.random.normal(64.5, 2.5, 1000), ax=ax[1][1])
ax[1][1].set_title(no_title)

sns.barplot(x=['outcome_1','outcome_2'], y=[sum(np.random.binomial(1,.5, 100)),100 - sum(np.random.binomial(1,.5, 100))], ax=ax[0][1], color='pink')
ax[0][1].set_title(no_title)

sns.barplot(x=list(range(1,21)), y=np.unique(np.random.randint(1,21,1000), return_counts=True)[1], ax=ax[0][2], color='teal')
ax[0][2].tick_params(labelrotation=45)
ax[0][2].set_title(no_title)

sns.barplot(list(range(1,21)), np.cumsum([1/20 for number in range(1,21)]), ax=ax[1][2])
ax[1][2].set_title(no_title)

plt.tight_layout()
```


![png](index_files/index_16_0.png)


# 2. PMFs, PDFs, and CDFs, oh my!

## PMF: Probability Mass Function


The $\bf{probability\ mass\ function\ (pmf)}$ for a random variable gives, at any value $k$, the probability that the random variable takes the value $k$. Suppose, for example, that I have a jar full of lottery balls containing:
- 50 "1"s,
- 25 "2"s,
- 15 "3"s,
- 10 "4"s

We then represent this function in a plot like so:


```python
# For each number, we calculate the probability that pull it from the jar by dividing

numbers = range(1,5)
counts = [50,25, 15, 10]

# calculate the probs by dividing each count by the total number of balls.

probs = [count/sum(counts) for count in counts]

lotto_dict = {number: prob for number,prob in zip(numbers, probs)}
lotto_dict
```




    {1: 0.5, 2: 0.25, 3: 0.15, 4: 0.1}




```python
# Plot here!

# x = range(1, 5)
# lotto_dict = {1: 0.5, 2: 0.25, 3: 0.15, 4:.1}
# y = [lotto_dict[num] for num in x]

x = list(lotto_dict.keys())
y = list(lotto_dict.values())

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(x, y, 'bo', ms=8, label='lotto pmf')
ax.vlines(x, 0, y, 'r', lw=5)
ax.legend(loc='best');
```


![png](index_files/index_21_0.png)


### Expected Value/Mean

The expected value, or the mean, describes the 'center' of the distribution (you may hear this called the first moment).  The 'center' refers loosely to the middle-values of a distribution, and is measured more precisely by notions like the mean, the median, and the mode.

For a discrete distribution, working from the vantage point of a collected sample of n data points:

mean = $\Large\mu = \frac{\Sigma^n_{i = 1}x_i}{n}$

If we are working from the vantage point of known probabilities, the mean is referred to as the expected value. The expected value of a discrete distribution is the weighted sum of all values of x, where the weight is their probability.
 
The expected value of the Lotto example is:
${\displaystyle \operatorname {E} [X]= \Sigma^n_{i=1}p(x_i)x_i}$

# Student input:
Help me calculate the expected value of the lotto example:



```python
# code
```

### Variance/Standard Deviation
Variance describes the spread of the data (it is also referred to as the second moment).  The 'spread' refers loosely to how far away the more extreme values are from the center.

Standard deviation is the square root of variance, and effectively measures the *average distance away from the mean*.

From the standpoint of a sample, the variance of a discrete distribution of n data points is:

std = $\Large\sigma = \sqrt{\frac{\Sigma^n_{i = 1}(x_i - \mu)^2}{n}}$


Variance is the expectation of the squared deviation of a random variable from its mean.

For our Lotto PMF, that means:

 $ \Large E((X-\mu)^2) = \sigma^2 = \Sigma^n_{i=1}p(x_i)(x_i - \mu)^2$

# Student input:
Help me calculate the variance for the Lotto Ball example



```python
# Code
```

## Uniform Distribution

The uniform distribution describes a set of discrete outcomes whose probabilities are all equally likely.

A common example is the roll of a die.  

![dice](https://media.giphy.com/media/3ohhwLh5dw0i7iLzOg/giphy.gif)

The pmf of a discrete uniform distribution is simply:

$ f(x)=\frac{1}{n} $

Let's take the example of a twelve-sided die.  

The probability for rolling any number, is 1/12.


```python
# expected value for a roll of a six-side die
expected_value = sum([1/12 * n for n in range(1,13)])
print(f'Expected value: {expected_value}')
# variance for a roll of a six-sided die
variance = sum([1/12 *(n - expected_value)**2 for n in range(1,13)])
print(f'Variance: {variance}')
```

    Expected value: 6.5
    Variance: 11.916666666666664


We can also calcalate the mean as follows:  
$\Large E(X)=\frac{a+b}{2}$

Where a is the lowest value and b is the highest. 




```python
# Let's check out that the two methods equal the same thing.
expected_value == (1+12)/2
```




    True



Variance can be calculated as follows:

$ \Large\sigma^2=\frac{(b-a+1)^2-1}{12} $


```python
# Again, let's check our math
round(variance,7) == round(((12-1+1)**2-1)/12, 7)
```




    True



![pear](https://media.giphy.com/media/fBS9UfNnOtkVDqR70I/giphy.gif)

# Short pair programming (2 minutes)
Create the pmf of a 12 sided die


```python
# Your code here

```

## PDF: Probability Density Function
> Probability density functions are similar to PMFs, in that they describe the probability of a result within a range of values.  But where PMFs can be descibed with barplots, PDFs are smooth curves.  

![](images/pdf_temp.png)



We can think of a pdf as a bunch of bars of probabilities getting smaller and smaller until each neighbor is indistinguishable from its neighbor.

It is then intuitive that you cannot calculate expected value and variance in the same way as we did with pmfs.  Instead, be have to integrate over the entirity of the curve to calculate the expected value.

### Expected value and variance for PDFs:
![](images/exp_v_pdf.png)


![](images/pdf_inter.png)

# Describing the PDF

Instead of calculating the mean and standard deviation by hand, we will rather get familiar with how they affect the shape of our PDF.


The mean of our PDF affects where it is centered on the x-axis.  In numpy and stats, mean is denoted by the loc parameter.

The two plots below have the same shape, but different centers.


```python
fig, ax = plt.subplots()

mean = 0
z_curve = np.linspace(stats.norm(mean,1).ppf(0.01),
             stats.norm(mean,1).ppf(0.99), 100)
ax.plot(z_curve, stats.norm(mean,1).pdf(z_curve),
     'r-', lw=5, alpha=0.6, label='z_curve')

mean = 1
z_curve = np.linspace(stats.norm(mean,1).ppf(0.01),
             stats.norm(mean,1).ppf(0.99), 100)
ax.plot(z_curve, stats.norm(mean,1).pdf(z_curve),
     'b-', lw=5, alpha=0.6, label='norm pdf')

ax.set_title("Two distributions differing only in mean")
```




    Text(0.5, 1.0, 'Two distributions differing only in mean')




![png](index_files/index_45_1.png)


The variance of our plots describes how closely the points are gathered around the mean.  Low variance means tight and skinny, high variance short and wide.


```python
# Mess around with the variance to see how the shape is altered.

fig, ax = plt.subplots()

mean = 1
var = 1
z_curve = np.linspace(stats.norm(mean,var).ppf(0.01),
             stats.norm(mean,var).ppf(0.99), 100)
ax.plot(z_curve, stats.norm(mean,var).pdf(z_curve),
     'r-', lw=5, alpha=0.6, label='z_curve')

mean = 1
var = 3
z_curve = np.linspace(stats.norm(mean,var).ppf(0.01),
             stats.norm(mean,var).ppf(0.99), 100)
ax.plot(z_curve, stats.norm(mean,var).pdf(z_curve),
     'b-', lw=5, alpha=0.6, label='norm pdf')

ax.set_title("Two distributions with different variance")
```




    Text(0.5, 1.0, 'Two distributions with different variance')




![png](index_files/index_47_1.png)


## Skew 

We will touch briefly on the third and fourth moments for the normal curve. Skew is a measure of assymemtry.  A skew of zero is perfectly symetrical about the mean.   
![skew](images/skew.png)


```python
# We can check skew with scipy
z_curve = np.random.normal(0,1, 1000)
print(stats.skew(z_curve))
```

    0.07098768308548697


To add right skew to the data, let's add some outliers to the left of the mean.

To learn about skew, let's take a normal distribution, and add values to skew it.


```python
# Update add right skew with data to skew it.
z_curve = np.random.normal(0,1, 1000)
add_right_skew = [0]
right_skewed_data = np.concatenate([z_curve, add_right_skew])

fig, ax = plt.subplots()
ax.hist(right_skewed_data)
ax.set_title(f"Right Skew {stats.skew(right_skewed_data)}");
```


![png](index_files/index_52_0.png)



```python
# Now, do the same for left skewed data

z_curve = np.random.normal(0,1, 1000)
add_left_skew = [0]
left_skewed_data = np.concatenate([z_curve, add_left_skew])

fig, ax = plt.subplots()
ax.hist(left_skewed_data)
ax.set_title(f"Left Skew {stats.skew(left_skewed_data)}");
```


![png](index_files/index_53_0.png)


### Transforming  Right/Positively Skewed Data

We may want to transform our skewed data to make it approach symmetry.

Common transformations of this data include 

#### Square root transformation:
Applied to positive values only. Hence, observe the values of column before applying.


#### The cube root transformation: 
involves converting x to x^(1/3). This is a fairly strong transformation with a substantial effect on distribution shape: but is weaker than the logarithm. It can be applied to negative and zero values too. Negatively skewed data.

#### The logarithm:
x to log base 10 of x, or x to log base e of x (ln x), or x to log base 2 of x, is a strong transformation and can be used to reduce right skewness.

## Left/Negatively Skewed Data

### Square transformation:
The square, x to x2, has a moderate effect on distribution shape and it could be used to reduce left skewness.
Another method of handling skewness is finding outliers and possibly removing them

## Pair: Report Back the effect of your transformation

Below, we have added some significant right skewed to the data by adding points between 2 and 4 standard deviations to to the right of the mean.

Apply each transformation mentioned above.
Hint: The data is in an array.  You can then feed the array into numpy functions, or broadcast a calculation across all elements.


```python
import numpy as np
from scipy import stats
no_skew_dist = np.random.normal(10,1, 1000)
add_right_skew = np.random.choice(np.random.normal(12,1,1000) , 100)
right_skewed_data = np.concatenate([no_skew_dist, add_right_skew])
print(f'Right Skew {stats.skew(right_skewed_data)}')

no_skew_dist_2 = np.random.normal(10,1, 1000)
add_left_skew = np.random.choice(np.random.normal(8,1,1000) , 100)
left_skewed_data = np.concatenate([no_skew_dist_2, add_left_skew])
print(f'Left Skew {stats.skew(left_skewed_data)}')
stats.skew(left_skewed_data)
```

    Right Skew 0.2878890743706462
    Left Skew -0.398050618316759





    -0.398050618316759



# Kurtosis

![kurtosis](images/kurtosis.png)


## CDF: Cumulative Distribution Function

![](images/cdf.png)

The cumulative distribution function describes the probability that your result will be of a value equal to or below a certain value. It can apply to both discrete or continuous functions.

For the scenario above, the CDF would describe the probability of drawing a ball equal to or below a certain number.  

In order to create the CDF from a sample, we:
- align the values from least to greatest
- for each value, count the number of values that are less than or equal to the current value
- divide that count by the total number of values

The CDF of the Lotto example plots how likely we are to get a ball less than or equal to a given example. 

Let's create the CDF for our Lotto example



```python
# align the values
lotto_dict = {0:0, 1:50, 2:25, 3:15, 4:10}
values = list(lotto_dict.keys())
# count the number of values that are less than or equal to the current value
count_less_than_equal = np.cumsum(list(lotto_dict.values()))
# divide by total number of values
prob_less_than_or_equal = count_less_than_equal/sum(lotto_dict.values()) 

```


```python
fig, ax = plt.subplots()
ax.plot(values, prob_less_than_or_equal, 'bo', ms=8, label='lotto pdf')
for i in range(0,5):
    ax.hlines(prob_less_than_or_equal[i], i,i+1, 'r', lw=5,)
for i in range(0,4):
    ax.vlines(i+1, prob_less_than_or_equal[i+1],prob_less_than_or_equal[i],  linestyles='dotted')
ax.legend(loc='best' )
ax.set_ylim(0);
```


![png](index_files/index_63_0.png)


# Pair Program
Taking what we know about cumulative distribution functions, create a plot of the CDF of a fair 12-sided die.

Take this in steps (no pun intended).
1. Create a list of possible rolls. 
2. Multiply the probability of each roll by the value of the roll.
3. Record the cumulative sum of each roll (hint: try np.cumsum()


```python
# Your Code Here
```

- For continuous random variables, obtaining probabilities for observing a specific outcome is not possible 
- Have to be careful with interpretation in PDF

We can, however, use the CDF to learn the probability that a variable will be less than or equal to a given value.



Consider the following normal distributions of heights (more on the normal distribution below).

The PDF and the CDF look like so:



```python
r = sorted(stats.norm.rvs(loc=70,scale=3,size=1000))
r_cdf = stats.norm.cdf(r, loc=70, scale=3)
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
sns.kdeplot(r, ax=ax1, shade=True)
ax1.set_title('PDF of Male Height in US')

ax2.plot(r, r_cdf, color='g')
ax2.set_title('CDF of Male Height in the US')


```




    Text(0.5, 1.0, 'CDF of Male Height in the US')




![png](index_files/index_68_1.png)


If we provide numpy with the underlying parameters of our distribution, we can calculate: 



```python
# the probability that a value falls below a specified value
r = stats.norm(70,3)
r.cdf(73)

```




    0.8413447460685429




```python
# the probability that a value falls between two specified values
r = stats.norm(70,3)
r.cdf(73) - r.cdf(67)

```




    0.6826894921370859



We can also calculate the value associated with a specfic percentile:


```python
r.ppf(.95)
```




    74.93456088085442



And from there, the value of ranges, such as the interquartile range:


```python
print(f'interquartile range {r.ppf(.25)} - {r.ppf(.75)}')

# We can see that the boxplot's interquartile range aligns with our ppf calculation above
box = plt.boxplot(stats.norm.rvs(loc=70,scale=3,size=1000));
print(box['boxes'][0].get_data())

```

    interquartile range 67.97653074941175 - 72.02346925058825
    (array([0.925, 1.075, 1.075, 0.925, 0.925]), array([68.00893806, 68.00893806, 72.09126527, 72.09126527, 68.00893806]))



![png](index_files/index_75_1.png)


# Common Discrete Distributions

# 3. Bernouli and Binomial Distributions

The Bernouli distribution is the discrete distribution that describes a two-outcome trial, such as heads or tails.  The distribution is described by the probability of one random variable of the value 1 associated with the probability p, and its correlary, the probability q, associated with 0  and taking the probability 1-p. 

PMF: 
${\displaystyle {\begin{cases}q=1-p&{\text{if }}k=0\\p&{\text{if }}k=1\end{cases}}}$

The simplest example is, once again, a coin flip.  In this scenario, we define either heads or tails as a "success", and assume, if the coin is fair, the probability of success to be .5

![](images/bernouli.png)

Another example would be a penalty kick in soccer.

![panenka](https://media.giphy.com/media/Jy1R6jdp8uXok/giphy.gif)

Let's assume the probability of scoring a goal is .75, the Bernouli distribution is:


```python
# probability of scoring
p = .75
# probability of missing
q = 1 -.75

fig, ax = plt.subplots()
ax.bar(['miss', 'score'],[q,p], color=['red','green'])
ax.set_title('Bernouli Distribution of Penalty Kicks')

```




    Text(0.5, 1.0, 'Bernouli Distribution of Penalty Kicks')




![png](index_files/index_81_1.png)


The expected value is the probability of success, i.e. **.75**
The variance is:  
$\sigma^2 = (.75)*(1-.75) = .1875 $

## Binomial

The Binomial distribution describes the number of successess of a set of bernouli trials. For example, say we have an unfair coin with a probability of landing heads of .8, if we designated our number of trials as 3, our PMF and CDF would look like what we see below:
![](images/binomial.png)

For the binomial our Expected Value and Variance can be calculated like so:
- Expected Value
> $E(X) = np$ <br>
- Variance
> $Var(X) = np(1-p)$<br>

If we want to see the probability of a certain number of successes, we use the pmf.
$\Large f(x) = {n \choose k}p^k(1 - p)^{n - k}$

Note: ${n\choose k} = \frac{n!}{k!(n - k)!}$, the number of ways of choosing $k$ objects from a total of $n$.

In our penalty kick example. Suppose we take for example a 10-kick penalty shoot after a extra time runs out.

The binomial distribution can tell me what the probability is that the shootout will result in exactly exactly $k$ goals out of $n$ shots ($k < n$).


```python
n = 10
p = 0.75
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
x = np.arange(stats.binom.ppf(0.001, n, p),
              stats.binom.ppf(.99, n, p)+1)

ax.plot(x, stats.binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
ax.vlines(x, 0, stats.binom.pmf(x, n, p), 'r', linewidth=5,
          label='pmf')
ax.legend(loc='best');
```


![png](index_files/index_86_0.png)


# Code along
What is the probability of a team scoring 7 goals in a shootout?



```python
mccalister = ['Adam', 'Amanda','Chum', 'Dann', 
 'Jacob', 'Jason', 'Johnhoy', 'Karim', 
'Leana','Luluva', 'Matt', 'Maximilian', ]
np.random.seed(42)
choice = np.random.choice(mccalister)
print(choice)
```

    Johnhoy



```python
# Start
n = None
k = None
p = None
```

# 4. Poisson Distribution

The Poisson distribution describes the probability of a certain number of a specific event occuring over a given interval. We assume that these events occur at a constant rate and independently.

Examples are:
- number of visitors to a website over an hour
- number of pieces of mail arriving at your door per day over a month
- number of births in a hospital per day


Shape of the Poisson Distribution is governed by the rate parameter lambda:

$\Large\lambda = \frac{Avg\ number\ of\ events}{period\ of\ time}$

${\displaystyle P(k)= {\frac {\lambda ^{k}e^{-\lambda }}{k!}}}$

Consider the scenario where a website receives 200 hits per hour.

The pmf of the Poisson distribution would be:



```python
rate = 40


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
x = np.arange(stats.poisson.ppf(0.1, rate),
              stats.poisson.ppf(.99, rate))

ax.plot(x, stats.poisson(rate).pmf(x), 'bo', ms=8, label='binom pmf')
ax.vlines(x, 0, stats.poisson(rate).pmf(x), 'r', linewidth=5,
          label='Poisson Distribution:\n Website Hits Over an Hour')
ax.legend(loc='best');
```


![png](index_files/index_94_0.png)


The Poisson distribution has a unique characteristic:
    
$\Large\mu = \sigma^2 = \lambda$

# Round Robin 

Northwestern Memorial is a very busy hospital.  The doctors there deliver, on average, 30 newborns per day.

Assume that newborns arrive at a constant rate and independently.

What is the probability of seeing exactly 40 newborns delivered on a given day.


```python
mccalister = ['Adam', 'Amanda','Chum', 'Dann', 
 'Jacob', 'Jason', 'Johnhoy', 'Karim', 
'Leana','Luluva', 'Matt', 'Maximilian', ]
```


```python
np.random.seed(42)
new_choice = np.random.choice(mccalister)
print(new_choice)
mccalister.remove(new_choice)
```

    Jacob



```python
# Code here
```

# 4. Normal Distribution

The normal distribution describes many phenomena. Think of anything that has a typical range:
- human body temperatures
- sizes of elephants
- sizes of stars
- populations of cities
- IQ
- Heart rate

Among human beings, 98.6 degrees Fahrenheit is an _average_ body temperature. Many folks' temperatures won't measure _exactly_ 98.6 degrees, but most measurements will be _close_. It is much more common to have a body temperature close to 98.6 (whether slightly more or slightly less) than it is to have a body temperature far from 98.6 (whether significantly more or significantly less). This is a hallmark of a normally distributed variable.

Similarly, there are large elephants and there are small elephants, but most elephants are near the average size.

The normal distribution is _very_ common in nature (**Why?**) and will arise often in your work. Get to know it well!

You will recognize it by its characteristic bell curve. 

![normal_curve](images/IQ_normal.png)

You may see the notation 

$ N(μ,σ2)$

where N signifies that the distribution is normal, μ is the mean, and σ2 is the variance. 


The PDF of the normal curve equals:

$\Large f(x) = \frac{1}{\sigma\sqrt{2\pi}}exp\left[\frac{-(x - \mu)^2}{2\sigma^2}\right]$



```python

fig, ax = plt.subplots()

mu = 0
sigma = 1
z_curve = np.linspace(stats.norm(mu,sigma).ppf(0.01),
             stats.norm(mu,sigma).ppf(0.99), 100)
ax.plot(z_curve, stats.norm(mu,sigma).pdf(z_curve),
     'r-', lw=5, alpha=0.6, label='z_curve')
```




    [<matplotlib.lines.Line2D at 0x1a1dcd6cf8>]




![png](index_files/index_104_1.png)


![](images/normal_2.png)

# Quick Solo Challenge

Turn off you cameras, turn them back on when you solved the problem, or when 1 minutes is up.

suppose the average height of an American woman is 65 inches with a standard deviation of 3.5 inches. 
Use numpy's random.normal to generate a sample of 1000 women and plot the histogram of the sample.



```python
# Code here
```


```python
fig, ax = plt.subplots()
ax.hist(np.random.normal(65, 3.5, 1000))
ax.set_title('Distribution of Heights of American Women')
ax.set_xlabel('Height in Inches');
```


![png](index_files/index_108_0.png)


# Standard Normal Distribution

A standard normal distribution has a mean of 0 and variance of 1. This is also known as a z distribution. 


![norm_to_z](images/norm_to_z.png)


```python
# Let's transform the normal distribtion centered on 5 with a standard deviation of 2 into a z curve
normal_dist = np.random.normal(5,2,1000)
z_dist = [(x - np.mean(normal_dist))/np.std(normal_dist) 
          for x in normal_dist]

fig, ax = plt.subplots()
sns.kdeplot(z_dist, ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1ed264e0>




![png](index_files/index_111_1.png)


![](images/empirical_rule.png)

## Empirical Rule
> The empirical or 68–95–99.7 states that 68% of the values of a normal distribution of data lie within 1 standard deviation of the mean, 95% within 2 stds, and 99.7 within three.  
> The empirical rule has countless applications in data science, which we will expand upon in the next few lectures.

By calculating the z-score of an individual point, we can see how unlikely a value is.

Consider, once again, the distribution of heights of American women, with a mean of 65 inches and a standard deviatio of 3.5 inches.

Calculate the zscore of a height of 75inches. 

Based on the empirical rule, if you were sampling heights of American women, speculate as to how improbable would that height be?


```python
# Your code here
```

# Exercise

Z score can be used to eliminate outliers.

For example, you may want to remove points that fall outside of 2.5 standard deviations of the mean.

In the diabetes dataset, the boxplot of bmi shows three outliers.


```python
from sklearn.datasets import load_diabetes
import pandas as pd

data = load_diabetes()
data.keys()
df = pd.DataFrame(data['data'])
df.columns = data['feature_names']

sns.boxplot(df['bmi'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1df426d8>




![png](index_files/index_119_1.png)


Using `stats.zscore`,remove all values that fall outside of  2.5 standard deviations on either side of the mean.


```python
# Your code here
```


```python

```

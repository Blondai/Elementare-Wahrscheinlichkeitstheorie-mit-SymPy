## A little Introduction 
This is my bachelor thesis for my degree in mathematics.
As I was studying in Germany the explanation stuff in the PDF file is done in german.
The Python files should also be understandable and usable for non german speakers.
(I think I did an okay job in naming variables, writing docstrings and so on.)

## The Goal
The goal of my thesis and therefore this project was making 'some' stochastics stuff with Python.

## The Story
As the goal was very broad, I was aimlessly reading some books about stochastics to get and idea about what I wanted to implement.
First I thought about [stochastic processes](https://en.wikipedia.org/wiki/Stochastic_process).
After a bit of digging and thinking about how to do it, I quickly realised that this was gonna be too hard.

So I thought about the most basic thing there is in [stochastics](https://en.wikipedia.org/wiki/Probability_theory).
The answer to my problem in finding the perfect topic for my thesis was [random variables](https://en.wikipedia.org/wiki/Random_variable).
These things are the building block of stochastics.

After selecting this topic I arrived at the first roadblock: How do you implement symbolic calculations in Python?!
A quick Google search revealed that there already is a package for symbolic math called [SymPy](https://www.sympy.org/en/index.html).
I promptly found out that to implement random variables in Python, I will most likely need a specific type of random variables which have a [density function](https://en.wikipedia.org/wiki/Probability_density_function).
This function allows easy calculation of [expected values](https://en.wikipedia.org/wiki/Expected_value) or more exotically [moment generating functions](https://en.wikipedia.org/wiki/Moment-generating_function).

Armed with this knowledge I started programming. And programming I did. After lots of trial and error I was to get the expected value of a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution).
Admittedly SymPy did a lot of the heavy lifting, mainly [integration](https://en.wikipedia.org/wiki/Integral) and later finding limits of [infinite series](https://en.wikipedia.org/wiki/Series_(mathematics)).
With this milestone achieved I started programming more exotic things like [characteristic functions](https://en.wikipedia.org/wiki/Characteristic_function_(probability_theory)), [cumulant generating function](https://en.wikipedia.org/wiki/Cumulant) and [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)).

After a while it was time to start writing the text part. This escalated very quickly and my introduction part was already over twenty pages long.
But [distribution functions](https://en.wikipedia.org/wiki/Distribution_function_(measure_theory)), [moments](https://en.wikipedia.org/wiki/Moment_(mathematics)) and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation) only scratched the tip of the iceberg.
So I asked my professor, and he replied that this part would already surffice for a bachelor thesis, but he won't restrict me.

This escalated even quicker and at the end I was at almost a hundred pages.

At the end I was awarded with the dream grade of 1.0, which was a perfect end to a good story.

## The Future
Currently, I am not planning to update this project.
Right at the end I thought that I could implement some ideas from the 'possible expansions' part, but I don't have any time for that.
I still think that this is a great project.
At least it could have some use in the stochastics part of a math degree, so you can cheat a little bit in the exercises.

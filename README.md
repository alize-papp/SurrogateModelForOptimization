
# Description of the project

Author: Alizé Papp (alize.papp@outlook.com)

## Technical description
Technical keywords: surrogate models, gaussian processes, kriging

This repository demos **methods to optimize a function when we know only a limited number of values it takes**. To do so, we must find model the function in the whole search space, using the values we know, and then optimize on the modeled function, using it as an estimate of the function we want to optimize. This is known as surrogate models.

One common technique is known as kriging, after [Danie Krige](https://en.wikipedia.org/wiki/Danie_G._Krige), a statistician and engineer who wanted to mine gold. The method can be used not only for gold-mining but also in biology, economics, marketing...


## A Real-life Application
Imagine a firm that provides access to ebooks. They have ebooks in all categories and they want to **encourage their customers to read more fiction and self-help books**. To do so, they decide to give some free credit to their customers, hoping after they use it they'll want to read more. They decided to give **120 free minutes of access to their fiction and self-help books to each customer**. By giving time for only the fiction and self-help categories, they should encourage their customers to discover new genres, but **how should they divide this free credit?** Should they give all 120 minutes for fiction books, or split 60 for fiction and 60 for self-help?

They ran some A/B testing on customers who had similar characteristics and **tested 12 different splits**. There were 1000 people per split. Of course, by testing all possible splits, they would end up finding the optimal one, but that would be costly and inefficient for all the customers who were in the suboptimal split. Then using only these 12 tests, how can we select the ideal split?

They want to know the best split under **two criteria**:
1. Optimizing the expected total reading time for fiction and self-help books
2. Optimizing the expected total reading time normalized by its standard deviation (known as Sharpe ratio)

## Dataset
I generated data in [this](notebooks/GenerateData.ipynb) notebook. I observed a few things, that were to be expected:
- the more people in a split we had, the better the estimate, 
- the more split, the better the estimate,
- when there are few splits, ie few values we know, the estimated function is very sensitive to the sampling scheme.

## File description
[data](data) The data I simulated.

[models](models) The different surrogate models. Not committed on the repo for space reasons.

[notebooks](notebooks) The notebooks with all analyses, surrogate models and optimizations.

[src](src) All reusable code

## Recommended Reading Order
Of course, you should first read this readme file, after that we advise reading the notebooks in the following order:
1. [Formalization](notebooks/FormalizationOfTheProblem.ipynb) Thoughs on the problem, notations and description of the method to solve it.
2. [Exploratory Data Analysis](notebooks/ExploratoryDataAnalysis.ipynb) First observation of the data. Intuitions that guided all choices for the modeling part hereafter.
3. Modeling and optimization strategies in the following notebooks:
   * [OptimizeWithRandomForestSurrogate](notebooks/OptimizeWithRandomForestSurrogate.ipynb)
   * [OptimizeWithSVRSurrogate](notebooks/OptimizeWithSVRSurrogate.ipynb)
   * [OptimizeWithGPSurrogate-separately](notebooks/OptimizeWithGPSurrogate-separately.ipynb)
   * [OptimizeWithGPSurrogate-TotalDirectly](notebooks/OptimizeWithGPSurrogate-TotalDirectly.ipynb)
4. [Generate data](notebooks/GenerateData.ipynb) If you are curious to know how the data was generated, or want to see what happens when you change some parameters

## Running the code
To be able to run the code on the repository:
1. Create a virtual environment (optional)
```
python -m venv venv/
source venv/bin/activate
```
2. Install python libraries by running in a terminal at the root of the repository:
```
pip install -r requirements.txt
```
You should be all set!

# Results

The following table sums up the results of the optimizations for all surrogate models:

| Notebook | Model | Optimum p for total reading (10^-2) | Maximized reading time (min, 10e0) | Optimum p for reading time / sd (10^-2) |
|-|-|-|-|-|
| [OptimizeWithGPSurrogate-TotalDirectly](notebooks/OptimizeWithGPSurrogate-TotalDirectly.ipynb) | Gaussian Process | 0 | 224 | 0.66 |
| [OptimizeWithRandomForestSurrogate](notebooks/OptimizeWithRandomForestSurrogate.ipynb) | Random Forest | 0.24 | 204 | 0.5 |
| [OptimizeWithSVRSurrogate](notebooks/OptimizeWithSVRSurrogate.ipynb) | SVR | 0 | 224 | 0.64 |


NB: the optimization with the Random Forest got stuck in a local extremum, an algorithm that left more room for the exploration of the space wouldn't have. The takeaway is for me is that except if we have good reason to except the underlying phenomenon to be best described by a tree-based method, Random Forests are not very good surogate models, because of their lack of convexity (and even monotonicity).

## First criteria: optimize expected reading

The [Exploratory Data Analysis](notebooks/ExploratoryDataAnalysis.ipynb) made it pretty clear the optimal allocation if optimizing only the expected reading time is to allocate all reading time to the self-help books. Further analyses with the surrogate only confirmed this.

The actual function:
![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/real.png)

Gaussian Process             |  Random Forest           | Support Vector Regression
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/GP.png)  |  ![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/RF.png)  | ![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/SVR.png)



## Second criteria: optimize expected reading standardized by its standard deviation

The optimization of the second criteria is more complex. The [Exploratory Data Analysis](notebooks/ExploratoryDataAnalysis.ipynb) suggests the demand for fiction books is more stable, so the optimal allocation is more balanced.

The actual function:
![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/real_sharpe.png)

Gaussian Process             |  Random Forest           | Support Vector Regression
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/GP_sharpe.png)  |  ![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/RF_sharpe.png)  | ![](https://github.com/alize-papp/SurrogateModelForOptimization/blob/main/images/SVR_sharpe.png)
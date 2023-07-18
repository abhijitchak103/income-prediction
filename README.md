# Insurance Premium Prediction README
-------------------------------------

### About the Repository
-------------------------------------

The repo contains the codebase for a machine learning prediction to calculate the expected premium for a customer given the basic information like `age`, `sex`, `bmi`, `children`, `region` and `smoker`. 

This project is an attempt at solving one of the iNeuron Internship projects. Link below:

[Insurance Premium Prediction](https://drive.google.com/file/d/1PUCqVKy21vtuKOYiBDhYJTUIa-7EP75g/view)

### About the Project
-------------------------------------

As the name of the repo suggests this projects into a basic prediction model where given the various inputs, the model predicts what the expected premium for the customer/user can be. 

### Learnings
-------------------------------------

- After implementing the project it was seen that out of different algorithms of `Linear Regression`, `Support Vector Machine`, `Decision Trees` and `Random Forest (RF)`, `RF` provided the highest accuracy on the test data. However, the difference between training and testing accuracy was large. Hence, to try toreduce the gap and reduce Overfitting. With `RandomizedSearchCV`, we were able to decrease the gap a bit, although a lot is yet to done.

- While deploying the project on AWS Beanstalk using CodePipeline, errors were encountered in setting up the cloud environment. With some basic search for EC2 instance setup, the model was successfully deployed.

- For the Frontend of the web app, CHATGPT came to the rescue. While developing the template files, basic usage of prompts to get a desired and decent output was achieved.

### Built With
--------------------------------------

The codebase has been built with the following languages and tools:

- [Python](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/index.html)

### Get Started
---------------------------------------

To use the project on your local systems use the following steps.

#### Prerequisites

There are no prerequisites to running the codebase.

#### Installation

1. Clone the repo
   
```
git clone https://github.com/abhijitchak103/insurance-premium-prediction.git
```
   
2. Create local virtual environment and activate the same

```
conda create -p venv python==3.9
conda activate venv/
```

3. Install the required packages
 
```
pip install -r requirements.txt
```

4. Run the app

```
python application.py
```

### Deployments
--------------------------------------------------------

User interface web-api has been deployed on AWS Beanstalk:
[Insurance Premium Predictor](http://insure-premium-env.eba-mimz6p2q.eu-north-1.elasticbeanstalk.com/)

### Next Steps
--------------------------------------------------------

Next steps would include 
- incorporating MLOps like MLFlow, DVC.
- increase database to introduce diversity.
- incorporate XGBoost, Boosting, Bagging to check whether we can increase accuracy.

### Feedback
---------------------------------------------------------

Each and every feedback is very much appreciated. It will help me implement better and increase efficiency in developing robust models.

Till then, take care and happy predicting....

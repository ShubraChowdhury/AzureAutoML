# Optimizing an ML Pipeline in Azure
# Note: My program Failed to perform solution using hyperdrive (See the Jupyter NoteBook udacity-project.ipynb)
## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset file name "bankmarketing_train.csv"  bank's campaigns . The marketing campaigns were based on phone calls to convince potenitial clients to subscribe to bank's .Potential solution was to predict if a prospect can become a custome.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The best performing model found using AutoML was a MaxAbsScaler LightGBM   with 91.44% accuracy, 
## My program Failed to perform solution using hyperdrive (See the Jupyter NoteBook udacity-project.ipynb)



## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
- Data link was provided by Udacity [Data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)
- Data was fetched using TabularDatasetFactory.from_delimited_files
- Data was cleaned using the clean_data(data) in train.py
- Data was split using sklearn.model_selection import train_test_split
- Data split ratio was training:testing::70:30
- Scikit-learn Logistic Regression ( from sklearn.linear_model import LogisticRegression) , RandomParameterSampling from (from azureml.train.hyperdrive.sampling import RandomParameterSampling) was used in addition to multiple hyperparameter (#"--learning_rate": normal(10,3), #"--keep_probability": uniform(0.05,0.1),#"--number_of_hidden_layers": choice(range(1,3)), "--P" : choice(0.01,0.1,1) ,   "--batch_size" : choice(32,64,128)) were used to test the solution. Batch Size with choice of [32,64,128] , Hidden layer with choice of [1,2,3], Learning Rate which returns a real value that's normally distributed with mean 10 and standard deviation 3, Keep Probablity that returns a value uniformly distributed between 0.05 and 0.1, P as inverse regularization C = 1/λ 
## Unfortunately my program Failed to perform solution using hyperdrive (See the Jupyter NoteBook udacity-project.ipynb) , even after using multiple combination , i lost time while debugging the program and that impacted the AutoML section as you can see I ran out of time prior to printing AutoML output , saving the automl model and deleteing the cluster (last 3 cell of my juypter notebook)
 

**What are the benefits of the parameter sampler you chose?**

As learned in the lessons I was expecting that random sampling over the hyperparameter search space using RandomParameterSampling in our parameter sampler would  reduces computation time and still find a reasonably models when compared to GridParameterSampling methodology where all the possible values from the search space are used.


**What are the benefits of the early stopping policy you chose?**
[BanditPolicy reference](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
For early stopping from BanditPolicy (azureml.train.hyperdrive.policy import BanditPolicy) was used , it takes evaluation_interval, slack_factor, slack_amount and delay_evaluation. 
Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.

- evaluation_interval: the frequency of applying the policy. Each time the training script logs the primary metric counts as one interval. An evaluation_interval of 1 will apply the policy every time the training script reports the primary metric. An evaluation_interval of 2 will apply the policy every other time. If not specified, evaluation_interval is set to 1 by default.
- delay_evaluation: delays the first policy evaluation for a specified number of intervals. This is an optional parameter that avoids premature termination of training runs by allowing all configurations to run for a minimum number of intervals. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.

- slack_factor or slack_amount: the slack allowed with respect to the best performing training run. slack_factor specifies the allowable slack as a ratio. slack_amount specifies the allowable slack as an absolute amount, instead of a ratio.

For example, consider a Bandit policy applied at interval 10. Assume that the best performing run at interval 10 reported a primary metric is 0.8 with a goal to maximize the primary metric. If the policy specifies a slack_factor of 0.2, any training runs whose best metric at interval 10 is less than 0.66 (0.8/(1+slack_factor)) will be terminated.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

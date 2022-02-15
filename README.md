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
[Reference](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)
- Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.

- The model used in this run are LightGBM, XGBoostClassifier, ExtremeRandomTrees, LogisticRegression, SGD,RandomForest.
- MaxAbsScaler LightGBM provided the best result with 91.44% accuracy.

Following configuration was used for AutoMl 
 AutoMLConfig(experiment_timeout_minutes=30, task='classification', primary_metric='accuracy',training_data=TRAIN_DATASET,label_column_name='y', n_cross_validations=4)

The cross validation checks overfitting and for computational reasons pre-defined timeout was set to 30 Minutes which limits number of Models that could be built.Model has Accuracy as primary metric.


## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
With AutoML the accuracy of best model MaxAbsScaler LightGBM was 91.44% accuracy and accuracy of HyperDrive model was inconclusive as my model errored out multiple times 

The architecture of both pipelines are different, however it follows the same process of  Load the data, instanciate the infrastructure to compute, set the parameters and call the compute method. Using AutoML allows many possibilities to increase the search for a better algorithm or a hyperparameter combination.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
AutoML generated the following alert:
TYPE:         Class balancing detection
STATUS:       ALERTED
DESCRIPTION:  To decrease model bias, please cancel the current run and fix balancing problem.
              Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
DETAILS:      Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class.

- In order to get over the above mentioned alert multiple other techniques may be used such as resampling training data, Adaptive Synthetic,Synthetic Minority Over-sampling Technique etc.


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

# I ran out of time prior to printing AutoML output , saving the automl model and deleteing the cluster (last 3 cell of my juypter notebook).

# Images
## Compute Instance Running
![Compute Instance Running](https://user-images.githubusercontent.com/32674614/153981412-4ad70cd7-490e-46bc-a78d-efff9e06cb2f.png)
## Compute Cluster
![Compute Cluster](https://user-images.githubusercontent.com/32674614/153981544-8d93f9a5-a3a0-4ae3-bc3d-de8418907ed4.png)
## Experiment
![Experiment](https://user-images.githubusercontent.com/32674614/153981589-301b812e-276d-47bb-9b67-6be02b1317d7.png)
## Hyperdrive Experiment Running
![Hyperdrive Experiment Running](https://user-images.githubusercontent.com/32674614/153981647-f208d9c7-742c-4b5d-92a9-296c8c018e38.png)
## Experiment With Hidden Layer & batch Size
![Experiment With Hidden Layer & batch Size](https://user-images.githubusercontent.com/32674614/153981700-7a529bfa-4b59-42cf-9aa0-3f84fa088fff.png)
## Environment
![Environment](https://user-images.githubusercontent.com/32674614/153981797-850eaa7d-a6be-4105-ad75-0f288450040b.png)
## DataStore
![DataStore](https://user-images.githubusercontent.com/32674614/153981835-c5076090-7af0-4194-b0b1-4b59505faa58.png)
## HyperDrive Error First Time
![HyperDrive Error First Time](https://user-images.githubusercontent.com/32674614/153981884-2e69a8ba-e95b-47e7-aad6-42cba2497429.png)
## Second Run with Hyperdrive
![Second Run with Hyperdrive](https://user-images.githubusercontent.com/32674614/153981980-f38c3c66-840e-4a29-93b1-6bd9665104b6.png)
## Run with IR and Batch Size
![Run with IR and Batch Size ](https://user-images.githubusercontent.com/32674614/153982040-90c6f1b3-a531-4dad-af94-979533e335c3.png)
## IR & BS Run Failed
![IR & BS Run Failed](https://user-images.githubusercontent.com/32674614/153982200-a8354e4e-56eb-4078-a8e4-063853c2bfdd.png)
## Third Run
![Third Run](https://user-images.githubusercontent.com/32674614/153982272-53836c09-3ba0-43b5-a59c-fd63339874fd.png)
## Third run later failed
![Third run later failed](https://user-images.githubusercontent.com/32674614/153982305-02421e98-edd9-49e1-995c-1c4e64a3ce64.png)


# References:
- how-to-configure-auto-train
[how-to-configure-auto-train](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)
- onnx_converte
[onnx_converter](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.onnx_convert.onnx_converter.onnxconverter?view=azure-ml-py)
- how-to-configure-auto-train
[how-to-configure-auto-train](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#configure-your-experiment-settings)
- Udacity lessons

















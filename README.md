# Optimizing an ML Pipeline in Azure
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

This dataset file name "bankmarketing_train.csv"  bank's campaigns . The marketing campaigns were based on phone calls to convince potenitial clients to subscribe to bank's product. Potential solution was to predict if a prospect can become a custome.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The best performing model found using AutoML was a VotingEnsemble    with 91.58% accuracy, while LogisticRegression has accuray of 91.077% using hyperdrive 
#


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
- Data link was provided by Udacity [Data](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)
- Data was fetched using TabularDatasetFactory.from_delimited_files
- Data was cleaned using the clean_data(data) in train.py
- Data was split using sklearn.model_selection import train_test_split
- Data split ratio was training:testing::70:30
- Scikit-learn Logistic Regression ( from sklearn.linear_model import LogisticRegression) , RandomParameterSampling from (from azureml.train.hyperdrive.sampling import RandomParameterSampling) was used  "--C" : choice(0.01,0.1,1) ,   "--max_iter" : choice(10,20,40), here  C as inverse regularization C = 1/λ  which had a choice between 0.01, 0.1 and 1 , max_iter is for max number of iteration which has a choice between 10 ,20 or 40.

 

**What are the benefits of the parameter sampler you chose?**

As learned in the lessons I was expecting that random sampling over the hyperparameter search space using RandomParameterSampling in our parameter sampler would  reduces computation time and still find a reasonably models when compared to GridParameterSampling methodology where all the possible values from the search space are used.


**What are the benefits of the early stopping policy you chose?**

BanditPolicy is used is an early stopping policy. It cuts more runs than a conservative policy like the MedianStoppingPolicy, hence saving the computational time significantly.

- BanditPolicy reference [BanditPolicy reference](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
For early stopping from BanditPolicy (azureml.train.hyperdrive.policy import BanditPolicy) was used , it takes evaluation_interval, slack_factor, slack_amount and delay_evaluation. 
Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.

- evaluation_interval: the frequency of applying the policy. Each time the training script logs the primary metric counts as one interval. An evaluation_interval of 1 will apply the policy every time the training script reports the primary metric. An evaluation_interval of 2 will apply the policy every other time. If not specified, evaluation_interval is set to 1 by default.
- delay_evaluation: delays the first policy evaluation for a specified number of intervals. This is an optional parameter that avoids premature termination of training runs by allowing all configurations to run for a minimum number of intervals. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.

- slack_factor or slack_amount: the slack allowed with respect to the best performing training run. slack_factor specifies the allowable slack as a ratio. slack_amount specifies the allowable slack as an absolute amount, instead of a ratio.

For example, consider a Bandit policy applied at interval 10. Assume that the best performing run at interval 10 reported a primary metric is 0.8 with a goal to maximize the primary metric. If the policy specifies a slack_factor of 0.2, any training runs whose best metric at interval 10 is less than 0.66 (0.8/(1+slack_factor)) will be terminated.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
- AutoML Reference [Reference](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)
- Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.

- The model used in this run are LightGBM, XGBoostClassifier, ExtremeRandomTrees, LogisticRegression, SGD,RandomForest.
- VotingEnsemble provided the best result with 91.58% accuracy.

Following configuration was used for AutoMl 
 AutoMLConfig(experiment_timeout_minutes=16, task='classification', primary_metric='accuracy',training_data=TRAIN_DATASET,label_column_name='y', n_cross_validations=4, ebable_early_stopping=True,enable_onnx_compatible_models=True)

The cross validation checks overfitting and for computational reasons pre-defined timeout was set to 15 Minutes (minimum allowed time is .25 hour and i used that) which limits number of Models that could be built.Model has Accuracy as primary metric.

### Following are AutoML Models and Accuracy.
![image](https://user-images.githubusercontent.com/32674614/154345574-eeabe14d-e656-4f0a-96dc-c6067d6b995a.png)



## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
With AutoML the accuracy of best model VotingEnsemble was 91.58% accuracy and accuracy of HyperDrive model was 91.077% 

The architecture of both pipelines are different, however it follows the same process of  Load the data, instanciate the infrastructure to compute, set the parameters and call the compute method. AutoML allows an increased search for a better algorithm or a hyperparameter combination.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
#### AutoML generated the following alert:
#### TYPE:         Class balancing detection
#### STATUS:       ALERTED
#### DESCRIPTION:  To decrease model bias, please cancel the current run and fix balancing problem.
              Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
#### DETAILS:      Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class.

- In order to get over the above mentioned alert multiple other techniques may be used such as resampling training data, Adaptive Synthetic,Synthetic Minority Over-sampling Technique etc.


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

- Deleting Cluster

![Deleting Cluster](https://user-images.githubusercontent.com/32674614/154344616-15012ea4-291e-4f02-9b2f-fca427ee6aee.png)

- Deleting Compute Instance

![image](https://user-images.githubusercontent.com/32674614/154344691-f6102a95-bbae-413d-8717-d5919639745a.png)



# References:
- how-to-configure-auto-train
[how-to-configure-auto-train](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)
- onnx_converte
[onnx_converter](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.onnx_convert.onnx_converter.onnxconverter?view=azure-ml-py)
- how-to-configure-auto-train
[how-to-configure-auto-train](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#configure-your-experiment-settings)
- Udacity lessons

















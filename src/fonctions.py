from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score,roc_auc_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
#import mlflow
#from mlflow.sklearn import log_model
#import tempfile
#import datetime
#from datetime import datetime as dt
#import os
#import warnings

import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_metrics
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
import os
import tempfile
import datetime
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')

def dupl(df):
    nDupl = df.duplicated().sum()
    print (f'Le dataset contient {nDupl} doublon(s)')   
   
def print_score(y_test,y_pred):
    '''Fonction permettant d'afficher les différents scores pertinents pour la classification'''
    print(f'Accuracy score = {accuracy_score(y_test, y_pred)}')
    print(f'Precision score = {precision_score(y_test, y_pred)}')
    print (f'Recall score = {recall_score(y_test,y_pred)}')
    print (f'F1 score = {f1_score(y_test,y_pred)}')
    print (f'ROC AUC score = {roc_auc_score(y_test,y_pred)}')   
    
def result(grid, log_target=0,transf_feat=0, features=''):
    ''' Fonction retournant un dataframe res recensant les différents résultats du GridSearchCv passé en fonction des paramétres utilisés'''
    res = pd.DataFrame(grid.cv_results_).round(2)
    cols = [i for i in res.columns if 'split' not in i ]
    res = res.loc[:,cols]
    ###
    
    res['log_target'] = log_target
    res['transf_feat'] = transf_feat
    res['features'] = features
    
    ###
    
    return res#.sort_values("", ascending =True)

def my_specificity_score(y_true,y_pred):
    """Metric to calculate specificity score. Used in GridSearchCV scoring"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    my_specificity = tn / (tn + fp)
    return my_specificity

def log_run(gridsearch: GridSearchCV, experiment_name: str, model_name: str, run_index: int, conda_env, tags={}): #
		"""
        Récupéré de liorshk/mlflow_gridsearch.py puis adapté pour mon notebook
        
        Logging of cross validation results to mlflow tracking server
		
		Args:
			experiment_name (str): experiment name
			model_name (str): Name of the model
			run_index (int): Index of the run (in Gridsearch)
			conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
			tags (dict): Dictionary of extra data and tags (usually features)
		"""
		
		cv_results = gridsearch.cv_results_
		with mlflow.start_run(run_name=str(run_index)) as run:  

			mlflow.log_param("folds", gridsearch.cv)

			print("Logging parameters")
			params = list(gridsearch.param_grid.keys())
			for param in params:
				mlflow.log_param(param, cv_results["param_%s" % param][run_index])

			print("Logging metrics")
			for score_name in [score for score in cv_results if "mean_test" in score]:
				mlflow.log_metric(score_name, cv_results[score_name][run_index])
				mlflow.log_metric(score_name.replace("mean","std"), cv_results[score_name.replace("mean","std")][run_index])

			print("Logging model")        
			log_model(gridsearch.best_estimator_, model_name, registered_model_name=model_name, conda_env=conda_env)

			print("Logging CV results matrix")
			tempdir = tempfile.TemporaryDirectory().name
			os.mkdir(tempdir)
			timestamp = dt.now().isoformat().split(".")[0].replace(":", ".")
			filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
			csv = os.path.join(tempdir, filename)
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				pd.DataFrame(cv_results).to_csv(csv, index=False)
			
			mlflow.log_artifact(csv, "cv_results") 

			print("Logging extra data related to the experiment")
			mlflow.set_tags(tags) 

			run_id = run.info.run_uuid
			experiment_id = run.info.experiment_id
			print(mlflow.get_artifact_uri())
			print("runID: %s" % run_id)
			mlflow.end_run()
   
def log_results(gridsearch: GridSearchCV, experiment_name, model_name, tags={}, log_only_best=False):
    
    """
    Récupéré de liorshk/mlflow_gridsearch.py puis adapté pour mon notebook
    
    Logging of cross validation results to mlflow tracking server
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        tags (dict): Dictionary of extra tags
        log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """
    conda_env = {
            'name': 'mlflow-env',
            'channels': ['defaults'],
            'dependencies': [
                'python=3.9.13',
                'scikit-learn>=0.21.3',
                {'pip': ['xgboost==1.7.5']}
            ]
        }

    best = gridsearch.best_index_
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment(experiment_name)

    if(log_only_best):
        log_run(gridsearch, experiment_name, model_name, best, conda_env, tags) 
    else:
        for i in range(len(gridsearch.cv_results_['params'])):
            log_run(gridsearch, experiment_name, model_name, i, conda_env, tags) 
        
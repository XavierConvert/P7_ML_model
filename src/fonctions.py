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

def decomposition_modele(best_model, X_train,X_test,y_train,):
    '''
    Fonction permettant de décomposer un modèle (pipeline) imbalanced avec d'un côté le preprocessing et d'un autre côté l'estimateur
    Le but étant de pouvoir gérer la feature importance en conservant le nom des colonnes de X_train
    
    Arguments:
    - best_model: de type pipeline
    - X_train
    - X_test
    - y_train (necessaire pour le resampling)
    '''
        
    # Si modèle imbalanced:
    
    if str(best_model[0]) == 'RandomUnderSampler()':
        
        #Sampling
        
        X_tr, y_train_rus=best_model[0].fit_resample(X_train,y_train)
    
        # Preprocessing X_train
    
        X_tr=best_model[1:-1].fit_transform(X_tr)
        X_tr_transf=pd.DataFrame(X_tr, columns=X_train.columns)
    
        # Preprocessing X_test
    
        X_te_transf=best_model[1:-1].fit_transform(X_test)
        X_te_transf=pd.DataFrame(X_te_transf, columns=X_train.columns)
        
    else: # si classes équilibrées
        
        y_train_rus = y_train
        X_tr=best_model[:-1].fit_transform(X_train)
        X_tr_transf=pd.DataFrame(X_tr, columns=X_train.columns)
    
        # Preprocessing X_test
    
        X_te_transf=best_model[:-1].fit_transform(X_test)
        X_te_transf=pd.DataFrame(X_te_transf, columns=X_train.columns)
    
       
    return X_tr_transf,X_te_transf, y_train_rus
    
    # Entrainement estimateur? A faire en dehors de la fonction
    
    #return X_tr,B,best_model[-1].fit(X_tr,B)

def calcul_gain_unit(ser, lost_coeff =0.3, taux=0.05): #y_true, y_pred
    
    '''Le but de cette fonction est de calculer le différentiel de gain grâce au modèle de prédiction comparé à la réalisation dans la vraie vie'''
        
    if ser['predictions']: # ==1 , pas de prêt
        if ser['y_true']: #==1; on ne perd plus d'argent grace au modèle de prédiction: gain
            return round(ser['AMT_CREDIT']-lost_coeff*ser['AMT_GOODS_PRICE'],2)
        else: # y_true = 0: on a refusé le crédit à tort: manque à gagner = interets non perçus = M*i*n (interets simples)
            return round(-(ser['AMT_CREDIT']*taux*ser['nb_annuité']),2)
    if not ser['predictions']: #==0, prêt accordé (pas de différence avec la réalité)
        return 0 
        #if y_true: #==1, prêt non remboursé, donc prêt accordé à tort=> perte = M-(30% * PP)
        #    return -(ser['AMT_CREDIT']-lost_coeff*ser['AMT_GOODS_PRICE'])
        #elif: # y_true = 0 pas de différence : le gain est le même avec et sans modèle de prédiction
        #    return 0
        
def calcul_gain_total(df):
    df['Gain']=0
    for i in range(len(df)):
        #calcul_gain_unit(df.iloc[i].T)
        df['Gain'].iloc[i]=calcul_gain_unit(df.iloc[i].T)
    return round(df['Gain'].sum(),2)

def calcul_gain_gross(ser, lost_coeff =0.3, taux=0.05): #y_true, y_pred
    
    if ser['predictions']:
        return 0
    
    if not ser['y_true']: # == 0 > prêt accordé et remboursé: gains = interets
        return round(ser['AMT_CREDIT']*taux*ser['nb_annuité'],2)
    
    if ser['y_true']: # ==1 prêt accordé à tort : perte K  - vente des biens
        return round(-ser['AMT_CREDIT']-lost_coeff * ser['AMT_GOODS_PRICE'],2)      
    
def calcul_gain_gross_total(df):
    gain=[]
    gain_total = 0
    for i in range(len(df)):
        gain.append(calcul_gain_gross(df.iloc[i].T))
        
        #df['Gain_net'].iloc[i]=calcul_gain_unit(df.iloc[i].T)
        gain_total+=gain[i]
    return round(gain_total,2)    

def calcul_gain_proba_unit(ser,seuil=0.50, lost_coeff =0.3, taux=0.05): 
    """ Sur base predict_proba du modèle retenu, fonction permettant le gain financier sur un crédit accordé.
    Arg:
    - ser: proba_0 de predict_proba
    - seuil: défualté à 0.5. Sera ajusté via la fonction calcul_seuil_optimal()
    - lost_coeff: coeff à appliquer au prix du bien financé: sert à minimiser la perte en cas de non remboursement du crédit
    - taux (= 0.05 par défaut): taux d'interet fictif du crédit
    """    
    if ser['proba_0'] < seuil:
        return 0    
            
    #if ser['predictions']:
    #    return 0
        
    if not ser['y_true']: # == 0 > prêt accordé et remboursé: gains = interets
        #Version où les interets sont perçus sur toutes les annuités
        #return round(ser['AMT_CREDIT']*taux*ser['nb_annuité'],2)
        
        #Version où l'on ne perçoit les int qu'une seule fois
        return round(ser['AMT_CREDIT']*taux,2)
    
    if ser['y_true']: # ==1 prêt accordé à tort : perte K  - vente des biens
        return round(-ser['AMT_CREDIT']-lost_coeff * ser['AMT_GOODS_PRICE'],2)
       
def calcul_seuil_optimal(df):
    """FOnction permettant de calculer le seuil à partir duquel on doit considérer 
    que l'on perd de l'argent si un crédité est accordé à tort
    
    Basé sur la fonction calcul_gain_proba_unit décrite plus haut """ 
    
    best_thresh=pd.DataFrame(columns=['threshold','Gain_total'])
       
    for thr,j in zip(np.arange(0,1,0.05),range(len(np.arange(0,1,0.05)))): 
        gain=[]
        gain_total = 0
        
        for i in range(len(df)):
            gain.append(calcul_gain_proba_unit(df.iloc[i].T,thr,taux=0.04,lost_coeff=0.3))
            gain_total+=gain[i]
        
        tmp=pd.DataFrame([[round(thr,2),gain_total]],columns=['threshold','Gain_total'])
        
        best_thresh=pd.concat([best_thresh,tmp],axis=0,ignore_index=True)
        
    return best_thresh #gain_total

def my_specificity_score(y_true,y_pred):
    """Custom metric to calculate specificity score. Used in GridSearchCV scoring"""
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
        log_run(gridsearch, experiment_name, model_name, best, tags) 
    else:
        for i in range(len(gridsearch.cv_results_['params'])):
            log_run(gridsearch, experiment_name, model_name, i, conda_env, tags) 
        
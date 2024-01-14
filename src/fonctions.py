from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score,roc_auc_score
import pandas as pd
import numpy as np

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
    - modele: de type pipeline
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
    best_thresh=pd.DataFrame(columns=['threshold','Gain_total'])
    #global best_thresh
    
    for thr,j in zip(np.arange(0,1,0.05),range(len(np.arange(0,1,0.05)))): 
        gain=[]
        gain_total = 0
        
        for i in range(len(df)):
            gain.append(calcul_gain_proba_unit(df.iloc[i].T,thr,taux=0.04,lost_coeff=0.3))
            gain_total+=gain[i]
        
        tmp=pd.DataFrame([[round(thr,2),gain_total]],columns=['threshold','Gain_total'])
        
        best_thresh=pd.concat([best_thresh,tmp],axis=0,ignore_index=True)
        
    return best_thresh #gain_total
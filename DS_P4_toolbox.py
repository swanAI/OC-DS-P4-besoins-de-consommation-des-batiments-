import pandas as pd 
import seaborn as sns 
import numpy as np 

import matplotlib.pyplot as plt

from IPython.display import Markdown

from tabulate import tabulate
from IPython.display import Markdown
from termcolor import colored
from google.colab import data_table
from scipy.stats import norm
from scipy.stats import shapiro
import scipy.stats as stats
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot

# fonction pour detecter les doublons dans plusieurs df 
def detecte_doublons(df):
  print("Les doublons dans le df :",len(df[df.duplicated()]))



#Fonction pour réduire la mémoire du df     
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


# fonction pour détecter NaN dans les colonnes 
def NaN_columns_df(df):
    
        
        dtypes = df.dtypes
        missing_count = df.isnull().sum()
        value_counts = df.isnull().count()
        missing_pct = (missing_count/value_counts)*100
        missing_total = df.isna().sum().sum()
        pct_NaN_total_in_dataset= (missing_total/df.size)*100
        df_missing = pd.DataFrame({'Count_NaN':missing_count, '%_NaN_col':missing_pct, 'Total_NaN_in_dataset': missing_total, '%_NaN_in_dataset':pct_NaN_total_in_dataset,'Types':dtypes})
        df_missing = df_missing.sort_values(by='%_NaN_col', ascending=False)
        
        print('Les valeurs manquantes dans chaque colonne :')
        print('')
        print(tabulate(round(df_missing,2),headers='keys',tablefmt='pretty'))
        
        plt.style.use('ggplot')
        plt.figure(figsize=(28,15))
        plt.title(f'Le pourcentage de valeurs manquantes pour les colonnes', size=25)
        plt.plot( df.isna().sum()/df.shape[0])
        plt.xticks(rotation = 90) 
        plt.xticks(fontsize=18) 
        plt.yticks(fontsize=20)
        plt.show()
        print('')
        print('----------------------------------------------------------------------------------')
        print('')


    # Graphique pour voir le pourcentage de valeurs manquantes pour les colonnes 
def plot_pourcentage_NaN_features(df):
    plt.style.use('ggplot')
    plt.figure(figsize=(20,18))
    plt.title('Le pourcentage de valeurs manquantes pour les features', size=20)
    plt.plot((df.isna().sum()/df.shape[0]*100).sort_values(ascending=True))
    plt.xlabel('Features dataset', fontsize=18)
    plt.ylabel('Pourcentage NaN dans features', fontsize=18)
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.show()

    pct_dataset = pd.DataFrame((df.isna().sum()/df.shape[0]*100).sort_values(ascending=False))
    pct_dataset = pct_dataset.rename(columns={0:'Pct_NaN_colonne'})
    pct_dataset = pct_dataset.style.background_gradient(cmap='YlOrRd')
    return pct_dataset
        

def NaN_rows (df):
    
    nb_cols = pd.DataFrame( df.T.isnull().count()).rename(columns={0:'Nb_cols'})
    missing_count_rows = pd.DataFrame(df.T.isnull().sum()).rename(columns={0:'Count_NaN'})
    pct_rows_NaN = pd.DataFrame((df.T.isnull().sum()/df.T.isnull().count()*100)).rename(columns={0:'Pct_NaN_rows'})
    df_rows_NaN = pd.concat([missing_count_rows,nb_cols, pct_rows_NaN], axis=1)
    df_rows_NaN = df_rows_NaN.sort_values(by=['Pct_NaN_rows'], ascending=False)
    df_rows_NaN         
   

# fonction afficher les valeurs uniques pour chaque colonne 
def unique_multi_cols(df):
  
  for col in list(df.columns):
    pct_nan = (df[col].isna().sum()/df[col].shape[0])
    unique = df[col].unique()
    nunique = df[col].nunique()
  
    print('')
    print(colored(col, 'red'))
    print('') 
    print((f'Le pourcentage NaN : {pct_nan*100}%'))
    print(f'Nombre de valeurs unique : {nunique}')
    print('')
    print(unique)
    print('')
    print('---------------------------------------------------------------------------------------')

# Fonction plot pour afficher distribution des variables afin de voir normalité et outliers
import scipy.stats as stats
def diagnostic_plots(df, variable):
    

    
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    print('Test Shapiro')
    data = df[variable].values
    stat, p = shapiro(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
         print('Probablement Gaussien ')
    else:
         print('Probablement pas  Gaussien ')
            
    print('Test normaltest')        
    data = df[variable].values
    stat, p = normaltest(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probablement Gaussien ')
    else:
        print('Probablement pas  Gaussien ')


    plt.show()



def detect_outliers_IQR(var):
    """Fonction pour détecter les outliers
    """
    Q1 = np.quantile(var, 0.25)  
    Q3 = np.quantile(var, 0.75)
    EIQ = Q3 - Q1
    LI = Q1 - (EIQ*1.5)
    LS = Q3 + (EIQ*1.5)    
    i = list(var.index[(var < LI) | (var > LS)])
    val = list(var[i])
    return i, val


# fonction afficher relation entre variable numérique et object
def boxplot_bivariée(df, var_cat, var_num):
  meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}
  plt.figure(figsize=(16,12))
  plt.title(f"Boxplot entre {var_cat} et {var_num}", size=22)
  sns.boxplot(x=df[var_cat], y=df[var_num], showmeans=True, meanprops=meanprops,data=df)
  plt.xticks(fontsize=9)


def evaluation(model, name):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
   
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=5, scoring='r2',
                                               train_sizes=np.linspace(0.25, 1.0, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Learning curve {name}', size=16)
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N,val_score.mean(axis=1), label='validation score')
    plt.xlabel('Training examples')
    plt.ylabel('R2 score')
    plt.legend()

    
    
def evaluation_CO2(model, name):
    
    model.fit(X_train_CO2, y_train_CO2)
    ypred = model.predict(X_test)
    
   
    
    N, train_score, val_score = learning_curve(model, X_train_CO2, y_train_CO2,
                                              cv=5, scoring='r2',
                                               train_sizes=np.linspace(0.25, 1.0, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Learning curve {name}', size=16)
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N,val_score.mean(axis=1), label='validation score')
    plt.xlabel('Training examples')
    plt.ylabel('R2 score')
    plt.legend()




def plot_features_importance(estimator, name_model, X_train, y_train, scoring=None):
    """
    Generate 1 plots: 
        1. The importance by feature
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.     
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning. 
        
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable. 
        If None, the estimator’s default scorer is used. 
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """     
    # Get the importance by feature
    results = permutation_importance(estimator, X_train, y_train, scoring=scoring)
    
    # Making a dataframe to work easily
    df_importance = pd.DataFrame({
                        "Feature" : X_train.columns,
                        "Importance" : results.importances_mean
                    })
    
    # Sorting by importance before plotting
    df_importance = df_importance.sort_values("Importance")
    
    # Initializing figure    
    fig = plt.subplots(figsize=(12, 8))
    
    plot = sns.barplot(data=df_importance, y=df_importance["Feature"], x=df_importance["Importance"])
    
    plt.title(name_model + " Features Importance", fontdict={ "fontsize": 16, "fontweight": "normal" })
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()



def plot_features_importance_CO2(estimator, name_model, X_train, y_train, scoring=None):
    """
    Generate 1 plots: 
        1. The importance by feature
    
    Parameters
    -----------------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    name_model : str
        Name of the model as title for the chart.     
        
    X_train : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y_train : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning. 
        
    scoring : string, callable or None, default=None
        Scorer to use. It can be a single string or a callable. 
        If None, the estimator’s default scorer is used. 
        
    Returns:
    -----------------
        None. 
        Plot the graph. 
        
    """     
    # Get the importance by feature
    results = permutation_importance(estimator, X_train_CO2_no_ESS, y_train_CO2, scoring=scoring)
    
    # Making a dataframe to work easily
    df_importance = pd.DataFrame({
                        "Feature" : X_train_CO2_no_ESS.columns,
                        "Importance" : results.importances_mean
                    })
    
    # Sorting by importance before plotting
    df_importance = df_importance.sort_values("Importance")
    
    # Initializing figure    
    fig = plt.subplots(figsize=(12, 8))
    
    plot = sns.barplot(data=df_importance, y=df_importance["Feature"], x=df_importance["Importance"])
    
    plt.title(name_model + " Features Importance", fontdict={ "fontsize": 16, "fontweight": "normal" })
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
"""Reproducible Ridge baseline and Optuna XGBoost experiment."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

SEED=42
NUMERIC=["GrLivArea","TotalBsmtSF","GarageArea","YearBuilt","OverallQual"]
CATEGORICAL=["MSZoning"]
FEATURES=NUMERIC+CATEGORICAL
DATA=Path(__file__).parent/'data'/'ames_train.csv'

def metrics(model,x,y):
    pred=model.predict(x)
    return {'r2':float(r2_score(y,pred)),'mae':float(mean_absolute_error(y,pred)),
            'rmse':float(np.sqrt(mean_squared_error(y,pred))),
            'mape':float(mean_absolute_percentage_error(y,pred))}

def prep(scale=False):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median')),
                      *(([('scale',StandardScaler())]) if scale else [])])
    return ColumnTransformer([('num',numeric,NUMERIC),('cat',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)])

def ridge():
    return Pipeline([('prep',prep(True)),('model',Ridge(alpha=1.0))])

def xgb(params):
    return Pipeline([('prep',prep()),('model',TransformedTargetRegressor(
        regressor=XGBRegressor(**params,random_state=SEED,n_jobs=1,objective='reg:squarederror'),
        func=np.log1p,inverse_func=np.expm1))])

def main():
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME','house-price'))
    df=pd.read_csv(DATA)
    df.columns=df.columns.str.strip()
    df=df[FEATURES+['SalePrice']].dropna(subset=['SalePrice'])
    x,y=df[FEATURES],df['SalePrice']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2,random_state=SEED)
    dataset=mlflow.data.from_pandas(df,source=str(DATA),targets='SalePrice',name='ames-housing')
    stats={'rows':len(df),'missing_fraction':df.isna().mean().to_dict(),
           'numeric_summary':df[NUMERIC+['SalePrice']].describe().to_dict(),
           'target_skew':float(y.skew()),'log_target_skew':float(np.log1p(y).skew())}
    fig,ax=plt.subplots(1,2,figsize=(9,4))
    ax[0].hist(y,bins=35);ax[0].set_title('SalePrice')
    ax[1].hist(np.log1p(y),bins=35);ax[1].set_title('log1p(SalePrice)')
    fig.tight_layout()
    with mlflow.start_run(run_name='ridge-baseline') as run:
        mlflow.log_input(dataset,context='train')
        model=ridge();model.fit(xtrain,ytrain)
        mlflow.log_params({'model':'ridge','alpha':1.0,'seed':SEED})
        mlflow.log_metrics(metrics(model,xtest,ytest))
        mlflow.sklearn.log_model(model,name='model',signature=infer_signature(xtrain,model.predict(xtrain)))
        baseline_run=run.info.run_id
    cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
    with mlflow.start_run(run_name='optuna-search'):
        def objective(trial):
            params={'n_estimators':trial.suggest_int('n_estimators',200,1000),
                'max_depth':trial.suggest_int('max_depth',2,8),
                'learning_rate':trial.suggest_float('learning_rate',.01,.3,log=True),
                'subsample':trial.suggest_float('subsample',.6,1.),
                'colsample_bytree':trial.suggest_float('colsample_bytree',.6,1.),
                'reg_lambda':trial.suggest_float('reg_lambda',0.,10.)}
            with mlflow.start_run(run_name=f'trial-{trial.number}',nested=True):
                score=float(cross_val_score(xgb(params),xtrain,ytrain,cv=cv,scoring='r2',n_jobs=1).mean())
                mlflow.log_params(params);mlflow.log_metric('cv_r2',score)
            return score
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective,n_trials=max(20,int(os.getenv('OPTUNA_TRIALS','20'))))
    with mlflow.start_run(run_name='xgboost-final') as run:
        model=xgb(study.best_params);model.fit(xtrain,ytrain)
        mlflow.log_input(dataset,context='train')
        mlflow.log_params({'model':'xgboost','seed':SEED,**study.best_params})
        mlflow.log_metrics({**metrics(model,xtest,ytest),'row_count':float(len(df)),
                            'target_skew':stats['target_skew'],'log_target_skew':stats['log_target_skew'],
                            'best_cv_r2':float(study.best_value)})
        mlflow.log_dict(stats,'data_stats.json')
        mlflow.log_figure(fig,'target_distributions.png')
        sample=xtrain.sample(min(len(xtrain),500),random_state=SEED)
        sample.to_parquet('/tmp/reference_data.parquet',index=False)
        mlflow.log_artifact('/tmp/reference_data.parquet')
        mlflow.sklearn.log_model(model,name='model',signature=infer_signature(xtrain,model.predict(xtrain)))
        final_run=run.info.run_id
    client=MlflowClient()
    # In MLflow 3, log_model(name=...) yields a logged model; runs:/run_id/model resolves its source.
    baseline=mlflow.register_model(f'runs:/{baseline_run}/model','house-price')
    version=mlflow.register_model(f'runs:/{final_run}/model','house-price')
    client.set_registered_model_alias('house-price','prod',version.version)
    print(json.dumps({'baseline_version':baseline.version,'prod_version':version.version,
                      'baseline_run':baseline_run,'final_run':final_run,'best_cv_r2':study.best_value}))
if __name__=='__main__':main()

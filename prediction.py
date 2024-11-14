import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import statsmodels.formula.api as sm
import pickle

def target_label(df):

    df['default_year'] = pd.to_datetime(df['def_date'], format="%d/%m/%Y", errors='coerce').dt.year
    df['target'] = ((df['default_year'].notna()) & (df['fs_year'] + 1 >= df['default_year'])).astype(int)

    return df

def null_imputation(data):
    df = data.copy()

    df = df.replace([float('inf'), -float('inf')], float('nan'))
    df['legal_struct_encoded'] = LabelEncoder().fit_transform(df['legal_struct'])
    features = df[['asst_tot', 'legal_struct_encoded', 'ateco_sector']].fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=10, random_state=42) 
    df['cluster'] = kmeans.fit_predict(scaled_features)

    for column in df.columns:
        if df[column].isnull().any():
            df[column] = df.groupby('cluster')[column].transform(lambda x: x.fillna(x.median()))

    return df


def features_engineering(df):
    df['def_date'] = pd.to_datetime(df['def_date'], dayfirst=True)
    df['stmt_date'] = pd.to_datetime(df['stmt_date'])
 
    # leverage
    df['td_ta'] = (df['asst_tot'] - df['eqty_tot']) / df['asst_tot']
    df['td_te'] = (df['asst_tot'] - df['eqty_tot']) / df['eqty_tot']
    df['td_ebitda'] = (df['asst_tot'] - df['eqty_tot']) / df['ebitda']
    # profitability
    df['operating_margin'] = df['prof_operations'] / df['rev_operating']
    df['earning_power'] = df['ebitda'] / df['asst_tot']
    # liquidity
    df['Liquidity'] = df['cash_and_equiv'] / (df['asst_tot'])
    df['current_ratio'] = df['asst_tot']/(df['asst_tot'] - df['wc_net'])
    df['cash_ratio'] = df['cash_and_equiv'] / (df['asst_tot'] - df['wc_net'])
    # debt coverage
    df['Debt_coverage'] = df['cf_operations'] / df['exp_financing']

    return df

def standardize(df):

    df['asst_tot_log'] = np.log(df['asst_tot'])

    columns_to_standardize = ['roa','td_ta','current_ratio','Debt_coverage', 'asst_tot_log']

    scaler = StandardScaler()

    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df


def preprocessor(data):

    df = data.copy()

    df_labeled = target_label(df)

    df_engineered = features_engineering(df_labeled)

    df_drop_na =  null_imputation(df_engineered[["fs_year",'target','roa','td_ta','current_ratio'\
                                                 ,'Debt_coverage', 'asst_tot',"legal_struct","ateco_sector"]])

    df_standardized = standardize(df_drop_na)
    
    final_columns = ["fs_year",'target','roa','td_ta','current_ratio','Debt_coverage', 'asst_tot_log']
    final_df = df_standardized[final_columns]
   
    return final_df

    
def predictor_harness(new_df, model, preprocessor, output_csv):
    
    preprocessed_data = preprocessor(new_df)
    predictions = model.predict(preprocessed_data)

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False, header=False)

    return output_csv
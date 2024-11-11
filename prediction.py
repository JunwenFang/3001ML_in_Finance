import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm
# import estimation

def target_label(df):

    df['default_year'] = pd.to_datetime(df['def_date'], format="%d/%m/%Y", errors='coerce').dt.year
    df['target'] = ((df['default_year'].notna()) & (df['fs_year'] + 1 >= df['default_year'])).astype(int)

    return df

def null_imputation(df):
    df['roe'] = df['roa'] * df['asst_tot']/df['eqty_tot']
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

    df_imputed = null_imputation(df_labeled)

    df_engineered = features_engineering(df_imputed)

    df_drop_na =  df_engineered[["fs_year",'target','roa','td_ta','current_ratio','Debt_coverage', 'asst_tot']]\
                    .replace([float('inf'), -float('inf')], float('nan'))\
                    .dropna()

    df_standardized = standardize(df_drop_na)
    
    final_columns = ["fs_year",'target','roa','td_ta','current_ratio','Debt_coverage', 'asst_tot_log']
    final_df = df_standardized[final_columns]
   
    return final_df


def estimator(df, formula):
    #f: "target ~ roa + td_ta + current_ratio + Debt_coverage + asst_tot"
    model = sm.logit(formula, data=df).fit()
    return model


def predictor(test_df, model):
    prob = model.predict(test_df)
    return prob


def preprocessor(data):
    df = data.copy()

    df_labeled = target_label(df)

    df_imputed = null_imputation(df_labeled)

    df_engineered = features_engineering(df_imputed)

    df_drop_na =  df_engineered[["fs_year",'target','roa','td_ta','current_ratio','Debt_coverage', 'asst_tot']]\
                    .replace([float('inf'), -float('inf')], float('nan'))\
                    .dropna()

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
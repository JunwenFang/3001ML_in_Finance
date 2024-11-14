import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import statsmodels.formula.api as sm
import pickle


def target_label(df):
    """
    Create target label for the dataset.
    If the default year within the fiscal year and the next year of the fiscal year (fis_year + 1), return 1
    Else (longer or NaT) return 0
    """

    df['default_year'] = pd.to_datetime(df['def_date'], format="%d/%m/%Y", errors='coerce').dt.year
    df['target'] = ((df['default_year'].notna()) & (df['fs_year'] + 1 >= df['default_year'])).astype(int)

    return df

def null_imputation(data):
    """
    Impute null values by cluster's median value
    """

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


def estimator(df, formula):
    #f: "target ~ roa + td_ta + current_ratio + Debt_coverage + asst_tot"
    model = sm.logit(formula, data=df).fit()
    return model


def predictor(test_df, model):
    prob = model.predict(test_df)
    return prob


def predictor_harness(new_df, model, preprocessor):

    df_test = preprocessor(new_df)
    
    X_test = df_test.drop(columns = ['target', 'fs_year'], axis=1)
    y_test = df_test['target']
    
    prob = predictor(X_test, model)
    predictions = {
        'Actual': y_test,
        'Predicted': prob
    }   
    
    return predictions 


def metrics(result_dict, year):
    y_actual = result_dict["Actual"]
    prob = result_dict["Predicted"]
    roc_auc = roc_auc_score(y_actual, prob)
    
    print(f"Year: {year}  ROC AUC: {roc_auc:.4f}")

    return {"Year": year, "AUC":roc_auc}



def walk_forward_harness(data, preprocessor, estimator, predictor_harness):

    df = data.copy()

    predictions = []
    model_list = []
    stats_list = []

    df['fs_year'] = df['fs_year'].astype(int)
    df = df.sort_values(by="fs_year",ascending=True)
    years = df['fs_year'].unique()

    print(years)
    
    for i in range(0, len(years)-1):

        year = years[i]
        print("Test year:", year)

        train_data = df[df['fs_year'] <= year]
        test_data = df[df['fs_year'] == year + 1]

        # pre-processing
        df_train = preprocessor(train_data)

        # estimator
        column_names = list(df_train.drop(["target","fs_year"], axis=1).columns)
        my_formula = "target ~ " + " + ".join(column_names)
        model = estimator(df_train, my_formula)
        model_list.append(model)

        # predictor
        result_dict = predictor_harness(test_data, model, preprocessor)
        predictions.append(result_dict)

        # performance metrics
        stats = metrics(result_dict, year)
        stats_list.append(stats)              

    return predictions, model_list, stats_list





original_train = pd.read_csv("Data/train.csv").drop("Unnamed: 0",axis=1)
df = original_train.copy()


predictions, model_list, stats_list = walk_forward_harness(df, preprocessor, estimator, predictor_harness)

print("\n")
mean_auc = [d["AUC"] for d in stats_list]
print("Mean AUC:", np.mean(mean_auc))

#overall predictions
all_actuals = pd.concat([d['Actual'] for d in predictions], ignore_index=True).values
all_predictions = pd.concat([d['Predicted'] for d in predictions], ignore_index=True).values

auc_score = roc_auc_score(all_actuals, all_predictions)
print("\n")
print(f"Overall AUC: {auc_score:.4f}")


# Final model

df_train = preprocessor(df)

X_train = df_train.drop(columns = ['target', 'fs_year'], axis=1)
y_train = df_train['target']

variables = list(X_train.columns)
my_formula = "target ~ " + " + ".join(variables)
model = estimator(df_train, my_formula)

# save the model
with open("final_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Calibration model
def calibrate_with_isotonic(df, model_output_col, default_label_col, k=20):
    df = df.sort_values(by=model_output_col, ascending=False).reset_index(drop=True)
    
    N = len(df)
    bucket_size = N // k
    
    default_rates = []
    quantiles = []
    
    for i in range(k):
        bucket = df.iloc[i * bucket_size: (i + 1) * bucket_size]
        
        default_rate = bucket[default_label_col].mean()
        default_rates.append(default_rate)

        quantiles.append(bucket[model_output_col].min())
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(quantiles, default_rates)
    

    with open('calibration_model.pkl', 'wb') as file:
        pickle.dump(iso_reg, file)
    
    calibrated_probs = iso_reg.transform(df[model_output_col].values)
    print("calibration done!")
    
    return calibrated_probs, iso_reg


predictions = model.predict(X_train)
df_calib = pd.DataFrame({"Actual":y_train, "Predicted":predictions})

cali_prob, iso_reg = calibrate_with_isotonic(df_calib,"Predicted","Actual")
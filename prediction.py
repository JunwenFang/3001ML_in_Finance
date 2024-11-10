import estimation


def preprocessor(df, preproc_params={}):

    data = df.copy()

    # Imputes the variables before feature engineering (avoid inf or -inf)
    df_nan_impute = nan_var_replace_pre_feat(data)

    # Do feature engineering
    df_feat_eng = feature_engineering(df_nan_impute)

    # Impute post feature engineering (impute ratios)
    df_feat_eng_nan_impute = nan_var_replace_post_feat(df_feat_eng)

    # Required features
    features = [ 'roa',
                 'asst_tot',
                 'debt_ratio',
                 'cash_return_assets',
                 'leverage']
    
    # Drop unnecessary columns
    df_features = df_feat_eng_nan_impute.drop(columns = [col for col in df_feat_eng 
                                            if col not in features])
    
    # Final imputation using medians
    result_df = nan_var_median_post_feat(df_features[features], preproc_params)

    return result_df
    
def predictor_harness(new_df, model, preprocessor, output_csv, preproc_params={}):
    
    preprocessed_data = preprocessor(new_df, preproc_params)
    predictions = model.predict(preprocessed_data)

    def curve_func(x, a, b, c):
        return a * x**2 + b * x + c

    def calibrate_prob(raw_prob, params):
        calibrate_prob = [curve_func(x, *params) for x in raw_prob]
        return calibrate_prob
    
    calculated_coefficients = [5.86333743e+02, -7.31859846e+00, 1.75116159e-02]
        
    predictions = calibrate_prob(predictions, calculated_coefficients)
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(output_csv, index=False, header=False)

    return output_csv
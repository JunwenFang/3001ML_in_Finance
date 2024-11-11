import pandas as pd
import pickle
import argparse
from prediction import preprocessor, predictor_harness

def load_model():
    with open('final_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_csv", required=True)

args = parser.parse_args()

input_csv = args.input_csv
output_csv = args.output_csv

df = pd.read_csv(input_csv)

model = load_model()

final_output = predictor_harness(df, model, preprocessor, output_csv )
                               
print('done')

# python3 harness.py --input_csv  <input file in csv> --output_csv <output csv file path to which the predictions are written> 
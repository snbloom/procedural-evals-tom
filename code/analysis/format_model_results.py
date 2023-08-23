"""
Get one df for all models, conditions, variables, methods, etc and store as raw_model_results.csv in same path.
"""
import os
import numpy as np
import pandas as pd


DATA_DIR = '../../data'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
INITIAL_BELIEF = ['0_forward', '1_forward'] # 0 hide initial belief, 1 show initial belief
VARIABLES = ['belief'] #'desire']
CONDITIONS = ['true_belief', 'false_belief']
models = ['meta-llama_Llama-2-7b-hf_0', 'openai_text-davinci-003_0', 'gpt-3.5-turbo_0']
methods = ['0shot']
temperatures = [0]


# Create a list to hold all DataFrames
model_results = []

for init_belief in INITIAL_BELIEF:
    for variable in VARIABLES:
        for condition in CONDITIONS:
            for model_name in models:
                for temperature in temperatures:
                    for method in methods:
                        if 'belief' in condition:
                            if 'gpt-4' in model_name:
                                if method == '0shot' or method == '0shot-cot':
                                    if method == '0shot':
                                        accuracy_file_1 = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/accuracy_{model_name}.{temperature}_chat-{method}_{variable}_{condition}.csv')
                                    elif method == '0shot-cot':
                                        accuracy_file_1 = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/accuracy_{model_name}.{temperature}_chat-{method}_{variable}_{condition}_0_100.csv')
                                    accuracy_file_2 = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/accuracy_{model_name}.{temperature}_chat-{method}_{variable}_{condition}_100_50.csv')
                                elif method == '1shot' or method == '1shot-cot':
                                    accuracy_file = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/accuracy_{model_name}.{temperature}_chat-{method}_{variable}_{condition}_0_50.csv')
                            elif True:
                                accuracy_file_1 = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/accuracy_{model_name}.{temperature}_{method}_{variable}_{condition}_0_50.csv')
                            
                        df_structure = ['correct']
                        df = pd.DataFrame(columns=df_structure)

                        if 'belief' in condition:
                            df = pd.read_csv(accuracy_file_1, header=None, names=['correct'])
                            df['correct'] = df['correct'].astype(int)
                        elif 'control' in condition:
                            df = pd.read_csv(accuracy_file, header=None, names=['correct'])
                            df['correct'] = df['correct'].astype(int)

                        meta_df = pd.DataFrame({
                            'init_belief': [init_belief.split('_')[0]] * len(df),
                            'direction': [init_belief.split('_')[1]] * len(df),
                            'variable': [variable] * len(df),
                            'true_false': [condition.split('_')[0]] * len(df),
                            'condition': [condition.split('_')[1]] * len(df),
                            'model_name': [model_name] * len(df),
                            'temperature': [temperature] * len(df),
                            'method': [method] * len(df),
                        })
                        # Concatenate metadata with data
                        df = pd.concat([df, meta_df], axis=1)
                        # Append to the list of DataFrames
                        model_results.append(df)
                            
                        
# Concatenate all data into a single DataFrame
model_results = pd.concat(model_results, ignore_index=True)
# Save the DataFrame as a CSV file
model_results.to_csv('raw_model_results.csv', index=False)

import pandas as pd

def process_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Apply the condition to the 'preds' column
    df['pred'] = df['pred'].apply(lambda x: 0 if x <= 0.5 else 1)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Provide the path to the input and output CSV files
input_file = 'DinoV2/dinov2_lance.csv'
output_file = 'dinov2_lance.csv'

# Process the CSV file
process_csv(input_file, output_file)

import pandas as pd
import os
# List of CSV files to combine
root_dir = '/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/generated_images/mistral_inpainting'
csv_files = ['ADE20K_mistral_subset.csv', 'CityScapes_mistral_subset.csv', 'OpenImages_mistral_subset.csv','SUN_RGBD_mistral_subset.csv','HumanParsing_mistral_subset.csv']
save_csv = 'mistral_subset.csv'
# Initialize an empty list to store DataFrames
dfs = []

# Read each CSV file and append its DataFrame to the list
for file in csv_files:
    dfs.append(pd.read_csv(os.path.join(root_dir,file)))

# Combine DataFrames into a single DataFrame
combined_df = pd.concat(dfs, axis=0, ignore_index=True)

# Optionally, you can save the combined DataFrame to a new CSV file
combined_df.to_csv(os.path.join(root_dir,save_csv), index=False)

# Print the combined DataFrame
# print(combined_df)
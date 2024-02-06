import pandas as pd 
import pdb 
import numpy as np

data = '/srv/hoffman-lab/share4/apal72/half-truths/csvs/lance_meta_fixed.csv'
df = pd.read_csv(data)
columns_fake = ['pertrubation_type', 'dataset', 'SA_change_category']
df_cleaned_fake = df[~((df['label'] == 'fake') & (df[columns_fake].eq('').any(axis=1) | df[columns_fake].isna().any(axis=1)))]
columns_real = ['image_path', 'dataset']
df_cleaned = df_cleaned_fake[~((df_cleaned_fake['label'] == 'real') & (df_cleaned_fake[columns_real].eq('').any(axis=1) | df_cleaned_fake[columns_real].isna().any(axis=1)))]
df_final = df_cleaned.dropna(subset=['label'], axis=0)
df_real = df_final[df_final['label']=='real']
indices = list(df_real.index)
np.random.shuffle(indices)


total_indices = len(indices)
train_indices = indices[:int(0.8 * total_indices)]
val_indices = indices[int(0.8 * total_indices):int(0.9 * total_indices)]
test_indices = indices[int(0.9 * total_indices):]

df_final.loc[train_indices, 'split'] = 'train'
df_final.loc[val_indices, 'split'] = 'val'
df_final.loc[test_indices, 'split'] = 'test'

df_fake = df_final[df_final['label']=='fake']
indices = list(df_fake.index)
np.random.shuffle(indices)


total_indices = len(indices)
train_indices = indices[:int(0.8 * total_indices)]
val_indices = indices[int(0.8 * total_indices):int(0.9 * total_indices)]
test_indices = indices[int(0.9 * total_indices):]


df_final.loc[train_indices, 'split'] = 'train'
df_final.loc[val_indices, 'split'] = 'val'
df_final.loc[test_indices, 'split'] = 'test'

df_final.to_csv('/srv/hoffman-lab/share4/apal72/half-truths/csvs/lance_meta_fixed_2.csv')
import pandas as pd 
import pdb
import numpy as np 

df = pd.read_csv('/srv/hoffman-lab/share4/apal72/half-truths/csvs/repaint_meta.csv')
df['change_value'] =""
df['pertrubation_type']=""
df['split'] = ''

def split_data(df,df_sub):
    indices = list(df_sub.index)
    np.random.shuffle(indices)



    total_indices = len(indices)
    train_indices = indices[:int(0.8 * total_indices)]
    val_indices = indices[int(0.8 * total_indices):int(0.9 * total_indices)]
    test_indices = indices[int(0.9 * total_indices):]
    df.loc[train_indices, 'split'] = 'train'
    df.loc[val_indices, 'split'] = 'val'
    df.loc[test_indices, 'split'] = 'test'
    return df 

df_inpaint = df[df['model']=='StableDiffusion']
df = split_data(df,df_inpaint)
df_repaint = df[df['model']=='RePaint']
df = split_data(df,df_repaint)
df_real = df[df['label']=='real']
df = split_data(df,df_real)

# pdb.set_trace()
colums_order = ['dataset','image_path','id','name','pertrubation_type','change_value','model','SA_change_category','label','split']
df = df[colums_order]

df.to_csv('/srv/hoffman-lab/share4/apal72/half-truths/csvs/repaint_meta_fixed.csv')
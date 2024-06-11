import pandas as pd 
import glob 
import pdb
import json 
import matplotlib.pyplot as plt
import os
import numpy as np
root_path = '/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/mag_csvs'
all_csvs = glob.glob(root_path+'/*_2.csv')

def convert_to_native_types(data):
    if isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(element) for element in data]
    else:
        return data


def find_diff(df, col1, col2):
    # pdb.set_trace()
    difference= (df[col1]- df[col2]).tolist()
    absolute_difference = [abs(num) for num in difference]
    max_val = max(absolute_difference)
    min_val = min(absolute_difference)
    return max_val, min_val

def find_corr(df,col1,col2):
    return df[col1].corr(df[col2])

def create_bins(df, col, file_path,var='whole'):
    # pdb.set_trace()

    max_val = df[col].max()
    min_val = df[col].min()
    plt.hist(df[col].tolist(), bins=10000, edgecolor='moccasin')
    # Add labels and title
    plt.xlabel('Magnitude of Change')
    plt.ylabel('Frequency')

    plt.title(col+' distribution')

    # Show the plot
    if(not(os.path.exists(os.path.join(file_path, var)))):
        os.makedirs(os.path.join(file_path, var))
    if(var == 'whole'):
        if(col == 'cc_cluters_rgb') or (col =='cluster_dist_rgb') or (col == 'largest_component_size_rgb'):
            mean_val = bin_values(df[col], col)

            return  max_val, min_val, mean_val
        else:
            small_max, large_max = bin_values(df[col], col)
            if(col == 'dreamsim'):
                ax = plt.gca()
                ax.set_xlim([0.0, 1.0])
            plt.axvline(x=small_max, color='r', linestyle='--', label='Small/Medium cutoff')
            plt.axvline(x=large_max, color='g', linestyle='--', label='Medium/Large cutoff')
            plt.legend()
            plt.savefig(os.path.join(file_path, var, (col+'.png')))
            plt.close()
            return max_val, min_val, small_max, large_max
    if(col == 'cc_cluters_rgb') or (col =='cluster_dist_rgb') or (col == 'largest_component_size_rgb'):
        return max_val, min_val, 0
    return max_val, min_val, 0, 0

def create_bins_5(df, col, file_path,var='whole'):
    # pdb.set_trace()

    max_val = df[col].max()
    min_val = df[col].min()
    plt.hist(df[col].tolist(), bins=1000, edgecolor='black')
    # Add labels and title
    plt.xlabel('Magnitude of Change')
    plt.ylabel('Frequency')
    plt.title(col+' distribution')

    # Show the plot
    if(not(os.path.exists(os.path.join(file_path, var)))):
        os.makedirs(os.path.join(file_path, var))
    plt.savefig(os.path.join(file_path, var, (col+'.png')))
    plt.close()
    if(var == 'whole'):
        if(col == 'cc_cluters_rgb') or (col =='cluster_dist_rgb') or (col == 'largest_component_size_rgb'):
            mean_val = bin_values_5(df[col], col)
            return  max_val, min_val, mean_val
        else:
            bin_1, bin2, bin3, bin4 = bin_values_5(df[col], col)
            return max_val, min_val, bin_1, bin2, bin3, bin4
    if(col == 'cc_cluters_rgb') or (col =='cluster_dist_rgb') or (col == 'largest_component_size_rgb'):
        return max_val, min_val, 0
    return max_val, min_val, 0, 0, 0, 0

def bin_values(values, col):
    # Define the bin boundaries
    # pdb.set_trace()
    small_max = np.percentile(np.array(values), 25)
    large_max = np.percentile(np.array(values), 75)
    if(col == 'cc_cluters_rgb') or (col =='cluster_dist_rgb') or (col == 'largest_component_size_rgb'):
        return np.mean(np.array(df[col]))
    else:
        return small_max, large_max


def bin_values_5(values, col):
    # Define the bin boundaries
    # pdb.set_trace()
    bin1 = np.percentile(np.array(values), 20)
    bin2 = np.percentile(np.array(values), 40)
    bin3 = np.percentile(np.array(values), 60)
    bin4 = np.percentile(np.array(values), 80)
    if(col == 'cc_cluters_rgb') or (col =='cluster_dist_rgb') or (col == 'largest_component_size_rgb'):
        return np.mean(np.array(df[col]))
    else:
        return bin1, bin2, bin3, bin4
    # for i in values:
    #     if(i < small_max):
    #         change.append('low')
    #     elif(small_max <= i < medium_max):
    #         change.append('medium')
    #     else:
    #         change.append('high')
    
    # return scene, small_max, medium_max

def diff_sem(df_all,dic, file_path, var='whole'):
    # pdb.set_trace()
    dic[var] = {}
    if(var == 'whole'):
        df = df_all
    else:
        df = (df_all[df_all['sem_magnitude'] == var]).reset_index(drop=True)
    
    print('DREAMSIM')
    dic[var]['dreamsim'] = {}
    dic[var]['dreamsim']['max'], dic[var]['dreamsim']['min'], dic[var]['dreamsim']['small_threshold'], dic[var]['dreamsim']['large_threshold'] = create_bins(df,'dreamsim',file_path, var)
    
    print('LPIPS')
    dic[var]['lpips_score'] = {}
    dic[var]['lpips_score']['max'], dic[var]['lpips_score']['min'], dic[var]['lpips_score']['small_threshold'], dic[var]['lpips_score']['large_threshold'] = create_bins(df,'lpips_score',file_path, var)
    
    print('SEN_SIM')
    dic[var]['sen_sim'] = {}
    dic[var]['sen_sim']['max'], dic[var]['sen_sim']['min'], dic[var]['sen_sim']['small_threshold'], dic[var]['sen_sim']['large_threshold'] = create_bins(df,'sen_sim',file_path, var)
    
    print('CLIP_SIM')
    dic[var]['clip_sim'] = {}
    dic[var]['clip_sim']['max'], dic[var]['clip_sim']['min'], dic[var]['clip_sim']['small_threshold'], dic[var]['clip_sim']['large_threshold'] = create_bins(df,'clip_sim',file_path, var)
    
    print('MSE')
    dic[var]['mse_rgb'] = {}
    dic[var]['mse_rgb']['max'], dic[var]['mse_rgb']['min'],dic[var]['mse_rgb']['small_threshold'], dic[var]['mse_rgb']['large_threshold'] = create_bins(df,'mse_rgb',file_path, var)
    dic[var]['mse_gray'] = {}
    dic[var]['mse_gray']['max'], dic[var]['mse_gray']['min'], dic[var]['mse_gray']['small_threshold'], dic[var]['mse_gray']['large_threshold'] = create_bins(df,'mse_gray',file_path,var)
    
    print('SSIM')
    dic[var]['ssim_rgb'] = {}
    dic[var]['ssim_rgb']['max'], dic[var]['ssim_rgb']['min'], dic[var]['ssim_rgb']['small_threshold'], dic[var]['ssim_rgb']['large_threshold'] = create_bins(df,'ssim_rgb',file_path,var)
    dic[var]['ssim_gray'] = {}
    dic[var]['ssim_gray']['max'], dic[var]['ssim_gray']['min'], dic[var]['ssim_gray']['small_threshold'], dic[var]['ssim_gray']['large_threshold'] = create_bins(df,'ssim_gray',file_path,var)
    
    print('RATIO')
    dic[var]['ratio_rgb'] = {}
    dic[var]['ratio_rgb']['max'], dic[var]['ratio_rgb']['min'], dic[var]['ratio_rgb']['small_threshold'], dic[var]['ratio_rgb']['large_threshold'] = create_bins(df,'ratio_rgb',file_path,var)
    
    dic[var]['ratio_gray'] = {}
    dic[var]['ratio_gray']['max'], dic[var]['ratio_gray']['min'], dic[var]['ratio_gray']['small_threshold'], dic[var]['ratio_gray']['large_threshold'] = create_bins(df,'ratio_gray',file_path,var)

    print('CC')
    dic[var]['largest_component_size_rgb'] = {}
    dic[var]['largest_component_size_rgb']['max'], dic[var]['largest_component_size_rgb']['min'], dic[var]['largest_component_size_rgb']['mean'] = create_bins(df,'largest_component_size_rgb',file_path,var)

    dic[var]['cc_cluters_rgb'] = {}
    dic[var]['cc_cluters_rgb']['max'], dic[var]['cc_cluters_rgb']['min'], dic[var]['cc_cluters_rgb']['mean'] = create_bins(df,'cc_cluters_rgb',file_path,var)

    dic[var]['cluster_dist_rgb'] = {}
    dic[var]['cluster_dist_rgb']['max'], dic[var]['cluster_dist_rgb']['min'],dic[var]['cluster_dist_rgb']['mean'] = create_bins(df,'cluster_dist_rgb',file_path,var)

    return dic


def diff_sem_5(df_all,dic, file_path, var='whole'):
    # pdb.set_trace()
    dic[var] = {}
    if(var == 'whole'):
        df = df_all
    else:
        df = (df_all[df_all['sem_magnitude'] == var]).reset_index(drop=True)
    
    print('DREAMSIM')
    dic[var]['dreamsim'] = {}
    dic[var]['dreamsim']['max'], dic[var]['dreamsim']['min'], dic[var]['dreamsim']['bin1'], dic[var]['dreamsim']['bin2'], dic[var]['dreamsim']['bin3'], dic[var]['dreamsim']['bin4'] = create_bins(df,'dreamsim',file_path, var)
    
    print('LPIPS')
    dic[var]['lpips_score'] = {}
    dic[var]['lpips_score']['max'], dic[var]['lpips_score']['min'], dic[var]['lpips_score']['bin1'], dic[var]['lpips_score']['bin2'],  dic[var]['lpips_score']['bin3'],  dic[var]['lpips_score']['bin4'] = create_bins(df,'lpips_score',file_path, var)
    
    print('SEN_SIM')
    dic[var]['sen_sim'] = {}
    dic[var]['sen_sim']['max'], dic[var]['sen_sim']['min'], dic[var]['sen_sim']['bin1'], dic[var]['sen_sim']['bin2'], dic[var]['sen_sim']['bin3'], dic[var]['sen_sim']['bin4'] = create_bins(df,'sen_sim',file_path, var)
    
    print('CLIP_SIM')
    dic[var]['clip_sim'] = {}
    dic[var]['clip_sim']['max'], dic[var]['clip_sim']['min'], dic[var]['clip_sim']['bin1'], dic[var]['clip_sim']['bin2'], dic[var]['clip_sim']['bin3'], dic[var]['clip_sim']['bin4']   = create_bins(df,'clip_sim',file_path, var)
    
    print('MSE')
    dic[var]['mse_rgb'] = {}
    dic[var]['mse_rgb']['max'], dic[var]['mse_rgb']['min'],dic[var]['mse_rgb']['bin1'], dic[var]['mse_rgb']['bin2'], dic[var]['mse_rgb']['bin3'], dic[var]['mse_rgb']['bin4'] = create_bins(df,'mse_rgb',file_path, var)
    dic[var]['mse_gray'] = {}
    dic[var]['mse_gray']['max'], dic[var]['mse_gray']['min'], dic[var]['mse_gray']['bin1'], dic[var]['mse_gray']['bin2'], dic[var]['mse_gray']['bin3'], dic[var]['mse_gray']['bin4'] = create_bins(df,'mse_gray',file_path,var)
    
    print('SSIM')
    dic[var]['ssim_rgb'] = {}
    dic[var]['ssim_rgb']['max'], dic[var]['ssim_rgb']['min'], dic[var]['ssim_rgb']['bin1'], dic[var]['ssim_rgb']['bin2'], dic[var]['ssim_rgb']['bin3'], dic[var]['ssim_rgb']['bin4'] = create_bins(df,'ssim_rgb',file_path,var)
    dic[var]['ssim_gray'] = {}
    dic[var]['ssim_gray']['max'], dic[var]['ssim_gray']['min'], dic[var]['ssim_gray']['bin1'], dic[var]['ssim_gray']['bin2'], dic[var]['ssim_gray']['bin3'], dic[var]['ssim_gray']['bin4']= create_bins(df,'ssim_gray',file_path,var)
    
    print('RATIO')
    dic[var]['ratio_rgb'] = {}
    dic[var]['ratio_rgb']['max'], dic[var]['ratio_rgb']['min'], dic[var]['ratio_rgb']['bin1'], dic[var]['ratio_rgb']['bin2'], dic[var]['ratio_rgb']['bin3'], dic[var]['ratio_rgb']['bin4'] = create_bins(df,'ratio_rgb',file_path,var)
    
    dic[var]['ratio_gray'] = {}
    dic[var]['ratio_gray']['max'], dic[var]['ratio_gray']['min'], dic[var]['ratio_gray']['bin1'], dic[var]['ratio_gray']['bin2'], dic[var]['ratio_gray']['bin3'], dic[var]['ratio_gray']['bin4']= create_bins(df,'ratio_gray',file_path,var)

    print('CC')
    dic[var]['largest_component_size_rgb'] = {}
    dic[var]['largest_component_size_rgb']['max'], dic[var]['largest_component_size_rgb']['min'], dic[var]['largest_component_size_rgb']['mean'] = create_bins(df,'largest_component_size_rgb',file_path,var)

    dic[var]['cc_cluters_rgb'] = {}
    dic[var]['cc_cluters_rgb']['max'], dic[var]['cc_cluters_rgb']['min'], dic[var]['cc_cluters_rgb']['mean'] = create_bins(df,'cc_cluters_rgb',file_path,var)

    dic[var]['cluster_dist_rgb'] = {}
    dic[var]['cluster_dist_rgb']['max'], dic[var]['cluster_dist_rgb']['min'],dic[var]['cluster_dist_rgb']['mean'] = create_bins(df,'cluster_dist_rgb',file_path,var)

    return dic

def print_vals(combined_df):

    max_dream_lpips,min_dream_lpips = find_diff(combined_df,'dreamsim','lpips_score')
    cor_dream_lpips = find_corr(combined_df,'dreamsim','lpips_score')

    max_sen_clip, min_sen_clip = find_diff(combined_df,'sen_sim','clip_sim')
    cor_sen_clip = find_corr(combined_df,'sen_sim','clip_sim')

    max_dif_rgb_gray, min_dif_rgb_gray = find_diff(combined_df,'mse_rgb','mse_gray')
    cor_rgb_gray = find_corr(combined_df,'mse_rgb','mse_gray')

    max_dif_ratio_rgb_gray, min_dif_ratio_rgb_gray = find_diff(combined_df,'ratio_rgb','ratio_gray')
    cor_ratio_rgb_gray = find_corr(combined_df,'ratio_rgb','ratio_gray')

    max_dif_ratio_mse, min_dif_ratio_mse = find_diff(combined_df,'mse_rgb','ratio_rgb')
    cor_ratio_mse = find_corr(combined_df,'ratio_rgb','mse_rgb')

    print('DREAM_LPIPS MAX: ',max_dream_lpips, 'MIN : ', min_dream_lpips, 'COR: ', cor_dream_lpips)
    print('SEN_CLIP MAX: ',max_sen_clip, 'MIN : ', min_sen_clip, 'COR: ', cor_sen_clip)
    print('MSE_RGB_GRAY MAX: ',max_dif_rgb_gray, 'MIN : ', min_dif_rgb_gray, 'COR: ', cor_rgb_gray)
    print('RATIO_RGB_GRAY MAX: ',max_dif_ratio_rgb_gray, 'MIN : ', min_dif_ratio_rgb_gray, 'COR: ', cor_ratio_rgb_gray)
    print('RATIO_MSE_RGB MAX: ',max_dif_ratio_mse, 'MIN : ', min_dif_ratio_mse, 'COR: ', cor_ratio_mse)

file_path='/srv/share4/apal72/half-truths/data/Half_Truths_Dataset/mag_plots/dist'

combined_data = []

for filename in all_csvs:
    df = pd.read_csv(filename)
    if(filename.split('/')[-1] == 'final_filtered_sample_data_mag-metrics_2.csv'):
        # pdb.set_trace()
        df = df[df['pass_qc'] == 1]
    if(df['dreamsim'].isna().any()):
        
        nan_indices = df[df['dreamsim'].isna()].index
        nan_count = df['dreamsim'].isna().sum()
        print('NaN: ', nan_count, ' : ', nan_indices)
    df1 = df.iloc[:, -16:]
    sem_mag_column = df['sem_magnitude']
    combined_columns = pd.concat([sem_mag_column, df1], axis=1)
    combined_data.extend(combined_columns.values.tolist())
combined_df = pd.DataFrame(combined_data, columns=combined_columns.columns)
combined_inpainting = []

for filename in all_csvs:
    df = pd.read_csv(filename)
    if not(filename.split('/')[-1] == 'final_filtered_sample_data_mag-metrics_2.csv'):
        # df = df['ratio']
        combined_inpainting.extend(df.values.tolist())
combined_df = pd.DataFrame(combined_inpainting, columns=df.columns)

dic = {}
dic = diff_sem(combined_df,dic,file_path,'whole')


native_data = convert_to_native_types(dic)
with open(file_path +'/mag_vals_3_.json', "w") as json_file:
    json.dump(native_data, json_file)




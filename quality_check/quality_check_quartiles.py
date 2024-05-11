import json
import pdb
import numpy as np
from tqdm import tqdm

DS = [
    "ADE20K", 
      "CelebAHQ", 
      "CityScapes", 
      "HumanParsing", 
      "OpenImages", 
      "SUN_RGBD"]
       

cap = []
img = []


DATA_FILE = f"/Users/mphute6/Documents/HalfTruths/postgen_quality_check_sem.csv"

data = json.load(open(DATA_FILE, "r"))

data_all = []

for key in data.keys():
    for edit in data[key]:
        # pdb.set_trace()
        try:
            cap.append(edit["cap2_img2_similarity"])
            img.append(edit["direct_similarity"])
        except:
            # print(edit)
            continue

cap = np.array(cap)
img = np.array(img)

# lower_cap = np.percentile(cap, 5)
# upper_cap = np.percentile(cap, 90)

# lower_img = np.percentile(img, 5)
# upper_img = np.percentile(img, 100)

# pdb.set_trace()

new_data = {}
# # DATA_FILE = f"/Users/mphute6/Documents/HalfTruths/postgen_quality_check_sem.csv"
# data = json.load(open(DATA_FILE, "r"))

# for key in data.keys():
#     for edit in data[key]:
#         try:
#             if edit["cap1_cap2_similarity"] < lower_cap or edit["cap1_cap2_similarity"] > upper_cap:
#                 continue
#             elif edit["img1_cap1_similarity"] < lower_img or edit["img1_cap1_similarity"] > upper_img:
#                 continue
#             else:
#                 new_data[key] = new_data.get(key, [])
#                 new_data[key].append(edit)
#                 # pdb.set_trace()
#         except:
#             # print(edit)
#             continue

# pdb.set_trace()
# i = 0
# for key in new_data.keys():
#     for edit in new_data[key]:
#         # print(edit)
#         n = edit["cap1_cap2_similarity"]
#         m = edit["img1_cap1_similarity"]
#         i+=1

# print(i)

# OUT_DIR = f"/raid/mphute6/HalfTruths/SemanticDefinition/outputs/quality_check/{ds}.json"
# with open(OUT_DIR, "w") as f:  
#     json.dump(new_data, f)

values = range(2,5)

desired_cap_sim = np.percentile(cap, 85)
desired_img_sim = [np.percentile(img,i) for i in values] 
desired_cap_sim = [np.percentile(cap,i) for i in values]
pdb.set_trace()

for key in data.keys():
    new_data[key] = []
    for edit in data[key]:
        # pdb.set_trace()
        try:
            if edit["cap2_img2_similarity"]  in desired_cap_sim:
                print("Original:  ", edit["original_caption"])
                print("Edited:  ", edit["edited_caption"])
                print(key)
                print("------------------------------")
            # if edit["img1_img2_similarity"]  == desired_img_sim:
            #     print(edit["edited_caption"])
        except Exception as e:
            # print(e)
            continue

# desired_cap_sim = np.percentile(cap, 85)
# values = range(2,5)
# desired_img_sim = [np.percentile(img,i) for i in values] 
# desired_cap_sim = [np.percentile(cap,i) for i in values]
# # pdb.set_trace()

# for key in data.keys():
#     new_data[key] = []
#     for edit in data[key]:
#         # pdb.set_trace()
#         try:
#             if edit["img1_cap1_similarity"] > min(desired_img_sim) and edit["img1_cap1_similarity"] < max(desired_img_sim):
#             # if edit["cap1_cap2_similarity"]  in desired_cap_sim:
#                 print("Original:  ", edit["original_caption"])
#                 print("Edited:  ", edit["edited_caption"])
#                 print(key)
#                 print("------------------------------")

#         except Exception as e:
#             # print(e)
#             continue
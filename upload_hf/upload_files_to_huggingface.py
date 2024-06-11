from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(

    folder_path="/raid/mphute6/HalfTruths/inpainting_data/half-truths/half-truths/inpainting/inpainting_data",

    repo_id="semi-truths/Semi-Truths",

    repo_type="dataset",

)

# api.delete_folder(

#     path_in_repo = "CityScapes",

#     repo_id="semi-truths/Semi-Truths",

#     repo_type="dataset",

# )

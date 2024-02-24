import cv2
from sentence_transformers import SentenceTransformer, util
import numpy as np 
import torch
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pdb 
import math
import matplotlib.colors as mcolors

_ = torch.manual_seed(123)
from scipy.ndimage import label
def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    print('Min: %.3f, Max: %.3f' % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
    arr_new = -1 + (scaled * 2)
    print('Min: %.3f, Max: %.3f' % (arr_new.min(), arr_new.max()))
    return arr_new

path_orig = '/srv/share4/apal72/half-truths/data/ADE20K_People_256/images/ADE_frame_00000008.jpg'
path_edit = '/srv/share4/apal72/half-truths/data/outputs/ade20k/object/ADE_frame_00000008/probe.jpeg'
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sen = 'a photo of a man in an orange work outfit is digging a skeleton in the ground with a hammer'
sen2 = 'a photo of a man in an orange work outfit is digging a skeleton in the ground with a probe'

sentences = [sen, sen2]
embeddings = model.encode(sentences)
val = util.pytorch_cos_sim(embeddings[0], embeddings[1])
semantic_change = val.numpy()[0][0]
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
mse_thresh = 0.8
img1 = cv2.imread(path_orig)
img2 = cv2.imread(path_edit)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # Fixing the resizing typo

score = lpips(torch.from_numpy(normalize(img1)).permute(2, 0, 1).unsqueeze(0), torch.from_numpy(normalize(img2)).permute(2, 0, 1).unsqueeze(0))
mse = np.linalg.norm(img1 - img2, axis=2)
mse = mse / (255 * math.sqrt(3))
mask = (mse*255).astype(np.uint8)
binary_image = (mask > 128).astype(np.uint8)  
num_labels, labeled_image = cv2.connectedComponents(binary_image)
flattened_image = labeled_image.flatten()
pdb.set_trace()
# Count occurrences of each label value
label_counts = np.bincount(flattened_image)

# Find the size of the largest connected component
largest_component_size = np.max(label_counts[1:])
print("Size of the largest connected component:", largest_component_size)
print('CC --', num_labels)
# Calculate centroids of connected components
centroids = []
for label_idx in range(1, num_labels):  # Exclude background label (0)
    labeled_indices = np.where(labeled_image == label_idx)
    centroid_x = np.mean(labeled_indices[1])
    centroid_y = np.mean(labeled_indices[0])
    centroids.append((centroid_x, centroid_y))

num_unique_values = np.max(labeled_image) + 1  # Number of unique values including 0
colors = plt.cm.get_cmap('tab20', num_unique_values - 1)  # Choose a colormap and get unique colors, excluding 0
black_color = np.array([[0, 0, 0, 1]])  # Black color for label 0
colors_array = colors(np.arange(num_unique_values - 1))  # Convert colormap object to numpy array
colors = np.vstack((black_color, colors_array))  # Add black color for label 0
cmap = mcolors.ListedColormap(colors)


centroids = np.array(centroids)

threshold = 10  # Adjust as needed based on your image and application

# Calculate the mean and standard deviation of centroid coordinates
mean_centroid = np.mean(centroids, axis=0)
std_centroid = np.std(centroids, axis=0)

print(mean_centroid, std_centroid)
if np.any(std_centroid > threshold):
    print("Connected components are diffused.")
else:
    print("Connected components are concentrated.")




mse2 = mse.flatten()
print(score, semantic_change, np.mean(mse2))
# Create img with mask
# img = np.ones((img2.shape[0], img2.shape[1]))
# img[mse >= mse_thresh] = 255

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot img1
axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('original image ')
axes[0, 0].axis('off')

# Plot img2
axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('changed image')
axes[0, 1].axis('off')

# Plot img
im = axes[1, 0].imshow(mse, cmap='hot')
cbar = fig.colorbar(im, ax=axes[1, 0])
cbar.set_label('MSE')
axes[1, 0].legend(['MSE'], loc='upper right')
axes[1, 0].imshow(mse,cmap='hot')
axes[1, 0].set_title('mse visualization')
axes[1, 0].axis('off')

# # Plot mse histogram
# axes[1, 1].hist(mse2)
# axes[1, 1].set_xlabel('mse values')
# axes[1, 1].set_ylabel('Frequency')
# axes[1, 1].set_title('Histogram of all values in mse')
# Plot mse histogram
axes[1, 1].imshow(labeled_image,cmap=cmap)
axes[1, 1].set_title('connected components ')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('ADE_frame_00000008_2.png')
import numpy as np
import torch
from pkg_resources import packaging
import clip
import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from collections import OrderedDict
import torch
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

data_dir="content/train/"
test_data_dir="content/test/"

descriptions = {
    "car1": "red car on white background",
    "car2": "front photo of green car",
    "car3": "red car on the road",
    "car4": "green car in service shop",
    "car5": "yellow sport car with black stripes",
    "car6": "police car on the road",
    "fiets1": "a foto of fiets bicycle for sale",
    "fiets2": "a foto of fiets bicycle for sale",
    "fiets3": "a foto of fiets bicycle for sale",
    "truck1": "a construction truck on the road",
    "truck2": "emerald color construction truck",
    "truck3": "a photo of black transportation truck",
    "truck4": "a truck without the load",
    "truck5": "a truck with logo printed on the back",
    "truck6": "a giant construction truck",
}
print(os.listdir("content/train"))
print(os.listdir("content/test"))
original_images = []
original_test_images = []
images = []
test_images = []
texts = []
plt.figure(figsize=(16, 5))

# Browsing through the input folder,Saving images to a list of images
for filename in [filename for filename in os.listdir(data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    # train images should have a description!
    if name not in descriptions:
        continue

    image = Image.open(os.path.join(f"{data_dir}", filename)).convert("RGB")

    plt.subplot(2, 8, len(images) + 1 )
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}")
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    # preprocessing and saving image to list
    images.append(preprocess(image))
    # saving descriptions
    texts.append(descriptions[name])

plt.tight_layout()

for filename in [filename for filename in os.listdir(test_data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:

    # converting, preprocessing, saving to list test images
    image = Image.open(os.path.join(f"{test_data_dir}", filename)).convert("RGB")
    original_test_images.append(image)
    test_images.append(preprocess(image))

# indexing images?
image_input = torch.tensor(np.stack(images)).cuda()
test_image_input = torch.tensor(np.stack(test_images)).cuda()
# and text_tokens
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

# encoding with CLIP text tokens and image tokens
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    test_image_features = model.encode_image(test_image_input).float()
    text_features = model.encode_text(text_tokens).float()


image_features /= image_features.norm(dim=-1, keepdim=True)
test_image_features /= test_image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)


text_descriptions = [f"This is a photo of a {label}" for label in descriptions.values()]
text_tokens = clip.tokenize(text_descriptions).cuda()
print(text_descriptions)

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

# softmax based 20 classes(All descriptions/images from train in order to define the nearest image for test images)
text_probs = (100.0 * test_image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
print(top_probs, top_labels)

plt.figure(figsize=(32, 8))

# Plotting out the results(attached)
for i, image in enumerate(original_test_images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [list(descriptions.values())[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace=0.5)
plt.show()

# --- PART2.Car,Truck,Fiets,Other classificator ---

labels = ["Car","Truck","Bicycle","Other"]
labels_tokens = clip.tokenize(labels).cuda()

with torch.no_grad():
    labels_features = model.encode_text(labels_tokens).float()
    labels_features /= labels_features.norm(dim=-1, keepdim=True)

labels_probs = (100.0 * test_image_features @ labels_features.T).softmax(dim=-1)
top_probs, top_labels = labels_probs.cpu().topk(4, dim=-1)
print(top_probs, top_labels)
# Visualisation results (attached)
plt.figure(figsize=(32, 8))

# Plotting out the results(attached)
for i, image in enumerate(original_test_images):
    plt.subplot(4, 4, 2 * i + 1)
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    plt.gca().invert_yaxis()
    plt.gca().set_axisbelow(True)
    plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace=0.5)
plt.show()


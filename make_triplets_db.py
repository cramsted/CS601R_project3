import json
import pickle

IMAGE_ROOT_DIR = "../lfw/images/"

json_data = open(IMAGE_ROOT_DIR+"../frequencies.json").read()
freq_data = json.loads(json_data)

maximum = 1
for key, value in freq_data["freq"].items():
    if value > maximum:
        maximum = value
print(maximum)
# json_data = open(IMAGE_ROOT_DIR+"../pairs_dev_train.json").read()
# train_data = json.loads(json_data)

# train_images = []
# for key, value in train_data.items():
#     for i in value:
#         train_images += i

# divided_images = {"all_images": train_images, "pos": {}, "neg": {}}
# for key, value in freq_data["freq"].items():
#     if value > 1:
#         divided_images["pos"][key] = freq_data["list"][key]
#     else:
#         divided_images["neg"][key] = freq_data["list"][key]

# with open('dataset_info.json', 'w') as outfile:
#     json.dump(divided_images, outfile, indent=4, sort_keys=True)
json_data = open(IMAGE_ROOT_DIR+"../pairs_dev_train.json").read()
data = json.loads(json_data)
_train_data = {}
_triplet_pairs = []
_images = []
for key, value in data.items():
    for images in value:
        for i in images:
            _images.append(i)
            key = i.split("/")[0]
            if key not in _train_data:
                _train_data[key] = [i]
            else:
                _train_data[key].append(i)
                if key not in _triplet_pairs:
                    _triplet_pairs.append(key)

with open("triplet_db.pkl", 'wb') as f:
    pickle.dump((_train_data, _triplet_pairs, _images), f)

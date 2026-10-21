import torch
import utils
import numpy as np
from voc_dataset import VOCDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from train_q2 import ResNet

classes = VOCDataset.CLASS_NAMES
num_classes = len(classes)

model_path = "q1_q2_classification/checkpoint-resnet-epoch50.pth"
# model = torch.load(model_path, weights_only=False, map_location='cuda')
model = ResNet(num_classes).cuda()

state_dict = torch.load(
    model_path,
    map_location="cuda",
    weights_only=False
)

model.load_state_dict(state_dict.state_dict())

model.eval()

test_dataset = VOCDataset(split='test', size=224)

indices = np.random.randint(0, len(test_dataset), 1000)

features_list = []
labels_list = []

def feature_hook(module, inputs, outputs):
    features_list.append(inputs[0].detach().cpu().numpy().squeeze())

model.resnet.fc.register_forward_hook(feature_hook)

with torch.no_grad():
    for idx in indices:
        image, label, _ = test_dataset[idx]
        image = image.cuda().unsqueeze(0)

        # with torch.no_grad():
        _ = model(image)
        
        labels_list.append(label.cpu().numpy())

features = np.array(features_list)
labels = np.stack(labels_list, axis=0)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=42
)

features_2d = tsne.fit_transform(features)

class_colors = plt.get_cmap("tab20")(
    np.linspace(0, 1, num_classes)
)[:, :3]

labels_bool = labels > 0

# import pdb; pdb.set_trace()

point_colors = (
    labels_bool.astype(float) @ class_colors
    / np.maximum(labels_bool.sum(axis=1, keepdims=True), 1)
)

plt.figure(figsize=(10, 10))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=point_colors, s=10)

for i in range(num_classes):
    plt.scatter([], [], c=class_colors[i], label=classes[i])

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.title("t-SNE")
plt.savefig('tsne_cov.png', dpi=200, bbox_inches="tight")
plt.show()

# ============================================================
# Scratch CNN baseline on the Food-101 21-class subset
#
# Standalone script (does not import or depend on the ResNet file):
#     python scratch_cnn.py
#
# What it does:
#   * trains several scratch-CNN configurations (some expected NOT to help)
#   * saves learning curves for every run as PNG files
#   * evaluates the best run once on the official test set
#   * saves per_class_scratch_cnn.csv, which the ResNet script reads
#     to build the scratch-vs-ResNet comparison
#
# The class selection, official train/test files, 10% validation holdout,
# seed and test_size MUST match the ResNet script exactly, or the two
# models are not trained and scored on the same data.
# ============================================================
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------- settings (edit these) ----------------
dataPath= r"D:\FALLSEMESTER\food-101\food-101\food-101"   # folder that contains images/ and meta/   <-- EDIT THIS
numClasses= 21
seed= 42
validationStop= 0.10
patience= 5
imgSize= 128           # the scratch CNN is small; 128 px keeps training tractable
batchSize= 64
epochs= 30
learningRate= 0.001
numWorkers= 0          # 0 is the safe default on Windows
showPlots= False       # True = also pop up each figure (PNG files are always saved)
quickTest= False       # True = tiny end-to-end check (2 configs, 1 epoch, small subset)

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean= [0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]

augmentTransform= transforms.Compose([
    transforms.RandomResizedCrop(imgSize, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # images are oversaturated/noisy
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
plainTransform= transforms.Compose([
    transforms.Resize(146),
    transforms.CenterCrop(imgSize),   # crop BEFORE ToTensor, so the tensor is imgSize x imgSize
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ---------------- data ----------------
class FoodDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths= paths
        self.labels= labels
        self.transform= transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image= Image.open(self.paths[i]).convert("RGB")
        return self.transform(image), self.labels[i]


def findMetaFile(split):
    for ext in (".txt", ".csv"):
        candidate= os.path.join(dataPath, "meta", split + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No {split}.txt or {split}.csv found in {os.path.join(dataPath, 'meta')}. Check dataPath.")


def readSplit(split, classToIdx):
    # each line looks like  apple_pie/1005649  (optionally with extra comma-separated fields)
    paths, labels= [], []
    with open(findMetaFile(split)) as f:
        for line in f:
            entry= line.strip().split(",")[0]
            if "/" not in entry:
                continue                      # header or blank line
            className= entry.split("/")[0]
            if className in classToIdx:
                if not entry.endswith(".jpg"):
                    entry += ".jpg"
                paths.append(os.path.join(dataPath, "images", entry))
                labels.append(classToIdx[className])
    return paths, labels


def subsample(paths, labels, n):
    idx= np.random.RandomState(seed).permutation(len(paths))[:n]
    return [paths[i] for i in idx], [labels[i] for i in idx]


def makeLoader(paths, labels, transform, shuffle):
    return DataLoader(FoodDataset(paths, labels, transform), batch_size=batchSize, shuffle=shuffle,
                      num_workers=numWorkers, pin_memory=(device.type == "cuda"),
                      persistent_workers=(numWorkers > 0))


# ---------------- model ----------------
class ScratchCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.classifier= nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, numClasses)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, dataLoader, lossFunction):
    model.eval()
    totalLoss, correct, total= 0, 0, 0
    with torch.no_grad():
        for images, labels in dataLoader:
            images, labels= images.to(device), labels.to(device)
            outputs= model(images)
            totalLoss += lossFunction(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return totalLoss / total, correct / total


def trainModel(config, trainPaths, trainLabels, valLoader):
    """
    config keys:
      name    : label for tables/plots
      dropout : dropout before the final linear layer
      augment : use the augmenting transform for training images
      epochs  : maximum epochs (early stopping may end sooner)
      lr      : learning rate
    """
    print(f"\n=== {config['name']} ===")
    torch.manual_seed(seed)
    trainLoader= makeLoader(trainPaths, trainLabels,
                            augmentTransform if config["augment"] else plainTransform, shuffle=True)

    model= ScratchCNN(config["dropout"]).to(device)
    lossFunction= nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=config["lr"])
    print(f"  {sum(p.numel() for p in model.parameters()):,} trainable params, lr={config['lr']}")

    history= {"trainLoss": [], "trainAcc": [], "valLoss": [], "valAcc": []}
    best= {"trainAcc": 0, "valAcc": 0, "epoch": 0, "weights": copy.deepcopy(model.state_dict())}

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        totalLoss, correct, total= 0, 0, 0
        for images, labels in trainLoader:
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs= model(images)
            loss= lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
        trainLoss, trainAcc= totalLoss / total, correct / total
        valLoss, valAcc= evaluate(model, valLoader, lossFunction)

        for key, value in zip(["trainLoss", "trainAcc", "valLoss", "valAcc"], [trainLoss, trainAcc, valLoss, valAcc]):
            history[key].append(value)
        print(f"    epoch {epoch}: Train Loss={trainLoss:.4f} Acc={trainAcc:.4f} | Val Loss={valLoss:.4f} Acc={valAcc:.4f}")

        if valAcc > best["valAcc"]:
            best= {"trainAcc": trainAcc, "valAcc": valAcc, "epoch": epoch,
                   "weights": copy.deepcopy(model.state_dict())}
        elif epoch - best["epoch"] >= patience:
            print(f"    early stop at epoch {epoch}; best val acc {best['valAcc']:.4f} at epoch {best['epoch']}")
            break

    state= {k: v.detach().cpu().clone() for k, v in best["weights"].items()}
    del model, best["weights"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": config["name"], "config": config, "trainAcc": best["trainAcc"], "valAcc": best["valAcc"],
            "bestEpoch": best["epoch"], "history": history, "state": state}


def saveFigure(fileName):
    plt.savefig(fileName, dpi=150)
    print(f"  saved {fileName}")
    if showPlots:
        plt.show()
    plt.close()


def main():
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  <-- CPU only, training will be slow"))
    if not os.path.isdir(os.path.join(dataPath, "images")):
        raise FileNotFoundError(f"Could not find {os.path.join(dataPath, 'images')}. Set dataPath at the top of this file.")

    # ---------- load data (identical logic to the ResNet script) ----------
    imagesDir= os.path.join(dataPath, "images")
    classes= sorted(d for d in os.listdir(imagesDir)
                    if os.path.isdir(os.path.join(imagesDir, d)) and not d.startswith("."))[:numClasses]
    classToIdx= {classes[i]: i for i in range(len(classes))}
    print("Classes:", classes)

    trainPathsALL, trainLabelsALL= readSplit("train", classToIdx)
    testPaths, testLabels= readSplit("test", classToIdx)
    trainPaths, validationPath, trainLabels, validationLabels= train_test_split(
        trainPathsALL, trainLabelsALL, test_size=validationStop, random_state=seed
    )
    print(f"Train: {len(trainPaths)}, Validation: {len(validationPath)}, Test: {len(testPaths)}")
    print("(expected for 21 classes: Train 14175, Validation 1575, Test 5250)")

    if quickTest:
        print("*** quickTest: small subset, 2 configs, 1 epoch each ***")
        trainPaths, trainLabels= subsample(trainPaths, trainLabels, 96)
        validationPath, validationLabels= subsample(validationPath, validationLabels, 64)
        testPaths, testLabels= subsample(testPaths, testLabels, 64)

    valLoader= makeLoader(validationPath, validationLabels, plainTransform, shuffle=False)
    testLoader= makeLoader(testPaths, testLabels, plainTransform, shuffle=False)
    testLabelsArray= np.array(testLabels)

    # ---------- experiment grid ----------
    # C1 is the plain baseline; each later run changes one thing.
    # C2 and C4 are expected NOT to help; their numbers are worth reporting too.
    base= dict(epochs=epochs, lr=learningRate)
    cnnConfigs= [
        dict(base, name="C1 dropout 0.5, augmentation",      dropout=0.5, augment=True),
        dict(base, name="C2 dropout 0.5, no augmentation",   dropout=0.5, augment=False),
        dict(base, name="C3 no dropout, augmentation",       dropout=0.0, augment=True),
        dict(base, name="C4 no dropout, no augmentation",    dropout=0.0, augment=False),
        dict(base, name="C5 dropout 0.3, augmentation",      dropout=0.3, augment=True),
    ]
    if quickTest:
        cnnConfigs= [dict(c, epochs=1) for c in cnnConfigs[:2]]

    cnnResults= [trainModel(cfg, trainPaths, trainLabels, valLoader) for cfg in cnnConfigs]

    # ---------- configuration comparison ----------
    comparison= pd.DataFrame({
        "Run": [r["name"] for r in cnnResults],
        "Dropout": [r["config"]["dropout"] for r in cnnResults],
        "Augmentation": [r["config"]["augment"] for r in cnnResults],
        "Train Acc": [round(r["trainAcc"], 3) for r in cnnResults],
        "Val Acc": [round(r["valAcc"], 3) for r in cnnResults],
        "Train-Val Gap": [round(r["trainAcc"] - r["valAcc"], 3) for r in cnnResults],
        "Best Epoch": [r["bestEpoch"] for r in cnnResults],
    })
    print("\nScratch CNN configuration comparison:")
    print(comparison.to_string(index=False))
    comparison.to_csv("cnn_configurations.csv", index=False)

    # ---------- learning curves for every run ----------
    print("\nLearning curves:")
    for k, r in enumerate(cnnResults):
        h= r["history"]
        epochRange= range(1, len(h["trainLoss"]) + 1)
        plt.figure(figsize=(12, 4))
        for panel, (metric, ylabel, title) in enumerate([("Loss", "Cross-Entropy Loss", "Loss"), ("Acc", "Accuracy", "Accuracy")], start=1):
            plt.subplot(1, 2, panel)
            plt.plot(epochRange, h["train" + metric], label="Train")
            plt.plot(epochRange, h["val" + metric], label="Validation")
            plt.axvline(r["bestEpoch"], color="gray", linestyle="--", label="Best epoch")
            plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend()
        plt.suptitle(f"Scratch CNN learning curves: {r['name']}")
        plt.tight_layout()
        saveFigure(f"curves_cnn_{k + 1}.png")

    # ---------- evaluate the best run once on the test set ----------
    bestCNN= max(cnnResults, key=lambda r: r["valAcc"])
    print(f"\nBest CNN run: {bestCNN['name']} (val acc {bestCNN['valAcc']:.3f})")
    torch.save(bestCNN["state"], "scratch_cnn_best.pt")
    print("  saved scratch_cnn_best.pt")

    model= ScratchCNN(bestCNN["config"]["dropout"]).to(device)
    model.load_state_dict(bestCNN["state"])
    model.eval()
    allPreds, allProbs= [], []
    with torch.no_grad():
        for images, labels in testLoader:
            outputs= model(images.to(device))
            allProbs.append(torch.softmax(outputs, dim=1).cpu())
            allPreds.append(outputs.argmax(1).cpu())
    testPreds= torch.cat(allPreds).numpy()
    testProbs= torch.cat(allProbs).numpy()

    testAccuracy= (testPreds == testLabelsArray).mean()
    print(f"Scratch CNN test accuracy: {testAccuracy:.3f}   (random guessing: {1 / numClasses:.3f})")

    # per-class accuracy -- this CSV is what the ResNet script reads
    matrix= confusion_matrix(testLabelsArray, testPreds, labels=list(range(numClasses)))
    rowTotals= matrix.sum(axis=1)
    perClass= pd.DataFrame({
        "Class": classes,
        "Correct": matrix.diagonal(),
        "Total": rowTotals,
        "Accuracy": (matrix.diagonal() / np.maximum(rowTotals, 1)).round(3),
    }).sort_values("Accuracy", ascending=False)
    print("\nScratch CNN per-class test accuracy:")
    print(perClass.to_string(index=False))
    perClass.to_csv("per_class_scratch_cnn.csv", index=False)

    # confusion matrix
    matrixPercent= matrix / np.maximum(rowTotals, 1)[:, None]
    plt.figure(figsize=(11, 9))
    plt.imshow(matrixPercent, cmap="Blues")
    plt.colorbar(label="Fraction of true class")
    plt.xticks(range(numClasses), classes, rotation=90, fontsize=8)
    plt.yticks(range(numClasses), classes, fontsize=8)
    plt.title(f"Scratch CNN confusion matrix (test accuracy {testAccuracy:.1%})")
    plt.xlabel("Predicted Class"); plt.ylabel("True Class")
    plt.tight_layout()
    saveFigure("confusion_scratch_cnn.png")

    confusedPairs= sorted(((matrixPercent[i, j], classes[i], classes[j])
                           for i in range(numClasses) for j in range(numClasses) if i != j), reverse=True)
    print("\nScratch CNN top 5 confusions (true -> predicted):")
    for frac, trueName, predName in confusedPairs[:5]:
        print(f"  {trueName} -> {predName}: {frac:.1%}")

    # most confident mistakes
    wrong= np.where(testPreds != testLabelsArray)[0]
    wrong= wrong[np.argsort(-testProbs[wrong].max(axis=1))][:12]
    if len(wrong) > 0:
        plt.figure(figsize=(13, 10))
        for n, i in enumerate(wrong):
            plt.subplot(3, 4, n + 1)
            plt.imshow(Image.open(testPaths[i]).convert("RGB"))
            plt.title(f"True: {classes[testLabelsArray[i]]}\nPred: {classes[testPreds[i]]} ({testProbs[i].max():.0%})", fontsize=9)
            plt.axis("off")
        plt.suptitle("Scratch CNN most confident mistakes")
        plt.tight_layout()
        saveFigure("mistakes_scratch_cnn.png")

    print("\nDone. Send per_class_scratch_cnn.csv to whoever runs the ResNet script.")


if __name__ == "__main__":
    main()

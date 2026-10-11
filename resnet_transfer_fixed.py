# ============================================================
# ResNet-50 transfer learning on the Food-101 21-class subset
#
# Standalone script (does not import or depend on the scratch-CNN file):
#     python resnet_transfer.py
#
# What it does:
#   * trains several ResNet-50 configurations (some expected NOT to help)
#   * prints which layers are frozen / trainable for every run
#   * saves learning curves for every run as PNG files
#   * evaluates the best run once on the official test set
#   * saves per-class accuracy, confusion matrix, confident mistakes
#   * if per_class_scratch_cnn.csv (written by the scratch-CNN script)
#     is in this folder, builds the scratch-vs-ResNet comparison
#
# The class selection, official train/test files, 10% validation holdout,
# seed and test_size are identical to the scratch-CNN script, so both
# models are trained and scored on exactly the same split.
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
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

# ---------------- settings (edit these) ----------------
dataPath= r"D:\FALLSEMESTER\food-101\food-101\food-101"   # folder that contains images/ and meta/   <-- EDIT THIS
numClasses= 21
seed= 42
validationStop= 0.10
patience= 5
resnetImgSize= 224     # ImageNet weights expect ~224 px
batchSize= 32          # 32 fits a 6 GB GPU; raise to 64 only if you have more memory
numWorkers= 4          # parallel JPEG decoding; set to 0 if you hit multiprocessing errors
usePretrained= True    # downloads ImageNet weights (~100 MB) on first run
showPlots= False       # True = also pop up each figure (PNG files are always saved)
quickTest= True       # True = tiny end-to-end check (2 configs, 1 epoch, small subset)
scratchCsv= "per_class_scratch_cnn.csv"

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean= [0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]

resnetAugment= transforms.Compose([
    transforms.RandomResizedCrop(resnetImgSize, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # images are oversaturated/noisy
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
resnetPlain= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(resnetImgSize),
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
def buildResNet(dropout):
    weights= ResNet50_Weights.IMAGENET1K_V1 if usePretrained else None
    model= models.resnet50(weights=weights)
    # replace the 1000-class ImageNet head with a 21-class head
    model.fc= nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, numClasses))
    return model.to(device)


def setTrainable(model, unfreeze):
    # unfreeze: top-level blocks to train, e.g. ["layer4"]; the new fc head is always trained
    for name, param in model.named_parameters():
        block= name.split(".")[0]
        param.requires_grad= (block == "fc") or (block in unfreeze)


def frozenBatchNormEval(model):
    # Frozen means frozen: keep BatchNorm layers whose weights are frozen in eval mode,
    # otherwise their running statistics would keep drifting during training.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def frozenSummary(model):
    rows= []
    for block in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]:
        params= [p for n, p in model.named_parameters() if n.split(".")[0] == block]
        if not params:
            continue
        rows.append({"Block": block,
                     "Params": sum(p.numel() for p in params),
                     "Status": "trainable" if all(p.requires_grad for p in params) else "frozen"})
    return pd.DataFrame(rows)


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


def runStage(model, trainLoader, valLoader, numEpochs, learningRate, history, label):
    lossFunction= nn.CrossEntropyLoss()
    trainable= [p for p in model.parameters() if p.requires_grad]
    optimizer= torch.optim.Adam(trainable, lr=learningRate)
    print(f"  {label}: {sum(p.numel() for p in trainable):,} trainable params, lr={learningRate}")

    best= {"trainAcc": 0, "valAcc": 0, "epoch": 0, "weights": copy.deepcopy(model.state_dict())}
    bestLocalEpoch= 0
    offset= len(history["trainLoss"])

    for epoch in range(1, numEpochs + 1):
        model.train()
        frozenBatchNormEval(model)
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
            best= {"trainAcc": trainAcc, "valAcc": valAcc, "epoch": offset + epoch,
                   "weights": copy.deepcopy(model.state_dict())}
            bestLocalEpoch= epoch
        elif epoch - bestLocalEpoch >= patience:
            print(f"    early stop at epoch {epoch}; best val acc {best['valAcc']:.4f} at epoch {bestLocalEpoch}")
            break

    model.load_state_dict(best["weights"])   # the next stage starts from this stage's best weights
    return best


def trainResNet(config, trainPaths, trainLabels, valLoader):
    """
    config keys:
      name        : label for tables/plots
      unfreeze    : blocks trained in stage 2 (fc always trained); [] = head only
      augment     : use the augmenting transform for training images
      dropout     : dropout before the new fc layer
      headEpochs  : stage-1 epochs (backbone frozen, head only)
      ftEpochs    : stage-2 epochs (unfrozen blocks + head); 0 skips stage 2
      headLR, ftLR: learning rates for the two stages
    """
    print(f"\n=== {config['name']} ===")
    torch.manual_seed(seed)
    trainLoader= makeLoader(trainPaths, trainLabels, resnetAugment if config["augment"] else resnetPlain, shuffle=True)

    model= buildResNet(config["dropout"])
    history= {"trainLoss": [], "trainAcc": [], "valLoss": [], "valAcc": []}

    # Stage 1: backbone frozen, train the new head
    setTrainable(model, unfreeze=[])
    print(frozenSummary(model).to_string(index=False))
    overall= runStage(model, trainLoader, valLoader, config["headEpochs"], config["headLR"], history, "Stage 1 (head only)")
    stageSwitch= len(history["trainLoss"])

    # Stage 2: unfreeze chosen blocks, small learning rate
    if config["ftEpochs"] > 0 and config["unfreeze"]:
        setTrainable(model, unfreeze=config["unfreeze"])
        print(frozenSummary(model).to_string(index=False))
        stage2= runStage(model, trainLoader, valLoader, config["ftEpochs"], config["ftLR"], history,
                         f"Stage 2 ({'+'.join(config['unfreeze'])} + head)")
        if stage2["valAcc"] >= overall["valAcc"]:
            overall= stage2

    state= {k: v.detach().cpu().clone() for k, v in overall["weights"].items()}
    del model, overall["weights"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"name": config["name"], "config": config, "trainAcc": overall["trainAcc"], "valAcc": overall["valAcc"],
            "bestEpoch": overall["epoch"], "stageSwitch": stageSwitch, "history": history, "state": state}


def saveFigure(fileName):
    plt.savefig(fileName, dpi=150)
    print(f"  saved {fileName}")
    if showPlots:
        plt.show()
    plt.close()


def main():
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  <-- CPU only, training will be very slow"))
    if not os.path.isdir(os.path.join(dataPath, "images")):
        raise FileNotFoundError(f"Could not find {os.path.join(dataPath, 'images')}. Set dataPath at the top of this file.")

    # ---------- load data (same logic as the scratch-CNN script) ----------
    classes= sorted(os.listdir(os.path.join(dataPath, "images")))[:numClasses]
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

    valLoader= makeLoader(validationPath, validationLabels, resnetPlain, shuffle=False)
    testLoader= makeLoader(testPaths, testLabels, resnetPlain, shuffle=False)
    testLabelsArray= np.array(testLabels)

    # ---------- experiment grid ----------
    # Ordered as a story for the slides: simplest transfer setup first, then one change at a time.
    # R4 and R5 are expected NOT to help; their numbers are worth reporting too.
    base= dict(dropout=0.5, augment=True, headEpochs=5, ftEpochs=15, headLR=1e-3, ftLR=1e-4)
    resnetConfigs= [
        dict(base, name="R1 frozen backbone, head only",           unfreeze=[], ftEpochs=0, headEpochs=15),
        dict(base, name="R2 + unfreeze layer4",                    unfreeze=["layer4"]),
        dict(base, name="R3 + unfreeze layer3 and layer4",         unfreeze=["layer3", "layer4"]),
        dict(base, name="R4 layer4, no augmentation",              unfreeze=["layer4"], augment=False),
        dict(base, name="R5 layer4, fine-tune LR 1e-3 (too high)", unfreeze=["layer4"], ftLR=1e-3),
        dict(base, name="R6 layer4, no dropout",                   unfreeze=["layer4"], dropout=0.0),
    ]
    if quickTest:
        resnetConfigs= [dict(c, headEpochs=1, ftEpochs=(1 if c["ftEpochs"] else 0)) for c in resnetConfigs[:2]]

    resnetResults= [trainResNet(cfg, trainPaths, trainLabels, valLoader) for cfg in resnetConfigs]

    # ---------- configuration comparison ----------
    resnetComparison= pd.DataFrame({
        "Run": [r["name"] for r in resnetResults],
        "Unfrozen blocks": ["+".join(r["config"]["unfreeze"]) or "none (head only)" for r in resnetResults],
        "Augment": [r["config"]["augment"] for r in resnetResults],
        "Dropout": [r["config"]["dropout"] for r in resnetResults],
        "FT LR": [r["config"]["ftLR"] if r["config"]["ftEpochs"] > 0 else "-" for r in resnetResults],
        "Train Acc": [round(r["trainAcc"], 3) for r in resnetResults],
        "Val Acc": [round(r["valAcc"], 3) for r in resnetResults],
        "Train-Val Gap": [round(r["trainAcc"] - r["valAcc"], 3) for r in resnetResults],
        "Best Epoch": [r["bestEpoch"] for r in resnetResults],
    })
    print("\nResNet-50 configuration comparison:")
    print(resnetComparison.to_string(index=False))
    resnetComparison.to_csv("resnet_configurations.csv", index=False)

    # ---------- learning curves for every run ----------
    print("\nLearning curves:")
    for k, r in enumerate(resnetResults):
        h= r["history"]
        epochRange= range(1, len(h["trainLoss"]) + 1)
        plt.figure(figsize=(12, 4))
        for panel, (metric, ylabel, title) in enumerate([("Loss", "Cross-Entropy Loss", "Loss"), ("Acc", "Accuracy", "Accuracy")], start=1):
            plt.subplot(1, 2, panel)
            plt.plot(epochRange, h["train" + metric], label="Train")
            plt.plot(epochRange, h["val" + metric], label="Validation")
            plt.axvline(r["bestEpoch"], color="gray", linestyle="--", label="Best epoch")
            if r["stageSwitch"] < len(epochRange):
                plt.axvline(r["stageSwitch"] + 0.5, color="red", linestyle=":", label="Unfreeze")
            plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.legend()
        plt.suptitle(f"ResNet-50 learning curves: {r['name']}")
        plt.tight_layout()
        saveFigure(f"curves_resnet_{k + 1}.png")

    # ---------- evaluate the best run once on the test set ----------
    bestResNet= max(resnetResults, key=lambda r: r["valAcc"])
    print(f"\nBest ResNet run: {bestResNet['name']} (val acc {bestResNet['valAcc']:.3f})")
    torch.save(bestResNet["state"], "resnet50_best.pt")
    print("  saved resnet50_best.pt")

    resnet= buildResNet(bestResNet["config"]["dropout"])
    resnet.load_state_dict(bestResNet["state"])
    resnet.eval()
    allPreds, allProbs= [], []
    with torch.no_grad():
        for images, labels in testLoader:
            outputs= resnet(images.to(device))
            allProbs.append(torch.softmax(outputs, dim=1).cpu())
            allPreds.append(outputs.argmax(1).cpu())
    resnetPreds= torch.cat(allPreds).numpy()
    resnetProbs= torch.cat(allProbs).numpy()

    resnetTestAccuracy= (resnetPreds == testLabelsArray).mean()
    print(f"ResNet-50 test accuracy: {resnetTestAccuracy:.3f}   (random guessing: {1 / numClasses:.3f})")

    # per-class accuracy
    resnetMatrix= confusion_matrix(testLabelsArray, resnetPreds, labels=list(range(numClasses)))
    rowTotals= resnetMatrix.sum(axis=1)
    resnetPerClass= pd.DataFrame({
        "Class": classes,
        "Correct": resnetMatrix.diagonal(),
        "Total": rowTotals,
        "Accuracy": (resnetMatrix.diagonal() / np.maximum(rowTotals, 1)).round(3),
    }).sort_values("Accuracy", ascending=False)
    print("\nResNet-50 per-class test accuracy:")
    print(resnetPerClass.to_string(index=False))
    resnetPerClass.to_csv("per_class_resnet50.csv", index=False)

    # confusion matrix
    resnetMatrixPercent= resnetMatrix / np.maximum(rowTotals, 1)[:, None]
    plt.figure(figsize=(11, 9))
    plt.imshow(resnetMatrixPercent, cmap="Blues")
    plt.colorbar(label="Fraction of true class")
    plt.xticks(range(numClasses), classes, rotation=90, fontsize=8)
    plt.yticks(range(numClasses), classes, fontsize=8)
    plt.title(f"ResNet-50 confusion matrix (test accuracy {resnetTestAccuracy:.1%})")
    plt.xlabel("Predicted Class"); plt.ylabel("True Class")
    plt.tight_layout()
    saveFigure("confusion_resnet50.png")

    confusedPairs= sorted(((resnetMatrixPercent[i, j], classes[i], classes[j])
                           for i in range(numClasses) for j in range(numClasses) if i != j), reverse=True)
    print("\nResNet-50 top 5 confusions (true -> predicted):")
    for frac, trueName, predName in confusedPairs[:5]:
        print(f"  {trueName} -> {predName}: {frac:.1%}")

    # most confident mistakes
    wrong= np.where(resnetPreds != testLabelsArray)[0]
    wrong= wrong[np.argsort(-resnetProbs[wrong].max(axis=1))][:12]
    if len(wrong) > 0:
        plt.figure(figsize=(13, 10))
        for n, i in enumerate(wrong):
            plt.subplot(3, 4, n + 1)
            plt.imshow(Image.open(testPaths[i]).convert("RGB"))
            plt.title(f"True: {classes[testLabelsArray[i]]}\nPred: {classes[resnetPreds[i]]} ({resnetProbs[i].max():.0%})", fontsize=9)
            plt.axis("off")
        plt.suptitle("ResNet-50 most confident mistakes")
        plt.tight_layout()
        saveFigure("mistakes_resnet50.png")

    # ---------- scratch CNN vs ResNet-50, and results vs literature ----------
    scratchTestAcc= None
    if os.path.exists(scratchCsv):
        scratch= pd.read_csv(scratchCsv)
        scratchTestAcc= round(scratch["Correct"].sum() / scratch["Total"].sum(), 3)
        perClassGain= pd.DataFrame({"Class": classes})
        perClassGain["Scratch CNN"]= perClassGain["Class"].map(scratch.set_index("Class")["Accuracy"])
        perClassGain["ResNet-50"]= perClassGain["Class"].map(resnetPerClass.set_index("Class")["Accuracy"])
        perClassGain["Gain"]= (perClassGain["ResNet-50"] - perClassGain["Scratch CNN"]).round(3)
        print("\nPer-class gain from transfer learning:")
        print(perClassGain.sort_values("Gain", ascending=False).to_string(index=False))
        perClassGain.to_csv("per_class_gain.csv", index=False)
    else:
        print(f"\n({scratchCsv} not found in this folder, skipping the scratch-vs-ResNet per-class comparison)")

    # Fill the literature rows from the papers/write-ups you surveyed. Note whether each reports on
    # the full 101 classes or a subset: that is part of explaining the gap (21-way is easier than 101-way).
    resultsTable= pd.DataFrame([
        {"Model": "Scratch CNN (ours)", "Classes": numClasses, "Test Acc": scratchTestAcc, "Source": "this project"},
        {"Model": f"ResNet-50 fine-tuned (ours): {bestResNet['name']}", "Classes": numClasses,
         "Test Acc": round(float(resnetTestAccuracy), 3), "Source": "this project"},
        {"Model": "<paper 1 model>", "Classes": 101, "Test Acc": None, "Source": "<author, year>"},
        {"Model": "<paper 2 model>", "Classes": 101, "Test Acc": None, "Source": "<author, year>"},
        {"Model": "<paper 3 model>", "Classes": 101, "Test Acc": None, "Source": "<author, year>"},
    ])
    print("\nResults vs literature (fill in the literature rows):")
    print(resultsTable.to_string(index=False))
    resultsTable.to_csv("results_vs_literature.csv", index=False)


if __name__ == "__main__":
    main()

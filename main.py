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

path= 'data.csv'
numClasses= 21
seed= 42
imgSize= 128
batchSize= 64
epochs= 30
validationStop= 0.10
learningRate= 0.001
patience= 5

#load data
classes= sorted(os.listdir(os.path.join(path, "images")))[:numClasses]
classToIdx= {classes[i]: i for i in range(len(classes))}
print("Classes:", classes)

def readSplit(fileName):
    paths, labels= [], []
    with open(os.path.join(path, "meta", fileName)) as f:
        for line in f:
            line= line.strip()
            className= line.split(",")[0]
            if className in classes:
                paths.append(os.path.join(path, "images", line + ".jpg"))
                labels.append(classToIdx[className])
    return paths, labels

trainPathsALL, trainLabelsALL= readSplit("train.csv")
testPaths, testLabels= readSplit("test.csv")
trainPaths, validationPath, trainLabels, validationLabels= train_test_split(
    trainPathsALL, trainLabelsALL, test_size=validationStop, random_state=seed
)
print("Train: {len(trainPaths)}, Validation: {len(validationPath)}, Test: {len(testPaths)}")

#image preprocessing 
mean= [0.485, 0.456, 0.406]
std= [0.229, 0.224, 0.225]

augmentTransform= transforms.Compose([
    transforms.RandomResizedCrop(imgSize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
planTransform= transforms.Compose([
    transforms.Resize(146),
    transforms.ToTensor(),
    transforms.CenterCrop(imgSize),
    transforms.Normalize(mean, std),
])


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
    
loader= DataLoader(FoodDataset(validationPath, validationLabels, planTransform), batch_size=batchSize, shuffle=False, num_workers=0)
testLoader= DataLoader(FoodDataset(testPaths, testLabels, planTransform), batch_size=batchSize, shuffle=False, num_workers=0)

#Scratch CNN
class ScratchCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features= nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=2),
        )
        self.classifier= nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, numClasses)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
    
#training function 
def trainModel(dropout, augment):
    trainTransform= augmentTransform if augment else planTransform
    trainLoader= DataLoader(FoodDataset(trainPaths, trainLabels, trainTransform), batch_size=batchSize, shuffle=True, num_workers=0)
    
    model= ScratchCNN(dropout).to(device)
    lossFunction= nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=learningRate)
    
    history= {"trainLoss": [], "trainAcc": [], "valLoss": [], "valAcc": []}
    bestValAcc, bestEpoch, bestModel= 0, 0, None
    
    for epoch in range(1, epochs + 1):
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
            trainLoss= totalLoss / total
            trainAcc= correct / total
        
        #validation
        model.eval()
        totalLoss, correct, total= 0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels= images.to(device), labels.to(device)
                outputs= model(images)
                
                totalLoss += lossFunction(outputs, labels).sum().item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
                
            valLoss= totalLoss / total
            valAcc= correct / total
            
            history["trainLoss"].append(trainLoss)
            history["trainAcc"].append(trainAcc)
            history["valLoss"].append(valLoss)
            history["valAcc"].append(valAcc)
            print(f"Epoch {epoch}: Train Loss={trainLoss:.4f}, Train Acc={trainAcc:.4f}, Val Loss={valLoss:.4f}, Val Acc={valAcc:.4f}")
            
            #early stop for best of everything 
            if valAcc > bestValAcc:
                bestValAcc= valAcc
                bestEpoch= epoch
                bestModel= copy.deepcopy(model.state_dict())
            elif epoch - bestEpoch >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc={bestValAcc:.4f} at epoch {bestEpoch}.")
                break
            
        model.load_state_dict(bestWeights)
        return model, history, bestValAcc, bestEpoch

# Part D: Comparative analysis and learning curves
comparison = pd.DataFrame({
    "Dropout": [r[0] for r in cnnResults],
    "Augmentation": [r[1] for r in cnnResults],
    "Training Accuracy": [round(r[2], 3) for r in cnnResults],
    "Validation Accuracy": [round(r[3], 3) for r in cnnResults],
    "Train-Val Gap": [round(r[2] - r[3], 3) for r in cnnResults],
    "Best Epoch": [r[4] for r in cnnResults],
})
print("\nComparative Analysis:")
print(comparison)
 
for r in cnnResults:
    history = r[5]
    epochRange = range(1, len(history["trainLoss"]) + 1)
    plt.figure(figsize=(12, 4))
 
    plt.subplot(1, 2, 1)
    plt.plot(epochRange, history["trainLoss"], label="Train")
    plt.plot(epochRange, history["valLoss"], label="Validation")
    plt.axvline(r[4], color="gray", linestyle="--", label="Best epoch")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
 
    plt.subplot(1, 2, 2)
    plt.plot(epochRange, history["trainAcc"], label="Train")
    plt.plot(epochRange, history["valAcc"], label="Validation")
    plt.axvline(r[4], color="gray", linestyle="--", label="Best epoch")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
 
    plt.suptitle(f"Scratch CNN Learning Curves (dropout={r[0]}, augmentation={r[1]})")
    plt.tight_layout()
    plt.show()
 
'''
1. What do the learning curves tell you? (underfitting, overfitting, where training stopped and why)
 
2. Which change helped the most, and did any change not help?
'''
 
 
# Part E: Final test evaluation of the best model
# Pick the run with the highest validation accuracy and score it once on the test set.
best = max(cnnResults, key=lambda r: r[3])
bestModel = best[6]
print(f"\nBest model: dropout={best[0]}, augmentation={best[1]}, validation accuracy {best[3]:.3f}")
 
bestModel.eval()
allPreds, allProbs = [], []
with torch.no_grad():
    for images, labels in testLoader:
        outputs = bestModel(images.to(device))
        allProbs.append(torch.softmax(outputs, dim=1).cpu())
        allPreds.append(outputs.argmax(1).cpu())
testPreds = torch.cat(allPreds).numpy()
testProbs = torch.cat(allProbs).numpy()
testLabelsArray = np.array(testLabels)
 
testAccuracy = (testPreds == testLabelsArray).mean()
print(f"Test Accuracy: {testAccuracy:.3f}   (random guessing: {1 / numClasses:.3f})")
 
# Per-class accuracy: of the test images that really are class k, how many were predicted as k
matrix = confusion_matrix(testLabelsArray, testPreds)
perClass = pd.DataFrame({
    "Class": classes,
    "Correct": matrix.diagonal(),
    "Total": matrix.sum(axis=1),
    "Accuracy": (matrix.diagonal() / matrix.sum(axis=1)).round(3),
}).sort_values("Accuracy", ascending=False)
print("\nPer-Class Test Accuracy:")
print(perClass.to_string(index=False))
perClass.to_csv("per_class_scratch_cnn.csv", index=False)
 
 
# Visualization: confusion matrix
matrixPercent = matrix / matrix.sum(axis=1, keepdims=True)
plt.figure(figsize=(11, 9))
plt.imshow(matrixPercent, cmap="Blues")
plt.colorbar(label="Fraction of true class")
plt.xticks(range(numClasses), classes, rotation=90, fontsize=8)
plt.yticks(range(numClasses), classes, fontsize=8)
plt.title(f"Scratch CNN Confusion Matrix (test accuracy {testAccuracy:.1%})")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()
 
# Most confused pairs
confusedPairs = []
for i in range(numClasses):
    for j in range(numClasses):
        if i != j:
            confusedPairs.append((matrixPercent[i, j], classes[i], classes[j]))
confusedPairs.sort(reverse=True)
print("\nTop 5 Confusions (true -> predicted):")
for p in confusedPairs[:5]:
    print(f"{p[1]} -> {p[2]}: {p[0]:.1%}")
 
 
# Visualization: examples the model got wrong (most confident mistakes first)
wrong = np.where(testPreds != testLabelsArray)[0]
wrong = wrong[np.argsort(-testProbs[wrong].max(axis=1))][:12]
 
plt.figure(figsize=(13, 10))
for n, i in enumerate(wrong):
    plt.subplot(3, 4, n + 1)
    plt.imshow(Image.open(testPaths[i]).convert("RGB"))
    plt.title(f"True: {classes[testLabelsArray[i]]}\nPred: {classes[testPreds[i]]} "
              f"({testProbs[i].max():.0%})", fontsize=9)
    plt.axis("off")
plt.suptitle("Most Confident Mistakes")
plt.tight_layout()
plt.show()
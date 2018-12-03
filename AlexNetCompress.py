import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import json, string
from torch.autograd import Variable
import torch.nn as nn
import numpy
import torch
import csv
import sys
import matplotlib.pyplot as plt
import os

path = 'data/../images'
for image_path in os.listdir(path):
    input_path = os.path.join(path, image_path)

    Alexnet = models.alexnet(pretrained=True)

    # Define the image pre-processing function.
    preprocessFn = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    # Load the imagenet class names.
    imagenetClasses = {int(idx): entry[1] for (idx, entry) in json.load(open('imagenet_class_index.json')).items()}


    # Set to evaluation mode so Dropoff layers don't add randomness.
    Alexnet.eval()

    # unsqueeze(0) adds a dummy batch dimension which is required for all models in pytorch.
    #testImage = sys.argv[1]
    image = Image.open(input_path).convert('RGB')
    inputVar =  Variable(preprocessFn(image).unsqueeze(0))
    predictions = Alexnet(inputVar)

    # Decode the top 10 classes predicted for this image.
    # We need to apply softmax because the model outputs the last linear layer activations and not softmax scores.
    probs, indices = (-nn.Softmax(dim = 1)(predictions).data).sort()
    probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
    preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

    print(preds)

    groundTruth = imagenetClasses[indices[0]]

    Alexnet.state_dict().keys()

    ######################
    with open('mycsv.csv', 'w', newline = '') as f:
        writer = csv.writer(f)

        writer.writerow(['Quality','Layer','Size','Kept', 'Removed','PercentRemoved','Prediction Lable','Prediction Prob'])

        model_dict = Alexnet.state_dict()

        qualities = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

        layers = ['features.0.weight',
              'features.3.weight',
              'features.6.weight',
              'features.8.weight',
              'features.10.weight',
              'classifier.1.weight',
              'classifier.4.weight',
              'classifier.6.weight'
              ]
        labelAccuracy = []
        probAccuracy = []
        reduction = []

        for quality in qualities:
                  print('Quality: ', quality)
                  totalSize = 0
                  totalRemoved = 0
                  totalProb = 0
                  for layer in layers:
                      opLayer = Alexnet.state_dict()[layer].numpy() #opLayer is the layer currently being worked on
                      stdDis = numpy.std(opLayer)*quality #Standard Distribution times a Quality Parameter

                      mask = abs(opLayer) >= stdDis

                      size = mask.size
                      totalSize += size
                      #print(layer, 'Size: ',mask.size)
                      kept = mask.sum()
                      #print(layer, '{} ({:.2f}%) kept'.format(mask.sum(), mask.sum()/mask.size * 100)) #The Number of non-zero elements divided by the Original Size
                      removed = mask.size - mask.sum()
                      totalRemoved += removed
                      #print(layer, '{} ({:.2f}%) removed'.format(mask.size - mask.sum(), 100 - mask.sum()/mask.size * 100)) #Number elements set to zero (The Original Size minus the .sum())
                      percentRemoved = 100 - mask.sum()/mask.size * 100
                      #print('\n')

                      result = torch.from_numpy(opLayer * mask)

                      model_dict[layer] = result



                      Alexnet.load_state_dict(model_dict)


                      predictions = Alexnet(inputVar)

                      # Decode the top 10 classes predicted for this image.
                      # We need to apply softmax because the model outputs the last linear layer activations and not softmax scores.
                      probs, indices = (-nn.Softmax(dim = 1)(predictions).data).sort()
                      probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
                      preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]

                      predictionLabel = imagenetClasses[indices[0]]
                      predictionProb = probs[0]





                      #print('New Pred: ')
                      #print(preds[0])
                      #print('\n')
                      writer.writerow([quality,layer,size,kept,removed,percentRemoved,predictionLabel,predictionProb])
                  labelAccuracy.append(predictionLabel)
                  probAccuracy.append(predictionProb)
                  reduction.append(totalRemoved/totalSize)
                  print('Total Size: ',totalSize)
                  print('Total Removed: ',totalRemoved)
                  print('{:.2f}% Removed'.format(totalRemoved/totalSize))
                  print('\n')
        plt.plot(qualities, probAccuracy, label = 'Accuracy')
        plt.plot(qualities, reduction, label = 'Reduction')
        plt.title('Ground Truth:\n'+groundTruth)
        plt.legend()
        plt.show()

        failIndex = 0
        for index in range(0, 10):
            if labelAccuracy[index] == groundTruth:
                failIndex += 1
            else:
                continue
        print('Model fails at Quality Parameter: ', qualities[failIndex-1])

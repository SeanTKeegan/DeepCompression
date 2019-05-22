import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import numpy
import torch
import matplotlib.pyplot as plt
import os
import json
import pickle
import collections
import huffman 
import csv 
import sys
from scipy.sparse import csr_matrix

###############
Alexnet = models.alexnet(pretrained=True)
model_dict = Alexnet.state_dict()
pickle_out = open("model_dict.pickle","wb")
pickle.dump(model_dict, pickle_out)
pickle_out.close()
pickle_in = open("model_dict.pickle","rb")
model_dict = pickle.load(pickle_in)

def threshold(quality1, quality2):
    
    totalRemoved = 0
    print('Evaluating Feature Layers...')
    print('\n')
    featureLayers = ['features.0.weight',
              'features.3.weight',
              'features.6.weight',
              'features.8.weight',
              'features.10.weight']
    classifierLayers = ['classifier.1.weight',
              'classifier.4.weight',
              'classifier.6.weight']
    
    # layers = featureLayers
    #model_dict = Alexnet.state_dict()
    pickle_in = open("model_dict.pickle","rb")
    model_dict = pickle.load(pickle_in)
    for layer in featureLayers:
        # opLayer is the layer currently being worked on
        opLayer = model_dict[layer].numpy()
        # Standard Distribution times a Quality Parameter
        stdDis = numpy.std(opLayer)*quality1
        mask = abs(opLayer) >= stdDis
        size = mask.size
        print(size)
        print(mask.sum())
        print(mask.size - mask.sum())
        removed = mask.size - mask.sum()
        print(print(layer, '{} ({:.2f}%) removed'.format(mask.size - mask.sum(), 100 - mask.sum()/mask.size * 100)))
        print('\n')
        result = torch.from_numpy(opLayer * mask)
        model_dict[layer] = result
        Alexnet.load_state_dict(model_dict)
        totalRemoved += removed
    # layers = classifierLAyers
    for layer in classifierLayers:
        # opLayer is the layer currently being worked on
        opLayer = model_dict[layer].numpy()
        # Standard Distribution times a Quality Parameter
        stdDis = numpy.std(opLayer)*quality2
        mask = abs(opLayer) >= stdDis
        size = mask.size
        print(size, 'total')
        print(mask.sum(), 'unaffected')
        print(mask.size - mask.sum(), 'removed')
        removed = mask.size - mask.sum()
        print(print(layer, '{} ({:.2f}%) removed'.format(mask.size - mask.sum(), 100 - mask.sum()/mask.size * 100)))
        print('\n')
        result = torch.from_numpy(opLayer * mask)
        model_dict[layer] = result
        Alexnet.load_state_dict(model_dict)
        totalRemoved += removed
    return totalRemoved, model_dict
    
###############
        
def evaluate():
    predictions = Alexnet(inputVar)
    # Decode the top 10 classes predicted for this image.
    # We need to apply softmax because the model outputs the last linear layer activations and not softmax scores.
    probs, indices = (-nn.Softmax(dim=1)(predictions).data).sort()
    probs = (-probs).numpy()[0][:10]
    indices = indices.numpy()[0][:10]
    preds = [imagenetClasses[idx] + ': '+ str(prob) for (prob, idx) in zip(probs, indices)]
    #print(preds)
#    plt.plot(abs(predictions.data.numpy()[0]), label = 'Pruned', alpha = 0.7)
#    plt.legend()
    prediction = imagenetClasses[indices[0]]
    Prob = probs[0]
    return prediction, Prob
        
        
##############
def Quantise(model_dict_threshold, bits):
    FCLayers = ['classifier.1.weight', 'classifier.4.weight', 'classifier.6.weight']
    uniqueBefore = 0
    uniqueAfter = 0 
    
    
    for layers in FCLayers:
        print('Quantising: ', layers)
        layer = model_dict_threshold[layers].numpy()
        layer.shape
        print("Unique before: ", numpy.unique(layer).size)
        uniqueBefore += numpy.unique(layer).size
        print("Quantising...")
    
        numpy.amin(layer)
        numpy.amax(layer)
    
        bins = numpy.linspace(numpy.amin(layer), numpy.amax(layer), bits)
        bins = numpy.append(bins, [0])
        bins.sort()
        digi = numpy.digitize(layer, bins)
        digi[:] = [x - 1 for x in digi] #list comprehention since all elements of digi need to be reduced by 1
    
        for index, x in numpy.ndenumerate(layer):
            layer[index] = bins[digi[index]]
        print('Quantising Complete')
        print("Unique After: ", numpy.unique(layer).size)
        uniqueAfter += numpy.unique(layer).size
        
        #Add new layer back into state_dict
        result = torch.from_numpy(layer)
        model_dict_threshold[layers] = result
        
        
        print('Finding Frequencies...')
        # Get unique values from complete 2D array
        uniqueValues, count = numpy.unique(layer, return_counts = True)
    
    
        full = numpy.sum(count)
        freqs = numpy.arange(len(count), dtype = float)
    
        for x in range(0, len(count)):
            freq = count[x]/full
            freqs[x] = freq
    
        tup = numpy.vstack((uniqueValues, freqs)).T
    
    
        test = [list(item) for item in tup]
        print('Frequencies Found')
    
    #Alexnet.load_state_dict(model_dict)
    return model_dict_threshold, uniqueBefore, uniqueAfter


##############
def encode(quantised_model_dict):
    print("\n")
    print("Generating Codebook...")
    concat = numpy.concatenate((numpy.ravel(quantised_model_dict['classifier.1.weight'].numpy()),numpy.ravel(quantised_model_dict['classifier.4.weight'].numpy()),numpy.ravel(quantised_model_dict['classifier.6.weight'].numpy())),axis = 0)
    huffmanUnique = numpy.unique(concat)
    codebook = huffman.codebook(collections.Counter(concat.tolist()).items())
    print("Codebook Generated")
    return codebook, huffmanUnique.size

    
###############
def uniqueOriginal():
    Alexnet = models.alexnet(pretrained=True)
    model_dict = Alexnet.state_dict()
    concat = numpy.concatenate((numpy.ravel(model_dict['classifier.1.weight'].numpy()),numpy.ravel(model_dict['classifier.4.weight'].numpy()),numpy.ravel(model_dict['classifier.6.weight'].numpy())),axis = 0)
    originalUnique = numpy.unique(concat)
    return originalUnique.size
        
###############

def calculateSize():
    
    model_dict = Alexnet.state_dict()
    totalSize = 0
    for key, value in model_dict.items():
        totalSize += model_dict[key].numpy().size
        #print(key)
    return totalSize

def sizeOnDisk(model, value):
    with open('AnalysisSize2.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Layer','Value','oSize in Bytes', 'cSize in Bytes'])
            FCLayers = ['classifier.1.weight', 'classifier.4.weight', 'classifier.6.weight']
            for layers in FCLayers:
                layer = model[layers]
                workLayer = layer.numpy().flatten()
                csr = csr_matrix(workLayer)
                writer.writerow([layers, value, sys.getsizeof(workLayer), sys.getsizeof(csr)])


path = './lion'
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


    # Set to evaluation mode so Drop-out does not add randomness.
    # Dropout is used to prevent over-fitting by randomly selecting weights to drop
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
    print('\n')

    groundTruth = imagenetClasses[indices[0]]
    groundTruthProb = probs[0]
#    plt.plot(abs(predictions.data.numpy()[0]), label = 'Original')
#    plt.xlabel('Class')
#    plt.ylabel('Probability')
#    plt.legend()
#    plt.title(groundTruth)
#    plt.show()

    Alexnet.state_dict().keys()
    with open('AnalysisValuesUmbrella.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Threshold Value','Total','Removed', 'oUniqueValues', 'oClass: class', 'oClass: prob','cClass: class','cClass: prob', 'Loss','PoF'])
#        testValues = numpy.arange(0.0, 2.1, 0.1)
        testValues = [1.0]
        for value in testValues:
            removed, model_dict_threshold = threshold(0.0, value)
            pickle_out = open("model_dict_threshold.pickle","wb")
            pickle.dump(model_dict_threshold, pickle_out)
            pickle_out.close()
            pickle_in = open("model_dict_threshold.pickle","rb")
            thres_dict = pickle.load(pickle_in)
            
            model_dict_quantised, uniqueBefore, uniqueAfter = Quantise(thres_dict, 256)
            pickle_out = open("model_dict_quantised.pickle","wb")
            pickle.dump(model_dict_quantised, pickle_out)
            pickle_out.close()
            pickle_in = open("model_dict_quantised.pickle","rb")
            quan_dict = pickle.load(pickle_in)
            Alexnet.load_state_dict(model_dict_quantised)
            
            sizeOnDisk(quan_dict, value)
            
#            huffmanCodebook, huffmanUnique = encode(quan_dict)
            
            originalUnique = uniqueOriginal()
            
            prediction, Prob = evaluate()
            size = calculateSize()
            
            
            
            
            print('\n')
            print('******************')
            print('Total: ', size,'Removed: ', removed)
            print('{:.2f}% Size reduction'.format(removed/size * 100))
            print('Quantisation: ')
            
            
            
            #print('Unique Before: ', uniqueBefore,'Unique After: ', uniqueAfter)
            
            #print('{:.3f}% less values'.format(100 - ((uniqueAfter/uniqueBefore) * 100)))
            print('Original Unique Values: ', originalUnique)
#            print('Resulting Unique Values: ', huffmanUnique)
            
            print('Original Classification: ', groundTruth, " ", groundTruthProb)
            print('Pruned Classification: ', prediction, " ", Prob)
            loss = groundTruthProb - Prob
            PoF = False
            if groundTruth == prediction: 
                print('Loss of Accuracy: ', loss)
            else: #In case the model no longer classifies to the same class
                print('Model has lost all Accuracy.')
                PoF = True
            print('\n')
            writer.writerow([value, size, removed, originalUnique, groundTruth, groundTruthProb, prediction, Prob, loss, PoF])

    ######################
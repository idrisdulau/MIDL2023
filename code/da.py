import numpy
import torch
import random
import torchvision

def geometryTransform(source, target, mode):
    assert(isImage(source) or isTensor(source))
    assert(isImage(target) or isTensor(target))
    if isImage(source):
        sourceTensor = toTensor(source)
    else: 
        sourceTensor = source
    if isImage(target):
        targetTensor = toTensor(target)
    else: 
        targetTensor = target
    subMode = random.choice(["hFlip","vFlip","dFlip"])
    if subMode == "hFlip":
        sourceTensor = torchvision.transforms.functional.hflip(sourceTensor)
        targetTensor = torchvision.transforms.functional.hflip(targetTensor)
    elif subMode == "vFlip":
        sourceTensor = torchvision.transforms.functional.vflip(sourceTensor)
        targetTensor = torchvision.transforms.functional.vflip(targetTensor)
    else:
        sourceTensor = torchvision.transforms.functional.hflip(sourceTensor)
        sourceTensor = torchvision.transforms.functional.vflip(sourceTensor)
        targetTensor = torchvision.transforms.functional.hflip(targetTensor)
        targetTensor = torchvision.transforms.functional.vflip(targetTensor)
    return sourceTensor, targetTensor     

def isImage(source):
    return isinstance(source, numpy.ndarray) 

def isTensor(source):
    return isinstance(source, torch.Tensor)
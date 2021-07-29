import itertools
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from data_manipulation.models import BaseModel
from data_manipulation.utils import to_torch_var, time_to_string
from torchvision import models as torchModels
from torchvision import transforms

import cv2
import sys
import os

def sliding_windowMosaicMask(image, mask, stepSize, windowSize):

    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]], mask[y:y + windowSize[1], x:x + windowSize[0]])

# Function to take a binary image and output the center of masses of its connected regions
def listFromBinary(im, patchSize):
	if im is None: return []
	else:
		mask = cv2.threshold(255-im, 40, 255, cv2.THRESH_BINARY)[1]

		# compute connected components
		numLabels, labelImage,stats, centroids = cv2.connectedComponentsWithStats(mask)

		newCentroids=[]
		for c in centroids:
			if not borderPoint(im,c,patchSize): newCentroids.append(c)

		return newCentroids[1:]

def borderPoint(img,p,w_size):

	imgH, imgW = img.shape
	margin = w_size // 2

	return p[0]<margin or (imgW-p[0])<margin or p[1]<margin or (imgH-p[1])<margin


def dsc_loss(pred, target, smooth=0.1):
    """
    Loss function based on a single class DSC metric.
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, n_classes, data_shape]
    :param target: Ground truth values. This tensor can have multiple shapes:
     - [batch_size, n_classes, data_shape]: This is the expected output since
       it matches with the predicted tensor.
     - [batch_size, data_shape]: In this case, the tensor is labeled with
       values ranging from 0 to n_classes. We need to convert it to
       categorical.
    :param smooth: Parameter used to smooth the DSC when there are no positive
     samples.
    :return: The mean DSC for the batch
    """
    dims = pred.shape
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    reduce_dims = tuple(range(1, len(dims)))
    num = (2 * torch.sum(pred * target, dim=reduce_dims))
    den = torch.sum(pred + target, dim=reduce_dims) + smooth
    dsc_k = num / den
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)

class myFeatureExtractor(BaseModel):
    def __init__(self,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    n_outputs=3,frozen=True,featEx="res",LR=1e-1,nChanInit=3):
        super().__init__()
        # Init values
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.n_outputs=n_outputs
        self.device = device
        #self.weights=torch.Tensor([1,1,1])
        self.norm = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        print("Creating myFeatureExtractor with device "+str(self.device))

        # add a convolution to move from 4 to 3 channels
        self.n_image_channels = nChanInit
        self.chanT = nn.Conv2d(self.n_image_channels,
                                       3,
                                       kernel_size=3,
                                       padding=1)

        #now add a feature extractor
        if featEx=="res":
            self.featEx = torchModels.resnet50(pretrained=True)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, self.n_outputs)
        elif featEx=="resBIG":
            self.featEx = torchModels.resnet152(pretrained=True)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, self.n_outputs)
        elif featEx=="resnext":
            self.featEx = torchModels.resnext101_32x8d(pretrained=True)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, self.n_outputs)
        elif featEx=="wideresnet":
            self.featEx = torchModels.wide_resnet101_2(pretrained=True)
            num_ftrs = self.featEx.fc.in_features
            self.featEx.fc = nn.Linear(num_ftrs, self.n_outputs)
        elif featEx=="dense":
            self.featEx = torchModels.densenet161(pretrained=True)
            num_ftrs = self.featEx.classifier.in_features
            self.featEx.classifier = nn.Linear(num_ftrs,self.n_outputs)
        elif featEx=="vgg":
            self.featEx = torchModels.vgg19_bn(pretrained=True)
            num_ftrs = self.featEx.classifier[6].in_features
            self.featEx.classifier[6] = nn.Linear(num_ftrs, self.n_outputs)
        elif featEx=="alex":
            self.featEx = torchModels.alexnet(pretrained=True)
            num_ftrs = self.featEx.classifier[6].in_features
            self.featEx.classifier[6] = nn.Linear(num_ftrs, self.n_outputs)
        elif featEx=="squeeze":
            self.featEx = torchModels.squeezenet1_0(pretrained=True)
            self.featEx.classifier[1] = nn.Conv2d(512, self.n_outputs, kernel_size=(1,1), stride=(1,1))
        else: raise Exception("Exception when constructing feature extractor, unknown architecture "+str(featEx))


        # for others see https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

        self.chanT = self.chanT.to(device)
        self.featEx = self.featEx.to(device)

        # <Loss function setup>
        #nn.CrossEntropyLoss()
        self.train_functions = [
            {
                'name': 'xentr01',
                'weight': 1,
                'f': lambda p, t: F.binary_cross_entropy(p, t.type_as(p).to(p.device))
            },

        ]
        self.val_functions = [
            {
                'name': 'xe01',
                'weight': 1,
                #'f': lambda p, t: F.binary_cross_entropy(p[1], t[1].type_as(p[1]).to(p[1].device),weight=self.weights.to(device))
                'f': lambda p, t: F.binary_cross_entropy(p, t.type_as(p).to(p.device))
            },
        ]

        # <Optimizer setup>
        # We do this last setp after all parameters are defined
        #allParams=list(self.parameters())

        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params, lr=LR)

        if frozen:
            for param in self.featEx.parameters():

                param.requires_grad = False

            #now Unfreeze classifier part
            if featEx in ["res","resBIG","resnext","wideresnet"]:
                for param in self.featEx.fc.parameters():
                    param.requires_grad = True
            elif featEx=="dense":
                for param in self.featEx.classifier:
                    param.requires_grad = True
            elif featEx in ["vgg","alex"]:
                for param in self.featEx.classifier[6]:
                    param.requires_grad = True
            elif featEx=="squeeze":
                for param in self.featEx.classifier[1]:
                    param.requires_grad = True
            else: raise Exception("Exception when unfreezing feature extractor classifier, unknown architecture "+str(featEx))

        #initially, do not define scheduler
        self.scheduler =None

    def addScheduler(self,LR,steps,ep):
        #define also LR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer_alg, max_lr=LR, steps_per_epoch=steps, epochs=ep)

    def forward(self, input_s):
        out = []
        for x_ in input_s:
            out.append(self.norm(x_.cpu()))
        out = torch.stack(out)

        output=self.featEx(out.to(self.device))

        #return torch.sigmoid(output)
        return torch.softmax(output,1)

    #def dropout_update(self):
    #    super().dropout_update()
    #    self.autoencoder.dropout = self.dropout

    def batch_update(self, epochs):
        if self.scheduler is not None:self.scheduler.step()

        #return None

    def classifyPatches(self,mosaic,stepSize, patchSize, classToRefine):
        # create empty output mask
        # create a mask of ones to be added each time a patch is found to contain the label
        outputMask = np.zeros((mosaic.shape[0], mosaic.shape[1]), dtype="uint8")
        detectionThreshold=0.95
        print("in classify patches, looking for class "+str(classToRefine))

        countPos=0
        countNeg=0

        for (x, y, mosaicW, maskW) in sliding_windowMosaicMask(mosaic, outputMask, stepSize, windowSize=(patchSize, patchSize)):

            # create a mask of ones to be added each time a patch is found to contain the label
            maskOfOnes = 1*np.ones((maskW.shape[0], maskW.shape[1]), dtype="uint8")
            # Can we do the following shit in a less ugly way???? (no, learner.predict(mosaicW) does not work)

            with torch.no_grad():
                currentImage=np.expand_dims(np.moveaxis(mosaicW,-1,0), axis=0)
                #print(currentImage.shape)
                seg_pi = self(torch.from_numpy(currentImage))
                probs=seg_pi[0]
            if probs[classToRefine]>detectionThreshold:
                countPos+=1
                #print("found one! "+str(probs)+" "+str(probs[classToRefine]))
                # maskW = np.add(maskW, maskOfOnes)
                maskW[:] += 1
                # print(str(np.count_nonzero(maskW)) +" "+str(np.count_nonzero(outputMask)))
            else:
                countNeg+=1


        # At the end of the loop, we have added ones every time a patch has been classified as belonging to the class
        oMaskMax = np.max(outputMask)
        print(countPos)
        print(countNeg)
        print(oMaskMax)
        heatMapPerc=0.75
        cv2.imwrite("HMres.jpg", outputMask*int(255/oMaskMax))
        outputMask[outputMask<int(oMaskMax*heatMapPerc)]=0
        outputMask[outputMask!=0]=255
        cv2.imwrite("HMresBLACK.jpg", outputMask)
        sys.exit(0)
        # print("output mask max!! "+str(oMaskMax))
        return outputMask



    def test(self, predictFolder, stepSize=50, patchSize=50, verbose=True, refine=False,classToRefine=0):
        print("entering test function")

        print("change hardcoded parameter!!!!!!!!!!!!!")
        label_to_find = "cargol_poma"

        self.eval()

        resultsFolder = os.path.join(predictFolder, "results")
        if not os.path.exists(resultsFolder):
            os.makedirs(resultsFolder)

        print("IN testing, results folder "+resultsFolder)

        for file in os.listdir(predictFolder):

            imageName, ext = os.path.splitext(file)

            if ext == ".jpg" or ext == ".JPG" or ext == ".jpeg":

                print("Analysing image "+file+"... ", end="", flush=True)
                image = cv2.imread(os.path.join(predictFolder,file), cv2.IMREAD_COLOR)
                if image is None: raise Exception("could not read image "+str(os.path.join(predictFolder,file)) )

                outputmask = self.classifyPatches(image, stepSize, patchSize,classToRefine)

                # outputmask = cv2.imread(predictFolder+"outputmask.jpg", cv2.IMREAD_GRAYSCALE)
                outputmask = np.invert(outputmask)
                centroids = listFromBinary(outputmask, patchSize)

                print(str(len(centroids)) + " " + label_to_find + " found ... ", end="", flush=True)
                for c in centroids:
                    image = cv2.circle(image, (int(c[0]), int(c[1])), 10, (0,0,255), 5)
                    # topleft = ( int(c[0])-int(classifierPatchSize/2), int(c[1])-int(classifierPatchSize/2) )
                    # bottomright = ( int(c[0])+int(classifierPatchSize/2), int(c[1])+int(classifierPatchSize/2) )
                    # image = cv2.rectangle(image, topleft, bottomright, (0,0,255), 3)

                scale_percent = 100  # percent of original size
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

                print("Saving result image & data ... ", end="", flush=True)
                cv2.imwrite(os.path.join(resultsFolder, imageName+"_res.jpg"), image)
                # csv_writer.writerow([imageName, gps['latitude'], gps['longitude'], len(centroids)])

                print("done")

            else:
                print("WARNING: Invalid image format. "+file+" should be a jpg image", "yellow")


#refine a mask created with patch predictions
# return a binary mask with the same size as im with 255 where the class is present
def refine(self, im, patch_size,c):
    #print("refining!"+str(c))
    #returnMask=np.zeros(im.shape[1:])
    #data_tensor = to_torch_var(np.expand_dims(im, axis=0))

    # make smaller images
    newPatchSize=patch_size//4

    # Initial results. Filled to 0.
    seg_i=[]
    for x in range(self.n_outputs):
        #seg_i.append([np.zeros(im.shape[1:])])
        seg_i.append(np.zeros(im.shape[1:]))

    #print("after initial filling, seg_i length "+str(len(seg_i)))

    limits = tuple(
        list(range(0, lim, newPatchSize))[:-1] + [lim - newPatchSize]
        for lim in im.shape[1:] #was for lim in data.shape[1:]
    )
    #print(limits)

    limits_product = list(itertools.product(*limits))
    #print(limits_product)

    n_patches = len(limits_product)

    # The following code is just a normal test loop with all the
    # previously computed patches.
    for pi, (xi, xj) in enumerate(limits_product):
        # Here we just take the current patch defined by its slice
        # in the x and y axes. Then we convert it into a torch
        # tensor for testing.
        xslice = slice(xi, xi + newPatchSize)
        yslice = slice(xj, xj + newPatchSize)

        currentImage=im[slice(None), xslice, yslice]


        if not uselessImage(currentImage):

            data_tensor = to_torch_var(np.expand_dims(currentImage, axis=0))

            # Testing itself.
            with torch.no_grad():
                seg_pi = self(data_tensor)
                seg_piFirst=seg_pi[0][0]
                seg_piSecond=seg_pi[1][0]

            probThreshold=0.4

            # we store the probability of this class in this patch
            #seg_i[x][xslice, yslice]=int(255*seg_piFirst.cpu().numpy()[x]*seg_piSecond.cpu().numpy()[x])
            if seg_piFirst.cpu().numpy()[c]>probThreshold:
                #print(seg_i[c][xslice, yslice].shape)
                seg_i[c][xslice, yslice]=255

    #print(seg_i[c].shape)

    return seg_i[c]

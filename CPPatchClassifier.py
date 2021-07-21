import argparse
import sys
import os
import re
import cv2
import time
import random
import numpy as np
from torch.utils.data import DataLoader
from data_manipulation.utils import color_codes, find_file
from datasets import CPDataset
from models import  myFeatureExtractor

import torch
import torchvision
from torchvision import models as torchModels
import torch.nn as nn

import dice
#import imagePatcherAnnotator as pa

def setRandomSeed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
#Remember to use num_workers=0 when creating the DataBunch.

def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-dtrain', '--trainingData-directory',
        dest='train_dir', default=None,
        help='Directory containing the training images'
    )
    parser.add_argument(
        '-dtest', '--testingData-directory',
        dest='test_dir', default=None,
        help='Directory containing the testing images'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int,  default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '-arch', '--architecture',
        dest='arch',
        default="res",
        help='type of architecture'
    )
    parser.add_argument(
        '-frozen', '--frozen',
        dest='frozen',
        default="True",
        help='whether or no the feature extractor is frozen'
    )
    parser.add_argument(
        '-p', '--patience',
        dest='patience',
        type=int, default=3,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-lr', '--learningRate',
        dest='lr',
        type=float, default=0.001,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '-B', '--batch-size',
        dest='batch_size',
        type=int, default=32,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-t', '--patch-size',
        dest='patch_size',
        type=int, default=128,
        help='Patch size'
    )
    parser.add_argument(
        '-st', '--step-size',
        dest='step_size',
        type=int, default=50,
        help='Step size'
    )

    options = vars(parser.parse_args())

    return options

def train_test_net(init_net_name, listOfClasses, importantClassCode, verbose=1):
    """
    :param net_name:
    :return:
    """
    # Init
    c = color_codes()
    options = parse_inputs()
    randomSeed=42

    save_model=True
    addScheduler=True

    # Data loading (or preparation)
    lrList=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.00001,0.00002,0.00003,0.00004,0.00005,0.00006,0.00007,0.00008,0.00009]

    for learningRate in lrList:
        resultsList=[]

        # READ TRAIN/VALIDATION set from a folder NO LEAVE ONE OUT HERE!
        val_split = 0.1
        allTrainData=CPDataset(options["train_dir"],listOfClasses)
        train,valid=allTrainData.breakTrainValid(val_split)
        #LATER READ TEST SET FROM ANOTHER FOLDER

        batch_size = int(options["batch_size"])

        overlap = (0, 0)
        num_workers = 1

        architecture=options['arch']
        #learningRate=options['lr']
        if options['frozen']=="True":frozenInit=True
        else: frozenInit=False

        net_name=init_net_name+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)

        model_name = '{:}.mosaic{:}.mdl'.format(net_name, str(learningRate))

        print("initializing feature extractor with architecture "+str(architecture)+" learningRate "+str(learningRate)+" and frozen "+str(frozenInit))

        net = myFeatureExtractor(n_outputs=allTrainData.numClasses(),frozen=frozenInit,featEx=architecture,LR=learningRate,nChanInit=3)

        print("FEATURE EXTRACTOR CREATED")

        training_start = time.time()
        d_path=options["train_dir"]
        try:
            net.load_model(os.path.join(d_path, model_name))
        except IOError:

            # Dataloader creation
            if verbose > 0:
                n_params = sum(
                    p.numel() for p in net.parameters() if p.requires_grad
                )
                print(
                    '%sStarting training %s (%d parameters)' %
                    (c['c'], c['nc'], n_params)
                )

            print('Training dataset (with validation) '+str(val_split)+" "+str(train.numClasses())+" "+str(valid.numClasses())+" train length "+str(len(train))+" valid length "+str(len(valid)))
            setRandomSeed(42)

            train_dataloader = DataLoader(train, batch_size, True, num_workers=num_workers)
            setRandomSeed(42)
            val_dataloader = DataLoader(valid, batch_size, num_workers=num_workers)

            setRandomSeed(42)
            epochs = parse_inputs()['epochs']
            patience = parse_inputs()['patience']

            if addScheduler:
                net.addScheduler(learningRate,len(train_dataloader),epochs)

            print("Starting")
            net.fit(
                train_dataloader,
                val_dataloader,
                epochs=epochs,
                patience=patience,
                verbose=True
            )

            if save_model: net.save_model(os.path.join(d_path, model_name))

        if verbose > 0:
            time_str = time.strftime(
                '%H hours %M minutes %S seconds',
                time.gmtime(time.time() - training_start)
            )
            print(
                '%sTraining finished%s (total time %s)\n' %
                (c['r'], c['nc'], time_str)
            )

            print("STARTING TESTING! ")
            #add a test parameter, it will be a list of image names

        yi = net.test(options["test_dir"],patchSize=options["patch_size"],stepSize=options["step_size"],verbose=True,refine=True,classToRefine=importantClassCode)

        print("ended testing")
        sys.exit()
        numClasses=len(listOfClasses)
        for counter in range(numClasses):
            print("starting prediction of "+str(layerNames[counter]))
            #print(str(mosaicDict[mosaic_names[i]]))

            predIm=yi[counter][0]
            currentGtImage=gtDict[mosaic_names[i]][counter]
            #currentGtImage=cv2.imread(mosaicDict[mosaic_names[i]].layerFileList[counter],0)

            if counter>0:
                #cv2.imwrite(
                #    os.path.join(d_path, "NONBpatch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+"layer"+str(counter)+'pred_{:}.jpg'.format(case)),
                #    predIm
                #)
                cv2.imwrite(
                    "./NONBpatch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+"layer"+str(counter)+'pred_{:}.jpg'.format(case),
                    predIm
                )

            binarizeMask(predIm)
            total,TP,FP,TN,FN=countPatchStatsCategory(currentGtImage,predIm,patch_size[0])
            currentResults=(total,TP,FP)

            print(str(layerNames[counter])+" TPR "+str(100*TP/(TP+FN)))
            print(str(layerNames[counter])+" FPR "+str(100*FP/(TN+FP)))
            print(str(layerNames[counter])+" ACCURACY "+str(100*(TP+TN)/(total)))
            #print(str(layerNames[counter])+" TNR "+str(100*TN/(TN+FP))  )
            #print(str(layerNames[counter])+" FNR "+str(100*FN/(TP+FN)))
            print(str(layerNames[counter])+" Stats: "+str(total)+" "+str(TP)+" "+str(FP)+" "+str(TN)+" "+str(FN))

            #print("Writing "+str("patch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+str(counter)+'pred_{:}.jpg'.format(case)))
            if counter>0:
                cv2.imwrite(
                    os.path.join(d_path, "patch"+str(sideOfPatch)+"LR"+str(learningRate)+"net"+architecture+"frozen"+str(frozenInit)+"layer"+str(counter)+'pred_{:}.jpg'.format(case)),
                    predIm
                )

            #sys.exit()

            print("mosaic"+str(case)+" DICE COEFFICIENT :"+str(dice.dice(255-currentGtImage,255-predIm)),flush=True)
            #print("mosaic"+str(case)+" DICE COEFFICIENT2 :"+str(dice.dice(currentGtImage,predIm)))
            #print("mosaic"+str(case)+" DICE COEFFICIENT3 :"+str(dice.dice(currentGtImage,255-predIm)))
        resultsList.append(currentResults)

        sumTotal=0
        sumTP=0
        sumFP=0
        for el in resultsList:
            print("starting first tuple "+str(el))
            sumTotal+=el[0]
            sumTP+=el[1]
            sumFP+=el[2]
        print("FINISHED LEAVE ONE OUT FOR LR"+str(learningRate)+" averageTotal "+str(sumTotal/len(resultsList))+" average FP "+str(sumTP/len(resultsList))+" average FP "+str(sumFP/len(resultsList)))


def main():
    # Init
    c = color_codes()
    listOfClasses=["cargolpoma","ground","highlight","plant"]
    importantClass="cargolpoma"
    classDict={}
    for i in range(len(listOfClasses)):classDict[listOfClasses[i]]=i
    importantClassCode=int(classDict[importantClass])
    print(listOfClasses)

    print(torchvision.__version__)

    print(
        '%s[%s] %s<Tree detection pipeline>%s' % (
            c['c'], time.strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    ''' <Detection task> '''
    net_name = 'floor-detection'

    train_test_net(net_name,listOfClasses,importantClassCode)


if __name__ == '__main__':
    main()

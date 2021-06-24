import numpy as np
import os
import sys
import re
import random
import cv2

from torch.utils.data.dataset import Dataset
#from data_manipulation.datasets import get_slices_bb

class CPDataset(Dataset):
    # Given a folder containing files stored following a certain regular expression,
    # Load all image files from the folder, put them into a list
    # At the same time, load all the labels from the folder names, put them into a list too!

    def __init__(self,dataFolder=None,listOfClasses=None):
        regex = re.compile('([^/]+)PATCH\d+.jpg$')
        classesDict={}
        self.listOfClasses=listOfClasses
        if listOfClasses is not None:
            for i in range(len(self.listOfClasses)):classesDict[self.listOfClasses[i]]=i
        #r'([^/]+)PATCH\d+.jpg$'

        self.folder=dataFolder
        self.imageList=[]
        self.labelList=[]

        if dataFolder is not None:
            for root, dirs, files in os.walk(dataFolder):
              for file in files:
                if regex.match(file):
                   #print(file)
                   currentClass= re.split(r'SP([^/]+)PATCH\d+.jpg$',file)[1]
                   currentImage=cv2.imread(os.path.join(self.folder,file))
                   if currentImage is None: raise Exception("CPDataset Constructor, problems reading file "+file)

                   self.imageList.append(np.moveaxis(currentImage,-1,0))#  a pytorch li agrada tenir el numero de canals davant
                   self.labelList.append(classesDict[currentClass])

        self.len=len(self.imageList)
        print("Read a CPDataset with "+str(self.len)+" images of the following classes "+str(self.labelList))


    def breakTrainValid(dataS,proportion):#Create a dataset from an existing one
        train=CPDataset()
        valid=CPDataset()


        for i in range(int(len(dataS)*proportion)):
            valid.imageList.append(dataS.imageList[i].copy())
            valid.labelList.append(dataS.labelList[i])

        for i in range(int(len(dataS)*proportion),len(dataS)):
            train.imageList.append(dataS.imageList[i].copy())
            train.labelList.append(dataS.labelList[i])

        print("breaking "+str(len(train))+" and "+str(len(valid)))


        return train,valid

    def numClasses(self):return len(np.unique(self.labelList))

    def __getitem__(self, index):

            inputs = self.imageList[index].astype(np.float32)

            # the target is a list of probabilities of belonging to each class
            target = np.zeros(len(self.listOfClasses))
            target[self.labelList[index]]=1

            #print("returning target "+str(target))

            return inputs, target

    def __len__(self):
        return len(self.imageList)

if __name__ == '__main__':
    da=CPDataset(sys.argv[1])

    tr,vd=da.breakTrainValid(0.4)

    print(len(tr))
    print(len(vd))

    print(vd.numClasses())


#    for x,y in tr:
        #print(x)
#        print(y)

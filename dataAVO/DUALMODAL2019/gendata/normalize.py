#: I.D.
#: normalize the groundtruth for vessels, veins and arteries subsets

import os
import sys
import cv2
import numpy
import subprocess

def main(argv):
    width, height = argv[1:3]

    curPath = os.getcwd()

    newtargetVesselsPathTest = curPath+"/../targetVessels/"+width+"x"+height+"/test/"
    newtargetVesselsPathTrain = curPath+"/../targetVessels/"+width+"x"+height+"/train/"  

    newExt = ".tif"

    for path in [newtargetVesselsPathTest, newtargetVesselsPathTrain]:
        for item in os.listdir(path):
            name, oldExt = item.split(".")
            if oldExt == "jpg":
                output = cv2.imread(path+item, cv2.IMREAD_UNCHANGED)  
                blue, green, red = cv2.split(output)
                output = red
                output = numpy.where(output==0, numpy.zeros(output.shape), numpy.ones(output.shape)*255)
                output = output.astype(numpy.uint8)   
                cv2.imwrite(path+name+newExt, output)
                subprocess.run(["rm", path+item])
            

    newtargetArteriesPathTest = curPath+"/../targetArteries/"+width+"x"+height+"/test/"
    newtargetArteriesPathTrain = curPath+"/../targetArteries/"+width+"x"+height+"/train/"

    for path in [newtargetArteriesPathTest, newtargetArteriesPathTrain]:
        for item in os.listdir(path):
            name, oldExt = item.split(".")
            if oldExt == "jpg":
                output = cv2.imread(path+item, cv2.IMREAD_UNCHANGED)  
                blue, green, red = cv2.split(output)
                output = green
                output = numpy.where(output==0, numpy.zeros(output.shape), numpy.ones(output.shape)*255)
                output = output.astype(numpy.uint8)  
                cv2.imwrite(path+name+newExt, output)
                subprocess.run(["rm", path+item])

    newtargetVeinsPathTest = curPath+"/../targetVeins/"+width+"x"+height+"/test/"
    newtargetVeinsPathTrain = curPath+"/../targetVeins/"+width+"x"+height+"/train/"

    for path in [newtargetVeinsPathTest, newtargetVeinsPathTrain]:
        for item in os.listdir(path):
            name, oldExt = item.split(".")
            if oldExt == "jpg":
                output = cv2.imread(path+item, cv2.IMREAD_UNCHANGED)  
                blue, green, red = cv2.split(output)
                output = blue
                output = numpy.where(output==0, numpy.zeros(output.shape), numpy.ones(output.shape)*255)
                output = output.astype(numpy.uint8)  
                cv2.imwrite(path+name+newExt, output)
                subprocess.run(["rm", path+item])

if __name__ == '__main__':
    main(sys.argv)

#: I.D.
#: Rename data from: original/ to successive positive integers

import os
import sys
import subprocess

def main(argv):
    width, height = argv[1:3]

    curPath = os.getcwd()
    originalSourcePath = curPath+"/../original/source/"
    newSourcePathTrain = curPath+"/../source/"+width+"x"+height+"/train/"
    newSourcePathTest = curPath+"/../source/"+width+"x"+height+"/test/"
   
    correspList = []
    for originalPath, trainPath, testPath in [(originalSourcePath,newSourcePathTrain,newSourcePathTest)]:
        nameIdx = 0
        for image in os.listdir(originalPath):
            name, ext = image.split(".")
            ext = "."+ext.lower()
            nameIdx+=1
            newName = str(nameIdx)+ext
            correspList += [(name,newName)]
            if nameIdx <= 25:
                subprocess.run(["cp", originalPath+image, testPath+newName])
            else:
                subprocess.run(["cp", originalPath+image, trainPath+newName])
            print("Old:",name,"=> New:",nameIdx)

    print(correspList)
    originalTargetAVPath = curPath+"/../original/targetAV/"
    newTargetVesselsPathTrain = curPath+"/../targetVessels/"+width+"x"+height+"/train/"
    newTargetVesselsPathTest = curPath+"/../targetVessels/"+width+"x"+height+"/test/"
    newTargetVeinsPathTrain = curPath+"/../targetVeins/"+width+"x"+height+"/train/"
    newTargetVeinsPathTest = curPath+"/../targetVeins/"+width+"x"+height+"/test/"
    newTargetArteriesPathTrain = curPath+"/../targetArteries/"+width+"x"+height+"/train/"
    newTargetArteriesPathTest = curPath+"/../targetArteries/"+width+"x"+height+"/test/"

    for originalPath, trainPath, testPath in [
    (originalTargetAVPath,newTargetVesselsPathTrain,newTargetVesselsPathTest), \
    (originalTargetAVPath,newTargetVeinsPathTrain,newTargetVeinsPathTest), \
    (originalTargetAVPath,newTargetArteriesPathTrain,newTargetArteriesPathTest)]:
        for image in os.listdir(originalPath):
            name, ext = image.split(".")
            n1, n2, _ = name.split("_") 
            correspName = n1+"_"+n2
            ext = "."+ext.lower()
            newName = [elem[1] for elem in correspList if elem[0] == correspName][0]
            if int(newName.split(".")[0]) <= 25:
                subprocess.run(["cp", originalPath+image, testPath+newName])
            else:
                subprocess.run(["cp", originalPath+image, trainPath+newName])
            print("Old:",name,"=> New:",newName)

if __name__ == '__main__':
    main(sys.argv)

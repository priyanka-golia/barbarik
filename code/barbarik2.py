from __future__ import print_function
import sys
import os
import math
import random
import argparse
import copy
import tempfile
cwd = os.getcwd()+"/WAPS"
sys.path.append(cwd)
from waps import sampler as samp
import weightcount.WeightCount as chainform
SAMPLER_UNIGEN = 1
SAMPLER_QUICKSAMPLER = 2
SAMPLER_STS = 3
SAMPLER_CUSTOM = 4
def parseWeights(inputFile, indVarList):
    f = open(inputFile, "r")
    lines = f.readlines()
    f.close()
    weight_map = {}
    for line in lines:
        if line.startswith("w"):
            variable, weight = line[2:].strip().split()
            variable = int(variable)
            weight = float(weight)
            if (0.0 < weight < 1.0):
                if (variable in indVarList):
                    weight_map[variable] = weight
            else:
                print("Weights should only be in (0,1) ")
                exit(-1)
    return weight_map
def tilt(weight_map, sample1, sample2, UserIndVarList):
    tilt = 1.0
    sample1_w = copy.deepcopy(sample1)
    sample2_w = copy.deepcopy(sample2)
    sample1_w.sort(key=abs)
    sample1_w.sort(key=abs)
    for i in range(len(sample1_w)):
        litWeight = weight_map.get(abs(sample1_w[i]),0.5)
        if (sample1_w[i] > sample2_w[i]):
            tilt *= litWeight/(1-litWeight)
        elif(sample1_w[i] < sample2_w[i]):
            tilt *= (1-litWeight)/litWeight
    return tilt
def parseIndSupport(indSupportFile):  # returns List of Independent Variables
    f = open(indSupportFile, "r")
    lines = f.readlines()
    f.close()
    indList = []
    numVars = 0
    for line in lines:
        if line.startswith("p cnf"):
            fields = line.split()
            numVars = int(fields[2])
        if line.startswith("c ind"):
            indList.extend(
                line.strip()
                .replace("c ind", "")
                .replace(" 0", "")
                .strip()
                .replace("v ", "")
                .split()
            )
    if len(indList) == 0:
        indList = [int(x) for x in range(1, numVars + 1)]
    else:
        indList = [int(x) for x in indList]
    return indList
def getSolutionFromUniGen(inputFile, numSolutions, indVarList):
    #inputFilePrefix = inputFile.split("/")[-1][:-4]
    tempOutputFile = inputFile + ".txt"
    f = open(tempOutputFile, "w")
    f.close()
    cmd = (
        "./samplers/scalmc "
        + inputFile
        + " --samples "
        + str(numSolutions)
        + " --sampleout "
        + str(tempOutputFile)
        #+ " > /dev/null 2>&1"
    )
    print(cmd)
    os.system(cmd)
    f = open(tempOutputFile, "r")
    lines = f.readlines()
    f.close()
    solList = []
    for line in lines:
        line = line.strip()
        freq = int(line.split(":")[0].strip())
        for _ in range(freq):
            sample = line.split(":")[1].strip()[:-2]
            sample = sample.split()
            sample = [int(i) for i in sample]
            solList.append(sample)
            if len(solList) == numSolutions:
                break
        if len(solList) == numSolutions:
            break
    cmd = "rm " + str(tempOutputFile)
    os.system(cmd)
    return solList
# @CHANGE_HERE : please make changes in the below block of code
""" this is the method where you could run your sampler for testing
Arguments : input file, number of solutions to be returned, list of independent variables
output : list of solutions """
def getSolutionFromCustomSampler(inputFile, numSolutions, indVarList):
    solreturnList = []
    """ write your code here """
    return solreturnList
""" END OF BLOCK """
def getSolutionFromSampler(seed, inputFile, numSolutions, samplerType, indVarList):
    if samplerType == SAMPLER_UNIGEN:
        return getSolutionFromUniGen(inputFile, numSolutions, indVarList)
    if samplerType == SAMPLER_QUICKSAMPLER:
        return getSolutionFromQuickSampler(inputFile, numSolutions, indVarList)
    if samplerType == SAMPLER_STS:
        return getSolutionFromSTS(seed, inputFile, numSolutions, indVarList)
    if samplerType == SAMPLER_CUSTOM:
        return getSolutionFromCustomSampler(inputFile, numSolutions, indVarList)
    else:
        print("Error")
        return None
def getSolutionFromSTS(seed, inputFile, numSolutions, indVarList):
    kValue = 50
    samplingRounds = numSolutions / kValue + 1
    inputFilePrefix = inputFile[:-4]
    outputFile = inputFilePrefix + ".output"
    cmd = (
        "./samplers/STS -k="
        + str(kValue)
        + " -nsamples="
        + str(samplingRounds)
        + " -rnd-seed="
        + str(seed)
        + " "
        + str(inputFile)
        + " > "
        + str(outputFile)
    )
    os.system(cmd)
    f = open(outputFile, "r")
    lines = f.readlines()
    f.close()
    solList = []
    shouldStart = False
    baseList = {}
    for j in range(len(lines)):
        if lines[j].strip() == "Outputting samples:" or lines[j].strip() == "start":
            shouldStart = True
            continue
        if lines[j].strip().startswith("Log") or lines[j].strip() == "end":
            shouldStart = False
        if shouldStart:
            if lines[j].strip() not in baseList:
                baseList[lines[j].strip()] = 1
            else:
                baseList[lines[j].strip()] += 1
            sol = []
            sample = list(lines[j].strip())
            for x in range(len(indVarList)):
                if sample[x] == "0":
                    sol.append(-1*indVarList[x])
                else:
                    sol.append(indVarList[x])
            sol.sort(key=abs)
            solList.append(sol)
            if len(solList) == numSolutions:
                break
    if len(solList) < numSolutions:
        print("STS Did not find required number of solutions")
        exit(1)
    cmd = "rm " + outputFile
    os.system(cmd)
    return solList
def getSolutionFromQuickSampler(inputFile, numSolutions, indVarList):
    cmd = (
        "./samplers/quicksampler -n "
        + str(numSolutions * 5)
        + " "
        + str(inputFile)
        + " > /dev/null 2>&1"
    )
    print(cmd)
    os.system(cmd)
    cmd = "./samplers/z3 " + str(inputFile) + " > /dev/null 2>&1"
    os.system(cmd)

    i = 0
    if numSolutions > 1:
        i = 0

    f = open(inputFile + ".samples", "r")
    lines = f.readlines()
    f.close()

    f = open(inputFile + ".samples.valid", "r")
    validLines = f.readlines()
    f.close()

    solList = []
    for j in range(len(lines)):
        if validLines[j].strip() == "0":
            continue
        fields = lines[j].strip().split(":")
        sol = []
        i = 0
        for x in list(fields[1].strip()):
            if x == "0":
                sol.append(-1*indVarList[i])
            else:
                sol.append(indVarList[i])
            i += 1
            if (i == len(indVarList)):
                break
        sol.sort(key=abs)
        solList.append(sol)
        if len(solList) == numSolutions:
            break

    cmd = "rm " + inputFile + ".samples"
    os.system(cmd)

    cmd = "rm " + inputFile + ".samples.valid"
    os.system(cmd)

    if len(solList) != numSolutions:
        print("Did not find required number of solutions")
        exit(1)

    return solList

def getSolutionFromWAPS(inputFile, numSolutions):
    sampler = samp(cnfFile=inputFile)
    sampler.compile()
    sampler.parse()
    sampler.annotate()
    samples = sampler.sample(totalSamples=numSolutions)
    solList = list(samples)
    solList = [i.strip().split() for i in solList]
    solList = [[int(x) for x in i] for i in solList]
    return solList

def getSolutionFromIdeal(inputFile, numSolutions):
    return getSolutionFromWAPS(inputFile, numSolutions)

def findWeightsForVariables(sampleSol, idealSol, numSolutions):
    """
    Finds rExtList
    """
    countList = []
    newVarList = []
    lenSol = len(sampleSol)
    for _ in range(min(int(math.log(numSolutions, 2)) + 4, lenSol - 3, 1)):
        countNum = 254  # random.randint(2, 64)
        countList.append(countNum)
        newVarList.append(8)
    rExtList = []
    oldVarList = []
    indexes = sorted(random.sample(range(len(sampleSol)), len(countList)))
    idealVarList = [idealSol[i] for i in indexes]
    sampleVarList = [sampleSol[i] for i in indexes]
    oldVarList.append(sampleVarList)
    oldVarList.append(idealVarList)
    rExtList.append(countList)
    rExtList.append(newVarList)
    rExtList.append(oldVarList)
    return rExtList

def pushVar(variable, cnfClauses):
    cnfLen = len(cnfClauses)
    for i in range(cnfLen):
        cnfClauses[i].append(variable)
    return cnfClauses

def getCNF(variable, binStr, sign, origTotalVars):
    cnfClauses = []
    binLen = len(binStr)
    if sign:
        cnfClauses.append([binLen + 1 + origTotalVars])
    else:
        cnfClauses.append([-(binLen + 1 + origTotalVars)])
    for i in range(binLen):
        newVar = int(binLen - i + origTotalVars)
        if sign == False:
            newVar = -1 * (binLen - i + origTotalVars)
        if binStr[binLen - i - 1] == "0":
            cnfClauses.append([newVar])
        else:
            cnfClauses = pushVar(newVar, cnfClauses)
    pushVar(variable, cnfClauses)
    return cnfClauses

def constructChainFormula(originalVar, solCount, newVars, origTotalVars, invert):
    writeLines = ""
    binStr = str(bin(int(solCount)))[2:-1]
    binLen = len(binStr)
    for i in range(newVars - binLen - 1):
        binStr = "0" + binStr
    firstCNFClauses = getCNF(-int(originalVar), binStr, invert, origTotalVars)
    addedClauseNum = 0
    for i in range(len(firstCNFClauses)):
        addedClauseNum += 1
        for j in range(len(firstCNFClauses[i])):
            writeLines += str(firstCNFClauses[i][j]) + " "
        writeLines += "0\n"
    CNFClauses = []
    for i in range(len(CNFClauses)):
        if CNFClauses[i] in firstCNFClauses:
            continue
        addedClauseNum += 1
        for j in range(len(CNFClauses[i])):
            writeLines += str(CNFClauses[i][j]) + " "
        writeLines += "0\n"
    return (writeLines, addedClauseNum)

# @returns whether new file was created and the list of independent variables
def constructNewFile(inputFile, tempFile, sampleSol, unifSol, rExtList, origIndVarList):
    sampleMap = {}
    unifMap = {}
    diffIndex = -1   #ensures that sampleSol != unifSol when projected on indVarList
    for i in sampleSol:
        if not (abs(int(i)) in origIndVarList):
            continue
        if int(i) != 0:
            sampleMap[abs(int(i))] = int(int(i) / abs(int(i)))
    for j in unifSol:
        if int(j) != 0:
            if not (abs(int(j)) in origIndVarList):
                continue
            if sampleMap[abs(int(j))] != int(j) / abs(int(j)):
                diffIndex = abs(int(j))
            unifMap[abs(int(j))] = int(int(j) / abs(int(j)))
    if diffIndex == -1:
        print("both samples are the same, error condition")
        print(sampleSol, unifSol)
        exit(-1)
    solClause = ""
    f = open(inputFile, "r")
    lines = f.readlines()
    f.close()
    countList = rExtList[0]
    newVarList = rExtList[1]
    sumNewVar = int(sum(newVarList))
    oldClauseStr = ""
    for line in lines:
        if line.strip().startswith("p cnf"):
            numVar = int(line.strip().split()[2])
            numClause = int(line.strip().split()[3])
        else:
            if line.strip().startswith("w"):
                oldClauseStr += line.strip()+"\n"
            elif not (line.strip().startswith("c")):
                oldClauseStr += line.strip()+"\n"
    #Adding constraints to ensure only two clauses
    for i in origIndVarList:
        if int(i) != diffIndex:
            numClause += 2
            solClause += (
                str(-(diffIndex ) * sampleMap[diffIndex])
                + " "
                + str(sampleMap[int(i)] * int(i))
                + " 0\n"
            )
            solClause += (
                str(-(diffIndex ) * unifMap[diffIndex])
                + " "
                + str(unifMap[int(i)] * int(i))
                + " 0\n"
            )
    invert = True
    seenVars = []
    currentNumVar = numVar
    for oldVarList in rExtList[2]:
        for i in range(len(oldVarList)):
            addedClause = ""
            addedClauseNum = 0
            if True or not (int(oldVarList[i]) in seenVars):
                sign = int(oldVarList[i]) / abs(int(oldVarList[i]))
                (addedClause, addedClauseNum) = constructChainFormula(
                    sign * (abs(int(oldVarList[i]))),
                    int(countList[i]),
                    int(newVarList[i]),
                    currentNumVar,
                    invert,
                )
            seenVars.append(int(oldVarList[i]))
            currentNumVar += int(newVarList[i])
            numClause += addedClauseNum
            solClause += addedClause
        invert = True #not (invert)
    tempIndVarList =[]
    indStr = "c ind "
    indIter = 1
    for i in origIndVarList:
        if indIter % 10 == 0:
            indStr += " 0\nc ind "
        indStr += str(i) + " "
        indIter += 1
        tempIndVarList.append(i)
    for i in range(numVar, currentNumVar + 1):
        if indIter % 10 == 0:
            indStr += " 0\nc ind "
        indStr += str(i) + " "
        indIter += 1
        tempIndVarList.append(i)
    indStr += " 0\n"
    headStr = "p cnf " + str(currentNumVar) + " " + str(numClause) + "\n"
    writeStr = headStr + indStr
    writeStr += solClause
    writeStr += oldClauseStr
    f = open(tempFile, "w")
    f.write(writeStr)
    f.close()
    return tempIndVarList

def constructKernel(inputFile, tempFile, samplerSample, idealSample, numSolutions, origIndVarList):
    rExtList = findWeightsForVariables(samplerSample, idealSample, numSolutions)
    tempIndVarList = constructNewFile(inputFile, tempFile, samplerSample, idealSample, rExtList, origIndVarList)
    return tempIndVarList

# Returns 1 if Ideal and 0 otherwise
def biasFind(sample, solList, indVarList):
    solMap = {}
    numSolutions = len(solList)
    for sol in solList:
        solution = ""
        solFields = sol
        solFields.sort(key=abs)
        for entry in solFields:
            if (abs(entry)) in indVarList:
                solution += str(entry) + " "
        if solution in solMap.keys():
            solMap[solution] += 1
        else:
            solMap[solution] = 1

    if not (bool(solMap)):
        print(solList,indVarList)
        print("No Solutions were given to the test")
        exit(1)
    print("c Printing solMap")
    print(solMap)
    solution = ""
    for i in solList[0]:
        if abs(i) in indVarList:
            solution += str(i) + " "
    if(len(set(solList[0]).intersection(set(sample))) == len(sample)):
        return solMap.get(solution, 0)*1.0/numSolutions
    else:
        return 1.0-(solMap.get(solution, 0)*1.0/numSolutions)

def barbarik2():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eta", type=float, help="default = 1.6", default=1.6, dest="eta"
    )
    parser.add_argument(
        "--epsilon", type=float, help="default = 0.1", default=0.1, dest="epsilon"
    )
    parser.add_argument(
        "--delta", type=float, help="default = 0.2", default=0.2, dest="delta"
    )
    parser.add_argument(
        "--sampler",
        type=int,
        help=str(SAMPLER_UNIGEN)
        + " for UniGen;\n"
        + str(SAMPLER_QUICKSAMPLER)
        + " for QuickSampler;\n"
        + str(SAMPLER_STS)
        + " for STS;\n",
        default=SAMPLER_STS,
        dest="samplertype",
    )
    parser.add_argument(
        "--nnf", type=int, default=0, help="set to 1 to keep the compiled nnf and other logs", dest="nnf")
    parser.add_argument("--seed", type=int, dest="seed", default=420)
    parser.add_argument("input", help="input file")
    #parser.add_argument("output", help="output file")

    args = parser.parse_args()
    UserInputFile = args.input

    print("This is the user input:--", UserInputFile)

    inputFilePrefix = UserInputFile.split("/")[-1][:-4]
    inputFile = inputFilePrefix + ".u.cnf"

    print("This is the output file after weighted to unweighted:", inputFile)

    UserIndVarList = parseIndSupport(UserInputFile)
    indVarList = list(chainform.Transform(UserInputFile, inputFile, 4))  # precision set to 4

    eta = args.eta
    epsilon = args.epsilon
    delta = args.delta
    samplerType = args.samplertype
    #outputFile = args.output
    nnf = args.nnf
    seed = args.seed

    random.seed(seed)

    #f = open(outputFile, "w")
    #f.close()

    samplerString = ""

    if samplerType == SAMPLER_UNIGEN:
        samplerString = "UniGen"
    if samplerType == SAMPLER_QUICKSAMPLER:
        samplerString = "QuickSampler"
    if samplerType == SAMPLER_STS:
        samplerString = "STS"
    if samplerType == SAMPLER_CUSTOM:
        samplerString = "CustomSampler"

    weight_map = parseWeights(UserInputFile, UserIndVarList)

    breakExperiment = False
    totalSolutionsGenerated = 0
    totalIdealSamples = 0
    t = int(math.ceil(math.log(2/delta)/math.log(10.0/(10 - (eta - 6*epsilon)*eta))))
    pre_numSolutions = int(2 * math.log(t/delta))
    lo = (1+epsilon)/(1-epsilon)
    hi = 1 + (eta+6*epsilon)/4

    print("# of times the test will run", t)

    tempFile = inputFile[:-6] + "_t.cnf"

    unifSol = getSolutionFromIdeal(UserInputFile, t)
    sampleSol = getSolutionFromSampler(seed, inputFile, t, samplerType, indVarList)

    print("This is the #of sample pairs we need to check "+str(t))

    for i in range(t):
        seed += 1

        idealSample = unifSol[i]
        samplerSample = sampleSol[i]
        projectedsamplerSample = []

        for s in samplerSample:
            if abs(s) in UserIndVarList:
                projectedsamplerSample.append(s)

        print("indVarList", indVarList)
        print("userIndVarList", UserIndVarList)

        # loop out?
        if set(projectedsamplerSample) == set(idealSample):
            print("no two different values found")
            print("Accepted")
            print( "totalsol :", totalSolutionsGenerated)
            if (i >= t-1):
                print("All rounds passed")
            continue

        # with the assumption that the variable ordering is same
        curTilt = tilt(weight_map, projectedsamplerSample, idealSample, UserIndVarList)

        print("curTilt :", curTilt)

        L = (curTilt*lo)/(1+curTilt*lo)
        H = (curTilt*hi)/(1+curTilt*hi)
        T = (H+L)/2
        numSolutions = int(pre_numSolutions*H/(T-L)**2)

        print("NumSol", numSolutions)

        totalIdealSamples += 1
        totalSolutionsGenerated += 1

        tempIndVarList = constructKernel(UserInputFile, tempFile, projectedsamplerSample,
                                                        idealSample, numSolutions, UserIndVarList)
        samplinglist = list(chainform.Transform(tempFile, tempFile, 4))  # precision set to 4

        print("file was constructed with these", projectedsamplerSample, idealSample)

        if(numSolutions > 10**8):
            print("Looking for more than 10**8 solutions", numSolutions)
            print("too many to ask ,so quitting here")
            print("Rejected at iteration", i)
            print("totalsol :", totalSolutionsGenerated)
            exit(1)

        print("samplingList:", samplinglist)

        solList = getSolutionFromSampler(seed, tempFile, numSolutions, samplerType, samplinglist)

        print("file was constructed with these", projectedsamplerSample, idealSample)

        seed += 1
        bias = biasFind(projectedsamplerSample, solList, UserIndVarList)
        cmd = "rm " + tempFile
        os.system(cmd)

        totalSolutionsGenerated += numSolutions

        print("loThresh:", T)
        print("bias", bias)

        if bias > T:
            #f = open(outputFile, "a")
            #f.write(
            #    """Sampler: {0} RejectIteration:{1} TotalSolutionsGenerated:{2} TotalIdealSamples:{3}\n""".format(
            #        samplerString, i, totalSolutionsGenerated, totalIdealSamples
            #    )
            #)
            #f.close()
            breakExperiment = True
            print( "totalsol :", totalSolutionsGenerated)
            print("Rejected at iteration:", i)
            break
        print("Accepted")

        if (i >= t-1):
            print( "totalsol :", totalSolutionsGenerated)
            print("All rounds passed")

    #if not (breakExperiment):
    #    f = open(outputFile, "a")
    #    f.write(
    #        """Sampler: {0} Accept:1 TotalSolutionsGenerated: {1} TotalIdealSamples:
    #        {2}\n""".format(
    #           samplerString, totalSolutionsGenerated, totalIdealSamples
    #        )
    #    )
    #    f.close()

    if (nnf == 0):
        os.system("rm *cnf.*  > /dev/null 2>&1 ")
        os.system("rm *.u.cnf")
if __name__ == "__main__":
    barbarik2()

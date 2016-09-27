import re
import sys
from copy import deepcopy
from decimal import Decimal
from itertools import product

import itertools

# ______________________________________________________________________________

listOfQueries = []
T, F = True, False
keyPrintValue =[]


# ______________________________________________________________________________

class ProbDist:
    def __init__(self, varname='?', freqs=None):

        self.prob = {}
        self.varname = varname
        self.values = []

        if freqs:
            for (v, p) in list(freqs.items()):
                self[v] = p
            self.normalize()

    def __getitem__(self, val):

        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val, p):

        if val not in self.values:
            self.values.append(val)
        self.prob[val] = p

    def normalize(self):

        total = sum(self.prob.values())
        if not isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total

        return self

    def show_approx(self, numfmt='%.3g'):

        return ', '.join([('%s: ' + numfmt) % (v, p)
                          for (v, p) in sorted(self.prob.items())])

# ______________________________________________________________________________

def event_values(event, variables):

    #print variables
    if isinstance(event, tuple) and len(event) == len(variables):
        #print event
        return event
    else:
        #print tuple([event[var] for var in variables])
        return tuple([event[var] for var in variables])

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):

    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# ______________________________________________________________________________

class BayesNet:



    def __init__(self, node_specs=[]):

        self.nodes = []
        self.variables = []
        for node_spec in node_specs:
            self.add(node_spec)

    def add(self, node_spec):

        node = BayesNode(*node_spec)
        assert node.variable not in self.variables
        assert every(lambda parent: parent in self.variables, node.parents)
        self.nodes.append(node)
        self.variables.append(node.variable)
        for parent in node.parents:
            self.variable_node(parent).children.append(node)

    def variable_node(self, var):

        for n in self.nodes:
            if n.variable == var:
                return n
        raise Exception("No such variable: %s" % var)

    def variable_values(self, var):

        return [True,False]

    def __repr__(self):
        return 'BayesNet(%r)' % self.nodes
    def getValues(self):
        return self

# ______________________________________________________________________________

def every(predicate, seq): # TODO: replace with all
    """True if every element of seq satisfies predicate."""

    return all(predicate(x) for x in seq)

def doRound(a):
    val = Decimal(str(a)).quantize(Decimal('.01'))
    return val

class BayesNode:



    def __init__(self, X, parents, cpt):

        if isinstance(parents, str):
            parents = parents.split()

        # We store the table always in the third form above.
        if isinstance(cpt, (float, int)):  # no parents, 0-tuple
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # one parent, 1-tuple
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = dict(((v,), p) for v, p in list(cpt.items()))

        assert isinstance(cpt, dict)
        for vs, p in list(cpt.items()):
            assert isinstance(vs, tuple) and len(vs) == len(parents)
            assert every(lambda v: isinstance(v, bool), vs)
            #assert 0 <= p <= 1

        self.variable = X
        self.parents = parents
        self.cpt = cpt
        self.children = []

    def p(self, value, event):

        assert isinstance(value, bool)
        t= event_values(event, self.parents)
        ptrue = self.cpt[event_values(event, self.parents)]
        if(ptrue !=1):
            return (ptrue if value else 1 - ptrue)
        else:
            return (ptrue if value else ptrue)


    def __repr__(self):

        return repr((self.variable, ' '.join(self.parents)))

# ______________________________________________________________________________

def calculateJointProbablity(X, e, bn):

    assert X not in e, "Query variable must be distinct from evidence"
    Q = ProbDist(X)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)

    return Q.prob

def enumeration_ask(X, e, bn):

    assert X not in e, "Query variable must be distinct from evidence"
    listOfVarC = X.split(', ')
    lenOfVar = len(listOfVarC)
    Q = ProbDist(X)
    for xi in itertools.product(bn.variable_values(X), repeat = lenOfVar):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()

def extend(s, var, val):

    s2 = s.copy()
    t1 = var.split(', ')
    if(isinstance(val, tuple)):
        for k in range(0, len(val)):
            s2[t1[k]] = val[k]
    else:
        s2[var] = val
    return s2


def enumerate_all(variables, e, bn):

    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.variable_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))
# ______________________________________________________________________________
def parserInputFile():
    listOfNode = []
    listOfParents =[]
    bayesNetObject =[]

    storeLine =()
    mainNode ={}
    with open(sys.argv[2], "r") as infile:
        #To Get all the  Queries in the List
        for line in infile:
            parseLine = line.strip()
            if(parseLine == "******"):
                break
            else:
                listOfQueries.append(parseLine)
        #print listOfQueries
        storeTheRelationship =[]
        parentNodes =[]
        for line in infile:
            parseLine = line.strip()
            #print parseLine
            if(parseLine == "***" or parseLine == "******"):
                if(len(mainNode) != 0):
                    storeLine = storeLine + (mainNode,)
                bayesNetObject.append(storeLine)
                #empty all the object
                currentNode=[]
                parentNode=[]
                storeLine=()
                mainNode={}

            else:

                startString = parseLine.split(' ')[0]
                t = re.match("^[A-Za-z]", startString)
                if(t != None):

                    if(t.string == "decision"):
                        storeLine = storeLine + (1,)
                    elif(re.match("^[A-Za-z]",startString ) ):

                        storeTheRelationship =parseLine.split('|')
                        currentNode = storeTheRelationship[0].strip()
                        storeLine = storeLine + (currentNode,)
                        parentNodes = storeTheRelationship[1:len(storeTheRelationship)]
                        if (len(parentNodes) == 0):
                            storeLine = storeLine +('',)
                        else:
                            allParentNode = ' '.join(parentNodes).strip()
                            storeLine = storeLine + (allParentNode,)
                else:
                    #print "Its a number"
                    numberNode = parseLine.split(' ')
                    probvalue = float(numberNode[0])
                    truthTable = numberNode[1: len(numberNode)]

                    if (len(truthTable) == 0):
                        storeLine = storeLine + (probvalue,)

                    elif (len(truthTable) == 1):

                        if (truthTable[0] == '+'):
                            mainNode[T] = probvalue
                        else:
                            mainNode[F] = probvalue
                        #storeLine = storeLine + (mainNode,)

                    elif (len(truthTable) == 2):
                        tempTuple =()
                        if(truthTable[0] == "+" and truthTable[1] == "+"):
                            tempTuple = (T, T)
                        elif (truthTable[0] == "+" and truthTable[1] == "-"):
                            tempTuple = (T, F)
                        elif (truthTable[0] == "-" and truthTable[1] == "+"):
                            tempTuple = (F, T)
                        else:
                            tempTuple = (F, F)
                        mainNode[tempTuple] = probvalue

                    else :
                        tempTuple = ()
                        if (truthTable[0] == "+" and truthTable[1] == "+" and  truthTable[2] == "+"):
                            tempTuple = (T, T, T)
                        elif (truthTable[0] == "+" and truthTable[1] == "+" and truthTable[2] == "-"):
                            tempTuple = (T, T, F)
                        elif (truthTable[0] == "+" and truthTable[1] == "-"  and truthTable[2] == "+"):
                            tempTuple = (T, F, T)
                        elif (truthTable[0] == "+" and truthTable[1] == "-"  and truthTable[2] == "-"):
                            tempTuple = (T, F, F)
                        elif (truthTable[0] == "-" and truthTable[1] == "+" and truthTable[2] == "+"):
                            tempTuple = (F, T, T)
                        elif (truthTable[0] == "-" and truthTable[1] == "+" and truthTable[2] == "-"):
                            tempTuple = (F, T, F)
                        elif (truthTable[0] == "-" and truthTable[1] == "-"  and truthTable[2] == "+"):
                            tempTuple = (F, F, T)
                        elif (truthTable[0] == "-" and truthTable[1] == "-"  and truthTable[2] == "-"):
                            tempTuple = (F, F, F)

                        mainNode[tempTuple] = probvalue

            #storeLine = storeLine + (mainNode,)
        storeLine = storeLine + (mainNode,)
        bayesNetObject.append(storeLine)
        #print "***********"
        #print bayesNetObject
    return bayesNetObject

def calculateProbablity(i,Prg):

    if ('|' in i):
        varDict = {}
        qItem =[]
        nameOfVariable =[]
        queryVar = i[i.index("(") + 1:i.index(" | ")].split(', ')
        evidenceVar = i[i.index("|") + 1:i.index(")")].split(', ')
        argsList =[]
        varDict = getEvidenceDictionary(evidenceVar)
        for k in queryVar:
            if('=' in k):
                qItem = k.split(' = ')
                if (qItem[1].strip() == "+"):
                    keyPrintValue.append(T)
                    nameOfVariable.append(qItem[0])
                else:
                    keyPrintValue.append(F)
                    nameOfVariable.append(qItem[0])
            else:
                keyPrintValue.append(T)
                nameOfVariable.append(k)
                #qItem.append(k)
        argString = ', '.join(nameOfVariable)


        value = enumeration_ask(argString, varDict, Prg)
        return value.prob

    else:

        listVariable = i[i.index("(") + 1:i.index(")")].split(',')
        varDict = getEvidenceDictionary(listVariable)
        value = calculateJointProbablity('', varDict, Prg)
        keyPrintValue.append(T)
        return value



def calculateParents(utility, Prg):
        parentList = Prg.variable_node(utility)
        return parentList.parents

def calculateUtility(utility, Prg):
        parentList = Prg.variable_node(utility)
        return parentList.cpt


def getEvidenceDictionary(listVariable):
    varDict = {}
    for j in listVariable:
        gItem = j.split(' = ')
        if (gItem[1].strip() == "+"):
            varDict[gItem[0].strip()] = T
        else:
            varDict[gItem[0].strip()] = F

    return varDict


def calculateExpectedUtility(Prg,i):
    deleteDict ={}
    vTemp = calculateUtility("utility", Prg)

    listOfParents = calculateParents("utility", Prg)
    listOfEve = i[i.index("(") + 1:i.index(")")]
    if ('|' in listOfEve):
        temp = listOfEve.split('|')
        listOfEve = ', '.join(temp)
    kTemp = listOfEve.split(', ')
    temp = getEvidenceDictionary(kTemp)
    kItems = temp.keys()
    listCopy = deepcopy(listOfParents)

    for x in listCopy:
            if (x in temp ):
                deleteDict[x] = listCopy.index(x)
                listCopy.remove(x)



    #print deleteDict

    if (len(listOfParents) == 1):
        X = ', '.join(listOfParents)
        probsValue = enumeration_ask(X, temp, Prg)
        probsDict = probsValue.prob
        value =0
        for key1, value1 in probsDict.iteritems():
            for key2, value2 in vTemp.iteritems():
                if (key1 == key2):
                    value = value + (value1 * value2)



    elif (len(listOfParents) == 2):

        if(len(listCopy) == 2):
            X = ', '.join(listCopy)
            probsValue = enumeration_ask(X, temp, Prg)
            probsDict = probsValue.prob
            value = 0
            for key1, value1 in probsDict.iteritems():
                for key2, value2 in vTemp.iteritems():
                    if (key1 == key2):
                        value = value + (value1 * value2)
        else:
            X = ', '.join(listCopy)
            probsValue = enumeration_ask(X, temp, Prg)
            probsDict = probsValue.prob
            #print probsDict
            kList =[None, None]
            for key, value in deleteDict.items():
                kList[value] = temp.get(key)
            #print kList
            for k in range(0,1):
                if(kList[k] == None):
                    kList[k] = T
                    v1 = tuple(kList,)
                    kList[k] = F
                    v2 = tuple(kList,)
                    value = probsDict[(T,)]*vTemp[v1] + probsDict[(F,)]*vTemp[v2]


    elif (len(listCopy) == 3):

        if (len(listCopy) == 3):
            X = ', '.join(listCopy)
            probsValue = enumeration_ask(X, temp, Prg)
            probsDict = probsValue.prob
            value = 0
            for key1, value1 in probsDict.iteritems():
                for key2, value2 in vTemp.iteritems():
                    if (key1 == key2):
                        value = value + (value1 * value2)

        elif(len(listCopy) == 2):

            X = ', '.join(listCopy)
            probsValue = enumeration_ask(X, temp, Prg)
            probsDict = probsValue.prob
            # print probsDict
            kList = [None, None, None]
            for key, value in deleteDict.items():
                kList[value] = temp.get(key)
            posList =[]
            for t in kList:
                if(t == None):
                    posList.append(kList.index(t))
            for xi in itertools.product([T,F], repeat = 2):
                kList[posList[0]] = xi[0]
                kList[posList[1]] = xi[1]
                t = tuple(kList)
                value = value + probsDict[xi] * vTemp[t]

        else:
            X = ', '.join(listCopy)
            probsValue = enumeration_ask(X, temp, Prg)
            probsDict = probsValue.prob
            # print probsDict
            kList = [None, None, None]
            for key, value in deleteDict.items():
                kList[value] = temp.get(key)
            # print kList
            for k in range(0, 2):
                if (kList[k] == None):
                    kList[k] = T
                    v1 = tuple(kList, )
                    kList[k] = F
                    v2 = tuple(kList, )
                    value = probsDict[(T,)] * vTemp[v1] + probsDict[(F,)] * vTemp[v2]



    return value





def maximumExpectedUtility(Prg, ii):
    #print "Inside MaximumUtilty"
    tempQuery = ii[ii.index("(") + 1:ii.index(")")]
    tempString = tempQuery.replace(', ', ' | ')
    tempList = tempString.split(' | ')
    # print tempList
    varValue = {}
    for i in range(0, len(tempList)):
        tempValue = tempList[i].split(" = ")
        if len(tempValue) == 2:
            tempList[i] = tempValue[0]
            varValue[i] = tempValue[1]
    # print varValue
    # print tempList
    length = len(tempList)
    varList = ['+', '-']
    max = 0
    value = 0
    tup = ()
    flag = 0
    #query = Query()
    for xi in itertools.product(varList, repeat=length):
        argumentList = []
        for i in range(0, length):
            if varValue.has_key(i):
                if xi[i] == varValue.get(i):
                    flag = 0
                else:
                    flag = 1
            argumentList.append(tempList[i] + " = " + xi[i])
        tempString = ", ".join(argumentList)
        tempString = "EU(" + tempString +")"
        #query.value = tempString
        if flag == 1:
            flag = 0
        else:
            value = calculateExpectedUtility(Prg, tempString)
            if value > max:
                max = value
                tup = xi
    tempString = ""
    for i in range(0, len(tup)):
        if varValue.has_key(i):
            flag = 1
        else:
            tempString += tup[i] + " "
    tempString += str(int(max + 0.5))
    return tempString


def writeFile(a,f1):
    str5 = str(a) + "\n"
    f1.write(str5)


def parseQuery(listOfQueries):
    bayesianNetwork = parserInputFile()
    f1 = open("output.txt", 'w')

    #print bayesianNetwork
    Prg = BayesNet(bayesianNetwork)
    for i in listOfQueries:
        firstChar = i[:1]
        if(firstChar== "P"):
            valueReturn = calculateProbablity(i,Prg)
            x = keyPrintValue[0]
            checkType = valueReturn.keys()

            if (isinstance(checkType[0], tuple)):
                if(len(checkType[0]) == 1):
                    writeFile(doRound(valueReturn.get((x,))),f1)
                elif(len(checkType[0]) == 2):
                    tup = ()
                    #print nameOfVariable
                    #print keyPrintValue
                    for i in keyPrintValue:
                        if(i == T):
                            tup += (T,)
                        else:
                            tup+= (F,)
                    writeFile(doRound(valueReturn.get(tup)),f1)
                else:
                    tup = ()
                    #When there are three variable in left of conditional probablity
                    for i in keyPrintValue:
                        if (i == T):
                            tup += (T,)
                        else:
                            tup += (F,)
                    writeFile(doRound(valueReturn.get(tup)),f1)

            #Is instance of String: When there is no Conditional Probablity
            else:
                writeFile(doRound(valueReturn[x]),f1)

            del keyPrintValue[:]

        elif(firstChar == "E"):
            finalEUValue = calculateExpectedUtility(Prg,i)
            writeFile(int(finalEUValue+0.5),f1)

        elif(firstChar == "M"):
            finalMEUValue = maximumExpectedUtility(Prg,i)
            writeFile(finalMEUValue,f1)
            #return 0

#Main Program Starts Here

parseQuery(listOfQueries)


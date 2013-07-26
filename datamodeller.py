import numpy as np
import scipy
import sys
from sklearn import linear_model,svm,naive_bayes

def classify(func,xTrain,xTest,yTrain,yTest):        
        clf = func()
        clf.fit(xTrain, yTrain);
        yPred = clf.predict(xTest); 
        resultLR =  clf.score(xTest,yTest);  
        print "Accuracy =" , resultLR.mean();
        return yPred   

def printResult(yTest,yPred):
    dict = {}
    dict[1.0] = 'Apple'
    dict[2.0] = 'Orange'
    dict[3.0] = 'Lemon'
    rows = len(yPred)
    print "------\t----------"
    print "Acutal \tPredicted";
    print "------\t----------"
    for i in range(0,rows):
      print str(dict[yTest[i]])+"\t"+str(dict[yPred[i]])       
        
class DataModeller:
            
    def __init__(self, training_file, test_file):
        self.training_file = training_file
        self.test_file = test_file        
        
    def runAnalysis(self):
        
        trainingData = np.loadtxt(open(self.training_file, 'rb'), delimiter = ',');
        testData = np.loadtxt(open(self.test_file,'rb'), delimiter = ',');

        xTrain =  trainingData[:, :trainingData.shape[1]-1]
        yTrain = trainingData[:,trainingData.shape[1]-1]
                  
        xTest = testData[:, :testData.shape[1] -1]
        yTest = testData[:, testData.shape[1]-1]
        
        #MultinomialNB classification
        print "\n********MultiNB********"
        yPred = classify(lambda:naive_bayes.MultinomialNB(),xTrain,xTest,yTrain,yTest)
        printResult(yTest,yPred);
                                  
        #SVM based classification
        print "\n********Support Vector Machines********"
        yPred = classify(lambda:svm.SVC(),
                 xTrain,xTest,yTrain,yTest)        
        printResult(yTest,yPred);
        
        #Logistic Regression classification
        print "\n********Log regression********"
        yPred = classify(lambda:linear_model.LogisticRegression(),
                 xTrain,xTest,yTrain,yTest)
        printResult(yTest,yPred);

              
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Please provide the training file and test file'
        print 'python datamodeller.py <training-file-path> <test_file> '
        sys.exit(1)
    training_file = sys.argv[1]
    test_file = sys.argv[2]

    model = DataModeller(training_file, test_file)
    model.runAnalysis()
        
        


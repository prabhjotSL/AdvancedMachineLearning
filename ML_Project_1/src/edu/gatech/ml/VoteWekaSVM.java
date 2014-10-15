package edu.gatech.ml;

import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class VoteWekaSVM {

    private static final String location = "data/vote.arff";

    public static void main(String[] args) throws Exception {

        Instances[] spliDataSets = MLUtils.getDatasets(location);
        Instances trainingSet = spliDataSets[0];
        Instances testSet = spliDataSets[2];

        // train classifier
        SMO classifier = new SMO(); // new instance of tree
        classifier.setOptions(new String[] {"-C", "1.0", "-L", "0.001", "-P", "1.0E-12", "-N", "0", "-M", "-V", "-1",
                "-W", "1", "-K", "weka.classifiers.functions.supportVector.NormalizedPolyKernel -C 250007 -E 2.0"});
        classifier.buildClassifier(trainingSet);

        MLUtils.runTest(classifier, trainingSet, testSet);
    }
}
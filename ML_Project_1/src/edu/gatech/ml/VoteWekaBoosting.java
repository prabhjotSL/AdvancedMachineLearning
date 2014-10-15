package edu.gatech.ml;

import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;

public class VoteWekaBoosting {

    private static final String location = "data/vote.arff";

    public static void main(String[] args) throws Exception {

        Instances[] spliDataSets = MLUtils.getDatasets(location);
        Instances trainingSet = spliDataSets[0];
        Instances testSet = spliDataSets[2];

        // train classifier
        AdaBoostM1 classifier = new AdaBoostM1(); // new instance of tree
        classifier.setOptions(new String[] {"-P", "100", "-S", "1", "-I", "10", "-W", "weka.classifiers.trees.J48",
                "--", "-R", "-N", "3", "-Q", "1", "-M", "2"});
        classifier.buildClassifier(trainingSet);

        MLUtils.runTest(classifier, trainingSet, testSet);

    }
}
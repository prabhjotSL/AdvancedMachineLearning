package edu.gatech.ml;

import weka.classifiers.trees.J48;
import weka.core.Instances;

public class VoteWekaDecisionTree {

    private static final String location = "data/vote.arff";

    public static void main(String[] args) throws Exception {

        Instances[] spliDataSets = MLUtils.getDatasets(location);
        Instances trainingSet = spliDataSets[0];
        Instances testSet = spliDataSets[2];

        // train classifier
        J48 classifier = new J48(); // new instance of tree
        // classifier.setOptions(new String[] {"-C", "0.25", "-M", "2"}); // set the options
        classifier.setOptions(new String[] {"-R", "-N", "3", "-Q", "1", "-M", "2"}); // set the options
        classifier.buildClassifier(trainingSet);

        MLUtils.runTest(classifier, trainingSet, testSet);

        System.out.println("\n" + classifier.toSummaryString());
    }
}
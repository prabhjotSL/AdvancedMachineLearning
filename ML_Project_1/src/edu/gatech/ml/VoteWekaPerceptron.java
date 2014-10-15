package edu.gatech.ml;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class VoteWekaPerceptron {

    private static final String location = "data/vote.arff";

    public static void main(String[] args) throws Exception {

        Instances[] spliDataSets = MLUtils.getDatasets(location);
        Instances trainingSet = spliDataSets[0];
        Instances testSet = spliDataSets[2];

        // train classifier
        MultilayerPerceptron classifier = new MultilayerPerceptron(); // new instance of tree
        // Classifier classifier = new Id3(); // new instance of tree
        classifier.setOptions(new String[] {"-L", "0.3", "-M", "0.2", "-N", "500", "-V", "0", "-S", "0", "-E", "20",
                "-H", "a"}); // set
                             // the
                             // options
        classifier.buildClassifier(trainingSet);

        MLUtils.runTest(classifier, trainingSet, testSet);
    }
}
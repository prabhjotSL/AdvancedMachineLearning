package edu.gatech.ml;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class VoteWekaKNN {

    private static final String location = "data/vote.arff";

    public static void main(String[] args) throws Exception {

        Instances[] spliDataSets = MLUtils.getDatasets(location);

        Instances trainingSet = spliDataSets[0];
        Instances testSet = spliDataSets[2];
        Remove rm = new Remove();
        rm.setAttributeIndices("1,3,5-8,10,12-16");
        rm.setInputFormat(trainingSet);

        trainingSet = Filter.useFilter(trainingSet, rm);
        testSet = Filter.useFilter(testSet, rm);

        // train classifier
        Classifier classifier = new IBk();
        classifier.setOptions(new String[] {"-K", "3"});
        classifier.buildClassifier(trainingSet);

        MLUtils.runTest(classifier, trainingSet, testSet);
    }
}
package edu.gatech.ml;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class MLUtils {

    public static boolean dataInfoPrinted = false;

    public static Instances[] getDatasets(String location) throws Exception {
        Instances data = getData(location);
        Instances[] spliDataSets = spliDataSets(data);
        if (!dataInfoPrinted) {
            System.out.println("Dataset size: " + data.numInstances());
            System.out.println("Training Set size: " + spliDataSets[0].numInstances());
            // System.out.println("cv size: " + spliDataSets[1].numInstances());
            System.out.println("Test Set size: " + spliDataSets[2].numInstances());
            dataInfoPrinted = true;
        }

        // print(trainingSet);
        // print(cvSet);
        // print(testSet);

        return spliDataSets;
    }

    public static void runTest(Classifier classifier, Instances trainingSet, Instances testSet) throws Exception {
        long start = System.currentTimeMillis();
        // evaluate classifier and print some statistics
        Evaluation eval = new Evaluation(trainingSet);
        eval.evaluateModel(classifier, testSet);

        long stop = System.currentTimeMillis();
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Execution time for testing: " + (stop - start));
    }

    public static void print(Instances trainingSet) {
        for (int i = 0; i < trainingSet.numInstances(); i++) {
            Instance instance = trainingSet.instance(i);
            System.out.println(instance);
        }
    }

    public static Instances getData(String location) throws Exception {
        /* Load a data set */
        DataSource source = new DataSource(location);
        Instances data = source.getDataSet();

        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    private static Instances[] spliDataSets(Instances data) throws Exception {

        // training data 60%
        Instances trainSet = filterData(data, true, false, 1, 70);
        Instances remainingSet = filterData(data, true, true, 1, 70);

        // cv 20%
        Instances crossValidationSet = filterData(remainingSet, true, false, 1, 50);

        // test set 20%
        Instances testSet = filterData(remainingSet, true, true, 1, 50);

        // return new Instances[] {trainSet, crossValidationSet, testSet};
        return new Instances[] {trainSet, null, remainingSet};

    }

    public static Instances filterData(Instances dataSet, boolean noReplacement, boolean invertSelection, int seed,
            int sampleSizePercent) throws Exception {
        Resample resampleFilter = new Resample();
        resampleFilter.setNoReplacement(noReplacement);
        resampleFilter.setInvertSelection(invertSelection);
        resampleFilter.setRandomSeed(seed);
        resampleFilter.setSampleSizePercent(sampleSizePercent);
        resampleFilter.setInputFormat(dataSet);
        Instances trainSet = Filter.useFilter(dataSet, resampleFilter); // apply filter
        return trainSet;
    }
}

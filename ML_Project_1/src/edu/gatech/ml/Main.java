package edu.gatech.ml;

public class Main {

    private static final String location = "data/vote.arff";

    public static void main(String[] args) throws Exception {

        MLUtils.getDatasets(location);

        System.out.println("------------------------------");
        System.out.println("Decision Tree");
        System.out.println("------------------------------");
        VoteWekaDecisionTree.main(args);
        System.out.println("------------------------------");
        System.out.println("\n");

        System.out.println("------------------------------");
        System.out.println("Support Vector Machine");
        System.out.println("------------------------------");
        VoteWekaSVM.main(args);
        System.out.println("\n");

        System.out.println("------------------------------");
        System.out.println("Neural Networks");
        System.out.println("------------------------------");
        VoteWekaPerceptron.main(args);
        System.out.println("------------------------------");
        System.out.println("\n");

        System.out.println("------------------------------");
        System.out.println("k-nearest neighbors");
        System.out.println("------------------------------");
        VoteWekaKNN.main(args);
        System.out.println("------------------------------");
        System.out.println("\n");

        System.out.println("------------------------------");
        System.out.println("Boosting");
        System.out.println("------------------------------");
        VoteWekaBoosting.main(args);
        System.out.println("------------------------------");
        System.out.println("\n");

    }

}

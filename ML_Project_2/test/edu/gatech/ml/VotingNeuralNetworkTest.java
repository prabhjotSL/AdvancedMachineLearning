package edu.gatech.ml;

import java.io.IOException;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer;
import shared.Instance;
import shared.SumOfSquaresError;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;

/**
 * Modified version of Andrew Guillory's optimization test for neural networks. Using the voting
 * dataset to test a neural network
 */
public class VotingNeuralNetworkTest {

  public static void main(String[] args) throws IOException {
    BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    Instance[][] dataSets = DataLoader.initializeInstances();
    Instance[] trainingData = dataSets[0];
    Instance[] testData = dataSets[1];

    BackPropagationNetwork network = factory
        .createClassificationNetwork(NetworkConfig.networkConfig);
    ErrorMeasure measure = new SumOfSquaresError();
    DataSet set = new DataSet(trainingData);
    NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(set, network,
        measure);
    OptimizationAlgorithm o = new RandomizedHillClimbing(nno);
    FixedIterationTrainer fit = new FixedIterationTrainer(o, NetworkConfig.trainingIterations);

    double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;

    fit.train();

    end = System.nanoTime();
    trainingTime = end - start;
    trainingTime /= Math.pow(10, 9);

    Instance opt = o.getOptimal();
    network.setWeights(opt.getData());

    double predicted, actual;
    start = System.nanoTime();
    for (Instance instance : testData) {
      network.setInputValues(instance.getData());
      network.run();
      System.out.println("~~");
      System.out.println(instance.getLabel());
      System.out.println(network.getOutputValues());

      predicted = Double.parseDouble(instance.getLabel().toString());
      actual = Double.parseDouble(network.getOutputValues().toString());

      double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
    }

    end = System.nanoTime();
    testingTime = end - start;
    testingTime /= Math.pow(10, 9);

    String results = MLUtil.outputResult("RHC", trainingTime, testingTime, correct, incorrect);
    System.out.println(results);
  }

}

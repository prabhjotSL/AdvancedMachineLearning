package edu.gatech.ml;

import static edu.gatech.ml.NetworkConfig.trainingIterations;

import java.io.IOException;
import java.text.DecimalFormat;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import util.linalg.Vector;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;

/**
 * Modified version of Hannah Lau's optimization test for neural networks using different
 * optimization algorithms. Using the voting dataset to test a neural network. Implementation of
 * randomized hill climbing, simulated annealing, and genetic algorithm to find optimal weights to a
 * neural network that is classifying voting data
 */
public class VotingOptimazationTest {

  static DecimalFormat decimalFormat = new DecimalFormat("0.000");

  public static void main(String[] args) throws IOException {
    Instance[][] dataSets = DataLoader.initializeInstances();
    Instance[] trainingDataSet = dataSets[0];
    DataSet trainingSet = new DataSet(trainingDataSet);
    Instance[] testData = dataSets[1];

    FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();

    ErrorMeasure measure = new SumOfSquaresError();

    FeedForwardNetwork networks[] = new FeedForwardNetwork[3];
    NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    String[] oaNames = { "RHC", "SA", "GA" };

    for (int i = 0; i < oa.length; i++) {
      networks[i] = factory.createClassificationNetwork(NetworkConfig.networkConfig);
      nnop[i] = new NeuralNetworkOptimizationProblem(trainingSet, networks[i], measure);
    }

    oa[0] = new RandomizedHillClimbing(nnop[0]);

    double startingTempature = 1E11;
    double coolingExponent = .95;
    oa[1] = new SimulatedAnnealing(startingTempature, coolingExponent, nnop[1]);

    int populationSize = 200;
    int toMate = 100;
    int toMutate = 10;
    oa[2] = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnop[2]);

    for (int i = 0; i < oa.length; i++) {
      double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
      boolean printCurrentError = false;

      OptimizationAlgorithm optimizationAlgorithm = oa[i];

      // training
      train(optimizationAlgorithm, networks[i], oaNames[i], printCurrentError, trainingDataSet,
          measure); // trainer.train();

      end = System.nanoTime();
      trainingTime = end - start;
      trainingTime /= Math.pow(toMutate, 9);

      Instance optimalInstance = optimizationAlgorithm.getOptimal();
      Vector weights = optimalInstance.getData();
      networks[i].setWeights(weights);

      double predicted, actual;
      start = System.nanoTime();
      for (Instance instance : testData) {
        networks[i].setInputValues(instance.getData());
        networks[i].run();

        predicted = Double.parseDouble(instance.getLabel().toString());
        actual = Double.parseDouble(networks[i].getOutputValues().toString());

        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

      }
      end = System.nanoTime();
      testingTime = end - start;
      testingTime /= Math.pow(toMutate, 9);

      System.out.println(MLUtil.outputResult(oaNames[i], trainingTime, testingTime, correct,
          incorrect));
    }

  }

  private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName,
      boolean printCurrentError, Instance[] instances, ErrorMeasure measure) {
    if (printCurrentError) {
      System.out.println("\nError results for " + oaName + "\n---------------------------");
    }

    for (int i = 0; i < trainingIterations; i++) {
      oa.train();

      if (printCurrentError) {
        double error = 0;
        for (Instance instance : instances) {
          network.setInputValues(instance.getData());
          network.run();

          Vector outputValues = network.getOutputValues();

          Instance output = instance.getLabel(), example = new Instance(outputValues);
          double parseDouble = Double.parseDouble(outputValues.toString());
          example.setLabel(new Instance(parseDouble));
          error += measure.value(output, example);
        }

        System.out.println(decimalFormat.format(error));
      }
    }
  }

}

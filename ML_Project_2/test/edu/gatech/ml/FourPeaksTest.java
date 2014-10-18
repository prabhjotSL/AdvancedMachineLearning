package edu.gatech.ml;

import java.text.DecimalFormat;
import java.util.Random;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

/**
 * Copied from ContinuousPeaksTest
 *
 * @version 1.0
 */
public class FourPeaksTest {

  static DecimalFormat df = new DecimalFormat("0.000");

  /** The n value */
  private static final int N = 50;
  /** The t value */
  private static final int T = N / 5;

  public static void main(String[] args) {
    int[] ranges = new int[N];
    Random random = new Random();
    for (int i = 0; i < ranges.length; i++) {
      ranges[i] = (random.nextInt(i + 1) % 2) + 1;
    }
    System.out.println("data size: " + N);
    EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
    Distribution odd = new DiscreteUniformDistribution(ranges);
    NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
    MutationFunction mf = new DiscreteChangeOneMutation(ranges);
    CrossoverFunction cf = new SingleCrossOver();
    Distribution df = new DiscreteDependencyTree(.1, ranges);
    HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
    GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
    ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

    FixedIterationTrainer fit = null;

    long start = System.nanoTime();
    SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
    fit = new FixedIterationTrainer(sa, 200000);
    fit.train();
    System.out.println("SA: " + ef.value(sa.getOptimal()));
    printExecTime(start);

    start = System.nanoTime();
    StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
    fit = new FixedIterationTrainer(ga, 1000);
    fit.train();
    System.out.println("GA: " + ef.value(ga.getOptimal()));
    printExecTime(start);

    start = System.nanoTime();
    MIMIC mimic = new MIMIC(200, 20, pop);
    fit = new FixedIterationTrainer(mimic, 1000);
    fit.train();
    System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
    printExecTime(start);
  }

  private static void printExecTime(long start) {
    long end = System.nanoTime();
    long testingTime = end - start;
    testingTime /= Math.pow(10, 9);
    System.out.println("execution time: " + df.format(testingTime) + "s");
  }
}

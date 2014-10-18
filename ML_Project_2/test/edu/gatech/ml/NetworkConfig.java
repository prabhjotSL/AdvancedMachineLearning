package edu.gatech.ml;

public class NetworkConfig {

  static int numOfAttributes = 16; // 16 attributes
  static int inputLayer = numOfAttributes;
  static int hiddenLayer = 9;
  static int outputLayer = 1;
  static int trainingIterations = 2000;
  static int[] networkConfig = new int[] { numOfAttributes, hiddenLayer, outputLayer };
}

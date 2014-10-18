package edu.gatech.ml;

import java.text.DecimalFormat;

public class MLUtil {

  static DecimalFormat df = new DecimalFormat("0.000");

  public static String outputResult(String algorithName, double trainingTime, double testingTime,
      double correct, double incorrect) {
    String results = "\nResults for network using " + algorithName
        + " Algorithm.\n Correctly classified " + correct + " instances."
        + "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
        + df.format((correct / (correct + incorrect)) * 100) + "%\nTraining time: "
        + df.format(trainingTime) + "s\nTesting time: " + df.format(testingTime) + "s\n";

    return results;
  }
}

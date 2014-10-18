package edu.gatech.ml;

import static edu.gatech.ml.NetworkConfig.numOfAttributes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import shared.Instance;

public class DataLoader {

  @SuppressWarnings("resource")
  static Instance[][] initializeInstances() throws IOException {

    int numOfRecords = 435;
    int testDataSetSize = 140;

    BufferedReader br = new BufferedReader(new FileReader(new File("data/house-votes-84.data.txt")));

    int size = numOfRecords - testDataSetSize;
    Instance[] trainingData = getRangeData(br, size);
    Instance[] testData = getRangeData(br, testDataSetSize);
    return new Instance[][] { trainingData, testData };
  }

  private static Instance[] getRangeData(BufferedReader br, int size) throws IOException {
    double[][][] attributes = new double[size][][];

    for (int i = 0; i < attributes.length; i++) {
      String[] scan = br.readLine().split(",");

      attributes[i] = new double[2][];
      attributes[i][0] = new double[numOfAttributes];
      attributes[i][1] = new double[1];

      attributes[i][1][0] = getClass(scan[0]);

      for (int j = 0; j < numOfAttributes; j++) {
        attributes[i][0][j] = getAttrValue(scan[j + 1]);
      }
// System.out.println(Arrays.toString(attributes[i][0]) + " - "
// + Arrays.toString(attributes[i][1]));
    }
    Instance[] instances = new Instance[attributes.length];

    for (int i = 0; i < instances.length; i++) {
      instances[i] = new Instance(attributes[i][0]);
      instances[i].setLabel(new Instance(attributes[i][1][0]));
    }

    return instances;
  }

  private static double getAttrValue(String attrVal) {

    if (attrVal.equals("?")) {
      return -1;
    } else if (attrVal.equals("n")) {
      return 0;
    } else if (attrVal.equals("y")) {
      return 1;
    } else {
      throw new IllegalArgumentException("couldn't map attribute");
    }
  }

  private static double getClass(String targetClass) {
    if (targetClass.equals("democrat")) {
      return 0;
    } else if (targetClass.equals("republican")) {
      return 1;
    } else {
      throw new IllegalArgumentException("couldn't map class");
    }
  }

  public static void main(String[] args) throws IOException {
    initializeInstances();
  }
}

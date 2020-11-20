/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Project Phase 3
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.metah.ga;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.genetic.genome.DoubleArrayGenome;

import java.util.Random;

/**
 * This class calculates the fitness of an individual chromosome or phenotype.
 */
class Circuit1Objective implements CalculateScore {

    public final static boolean DEBUGGING = false;
    public final static String TEAM = "Abuchi Obiegbu";
    public final static int NUM_WEIGHTS = 10;
    public final static double RANGE_MAX = 10.0;
    public final static double RANGE_MIN = -10.0;
    protected static Random ran = null;
    static {
        long seed = System.nanoTime();
        if(DEBUGGING)
            seed = TEAM.hashCode();
        ran = new Random(seed);
    }

    /**
     * The input necessary for XOR.
     */
    public final static double XOR_INPUTS[][] = {
            /* x1   x2   x3 */
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 1.0},
            {0.0, 1.0, 0.0},
            {0.0, 1.0, 1.0},
            {1.0, 0.0, 0.0},
            {1.0, 0.0, 1.0},
            {1.0, 1.0, 0.0},
            {1.0, 1.0, 1.0}};

    /**
     * The ideal data necessary for XOR.
     */
    public final static double XOR_IDEALS[][] = {
            {1.0},
            {1.0},
            {0.0},
            {1.0},
            {0.0},
            {0.0},
            {0.0},
            {1.0}};

    /**
     * Get a uniform random deviated from the weights constraints
     * @return random weight
     */
    public static double getRandomWeight() {
        ran = new Random(); // instance of random class

        return ran.nextDouble() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN;
    }

    /**
     * Sigmoid activation function
     * @param z input
     * @return sigmoid value of z
     */
    protected double sigmoid(double z) {
        double y = 1.0 / (1 + Math.exp(-z));
        return y;
    }

    /**
     * Implements the three feedforward equations to calculate the actual output
     * @param x1 XOR input
     * @param x2 XOR input
     * @param x3 XOR input
     * @param ws Weights w1-w8 and b1 and b2 -- in this order
     * @return Actual, that is, Y1
     */
    public double feedforward(double x1, double x2, double x3, double[] ws) {
        // local variables to hold w1-w8, b1, b2 values
        double w1 = ws[0];
        double w2 = ws[1];
        double w3 = ws[2];
        double w4 = ws[3];
        double w5 = ws[4];
        double w6 = ws[5];
        double w7 = ws[6];
        double w8 = ws[7];
        double b1 = ws[8];
        double b2 = ws[9];

        /* Implements Equations 1, 2 and 3 */
        double zh1 = w1 * x1 + w3 * x2 + b1 * 1.0 + w7 * x3;
        double zh2 = w2 * x1 + w4 * x2 + b1 * 1.0 + w8 * x3;
        double h1 = sigmoid(zh1);
        double h2 = sigmoid(zh2);
        double zy1 = w5 * h1 + w6 * h2 + b2 * 1.0;
        double y1 = sigmoid(zy1);

        return y1;
    }

    /**
     * Calculates the fitness of interneuron weights using the root mean square error
     * @param ws interneuron weights
     * @return Fitness value
     */
    public double getFitness(double[] ws) {
        // Sum of square error
        double sumSqrErr = 0.0;

        // This is the encoded column header
        String reportHeader = String.format("%2s %4s %7s %7s %7s %7s","#", "x1", "x2", "x3", "t1", "y1");

        // display only when debug is turned on
        if (DEBUGGING){
            System.out.println(reportHeader);
        }

        // loop through all the XOR_INPUTS
        for(int k=0; k < XOR_INPUTS.length; k++) {
            double x1 = XOR_INPUTS[k][0];
            double x2 = XOR_INPUTS[k][1];
            double x3 = XOR_INPUTS[k][2];

            // calculate actual output
            double y1 = feedforward(x1, x2, x3, ws);

            // ideal output
            double t1 = XOR_IDEALS[k][0];

            // Square error
            double delta = (y1 - t1);
            double sqrError = square(delta);

            // Sum the square error
            sumSqrErr += sqrError;

            // display only when debug is turned on
            if (DEBUGGING) {
                System.out.printf("%2d %8.4f %7.4f %7.4f %7.4f %7.4f",
                        k+1, x1, x2, x3, t1, y1);
                System.out.println();
            }
        }

        // calculate RMSE
        double rmse = Math.sqrt(sumSqrErr / XOR_INPUTS.length);

        return rmse;
    }

    /**
     * Calculates the square of a number
     * @param num number input
     * @return  square of number in double precision
     */
    public double square(double num) {
        return num * num;
    }

    /**
     * Calculates the fitness.
     * @param phenotype Individual
     * @return Objective
     */
    @Override
    public double calculateScore(MLMethod phenotype) {
        DoubleArrayGenome genome = (DoubleArrayGenome) phenotype;

        // get the interneuron weights from the genome and returns fitness
        return getFitness(genome.getData());
    }

    /**
     * Specifies the objective
     * @return True to minimize, false to maximize.
     */
    @Override
    public boolean shouldMinimize() {
        return true;
    }

    /**
     * Specifies the threading approach.
     * @return True to use single thread, false for multiple threads
     */
    @Override
    public boolean requireSingleThreaded() {
        return true;
    }

    /**
     * Objective function
     * @param x Domain parameter.
     * @return y
     */
    protected int f(int x) {
        return (x - 3)*(x - 3);
    }

    /**
     * The main method.
     * @param args No arguments are used.
     */
    public static void main(String[] args) {

        // array of weights
        double[] ws = new double[NUM_WEIGHTS];
        // Holds the wt names
        String[] wtNames = {"w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "b1", "b2"};

        // connect fitness to getFitness -> call it inside
        String wsValue;

        // randomize the interneuron weights
        for(int k=0; k < ws.length; k++)
            ws[k] = getRandomWeight();

        // populate the weight header
        String header = String.format("%2s %10s", "Wt","Value");

        // output the weight header
        System.out.println(header);

        // output the weight report
        for(int k=0; k < ws.length; k++) {
            System.out.printf("%2s", wtNames[k]);
            if (ws[k] < 0) {
                wsValue = String.format("%11.5f", ws[k]);
            } else {
                wsValue = String.format(" %10.5f", ws[k]);
            }
            System.out.println(wsValue);
        }

        // instance of Circuit1
        Circuit1Objective objective = new Circuit1Objective();

        // get fitness value
        double fitness = objective.getFitness(ws);

        // display only when debug is turned on
        if (DEBUGGING) {
            System.out.println("fitness = " + fitness);
        }
    }
}
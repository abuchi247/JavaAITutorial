/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Project Phase 2
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.metah.ga;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.ml.genetic.genome.IntegerArrayGenome;

import java.util.Random;

import static javaai.util.Helper.asInt;

/**
 * This class calculates the fitness of an individual chromosome or phenotype.
 */
class Circuit1Objective implements CalculateScore {

    public final static boolean DEBUGGING = true;
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
     * Calculates the sigmoid activation function
     * @param z value to calculate
     * @return sigmoid value of z
     */
    protected double sigmoid(double z) {
        return (1/(1 + Math.pow(Math.E, (-1*z))));
    }

    /**
     * Implements the three feedforward equations to calculate the actual output
     * @param x1 XOR input
     * @param x2 XOR input
     * @param x3 XOR input
     * @param ws interneuron weights
     * @return double precision value of y1
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
        double h1 = sigmoid(w1 * x1 + w3 * x2 + b1 * 1 + w7 * x3);
        double h2 = sigmoid(w2 * x1 + w4 * x2 + b1 * 1 + w8 * x3);
        double y1 = sigmoid(w5 * h1 + w6 * h2 + b2 * 1);

        return y1;
    }

    /**
     * Calculates the fitness of interneuron weights using the root mean square error
     * @param ws interneuron weights
     * @return double precision root mean square error value
     */
    public double getFitness(double[] ws) {
        double y1 = 0.0;
        double t1;
        double squareValue = 0.0;
        double mean = 0.0;
        double root = 0.0;

        // This is the encoded column header
        String reportHeader = String.format("%2s %4s %7s %7s %7s %7s","#", "x1", "x2", "x3", "t1", "y1");

        // display only when debug is turned on
        if (DEBUGGING){
            System.out.println(reportHeader);
        }

        // loop through all the XOR_INPUTS
        for(int row=0; row < XOR_INPUTS.length; row++) {
            // cslculate actual output
            y1 = feedforward(XOR_INPUTS[row][0], XOR_INPUTS[row][1], XOR_INPUTS[row][2], ws);

            // ideal output
            t1 = XOR_IDEALS[row][0];

            // compute square value
            squareValue += square(y1 - t1);

            // display only when debug is turned on
            if (DEBUGGING) {
                System.out.printf("%2d %8.4f %7.4f %7.4f %7.4f %7.4f",
                        row+1, XOR_INPUTS[row][0], XOR_INPUTS[row][1], XOR_INPUTS[row][2], t1, y1);
                System.out.println();
            }
        }

        // calculate mean
        mean = (squareValue / (double)XOR_INPUTS.length);

        // calculate root
        root = (double)Math.sqrt(mean);

        return root;
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
        IntegerArrayGenome genome = (IntegerArrayGenome) phenotype;

        int x = asInt(genome);

        double y = f(x);

        return y;
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

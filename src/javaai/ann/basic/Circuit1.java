/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 4
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */

package javaai.ann.basic;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

/**
 * This class implements the design and train a multilayer perceptron to
 * learn a novel circuit
 *
 * @author Abuchi Obiegbu
 * @date 15 Sept 2020
 */
public class Circuit1 {
    /** Error tolerance */
    public final static double TOLERANCE = 0.01;

    /**
     * The input necessary for XOR.
     */
    public static double XOR_INPUTS[][] = {
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
    public static double XOR_IDEALS[][] = {
            {1.0},
            {1.0},
            {0.0},
            {1.0},
            {0.0},
            {0.0},
            {0.0},
            {1.0}};

    /**
     * The main method.
     *
     * @param args No arguments are used.
     */
    public static void main(final String args[]) {

        // Create a neural network, without using a factory
        BasicNetwork network = new BasicNetwork();

        // Add input layer with no activation function, bias enabled, and three neurons
        network.addLayer(new BasicLayer(null, true, 3));

        // Add hidden layer with ramped activation, bias enabled, and five neurons
        // NOTE: ActivationReLU is not in javadoc but can be found here http://bit.ly/2zyxk7A.
        // network.addLayer(new BasicLayer(new ActivationReLU(), true, 5));

        // Add hidden layer with sigmoid activation, bias enabled, and two neurons
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 2));

        // Add output layer with sigmoid activation, bias disable, and one neuron
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));

        // No more layers to be added
        network.getStructure().finalizeStructure();

        // Randomize the weights
        network.reset();

        // Create training data
        MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUTS, XOR_IDEALS);

        // Train the neural network.
        // Use a training object to train the network, in this case, an improvement
        // back propagation. For details on what this does see the javadoc.
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        int epoch = 1;

        // Never train on a specific error rate but an acceptable tolerance and
        // if the error drops below that tolerance, the network has converged.
        do {
            long then = System.nanoTime();

            train.iteration();

            long now = System.nanoTime();

            long elapsed = now - then;

            System.out.println("dt: "+elapsed+ " epoch #" + epoch + " error: " + train.getError());

            epoch++;
        } while (train.getError() > TOLERANCE);

        train.finishTraining();

        // Test the neural network
        System.out.println("Neural Network Results:");

        for (MLDataPair pair : trainingSet) {

            final MLData output = network.compute(pair.getInput());

            System.out.println(pair.getInput().getData(0) + "," + pair.getInput().getData(1)
                    + "," + pair.getInput().getData(2) + ", actual=" + output.getData(0)
                    + ", ideal=" + pair.getIdeal().getData(0));
        }

        Encog.getInstance().shutdown();
    }
}

/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 10
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.ann.basic;

import javaai.util.Helper;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.Equilateral;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.arrayutil.NormalizationAction;
import org.encog.util.arrayutil.NormalizedField;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static javaai.ann.output.Ontology.parsers;

/**
 *  This class constructs an Encog neural network to predict the report
 *  from the XOR operator. It extracts, transforms, and loads data in
 *  column-major form to make normalization easier and then flips the
 *  column-major normalized data into a row-major form required by
 *  Encog. A training and testing set is generated to be use for testing the
 *  multilayer perceptron.
 *
 * @author Abuchi Obiegbu
 * @date 30 Oct 2020
 */
public class AbuchizIris {
    /** Error tolerance */
    public final static double TOLERANCE = 0.01;

    /** data to be used for training purposes */
    public static final double TRAINING_PERCENT = 0.80;

    /** The high range index */
    public final static int HI = 1;

    /** The low range index */
    public final static int LO = 0;

    public final static int SPECIES = 4;

    /** To store the report header */
    public static String[] reportHeader = new String[4];

    public static Equilateral eq;

    public static List<String> subtypes;

    /** For training purposes */
    public static double TRAINING_INPUTS[][];
    public static double TRAINING_IDEALS[][];

    /** For testing purposes */
    public static double TESTING_INPUTS[][];
    public static double TESTING_IDEALS[][];

    /**
     * The main method.
     * @param args No arguments are used.
     */
    public static void main(final String args[]) {
        // Initialize init method
        init();

        // Build the network
        BasicNetwork network = new BasicNetwork();

        // Input layer plus bias node
        network.addLayer(new BasicLayer(null, true, 4));

        // Hidden layer plus bias node
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 4));

        // Output layer
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 2));

        // No more layers to be added
        network.getStructure().finalizeStructure();

        // Randomize the weights
        network.reset();

        System.out.println("Network description: before training");

        Helper.describe(network);

        // Create training data
        MLDataSet trainingSet = new BasicMLDataSet(TRAINING_INPUTS, TRAINING_IDEALS);

        // Use a training object for the learning algorithm, in this case, an improved
        // backpropagation. For details on what this does see the javadoc.
        final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

        // Set learning batch size: 0 = batch, 1 = online, n = batch size
        // See org.encog.neural.networks.training.BatchSize
        // train.setBatchSize(0);

        int epoch = 0;

        Helper.log(epoch, train,false);
        do {
            train.iteration();

            epoch++;

            Helper.log(epoch, train,false);

        } while (train.getError() > TOLERANCE && epoch < Helper.MAX_EPOCHS);

        train.finishTraining();

        Helper.log(epoch, train,true);
        Helper.report(trainingSet, network);
        System.out.println("Network description: after training");
        Helper.describe(network);

        Encog.getInstance().shutdown();

        // to keep count of the number of mistakes the ANN makes
        int missed = 0;

        // to store the equilateral values on which the network was trained
        double[] output = new double[2];

        // display testing result title
        System.out.println("Network testing results:");

        // populating header display format
        String header = String.format("%2s %11s %11s ","#","Ideal","Actual");
        String report;
        int NUM_TESTS = TESTING_INPUTS.length;

        System.out.println(header);

        for(int k=0; k < TESTING_INPUTS.length; k++) {
            double[] input = TESTING_INPUTS[k];

            // Send input into MLP and get its output
            network.compute(input, output);
            // Decode output to its actual subtype index.
            int actualIndex = eq.decode(output);

            // Decode ideal to its subtype index.
            int idealIndex = eq.decode(TESTING_IDEALS[k]);

            // Output actual & ideal specie string names.
            report = String.format("%2d %11s %11s ", k+1, subtypes.get(idealIndex), subtypes.get(actualIndex));

            // If actual != ideal, output MISSED! in the right margin
            if (actualIndex != idealIndex) {
                // add missed to the report body
                report += "MISSED!";
                missed ++;  // increment missed.
            }

            System.out.println(report);
        }
        double rate = (double)(NUM_TESTS-missed) / NUM_TESTS * 100;
        int success = NUM_TESTS - missed;

        System.out.printf("success rate = %d/%d (%3.1f%%)\n",success, NUM_TESTS, rate);

    }

    /**
     * Gets hi-lo range using an elementary form of unsupervised learning.
     * @param list List
     * @return 2-tuple of doubles for low and high range
     */
    protected static double[] getRange(List<Double> list) {
        // Initial low and high values
        double[] range = {Double.MAX_VALUE, -Double.MAX_VALUE};

        // Go through each value in the list
        for(Double value: list) {
            // if value greater than range[HI], update range[HI].
            if (value > range[HI]) {
                range[HI] = value;
            }

            // if value less than range[LO], update range[LO].
            if (value < range[LO]) {
                range[LO] = value;
            }
        }
        return range;
    }

    /**
     * Gets the column-major form normalized data and flips it a row-major form
     * @return all the input data in normalized form as a 2D array
     */
    protected double[][] getInputs() {
        HashMap<String, List<Double>> normals = new HashMap<>();

        // traverse all the headers
        for (String title: Helper.headers) {
            // Get the column data for this title
            List<Double> col = (List<Double>) Helper.data.get(title);

            // Make sure the list contains doubles, not empty and not null
            if(col == null || col.isEmpty() || !(col.get(0) instanceof Double))
                continue;

            // Get range for this column using elementary form of unsupervised learning
            double[] range = getRange(col);

            // NormalizedField instance using the hi-lo range.
            NormalizedField norm = new NormalizedField(NormalizationAction.Normalize,
                    null,range[HI],range[LO],1,-1);

            // List will contain normalized iris data for this column.
            List<Double> normalized = new ArrayList<>();

            // normalize each column element and add it to normalized.
            for (Double element: col) {
                normalized.add(norm.normalize(element));
            }

            // Add normalized data to the normals for this title.
            normals.put(title, normalized);
        }

        // keySet returns an Object array, not a String array
        Object[] keys = normals.keySet().toArray();

        // Transfer header names to String array
        String[] cols = new String[keys.length];
        System.arraycopy(keys, 0, cols, 0, keys.length);

        // copy cols the header report
        System.arraycopy(cols, 0, reportHeader, 0, keys.length);

        // Allocate the 2D storage
        int numRows = Helper.getRowCount();
        int numCols = cols.length;

        double[][] inputs = new double[numRows][numCols];

        // transfer normals to inputs in a row-major form
        for(int row=0; row < numRows; row++) {
            for(int col=0; col < numCols; col++) {
                inputs[row][col] = normals.get(cols[col]).get(row);
            }
        }

        return inputs;
    }

    /**
     * Initializes and Loads the iris data and populate the input
     */
    protected static void init() {
        try {
            Helper.loadCsv("iris.csv", parsers);

            AbuchizIris zIris = new AbuchizIris(); // instance of iris class

            double[][] inputs = zIris.getInputs();

            double[][] ideals= zIris.getIdeals();

            System.out.println("Iris normalized data inputs");
            System.out.println("---------------------------");

            // This is the normalized column header
            for(String key: zIris.reportHeader) {
                System.out.printf("%15s ",key);
            }

            System.out.println();

            // display normalized data content
            for(int row=0; row < Helper.getRowCount(); row++) {
                System.out.printf("%3d ", row);
                for(int col=0; col < 4; col++) {
                    if (col == 0)
                        System.out.printf("%9.2f", inputs[row][col]);
                    else
                        System.out.printf("%16.2f", inputs[row][col]);
                }
                System.out.println();
            }

            System.out.println("Iris encoded data outputs");
            System.out.println("-------------------------");

            // This is the encoded column header
            String idealHeader = String.format("%3s %5s %7s %10s","Index", "y1", "y2", "Decoding");
            System.out.println(idealHeader);

            for (int row=0; row < ideals.length; row++) {
                System.out.printf("%3d", row+1);
                for(int col=0; col < ideals[0].length; col++) {
                    if (col == 0)
                        System.out.printf("%10.4f", ideals[row][col]);
                    else
                        System.out.printf("%8.4f", ideals[row][col]);
                }

                String col = Helper.headers.get(SPECIES);
                String nominal = (String) Helper.data.get(col).get(row);
                System.out.println(" " + nominal);
            }

            int numRows = getTrainingEndIndex() - getTrainingStartIndex() + 1;

            assert(numRows == 120);

            int inputsNumCols = inputs[0].length;

            TRAINING_INPUTS = new double[numRows][inputsNumCols];

            // populate training INPUTS
            for(int row=0; row < numRows; row++) {
                for(int col=0; col < inputsNumCols; col++) {
                    TRAINING_INPUTS[row][col] = inputs[row][col];
                }
            }

            int idealsNumCols = ideals[0].length;
            TRAINING_IDEALS = new double[numRows][idealsNumCols];
            // populate training IDEALS
            for (int row=0; row < numRows; row++) {
                for(int col=0; col < idealsNumCols; col++) {
                    TRAINING_IDEALS[row][col] = ideals[row][col];
                }
            }


            int testRows = getTestingEndIndex() - getTestingStartIndex() + 1;
            // Allocate storage for testing inputs and ideals
            TESTING_INPUTS = new double[testRows][inputsNumCols];
            TESTING_IDEALS = new double[testRows][idealsNumCols];

            // populate testing INPUTS
            for(int row=0; row < testRows; row++) {
                for(int col=0; col < inputsNumCols; col++) {
                    TESTING_INPUTS[row][col] = inputs[row][col];
                }
            }

            // populate testing IDEALS
            for(int row=0; row < testRows; row++) {
                for(int col=0; col < idealsNumCols; col++) {
                    TESTING_IDEALS[row][col] = ideals[row][col];
                }
            }

        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Gets all the ideal output in a 2D array format
     * @return a 2D array of ideal output in doubles
     */
    protected double[][] getIdeals() {
        final int SPECIES = 4;
        subtypes = Helper.getNominalSubtypes(SPECIES);

        // instantiate 2D array to contain the ideals
        double[][] ideals = new double[Helper.getRowCount()][];

        try {
            HashMap<String,Integer> subtypeToNumber = new HashMap<>();
            Integer number = 0;
            for(String subtype: subtypes) {
                subtypeToNumber.put(subtype, number);
                number++;
            }

            // construct the equilateral encoder
            eq = new Equilateral(subtypes.size(), 1.0, -1.0);

            // get species column name
            String col = Helper.headers.get(SPECIES);

            // Convert every species string name in that column to an equilateral encoding
            for(int row=0; row < Helper.getRowCount(); row++) {
                // Get the nominal as a string name
                String nominal = (String) Helper.data.get(col).get(row);
                // Convert the name to a subspecies index number
                number = subtypeToNumber.get(nominal);
                if(number == null)
                    throw new Exception("bad nominal: "+nominal);
                // Encode the number as vertex in n-1 dimensions
                double[] encoding = eq.encode(number);
                // Save the vertex encoding as columns for this row
                ideals[row] = encoding;
            }

        } catch(Exception e) {
            e.printStackTrace();
        }
        return ideals;
    }

    /**
     * Gets the start index for training purpose
     * @return  zero as the start index
     */
    public static final int getTrainingStartIndex() {
        return 0;
    }

    /**
     * Gets the end index for training purpose
     * @return  end index for training
     */
    public static final int getTrainingEndIndex() {
        return (int)(Helper.getRowCount() * TRAINING_PERCENT + 0.50 - 1);
    }

    /**
     * Gets the start index for testing purpose
     * @return  start index for testing
     */
    public static final int getTestingStartIndex() {
        return (int)(Helper.getRowCount() * TRAINING_PERCENT + 0.50);
    }

    /**
     * Gets the end index for testing purpose
     * @return  end index for testing
     */
    public static final int getTestingEndIndex() {
        return Helper.getRowCount() - 1;
    }
}

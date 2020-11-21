/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Project Phase 3
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.metah.ga;

import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.species.BasicSpecies;
import org.encog.ml.ea.species.Species;
import org.encog.ml.ea.train.basic.TrainEA;
import org.encog.ml.genetic.crossover.Splice;
import org.encog.ml.genetic.genome.DoubleArrayGenome;
import org.encog.ml.genetic.genome.DoubleArrayGenomeFactory;

import static javaai.util.Helper.asString;

/**
 * This class uses a genetic algorithm for unsupervised learning to solve y = (x-3)^2.
 */
public class Circuit1Ga {

    /** Stopping criteria as difference between best solution and last best one */
    public final double TOLERANCE = 0.01;

    /** Convergence criteria: number of times best solution stays best */
    public final static int MAX_SAME_COUNT = 100;

    /** Population has this many individuals. */
    public final static int POPULATION_SIZE = 10000;

    /** Chromosome size (ie, number of genes): domain is [0, (2^n-1)]. */
    public final static int GENOME_SIZE = 10;

    /** Mutation rate */
    public final static double MUTATION_RATE = 0.01;

    /** Same count for test of convergence */
    protected int sameCount = 0;

    /** Last y value of training iteration */
    protected double yLast;

    double fitness = 0.0;

    /**
     * Runs the program.
     * @param args Command line arguments not used.
     */
    public static void main(String[] args) {
        Circuit1Ga ga = new Circuit1Ga();

        DoubleArrayGenome best = ga.solve();

        System.out.println("best ="+asString(best) + " fitness = " + ga.fitness);
    }

    /**
     * Solves the objective.
     * @return Best individual
     */
    public DoubleArrayGenome solve() {
        // Initialize a population
        Population population = initPop();
        // output("before", population);

        // instance of Circuit1Objective
        Circuit1Objective objective = new Circuit1Objective();

        // Create the evolutionary training algorithm
        TrainEA ga = new TrainEA(population, objective);

        // Set the mutation rate: 2nd operation tends to give better results.
        // ga.addOperation(MUTATION_RATE, new MutateShuffle());
        ga.addOperation(MUTATION_RATE, new MutateDoubleArrayGenome(0.001));

        // Set up to splice along the middle of the genome
        ga.addOperation(0.9, new Splice(GENOME_SIZE /2));

        // Do the learning algorithm
        train(ga);

        // Return the best individual
        DoubleArrayGenome best = (DoubleArrayGenome)ga.getBestGenome();

        ga.getPopulation();

        // output("after", population);

        // get the fitness value
        fitness = getFitness(best.getData(), objective);


        return best;
    }

    /**
     * Runs the learning algorithm.
     * @param ga
     */
    protected void train(TrainEA ga) {
        int iteration = 0;

        boolean converged = false;

        // display header
        System.out.printf("%3s %8s %5s %5s", "#", "fitness", "same", "best");

        System.out.println();

        // Loop until the best answer doesn't change for a while
        while(!converged) {

            ga.iteration();

            // Get the value of the best solution for predict(x)
            double fitness = ga.getError();

            DoubleArrayGenome best = (DoubleArrayGenome) ga.getBestGenome();

            System.out.printf("%3d %7.4f %4d %82s\n",iteration, fitness, sameCount, asString(best));

            iteration++;

            converged = didConverge(fitness, ga.getPopulation());
        }
    }

    /**
     * Tests whether GA has converged.
     * @param y Y value in y=predict(x)
     * @param pop Population of individuals
     * @return True if the GA has converge, otherwise false
     */
    public boolean didConverge(double y, Population pop) {
        if(sameCount >= MAX_SAME_COUNT)
            return true;

        if(Math.abs(yLast - y) < TOLERANCE) {
            sameCount++;
        }
        else
            sameCount = 0;

        yLast = y;

        return false;
    }

    /**
     * Initializes a population.
     * @return Population
     */
    protected Population initPop() {
        Population pop = new BasicPopulation(POPULATION_SIZE, null);

        BasicSpecies species = new BasicSpecies();

        species.setPopulation(pop);

        for(int k=0; k < POPULATION_SIZE; k++) {
            final DoubleArrayGenome genome = randomGenome(GENOME_SIZE);

            species.getMembers().add(genome);
        }

        pop.setGenomeFactory(new DoubleArrayGenomeFactory(GENOME_SIZE));
        pop.getSpecies().add(species);

        return pop;
    }

    /**
     * Gets a random individual
     * @param sz Number of genes
     * @return
     */
    public DoubleArrayGenome randomGenome(int sz) {
        DoubleArrayGenome genome = new DoubleArrayGenome(sz);

        final double[] organism = genome.getData();

        for(int k=0; k < organism.length; k++) {
            // get a random weight
            organism[k] = Circuit1Objective.getRandomWeight();
        }

        return genome;
    }

    /**
     * Dumps the population.
     * @param title Title
     * @param pop Population
     */
    protected void output(final String title, final Population pop) {
        final Species species = pop.getSpecies().get(0);

        System.out.println("----- "+title);

        int n = 1;

        for (Genome genome : species.getMembers()) {
            DoubleArrayGenome individual = (DoubleArrayGenome) genome;

            System.out.printf("%5d %s\n",n, asString(individual));

            n++;
        }
    }

    /**
     * Calculates the fitness of interneuron weights using the root mean square error
     * @param ws interneuron weights
     * @param objective instance of circuit1 objective
     * @return Fitness value
     */
    public double getFitness(double[] ws, Circuit1Objective objective) {
        // Sum of square error
        double sumSqrErr = 0.0;

        // This is the encoded column header
        String reportHeader = String.format("%3s %4s %4s %4s %4s", "x1", "x2", "x3", "t1", "y1");

        System.out.println(reportHeader);


        // loop through all the XOR_INPUTS
        for(int k=0; k < objective.XOR_INPUTS.length; k++) {
            double x1 = objective.XOR_INPUTS[k][0];
            double x2 = objective.XOR_INPUTS[k][1];
            double x3 = objective.XOR_INPUTS[k][2];

            // calculate actual output
            double y1 = objective.feedforward(x1, x2, x3, ws);

            // ideal output
            double t1 = objective.XOR_IDEALS[k][0];

            // Square error
            double delta = (y1 - t1);
            double sqrError = objective.square(delta);

            // Sum the square error
            sumSqrErr += sqrError;

            System.out.printf("%1.1f %4.1f %4.1f %4.1f %10.6f", x1, x2, x3, t1, y1);
            System.out.println();
        }

        // calculate RMSE
        double rmse = Math.sqrt(sumSqrErr / objective.XOR_INPUTS.length);

        return rmse;
    }
}



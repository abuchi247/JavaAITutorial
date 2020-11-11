/*
 * Author:               Abuchi Obiegbu
 * Assignment:           Lab 2
 * Class:                Artificial Intelligence
 * Copyright (c) 2020 Abuchi Obiegbu
 */
package javaai.aann;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class reads the iris csv file, populates a NNS hashmap for
 * classification purposes and tests classification.
 * @author Abuchi Obiegbu
 * @version 1.0
 */
public class ML2 extends ML1{
    public static void main(String[] args) {
        // NNS hashmap
        NNSMap map = new NNSMap();
        // populate species hashmap
        load(map);

        // list of test cases
        List<Measure> tests =
                new ArrayList<>(Arrays.asList(
                        // failing test case from ML1
                        new Measure(5.1,3.5,1.4,0.2),
                        // Additional edge test cases
                        new Measure(5.1,3.6,1.5,0.4),
                        new Measure(0.0,0.0, 0.0, 0.0),
                        new Measure(100.0,100.0, 100.0, 100.0),
                        // there are no -1.0 position. the data is garbage
                        new Measure(-1.0, -1.0, -1.0, -1.0)));

        // testing classification
        for(Measure measure: tests) {
            System.out.println(map.get(measure)+" "+measure);
        }
    }
}

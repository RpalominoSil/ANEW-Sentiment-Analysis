/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.examples;

import java.io.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;

import clus.algo.kNN.KNNClassifier;
import mulan.ASBCT.*;
import mulan.ASBCT.Attribute;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.ClassifierChain;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.dimensionalityReduction.Ranker;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.experiments.ENTCS13FeatureSelection;
import sun.security.x509.AttributeNameEnumeration;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.J48;
import weka.core.*;
import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.GraphBasedClassifier;
import weka.core.stemmers.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import static mulan.experiments.ENTCS13FeatureSelection.buildMultiLabelFeatureSelection;
import static mulan.experiments.ENTCS13FeatureSelection.featureIndicesByThreshold;
import static mulan.experiments.ENTCS13FeatureSelection.sortedEvaluatedAttributeList;

public class CrossValidationExperiment {
    public static void main(String[] args) {
        String attributesFilepath = StringUtils.ANEW_DICTIONARY_CSV;
        String commentsFilepath = StringUtils.TWITTER_COMMENTS_CSV;

        List<Attribute> attributeList = Reading.readAttributes(attributesFilepath);
        List<Comment> commentList = Reading.readComments(commentsFilepath);

        List<Comment> sentiCommentList = Tagging.tagInstances(commentList,attributeList);
        String arffpath = DatasetGenerator.generateDataset(sentiCommentList,attributeList);
        String xmlpath = DatasetGenerator.generatedatasetHeader();
        Writing.writeResults(sentiCommentList);

        String[] newargs = {"-arff",arffpath,"-xml",xmlpath};

        try {
            String arffFilename = Utils.getOption("arff", newargs);
            String xmlFilename = Utils.getOption("xml", newargs);

            System.out.println("Loading the dataset");
            MultiLabelInstances dataset = new MultiLabelInstances(arffFilename, xmlFilename);

            /*FEATURE SELECTION*/
            String FSmethod = "IG-BR";
            ASEvaluation multiLabelFeatureSelectionMethod = buildMultiLabelFeatureSelection(FSmethod, dataset);
            System.out.println("\nEvaluating features by " + FSmethod + "\n");
            Ranker r = new Ranker();
            r.search((AttributeEvaluator) multiLabelFeatureSelectionMethod, dataset);

            System.out.println("\nOutputting the sorted evaluated attribute list\n");
            double[][] sortedEvaluatedAttributeList = sortedEvaluatedAttributeList(dataset.getFeatureIndices(), (AttributeEvaluator) multiLabelFeatureSelectionMethod);
            int numberFeatures = sortedEvaluatedAttributeList.length;
            for (int i = 0; i < numberFeatures; i++) {
                System.out.println(sortedEvaluatedAttributeList[i][1] + " " + (((int) (sortedEvaluatedAttributeList[i][0])) + 1)); //This line prints the feature index as Weka does: actual feature index + 1
            }

            System.out.println("\nSetting the number of features to be returned from ranking\n");
            int[] featureIndices = featureIndicesByThreshold(0.001, sortedEvaluatedAttributeList); /*Used in ENTCS2013 paper*/
            //int [] featureIndices = featureIndicesByKBest(10, sortedEvaluatedAttributeList); /*Alternative function. See function documentation*/
            //int [] featureIndices = featureIndicesbyTPercent(0.25, sortedEvaluatedAttributeList); /*Alternative function. See function documentation*/

            System.out.println("\nOutputting the indices of the chosen features\n");
            numberFeatures = featureIndices.length;
            for (int i = 0; i < numberFeatures; i++) {
                System.out.println(featureIndices[i] + 6); //This line prints the feature index as Weka does: actual feature index + 1
            }

            /*
             * Feature ranking ends
             */

            dataset = buildReducedMultiLabelDataset(featureIndices, dataset);

            /*PROCESSING*/
            Map<String,DistanceFunction> distanceFunctionMap = new HashMap<>();
            distanceFunctionMap.put("Euclidean",new EuclideanDistance());
            //distanceFunctionMap.put("Manhattan", new ManhattanDistance());
            //distanceFunctionMap.put("Minkowski", new MinkowskiDistance());
            distanceFunctionMap.put("Chebyshev", new ChebyshevDistance());
            distanceFunctionMap.put("Cosine",new CosineDistance());

            int minNumOfNeighbors = 1;
            int maxNumOfNeighbors = 20;
            int minNumFolds = 10;
            int maxNumFolds = 10;
            List<String> classifiers = new ArrayList<>();
            classifiers.add("GeneralB");
            classifiers.add("mLkNN");
            classifiers.add("bRkNN");
            classifiers.add("bRkNN_a");
            classifiers.add("bRkNN_b");
            classifiers.add("mlmut");
            classifiers.add("mlmut_a");
            classifiers.add("mlmut_b");
            classifiers.add("mlnotmut");
            classifiers.add("mlnotmut_a");
            classifiers.add("mlnotmut_b");

            for (int numFolds = minNumFolds; numFolds <= maxNumFolds ; numFolds++) {
                System.out.println("Number of Cross-Validation Folds: " + numFolds);

                for (String distanceName : distanceFunctionMap.keySet()) {
                    System.out.println("Distance Function: " + distanceName);
                    DistanceFunction distanceFunction = distanceFunctionMap.get(distanceName);
                    Vector<Vector<String>> matrix = new Vector<>();
                    for (int numOfNeighbors = minNumOfNeighbors; numOfNeighbors <= maxNumOfNeighbors; numOfNeighbors++) {
                        System.out.println("Number of K Neighbours: " + numOfNeighbors);

                        Vector<String> cvResults = new Vector<>();
                        Evaluator eval = new Evaluator();
                        MultipleEvaluation results;

                        if (classifiers.contains("GeneralB")) {
                            System.out.println("GeneralB");
                            CardinalityBaseline generalB = new CardinalityBaseline();
                            generalB.build(dataset);
                            results = eval.crossValidate(generalB, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mLkNN")) {
                            System.out.println("mLkNN");
                            MLkNN mLkNN = new MLkNN(numOfNeighbors, 1.0);
                            mLkNN.setDfunc(distanceFunction);
                            mLkNN.build(dataset);
                            results = eval.crossValidate(mLkNN, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("bRkNN")) {
                            System.out.println("bRkNN");
                            BRkNN bRkNN = new BRkNN(numOfNeighbors, BRkNN.ExtensionType.NONE); bRkNN.setDfunc(distanceFunction);
                            bRkNN.build(dataset);
                            results = eval.crossValidate(bRkNN, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("bRkNN_a")) {
                            System.out.println("bRkNN_a");
                            BRkNN bRkNN_a = new BRkNN(numOfNeighbors, BRkNN.ExtensionType.EXTA); bRkNN_a.setDfunc(distanceFunction);
                            bRkNN_a.build(dataset);
                            results = eval.crossValidate(bRkNN_a, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("bRkNN_b")) {
                            System.out.println("bRkNN_b");
                            BRkNN bRkNN_b = new BRkNN(numOfNeighbors, BRkNN.ExtensionType.EXTB); bRkNN_b.setDfunc(distanceFunction);
                            bRkNN_b.build(dataset);
                            results = eval.crossValidate(bRkNN_b, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mlmut")) {
                            System.out.println("mlmut");
                            GraphBasedClassifier mlmut = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.NONE, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL); mlmut.setDfunc(distanceFunction);
                            mlmut.build(dataset);
                            results = eval.crossValidate(mlmut, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mlmut_a")) {
                            System.out.println("mlmut_a");
                            GraphBasedClassifier mlmut_a = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTA, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL); mlmut_a.setDfunc(distanceFunction);
                            mlmut_a.build(dataset);
                            /*
                            //EMPIEZA
                            String unlabeledDataFilename = Utils.getOption("unlabeled", args);
                            //String unlabeledDataFilename = StringUtils.EXP_TEST_CSV;
                            MultiLabelInstances unlabeledData = new MultiLabelInstances(unlabeledDataFilename, xmlFilename);
                            int numInstances = unlabeledData.getNumInstances();
                            for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
                                Instance instance = unlabeledData.getDataSet().instance(instanceIndex);
                                MultiLabelOutput output = mlmut_a.makePrediction(instance);
                                if (output.hasBipartition()) {
                                    String bipartion = Arrays.toString(output.getBipartition());
                                    System.out.println("Predicted bipartion: " + bipartion);
                                }
                                System.out.println(output);
                            }
                            */
                            results = eval.crossValidate(mlmut_a, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mlmut_b")) {
                            System.out.println("mlmut_b");
                            GraphBasedClassifier mlmut_b = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTB, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL); mlmut_b.setDfunc(distanceFunction);
                            mlmut_b.build(dataset);
                            results = eval.crossValidate(mlmut_b, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mlnotmut")) {
                            System.out.println("mlnotmut");
                            GraphBasedClassifier mlnotmut = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.NONE, GraphBasedClassifier.GraphType.NEIGHBOR_NOT_MUTUAL); mlnotmut.setDfunc(distanceFunction);
                            mlnotmut.build(dataset);
                            results = eval.crossValidate(mlnotmut, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mlnotmut_a")) {
                            System.out.println("mlnotmut_a");
                            GraphBasedClassifier mlnotmut_a = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTA, GraphBasedClassifier.GraphType.NEIGHBOR_NOT_MUTUAL); mlnotmut_a.setDfunc(distanceFunction);
                            mlnotmut_a.build(dataset);
                            results = eval.crossValidate(mlnotmut_a, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        if (classifiers.contains("mlnotmut_b")) {
                            System.out.println("mlnotmut_b");
                            GraphBasedClassifier mlnotmut_b = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTB, GraphBasedClassifier.GraphType.NEIGHBOR_NOT_MUTUAL); mlnotmut_b.setDfunc(distanceFunction);
                            mlnotmut_b.build(dataset);
                            results = eval.crossValidate(mlnotmut_b, dataset, numFolds);
                            cvResults.add(results.getHammingLoss());
                        }
                        matrix.add(cvResults);
                    }
                    //String filename = "data\\MEDICAL\\EXPMED\\CV" + numFolds + "\\CV" + numFolds + distanceName + ".csv";
                    String filename = StringUtils.EXP_DIR + "CV" + numFolds + "\\CV" + numFolds + distanceName + ".csv";
                    File file = new File(filename);
                    try {
                        Writer sb = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "ISO-8859-1"));

                        /*for (Vector<String> results : matrix) {
                            for (String result : results) {
                                sb.append(result);
                                sb.append(";");
                            }
                            sb.append('\n');
                        }*/

                        sb.append("");
                        sb.append(";");
                        for(int k = minNumOfNeighbors-1; k < maxNumOfNeighbors; k++) {
                            sb.append("k=" + String.valueOf(k + 1));
                            sb.append(";");
                        }
                        sb.append('\n');
                        for (int k = 0; k < classifiers.size() ; k++) {
                            sb.append(classifiers.get(k));
                            sb.append(";");
                            for (Vector<String> results : matrix) {
                                sb.append(results.get(k));
                                sb.append(";");
                            }
                            sb.append('\n');
                        }

                        sb.flush();
                        sb.close();
                        System.out.println(filename);
                    }  catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (UnsupportedEncodingException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }

            //ClassifierChain classifierChain = new ClassifierChain(new J48());

            /*BRkNN bRkNN = new BRkNN(numOfNeighbors, BRkNN.ExtensionType.NONE);
            BRkNN bRkNN_a = new BRkNN(numOfNeighbors, BRkNN.ExtensionType.EXTA);
            BRkNN bRkNN_b = new BRkNN(numOfNeighbors, BRkNN.ExtensionType.EXTB);
            //GraphBasedClassifier mlnone_n = new GraphBasedClassifier(14, GraphBasedClassifier.ExtensionType.NONE, GraphBasedClassifier.GraphType.NONE);
            //GraphBasedClassifier mlnone_a = new GraphBasedClassifier(14, GraphBasedClassifier.ExtensionType.EXTA, GraphBasedClassifier.GraphType.NONE);
            //GraphBasedClassifier mlnone_b = new GraphBasedClassifier(14, GraphBasedClassifier.ExtensionType.EXTB, GraphBasedClassifier.GraphType.NONE);
            GraphBasedClassifier mlmut_n = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.NONE, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL);
            GraphBasedClassifier mlmut_a = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTA, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL);
            GraphBasedClassifier mlmut_b = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTB, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL);
            GraphBasedClassifier mlnotmut_n = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.NONE, GraphBasedClassifier.GraphType.NEIGHBOR_NOT_MUTUAL);
            GraphBasedClassifier mlnotmut_a = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTA, GraphBasedClassifier.GraphType.NEIGHBOR_NOT_MUTUAL);
            GraphBasedClassifier mlnotmut_b = new GraphBasedClassifier(numOfNeighbors, GraphBasedClassifier.ExtensionType.EXTB, GraphBasedClassifier.GraphType.NEIGHBOR_NOT_MUTUAL);
*/
            /*results = eval.crossValidate(bRkNN, dataset, numFolds);
            System.out.println("BRkNN: " );
            System.out.println(results);

            results = eval.crossValidate(bRkNN_a, dataset, numFolds);
            System.out.println("BRkNN-a: " );
            System.out.println(results);

            results = eval.crossValidate(bRkNN_b, dataset, numFolds);
            System.out.println("BRkNN-b: " );
            System.out.println(results);

            /*results = eval.crossValidate(mlnone_n, dataset, numFolds);
            System.out.println("ML: " );
            System.out.println(results);*/

            /*results = eval.crossValidate(mlnone_a, dataset, numFolds);
            System.out.println("ML-a: " );
            System.out.println(results);*/

            /*results = eval.crossValidate(mlnone_b, dataset, numFolds);
            System.out.println("ML-b: " );
            System.out.println(results);*/

            /*results = eval.crossValidate(mlmut_n, dataset, numFolds);
            System.out.println("MLMUT: " );
            System.out.println(results);

            results = eval.crossValidate(mlmut_a, dataset, numFolds);
            System.out.println("MLMUT-a: " );
            System.out.println(results);

            results = eval.crossValidate(mlmut_b, dataset, numFolds);
            System.out.println("MLMUT-b: " );
            System.out.println(results);

            results = eval.crossValidate(mlnotmut_n, dataset, numFolds);
            System.out.println("MLnotMUT: " );
            System.out.println(results);

            results = eval.crossValidate(mlnotmut_a, dataset, numFolds);
            System.out.println("MLnotMUT-a: " );
            System.out.println(results);

            results = eval.crossValidate(mlnotmut_b, dataset, numFolds);
            System.out.println("MLnotMUT-b: " );
            System.out.println(results);*/

        } catch (InvalidDataFormatException ex) {
            Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    private static MultiLabelInstances buildReducedMultiLabelDataset(int[] featureIndices, MultiLabelInstances dataSet) {
        //System.out.println("\nBuilding the reduced dataset from the chosen features\n");
        int[] toKeep = new int[featureIndices.length + dataSet.getNumLabels()];
        System.arraycopy(featureIndices, 0, toKeep, 0, featureIndices.length);
        int[] labelIndices = dataSet.getLabelIndices();
        System.arraycopy(labelIndices, 0, toKeep, featureIndices.length, dataSet.getNumLabels());

        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(toKeep);
        filterRemove.setInvertSelection(true);
        MultiLabelInstances mlFiltered = dataSet;
        try {
            filterRemove.setInputFormat(dataSet.getDataSet());
            Instances filtered = Filter.useFilter(dataSet.getDataSet(), filterRemove);
            mlFiltered = new MultiLabelInstances(filtered, dataSet.getLabelsMetaData());

            // You can now work on the reduced multi-label dataset mlFiltered
        } catch (Exception ex) {
            Logger.getLogger(ENTCS13FeatureSelection.class.getName()).log(Level.SEVERE, null, ex);
        }
        return mlFiltered;
    }
}
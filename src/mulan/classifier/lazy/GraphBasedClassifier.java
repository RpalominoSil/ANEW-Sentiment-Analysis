package mulan.classifier.lazy;

import mulan.classifier.MultiLabelOutput;
import mulan.core.Util;
import mulan.data.MultiLabelInstances;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

@SuppressWarnings("serial")
public class GraphBasedClassifier extends MultiLabelKNN {

    Random random;
    /**
     * Stores the average number of labels among the knn for each instance Used
     * in BRkNN-b extension
     */
    int avgPredictedLabels;
    /**
     * The value of kNN provided by the user. This may differ from
     * numOfNeighbors if cross-validation is being used.
     */
    private int cvMaxK;
    /**
     * Whether to select k by cross validation.
     */
    private boolean cvkSelection = false;

    /**
     * The two types of extensions
     */
    public enum ExtensionType {

        /**
         * Standard BR
         */
        NONE,
        /**
         * Predict top ranked label in case of empty prediction set
         */
        EXTA,
        /**
         * Predict top n ranked labels based on size of labelset in neighbors
         */
        EXTB
    };
    /**
     * The type of extension to be used
     */
    private ExtensionType extension = ExtensionType.NONE;


    public enum GraphType {

        /**
         * Standard KNN
         */
        NONE,

        /**
         * Examples are neighbors when the distance among them are lower than the user-defined threshold
         */
        THRESHOLD,

        /**
         * Two examples are neighbors when both consider each other as neighbors
         */
        NEIGHBOR_MUTUAL,

        /**
         * Two examples are neighbors when one of them considers the other as neighbor
         */
        NEIGHBOR_NOT_MUTUAL
    };
    private GraphType graphtype = GraphType.NEIGHBOR_MUTUAL;


    /**
     * Threshold value used for THRESHOLD GraphType strategy
     */
    private double e_threshold;

    /**
     * The default constructor
     *
     * @param numOfNeighbors
     */
    public GraphBasedClassifier(int numOfNeighbors) {
        this(numOfNeighbors, ExtensionType.EXTB, GraphType.NEIGHBOR_MUTUAL);
    }

    /**
     * Constructor giving the option to select an extension of the base version
     *
     * @param numOfNeighbors
     * @param ext the extension to use (see {@link ExtensionType})
     *
     */
    public GraphBasedClassifier(int numOfNeighbors, ExtensionType ext, GraphType graph) {
        super(numOfNeighbors);
        random = new Random(1);
        extension = ext;
        graphtype = graph;
        distanceWeighting = WEIGHT_NONE; // weight none
    }

    public GraphBasedClassifier(double threshold, ExtensionType ext, GraphType graph) {
        super(1);
        random = new Random(1);
        extension = ext;
        graphtype = graph;
        distanceWeighting = WEIGHT_NONE; // weight none
        e_threshold = threshold;

    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(
                Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR,
                "Everton Alvares Cherman, Newton Spolaôr, Jorge Carlos Valverde-Rebaza, Maria Carolina Monard");
        result.setValue(Field.TITLE,
                "Algoritmos de Aprendizado Baseado em Grafos para Classificação Multirrótulo");
        result.setValue(Field.BOOKTITLE,
                "Anais do X Encontro Nacional de Inteligência Artificial e Computacional");
        result.setValue(Field.LOCATION, "Fortaleza, Brasil");
        result.setValue(Field.YEAR, "2013");
        return result;
    }

    double[][] matrix = null;

    @Override
    protected void buildInternal(MultiLabelInstances aTrain) throws Exception {
        super.buildInternal(aTrain);
        matrix = calculateDistanceMatrix(train);

        if (cvkSelection == true) {
            crossValidate();
        }
    }

    public String getSomething(String name)
    {
        int knn[][];
        StringBuilder aux = new StringBuilder();

        switch (graphtype) {
            case THRESHOLD:

                for (int i=0; i<matrix.length-1; i++)
                {
                    for (int j=i+1; j<matrix.length; j++ )
                    {
                        if (matrix[i][j] <= e_threshold)
                        {
                            aux.append("node");
                            aux.append(i);
                            aux.append(";");
                            aux.append("node");
                            aux.append(j);
                            aux.append(";");
                            aux.append(matrix[i][j]);
                            aux.append("\n");
                        }
                    }
                }

                break;

            case NEIGHBOR_MUTUAL:

                knn= sortMatrix(matrix);

                for (int i=0; i<knn.length; i++)
                {
                    int[] knnaux = Arrays.copyOfRange(knn[i],0, numOfNeighbors);

                    for (int j=0; j<knnaux.length; j++)
                    {
                        if (knnaux[j] > i){
                            int[] knnaux2 = Arrays.copyOfRange(knn[knnaux[j]],0, numOfNeighbors);


                            if (contains(knnaux2,i))
                            {
                                aux.append("node");
                                aux.append(i);
                                aux.append(";");
                                aux.append("node");
                                aux.append(knnaux[j]);
                                aux.append(";");
                                aux.append(matrix[i][knnaux[j]]);
                                aux.append("\n");
                            }
                        }
                    }
                }

                break;

            case NEIGHBOR_NOT_MUTUAL:
                knn = sortMatrix(matrix);

                for (int i=0; i<knn.length; i++)
                {
                    int[] knnaux = Arrays.copyOfRange(knn[i],0, numOfNeighbors);

                    for (int j=0; j<knnaux.length; j++)
                    {
                        if (knnaux[j] > i || (!contains(Arrays.copyOfRange(knn[knnaux[j]],0, numOfNeighbors),i)))
                        {
                            aux.append("node");
                            aux.append(i);
                            aux.append(";");
                            aux.append("node");
                            aux.append(knnaux[j]);
                            aux.append(";");
                            aux.append(matrix[i][knnaux[j]]);
                            aux.append("\n");
                        }
                    }
                }

                break;

        }


        return name + "\n" + aux.toString();
    }

    /**
     *
     * @param flag
     *            if true the k is selected via cross-validation
     */
    public void setkSelectionViaCV(boolean flag) {
        cvkSelection = flag;
    }

    /**
     * Select the best value for k by hold-one-out cross-validation. Hamming
     * Loss is minimized
     *
     * @throws Exception
     */
    protected void crossValidate() throws Exception {
        try {
            // the performance for each different k
            double[] hammingLoss = new double[cvMaxK];

            for (int i = 0; i < cvMaxK; i++) {
                hammingLoss[i] = 0;
            }

            Instances dataSet = train;
            Instance instance; // the hold out instance
            Instances neighbours; // the neighboring instances
            double[] origDistances, convertedDistances;
            for (int i = 0; i < dataSet.numInstances(); i++) {
                if (getDebug() && (i % 50 == 0)) {
                    debug("Cross validating " + i + "/" + dataSet.numInstances() + "\r");
                }
                instance = dataSet.instance(i);
                neighbours = lnn.kNearestNeighbours(instance, cvMaxK);
                origDistances = lnn.getDistances();

                // gathering the true labels for the instance
                boolean[] trueLabels = new boolean[numLabels];
                for (int counter = 0; counter < numLabels; counter++) {
                    int classIdx = labelIndices[counter];
                    String classValue = instance.attribute(classIdx).value(
                            (int) instance.value(classIdx));
                    trueLabels[counter] = classValue.equals("1");
                }
                // calculate the performance metric for each different k
                for (int j = cvMaxK; j > 0; j--) {
                    convertedDistances = new double[origDistances.length];
                    System.arraycopy(origDistances, 0, convertedDistances, 0,
                            origDistances.length);
                    double[] confidences = this.getConfidences(neighbours,
                            convertedDistances);
                    boolean[] bipartition = null;

                    switch (extension) {
                        case NONE:
                            MultiLabelOutput results;
                            results = new MultiLabelOutput(confidences, 0.5);
                            bipartition = results.getBipartition();
                            break;
                        case EXTA:
                            bipartition = labelsFromConfidences2(confidences);
                            break;
                        case EXTB:
                            bipartition = labelsFromConfidences3(confidences);
                            break;
                    }

                    double symmetricDifference = 0; // |Y xor Z|
                    for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
                        boolean actual = trueLabels[labelIndex];
                        boolean predicted = bipartition[labelIndex];

                        if (predicted != actual) {
                            symmetricDifference++;
                        }
                    }
                    hammingLoss[j - 1] += (symmetricDifference / numLabels);

                    neighbours = new IBk().pruneToK(neighbours,
                            convertedDistances, j - 1);
                }
            }

            // Display the results of the cross-validation
            if (getDebug()) {
                for (int i = cvMaxK; i > 0; i--) {
                    debug("Hold-one-out performance of " + (i) + " neighbors ");
                    debug("(Hamming Loss) = " + hammingLoss[i - 1] / dataSet.numInstances());
                }
            }

            // Check through the performance stats and select the best
            // k value (or the lowest k if more than one best)
            double[] searchStats = hammingLoss;

            double bestPerformance = Double.NaN;
            int bestK = 1;
            for (int i = 0; i < cvMaxK; i++) {
                if (Double.isNaN(bestPerformance) || (bestPerformance > searchStats[i])) {
                    bestPerformance = searchStats[i];
                    bestK = i + 1;
                }
            }
            numOfNeighbors = bestK;
            if (getDebug()) {
                System.err.println("Selected k = " + bestK);
            }

        } catch (Exception ex) {
            throw new Error("Couldn't optimize by cross-validation: " + ex.getMessage());
        }
    }

    double max = 0;
    protected double[][] calculateDistanceMatrix(Instances train)
    {
        double[][] mdist = new double[train.numInstances()][train.numInstances()];

        int numInstances = train.numInstances();
        for (int i=0; i<numInstances-1; i++)
        {
            mdist[i][i] = 0;
            for (int j=i+1; j<numInstances; j++)
            {
                Instance inst1 = train.instance(i);
                Instance inst2 = train.instance(j);

                mdist[i][j] = mdist[j][i] = dfunc.distance(inst1,inst2);

                if (max < mdist[i][j])
                    max = mdist[i][j];

            }
        }

        for (int i=0; i<numInstances-1; i++)
        {
            mdist[i][i] = Double.MAX_VALUE;
            for (int j=i+1; j<numInstances; j++)
            {
                mdist[i][j] = mdist[j][i] = mdist[i][j]/max;
            }
        }
        mdist[numInstances-1][numInstances-1] = Double.MAX_VALUE;

        return mdist;
    }


    protected double[][] updateDistanceMatrix(double[][] mdist, Instances train, Instance test)
    {
        int numInstances = train.numInstances();
        double [][] mdistnew = new double[numInstances+1][numInstances+1];

        // copy vector

        for (int i=0; i<numInstances;i++)
            for (int j=i; j<numInstances; j++)
                mdistnew[i][j] = mdistnew[j][i] = mdist[i][j];

        for (int i=0; i<numInstances; i++){
            Instance inst = train.instance(i);
            mdistnew[i][numInstances] = mdistnew[numInstances][i] = dfunc.distance(inst, test)/max;
        }
        mdistnew[numInstances][numInstances] = Double.MAX_VALUE;

        return mdistnew;
    }




    private int[] getKnnIndexes(double[][] distanceMatrix, int index,
                                int numOfNeighbors) {

        double[] line = distanceMatrix[index];
        int[] indexs = weka.core.Utils.sort(line);

        return Arrays.copyOfRange(indexs, 0, numOfNeighbors);
    }

    private int[][] sortMatrix(double[][] distanceMatrix) {

        int n = distanceMatrix.length;
        int[][] knn = new int[n][n];

        for (int i=0; i <n; i++)
        {
            double[] line = distanceMatrix[i];
            knn[i] = weka.core.Utils.sort(line);
        }
        return knn;
    }


    private boolean contains(int[] array, int key)
    {
        for (int a: array)
        {
            if (a == key)
                return true;
        }
        return false;
    }

    /**
     * weka Ibk style prediction
     *
     * @throws Exception if nearest neighbours search fails
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        Instances  knn = new Instances(train,0);
        double[][] matrixcopy = matrix.clone();

        ArrayList<Double> vConfidences = new ArrayList<Double>();
        double[] distances = new double[0];
        int[]  knnindexs;

        switch (graphtype) {
            case THRESHOLD:

                matrixcopy = updateDistanceMatrix(matrixcopy, train, instance);
                for (int i=0; i<train.numInstances(); i++)
                {
                    double dist = matrixcopy[i][matrix.length];
                    if (dist <= e_threshold){
                        knn.add(train.instance(i));
                        vConfidences.add(dist);
                    }
                }

                // convert ArrayList<Double> to double[]
                distances = new double[vConfidences.size()];
                for (int i =0; i<vConfidences.size(); i++)
                {
                    distances[i] = vConfidences.get(i);
                }
                break;

            case NEIGHBOR_MUTUAL:

                matrixcopy = updateDistanceMatrix(matrixcopy, train, instance);
                knnindexs = getKnnIndexes(matrixcopy,matrix.length,numOfNeighbors);

                for (int i : knnindexs)
                {
                    int[] knnindexs2 = getKnnIndexes(matrixcopy,i,numOfNeighbors);

                    if (contains(knnindexs2, matrix.length)){
                        knn.add(train.instance(i));
                        vConfidences.add(dfunc.distance(train.instance(i),instance));
                    }
                }
                // convert ArrayList<Double> to double[]
                distances = new double[vConfidences.size()];
                for (int i =0; i<vConfidences.size(); i++)
                {
                    distances[i] = vConfidences.get(i);
                }

                break;

            case NEIGHBOR_NOT_MUTUAL:

                matrixcopy = updateDistanceMatrix(matrixcopy, train, instance);
                knnindexs  = getKnnIndexes(matrixcopy,matrix.length,numOfNeighbors);

                for (int i: knnindexs)
                {
                    knn.add(train.instance(i));
                    vConfidences.add(matrixcopy[i][matrix.length]);
                }

                for (int i = 0; i < matrix.length; i++) {

                    if (!contains(knnindexs, i)){
                        int[] knnindexs2 = getKnnIndexes(matrixcopy, i, numOfNeighbors);

                        if (contains(knnindexs2, matrix.length)){
                            knn.add(train.instance(i));
                            vConfidences.add(matrixcopy[i][matrix.length]);
                        }

                    }
                }

                // convert ArrayList<Double> to double[]
                distances = new double[vConfidences.size()];
                for (int i =0; i<vConfidences.size(); i++)
                {
                    distances[i] = vConfidences.get(i);
                }

                break;

            case NONE:
                knn = lnn.kNearestNeighbours(instance, numOfNeighbors);
                distances = lnn.getDistances();
                break;
        }

        // if there isn't any selected neighbour
        if (distances.length <= 0)
        {
            knn = lnn.kNearestNeighbours(instance, 1);
            distances = lnn.getDistances();
        }



        double[] confidences = getConfidences(knn, distances);
        boolean[] bipartition;

        MultiLabelOutput results = null;
        switch (extension) {
            case NONE: // BRknn
                results = new MultiLabelOutput(confidences, 0.5);
                break;
            case EXTA: // BRknn-a
                bipartition = labelsFromConfidences2(confidences);
                results = new MultiLabelOutput(bipartition, confidences);
                break;
            case EXTB: // BRknn-b
                bipartition = labelsFromConfidences3(confidences);
                results = new MultiLabelOutput(bipartition, confidences);
                break;
        }
        return results;

    }

    /**
     * Calculates the confidences of the labels, based on the neighboring
     * instances
     *
     * @param neighbours
     *            the list of nearest neighboring instances
     * @param distances
     *            the distances of the neighbors
     * @return the confidences of the labels
     */
    private double[] getConfidences(Instances neighbours, double[] distances) {
        double total = 0, weight;
        double neighborLabels = 0;
        double[] confidences = new double[numLabels];

        // Set up a correction to the estimator
        for (int i = 0; i < numLabels; i++) {
            confidences[i] = 1.0 / Math.max(1, train.numInstances());
        }
        total = (double) numLabels / Math.max(1, train.numInstances());

        for (int i = 0; i < neighbours.numInstances(); i++) {
            // Collect class counts
            Instance current = neighbours.instance(i);
            distances[i] = distances[i] * distances[i];
            distances[i] = Math.sqrt(distances[i] / (train.numAttributes() - numLabels));
            switch (distanceWeighting) {
                case WEIGHT_INVERSE:
                    weight = 1.0 / (distances[i] + 0.001); // to avoid division by
                    // zero
                    break;
                case WEIGHT_SIMILARITY:
                    weight = 1.0 - distances[i];
                    break;
                default: // WEIGHT_NONE:
                    weight = 1.0;
                    break;
            }
            weight *= current.weight();

            for (int j = 0; j < numLabels; j++) {
                double value = Double.parseDouble(current.attribute(
                        labelIndices[j]).value(
                        (int) current.value(labelIndices[j])));
                if (Utils.eq(value, 1.0)) {
                    confidences[j] += weight;
                    neighborLabels += weight;
                }
            }
            total += weight;
        }

        avgPredictedLabels = (int) Math.round(neighborLabels / total);
        // Normalise distribution
        if (total > 0) {
            Utils.normalize(confidences, total);
        }
        return confidences;
    }

    /**
     * used for BRknn-a
     *
     * @param confidences the probabilities for each label
     * @return a bipartition
     */
    protected boolean[] labelsFromConfidences2(double[] confidences) {
        boolean[] bipartition = new boolean[numLabels];
        boolean flag = false; // check the case that no label is true

        for (int i = 0; i < numLabels; i++) {
            if (confidences[i] >= 0.5) {
                bipartition[i] = true;
                flag = true;
            }
        }
        // assign the class with the greater confidence
        if (flag == false) {
            int index = Util.RandomIndexOfMax(confidences, random);
            bipartition[index] = true;
        }
        return bipartition;
    }

    /**
     * used for BRkNN-b (break ties arbitrarily)
     *
     * @param confidences the probabilities for each label
     * @return a bipartition
     */
    protected boolean[] labelsFromConfidences3(double[] confidences) {
        boolean[] bipartition = new boolean[numLabels];

        int[] indices = Utils.stableSort(confidences);

        ArrayList<Integer> lastindices = new ArrayList<Integer>();

        int counter = 0;
        int i = numLabels - 1;

        while (i >= 0) {
            if (confidences[indices[i]] > confidences[indices[numLabels - avgPredictedLabels]]) {
                bipartition[indices[i]] = true;
                counter++;
            } else if (confidences[indices[i]] == confidences[indices[numLabels - avgPredictedLabels]]) {
                lastindices.add(indices[i]);
            } else {
                break;
            }
            i--;
        }

        int size = lastindices.size();

        int j = avgPredictedLabels - counter;
        while (j > 0) {
            int next = random.nextInt(size);
            if (bipartition[lastindices.get(next)] != true) {
                bipartition[lastindices.get(next)] = true;
                j--;
            }
        }

        return bipartition;
    }

    /**
     * set the maximum number of neighbors to be evaluated via cross-validation
     *
     * @param cvMaxK
     */
    public void setCvMaxK(int cvMaxK) {
        this.cvMaxK = cvMaxK;
    }

    //@Override
    public String globalInfo() {
        // TODO Auto-generated method stub
        return null;
    }

}



package mulan.ASBCT;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.GraphBasedClassifier;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.dimensionalityReduction.Ranker;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.examples.CrossValidationExperiment;
import mulan.examples.GettingPredictionsOnUnlabeledData;
import mulan.experiments.ENTCS13FeatureSelection;
import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;
import twitter4j.*;
import twitter4j.conf.ConfigurationBuilder;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.sql.*;
import java.sql.Connection;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static mulan.experiments.ENTCS13FeatureSelection.buildMultiLabelFeatureSelection;
import static mulan.experiments.ENTCS13FeatureSelection.featureIndicesByThreshold;
import static mulan.experiments.ENTCS13FeatureSelection.sortedEvaluatedAttributeList;

public class TwitterData {

    private static String oAuthConsumerKey="PjZvyDOi75monKa97Dn3lImeB";
    private static String oAuthConsumerSecret="HXD3Cjksdo8F4BJq2OmSVhlO1r8xQNjKfIL28AAFi09H1MP9Nb";
    private static String oAuthAccessToken="1067466193-Dz65p3lM2fiOOZZaJy8WvTZuEURGwbdm1niUNo1";
    private static String oAuthAccessTokenSecret= "uf1bdx1LYJl1vcHuuuKbhZ2MG1MP56H9iVX3Y5ri3idNF";
    private static String URL="jdbc:postgresql://postgres:5432/tweeter1";
    private static String USER="postgres";
    private static String PASS="root";

    static List<String> tokenlist = new ArrayList<>();
    static List<String> TweetsCrudo = new ArrayList<>();
    static List<String> TweetsLemma = new ArrayList<>();

    public static void request(){
        String key = "fdbd9f18e0abe53892ab86bb58a53d3d";
        String lang = "es";
        //String txt = "Descubrí mi blog de filosofía y pensamiento contemporáneo, Anuncios de cursos y talleres, Entrevistas. Escritos sob… https://t.co/yIg7QuNWxz\n";

        Integer count = 0;
        String line = "";
        try {
            while ((line = TweetsCrudo.get(0)) != null) {
                TweetsCrudo.remove(0);
                tokenlist = new ArrayList<>();
                innerrequest(key,lang,line);
                Thread.sleep(500);
                String sentence = new String();
                for (String token: tokenlist) {
                    sentence=sentence.concat(token + " ");
                }
                TweetsLemma.add(sentence);
            }
        } catch (Exception ex) {
            System.out.println(ex);
        }
    }
    private static void innerrequest(String key, String lang, String txt) {
        OkHttpClient client = new OkHttpClient();
        //MediaType mediaType = MediaType.parse("application/x-www-form-urlencoded");

        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("key", key)
                .addFormDataPart("lang",lang)
                .addFormDataPart("txt",txt)
                //.addFormDataPart("doc",doc.getName(),RequestBody.create(MediaType.parse("text/csv"),ous.toByteArray()))
                .build();
        Request request = new Request.Builder()
                .url("http://api.meaningcloud.com/parser-2.0")
                .post(body)
                .addHeader("content-type", "application/x-www-form-urlencoded")
                .build();
        try {
            Response response = client.newCall(request).execute();
            JSONObject jsonObj1 = new JSONObject(response.body().string());
            //System.out.println(jsonObj1.toString());
            JSONArray sentencesPrev = jsonObj1.getJSONArray("token_list");
            getToken(sentencesPrev);
            System.out.println(response.body().string());
            //System.out.println(tokens.toString());
        } catch (Exception e) {
            //System.out.println(e);
        }
    }
    private static void getToken(JSONArray sentence){
        for (int i=0;i<sentence.length();i++) {
            try {
                JSONObject jsonObj = sentence.getJSONObject(i);
                JSONArray phrase = jsonObj.getJSONArray("token_list");
                //System.out.println(phrase.toString());
                getToken(phrase);
            } catch (Exception e) {
                try {
                    JSONObject jsonObj = sentence.getJSONObject(i);
                    JSONArray analisislist = jsonObj.getJSONArray("analysis_list");
                    JSONObject analisis = analisislist.getJSONObject(0);
                    Object tag = analisis.get("tag");
                    if (!tag.toString().equals("1D--")) {
                        Object word = analisis.get("lemma");
                        tokenlist.add(word.toString());
                        //System.out.println(word.toString());
                    }
                }catch (Exception ex) {
                    try {
                        JSONObject jsonObj = sentence.getJSONObject(i);
                        Object word = jsonObj.get("form");
                        tokenlist.add(word.toString());
                        //System.out.println(word.toString());
                    } catch (Exception exc) {
                        System.out.println(exc);
                    }
                }
            }
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

    public static boolean ASBCT_MLmut_a_prediction(int num){
        boolean valor = false;
        try {
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

                System.out.println("Loading the dataset...");
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

                System.out.println("\nOutputting the indices of the chosen features\n");
                numberFeatures = featureIndices.length;
                for (int i = 0; i < numberFeatures; i++) {
                    System.out.println(featureIndices[i] + 1); //This line prints the feature index as Weka does: actual feature index + 1
                }

                dataset = buildReducedMultiLabelDataset(featureIndices, dataset);

                MultipleEvaluation results;
                Evaluator eval = new Evaluator();
                DistanceFunction distanceFunction = new EuclideanDistance();
                GraphBasedClassifier mlmut_a = new GraphBasedClassifier(10, GraphBasedClassifier.ExtensionType.EXTA, GraphBasedClassifier.GraphType.NEIGHBOR_MUTUAL); mlmut_a.setDfunc(distanceFunction);
                mlmut_a.build(dataset);
                mlmut_a.setCvMaxK(10);
                mlmut_a.setDfunc(distanceFunction);
                results = eval.crossValidate(mlmut_a, dataset, 10);

                String unlabeledDataFilename = Utils.getOption("unlabeled", newargs);
                MultiLabelInstances unlabeledData = new MultiLabelInstances(unlabeledDataFilename, xmlFilename);

                int numInstances = unlabeledData.getNumInstances();
                for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
                    Instance instance = unlabeledData.getDataSet().instance(instanceIndex);
                    MultiLabelOutput output = mlmut_a.makePrediction(instance);
                    if (output.hasBipartition()) {
                        String bipartion = Arrays.toString(output.getBipartition());
                        System.out.println("Predicted bipartion: " + bipartion);
                    }
                    switch (num)
                    {
                        case 1: valor=output.equals("agrado")? true:false;
                        break;
                        case 2: valor=output.equals("desagrado")? true:false;
                            break;
                        case 3: valor=output.equals("emocion")? true:false;
                            break;
                        case 4: valor=output.equals("calma")? true:false;
                            break;
                        case 5: valor=output.equals("control")? true:false;
                            break;
                        case 6: valor=output.equals("descontrol")? true:false;
                            break;
                    }
                }
            } catch (InvalidDataFormatException e) {
                System.err.println(e.getMessage());
            } catch (Exception ex) {
                Logger.getLogger(GettingPredictionsOnUnlabeledData.class.getName()).log(Level.SEVERE, null, ex);
            }
        } catch (Exception ex) {
            Logger.getLogger(CrossValidationExperiment.class.getName()).log(Level.SEVERE, null, ex);
        }
        return valor;
    }
    public static void conexion(String tweet) throws SQLException, ClassNotFoundException{
        Class.forName("org.postgresql.Driver");
        Connection con= DriverManager.getConnection(URL, USER, PASS);
        if(con!=null){
            System.out.println("conexion realizada correctamente");
            try{
                int text=1;
                int agrado=2;
                int desagrado=3;
                int emocion=4;
                int calma=5;
                int control=6;
                int descontrol=7;
                int fecha=8;

                Timestamp date=new Timestamp(System.currentTimeMillis());
                String query="INSERT INTO tweets(tweet, agrado, desagrado, emocion,calma, control, descontrol, fecha) Values(?,?,?,?,?,?,?,?)";

                PreparedStatement pst =con.prepareStatement(query);
                pst.setString(text, tweet);
                pst.setBoolean(agrado, ASBCT_MLmut_a_prediction(1));
                pst.setBoolean(desagrado,ASBCT_MLmut_a_prediction(2));
                pst.setBoolean(emocion, ASBCT_MLmut_a_prediction(3));
                pst.setBoolean(calma, ASBCT_MLmut_a_prediction(4));
                pst.setBoolean(control, ASBCT_MLmut_a_prediction(5));
                pst.setBoolean(descontrol, ASBCT_MLmut_a_prediction(6));
                pst.setTimestamp(fecha, date);
                pst.executeUpdate();

                System.out.println("fECHA: " + date.toString());
            }catch(SQLException e){
                System.out.println("Error: " + e);
            }
        }
        else{
            System.out.println("Error");
        }
    }

    public static void main(String[] args) throws TwitterException, IOException, SQLException, ClassNotFoundException, InterruptedException {

        // TODO code application logic here
        Thread.sleep(120000);
        ConfigurationBuilder cb=new ConfigurationBuilder();
        //creacion de csv
        //PrintWriter pw=null;
        try{
            // File data=new File("Data.csv");
            FileWriter fw=new FileWriter("Data.csv",true);
            BufferedWriter bw=new BufferedWriter(fw);
            PrintWriter pw=new PrintWriter(bw);

            //Obtener datos Twitter
            cb.setDebugEnabled(true)
                    .setOAuthConsumerKey(oAuthConsumerKey)
                    .setOAuthConsumerSecret(oAuthConsumerSecret)
                    .setOAuthAccessToken(oAuthAccessToken)
                    .setOAuthAccessTokenSecret(oAuthAccessTokenSecret);

            TwitterFactory tf= new TwitterFactory(cb.build());
            twitter4j.Twitter twitter=tf.getInstance();
            Double lat=12.0464;
            Double longi=77.0428; //latitud y longitud ubicacion de Lima
            String resUnit="km";
            Query query=new Query(/*Aqui va el query a buscar*/"Futbol OR Mundial OR FIFA OR Peru OR blanquirroja OR \"36 OR años\" OR Bicolor OR Gareca OR Cueva OR Guerrero OR Carrillo OR Farfán OR Flores OR Gallese OR #rusia2018 OR #WorldCup OR #MundialRusia2018 OR #VamosPeru OR #AUSPER OR #PerúvsAustralia OR #AbrazoDeGol OR #ArribaPerú OR #PER OR #CHONGOPERU4NO OR #ContigoPeru OR #ElCanalDelMundial OR #peruvsfrancia OR #ArribaPeruCarajo OR #PerúEnRusia OR #Flores OR #farfan OR #Guerrero OR #gareca OR @fifaworldcup_es OR @SeleccionPeru -filter:retweets -filter:replies").lang(/*idioma*/"es");
            QueryResult result;
            int cont=0;


            StringBuilder builder = new StringBuilder();
        /*String ColumnNamesList = "Username"+"\t"+"location"+"\t"+"verified"+"\t"+"text";
        builder.append(ColumnNamesList +"\n");*/
            do{
                result=twitter.search(query);
                List<Status> status = result.getTweets();
                //Mostrar y escribir datos en archivo
                for(Status st : status){
                    builder.append(st.getUser().getName()+"\t");
                    builder.append(st.getUser().getLocation()+"\t");
                    builder.append(st.getUser().isVerified()+"\t");
                    builder.append(st.getText());
                    builder.append('\n');
                    TweetsCrudo.add(st.getText());
                    cont++;
                    System.out.println(st.getText());
                }

                for(String st1: TweetsLemma){
                    conexion(st1);
                }
                //System.out.println(cont);
            }while((query=result.nextQuery())!=null && cont<=300);
            //pw.write());
            pw.append(builder.toString());
            pw.close();
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }
    }
}

import java.io.*;
import java.util.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import edu.stanford.nlp.coref.CorefCoreAnnotations;

import edu.stanford.nlp.coref.data.CorefChain;
import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;

public class ASBCT {
  /** Usage: java -cp "*" StanfordCoreNlpDemo [inputFile [outputTextFile [outputXmlFile]]] */
  public static void main(String[] args) throws IOException {
    PrintWriter out;
    if (args.length > 1) {
      out = new PrintWriter(args[1]);
    } else {
      out = new PrintWriter(System.out);
    }
    PrintWriter xmlOut = null;
    if (args.length > 2) {
      xmlOut = new PrintWriter(args[2]);
    }

    Properties props = new Properties();
    props.load(IOUtils.readerFromString("StanfordCoreNLP-Spanish.properties"));
    props.setProperty("annotators", "tokenize, ssplit, pos,lemma, ner, parse");

    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

    //Get Info CSV
    String csvFile = "F:/Universidad/Tesis/Twitter Dataset/prepro/Stopwords.csv";
    BufferedReader br = null;
    String line = "";
    String cvsSplitBy = ",";

    try {

        br = new BufferedReader(new FileReader(csvFile));
        while ((line = br.readLine()) != null) {
            String[] sstopword = line.split(cvsSplitBy);
            //System.out.println("Stopword: [" + sstopword[0] + "]");
        }
    } catch (FileNotFoundException e) {
        e.printStackTrace();
    } catch (IOException e) {
        e.printStackTrace();
    } finally {
        if (br != null) {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /*Another Form
    CoreDocument document = new CoreDocument(sstopword);
    pipeline.annotate(document);*/

    Annotation annotation;
    if (args.length > 0) {
      annotation = new Annotation(IOUtils.slurpFileNoExceptions(args[0]));
    } else {
      annotation = new Annotation(sstopword);
    }

    pipeline.annotate(annotation);

    pipeline.prettyPrint(annotation, out);
    if (xmlOut != null) {
      pipeline.xmlPrint(annotation, xmlOut);
    }

    out.println();
    out.println("The top level annotation");
    out.println(annotation.toShorterString());
    out.println();

    List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);

    for(CoreMap sentence: sentences) {
      for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
        String word = token.get(TextAnnotation.class);
        String pos = token.get(PartOfSpeechAnnotation.class);
        String ne = token.get(NamedEntityTagAnnotation.class);
      }
      Tree tree = sentence.get(TreeAnnotation.class);
      SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
    }
    Map<Integer, CorefChain> graph = 
    annotation.get(CorefChainAnnotation.class);

    /*Parse tree si no me equivoco es lo que quieres faltaria reescribir eso*/
    }
    IOUtils.closeIgnoringExceptions(out);
    IOUtils.closeIgnoringExceptions(xmlOut);
  }

}

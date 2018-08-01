package mulan.ASBCT;

import java.io.*;
import java.io.Writer;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.Charset;
import java.util.List;

public class Writing {

    public static void writeResults(List<Comment> commentList) {
        //PrintWriter pw = null;
        File file = new File(StringUtils.PRECLASSIFICATION_RESULTS_CSV);

        try {
            Writer sb = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "ISO-8859-1"));
            //pw = new PrintWriter(,"UTF-8");
            //StringBuilder sb = new StringBuilder();
            for (Comment comment : commentList) {
                //byte[] ptext = comment.comment.getBytes(Charset.forName("ISO-8859-1"));
                //sb.append(new String(ptext, Charset.forName("UTF-8")));
                sb.append(comment.comment);
                sb.append(";");
                sb.append(String.valueOf(comment.unpleasant));
                sb.append(";");
                sb.append(String.valueOf(comment.pleasant));
                sb.append(";");
                sb.append(String.valueOf(comment.calm));
                sb.append(";");
                sb.append(String.valueOf(comment.excited));
                sb.append(";");
                sb.append(String.valueOf(comment.outOfControl));
                sb.append(";");
                sb.append(String.valueOf(comment.inControl));
                sb.append('\n');
            }
            //byte[] ptext = sb.toString().getBytes(Charset.forName("ISO-8859-1"));
            //pw.write(new String(ptext, Charset.forName("UTF-8")));
            //pw.close();
            sb.flush();
            sb.close();
            System.out.println("results.csv done!");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

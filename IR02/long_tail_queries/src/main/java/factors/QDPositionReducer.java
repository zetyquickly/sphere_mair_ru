package clickmodels;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;

import java.io.IOException;
import java.util.HashMap;

public class QDPositionReducer {
    public static final String MARKER = "QDPos";
    @SuppressWarnings("unchecked")
    public static void reduce(String queryurl,
                              Iterable<Text> values,
                              HashMap<String, Long> trainMarks,
                              HashMap<String, String> sampleNames,
                              MultipleOutputs<Text, Text> out) throws IOException, InterruptedException {

        final String DELIMETER = "::::";
        final String CLCK = "CLCK";
        final String SHOW = "SHOW";
        Double clicks_pos = 0.0;
        Double shows_pos = 0.0;
        Double clicks_cnt = 0.0;
        Double shows_cnt = 0.0;
        Double shows_min_pos = 11.0;
        Double shows_max_pos = -1.0;
        Double clicks_min_pos = 11.0;
        Double clicks_max_pos = -1.0;
        //System.out.println(queryurl);
        for (Text i : values) {
            String[] vals = i.toString().split(DELIMETER);
            Double pos = Double.parseDouble(vals[1]);
            if (vals[0].equals(CLCK)) {
                clicks_pos += pos;
                clicks_cnt += 1.0;
                if (pos <= clicks_min_pos) {
                    clicks_min_pos = pos;
                }
                if (pos >= clicks_max_pos) {
                    clicks_max_pos = pos;
                }
            } else if (vals[0].equals(SHOW)) {
                shows_pos += pos;
                shows_cnt += 1.0;
                if (pos <= shows_min_pos) {
                    shows_min_pos = pos;
                }
                if (pos >= shows_max_pos) {
                    shows_max_pos = pos;
                }
            }
        }

        Double shows_mean = shows_pos / shows_cnt, clicks_mean = clicks_pos / clicks_cnt;
        boolean flag = true;
        if (trainMarks.containsKey(queryurl)) {
            out.write("xgb", new Text(queryurl + "\t"
                            + shows_mean.toString() + " "
                            + shows_cnt.toString() + " "
                            + shows_max_pos.toString() + " "
                            + shows_min_pos.toString() + " "

                            + clicks_mean.toString() + " "
                            + clicks_cnt.toString() + " "
                            + clicks_max_pos.toString() + " "
                            + clicks_min_pos.toString() + " "),
                    new Text(trainMarks.get(queryurl).toString()), "QDPosTrain");
            flag = false;
        }
        if (sampleNames.containsKey(queryurl)) {
            out.write("xgb", new Text(shows_mean.toString() + " "
                    + shows_cnt.toString() + " "
                    + shows_max_pos.toString() + " "
                    + shows_min_pos.toString() + " "

                    + clicks_mean.toString() + " "
                    + clicks_cnt.toString() + " "
                    + clicks_max_pos.toString() + " "
                    + clicks_min_pos.toString() + " "), new Text(sampleNames.get(queryurl)), "QDPosTest");
            flag = false;
        }
        if (flag) {
            throw new InterruptedException("Some problems in reducer");
        }
    }
}


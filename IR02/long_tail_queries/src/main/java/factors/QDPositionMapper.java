package clickmodels;

import SERPLog.SERPLog;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

public class QDPositionMapper {
    private final String DELIMETER = "::::";
    private final String MARKER = "QDPos" + DELIMETER;
    private final String CLCK = "CLCK" + DELIMETER;
    private final String SHOW = "SHOW" + DELIMETER;
    HashSet<String> inTrainURLs;
    HashSet<String> inSampleURLs;

    public QDPositionMapper(HashSet<String> inTrainURLs, HashSet<String> inSampleURLs) {
        this.inSampleURLs = inSampleURLs;
        this.inTrainURLs = inTrainURLs;
    }

    @SuppressWarnings("unchecked")
    public void map(Mapper.Context context, SERPLog log) throws IOException, InterruptedException {

        for (int i = 0; i < log.shownLinks.length; i++) {
            String link = log.shownLinks[i];
            if (inTrainURLs.contains(log.query + " " + link) | inSampleURLs.contains(log.query + " " + link)) {
                context.write(new Text(MARKER + log.query + " " + link), new Text(SHOW + Integer.toString(i)));
                if (log.clickedPositions.contains(i)) {
                    context.write(new Text(MARKER + log.query + " " + link), new Text(CLCK + Integer.toString(i)));
                }
            }
        }

    }
}
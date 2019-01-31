package SERPLog;

import java.util.ArrayList;

public class SERPLogUnhashed extends SERPLog {
    public SERPLogUnhashed(String val) {
        super();
        String[] valArray = val.trim().split("\t");

        String[] queryAndGeo = valArray[0].split("@");
        this.query = queryAndGeo[0];
        this.geo = Long.parseLong(queryAndGeo[1]);


        // заменяем https:// в началах на http://
        String tmp = valArray[1].replace(",https://",",http://");
        tmp = tmp.replaceFirst("^https://", "http://");
        // добавляем во все колдунщики, которые идут не первыми, в начало http://
        tmp = tmp.replaceAll(",(?!http://)([a-zA-Z]+)", ",http://$1");
        tmp = tmp.replaceFirst("^http://", "");
        // делаем разбиение по ,http://
        String[] shownLinksUnhashed = tmp.split(",http://");

        this.shownLinks = new String[shownLinksUnhashed.length];
        for (int i = 0; i < shownLinksUnhashed.length; i++){
            this.shownLinks[i] = shownLinksUnhashed[i];
        }

        // заменяем https:// в началах на http://
        tmp = valArray[2].replace(",https://",",http://");
        tmp = tmp.replaceFirst("^https://", "http://");
        //System.out.println("After removing https://   " + tmp);
        // добавляем во все колдунщики, которые идут не первыми, в начало http://
        tmp = tmp.replaceAll(",(?!http://)([a-zA-Z]+)", ",http://$1");
        tmp = tmp.replaceFirst("^http://", "");
        //System.out.println("After adding http://   " + tmp);
        // делаем разбиение по ,http://
        String[] clickedLinks = tmp.split(",http://");

        this.clickedPositions = new ArrayList<Integer>();
        for (int i = 0; i < clickedLinks.length; i++){
            String currlink = clickedLinks[i];
            for (int j = 0; j < shownLinksUnhashed.length; j++){
                if (currlink.equals(shownLinksUnhashed[j])){
                    this.clickedPositions.add(j);
                }
            }
        }

        String[] timestamp_arr = valArray[3].split(",");
        this.timestamps = new long[timestamp_arr.length];
        for (int i=0; i<timestamp_arr.length; i++){
            this.timestamps[i] = Long.parseLong(timestamp_arr[i]);
        }
    }
}

import org.apache.hadoop.util.ToolRunner;
import SERPLog.SERPLogUnhashed;

public class RegexTest {
    static public void main(String[] args) throws Exception {
        String value = "zgjkycrbt bthjukbas@188\timages,videos,http://nihongo.aikidoka.ru/all_kanji.html,http://waysamurai.ru/articles/Japan_ieroglify,http://japancalligraphy.eu/ru/kanji,http://evelhyld.beon.ru/3470-548-japonskie-ieroglify.zhtml,http://hi-braa.spb.ru/nihhon/calligraphy.php,http://nihon-go.ru/yaponskie-ieroglifyi,http://kanjiname.ru/,http://kakprosto.ru/kak-67551-kak-perevesti-yaponskie-ieroglify,http://mytattoo.ru/2008/03/08/japonskii_simvoty_.html,http://miuki.info/2011/10/nekotorye-yaponskie-ieroglify-i-ix-znacheniya\timages,videos,http://miuki.info/2011/10/nekotorye-yaponskie-ieroglify-i-ix-znacheniya\t1494530497000\n";
        SERPLogUnhashed log = new SERPLogUnhashed(value);
        for (int i = 0; i < log.shownLinks.length; i++){
            System.out.println(log.shownLinks[i]);
        }
        for (int i = 0; i < log.clickedPositions.size(); i++){
            System.out.println(log.clickedPositions.get(i));
        }


    }
}

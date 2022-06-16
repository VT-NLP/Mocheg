package extractor;

import java.util.regex.Pattern;

import constants.Constants;
import constants.Args;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import edu.uci.ics.crawler4j.crawler.CrawlController;
import edu.uci.ics.crawler4j.crawler.Page;
import edu.uci.ics.crawler4j.crawler.WebCrawler;
import edu.uci.ics.crawler4j.parser.HtmlParseData;
import edu.uci.ics.crawler4j.url.WebURL;
import source.*;
import utils.MyFileWriter;


/**
 * This class is responsible to create Corpus 1 which contains all the
 * fact-check URLs present in Snopes fact-check page.
 *
 * @author Arpitha Nagaraja
 *
 */
public class FactCheckUrlExtractor extends WebCrawler {
    private MyFileWriter myFileWriter;
    private Source source;


    public FactCheckUrlExtractor() {
        this.source=Source.get_source(Args.source_str);
        String running_dir=Args.running_dir;//TODO  use Thread variable
        
        myFileWriter = new MyFileWriter(running_dir);
    }

    /**
     * This function is called before parsing the URL. This acts as a filter to
     * check if a particular URL has to be visited or not. Only if this function
     * returns true, the category page is visited to fetch the list of fact
     * check URLs that has to be validated.
     *
     * @param referringPage
     *            snopes.com page that has to be parsed to fetch the list of
     *            fact check URLs
     * @param url
     *            URL of the Snopes website
     * @return true if the URL is pointing to a category related to fact check
     *         else false is returned
     */
    @Override
    public boolean shouldVisit(Page referringPage, WebURL url) {
        String categoryUrl = url.getURL();
        // Check if the URL is a valid fact check URL
     
        if (Pattern.matches(this.source.next_page_pattern, categoryUrl)) {
//            myFileWriter.openWriteConnection("accessd.txt");
//            myFileWriter.writeLine(categoryUrl);
//            myFileWriter.closeWriteConnection();
            logger.info( "visit {}",categoryUrl);
            return true;
        }
        return false;
    }

    /**
     * This function will parse through all the pages present in the
     * 'www.snopes.com' archive to fetch the fact check URLs and updates this in
     * the file 'FactCheckUrlCorpus.csv'
     *
     * @param page
     *            The page that has to be parsed to fetch the fact check URLs.
     */
    @Override
    public void visit(Page page) {

        if (page.getParseData() instanceof HtmlParseData) {

            HtmlParseData parseData = (HtmlParseData) page.getParseData();
            Document htmlDocument = Jsoup.parseBodyFragment(parseData.getHtml());
            Elements factCheckUrls = htmlDocument.getElementsByTag("article");
            myFileWriter.openWriteConnection(Constants.SNOPES_URLS_CORPUS);
            for (int i = 0; i < factCheckUrls.size(); i++) {
//                if (factCheckUrls.get(i).select("a").attr("href").contains(Constants.SNOPES_FACT_CHECK))
                try{
                    myFileWriter.writeLine(this.source.extract_source_url(factCheckUrls,i));
                }    catch(Exception e){
                    System.out.print(" "+e);
                }
                

            }
            myFileWriter.closeWriteConnection();
        }
    }

    


    // public static void main(String[] args) {
    //     MyFileWriter myFileWriter = new MyFileWriter();
    //     myFileWriter.openWriteConnection(Constants.SNOPES_URLS_CORPUS);
    //     for (int i = 0; i < 3; i++) {
    // //                if (factCheckUrls.get(i).select("a").attr("href").contains(Constants.SNOPES_FACT_CHECK))
    //             myFileWriter.writeLine( i+"");

    //     }
    //     myFileWriter.closeWriteConnection();
    // }
}

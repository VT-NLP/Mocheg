package crawler;


import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import constants.Constants;
import edu.uci.ics.crawler4j.crawler.CrawlConfig;
import edu.uci.ics.crawler4j.crawler.CrawlController;
import edu.uci.ics.crawler4j.fetcher.PageFetcher;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtConfig;
import edu.uci.ics.crawler4j.robotstxt.RobotstxtServer;
import extractor.FactCheckUrlExtractor;
import source.Source;
import utils.FileUtil;
import utils.MyFileWriter;



public class UrlCrawler {
    private Logger logger = LoggerFactory.getLogger(getClass());
    public Source source;
    public int max_pages_to_fetch;
    public String running_dir;
    public UrlCrawler(String running_dir,Source source,int max_pages_to_fetch ){
        this.running_dir=running_dir;
        this.source=source;
        this.max_pages_to_fetch=max_pages_to_fetch;
    }
    /**
     * Crawler4j crawl latest fact-checking urls on the Snopes, filter out repeated urls.
     */
    public void urlCorpusConstruct(){
        File f2 = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.UNIQUE_URLS_CORPUS);
        if(f2.exists()){
            System.out.println("Corpus1.txt has been done, start extracting!");
            return;
        }
        logger.info("urlCorpusConstruct start");
        CrawlController factCheckUrlController = factCheckUrlCrawlConfig(running_dir,max_pages_to_fetch);
        if(factCheckUrlController!=null){
            // factCheckUrlController.addSeed(Constants.SNOPES_FACTCHECK_WEBSITE);
            factCheckUrlController.addSeed(this.source.seed_url);
            factCheckUrlController.startNonBlocking(FactCheckUrlExtractor.class,40);

            if (factCheckUrlController !=null){
                factCheckUrlController.waitUntilFinish();
            }

        }
        String snopesURLCorpus=running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_URLS_CORPUS;
        ArrayList<String> snopesUrls = FileUtil.readUlrs(snopesURLCorpus);
        Set<String> uniqueUrls = new HashSet<String>();
        for (String url : snopesUrls){
            uniqueUrls.add(url);
        }

        MyFileWriter myFileWriter = new MyFileWriter(running_dir);
        for (String url : uniqueUrls){
            myFileWriter.openWriteConnection(Constants.UNIQUE_URLS_CORPUS);
            myFileWriter.writeLine(url);
            myFileWriter.closeWriteConnection();
        }

        logger.info("urlCorpusConstruct end");
    }

    /**
     * This function initializes a controller for crawling snopes web pages to
     * fetch the list of all fact check URLs
     *
     * @return An instance of the crawl controller
     */
    private CrawlController factCheckUrlCrawlConfig(String running_dir,int max_pages_to_fetch) {
        // Specify the configurations required to perform web crawl
        CrawlConfig oCrawlConfig = new CrawlConfig();
        oCrawlConfig.setCrawlStorageFolder(running_dir+Constants.RESULT_STORAGE_DIRECTORY);
        oCrawlConfig.setMaxPagesToFetch( max_pages_to_fetch);
        // Initialize the controller that manages the crawling session
        PageFetcher oPageFetcher = new PageFetcher(oCrawlConfig);
        RobotstxtConfig oRobotstxtConfig = new RobotstxtConfig();
        RobotstxtServer oRobotstxtServer = new RobotstxtServer(oRobotstxtConfig, oPageFetcher);
        CrawlController oCrawlController = null;
        try {
            oCrawlController = new CrawlController(oCrawlConfig, oPageFetcher, oRobotstxtServer);
        } catch (Exception e) {
            System.err.println("Unable to initialize the fact check crawler controller");
            e.printStackTrace();
        }
        return oCrawlController;
    }
}

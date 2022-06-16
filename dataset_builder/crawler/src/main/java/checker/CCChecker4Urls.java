package checker;

import utils.AccessURL;
import utils.MyFileWriter;

import java.util.ArrayList;


/**
 *
 */

public class CCChecker4Urls extends AbstractChecker {


    private String notFoundUrls;
    private String foundUrls;
    private String logFile;
    private String commonCrawlIndex="http://index.commoncrawl.org";
    private String connectUrl = "-index?url=";
    private String running_dir;
    private ArrayList<String> indexServers = new ArrayList<String>(){
        {
            add("/CC-MAIN-2018-05");
            add("/CC-MAIN-2017-51");
            add("/CC-MAIN-2017-47");
            add("/CC-MAIN-2017-43");
            add("/CC-MAIN-2017-39");
            add("/CC-MAIN-2017-34");
            add("/CC-MAIN-2017-30");
            add("/CC-MAIN-2017-26");
            add("/CC-MAIN-2017-22");
            add("/CC-MAIN-2017-17");
            add("/CC-MAIN-2017-13");
            add("/CC-MAIN-2017-09");
            add("/CC-MAIN-2017-04");
            add("/CC-MAIN-2016-50");
            add("/CC-MAIN-2016-44");
            add("/CC-MAIN-2016-40");
            add("/CC-MAIN-2016-36");
            add("/CC-MAIN-2016-30");
            add("/CC-MAIN-2016-26");
            add("/CC-MAIN-2016-22");
            add("/CC-MAIN-2016-18");
            add("/CC-MAIN-2016-07");
            add("/CC-MAIN-2015-48");
            add("/CC-MAIN-2015-40");
            add("/CC-MAIN-2015-35");
            add("/CC-MAIN-2015-32");
            add("/CC-MAIN-2015-27");
            add("/CC-MAIN-2015-22");
            add("/CC-MAIN-2015-18");
            add("/CC-MAIN-2015-14");
            add("/CC-MAIN-2015-11");
            add("/CC-MAIN-2015-06");
            add("/CC-MAIN-2014-52");
            add("/CC-MAIN-2014-49");
            add("/CC-MAIN-2014-42");
            add("/CC-MAIN-2014-41");
            add("/CC-MAIN-2014-35");
            add("/CC-MAIN-2014-23");
            add("/CC-MAIN-2014-15");
            add("/CC-MAIN-2014-10");
            add("/CC-MAIN-2013-48");
            add("/CC-MAIN-2013-20");
        }
    };

    public CCChecker4Urls(String notFoundUrls, String foundUrls, String logFile,String running_dir){
        this.notFoundUrls=notFoundUrls;
        this.foundUrls = foundUrls;
        this.logFile = logFile;
        this.running_dir=running_dir;
    }

    /**
     * check if the link exists in the common crawl, and stores the index_server url into the FoundUrls.txt
     * @param url
     */
    public void checkUrl(String url){
        String downloadUrl = "";
        for (String indexServer : indexServers){
            downloadUrl = commonCrawlIndex+indexServer+connectUrl+url;
            int responseCode = getResponseCode(downloadUrl);
            if (responseCode==200){
                writeToFile(foundUrls,downloadUrl);
                writeToFile(logFile,url);
                System.out.println(downloadUrl+" has been found on common crawl");
                return;
            }
        }
        writeToFile(notFoundUrls,url);
        writeToFile(logFile,url);
    }

    /**
     * update the url not in the CC into NotFoundUrls file
     * @param file
     * @param content
     */
    private synchronized void writeToFile(String file, String content){
        MyFileWriter myFileWriter = new MyFileWriter(running_dir);
        myFileWriter.openWriteConnection(file);
        myFileWriter.writeLine(content);
        myFileWriter.closeWriteConnection();
    }

    /**
     * check the response code of index servers, show if it is crawled by CC
     * @param url
     * @return
     */
    private int getResponseCode(String url) {
        try {
            AccessURL accessURL = new AccessURL(url);
            String html = accessURL.getUrlContent();
            if (html!=null && html.length()>10){
                return 200;
            }else {
                return 404;
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return 404;
    }

}

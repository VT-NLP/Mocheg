package checker;

import constants.Constants;
import extractor.OriginDocExtractorOnWeb;
import utils.AccessURL;
import utils.MyCsvFileWriter;
import utils.MyFileWriter;


import java.util.regex.Pattern;
import java.util.regex.Matcher;

/**
 * check if the link crawled by the webarchive, if it is, then extract the clean text from the webarchive and update the corpus
 */

public class WebArchiveChecker extends AbstractChecker {

    private static final String webArchive = "http://archive.org/wayback/available?url=";
    private static final String filterInfo = "&timestamp=20170701";
    private AccessURL accessURL;
    private MyCsvFileWriter myCsvFileWriter;
    private MyFileWriter myFileWriter;
    private String logFile;
    private String running_dir;
    public WebArchiveChecker(String logFile,String running_dir){
        myCsvFileWriter=new MyCsvFileWriter();
        myFileWriter = new MyFileWriter(running_dir);
        this.logFile = logFile;
        this.running_dir=running_dir;
    }

    /**
     * check if the link exists in the webarchive, and get the wayback machine url, and extract clean text from wayback machine url.
     * @param link
     * @param snopesUrl
     */
    public void checkLink(String link,String snopesUrl){
        accessURL = new AccessURL(webArchive+link+filterInfo);
        String snapshotUrl = null;
        String content = null;

        try {
            content = accessURL.getUrlContent();
        }catch (Exception e){
            e.printStackTrace();
        }


        if (content!=null && content.length()>0){
            Pattern archivedPattern = Pattern.compile("\"archived_snapshots\":\\s*\\{([^\\}]*)");
            Matcher matcher = archivedPattern.matcher(content);
            String archivedInfo = new String();
            if(matcher.find()){
                archivedInfo=matcher.group(1);
            }
            if (archivedInfo.length()!=0){
                Pattern urlPattern = Pattern.compile("\"url\":\\s*\"([^\"]*)");
                Matcher urlMatcher = urlPattern.matcher(archivedInfo);
                if (urlMatcher.find()){
                    snapshotUrl = urlMatcher.group(1);
                }
            }
            if (snapshotUrl!=null && snapshotUrl.length()>0){
                OriginDocExtractorOnWeb originExtractor = new OriginDocExtractorOnWeb(snapshotUrl);
                String text = null;
                try{
                    text = originExtractor.getDoc();
                    if (text!=null && text.length()>0){
                        String[] newLine = {snopesUrl,link,snapshotUrl,text};
                        synchronized (this){
                            updateCorpus(newLine);
                            updateLog(snopesUrl+";"+link);
                        }
                    }else {
                        writeToFile(snopesUrl+";"+link);
                        updateLog(snopesUrl+";"+link);

                    }
                }catch (Exception e){
                    synchronized (this){
                        writeToFile(snopesUrl+";"+link);
                        updateLog(snopesUrl+";"+link);

                    }
                    e.printStackTrace();
                }

            }else {
                synchronized (this){
                    writeToFile(snopesUrl+";"+link);
                    updateLog(snopesUrl+";"+link);

                }            }
        }else {
            synchronized (this){
                writeToFile(snopesUrl+";"+link);
                updateLog(snopesUrl+";"+link);
            }
        }

    }

//    /**
//     * update the url not in the webarchive into NotFoundLinks file
//     * @param file
//     * @param content
//     */
//    private synchronized void writeToFile(File file, String content){
//        try {
//            BufferedWriter writer = new BufferedWriter(new FileWriter(file,true));
//            writer.write(content+'\n');
//            writer.flush();
//
//        } catch (Exception e) {
//             
//            e.printStackTrace();
//        }
//    }

    private void writeToFile(String content){
        myFileWriter.openWriteConnection(Constants.NOT_FOUND_LINKS_IN_AV);
        myFileWriter.writeLine(content);
        myFileWriter.closeWriteConnection();
    }

    /**
     * update extract info into corpus3 file
     * @param content
     */
    private void updateCorpus(String[] content){
        myCsvFileWriter.openWriteConnection(Constants.ORIGIN_DOC_CORPUS,running_dir);
        myCsvFileWriter.writeLine(content);
        myCsvFileWriter.closeWriteConnection();
    }

    private void updateLog(String content){
        myFileWriter.openWriteConnection(logFile);
        myFileWriter.writeLine(content);
        myFileWriter.closeWriteConnection();
    }
}

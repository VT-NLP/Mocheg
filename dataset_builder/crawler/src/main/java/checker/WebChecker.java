package checker;

import constants.Constants;
import extractor.OriginDocExtractorOnWeb;
import utils.MyCsvFileWriter;
import utils.MyFileWriter;


public class WebChecker extends AbstractChecker{


    private MyCsvFileWriter myCsvFileWriter;
    private MyFileWriter myFileWriter;
    private String logFile;
    private String running_dir;
    public WebChecker(String logFile,String running_dir){
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
        OriginDocExtractorOnWeb originDocumentExtractor = new OriginDocExtractorOnWeb(link);
        try{
            String text = originDocumentExtractor.getDoc();

            if (text!=null && text.length()>0){
                String[] newLine = {snopesUrl,link," ",text};
                synchronized (this){
                    updateCorpus(newLine);
                    updateLog(snopesUrl+";"+link);
                }
            }else {
                synchronized (this){
                    writeToFile(snopesUrl+";"+link);
                    updateLog(snopesUrl+";"+link);

                }
            }
        }catch (Exception e ){
            synchronized (this){
                writeToFile(snopesUrl+";"+link);
                updateLog(snopesUrl+";"+link);

            }
            e.printStackTrace();
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
        myFileWriter.openWriteConnection(Constants.CORRUPTED_LINKS);
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

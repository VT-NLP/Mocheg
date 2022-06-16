package extractor;

import constants.Constants;
import utils.MyCsvFileWriter;
import utils.MyFileWriter;
import utils.WARCDownloader;
import utils.WARCReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;

/**
 *
 * This class is used to implement multithreading for extract snopes information and stores them into the corpus.
 */


public class ClaimEvidenceExtractorThread implements Runnable{
    private int threadId;
    private ArrayList<String> urls;
    private MyFileWriter fileWriter ;
    private int partLength;
    private MyCsvFileWriter myCsvFileWriter;
    private String running_dir;


    public ClaimEvidenceExtractorThread(int threadId,ArrayList<String> urls,int partLength,String running_dir){
        this.threadId = threadId;
        this.urls = urls;
        this.partLength = partLength;
        myCsvFileWriter = new MyCsvFileWriter();
        this.running_dir=running_dir;
        this.fileWriter = new MyFileWriter(running_dir);
    }


    public void run() {
        int urlStart = threadId*partLength;
        int urlEnd = (threadId+1)*partLength;
        if (urls.size()<urlStart)
            urlStart = urls.size();
        if (urls.size()<urlEnd)
            urlEnd = urls.size();
        if (urlStart == urlEnd)
            return;

        ArrayList<String> partUrls = new ArrayList<String>(urls.subList(urlStart,urlEnd));

        for (String url : partUrls){
            try {
                process (url);
            }catch (Exception e){
                e.printStackTrace();
            }
        }
        System.out.println("thread" +threadId+" has finished his work! waiting");
    }

    public void process(String url) throws IOException{
        FilenameExtractor filenameExtractor = new FilenameExtractor(url);
        filenameExtractor.getFileInfo();
        String filePath = filenameExtractor.getFilePath();
        Long offset = filenameExtractor.getOffset();
        int length = filenameExtractor.getLength();
        String snopesUrl = url.split("=")[1];
        if(!(filePath==null || offset==null || length==0)){
            String downloadUrl = Constants.AMAZON_FILE_SERVER + filePath;
            WARCDownloader warcDownloader = new WARCDownloader(downloadUrl,Constants.DOWNLOAD_STORAGE_DIRECTORY,offset,length);
            warcDownloader.download();

            WARCReader warcReader = new WARCReader(Constants.DOWNLOAD_STORAGE_DIRECTORY,warcDownloader.getFileName(),snopesUrl);
            String html = warcReader.readFile();
            if (html==null || html.length()<1){
                System.out.println("html content in url " +url +" is null");
                updatedNotFoundUrl(snopesUrl);
            }else {
                ClaimEvidenceExtractor claimEvidenceExtractor = new ClaimEvidenceExtractor(html,snopesUrl,downloadUrl,offset,length,running_dir);
                update(claimEvidenceExtractor);
            }
        }
        else {
            System.out.println("Cannot download information from "+url);
            updatedNotFoundUrl(snopesUrl);
        }
        updateCCLog(url);
    }

    public synchronized void update(ClaimEvidenceExtractor claimEvidenceExtractor) throws IOException{
        String claim = claimEvidenceExtractor.getClaim();
        String offset = Long.toString(claimEvidenceExtractor.getOffset());
        String length = Integer.toString(claimEvidenceExtractor.getLength());
        String url = claimEvidenceExtractor.getUrl();
        String serverURL = claimEvidenceExtractor.getServerURL();
        if (claim!=null && !(claim.length() < 1)) {
            String truthfulness = claimEvidenceExtractor.getTruthfulness();
            if (!(truthfulness.length() < 1)) {
                HashSet<String> evidenceSet = claimEvidenceExtractor.getEvidenceSet();
                if (evidenceSet.size() != 0) {
                    String workedLog = url + " works for extraction!";
                    updateLogFile(workedLog);
                    String headline = claimEvidenceExtractor.getHeadline();
                    String category = claimEvidenceExtractor.getCategory();
                    String subCategory = claimEvidenceExtractor.getSubCategory();
                    String description = claimEvidenceExtractor.getDescription();
                    String source = claimEvidenceExtractor.getSource();
                    String origin = claimEvidenceExtractor.getOrigin();

                    if (origin.length() < 200) {
                        String originLog = url + " has no origin or bad origin";
                        updateLogFile(originLog);
                        System.out.println(url + " has no origin");
                    }
                    for (String e : evidenceSet) {
                        if (e.length() != 0) {
                            String[] infos = {url,serverURL,offset,length,category,subCategory,headline,
                                    description,source,claim,truthfulness,e,origin};
                            synchronized (this){
                                updateCsvFile(infos,Constants.CLAIM_EVIDENCE_CORPUS);
                            }

                        }
                    }

                    HashSet<String> originalDocumentLinkSet = claimEvidenceExtractor.getOriginalDocumentLinkSet();
                    if (originalDocumentLinkSet.size() > 0) {
                        for (String link : originalDocumentLinkSet){
                            if (link.length()!=0){
                                String[] linkInfo = {url,link};
                                synchronized (this){
                                    updateCsvFile(linkInfo,Constants.ORIGIN_LINK_CORPUS);
                                }

                            }
                        }

                    } else {
                        String linkLog = url + " has no links";
                        updateLogFile(linkLog);
//                        System.out.println(url + " has no links!");
                    }
                } else {
                    String evidenceLog = url + " has no evidence";
                    updateLogFile(evidenceLog);
                }
            } else {
                String ratingLog = url + " has no rating";
                updateLogFile(ratingLog);
            }

        } else {
            String claimLog = url + " has no claim";
            updateLogFile(claimLog);
        }
    }

    private synchronized void updatedNotFoundUrl(String snopesUrl){
        fileWriter.openWriteConnection(Constants.NOT_FOUND_URLS);
        fileWriter.writeLine(snopesUrl);
        fileWriter.closeWriteConnection();
    }

    private synchronized void updateLogFile(String content){
        fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
        fileWriter.writeLine(content);
        fileWriter.closeWriteConnection();
    }

    private synchronized void updateCCLog(String url){
        fileWriter.openWriteConnection(Constants.CC_EXTRACTOR_LOGGER);
        fileWriter.writeLine(url);
        fileWriter.closeWriteConnection();
    }

    private void updateCsvFile(String[] content,String fileName){
        myCsvFileWriter.openWriteConnection(fileName,this.running_dir);
        myCsvFileWriter.writeLine(content);
        myCsvFileWriter.closeWriteConnection();
    }


}

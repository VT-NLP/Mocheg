package extractor;

import java.util.HashSet;

import constants.Constants;
import utils.MyCsvFileWriter;
import utils.MyFileWriter;

/**
 *Extract information with snopes URLs that are not found on the Common Crawl.
 */
public class SnopesExtractorThread{

    private String snopesUrl;
    private String html;
    private MyCsvFileWriter myCsvFileWriter;
    private String running_dir;
    private String source_name;

    public SnopesExtractorThread(String snopesUrl,String html, String running_dir,String source_name){

        this.snopesUrl = snopesUrl;
        this.html = html;
        myCsvFileWriter = new MyCsvFileWriter();
        this.running_dir=running_dir;
        this.source_name=source_name;
    }

    public void process(){
        MyFileWriter fileWriter = new MyFileWriter(running_dir);
        ClaimEvidenceExtractor claimEvidenceExtractor =null;
        if(source_name=="Snopes"){
            claimEvidenceExtractor = new  ClaimEvidenceExtractor(html,snopesUrl," ",0L,0,running_dir);
        }else{
            claimEvidenceExtractor = new PolitifactClaimEvidenceExtractor(html,snopesUrl," ",0L,0,running_dir);
        }
 
        String claim = claimEvidenceExtractor.getClaim();
        String offset = Long.toString(claimEvidenceExtractor.getOffset());
        String length = Integer.toString(claimEvidenceExtractor.getLength());
        String url = claimEvidenceExtractor.getUrl();
        String serverURL = claimEvidenceExtractor.getServerURL();
        if (!(claim.length() < 1)) {
            String truthfulness = claimEvidenceExtractor.getTruthfulness();
            if (!(truthfulness.length() < 1)) {
                HashSet<String> evidenceSet = claimEvidenceExtractor.getEvidenceSet();


                String headline = claimEvidenceExtractor.getHeadline();
                String category = claimEvidenceExtractor.getCategory();
                String subCategory = claimEvidenceExtractor.getSubCategory();
                String description = claimEvidenceExtractor.getDescription();
                String source = claimEvidenceExtractor.getSource();
                String origin = claimEvidenceExtractor.getOrigin();
                String rulingOutline=claimEvidenceExtractor.getRulingOutlineExtactor();
                if (origin.length() < 200) {
                    fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
                    fileWriter.writeLine(url+" has no origin or bad origin");
                    fileWriter.closeWriteConnection();
                    System.out.println(url + " has no origin");
                }
                if (rulingOutline.length() < 200) {
                    fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
                    fileWriter.writeLine(url+" has no rulingOutline or bad rulingOutline");
                    fileWriter.closeWriteConnection();
                    System.out.println(url + " has no rulingOutline");
                }
                if (evidenceSet.size() != 0) {
                    fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
                    fileWriter.writeLine(url + " works for extraction!");
                    fileWriter.closeWriteConnection();
                    for (String e : evidenceSet) {
                        if (e.length() != 0) {
                            String[] infos = {url, serverURL, offset, length, category, subCategory, headline,
                                    description, source, claim, truthfulness, e, origin,rulingOutline};
                            updateCsvFile(infos,Constants.CLAIM_EVIDENCE_CORPUS);
                        }
                    }

                    
                } 
                else if (origin.length() >0|| rulingOutline.length() >0){ 
                    String[] infos = {url, serverURL, offset, length, category, subCategory, headline,
                        description, source, claim, truthfulness, "", origin,rulingOutline};
                    updateCsvFile(infos,Constants.CLAIM_EVIDENCE_CORPUS);
                    fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
                    fileWriter.writeLine(url+" has no evidence, write empty evidence");
                    fileWriter.closeWriteConnection();
                    System.out.println(url + " has no evidence! , write empty evidence");
                }
                HashSet<String> originalDocumentLinkSet = claimEvidenceExtractor.getOriginalDocumentLinkSet();
                if (originalDocumentLinkSet.size() > 0) {
                    for (String link : originalDocumentLinkSet) {
                        if (link.length()!= 0) {
                            String[] linkInfo = {url, link};
                            updateCsvFile(linkInfo,Constants.ORIGIN_LINK_CORPUS);
                        }
                    }

                } else {
                    fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
                    fileWriter.writeLine(url+" has no links");
                    fileWriter.closeWriteConnection();
                    System.out.println(url + " has no links!");
                }
            } else {
                fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
                fileWriter.writeLine(url+" has no rating");
                fileWriter.closeWriteConnection();
                System.out.println(url + " has no rating!");
            }

        } else {
            fileWriter.openWriteConnection(Constants.EXTRACTOR_LOGS);
            fileWriter.writeLine(url+" has no claim");
            fileWriter.closeWriteConnection();
            System.out.println(url + " has no claim");
        }
    }

    private synchronized void updateCsvFile(String[] content,String filename){
        myCsvFileWriter.openWriteConnection(filename,running_dir);
        myCsvFileWriter.writeLine(content);
        myCsvFileWriter.closeWriteConnection();
    }
}

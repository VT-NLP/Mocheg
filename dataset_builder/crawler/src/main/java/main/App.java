package main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import au.com.bytecode.opencsv.CSVReader;
import checker.CCChecker4Urls;
import checker.Checker;
import checker.CheckerThread4Links;
import checker.CheckerThread4Urls;
import checker.WebArchiveChecker;
import checker.WebChecker;
import constants.Args;
import constants.Constants;
import crawler.UrlCrawler;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import extractor.ClaimEvidenceExtractorThread;
import extractor.SnopesExtractorThread;
import source.Source;
import utils.AccessURL;
import utils.FileUtil;
import utils.MyCsvFileWriter;
import utils.MyFileWriter;




public class App{
     


    private Logger logger = LoggerFactory.getLogger(getClass());
    public Source source;
    public String running_dir;
    public int max_pages_to_fetch;
    public App(String running_dir,String source_enum_str,int max_pages_to_fetch){
        this.running_dir=running_dir;
        this.source=Source.get_source(source_enum_str);
        this.max_pages_to_fetch=max_pages_to_fetch;
    }

    public static void main(String[] args) throws Exception{
        String mode=args[0];
        String running_dir=Args.running_dir;
        String source_enum_str=Args.source_str;//Politifact
        int max_pages_to_fetch=Args.max_pages_to_fetch;
        if( args.length>1){
             running_dir=args[1];
            
             source_enum_str=args[2];
        }
  
            
        
        
        App app = new App(running_dir,source_enum_str,max_pages_to_fetch);
        
        app.start(mode );

    }

    public void start(String mode) throws Exception{

        long startTime = System.nanoTime();
 
        

        if (  mode.equals("mode3")){
            
            urlCorpusConstruct();
            claimEvideceExtractorOnSnopes(Constants.UNIQUE_URLS_CORPUS,running_dir);
        }else{
            String filePath = Constants.INITIAL_DATA+Constants.SNOPES_URLS;
            globalUrlsCheckInCrawl(filePath,running_dir);
            claimEvideceExtractorWithCrawl(running_dir);
            deleteDownloads(running_dir);
            claimEvideceExtractorOnSnopes(Constants.NOT_FOUND_URLS,running_dir);//TODO need modify annotatingLabel and predict.py since we have empty evidence
        }

        if (mode.equals("mode1") || mode.equals("mode2")){
            annotatingLabel(  running_dir);
        }

        if (mode.equals("mode1") || mode.equals("mode3")){
            gen_relevant_document(running_dir);
        }

        long endTime = System.nanoTime();
        long totalTime = endTime-startTime;
        System.out.println("total time cost"+(totalTime));
    }

    public void gen_relevant_document(String running_dir)throws Exception{
        localLinksCheckInArchive(  running_dir);
        localLinksExtractorInWeb(running_dir);
        snopesLinksHandler(running_dir);

    }

    /**
     * delete the download snapshots of Snopes URLs
     */

    public void deleteDownloads(String running_dir){
        File file = new File(System.getProperty("user.dir"));
        File downloadDir = new File(file+"/" + Constants.DOWNLOAD_STORAGE_DIRECTORY);// "/DownloadFiles"
        try {
            FileUtils.deleteDirectory(downloadDir);
        }catch (Exception e){
            System.out.println("Download Files directory doesn't exist. "+e );
        }
    }
    private void urlCorpusConstruct(){
        UrlCrawler urlCrawler=new UrlCrawler(running_dir, source,max_pages_to_fetch);
        urlCrawler.urlCorpusConstruct();
    }
    

    /**
     * read the snopes ulr from local file and check if they are crawled by common crawl
     * output: Found_URLs file stores url crawled by common crawl, NOT_FOUND_URLS
     * @throws Exception
     */
    private void globalUrlsCheckInCrawl(String filePath,String running_dir) throws Exception{

        ArrayList<String> urls = readUlrs(filePath);
        String notFoundFile = Constants.NOT_FOUND_URLS;
        String foundFile = Constants.FOUND_URLS;
        String logFile = Constants.CHECKE_LOGGER;

        File f = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CLAIM_EVIDENCE_CORPUS);
        File f1 = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CHECKE_LOGGER);
        if (f.exists()){
            System.out.println("Checking on Common Crawl has been done, start extracting!");
            return;
        }
        if (f1.exists()){
           ArrayList<String> crawledUrls = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CHECKE_LOGGER);
           urls = removeCrawledUrls(urls,crawledUrls);
        }
        System.out.println(urls.size());


        //numThread should be carefully selected in order to not overload the commoncrawl server.
        int numThread = 10;
        ExecutorService es = Executors.newFixedThreadPool(numThread);

        int partLength = urls.size()%numThread ==0 ? urls.size()/numThread : urls.size()/numThread +1;

        for (int i=0;i<numThread;i++){
            Checker checker = new CCChecker4Urls(notFoundFile,foundFile,logFile,running_dir);
            es.execute(new CheckerThread4Urls(i,urls,partLength,checker));
        }
        es.shutdown();
        es.awaitTermination(600, TimeUnit.MINUTES);

        System.out.println("Global urls check if they are on commoncrawl has been done!");
    }


    /**
     *read snopes ulr form FOUND_URLS, and download warc file from common crawl contains html content, and extract wanted info from these files
     * output: corpus2 stored evidence, claim and other relevant information w.r.t the corresponding html content
     *         LinkCorpus stored the snopes url and links in the origin text.
     *         the url with null html content or cannot download will be stored into the NOT_FOUND_URLs file.
     * @throws Exception
     */
    public void claimEvideceExtractorWithCrawl(String running_dir) throws Exception{
        String foundUrls = running_dir+Constants.RESULT_STORAGE_DIRECTORY+ Constants.FOUND_URLS;
        ArrayList<String> urls = readUlrs(foundUrls);
        ArrayList<String> processedUrls = new ArrayList<String>();
        for (String url : urls){
            processedUrls.add(url);
        }

        File f1 = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CLAIM_EVIDENCE_CORPUS);
        File f = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CC_EXTRACTOR_LOGGER);
        File f2 = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ORIGIN_DOC_CORPUS);

        //write csv file headers
        if (!f1.exists()){
            MyCsvFileWriter myCsvFileWriter = new MyCsvFileWriter();
            myCsvFileWriter.openWriteConnection(Constants.CLAIM_EVIDENCE_CORPUS,running_dir);
            myCsvFileWriter.writeLine(Constants.SNOPES_CLAIM_EVIDENCE_HEADERS.split(","));
            myCsvFileWriter.closeWriteConnection();
            myCsvFileWriter.openWriteConnection(Constants.ORIGIN_LINK_CORPUS,running_dir);
            myCsvFileWriter.writeLine(Constants.SNOPES_URL_LINKS_HEADERS.split(","));
            myCsvFileWriter.closeWriteConnection();
        }

        if(f2.exists()){
            System.out.println("URLs on Common crawl has been crawled, starting processing the reset URLs.");
            return;
        }

        if (f.exists()){
            ArrayList<String> crawledUrls = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CC_EXTRACTOR_LOGGER);
            processedUrls = removeCrawledUrls(processedUrls,crawledUrls);
        }



        //numThread should be carefully selected to not overload the commoncrawl server.
        int threadSize = 5;


        ExecutorService es = Executors.newFixedThreadPool(threadSize);

        int length = processedUrls.size();
        System.out.println(processedUrls.size());
        int chunkLength = length % threadSize == 0 ? length / threadSize : length / threadSize + 1;
        for(int threadId =0; threadId<threadSize;threadId++){
            es.execute(new ClaimEvidenceExtractorThread(threadId,processedUrls,chunkLength,running_dir));
        }

        es.shutdown();
        es.awaitTermination(600, TimeUnit.MINUTES);


        System.out.println("Claim evidence extractor on common crawl has been done!");

    }

    /**
     * read snopes url from NotFoundURls which contatins urls not found in common crawl and extract wanted info from the website
     * output: corpus2 stored evidence, claim and other relevant information w.r.t the corresponding html content
     *         LinkCorpus stored the snopes url and links in the origin text.
     * @throws Exception
     */
    public void claimEvideceExtractorOnSnopes(String filename,String running_dir){

        String notFoundUrls = running_dir+Constants.RESULT_STORAGE_DIRECTORY+filename;
        ArrayList<String> urls = readUlrs(notFoundUrls);
        ArrayList<String> processedUrls = new ArrayList<String>();

        for (String url : urls){
            url = url.replaceFirst("http://","https://");
            processedUrls.add(url);
        }
        processedUrls = new ArrayList<String>(new HashSet<String>(processedUrls));

        File f = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_EXTRACTOR_LOGGER);
        File f2 = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ORIGIN_DOC_CORPUS);
        if(f2.exists()){
            System.out.println("Checking on Common Crawl has been done, start extracting!");
            return;
        }

        if (f.exists()){
            ArrayList<String> crawledUrls = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_EXTRACTOR_LOGGER);
            processedUrls = removeCrawledUrls(processedUrls,crawledUrls);
        }
        System.out.println(processedUrls.size());
        MyFileWriter myFileWriter = new MyFileWriter(running_dir);

        for (String url : processedUrls){
            try{
                AccessURL accessURL = new AccessURL(url);
                String html = accessURL.getUrlContent();
                SnopesExtractorThread snopesExtractorThread = new SnopesExtractorThread(url,html,running_dir,this.source.name);
                snopesExtractorThread.process();
            }catch (Exception e){
                myFileWriter.openWriteConnection(Constants.CORRUPTED_URLS_SNOPES);
                myFileWriter.writeLine(url);
                myFileWriter.closeWriteConnection();
                e.printStackTrace();
            }
            updateSnopesLog(url,running_dir);

        }
        logger.info("claimEvideceExtractorOnSnopes end");
    }

    /**
     * read snopes urls and links corresponding to these urls, firstly extract originDoc from link contains archive, for the links which is belonging to snopes,
     * stored them to a local file, then check theses links if they crawled by web archive and extract orginDoc from the webarchive.
     * output: SnopeLinks store links which contain snopes.com
     *         Corpus3 stores information of snopes url, link, webarchive url and original document
     *         NotFoundLinks stores links not crawled by webarchive
     * @throws Exception
     */
    public void localLinksCheckInArchive(String running_dir) throws Exception{
        ArrayList<String[]> lines = new ArrayList<String[]>();
        CSVReader reader = new CSVReader(new FileReader(running_dir+Constants.RESULT_STORAGE_DIRECTORY + Constants.ORIGIN_LINK_CORPUS));

        //writer csv file headers
        File docFile = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ORIGIN_DOC_CORPUS);
        if (!docFile.exists()){
            MyCsvFileWriter corpus3writer = new MyCsvFileWriter();
            corpus3writer.openWriteConnection(Constants.ORIGIN_DOC_CORPUS,running_dir);
            corpus3writer.writeLine(Constants.ORIGIN_DOC_HEADERS.split(","));
            corpus3writer.closeWriteConnection();
        }


        String[] nextLine;
        nextLine = reader.readNext();
        MyFileWriter myFileWriter = new MyFileWriter(running_dir);

        while ((nextLine = reader.readNext()) != null) {
            if (  nextLine.length<2){
                continue;
            }
            if ( nextLine[1].length() == 0 || nextLine[0].length()==0){
                continue;
            }
            lines.add(nextLine);
        }

        File logFile = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ALL_LINKS_LOGGER);
        if (logFile.exists()){
            ArrayList<String> crawledLines = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ALL_LINKS_LOGGER);
            lines = removeCrawledLines(lines,crawledLines);
        }

        System.out.println(lines.size());
        List<String[]> archiveLinks = new ArrayList<String[]>();
        List<String[]> processedLinks = new ArrayList<String[]>();
        for (String[] line : lines) {
            if (line[1].contains("archive.")) {
                archiveLinks.add(line);
            }else if(line[1].contains("snopes.com")){
                myFileWriter.openWriteConnection(Constants.SNOPES_LINKS);
                myFileWriter.writeLine(line[0]+";"+line[1]);
                myFileWriter.closeWriteConnection();
                updateLinksLog(line[0]+";"+line[1],Constants.ALL_LINKS_LOGGER,running_dir);
                continue;
            }else {
                processedLinks.add(line);
            }
        }
        reader.close();

        int numThread = 5;
        ExecutorService es = Executors.newFixedThreadPool(numThread);
        int partLength = archiveLinks.size()%numThread ==0 ? archiveLinks.size()/numThread : archiveLinks.size()/numThread +1;
        for (int i=0;i<numThread;i++){
            Checker checker = new WebChecker(Constants.ALL_LINKS_LOGGER,running_dir);
            es.execute(new CheckerThread4Links(i,archiveLinks,partLength,checker));
        }
        es.shutdown();
        es.awaitTermination(600, TimeUnit.MINUTES);

        es = Executors.newFixedThreadPool(numThread);
        partLength = processedLinks.size()%numThread ==0 ? processedLinks.size()/numThread : processedLinks.size()/numThread +1;
        for (int i=0;i<numThread;i++){
            Checker checker = new WebArchiveChecker(Constants.ALL_LINKS_LOGGER,running_dir);
            es.execute(new CheckerThread4Links(i,processedLinks,partLength,checker ));
        }
        es.shutdown();
        es.awaitTermination(600, TimeUnit.MINUTES);

        System.out.println("Local links check if they are on archive has been done!");
        logger.info("localLinksCheckInArchive end");
    }

    /**
     * Crawl Links that cannot find on the web archive with its original url
     * @throws Exception
     */
    public void localLinksExtractorInWeb(String running_dir) throws Exception{
        ArrayList<String> lines = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.NOT_FOUND_LINKS_IN_AV);
        List<String[]> restPairs = new ArrayList<String[]>();

        File logFile = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.NOT_FOUND_LINKS_LOGGER);
        if (logFile.exists()){
            ArrayList<String> crawledLines = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.NOT_FOUND_LINKS_LOGGER);
            lines = removeCrawledUrls(lines,crawledLines);
        }


        for (String line : lines){
            String[] eachLine = line.split(";",2);
            restPairs.add(eachLine);
        }

        System.out.println(restPairs.size());
        int numThread = 5;
        ExecutorService es = Executors.newFixedThreadPool(numThread);
        int partLength = restPairs.size()%numThread ==0 ? restPairs.size()/numThread : restPairs.size()/numThread +1;

        for (int i=0;i<numThread;i++){
            Checker checker = new WebChecker(Constants.NOT_FOUND_LINKS_LOGGER,running_dir);
            es.execute(new CheckerThread4Links(i,restPairs,partLength,checker));
        }
        es.shutdown();
        es.awaitTermination(600, TimeUnit.MINUTES);

        System.out.println("Original Document extraction on original website has been done!");
        logger.info("localLinksExtractorInWeb end");
    }

    /**
     * handle links in Doc Link Corpus that are the Snopes URLs
     * @throws Exception
     */
    public void snopesLinksHandler(String running_dir) throws Exception{
        ArrayList<String> lines = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_LINKS);
        List<String[]> snopesPairs = new ArrayList<String[]>();

        File docFile = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_LINKS_LOGGER);
        if (docFile.exists()){
            ArrayList<String> crawledLines = readUlrs(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_LINKS_LOGGER);
            lines = removeCrawledUrls(lines,crawledLines);
        }
        lines = new ArrayList<String>(new HashSet<String>(lines));
        System.out.println(lines.size());

        String logFile = Constants.SNOPES_LINKS_LOGGER;
        int count = 0;
        boolean isProcessed=false;
        for (String snopesLink: lines){
            count+=1;
            String[] snopesLinkpair = snopesLink.split(";",2);
            CSVReader reader = new CSVReader(new FileReader(running_dir+Constants.RESULT_STORAGE_DIRECTORY + Constants.CLAIM_EVIDENCE_CORPUS));

            //skip the headr
            String[] nextLine;
            nextLine = reader.readNext();

            while ((nextLine = reader.readNext()) != null) {
                if (snopesLinkpair[1].equals(nextLine[0])){
                    String[] content ={snopesLinkpair[0],snopesLinkpair[1]," ",nextLine[12]};
                    updateCorpus(Constants.ORIGIN_DOC_CORPUS,content,running_dir);
                    updateLinksLog(snopesLink,logFile,running_dir);
                    isProcessed = true;
                    break;
                }
            }
            reader.close();
            if (!isProcessed){
                snopesPairs.add(snopesLinkpair);
            }else {
                isProcessed = false;
            }
        }

        int numThread = 5;
        ExecutorService es = Executors.newFixedThreadPool(numThread);
        int partLength = snopesPairs.size()%numThread ==0 ? snopesPairs.size()/numThread : snopesPairs.size()/numThread +1;

        for (int i=0;i<numThread;i++){
            Checker checker = new WebChecker(logFile,running_dir);
            es.execute(new CheckerThread4Links(i,snopesPairs,partLength,checker));
        }
        es.shutdown();
        es.awaitTermination(600, TimeUnit.MINUTES);

        System.out.println("original document extraction on Snopes has been done!");
        logger.info("snopesLinksHandler end");
    }

//    /**
////     * find the claim-evidence both in the newly constructed Corpus2 and annotated Corpus2, and label the pairs with the label in annotated Corpus 2
////     * @throws Exception
////     */
////
////    private void annotatingLabel() throws Exception{
////
////        CSVReader reader = new CSVReader(new FileReader(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CLAIM_EVIDENCE_CORPUS));
////        //skip the header
////        String[] nextLine;
////        nextLine = reader.readNext();
////
////        MyCsvFileWriter myCsvFileWriter = new MyCsvFileWriter();
////        myCsvFileWriter.openWriteConnection(Constants.OUR_ANNOTATED_CORPUS);
////        myCsvFileWriter.writeLine(Constants.ANNOTATED_CLAIM_EVIDENCE_HEADERS.split(","));
////        myCsvFileWriter.closeWriteConnection();
////
////        int count = 0;
////        while ((nextLine = reader.readNext()) != null) {
////            Boolean isIncoporate = false;
////            String[] content = new String[15];
////            System.arraycopy(nextLine,0,content,0,nextLine.length);
////            String snopeUrl = nextLine[0];
////            Set<String> evidenceSet = new HashSet<String>();
////            StringTokenizer evidenceTokenizer = new StringTokenizer(nextLine[11]," \t\n\r\f,.:;?![]'");
////
//////            System.out.println("IN OUR corpus: "+nextLine[11]);
////            while (evidenceTokenizer.hasMoreElements()){
////                String word = evidenceTokenizer.nextElement().toString();
////                word = StringUtils.lowerCase(word);
////                if (StringUtils.isAlpha(word)){
////                    evidenceSet.add(word);
////                }
////            }
////
////            CSVReader newCorpusReader = new CSVReader(new InputStreamReader(new FileInputStream(running_dir+Constants.RESULT_STORAGE_DIRECTORY + Constants.ANNOTATED_CORPUS),"UTF-8"));
////
////            //skip the header
////            String[] annotatedLine;
////            annotatedLine = newCorpusReader.readNext();
////
////            snopeUrl = snopeUrl.replaceFirst("http[s]?://","");
////            while (((annotatedLine=newCorpusReader.readNext())!=null)){
////                String annotatedUrl = annotatedLine[2].replaceFirst("http[s]?://","");
////                if (snopeUrl.equals(annotatedUrl)){
////                    Set<String> annotatedWordSet = new HashSet<String>();
////                    String evidence = annotatedLine[8];
////                    evidence = evidence.replaceAll("[0-9][0-9]?_\\{","");
////                    evidence = evidence.replaceAll("}","");
////                    evidence = evidence.replaceAll("<p>|</p>","");
////                    StringTokenizer annotatedTokenzier = new StringTokenizer(evidence," \t\n\r\f,.:;?![]'");
////                    while (annotatedTokenzier.hasMoreElements()){
////                        String word = annotatedTokenzier.nextToken().toString();
////                        word = StringUtils.lowerCase(word);
////                        if (StringUtils.isAlpha(word))
////                            annotatedWordSet.add(word);
////                    }
////                    Set<String> intersection = new HashSet<String>(evidenceSet);
////                    intersection.retainAll(annotatedWordSet);
////
////                    float divison = (float)intersection.size()/evidenceSet.size();
////                    if (divison>0.9){
////                        System.out.println(nextLine[11]);
////                        System.out.println(evidence);
////                        content[13]=annotatedLine[10];
////                        updateCorpus(Constants.OUR_ANNOTATED_CORPUS,content);
////                        isIncoporate= true;
////                        count += 1;
////                        break;
////                    }
////                }
////            }
////            newCorpusReader.close();
////            if (!isIncoporate){
////                content[13] = " ";
////                updateCorpus(Constants.OUR_ANNOTATED_CORPUS,content);
////            }
////
////        }
////        reader.close();
////        System.out.println(count + "entries has found its label!");
////    }

//    /**
//     * read snopes urls and links corresponding not found in webarchive, and check if they are crawled by cc and extract original document from CC
//     * output: Corpus3 stores information of snopes url, link, webarchive url and original document
//     *         NotFoundRestLinks stores links not crawled by CC and Webarchive
//     * @throws Exception
//     */
//    private void localLinksCheckInCrawl() throws Exception{
//        ArrayList<String> lines = readUlrs(Constants.NOT_FOUND_LINKS_IN_AV);
//        List<String[]> restPairs = new ArrayList<String[]>();
//        File notFoundFile = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.NOT_FOUND_LINKS_IN_CC);
//
//
//        for (String line : lines){
//            String[] eachLine = line.split(";");
//            restPairs.add(eachLine);
//        }
//
//        int numThread=20;
//        ExecutorService es = Executors.newFixedThreadPool(numThread);
//
//        int partLength = restPairs.size()%numThread ==0 ? restPairs.size()/numThread : restPairs.size()/numThread +1;
//
//        for (int i=0;i<numThread;i++){
//            Checker checker = new CCChecker4Links(notFoundFile);
//            es.execute(new CheckerThread4Links(i,restPairs,partLength,checker));
//        }
//        es.shutdown();
//        es.awaitTermination(Long.MAX_VALUE, TimeUnit.HOURS);
//
//        System.out.println("Original Document extraction has been done!");
//    }

    /**
     * Read urls or url and link pairs from local file
     * @param filePath
     * @return
     */
    private ArrayList<String> readUlrs(String filePath){
        return FileUtil.readUlrs(filePath);
    }



    

    private void updateCorpus(String filename,String[] content,String running_dir){
        MyCsvFileWriter myCsvFileWriter = new MyCsvFileWriter();
        myCsvFileWriter.openWriteConnection(filename,running_dir);
        myCsvFileWriter.writeLine(content);
        myCsvFileWriter.closeWriteConnection();
    }


    /**
     * find the claim-evidence both in the newly constructed Corpus2 and annotated Corpus2, and label the pairs with the label in annotated Corpus 2
     * @throws Exception
     */


    private void annotatingLabel(String running_dir) throws Exception{

        CSVReader reader = new CSVReader(new FileReader(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CLAIM_EVIDENCE_CORPUS));
        //skip the header
        String[] nextLine;
        nextLine = reader.readNext();

        File f = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.OUR_ANNOTATED_CORPUS);
        if (f.exists()){
            System.out.println("Annotation has been done");
            return;
        }

        MyCsvFileWriter myCsvFileWriter = new MyCsvFileWriter();
        myCsvFileWriter.openWriteConnection(Constants.OUR_ANNOTATED_CORPUS,running_dir);
        myCsvFileWriter.writeLine(Constants.ANNOTATED_CLAIM_EVIDENCE_HEADERS.split(","));
        myCsvFileWriter.closeWriteConnection();

        int count = 0;
        while ((nextLine = reader.readNext()) != null) {
            Boolean isIncoporate = false;
            String[] content = new String[16];
            System.arraycopy(nextLine,0,content,0,nextLine.length);
            String snopeUrl = nextLine[0];
            String evidence = nextLine[11];
            evidence = evidence.replaceAll("<p>|</p>","");
            evidence = evidence.replaceAll("[^A-Za-z0-9]","");
            evidence =evidence.toLowerCase();
            String chars = new String();
            if (evidence.length()>200){
                chars = evidence.substring(0,200);
            }else {
                chars = evidence;
            }
            String evidenceSents = evidenceSplitting(nextLine[11]);

            CSVReader newCorpusReader = new CSVReader(new InputStreamReader(new FileInputStream(Constants.INITIAL_DATA + Constants.ANNOTATED_CORPUS),"UTF-8"));

            //skip the header
            String[] annotatedLine;
            annotatedLine = newCorpusReader.readNext();

            snopeUrl = snopeUrl.replaceFirst("http[s]?://","");
            while (((annotatedLine=newCorpusReader.readNext())!=null)) {
                String annotatedUrl = annotatedLine[1].replaceFirst("http[s]?://", "");
                if (snopeUrl.equals(annotatedUrl)) {
                    String annoChars = annotatedLine[2];
                    if (annoChars.equals(chars)){
                        content[13]= evidenceSents;
                        content[14]=annotatedLine[3];
                        content[15]=annotatedLine[4];
                        updateCorpus(Constants.OUR_ANNOTATED_CORPUS,content,running_dir);
                        isIncoporate= true;
                        count += 1;
                        break;
                    }

                }
            }
            newCorpusReader.close();
            if (!isIncoporate){
                content[13] = evidenceSents;
                content[14] = " ";
                content[15] = " ";
                updateCorpus(Constants.OUR_ANNOTATED_CORPUS,content,running_dir);
            }

        }
        reader.close();
        System.out.println(count + "entries has found its label!");
    }


    private String evidenceSplitting(String evidence){
//        Matcher m = Pattern.compile("\\.[A-Z]").matcher(evidence);
//        while (m.find()){
//            evidence = evidence.replace(m.group(),". "+m.group().substring(1,2));
//        }
        Reader stringReader = new StringReader(evidence);
        DocumentPreprocessor dpt = new DocumentPreprocessor(stringReader);
        ArrayList<String> sentences = new ArrayList<String>();
        for (List<HasWord> sentence : dpt){
            String sentenceString = sentence.get(0).toString();
            for (int j=1; j<sentence.size();j++){
                String tkn = sentence.get(j).toString();
                if ((tkn.equals("."))|| tkn.equals(",")||tkn.equals("!")||tkn.equals(":")||tkn.equals(";")){
                    sentenceString += tkn;
                }else {
                    sentenceString += " "+tkn;
                }
            }
            sentenceString = normalizeText(sentenceString);
            sentences.add(sentenceString);
        }
        String splitSents = new String();
        for (int i=0;i<sentences.size();i++){
            String sent = normalizeText(sentences.get(i));
            System.out.println(sent);
            if (sent.length()==0){
                continue;
            }
            splitSents += " "+i+"_{"+sent+"}";
        }
        return splitSents;
    }

    private static String normalizeText(String txt){
        txt = txt.replace("-LSB- ", "[");
        txt = txt.replace(" -RSB-", "]");
        txt = txt.replace("-LRB- ", "(");
        txt = txt.replace(" -RRB-", ")");
        txt = txt.replace(" '", "'");
        txt = txt.replace("`` ", "\"");
        txt = txt.replace(" ,", ",");
        txt = txt.replace("` ", "'");
        txt = txt.replace(" n't", "n't");
        txt = txt.replace("<p> ", "");
        txt = txt.replace("</p> ", "");
        txt = txt.replace(" </p> ","");
        txt = txt.replace("</p>","");
        return txt;
    }

    private ArrayList<String> removeCrawledUrls(ArrayList<String> urls, ArrayList<String> crawledUrls){
        urls.removeAll(crawledUrls);
        return urls;
    }

    private ArrayList<String []> removeCrawledLines(ArrayList<String[]> lines, ArrayList<String> crawledLines){
        ArrayList<String> allLines = new ArrayList<String>();
        for (String [] line : lines){
            allLines.add(line[0]+";"+line[1]);
        }
        allLines.removeAll(crawledLines);
        ArrayList<String []> cleanLines = new ArrayList<String[]>();
        for (String line : allLines){
            cleanLines.add(line.split(";"));
        }
        return cleanLines;
    }

    private void updateSnopesLog(String url, String running_dir){
        MyFileWriter fileWriter = new MyFileWriter(running_dir);
        fileWriter.openWriteConnection(Constants.SNOPES_EXTRACTOR_LOGGER);
        fileWriter.writeLine(url);
        fileWriter.closeWriteConnection();
    }

    private void updateLinksLog(String url,String filename, String running_dir){
        MyFileWriter fileWriter = new MyFileWriter(running_dir);
        fileWriter.openWriteConnection(filename);
        fileWriter.writeLine(url);
        fileWriter.closeWriteConnection();
    }
}
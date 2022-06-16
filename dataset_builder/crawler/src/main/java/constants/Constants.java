package constants;

    /**
     * This class contains all the constants used in the program.
     */
public class Constants {
        // public static String RUNNING_DIR="src/test/resources/";
        public static String TEST_DIR="src/test/resources/testcase/testClaimEvideceExtractorWithCrawl1/";
        public static String RUNNING_DIR="";
        public static String RESULT_STORAGE_DIRECTORY=RUNNING_DIR+"Results/"; // Directory path to store all the results
        public static String INITIAL_DATA = RUNNING_DIR+"Data/"; //Directory to store the Annotations and provided Corpus1
        public static String DOWNLOAD_STORAGE_DIRECTORY=RUNNING_DIR+"DownloadFiles/";

        public static final String CSV_SEPARATOR = ","; // Separator used in csv files
        public static final String CSV_QUOTE = "\""; // Quote used to update the data to csv files



        // Headers for the csv files
        public static final String SNOPES_CLAIM_EVIDENCE_HEADERS = "Snopes URL,Commoncrawl URL,Offset,Length,Category,SubCategory,Headline,Description,Source,Claim,Truthfulness,Evidence,Origin";
        public static final String SNOPES_URL_LINKS_HEADERS="Snopes URL,Original Link URL";
        public static final String ORIGIN_DOC_HEADERS="Snopes URL,Link URL,Archive Url,Origin Document";
        public static final String ANNOTATED_CLAIM_EVIDENCE_HEADERS= "Snopes URL,Commoncrawl URL,Offset,Length,Category,SubCategory,Headline,Description,Source,Claim,Truthfulness,Evidence,Origin,Snippets,Stance,Evidence_Sentences";

        public static final String AMAZON_FILE_SERVER = "https://commoncrawl.s3.amazonaws.com/";
        public static final String SNOPES_FACT_CHECK_PATTERN =  ".+(fact-check/page)/.*";///page 
        public static final String SNOPES_CATEGORY_PATTERN = ".+(fact-check/category)/.*";
        public static final String SNOPES_FACTCHECK_WEBSITE = "https://www.snopes.com/fact-check/"; // Snopes website
        public static final String SNOPES_WEBSITE="https://www.snopes.com/";
        public static final String SNOPES_FACT_CHECK="fact-check";

 

        //files to store urlschecker results
        public static final String NOT_FOUND_URLS = "NotFoundURLs.txt";
        public static final String FOUND_URLS="FoundURLs.txt";
        public static final String NOT_FOUND_LINKS_IN_AV="NotFoundLinks.txt";
        public static final String NOT_FOUND_LINKS_IN_CC="NotFoundRestLinks.txt";
        public static final String CORRUPTED_URLS_SNOPES="CorruptedUrls.txt";
        public static final String SNOPES_LINKS="SnopesLinks.txt";
        public static final String EXTRACTOR_LOGS="Log.txt";
        public static final String CORRUPTED_LINKS="CorruptedLinks.txt";

        //files for genereated corpus
        public static final String SNOPES_URLS_CORPUS="URLCorpus.txt";
        public static final String UNIQUE_URLS_CORPUS="Corpus1.txt";
        public static final String CLAIM_EVIDENCE_CORPUS="Corpus2.csv";
        public static final String ORIGIN_DOC_CORPUS="Corpus3.csv";
        public static final String DOC_EVIDENCE_CORPUS="Corpus4.csv";
        public static final String ORIGIN_LINK_CORPUS="LinkCorpus.csv";
//        public static  final String ANNOTATED_CORPUS="SnopesCorpus14June2017_stance_sentence1.csv";
        public static final String ANNOTATED_CORPUS = "Annotations.csv";
        public static final String OUR_ANNOTATED_CORPUS="Corpus2_annotated.csv";
//        public static final String SNOPES_URLS="UpdatedURLCorpus.txt";
        public static final String SNOPES_URLS="URLS.txt";



            //Log Files
        public static final String CHECKE_LOGGER = "CheckerLog.txt";
        public static final String CC_EXTRACTOR_LOGGER = "CCExtractorLog.txt";
        public static final String SNOPES_EXTRACTOR_LOGGER = "SnopesExtractorLog.txt";
        public static final String ALL_LINKS_LOGGER = "AllLinksLog.txt";
        public static final String NOT_FOUND_LINKS_LOGGER = "NotFoundLinksLog.txt";
        public static final String SNOPES_LINKS_LOGGER = "SnopesLinksLog.txt";


}


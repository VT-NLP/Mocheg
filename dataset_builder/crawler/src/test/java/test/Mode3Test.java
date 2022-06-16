package test;

import java.io.File;

import constants.Constants;
import main.App;

public class Mode3Test{
    

    // @Test 
    public void testClaimEvidenceExtractorOnSnopes2 ( ) throws Exception{
        String running_dir="src/test/resources/testcase/testClaimEvidenceExtractorOnSnopes_mode3/"; 
         
        File file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_EXTRACTOR_LOGGER);
        file.delete();
        // String running_dir="Results/run1/";
        App app = new App(running_dir,"Snopes",1);
        app.claimEvideceExtractorOnSnopes(Constants.UNIQUE_URLS_CORPUS,running_dir);
    }   
    // @Test 
    // public void test_politifact_snopesLinksHandler_mode3() throws Exception{
    //     String running_dir="src/test/resources/testcase/test_politifact_snopesLinksHandler_mode3/"; 
         
    //     // File file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_EXTRACTOR_LOGGER);
    //     // file.delete();
    //     // file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CLAIM_EVIDENCE_CORPUS);
    //     // file.delete();
    //     // file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ORIGIN_LINK_CORPUS);
    //     // file.delete();
    //     // String running_dir="Results/run1/";
    //     App app = new App(running_dir,"Politifact",1);
    //     app.snopesLinksHandler(running_dir);
    // }

    // @Test 
    public void test_politifact_ClaimEvidenceExtractorOnSnopes_mode3() throws Exception{
        String running_dir="src/test/resources/testcase/test_politifact_ClaimEvidenceExtractorOnSnopes_mode3/"; 
         
        File file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_EXTRACTOR_LOGGER);
        file.delete();
        file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CLAIM_EVIDENCE_CORPUS);
        file.delete();
        file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.ORIGIN_LINK_CORPUS);
        file.delete();
        // String running_dir="Results/run1/";
        App app = new App(running_dir,"Politifact",1);
        app.claimEvideceExtractorOnSnopes(Constants.UNIQUE_URLS_CORPUS,running_dir);
    }
    // @Test
    // public void testLocalLinksCheckInArchive() throws Exception{
    //     App app = new App();
    //     app.localLinksCheckInArchive();
    // }
    // // @Test
    // public void testLocalLinksExtractorInWeb() throws Exception{
    //     App app = new App();
    //     app.localLinksExtractorInWeb();
    //     // assertEquals(20, calculator.multiply(4,5),      
    //     // "Regular multiplication should work");   
    // }
    // @Test
    public void testLocalLinksCheckInArchive() throws Exception{
        String running_dir="src/test/resources/testcase/test_politifact_relevant_document_mode3/"; 
        App app = new App(running_dir,"Politifact",1);
        app.gen_relevant_document(running_dir);
    } 
}
package test;

import java.io.File;

import constants.Args;
import constants.Constants;
import main.App;

public class Mode1Test{
    String run_dir="";
    // @Test
    public void testLocalLinksCheckInArchive() throws Exception{
        String running_dir="Results/run1/";
        App app = new App(running_dir,Args.source_str,40);
 
        
        app.localLinksCheckInArchive(run_dir);
    }
    // @Test
    public void testLocalLinksExtractorInWeb() throws Exception{
        String running_dir="Results/run1/";
        App app = new App(running_dir,Args.source_str,40);
        app.localLinksExtractorInWeb(run_dir);
        // assertEquals(20, calculator.multiply(4,5),      
        // "Regular multiplication should work");   
    }

    // @Test 
    public void testClaimEvidenceExtractorOnSnopes1 ( ) throws Exception{
        String running_dir="src/test/resources/testcase/testClaimEvidenceExtractorOnSnopes1/"; 
        File file = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.SNOPES_EXTRACTOR_LOGGER);
        file.delete();
        
        // String running_dir="Results/run1/";
        App app = new App(running_dir,Args.source_str,40);
        app.claimEvideceExtractorOnSnopes(Constants.NOT_FOUND_URLS,running_dir);
    }   


    // @Test
    public void testClaimEvideceExtractorWithCrawl1() throws Exception{
        String running_dir="src/test/resources/testcase/testClaimEvideceExtractorWithCrawl1/"; 
        File f = new File(  running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CC_EXTRACTOR_LOGGER);
        f.delete();
        // String running_dir="Results/run1/";
        App app = new App(running_dir,Args.source_str,40);
        

        app.claimEvideceExtractorWithCrawl(running_dir);
        app.deleteDownloads( running_dir);
    }

    // @Test
    public void testLocalLinksCheckInArchive1() throws Exception{
        String running_dir="src/test/resources/testcase/testLocalLinksCheckInArchive1/"; 
        File f = new File(  running_dir+Constants.RESULT_STORAGE_DIRECTORY+Constants.CC_EXTRACTOR_LOGGER);
        f.delete();
        // String running_dir="Results/run1/";
        App app = new App(running_dir,Args.source_str,40);
        

        app.localLinksCheckInArchive(running_dir);
         
    }

    
}
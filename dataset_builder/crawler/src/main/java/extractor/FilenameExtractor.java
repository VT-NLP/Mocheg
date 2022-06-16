package extractor;

import utils.AccessURL;

import java.util.regex.Pattern;
import java.util.regex.Matcher;



public class FilenameExtractor {

    private String crawlUrl;
    private String filePath;
    private Long offset;
    private int length=0;
    private AccessURL accessURL;

    public FilenameExtractor(String crawlUrl){
        this.crawlUrl = crawlUrl;
        this.getFileInfo();
    }

    public Integer getLength() {
        return length;
    }

    public Long getOffset() {
        return offset;
    }

    public String getFilePath() {
        return filePath;
    }

    /**
     * retrieve filepath on the Amazon web Service with the retruned json file
     */
    public void getFileInfo (){
        try{
            accessURL = new AccessURL(crawlUrl);
            String content = accessURL.getUrlContent();
            if (!(content.length()<1)){
                String[] infos = content.split("\n");
                int maxLength = 0;
                for(String info : infos){
                    Pattern statusPattern = Pattern.compile("\"status\":\\s*\"([^\"]+)");
                    Matcher statusMatcher = statusPattern.matcher(info);
                    if(statusMatcher.find()){
                        int status = Integer.parseInt(statusMatcher.group(1));
                        if (status==200){
                            Pattern namePattern = Pattern.compile("\"filename\":\\s*\"([^\"]+)");
                            Matcher nameMatcher = namePattern.matcher(info);
                            nameMatcher.find();
                            filePath = nameMatcher.group(1);

                            Pattern offsetPattern = Pattern.compile("\"offset\":\\s*\"([^\"]+)");
                            Matcher offsetMatcher = offsetPattern.matcher(info);
                            offsetMatcher.find();
                            offset = Long.parseLong(offsetMatcher.group(1));

                            Pattern lengthPattern = Pattern.compile("\"length\":\\s*\"([^\"]+)");
                            Matcher lengthMatcher = lengthPattern.matcher(info);
                            lengthMatcher.find();
                            length = Integer.parseInt(lengthMatcher.group(1));

                            break;
                        }
                    }else {
                        Pattern lengthPattern = Pattern.compile("\"length\":\\s*\"([^\"]+)");
                        Matcher lengthMatcher = lengthPattern.matcher(info);
                        lengthMatcher.find();
                        length = Integer.parseInt(lengthMatcher.group(1));
                        if (length>maxLength){
                            Pattern namePattern = Pattern.compile("\"filename\":\\s*\"([^\"]+)");
                            Matcher nameMatcher = namePattern.matcher(info);
                            nameMatcher.find();
                            filePath = nameMatcher.group(1);

                            Pattern offsetPattern = Pattern.compile("\"offset\":\\s*\"([^\"]+)");
                            Matcher offsetMatcher = offsetPattern.matcher(info);
                            offsetMatcher.find();
                            offset = Long.parseLong(offsetMatcher.group(1));
                            maxLength = length;
                        }
                    }
                }

            }


        }catch (Exception e){
            e.printStackTrace();
        }
    }

}

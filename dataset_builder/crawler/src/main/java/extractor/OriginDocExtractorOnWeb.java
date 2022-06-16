package extractor;

import utils.AccessURL;


public class OriginDocExtractorOnWeb {

    private String accessUrl;

    public OriginDocExtractorOnWeb(String accessUrl){
        this.accessUrl = accessUrl;
    }

    public String getDoc(){
        String content = null;


        try{
            AccessURL accessURL = new AccessURL(accessUrl);
            String html = accessURL.getUrlContent();
            if (html!=null && html.length()>0){
                OriginDocumentExtractor originDocumentExtractor = new OriginDocumentExtractor(html);
                content = originDocumentExtractor.extractDocument();
            }else {
                return null;
            }
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }



        return content;
    }
}

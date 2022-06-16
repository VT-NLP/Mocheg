package utils;

import java.io.FileInputStream;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.StringUtils;
import org.archive.io.ArchiveReader;
import org.archive.io.ArchiveRecord;
import org.archive.io.warc.WARCReaderFactory;


public class WARCReader {

    private String storagePath;
    private String filename;
    private String url;

    public WARCReader(String storagePath, String filename, String url){
        this.storagePath = storagePath;
        this.filename = filename;
        this.url = url;
    }

    /**
     * read warc file, and remove the Request, Response header, and get the clean HTML content.
     * @return
     */
    public String readFile(){

        try{
            String filePath = storagePath+filename;
            FileInputStream fileInputStream = new FileInputStream(filePath);
            ArchiveReader archiveReader = WARCReaderFactory.get(filePath,fileInputStream, true);

            for (ArchiveRecord r : archiveReader){
                String urlInContent = r.getHeader().getUrl();

                if (urlInContent == null || (!urlInContent.startsWith("http://")&& !urlInContent.startsWith("https://"))){
                    continue;
                }
                urlInContent = urlInContent.replaceFirst("http[s]?://","");
                url =url.replaceFirst("http[s]?://","");
//                System.out.println(url);

                if (url.equals(urlInContent)){

                    byte[] rawData = IOUtils.toByteArray(r,r.available());
                    String content = new String(rawData);
                    content = StringEscapeUtils.unescapeHtml(content);
                    if (content.contains("<html")){
                        content = StringUtils.substringBetween(content,"<html","</html");
                    }else if (content.contains("<HTML")){
                        content = StringUtils.substringBetween(content,"<HTML","</HTML>");
                    }
                    content = "<html" + content +"</html>";
                    return content;
                }
            }
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("No content found or File Not Found!");
        }
        return null;
    }


}

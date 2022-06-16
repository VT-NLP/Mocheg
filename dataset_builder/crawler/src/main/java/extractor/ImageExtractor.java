package extractor;

 

import java.io.File;
import java.net.URL;
import org.apache.commons.io.FileUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import constants.Constants;
 
public class ImageExtractor { 
	public static void main(String[] args) throws Exception {
        
	}


	
	public static void findImageUrl() throws Exception{
		String url = "https://www.snopes.com/fact-check/beach-handball-uniforms-photo/"; // double quotes in the content of the website url you want to climb here skipped here
        Document document = Jsoup.connect(url).post();//Get url website html content
		String running_dir="";
		Elements elements = document.getElementsByClass("block");
        for (int i = 0; i < elements.size(); i++) {//cyclic output
            Element yuansu= elements.get(i);// used to get a single class with the content of "block"
            Elements yuansu_zi = yuansu.getElementsByTag("img");//Get the child whose class is "block" as the content of img
            String imgurl = yuansu_zi.attr("src");//Get the src attribute value in img
            String imgname = yuansu.attr("title");//Get the class src attribute value in "block"
            
            obtainImage(imgurl, running_dir+Constants.RESULT_STORAGE_DIRECTORY+"img/", imgname);//The first parameter is the image path to be downloaded. The second is the stored path. The third is the name of the image.
		}
	}
	public static void obtainImage(String imgurl,String dizhi,String imgname) throws Exception{
		 URL url = new URL(imgurl);//Create url class
		 String path =dizhi+imgname+".jpg"; //The name of the image must be suffixed
		File file = new File(path);
		 FileUtils.copyURLToFile(url, file);//Download as the image address of the url and save the downloaded image to the address of the file.
		 System.out.println("Read Successful");
	}
}

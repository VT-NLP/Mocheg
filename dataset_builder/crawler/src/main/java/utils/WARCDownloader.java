package utils;

import constants.Constants;

import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.File;

/**
 * This class is to download the file captured by the Common Crawl.
 */
public class WARCDownloader {

    private String downloaderUrl;
    private String storagePath;
    private String fileName;
    private long offset;
    private int length;

    public WARCDownloader(String downloaderUrl, String storagePath,long offset,int length){
        this.downloaderUrl = downloaderUrl;
        this.storagePath = storagePath;
        this.offset = offset;
        this.length = length;
    }

    public String getFileName(){    return this.fileName;}

    public void download(){

        File theDir = new File(Constants.DOWNLOAD_STORAGE_DIRECTORY);

        // if the directory does not exist, create it
        if (!theDir.exists()) {
            try{
                System.out.println("Crete Download directory!");
                theDir.mkdir();
            }
            catch(SecurityException se){
                //handle it
                se.printStackTrace();
            }
        }
        try{
            URL url = new URL(downloaderUrl);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();

            fileName = downloaderUrl.substring(downloaderUrl.lastIndexOf("/")+1);
            String storePath = storagePath+fileName;

            File file =new File(storePath);
            if (file.exists()){
                System.out.println("This file has been downloaded before, it may becauese two webpage are stored into a same warc file");
                file.delete();
            }else {
                file.createNewFile();
            }
            FileOutputStream fileOutputStream = new FileOutputStream(file);

            connection.setRequestMethod("GET");
            connection.setConnectTimeout(5000);
            connection.setRequestProperty("Range","bytes="+offset+"-"+(offset+length-1));
            while(connection.getResponseCode()==503){
                connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("GET");
                connection.setConnectTimeout(5000);
                connection.setRequestProperty("Range","bytes="+offset+"-"+(offset+length-1));
            }
            if (connection.getResponseCode() == 206){
                InputStream inputStream = connection.getInputStream();

                byte[] buffer = new byte[4096];
                int len=0;
                while ((len = inputStream.read(buffer))!=-1){
                    fileOutputStream.write(buffer,0,len);
                }
                fileOutputStream.close();
                inputStream.close();
            }
            connection.disconnect();
            Thread.sleep(5000);

        }catch (Exception e){
            e.printStackTrace();
            System.out.println("download fail!");
        }

    }

}

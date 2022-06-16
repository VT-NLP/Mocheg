package utils;

import java.util.ArrayList;

public class FileUtil {
    public static ArrayList<String> readUlrs(String filePath){
        MyFileReader myFileReader = new MyFileReader();
        myFileReader.openReadConnection(filePath);
        String line;
        if (filePath.endsWith(".csv")){
            line = myFileReader.readLine();
        }
        ArrayList<String> urls = new ArrayList<String>();
        while ((line=myFileReader.readLine())!=null){
            urls.add(line);
        }
        return urls;
    }
    
}

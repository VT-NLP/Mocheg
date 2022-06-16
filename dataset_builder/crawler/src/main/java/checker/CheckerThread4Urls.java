package checker;

import java.util.ArrayList;

/**
 * This class is used to implement multithreading for snopes url checking if they are crawled by some web archive.
 */

public class CheckerThread4Urls implements Runnable{

    private int threadId;
    private int partLength;
    private ArrayList<String> urls;
    private Checker checker;
    private String running_dir;
    public CheckerThread4Urls(int thredId, ArrayList<String> urls, int partLength, Checker checker ){
        this.threadId = thredId;
        this.urls = urls;
        this.partLength = partLength;
        this.checker = checker;
      
    }



    public void run(){
        int urlStart = threadId*partLength;
        int urlEnd = (threadId+1)*partLength;
        if (urls.size()<urlStart)
            urlStart = urls.size();
        if (urls.size()<urlEnd)
            urlEnd = urls.size();
        if (urlStart == urlEnd)
            return;

        ArrayList<String> partUrls = new ArrayList<String>(urls.subList(urlStart,urlEnd));

        for (String url : partUrls){
            checker.checkUrl(url);
        }
    }

}

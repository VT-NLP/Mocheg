package checker;


import java.util.ArrayList;
import java.util.List;

/**
 * This class is used to implement multithreading for link checking if they are crawled by some web archive.
 */

public class CheckerThread4Links implements Runnable{

    private int threadId;
    private int partLength;
    private List<String[]> urls;
    private Checker checker;
    

    public CheckerThread4Links(int thredId, List<String[]> urls, int partLength, Checker checker ){
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

        List<String[]> partUrls = new ArrayList<String[]>(urls.subList(urlStart,urlEnd));

        for (String[] url : partUrls){
            checker.checkLink(url[1],url[0]);
        }
    }
}

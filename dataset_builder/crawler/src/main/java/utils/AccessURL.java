package utils;


import org.apache.http.HttpResponse;
import org.apache.http.NoHttpResponseException;
import org.apache.http.client.HttpRequestRetryHandler;
import org.apache.http.client.ServiceUnavailableRetryStrategy;
import org.apache.http.client.config.CookieSpecs;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.conn.HttpHostConnectException;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.impl.client.LaxRedirectStrategy;
import org.apache.http.protocol.HttpContext;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.*;
import java.util.Random;



public class AccessURL {

    private String url;
    //    private Map<String,Integer> proxies;
    private Boolean isRetrieveUrl=false;

    public  AccessURL(String url){
        this.url = url;
        this.url = url.replace("\"","");
    }


    /**
     * access the url and convert the content on url into string for continue processing
     * @return
     */
    public String getUrlContent() throws Exception{
        String html =null;
        String decodedURL = URLDecoder.decode(url, "UTF-8");
        URL uRL = new URL(decodedURL);
        URI uri = new URI(uRL.getProtocol(), uRL.getUserInfo(), uRL.getHost(), uRL.getPort(), uRL.getPath(), uRL.getQuery(), uRL.getRef());
        url = uri.toASCIIString();

        RequestConfig config = RequestConfig.custom().setCookieSpec(CookieSpecs.STANDARD).setMaxRedirects(10).setCircularRedirectsAllowed(true).setSocketTimeout(60000).build();


        CloseableHttpClient httpClient = HttpClients.custom().setServiceUnavailableRetryStrategy(createStatusRetry()).setRedirectStrategy(new LaxRedirectStrategy())
                .setDefaultRequestConfig(config).setRetryHandler(createRetryHandler()).build();
        HttpGet httpGet = new HttpGet(url);
//        httpGet.addHeader("Authorization", "Basic "+Base64.encodeToString("rat#1:rat".getBytes(),Base64.NO_WRAP));
        httpGet.setHeader("User-Agent","Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36");

        try {
            HttpResponse response = httpClient.execute(httpGet);
            System.out.println(url + " : " + response.getStatusLine().getStatusCode());
            int code = response.getStatusLine().getStatusCode();
            if (code < 200 || code > 299) {
                html = null;
            } else {
                BufferedReader rd = new BufferedReader(
                        new InputStreamReader(response.getEntity().getContent()));

                StringBuffer result = new StringBuffer();
                String line = "";
                while ((line = rd.readLine()) != null) {
                    result.append(line + "\n");
                }
                html = result.toString();
            }

        } finally {
            httpClient.close();
        }
        Thread.sleep(1000);
        return html;

    }

    /**
     *
     * Retry if a url is no accessible.
     * @return
     */

    private HttpRequestRetryHandler createRetryHandler() {

        return new HttpRequestRetryHandler() {
            public boolean retryRequest(IOException e, int numRetry, org.apache.http.protocol.HttpContext httpContext) {
                if (numRetry>5){
                    return false;
                }
                if (e instanceof NoHttpResponseException ||
                        e instanceof SocketTimeoutException ||
                        e instanceof SocketException){
                    Random random = new Random();
                    int blockTime = random.nextInt(15);
                    try{
                        Thread.sleep(blockTime*1000);
                    }catch (InterruptedException ie){
                        ie.printStackTrace();
                    }
                    return true;
                }
                if (e instanceof HttpHostConnectException){
                    if (numRetry>10){
                        return false;
                    }else {
                        return true;
                    }
                }
                return false;
            }
        };

    }

    /**
     * Retry to access the url if the response code is between 500 and 599.
     * @return
     */
    private ServiceUnavailableRetryStrategy createStatusRetry(){
        return new ServiceUnavailableRetryStrategy() {
            public boolean retryRequest(HttpResponse httpResponse, int retryCount, HttpContext httpContext) {
                if (retryCount>10){
                    return false;
                }
                int code = httpResponse.getStatusLine().getStatusCode();
                if (code>=500 && code <=599){
                    return true;
                }

                return false;
            }

            public long getRetryInterval() {
                Random random = new Random();
                long sleepTime = random.nextInt(15);
                return sleepTime*1000;
            }
        };
    }


}

package extractor;


import org.apache.commons.lang.StringEscapeUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.safety.Whitelist;
import utils.Utils;
import org.jsoup.select.Elements;
import org.jsoup.nodes.Element;
import com.kohlschutter.boilerpipe.extractors.ArticleExtractor;


public class OriginDocumentExtractor {

    private String htmlContent;
    private Utils dkProUtils;
    private String originDocument;
    private Document doc;


    public OriginDocumentExtractor(String htmlContent){
        this.htmlContent = htmlContent;
        this.dkProUtils = new Utils();
        doc = Jsoup.parseBodyFragment(htmlContent);

    }


    /**
     * This function use the boilerpipe to remove templates on the web page
     * @return plain text
     */
    public String extractDocument() {
        if (htmlContent.length()>0 && htmlContent!=null){
            Element body = doc.select("body").first();
            Elements elements = body.getElementsByAttributeValue("id","wm-ipp");
            for (Element e: elements){
                e.remove();
            }
            String newBody = body.toString();
            try {
                originDocument = ArticleExtractor.getInstance().getText(newBody);
                Document doc1 = Jsoup.parse(originDocument);
                originDocument = stringNormalize(doc1.text(),true);
            }catch (Exception e){
                originDocument = null;
            }
        }
        return originDocument;
    }

    /**
     * This function is responsible to normalize the sting by removing the HTML
     * tags.
     *
     * @param text
     *            String that has to be normalized.
     * @param tag
     *            Boolean value. 'true' indicates that all the HTML tags except
     *            paragraph tags are removed, else all the tags are removed.
     * @return normalized string
     */
    private String stringNormalize(String text, boolean tag) {
        // Remove all anchor and img tags
        Whitelist whitelist;
        if (tag) {
            whitelist = new Whitelist().addTags("p");
            text = Jsoup.clean(text, whitelist);
            text = text.replaceAll("<p></p>", "");
            text = text.replaceAll("<p>&nbsp;*</p>", "");
            text = text.replaceAll("<p>\\s*</p>", "");
            if (!text.startsWith("<p>")) {
                text = text.replaceFirst("<p>", "</p> <p>");
                text = "<p>" + text;
            }
            if (!text.endsWith("</p>")) {
                String toReplace = "</p>";
                int pos = text.lastIndexOf(toReplace);
                if (pos > -1) {
                    text = text.substring(0, pos) + "</p> <p>"
                            + text.substring(pos + toReplace.length(), text.length());
                }
                text = text + "</p>";
            }
        } else {
            text = Jsoup.clean(text, Whitelist.none());
        }

        // Replace space and extra para tags if present
        text = text.replaceAll("&nbsp;", " ");
        text = text.replaceAll("<p></p>", "");
        text = text.replaceAll("<p>\\s*</p>", "");

        // Remove unwanted texts
        text = text.replaceAll("\\{.*\\}", "");
        text = text.replaceAll("<p></p>", "");
        text = text.replaceAll("<p>\\s*</p>", "");
        text = text.replaceAll("<p>\\s*<p>", "<p>");
        text = text.replaceAll("</p>\\s*</p>", "</p>");

        text = dkProUtils.normalizeBreaks(text);
        text = dkProUtils.normalize(text);

        text = StringEscapeUtils.unescapeHtml(text);
        text = text.replaceAll("&gt;", ">");
        text = text.replaceAll("&lt;", "<");
        text = text.replaceAll("&amp;", "&");

        // Replace all double quotes quotes
        text = replaceQuotation(text);

        // Replace new line character with space
//        text = text.replaceAll("\n", " ");
        text = text.replaceAll("\r", " ");

        return text;
    }


    /**
     * This function is responsible to replace the quotation with right and left
     * quotations.
     *
     * @param data
     *            The string in which the quotation has to be replaced.
     * @return The modified string
     */
    private String replaceQuotation(String data) {
        String newData = data;
        int count = 2;
        while (newData.indexOf("\"") > -1) {
            if ((count % 2) == 0) {
                newData = newData.replaceFirst("\"", "'");
            } else {
                newData = newData.replaceFirst("\"", "'");
            }
            count++;
        }
        return newData;
    }
}

package extractor;

import java.util.HashSet;

import org.apache.commons.lang.StringEscapeUtils;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.safety.Whitelist;
import org.jsoup.select.Elements;

import utils.MyFileWriter;
import utils.Utils;

/**
 * This class is implemented the parsing rules for retrieve useful information on the Snopes fact-check pages.
 * Each function retrieve one kind of information.
 * The results can retrieve with all the get fucntions.
 */

public class ClaimEvidenceExtractor {
    protected String url;
    protected String htmlContent;
    protected String claim="";
    protected String category;
    protected String subCategory;
    protected String headline;
    protected String description;
    protected String truthfulness="";
    protected HashSet<String> originalDocumentLinkSet = new HashSet<String>();
    protected String source;
    protected HashSet<String> evidenceSet = new HashSet<String>();
    protected String origin;
    protected String rulingOutline;
    protected Utils dkProUtils;
    protected String pageType="New";
    protected String serverURL;
    protected long offset;
    protected int length;
    protected MyFileWriter fileWriter;
    protected Document doc;


    public ClaimEvidenceExtractor(String htmlContent,String url,String serverURL,long offset,int length,String running_dir){
        this.htmlContent = htmlContent;
        this.url = url;
        this.dkProUtils = new Utils();
        this.serverURL = serverURL;
        this.offset = offset;
        this.length = length;
        this.fileWriter = new MyFileWriter(running_dir);
        doc = text2Document(htmlContent);
    }

    /**
     * convert the HTML content into a Jsoup document
     * @param htmlContent
     */
    public ClaimEvidenceExtractor(String htmlContent){
        this.htmlContent = htmlContent;
        doc = text2Document(htmlContent);
    }

    public ClaimEvidenceExtractor(){}

    public Document text2Document(String text){
        Document doc = Jsoup.parse(text);
        return doc;
    }


    protected String categoryExtractor(Document doc){
        Elements mainArticle =  doc.select("article");
        Elements category ;
        String newsCategory = "";
        try{
          category = mainArticle.first().previousElementSibling().select("a");
        }catch(Exception e){
            category=  doc.select("article > nav > div:nth-child(2) > a");
        }
   


        if (category.size()>0){
            newsCategory = stringNormalize(category.get(0).text(),true,true);
            return newsCategory;
        }
        return newsCategory;
    }

    protected String subCategoryExtractor(Document doc){
        Elements mainArticle =  doc.select("article");
        Elements category ;
        try{
              category = mainArticle.first().previousElementSibling().select("a");
        }catch(Exception e){
            category=  doc.select("article > nav > div:nth-child(3) > a");
        }
        String newsSubCategory="";
        if (category.size()>1){
            newsSubCategory = stringNormalize(category.get(1).text(),true,true);
            return newsSubCategory;
        }
        return newsSubCategory;
    }

    protected String headlineExtractor(Document doc){
         
        Elements mainArticle = doc.select("article");
        Elements header = mainArticle.first().select("header");
        String headline  = (header.first().getElementsByAttributeValue("itemprop","headline")).text();
        if (headline.length()<1 ){
            headline  = (header.first()).text();
             
             
        }

        if (headline.length()>0){
            headline = stringNormalize(headline,true,true);
            return headline;
        }else{
            return "";
        }
        
    }

    protected String descriptionExtractor(Document doc){
        Elements mainArticle = doc.select("article");
        Elements header = mainArticle.first().select("header");
        String description  = (header.first().getElementsByAttributeValue("itemprop","description")).text();
        if (description.length()>0){
            description = stringNormalize(description,true,true);
            return description;
        }
        return "";
    }

    private Elements extractClaimFromSpecialLocation(Document doc){
        Elements claimElement =doc.getElementsByAttributeValue("itemprop","claimReviewed");
        if (claimElement.text().length()>0){
            return claimElement;
        }else{
            claimElement = doc.select("div.claim-text.card-body");
            // System.out.println(claimElement.text());
        }
        return claimElement ;
    }
    protected String claimExtractor(Document doc){
        Elements articleBody = getArticleBody( doc);
        Elements claimPara = extractClaimFromSpecialLocation(doc);
//        System.out.println(articleBody);
        String claim = "";
        if (claimPara.text().length()>0){
            claim = claimPara.text();
        } else {
            Elements temArticleBody = articleBody;
            if (articleBody.select("div.article-text-inner").size()!= 0){
                temArticleBody = articleBody.select("div.article-text-inner");
            }
            Elements paraList = articleBody.select("> p");
            if (paraList.size() == 0) {
                paraList = temArticleBody.select("> div > p");
            }
            for (int i = 0; i < paraList.size(); i++) {
                String paraText = paraList.get(i).text();
//                System.out.println(paraText);
                if ((paraText).contains("Claim:")) {
                    if (paraList.get(i).select("noindex").size() > 0) {
                        if (paraText.contains("Status:")) {
                            claim = ((paraText.split("Status:"))[0])
                                    .replaceFirst("Claim:", "");
                        } else {
                            paraText =paraList.get(i).ownText();
                            claim = paraText.replaceFirst("Claim:", "");
                        }
                    } else {
                        claim = paraText.replaceFirst("Claim:", "");
                    }
                    break;
                }
            }
        }
        if (claim.length()==0){
            claimPara  = articleBody.first().children();
            for (Element e : claimPara){
                if (e.text().equals("CLAIM")){
                    claim = e.nextElementSibling().text();
                    break;
                }
            }

        }
        if (claim.length()==0){
            claimPara = articleBody.first().children();
            Element contentPara = claimPara.get(0);
            if (contentPara.children().size()==1){
                contentPara = claimPara.get(1);
            }

            Elements paraList = contentPara.children();
            for (Element p: paraList){
                if (p.tagName()=="h3" & p.text().contains("CLAIM")){
                    claim = p.nextElementSibling().text();
                }else if (p.text().contains("Claim") || p.text().contains("CLAIM")){
                    String paraText = p.ownText();
                    claim = paraText.replaceFirst(":","");
                }

            }
        }
        int index = claim.lastIndexOf(".");
        if (index>30){
            claim = claim.substring(0,claim.lastIndexOf(".")+1);
        }
        claim = claim.replace("See Example( s )","");
        claim = claim.replace("See Example( s )","");
        claim = stringNormalize(claim,true,true);
        return claim;
    }


    private Elements extractTruthfulnessParaFromSpecifiedLocation(Document doc){
        Elements truthfulnesPara=doc.getElementsByAttributeValue("itemprop","reviewRating");
        if (truthfulnesPara.text().length()>0){
            return truthfulnesPara;
        }else{
            truthfulnesPara = doc.select("div.media-body.d-flex.flex-column.align-self-center > span");
            return truthfulnesPara;
        }
        
    }
    protected String truthfulnessExtractor(Document doc){

        Elements articleBody = getArticleBody( doc);
        Elements truthfulnesPara = extractTruthfulnessParaFromSpecifiedLocation(doc);
        String sTruthfulness = "";
        if (truthfulnesPara.text().length()>0){
            sTruthfulness = truthfulnesPara.text();
        } else {

            Elements paraList = articleBody.select("> p");
            if (paraList.size() == 0) {
                paraList = articleBody.select("> div > p");
            }
            for (int i = 0; i < paraList.size(); i++) {
                String paraText = paraList.get(i).text();
                if (paraText.startsWith("Status:")) {
                    sTruthfulness = paraText.replaceFirst("Status:", "");
                    break;
                } else if ((paraText.startsWith("Example:")) || (paraText.startsWith("Origins:"))) {
                    break;
                }
            }
            if (sTruthfulness.length() < 1) {
                truthfulnesPara = articleBody.first().select("font.status_color");
                if (truthfulnesPara.hasText()) {
                    sTruthfulness = truthfulnesPara.first().text();
                } else {
                    truthfulnesPara = articleBody.first().select("div.claim-old");
                    if (truthfulnesPara.text().length() > 0) {
                        sTruthfulness = truthfulnesPara.text();
                    }
                }
            }
            //add new case to extract ratings for some usual cases.
            if (sTruthfulness.length()<1){
                truthfulnesPara = articleBody.select("noindex");
                sTruthfulness = truthfulnesPara.text();
            }
            if (sTruthfulness.length()<1){
                truthfulnesPara = articleBody.select("table");
                sTruthfulness = truthfulnesPara.text();
            }
        }
        if (sTruthfulness.length()<1){
            Elements ratingPara  = articleBody.first().children();
            for (Element e : ratingPara){
                if (e.text().equals("RATING")){
                    sTruthfulness = e.nextElementSibling().text();
                    break;
                }
            }
        }
        if (sTruthfulness.length()<1){
            Elements titles = doc.getElementsByTag("title");
            if(titles.text().contains(":")){
                String[] words = titles.text().split(":",2);
                if (words[0].contains(" ")){
                    sTruthfulness = "";
                }else {
                    sTruthfulness = words[0];
                }
            }
        }
        if (sTruthfulness.length()<1){
            for (Element e: articleBody){
                Elements  paraRating = e.getElementsByAttributeValueContaining("style","xx-large");
                if (paraRating.size()==0){
                    paraRating = e.getElementsByAttributeValueContaining("style","color: #1aa315; font-size: x-large;");
                }
                if (paraRating.size()==0){
                    paraRating = e.getElementsByAttributeValueContaining("style","color: #e51919; font-size: medium;");
                }
                if (paraRating.size()==0){
                    paraRating = e.getElementsByAttributeValueContaining("style","white-space: nowrap");
                }
                for (Element ele : paraRating){
                    sTruthfulness = ele.text();
                    break;
                }
            }

        }
        if (sTruthfulness.length()==0){
            Elements paras = articleBody.first().children();
            Element contentPara = paras.get(0);
            Elements paraList = contentPara.children();
            for (Element p: paraList){
                if (p.tagName()=="h3" & p.text().contains("RATING")){
                    sTruthfulness = p.nextElementSibling().text();
                }
            }
        }
        sTruthfulness = stringNormalize(sTruthfulness,true,true);
        return sTruthfulness;

    }

    protected HashSet<String> evidencesExtractor(Document doc){
        HashSet<String> evidences = new HashSet<String>();
        Elements articleBody = getArticleBody( doc);
        Elements temArticleBody = articleBody;
        if (articleBody.select("div.article-text-inner").size()!= 0){
            temArticleBody = articleBody.select("div.article-text-inner");
        }
//        System.out.println(temArticleBody);
        Elements evidenceList = temArticleBody.select("> blockquote");
        Elements evidenceListOld = temArticleBody.select("> div.quoteBlock").select("> blockquote");
        if (evidenceList.size()==0){
            evidenceList = temArticleBody.select("> div > blockquote");
        }
        if (evidenceList.size()>0 || evidenceListOld.size()>0){
            pageType = "New";
            for (Element evidence : evidenceList){
                if (evidence.text().length()>0){
//                    String newEvidence = stringNormalize(evidence.text().toString(),true,false);
                    String newEvidence = stringNormalize(evidence.toString(),true,false);

                    if (newEvidence.length()>0){
                        evidences.add(newEvidence);
                    }
                }
            }
            for (Element evidence : evidenceListOld){
                if (evidence.text().toString().length()>0){
//                    String newEvidence = stringNormalize(evidence.text().toString(),true,false);
                    String newEvidence = stringNormalize(evidence.toString(),true,false);

                    if (newEvidence.length()>0){
                        evidences.add(newEvidence);
                    }
                }
            }
        }else{
            evidenceList = articleBody.select("> div.quote_style");
            if (evidenceList.size()>0){
                pageType = "Old";
                for (Element evidence : evidenceList){
                    if (evidence.text().length()>0){
//                        String newEvidence = stringNormalize(evidence.text().toString(),true,false);
                        String newEvidence = stringNormalize(evidence.toString(),true,false);

                        if (newEvidence.length()>0){
                            evidences.add(newEvidence);
                        }
                    }
                }
            }
        }

        if (evidences.size() ==0){
            Elements evidenceListRecent = temArticleBody.select("> div.quote_style");
            for (Element evidence : evidenceListRecent){
                if (evidence.text().toString().length()>0){
//                    String newEvidence = stringNormalize(evidence.text().toLowerCase(),true,false);
                    String newEvidence = stringNormalize(evidence.toString(),true,false);
                    if (newEvidence.length()>0){
                        evidences.add(newEvidence);
                    }
                }
            }
        }

        return evidences;
    }

    protected String sourceExtractor(Document doc){
        Elements articleBody = getArticleBody( doc);
        Elements articleSourceBox = articleBody.first().getElementsByClass("article-sources-box");
        String sSources = "";
        if (articleSourceBox.size() > 0) {
            Elements articleSource = articleSourceBox.first().select("> p");
            for (int i = 0; i < articleSource.size(); i++) {
                sSources = articleSource.get(i).text() + "; " + sSources;
            }
        }
        if (sSources.length()==0) {
            articleSourceBox = articleBody.select("table");
            for (int i = 0; i < articleSourceBox.size(); i++) {
                if (articleSourceBox.get(i).text().toLowerCase().contains("sources")) {
                    Element sources = articleSourceBox.get(i).nextElementSibling();
                    while (sources.tagName().toLowerCase().equals("br")) {
                        sources = sources.nextElementSibling();
                    }
                    sSources = sources.text();
                }
            }
        }
        sSources = stringNormalize(sSources,true,true);
        return sSources;
    }

    protected Elements getArticleBody(Document doc){
        Elements articleBody =doc.select("div.article-text");
        if (articleBody.text().length()<1){
            articleBody=doc.select("main > article");
        }
        return articleBody;
    }

    protected String rulingOutlineExtactor(Document doc){
        return "";
    }
    
    protected String originExtactor(Document doc){
        String sOrigin = "";
        Elements articleBody = getArticleBody( doc);
        Element articleText = null;
        if (articleBody.size()==1){
            articleText = articleBody.first();
        }else {
            for (Element e : articleBody){
                if (e.text().contains("This article has been moved")){
                    continue;
                }else {

                    articleText = e;
                }
            }
            System.out.println(url+" has not only a article-text");
        }
        if (articleText!=null){
            Elements paras = articleText.children();
            boolean startExtractor = false;
            for (Element para : paras){

                if (para.tagName()=="h3" && para.text().contains("ORIGIN")){
                    startExtractor=true;
                    continue;
                }else if(para.tagName()=="h3" && para.text().contains("Origin")){
                    startExtractor=true;
                    continue;
                }

                if (para.text().contains("Origins:")){
                    startExtractor=true;
                }

                if (para.text().contains("Last updated:")){
                    break;
                }
                if (startExtractor) {
                    sOrigin += para.text();
                }

            }
            if (sOrigin.length()==0){
                Elements paragraphs = articleText.getElementsByAttributeValue("style","text-align: justify;");
                articleText = paragraphs.first();
                try {
                    Elements paraList = articleText.children();
                    if (paraList!=null){
                        paraList = articleText.children();
                        startExtractor = false;
                        for (Element  para : paraList){
                            if (para.text().contains("Origins:")){
                                startExtractor = true;
                            }else if (para.text().contains("Last updated:")){
                                break;
                            }
                            if (startExtractor){
                                sOrigin += para.text();
                            }

                        }
                    }
                }catch (Exception e){

                }
            }

        }

        if (sOrigin.length()==0){
            Elements paras = articleBody.first().children();
            Element contentPara = paras.get(0);
            Elements paraList = contentPara.children();
            boolean startExtractor = false;
            for (Element p: paraList){
                if (p.tagName()=="h3" & p.text().contains("ORIGIN")){
                    startExtractor = true;
                }
                if (p.text().contains("Origins:")){
                    startExtractor=true;
                }
                if (p.text().contains("Last updated:")){
                    break;
                }
                if (startExtractor) {
                    sOrigin += p.text();
                }
            }
        }

        if (sOrigin.contains("Origins:")){
            sOrigin = sOrigin.split(":",2)[1];
        }
        sOrigin = stringNormalize(sOrigin,true,true);
        return sOrigin;
    }



    protected HashSet<String> originalLinksExtractor(Document doc){
        HashSet<String> evidenceLinks = new HashSet<String>();
        Elements articleBody = getArticleBody( doc);
        Elements paraList = articleBody.select("> p");
        if (paraList.size()==0){
            paraList = articleBody.select("> div > p");
        }
        for (Element para : paraList){
            Elements anchors = para.select("a[href]");
            for (Element anchor : anchors){
                String link = anchor.attr("href");
                if (link.endsWith(".jpg") || link.endsWith(".txt") || link.endsWith(".pdf") || link.endsWith(".jpeg")||link.endsWith(".png"))
                    continue;
                if (link.contains("http:")|| link.contains("https:")) {
                    link = link.replaceAll("\\s", "");
                    link = link.replaceAll("\n", "");
                    evidenceLinks.add(link);
                }
            }
        }

        return evidenceLinks;
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
     * @param allTag
     *            Boolean value. 'true' indicates that all the HTML tags except
     *            paragraph and the blockquotes are removed, else all the tags
     *            are removed.
     * @return normalized string
     */
    protected String stringNormalize(String text, boolean tag, boolean allTag) {
        // Remove all anchor and img tags
        Whitelist whitelist;
        if (allTag) {
            if (pageType.equals("New")) {
                whitelist = new Whitelist().addTags("p", "blockquote");
                text = Jsoup.clean(text, whitelist);
            } else if (pageType.equals("Old")) {
                whitelist = new Whitelist().addTags("p");
                whitelist = whitelist.addAttributes("div", "class");
                text = Jsoup.clean(text, whitelist);
            }
        } else if (tag) {
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
//        text = text.replaceAll("\n","");

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
        text = text.replaceAll("\n", " ");
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
    protected String replaceQuotation(String data) {
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


    public String getUrl() {
        return url;
    }

    public String getClaim() {
        claim = claimExtractor(doc);
        return claim;
    }

    public String getCategory() {
        category = categoryExtractor(doc);
        return category;
    }

    public String getSubCategory() {
        subCategory = subCategoryExtractor(doc);
        return subCategory;
    }

    public String getHeadline() {
        headline = headlineExtractor(doc);
        return headline;
    }

    public String getDescription() {
        description = descriptionExtractor(doc);
        return description;
    }

    public String getTruthfulness() {
        truthfulness = truthfulnessExtractor(doc);
        return truthfulness;
    }

    public HashSet<String> getOriginalDocumentLinkSet() {
        originalDocumentLinkSet=originalLinksExtractor(doc);
        return originalDocumentLinkSet;
    }

    public String getSource() {
        source = sourceExtractor(doc);
        return source;
    }

    public HashSet<String> getEvidenceSet() {
        evidenceSet = evidencesExtractor(doc);
        return evidenceSet;
    }

    public String getRulingOutlineExtactor(){
        rulingOutline=rulingOutlineExtactor(doc);
        return rulingOutline;
    }

    public String getOrigin() {
        origin = originExtactor(doc);
        return origin;
    }

    public String getServerURL() {
        return serverURL;
    }

    public long getOffset() {
        return offset;
    }

    public int getLength() {
        return length;
    }

}

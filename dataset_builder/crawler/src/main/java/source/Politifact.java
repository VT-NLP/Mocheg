package source;
import org.jsoup.select.Elements;
public class Politifact extends Source{
    public Politifact(  ) {
        this.seed_url= "https://www.politifact.com/factchecks/";
        this.next_page_pattern= ".*factchecks/.page.*";
        this.name="Politifact";
    }

    public String extract_source_url(Elements elements_in_dict_page, int i){
        String url= elements_in_dict_page.get(i).select("div[class*=m-statement__quote]").select("a").attr("href");
        url="https://www.politifact.com" +url;
        return url;
    }
}

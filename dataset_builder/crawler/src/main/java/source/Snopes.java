package source;
import org.jsoup.select.Elements;
public class Snopes extends Source  {
    public Snopes(  ) {
        this.seed_url="https://www.snopes.com/fact-check/";
        this.next_page_pattern= ".+(fact-check/page)/.*";
        this.name="Snopes";
    }
    
    public String extract_source_url(Elements elements_in_dict_page, int i){
        return elements_in_dict_page.get(i).select("a").attr("href");
    }

}

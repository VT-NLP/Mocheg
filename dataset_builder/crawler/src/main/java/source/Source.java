package source;
import org.jsoup.select.Elements;

import enums.SourceEnum;
public class Source {
    public String seed_url;
    public String next_page_pattern;
    public String name;
    public String extract_source_url(Elements elements_in_dict_page, int i){
        return null;
    }


    public static Source get_source(String source_enum_str){
        SourceEnum sourceEnum=SourceEnum.valueOf(source_enum_str);
        return sourceEnum.source;
    }
}

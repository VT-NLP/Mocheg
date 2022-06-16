package enums;
import source.*;


public enum SourceEnum {
    Snopes  (new  Snopes()),   
    Politifact(new Politifact()  )   
    
    ;  


 
    public  Source source;
 
    private SourceEnum( Source source ) {
      
        this.source=source;
        
    }
}

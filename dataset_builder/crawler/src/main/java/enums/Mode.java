package enums;

import java.util.ArrayList;
import java.util.List;

public enum Mode {
    Mode1  ("get_evidence_by_url,annotation,get_relevant_document","mode1"),   
    Mode2("get_evidence_by_url,annotation","mode2"),  
    Mode3   ("get_evidence_from_scratch,get_relevant_document","mode3")    ,
    Mode4   ("get_evidence_from_scratch,get_relevant_document","mode4")    
    ;  


    public String[] phase_list;
    public String name;

    private Mode(String phase_list_str,String name) {
        this.name=name;
 
        this.phase_list=phase_list_str.split(",");
    }

    public boolean is_in(String in_phase){
        for (String phase: this.phase_list){
            if (phase==in_phase){
                return true;
            }
        }
        return false;
    }
}
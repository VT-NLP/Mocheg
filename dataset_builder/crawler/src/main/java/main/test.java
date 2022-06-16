package main;

import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

public class test {

    public static void main(String[] args) throws Exception {

//        String evidence = evidenceSplitting(a);
//        System.out.println(evidence);
        test2();
    }

    public static void test2() throws Exception{
        
        String running_dir="Results/run004_politifact_evidence/";
        App app = new App(running_dir,"Politifact",1);
         
        String mode="mode3";
        app.start(mode );

    }

    public static void test3(){
 
        Pattern pattern = Pattern.compile(".*factchecks/.page.*" );
        // Pattern pattern = Pattern.compile("w3schools", Pattern.CASE_INSENSITIVE );
        Matcher matcher = pattern.matcher("https://www.politifact.com/factchecks/?page=5&");
        boolean matchFound = matcher.find();
        if(matchFound) {
        System.out.println("Match found");
        } else {
        System.out.println("Match not found");
        }
    }

    public static void test1(){
        String a = "<p>Just because the test helped this man get a positive cancer diagnosis doesn't mean it's a reliable tool everyone should use, according to the American Cancer Society.The organization put the question to Ted Gansler, director of medical content, who wrote that 'only a small minority of men' with testicular cancer have HCG levels high enough to be detected by a home pregnancy test.He added that 'several non-cancerous conditions can cause false positive results.'</p> <p>According to Gansler, 'current evidence does not indicate that screening the general population of men with a urine test for HCG (or with urine or blood tests for any other tumour marker) can find testicular cancer early enough to reduce testicular cancer death rates.'</p> <p>One thing men can do is be on the lookout for lumps in the testicles and see your doctor if you find one. Testicle pain or swelling and heaviness or aching in the lower abdomen are also possible signs of testicular cancer.</p>";
        Matcher m = Pattern.compile("\\.[A-Z]").matcher(a);
        while (m.find()){
            a = a.replace(m.group(),". "+m.group().substring(1,2));
        }
        System.out.println(a);
    }

    public static String evidenceSplitting(String evidence){
        Reader stringReader = new StringReader(evidence);
        DocumentPreprocessor dpt = new DocumentPreprocessor(stringReader);
        ArrayList<String> sentences = new ArrayList<String>();
        for (List<HasWord> sentence : dpt){
            String sentenceString = sentence.get(0).toString();
            for (int j=1; j<sentence.size();j++){
                String tkn = sentence.get(j).toString();
                if ((tkn.equals("."))|| tkn.equals(",")||tkn.equals("!")||tkn.equals(":")||tkn.equals(";")){
                    sentenceString += tkn;
                }else {
                    sentenceString += " "+tkn;
                }
            }
            sentences.add(sentenceString);
        }
        String splitSents = new String();
        for (int i=0;i<sentences.size();i++){
            String sent = sentences.get(i);
            System.out.println(sent);
            if (sent.length()==0){
                continue;
            }
            splitSents += " "+i+"_{"+sent+"}";
        }
        return splitSents;
    }
}

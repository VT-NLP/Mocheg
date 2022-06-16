package utils;

import constants.Constants;

import java.io.*;

public class MyFileWriter {
    private BufferedReader bufferedReader = null;
    private Writer fileWriter = null;
    private String sLine = "";
    private String running_dir;


    public MyFileWriter(String running_dir) {
        File file = new File(System.getProperty("user.dir"));
        File resultDir = new File(file + "/Results");
        if (!resultDir.exists()){
            createDirectory();
        }
        this.running_dir=running_dir;


    }

    /**
     * This function is responsible to create the the Directory named 'Results'
     * which stores all the intermediate results of and also the Corpora.
     */
    public void createDirectory() {
        File resultDir = new File( running_dir+ Constants.RESULT_STORAGE_DIRECTORY);
        if (!resultDir.exists()) {
            resultDir.mkdir();
        }
    }

    /**
     * This function is used to create a new file.
     *
     * @param filename
     *            Name of the file to be created
     */
    public void createFile(String filename) {
        try {
            File fFile = new File( running_dir+ Constants.RESULT_STORAGE_DIRECTORY + filename);
            if (fFile.exists()) {
                fFile.delete();
            }
            fFile.createNewFile();
        } catch (Exception e) {
            System.err.println(
                    "Unable to create a file " + filename + " in the directory " +  running_dir+Constants.RESULT_STORAGE_DIRECTORY);
            e.printStackTrace();
        }
    }

    /**
     * This function is used to open the write connection
     *
     * @param filename
     *            Name of the file which needs to be updated
     */
    public void openWriteConnection(String filename) {
        File fFile = new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY + filename);
        try {
            fileWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fFile, true), "UTF8"));
        } catch (Exception e) {
            System.err.println("Unable to open write connection to the file " + filename);
            e.printStackTrace();
        }
    }

    /**
     * This function is used to close all the opened write connections
     */
    public void closeWriteConnection() {
        try {
            if (fileWriter != null) {
                fileWriter.close();
            }
        } catch (Exception ex) {
            System.err.println("Unable to close the write connections");
            ex.printStackTrace();
        }
    }

    /**
     * This function is used to update the file.
     *
     * @param data
     *            Line that has to be written to the file
     */
    public void writeLine(String data) {
        try {
            fileWriter.append(data);
            fileWriter.append("\n");
        } catch (Exception e) {
            System.err.println("Unable to write line to the file");
            e.printStackTrace();
        }
    }

}

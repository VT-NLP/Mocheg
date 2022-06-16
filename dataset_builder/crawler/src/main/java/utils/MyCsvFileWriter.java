package utils;

import au.com.bytecode.opencsv.CSVWriter;
import constants.Constants;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;

public class MyCsvFileWriter {


    private CSVWriter writer = null;


    /**
     * This function is responsible to create the the Directory named 'Results'
     * which stores all the intermediate results of and also the Corpora.
     */
    // public void createDirectory() {
    //     File file = new File(System.getProperty("user.dir"));
    //     File resultDir = new File(file + "/Results");
    //     if (!resultDir.exists()) {
    //         resultDir.mkdir();
    //     }
    //     running_dir+Constants.RESULT_STORAGE_DIRECTORY = resultDir.getPath();
    // }

    /**
     * This function is used to create a new file.
     *
     * @param filename
     *            Name of the file to be created
     */
    // public void createFile(String filename) {
    //     try {
    //         File fFile = new File(running_dir+Constants.RESULT_STORAGE_DIRECTORY + filename);
    //         if (fFile.exists()) {
    //             fFile.delete();
    //         }
    //         fFile.createNewFile();
    //     } catch (Exception e) {
    //         System.err.println(
    //                 "Unable to create a file " + filename + " in the directory " + running_dir+Constants.RESULT_STORAGE_DIRECTORY);
    //         e.printStackTrace();
    //     }
    // }

    /**
     * This function is used to open the write connection
     *
     * @param filename
     *            Name of the file which needs to be updated
     */
    public void openWriteConnection(String filename,String running_dir ) {//
        try {
            FileOutputStream fileStream = new FileOutputStream(new File( running_dir+Constants.RESULT_STORAGE_DIRECTORY+filename),true);
            OutputStreamWriter outputStream = new OutputStreamWriter(fileStream, "UTF-8");
            writer = new CSVWriter(outputStream,CSVWriter.DEFAULT_SEPARATOR,CSVWriter.DEFAULT_QUOTE_CHARACTER);
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
            if (writer != null) {
                writer.close();
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
    public void writeLine(String[] data) {
        try {
            writer.writeNext(data);
        } catch (Exception e) {
            System.err.println("Unable to write line to the file");
            e.printStackTrace();
        }
    }
}

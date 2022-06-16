package utils;


import java.io.*;
import constants.Constants;

public class MyFileReader {

    private BufferedReader bufferedReader = null;
    private String sLine = "";

    /**
     * This function is used to open a read connection to the file.
     *
     * @param filename
     *            Name of the file from which the data has to be read from
     */
    public void openReadConnection(String filename) {
        try {
            bufferedReader = new BufferedReader(new FileReader(filename));
        } catch (Exception e) {
            System.err.println("Unable to open read connection to the file " + filename);
            e.printStackTrace();
        }
    }

    /**
     * This function is used to close all the read connections.
     */
    public void closeReadConnection() {
        if (bufferedReader != null) {
            try {
                bufferedReader.close();
            } catch (Exception e) {
                System.err.println("unable to close the read connection");
                e.printStackTrace();
            }
        }
    }

    /**
     * This function is used to read a single line from the file.
     *
     * @return A single line read from the file
     */
    public String readLine() {
        try {
            sLine = bufferedReader.readLine();
            if (sLine != null) {
                return (sLine);
            }
        } catch (Exception e) {
            System.err.println("Unable to read lines from the file");
            e.printStackTrace();
        }
        return null;
    }
}

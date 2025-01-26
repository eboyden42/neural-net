package research.neuralnetwork.loader.heart;

import research.neuralnetwork.loader.BatchData;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.MetaData;
import research.neuralnetwork.loader.image.ImageBatchData;
import research.neuralnetwork.loader.image.ImageMetaData;
import research.neuralnetwork.loader.image.LoaderException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class CSVLoader implements Loader {

	private String fileName;
	private int batchSize;
	
	private Scanner scan;
	
	private CSVMetaData metaData;
	
	private double[][] trainingData;
	private double[] labelData;
	
	private int dataRows = 303;
	private int dataCols = 13;
	
	private Lock readLock = new ReentrantLock();
	
	public CSVLoader(String fileName, int batchSize) {  
		this.fileName = fileName;  
		this.batchSize = batchSize;  
	} 
	 
	@Override 
	public CSVMetaData open() {  
		
		try {  
			scan = new Scanner(new File(fileName)); 
		} catch (FileNotFoundException e) { 
			e.printStackTrace(); 
		} 
		
		scan.useDelimiter(",");  
		scan.nextLine();
		
		trainingData = new double[dataRows][dataCols];
		labelData = new double[dataRows];
		
		int row = 0;
		while (scan.hasNextLine()) {
			
			String stringLine = scan.nextLine();
			String[] line = stringLine.split(",");
			double[] data = new double[line.length];
			for (int i = 0; i < line.length; i ++) {
				if (i != 13) {
					trainingData[row][i] = Double.parseDouble(line[i]);
					//System.out.print(trainingData[row][i]+", ");
				}
				else {
					labelData[row] = Double.parseDouble(line[i]);
					//System.out.print("Expected: "+labelData[row]+", ");
				}
			}
			row ++;
			//System.out.println();
		}
		
		this.metaData = readMetaData();
		return metaData;
	}

	
	private CSVMetaData readMetaData() {
		metaData = new CSVMetaData();
		
		int numberItems = dataRows;
		
		
		metaData.setNumberItems(numberItems);
			
		metaData.setInputSize(dataCols);
		
		metaData.setExpectedSize(2);
		
		int numberBatches = (int) Math.ceil((double)numberItems / batchSize);
		
		metaData.setNumberBatches(numberBatches);
		
		return metaData;
	}
	
	@Override
	public void close() {
		scan.close();
	}

	@Override
	public CSVMetaData getMetaData() {
		return metaData;
	}

	@Override
	public CSVBatchData readBatch() {
		
		readLock.lock();
		try {
			CSVBatchData batchData = new CSVBatchData();
			
			int inputItemsRead = readInputBatch(batchData);
			int expectedItemsRead = readExpectedBatch(batchData);
			
			if (inputItemsRead != expectedItemsRead) {
				//throw new LoaderException("The number of images read does not equal the number of labels read");
			}
			
			metaData.setItemsRead(expectedItemsRead);
			
			return batchData;
		}
		finally {
			readLock.unlock();
		}
		
	}

	private int readExpectedBatch(CSVBatchData batchData) {
		//try {
			var totalItemsRead = metaData.getTotalItemsRead();
			var numberItems = metaData.getNumberItems();
			var numberToRead = Math.min(numberItems - totalItemsRead, batchSize);
			var expectedSize = metaData.getExpectedSize();
			
			var numberRead = numberToRead;
			
			if (numberRead != numberToRead) {
				throw new LoaderException("Couldn't read sufficient bytes from image data");
			}
			
			double[] data = new double[numberToRead*expectedSize];
			
			for (int i = 0; i < numberToRead; i ++) {
				double label = labelData[totalItemsRead + i];
				
				if (label == 1.0) {
					data[i*expectedSize] = 1;
				}
				else {
					data[i*expectedSize+1] = 1;
				}
			}
				
			batchData.setExpectedBatch(data);
			
			return numberToRead;
		//}
		//catch (IOException e) {
			//throw new LoaderException("Error occured reading image data ", e);
		//}
	}

	private int readInputBatch(CSVBatchData batchData) {
		var totalItemsRead = metaData.getTotalItemsRead();
		var numberItems = metaData.getNumberItems();
		var numberToRead = Math.min(numberItems - totalItemsRead, batchSize);
	
		var inputSize = metaData.getInputSize();
		var numberItemsToRead = numberToRead*inputSize;
	
		double[] data = new double[numberItemsToRead];
	
		var numberRead = numberToRead;
		
		if (numberRead != numberToRead) {
			throw new LoaderException("Couldn't read sufficient bytes from image data");
		}
		
		for (int row = 0; row < numberToRead; row ++) {
			double[] oneBatch = trainingData[(totalItemsRead+row)];
			for (int i = 0; i < oneBatch.length; i ++) {
				data[row*oneBatch.length+i] = oneBatch[i];
			}
		}
		
		batchData.setInputBatch(data);
		
		return numberToRead;
	}

	
	public double[][] getTrainingData() {
		return trainingData;
	}
	
	public double[] getLabelData() {
		return labelData;
	}
	
	public static void main(String[] args) {
		CSVLoader c = new CSVLoader("heart.csv", 32);
		c.open();
		MetaData m = c.getMetaData();
		System.out.println(m.getExpectedSize());
		for (int n = 0; n < 9; n ++) {
		CSVBatchData b = c.readBatch();
		double[] expected = b.getInputBatch();
		for (int i = 0; i < expected.length; i ++) {
			System.out.print(expected[i]+", ");
			if ((i+1) % 13 == 0) {
				System.out.println();
			}
			
		}
		}
		
	}

}

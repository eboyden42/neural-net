package research.neuralnetwork.loader.image;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import research.neuralnetwork.loader.BatchData;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.MetaData;

public class ImageLoader implements Loader {
	private String imageFileName;
	private String labelFileName;
	private int batchSize;
	
	private DataInputStream dsImages;
	private DataInputStream dsLabels;
	
	private ImageMetaData metaData;
	
	private Lock readLock = new ReentrantLock();
	
	public ImageLoader(String imageFileName, String lableFileName, int batchSize) {
		this.imageFileName = imageFileName;
		this.labelFileName = lableFileName;
		this.batchSize = batchSize;
	}

	@Override
	public ImageMetaData open() {
		try {
			dsImages = new DataInputStream(new FileInputStream(imageFileName));
		} 
		catch (Exception e) {
			throw new LoaderException("Cannot Open "+imageFileName, e);
		}
		
		try {
			dsLabels = new DataInputStream(new FileInputStream(labelFileName));
		} 
		catch (Exception e) {
			throw new LoaderException("Cannot Open "+labelFileName, e);
		}
		
		metaData = readMetaData();
		return metaData;
	}

	private ImageMetaData readMetaData() {
		
		metaData = new ImageMetaData();
		
		int numberItems;
		
		try {
			int magicLabelNumber = dsLabels.readInt();
			if (magicLabelNumber != 2049) {
				throw new LoaderException("Label file "+labelFileName+"has wrong format");
			}
			
			numberItems = dsLabels.readInt();
			metaData.setNumberItems(numberItems);
			
		} catch (IOException e) {
			throw new LoaderException("Unable to read MetaData from "+labelFileName, e);
		}
		
		try {
			int magicImageNumber = dsImages.readInt();
			if (magicImageNumber != 2051) {
				throw new LoaderException("Label file "+imageFileName+" has wrong format");
			}
			
			if (dsImages.readInt() != numberItems) {
				throw new LoaderException("Image File "+imageFileName+" has different number of items than "+labelFileName);
			}
			
			int height = dsImages.readInt();
			int width = dsImages.readInt();
			
			metaData.setHeight(height);
			metaData.setWidth(width);
			metaData.setInputSize(width*height);
			
		} catch (IOException e) {
			throw new LoaderException("Unable to read MetaData from "+imageFileName, e);
		}
		
		metaData.setExpectedSize(10);
		metaData.setNumberBatches((int) Math.ceil((double)numberItems / batchSize));
		
		return metaData;
	}
	
	@Override
	public void close() {
		
		metaData = null;
		
		try {
			dsImages.close();
		}
		catch(Exception e) {
			throw new LoaderException("Cannot close "+imageFileName, e);
		}
		
		try {
			dsLabels.close();
		}
		catch (Exception e) {
			throw new LoaderException("Cannot close "+labelFileName, e);
		}
	}

	@Override
	public ImageMetaData getMetaData() {
		return metaData;
	}

	@Override
	public BatchData readBatch() {
		readLock.lock();
		try {
			ImageBatchData batchData = new ImageBatchData();
			
			int inputItemsRead = readInputBatch(batchData);
			int expectedItemsRead = readExpectedBatch(batchData);
			
			if (inputItemsRead != expectedItemsRead) {
				throw new LoaderException("The number of images read does not equal the number of labels read");
			}
			
			metaData.setItemsRead(inputItemsRead);
			
			return batchData;
		}
		finally {
			readLock.unlock();
		}
		
	}

	private int readExpectedBatch(ImageBatchData batchData) {
		try {
			var totalItemsRead = metaData.getTotalItemsRead();
			var numberItems = metaData.getNumberItems();
			var numberToRead = Math.min(numberItems - totalItemsRead, batchSize);
		
			var labelData = new byte[numberToRead];
			var expectedSize = metaData.getExpectedSize();
			
			var numberRead = dsLabels.read(labelData, 0, numberToRead);
			
			if (numberRead != numberToRead) {
				throw new LoaderException("Couldn't read sufficient bytes from image data");
			}
			
			double[] data = new double[numberToRead*expectedSize];
			
			for (int i = 0; i < numberToRead; i ++) {
				byte label = labelData[i];
				
				data[i*expectedSize + label] = 1;
			}
				
			batchData.setExpectedBatch(data);
			
			return numberToRead;
		}
		catch (IOException e) {
			throw new LoaderException("Error occured reading image data ", e);
		}
	}

	private int readInputBatch(ImageBatchData batchData) {
		try {
			var totalItemsRead = metaData.getTotalItemsRead();
			var numberItems = metaData.getNumberItems();
			var numberToRead = Math.min(numberItems - totalItemsRead, batchSize);
		
			var inputSize = metaData.getInputSize();
			var numberBytesToRead = numberToRead*inputSize;
		
			byte[] imageData = new byte[numberBytesToRead];
		
			var numberRead = dsImages.read(imageData, 0, numberBytesToRead);
			
			if (numberRead != numberBytesToRead) {
				throw new LoaderException("Couldn't read sufficient bytes from image data");
			}
			
			double[] data = new double[numberBytesToRead];
			
			for (int i = 0; i < numberBytesToRead; i++) {
				data[i] = (imageData[i] & 0xFF)/256.0;
			}
			
			batchData.setInputBatch(data);
			
			return numberToRead;
		}
		catch (IOException e) {
			throw new LoaderException("Error occured reading image data ", e);
		}
	}
	
	
}

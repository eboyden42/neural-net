package research.neuralnetwork;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.LinkedList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import research.matrix.Matrix;
import research.neuralnetwork.loader.BatchData;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.MetaData;

public class NeuralNetwork implements Serializable {
	private static final long serialVersionUID = 1L;

	private Engine engine;
	
	private int epochs = 20;
	private double initalLearningRate = 0.01;
	private double finalLearningRate = 0.001;
	private int threads = 2;
	
	private double finalLoss = 0.0;
	private double finalPercentCorrect = 0.0;
	
	private transient double learningRate;
	private transient Object lock = new Object();
	
	public NeuralNetwork() {
		engine = new Engine();
	}
	
	public void setThreads(int threads) {
		this.threads = threads;
	}
	
	public void setScaleInitalWeights(double scale) {
		engine.setScaleInitalWeights(scale);
	}
	
	public void add(Transform transform, double...params) {
		engine.add(transform, params);
	}

	
	public void setLearningRates(double initalRate, double finalRate) {
		initalLearningRate = initalRate;
		finalLearningRate = finalRate;
	}
	
	public double getLoss() {
		return finalLoss;
	}
	
	public double getPercentCorrect() {
		return finalPercentCorrect;
	}
	
	public void setEpochs(int epochs) {
		this.epochs = epochs;
	}
	
	public double[] predict(double[] inputData) {
		Matrix input = new Matrix(inputData.length, 1, i -> inputData[i]);
		
		BatchResult batchResult = engine.runForwards(input);
		return batchResult.getOutput().get();
	}
	
	public void fit(Loader trainLoader, Loader evalLoader) {
		learningRate = initalLearningRate;
		
		for (int epoch = 0; epoch < epochs; epoch ++) {
			System.out.printf("Epoch %3d ", epoch+1);
			
			long start = System.currentTimeMillis();
			runEpoch(trainLoader, true);
			
			if (evalLoader != null) {
				runEpoch(evalLoader, false);
			}
			
			long end = System.currentTimeMillis();
			double time = ((end - start)/1000.0)/60.0;
			double timeRemaining = time*(epochs - (epoch+1));
			
			System.out.printf(" | Estimated Time Remaining: %.2f m | Learning Rate: %.6f", timeRemaining, learningRate);
			
			System.out.println();
			learningRate -= (initalLearningRate - finalLearningRate)/epochs;
		}
	}
	
	private void runEpoch(Loader loader, boolean trainingMode) {
		loader.open();
		
		var queue = createBatchTasks(loader, trainingMode);
		consumeBatchTasks(queue, trainingMode);
		
		loader.close();
	}

	private void consumeBatchTasks(LinkedList<Future<BatchResult>> batches, boolean trainingMode) {
		
		var numberBatches = batches.size();
		int index = 0;
		
		double averageLoss = 0.0;
		double averagePercentCorrect = 0.0;
		
		for (var batch : batches) {
			try {
				var batchResult = batch.get();
				if (!trainingMode) {
					averageLoss += batchResult.getLoss();
					averagePercentCorrect += batchResult.getPercentCorrect();
				}
			} catch (Exception e) {
				throw new RuntimeException("Execution Error: ", e);
			}
			/**
			 * dotNumber should vary based on the size of the data set;
			 * Never set dotNumber greater than numberBatches
			 */
			int dotNumber = 30;
			
			int printDot = numberBatches/dotNumber;
			if (trainingMode && index ++ % printDot == 0) {
				System.out.print(".");
			}
		}
		
		if (!trainingMode) {
			averageLoss /= batches.size();
			averagePercentCorrect /= batches.size();
			
			finalLoss = averageLoss;
			finalPercentCorrect = averagePercentCorrect;
			
			System.out.printf("Loss: %.3f — Percent Correct %.2f", averageLoss, averagePercentCorrect);
		}	
	}

	private LinkedList<Future<BatchResult>> createBatchTasks(Loader loader, boolean trainingMode) {
		LinkedList<Future<BatchResult>> batches = new LinkedList<>();
		
		MetaData metaData = loader.getMetaData();
		int numberBatches = metaData.getNumberBatches();
		
		var executor = Executors.newFixedThreadPool(threads);
		
		for (int i = 0; i < numberBatches; i ++) {
			batches.add(executor.submit(()->runBatch(loader, trainingMode)));
		}
		
		executor.shutdown();
		
		return batches;
	}

	private BatchResult runBatch(Loader loader, boolean trainingMode) {
		MetaData metaData = loader.getMetaData();
		BatchData batchData = loader.readBatch();
		int itemsRead = metaData.getItemsRead();
		int inputSize = metaData.getInputSize();
		int expectedSize = metaData.getExpectedSize();
		
		Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
		Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());
		
		BatchResult batchResult = engine.runForwards(input);
		
		if(trainingMode) {
			engine.runBackwards(batchResult, expected);
			
			synchronized (lock) {
				engine.adjust(batchResult, learningRate);
			}
		}
		else {
			engine.evaluate(batchResult, expected);
		}
		
		return batchResult;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		sb.append("Network Configuration Data:\n");
		sb.append("---------------------------\n");
		sb.append(String.format("Epochs: %d\n", epochs));
		sb.append(String.format("Inital Learning Rate: %.5f\n", initalLearningRate));
		sb.append(String.format("Final Learning Rate : %.5f\n", finalLearningRate));
		sb.append(String.format("Thread Count: %d\n", threads));
		
		sb.append("\nEngine Configuration Data:\n");
		sb.append("--------------------------\n");
		sb.append(engine);
		
		return sb.toString();
	}

	public boolean save(String file) {
		try (var ds = new ObjectOutputStream(new FileOutputStream(file))) {
			ds.writeObject(this);
		}
		catch(IOException e) {
			System.err.println("Unable to be saved to "+file);
			return false;
		}
		return true;
	}
	
	public static NeuralNetwork load(String file) {
		
		NeuralNetwork network = null;
		
		try (var ds = new ObjectInputStream(new FileInputStream(file))) {
			network = (NeuralNetwork)ds.readObject();
		}
		catch(Exception e) {
			System.err.println("Unable to load from "+file);
		}
		return network;
	}
	
	public Object readResolve() {
		lock = new Object();
		return this;
	}
	
}

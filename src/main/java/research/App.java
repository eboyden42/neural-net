package research;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import research.neuralnetwork.NeuralNetwork;
import research.neuralnetwork.Transform;
import research.neuralnetwork.loader.BatchData;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.MetaData;
import research.neuralnetwork.loader.image.ImageLoader;
import research.neuralnetwork.loader.test.TestLoader;

public class App {

	public static void main(String[] args) {
		
		final String filename = String.format("mnistNeural36x16.ntw");
		
		if (args.length == 0) {
			System.out.println("usage: [app] <MNIST DATA DIRECTORY>");
			return;
		}

		String directory = args[0];
		
		if (!new File(directory).isDirectory()) {
			System.out.println("'"+directory+"'"+" is not a directory");
			return;
		}
		
		
		final String trainImages = String.format("%s%s%s", directory, File.separator, "train-images-idx3-ubyte");
		final String trainLabels = String.format("%s%s%s", directory, File.separator, "train-labels-idx1-ubyte");
		final String testImages = String.format("%s%s%s", directory, File.separator, "t10k-images-idx3-ubyte");
		final String testLabels = String.format("%s%s%s", directory, File.separator, "t10k-labels-idx1-ubyte");
		
		System.out.println(trainImages);
		
		Loader trainLoader = new ImageLoader(trainImages, trainLabels, 32);
		Loader testLoader = new ImageLoader(testImages, testLabels, 32);
		
		MetaData metaData = trainLoader.open();
		int inputSize = metaData.getInputSize();
		int outputSize = metaData.getExpectedSize();
		trainLoader.close();
		
		
				
		NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);
				
		if (neuralNetwork == null) {
			System.out.println("Unable to load network from save, Creating from scratch...");
					
			int epochNumber = 100;
					
			neuralNetwork = new NeuralNetwork();
			
			neuralNetwork.setEpochs(epochNumber);
			neuralNetwork.setLearningRates(0.02, 0.01);
			neuralNetwork.setThreads(12);
			neuralNetwork.setScaleInitalWeights(0.2);
			
			neuralNetwork.add(Transform.DENSE, 36, inputSize);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, 16);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, outputSize);
			neuralNetwork.add(Transform.SOFTMAX);
					
		}
		else {
			System.out.println("Loaded from "+filename);
		}
		
		System.out.println(neuralNetwork);
		
		neuralNetwork.fit(trainLoader, testLoader);
		
		neuralNetwork.save(filename);
		
		/**
		StringBuilder sb = new StringBuilder().append(String.format("%s: 100EpochLoss: %.3f - 100EpochPC: %.2f \n", filename, neuralNetwork.getLoss(), neuralNetwork.getPercentCorrect()));
		try {
			FileWriter fw = new FileWriter("NetworkTrainingData.txt", true);
			fw.write(sb.toString());
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		}
		**/
	}

}

package research;

import java.io.File;

import research.neuralnetwork.NeuralNetwork;
import research.neuralnetwork.Transform;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.MetaData;
import research.neuralnetwork.loader.heart.CSVLoader;
import research.neuralnetwork.loader.image.ImageLoader;

public class App2 {

	public static void main(String[] args) {
		final String filename = String.format("heartNeural250.ntw");
		
		/**
		if (args.length == 0) {
			System.out.println("usage: [app] <MNIST DATA DIRECTORY>");
			return;
		}

		String directory = args[0];
		
		if (!new File(directory).isDirectory()) {
			System.out.println("'"+directory+"'"+" is not a directory");
			return;
		}
		
		**/
		final String heartFileName = "heart.csv";
		
		System.out.println(heartFileName);
		
		Loader trainLoader = new CSVLoader(heartFileName, 32);
		Loader testLoader = new CSVLoader(heartFileName, 32);
		
		MetaData metaData = trainLoader.open();
		int inputSize = metaData.getInputSize();
		int outputSize = metaData.getExpectedSize();
		trainLoader.close();
		
		
				
		NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);
				
		if (neuralNetwork == null) {
			System.out.println("Unable to load network from save, Creating from scratch...");
					
			int epochNumber = 10000;
					
			neuralNetwork = new NeuralNetwork();
			
			neuralNetwork.setEpochs(epochNumber);
			neuralNetwork.setLearningRates(0.001, 0.00001);
			neuralNetwork.setThreads(12);
			neuralNetwork.setScaleInitalWeights(0.01);
			
			neuralNetwork.add(Transform.DENSE, 250, inputSize);
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

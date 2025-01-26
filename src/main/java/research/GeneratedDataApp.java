package research;
import research.neuralnetwork.*;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.test.TestLoader;

public class GeneratedDataApp {

	public static void main(String[] args) {
		//System.out.println(Runtime.getRuntime().availableProcessors());
		
		String filename = "network1.net";
		
		NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);
		
		if (neuralNetwork == null) {
			System.out.println("Unable to load network from save, Creating from scratch...");
			
			int inputRows = 500;
			int outputRows = 3;
			int epochNumber = 20;
			
			neuralNetwork = new NeuralNetwork();
			neuralNetwork.add(Transform.DENSE, 100, inputRows);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, 50);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, outputRows);
			neuralNetwork.add(Transform.SOFTMAX);
			
			neuralNetwork.setEpochs(epochNumber);
			neuralNetwork.setLearningRates(0.02, 0.001);
			neuralNetwork.setThreads(12);
			neuralNetwork.setScaleInitalWeights(1.0);
		}
		else {
			System.out.println("Loaded from "+filename);
		}
		
		
		
		System.out.println(neuralNetwork);
		
		Loader trainLoader = new TestLoader(60_000, 32);
		Loader testLoader = new TestLoader(10_000, 32);
		
		neuralNetwork.fit(trainLoader, testLoader);
		
		neuralNetwork.save(filename);
		
	}

}

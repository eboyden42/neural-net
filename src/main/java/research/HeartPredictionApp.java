package research;

import research.neuralnetwork.NeuralNetwork;
import research.neuralnetwork.loader.heart.CSVLoader;

public class HeartPredictionApp {

	public static void main(String[] args) {
		String fileName = "heartNeural24.ntw";
		
		NeuralNetwork heart24 = NeuralNetwork.load(fileName);
		
		System.out.println(fileName);
		System.out.println(String.format("Average Percent Correct: %.4f", heart24.getPercentCorrect()));
		System.out.println(String.format("Average Loss: %.4f", heart24.getLoss()));
		
		CSVLoader loader = new CSVLoader("heart.csv", 32);
		loader.open();
		double[][] train = loader.getTrainingData();
		double[] label = loader.getLabelData();
		
		for (int i = 0; i < train.length; i ++) {
			double[] predicted = heart24.predict(train[i]);
			double first = 0.0;
			double prediction = 0.0;
			double confidence = 0.0;
			for (int n = 0; n < predicted.length; n ++) {
				//System.out.print(String.format("%.2f, ", predicted[n]));
				if (n == 0) {
					first = predicted[n];
				}
				else {
					if (first > predicted[n]) {
						System.out.print("Network Prediction: "+1.0+", ");
						prediction = 1.0;
						confidence = predicted[n-1];
					}
					else {
						System.out.print("Network Prediction: "+0.0+", ");
						prediction = 0.0;
						confidence = predicted[n];
					}
				}
			}
			System.out.println(String.format("Actual: %.2f | Confidence: %.2f <<%b>>", label[i], confidence, prediction == label[i]));
		}

	}

}

package research;

import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import research.neuralnetwork.NeuralNetwork;
import research.neuralnetwork.Transform;
import research.neuralnetwork.loader.BatchData;
import research.neuralnetwork.loader.Loader;
import research.neuralnetwork.loader.MetaData;
import research.neuralnetwork.loader.image.ImageLoader;
import research.neuralnetwork.loader.image.ImageWriter;
import research.neuralnetwork.loader.test.TestLoader;

public class App {

	public static void main(String[] args) {

		String directory = "MNISTdata/MNIST";
		if (args.length > 0) {
			directory = args[0];
		}
		
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

		Scanner scan = new Scanner(System.in);
		System.out.println("Enter the name of your network, if the file cannot be found a new network will be created:");
		final String filename = scan.nextLine()+".ntw";
				
		NeuralNetwork neuralNetwork = NeuralNetwork.load(filename);
				
		if (neuralNetwork == null) {
			System.out.println("Unable to load network from save, Creating from scratch...");

			//TRAINING SETTINGS - settings are set to default values but can be changed manually
			int epochNumber = 100;
			double initialLearningRate = 0.02;
			double finalLearningRate = 0.01;
			int numberOfThreads = 12; //you may want to change this based on your computer (so the network trains faster), nothing will break if you don't
			final double scaleInitialWeights = 0.2; //DO NOT CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING

			neuralNetwork = new NeuralNetwork();
			neuralNetwork.setEpochs(epochNumber);
			neuralNetwork.setLearningRates(initialLearningRate, finalLearningRate);
			neuralNetwork.setThreads(numberOfThreads);
			neuralNetwork.setScaleInitalWeights(scaleInitialWeights);

			//NETWORK ARCHITECTURE SETTINGS - determined dynamically upon program run
			String line = "";
			System.out.println("Do you want to use a default network architecture? (Y/N)");
			line = scan.nextLine();
			while(!line.equals("Y") && !line.equals("N")) {
				System.out.println("Must enter Y or N. Do you want to use a default network architecture?");
				line = scan.nextLine();
			}
			if (line.equals("Y")) {
				neuralNetwork.add(Transform.DENSE, 36, inputSize);
				neuralNetwork.add(Transform.RELU);
				neuralNetwork.add(Transform.DENSE, 16);
				neuralNetwork.add(Transform.RELU);
				neuralNetwork.add(Transform.DENSE, outputSize);
				neuralNetwork.add(Transform.SOFTMAX);
			}
			else {
				System.out.println("Enter the number of layers you would like in your network (typically 1-5):");
				line = scan.nextLine();
				while(!isInteger(line)) {
					System.out.println("Must enter an integer.");
					line = scan.nextLine();
				}
				int numberOfLayers = Integer.parseInt(line);

				System.out.println("Enter the number of nodes you would like in layer 1. (typically 5-300, descending with each layer):");
				line = scan.nextLine();
				while(!isInteger(line)) {
					System.out.println("Must enter an integer.");
					line = scan.nextLine();
				}
				int nodes = Integer.parseInt(line);
				neuralNetwork.add(Transform.DENSE, nodes, inputSize);
				neuralNetwork.add(Transform.RELU);

				for (int i = 1; i < numberOfLayers; i ++) {
					System.out.printf("Enter the number of nodes you would like in layer %d. (typically 5-300, descending with each layer):\n", i+1);
					line = scan.nextLine();
					while(!isInteger(line)) {
						System.out.println("Must enter an integer.");
						line = scan.nextLine();
					}
					nodes = Integer.parseInt(line);
					neuralNetwork.add(Transform.DENSE, nodes);
					neuralNetwork.add(Transform.RELU);
				}

				//append output layer
				neuralNetwork.add(Transform.DENSE, outputSize);
				neuralNetwork.add(Transform.SOFTMAX);
			}
					
		}
		else {
			System.out.println("Loaded from "+filename);
			String line = "";
			System.out.println("Do you want to train this loaded network further? (Y/N)");
			line = scan.nextLine();
			while(!line.equals("Y") && !line.equals("N")) {
				System.out.println("Must enter Y or N. Do you want to use a default network architecture?");
				line = scan.nextLine();
			}
			if (line.equals("N")) {
				System.out.println("Writing images to files...");
				new ImageWriter().run(directory, filename);
				return;
			}
		}

		//for newly generated network
		System.out.println(neuralNetwork);
		neuralNetwork.fit(trainLoader, testLoader);
		neuralNetwork.save(filename);
		System.out.println("Writing images to files...");
		new ImageWriter().run(directory, filename);

		//debug code
		/**
		StringBuilder sb = new StringBuilder().append(String.format("%s: 100EpochLoss: %.3f - 100EpochPC: %.2f \n", filename, neuralNetwork.getLoss(), neuralNetwork.getPercentCorrect()));
		try {
			FileWriter fw = new FileWriter("NetworkTrainingData.txt", true);
			fw.write(sb.toString());
			fw.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		}
		**/
	}

	public static boolean isInteger(String input) {
		try {
			Integer.parseInt(input);
			return true;
		} catch (NumberFormatException e) {
			return false;
		}
	}
}

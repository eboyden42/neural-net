package research.neuralnetwork;
import static org.junit.Assert.*;
import org.junit.Test;
import research.matrix.Matrix;
import java.util.Random;
import java.util.ArrayList;
import java.io.FileOutputStream;

public class NeuralNetTest {
	
	private Random random = new Random(); 
	
	@Test
	public void testTrainEngine() {
		int inputRows = 500;
		int cols = 32;
		int outputRows = 3;
		
		Engine engine = new Engine();
		engine.add(Transform.DENSE, 100, inputRows);
		engine.add(Transform.RELU);
		engine.add(Transform.DENSE, outputRows);
		engine.add(Transform.SOFTMAX);
		
		ArrayList<Double> percentageOverTime = new ArrayList<Double>();
		ArrayList<Double> lossOverTime = new ArrayList<Double>();
		
		RunningAverages runningAverages = new RunningAverages(2, 500, (callnumber, averages) -> {
			assertTrue(averages[0] < 6.0);
			System.out.printf("%d. Loss: %.3f -- Percent correct: %.2f\n", callnumber, averages[0], averages[1]);
		});
		
		int iterations = 500;
		double initalLearningRate = 0.02;
		double learningRate = initalLearningRate;
		
		for (int i = 0; i <= iterations; i ++) {
			var tm = Util.generateTrainingMatrixes(inputRows, outputRows, cols);
			var input = tm.getInput();
			var expected = tm.getOutput();
			
			BatchResult b = engine.runForwards(input);
			engine.runBackwards(b, expected);
			engine.adjust(b, learningRate);
			engine.evaluate(b, expected);
			double loss = b.getLoss();
			runningAverages.add(loss, b.getPercentCorrect());
			
			learningRate -= initalLearningRate/iterations;
		}
		
		/*
		double loss = 500.0;
		double percent = 0.0;
		int batchNumber = 0;
		while (percent < 99.0) {
			BatchResult b = engine.runForwards(input);
			engine.evaluate(b, expected);
			loss = b.getLoss();
			lossOverTime.add(loss);
			percent = b.getPercentCorrect();
			percentageOverTime.add(percent);
			engine.runBackwards(b, expected);
			engine.adjust(b, 0.01);
			System.out.println(b.getOutput());
			System.out.println(expected);
			System.out.println("Loss: "+loss);
			System.out.println("Percent Correct: "+percent);
			batchNumber ++;
		}
		*/
	}
	
	@Test
	public void testWeightGradient() {
		int inputRows = 4;
		int outputRows = 5;
		
		Matrix weights = new Matrix(outputRows, inputRows, i -> random.nextGaussian());
		Matrix input = Util.generateInputMatrix(inputRows, 1);
		Matrix expected = Util.generateExpectedMatrix(outputRows, 1);
		
		Matrix output = weights.multiply(input).softmax();
		
		Matrix loss = LossFunctions.crossEntropy(expected, output);
		Matrix calculatedError = output.apply((index, value) -> value-expected.get(index));
		
		Matrix calculatedWeightError = calculatedError.multiply(input.transpose());
		
		Matrix approximatedWeightError = Approximator.weightGradient(weights, w -> {
			Matrix mult = w.multiply(input).softmax();
			return LossFunctions.crossEntropy(expected, mult);
		});
		
		assertTrue(calculatedWeightError.equals(approximatedWeightError));
		
	}
	
	@Test
	public void testEngine() {
		Engine engine = new Engine();
			
		int inputRows = 5;
		int cols = 3;
		int outputRows = 4;
		
		
		engine.add(Transform.DENSE, 8, 5);
		engine.add(Transform.RELU);
		engine.add(Transform.DENSE, 5);
		engine.add(Transform.RELU);
		engine.add(Transform.DENSE, 4);
		
		engine.add(Transform.SOFTMAX);
		engine.setStoreInputError(true);
		
		Matrix input = Util.generateInputMatrix(inputRows, cols);
		Matrix expected = Util.generateExpectedMatrix(outputRows, cols);
		
		Matrix approximatedError = Approximator.gradient(input, in->{
			BatchResult batchResult = engine.runForwards(in);
			return LossFunctions.crossEntropy(expected, batchResult.getOutput());
		});
		
		BatchResult output = engine.runForwards(input);
		engine.runBackwards(output, expected);
		
		Matrix calculatedError = output.getInputError();
		
		calculatedError.setTolerance(0.01);
		assertTrue(calculatedError.equals(approximatedError));
	}
	
	@Test
	public void testBackprop() {
		final boolean printValues = false;
		
		interface NeuralNet {
			Matrix apply(Matrix m);
		}
		
		final int inputRows = 4;
		final int cols = 5;
		final int outputRows = 4;
		
		Matrix input = new Matrix(inputRows, cols, i-> random.nextGaussian());
		Matrix expected = new Matrix(outputRows, cols, i->0);
		
		Matrix weights = new Matrix(outputRows, inputRows, i-> random.nextGaussian());
		Matrix biases = new Matrix(outputRows, 1, i->random.nextGaussian());
		
		for (int col = 0; col < cols; col ++) {
			int randRow = random.nextInt(outputRows);
			expected.set(randRow, col, 1);
		}
		
		NeuralNet neuralNet = m -> {
			Matrix out = m.apply((index, value)-> value > 0 ? value: 0); //relu
			out = weights.multiply(out); //weights
			out.modify((row, col, value)->value + biases.get(row)); //biases
			out = out.softmax(); //softmax
			
			return out;
		};
		
		Matrix softmaxOutput = neuralNet.apply(input);
		
		Matrix approximatedResult = Approximator.gradient(input, in->{
			return LossFunctions.crossEntropy(expected, neuralNet.apply(in));
		});
		
		Matrix calculatedResult = softmaxOutput.apply((index, value) -> value - expected.get(index)); //Backpropagation through softmax
		calculatedResult = weights.transpose().multiply(calculatedResult); //backpropagation through weighted sum
		calculatedResult = calculatedResult.apply((index, value) -> input.get(index) > 0 ? value: 0); //backpropagation through relu
		//System.out.println(approximatedResult);
		//System.out.println(calculatedResult);
		
		assertTrue(approximatedResult.equals(calculatedResult));
		
		if (printValues) {
			System.out.println("Inital Random Input:");
			System.out.println(input);
			Matrix m1 = input.apply((index, value)-> value > 0 ? value: 0);
			System.out.println("Relu(input):");
			System.out.println(m1);
			System.out.println("Weighted Sum:");
			System.out.println(weights.multiply(m1).modify((row, col, value)->value + biases.get(row)));
			System.out.println("Softmaxed Weighed Sum:");
			System.out.println(neuralNet.apply(input));
			System.out.println("Expected/Ideal Results:");
			System.out.println(expected);
			System.out.println("Loss Between inital And Expected:");
			System.out.println(LossFunctions.crossEntropy(expected, neuralNet.apply(input)));
			System.out.println("Differental for each space in the input Matrix:");
			System.out.println(approximatedResult);
		}
	
	}
	
	@Test
	public void testBackpropWeights() {
		final boolean printValues = false;
		
		interface NeuralNet {
			Matrix apply(Matrix m);
		}
		
		final int inputRows = 4;
		final int cols = 5;
		final int outputRows = 4;
		
		Matrix input = new Matrix(inputRows, cols, i-> random.nextGaussian());
		Matrix expected = new Matrix(outputRows, cols, i->0);
		
		Matrix weights = new Matrix(outputRows, inputRows, i-> random.nextGaussian());
		Matrix biases = new Matrix(outputRows, 1, i->random.nextGaussian());
		
		for (int col = 0; col < cols; col ++) {
			int randRow = random.nextInt(outputRows);
			expected.set(randRow, col, 1);
		}
		
		NeuralNet neuralNet = m -> weights.multiply(m).modify((row, col, value)->value + biases.get(row)).softmax();
		Matrix softmaxOutput = neuralNet.apply(input);
		
		Matrix approximatedResult = Approximator.gradient(input, in->{
			return LossFunctions.crossEntropy(expected, neuralNet.apply(in));
		});
		
		Matrix calculatedResult = weights.transpose().multiply(softmaxOutput.apply((index, value) -> value - expected.get(index)));
		
		//System.out.println(approximatedResult);
		//System.out.println(calculatedResult);
		
		assertTrue(approximatedResult.equals(calculatedResult));
		
		if (printValues) {
			System.out.println("Inital Random Input:");
			System.out.println(input);
			System.out.println("Weighted Sum:");
			System.out.println(weights.multiply(input).modify((row, col, value)->value + biases.get(row)));
			System.out.println("Softmaxed Weighed Sum:");
			System.out.println(neuralNet.apply(input));
			System.out.println("Expected/Ideal Results:");
			System.out.println(expected);
			System.out.println("Error Between inital And Expected:");
			System.out.println(LossFunctions.crossEntropy(expected, weights.multiply(input).modify((row, col, value)->value + biases.get(row)).softmax()));
			System.out.println("Differental for each space in the input Matrix:");
			System.out.println(approximatedResult);
		}
	
	}
	
	@Test
	public void testSoftmaxCrossEntropyGradient() {
		final boolean printValues = false;
		
		final int rows = 4;
		final int cols = 5;
		Matrix input = new Matrix(rows, cols, i-> random.nextGaussian());
		
		Matrix expected = new Matrix(rows, cols, i->0);
		
		for (int col = 0; col < cols; col ++) {
			int randRow = random.nextInt(rows);
			expected.set(randRow, col, 1);
		}
		
		Matrix softmaxOutput = input.softmax();
		
		Matrix result = Approximator.gradient(input, in->{
			return LossFunctions.crossEntropy(expected, in.softmax());
		});
		
		result.forEach((index, value) -> {
			double softOutput = softmaxOutput.get(index);
			double expectedOutput = expected.get(index);
			
			//the rate of change of the losses with respect to a Input Matrix, is equal to the softmax() of that Matrix minus the expected output (ie: [0, 0, 1, 0])
			if (printValues)
				System.out.println("("+value+", "+(softOutput - expectedOutput)+")");
			
			assertTrue(Math.abs(value-(softOutput - expectedOutput)) < 0.01);
		});
		
		if (printValues) {
			System.out.println("Inital Random Input:");
			System.out.println(input);
			System.out.println("Expected/Ideal Results:");
			System.out.println(expected);
			System.out.println("Loss Between inital.softmax() And Expected:");
			System.out.println(LossFunctions.crossEntropy(expected, input.softmax()));
			System.out.println("Differental for each space in the input Matrix:");
			System.out.println(result);
		}
	
	}
	
	@Test
	public void testApproximator() {
		final boolean printValues = false;
		
		final int rows = 4;
		final int cols = 5;
		Matrix input = new Matrix(rows, cols, i-> random.nextGaussian()).softmax();
		
		Matrix expected = new Matrix(rows, cols, i->0);
		
		for (int col = 0; col < cols; col ++) {
			int randRow = random.nextInt(rows);
			expected.set(randRow, col, 1);
		}
		
		Matrix result = Approximator.gradient(input, in->{
			return LossFunctions.crossEntropy(expected, in);
		});
		
		input.forEach((index, value) -> {
			double resultValue = result.get(index);
			double expectedValue = expected.get(index);
			
			if(expectedValue < 0.00001) {
				assertTrue(resultValue < 0.01);
			}
			else {
				//remember the derivative of f(x) = -ln(x) (the Loss Function) is f'(x) = -1/x
				assertTrue(Math.abs(resultValue - (-1.0/value)) < 0.01);
			}
		});
		
		if (printValues) {
			System.out.println("Pre-Softmaxed Inital Random Input:");
			System.out.println(input);
			System.out.println("Expected/Ideal Results:");
			System.out.println(expected);
			System.out.println("Loss Between Inital And Expected:");
			System.out.println(LossFunctions.crossEntropy(expected, input));
			System.out.println("Differental for each space in the input Matrix:");
			System.out.println(result);
		}
	
	}
	
	@Test
	public void testCrossEntropy() {
		double[] expectedValues = {1, 0, 0, 0, 0, 1, 0, 1, 0};
		
		Matrix expected = new Matrix(3, 3, i->expectedValues[i]);
		Matrix actual = new Matrix(3, 3, i->0.05*i*i).softmax();
		Matrix lossMatrix = LossFunctions.crossEntropy(expected, actual);
		
		actual.forEach((row, col, index, value)->{
			double expectedValue = expected.get(index);
			double loss = lossMatrix.get(col);
			
			if(expectedValue > 0.9) {
				assertTrue(Math.abs(-Math.log(value) - loss) < 0.001);
			}
		});
	}
	
	//@Test
	public void testTemp() {
		int inputSize = 5;
		int layer1Size = 6;
		int layer2Size = 4;
		
		Matrix input = new Matrix(inputSize, 1, i->random.nextGaussian());
		Matrix layer1Weights = new Matrix(layer1Size, input.getRows(), i->random.nextGaussian());
		Matrix layer1Biases = new Matrix(layer1Size, 1, i->random.nextGaussian());
		
		Matrix layer2Weights = new Matrix(layer2Size, layer1Weights.getRows(), i->random.nextGaussian());
		Matrix layer2Biases = new Matrix(layer2Size, 1, i->random.nextGaussian());
		
		
		//inputs
		System.out.println("Input:");
		var output = input;
		System.out.println(output);
		
		//layer 1 weighted sums
		System.out.println("layer 1 weighted sums:");
		output = layer1Weights.multiply(output);
		System.out.println(output);
		
		//biases added
		System.out.println("biases added:");
		output = output.modify((row, col, value)->value + layer1Biases.get(row));
		System.out.println(output);
		
		//RELU values/inputs for layer 2
		System.out.println("RELU values/inputs for layer 2:");
		output = output.modify(value -> value > 0 ? value: 0);
		System.out.println(output);
		
		//layer 2 weighted sums
		System.out.println("layer 2 weighted sums:");
		output = layer2Weights.multiply(output);
		System.out.println(output);
		
		//biases added
		System.out.println("biases added:");
		output = output.modify((row, col, value) -> layer2Biases.get(row));
		System.out.println(output);
		
		//SOFTMAX/final output
		System.out.println("SOFTMAX/final output:");
		output = output.softmax();
		System.out.println(output);
		
	}
	
	@Test
	public void testAddBias() {

		Matrix input = new Matrix(3, 3, i->i+1);
		Matrix weights = new Matrix(3, 3, i->i+1);
		Matrix biases = new Matrix(3, 1, i->i+1);
		
		Matrix result = weights.multiply(input).modify((row, col, value)->value + biases.get(row));
		
		double[] expectedValues = { +31.00000, +37.00000, +43.00000, 
				                    +68.00000, +83.00000, +98.00000,
				                    +105.00000, +129.00000, +153.00000};
		Matrix expected = new Matrix(3, 3, i->expectedValues[i]);
		
		assertTrue(expected.equals(result));
	}

	@Test
	public void testReLu() {
		final int numberNeurons = 5;
		final int numberInputs = 6;
		final int inputSize = 4;

		Matrix input = new Matrix(inputSize, numberInputs, i-> random.nextDouble());
		Matrix weights = new Matrix(numberNeurons, inputSize, i-> random.nextGaussian());
		Matrix biases = new Matrix(numberNeurons, 1, i-> random.nextGaussian());
		
		Matrix result1 = weights.multiply(input).modify((row, col, value)->value + biases.get(row));
		Matrix result2 = weights.multiply(input).modify((row, col, value)->value + biases.get(row)).modify(value -> value > 0 ? value: 0);
		
		result2.forEach((index, value) -> {
			double originalValue = result1.get(index);
			if(originalValue > 0) {
				assertTrue(Math.abs(originalValue-value) < 0.000001);
			}
			else {
				assertTrue(Math.abs(value) < 0.000001);
			}
		});

	}
	
}

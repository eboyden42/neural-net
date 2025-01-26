package research.neuralnetwork;
import java.io.Serializable;
import java.util.LinkedList;
import java.util.Random;
import research.matrix.Matrix;

public class Engine implements Serializable {
	private static final long serialVersionUID = 1L;
	
	private LinkedList<Transform> transforms = new LinkedList<Transform>();
	private LinkedList<Matrix> weights = new LinkedList<Matrix>();
	private LinkedList<Matrix> biases = new LinkedList<Matrix>();
	
	private LossFunction lossFunction = LossFunction.CROSSENTROPY;
	private double scaleInitalWeights = 1;
	private boolean storeInputError = false;
	
	public void setScaleInitalWeights(double scale) {
		scaleInitalWeights = scale;
		
		if (weights.size() != 0) {
			throw new RuntimeException("Must call scaleInitalWeights BEFORE adding transforms!");
		}
	}
	
	public void evaluate(BatchResult batchResult, Matrix expected) {
		if (lossFunction != LossFunction.CROSSENTROPY) {
			throw new UnsupportedOperationException("CrossEntropy is the ony supported loss function.");
		}
		
		//System.out.println(batchResult.getOutput());
		
		double loss = LossFunctions.crossEntropy(expected, batchResult.getOutput()).averageColumn().get(0);
		batchResult.setLoss(loss);
		
		Matrix predictions = batchResult.getOutput().getGreatestRowNumber();
		Matrix actual = expected.getGreatestRowNumber();
		
		int correct = 0;
		
		
		for (int i = 0 ; i < actual.getCols(); i ++) {
			int p = (int)predictions.get(i);
			int a = (int)actual.get(i);
			
			if (p == a) {
				correct ++;
			}
		}
		
		double percentCorrect = (100.0 * correct)/actual.getCols();
		
		batchResult.setPercentCorrect(percentCorrect);
	}
	
	public BatchResult runForwards(Matrix input) {
		Matrix output = input;
		BatchResult batchResult = new BatchResult();
		batchResult.add(output);
		
		int denseIndex = 0;
		
		for (var t: transforms) {
			if (t == Transform.DENSE) {
				batchResult.addWeightInput(output);
				
				Matrix weight = weights.get(denseIndex);
				Matrix bias = biases.get(denseIndex);
				
				output = weight.multiply(output).modify((row, col, value)->value + bias.get(row));
			    
				++ denseIndex;		
			}
			else if(t == Transform.RELU) {
				output = output.modify(value -> value > 0 ? value: 0);
			}
			else if(t == Transform.SOFTMAX) {
				output = output.softmax();
			}
			
			batchResult.add(output);
		}
		
		return batchResult;
	}
	
	public void setStoreInputError(boolean storeInputError) {
		this.storeInputError = storeInputError;
	}

	public void adjust(BatchResult batchResult, double learningRate) {
		var weightInputs = batchResult.getWeightInputs();
		var weightErrors = batchResult.getWeightErrors();
		
		assert weightInputs.size() == weightErrors.size();
		assert weightInputs.size() == weights.size();
		
		for (int i = 0; i < weights.size(); i ++) {
			var weight = weights.get(i);
			var bias = biases.get(i);
			var error = weightErrors.get(i);
			var input = weightInputs.get(i);
			
			assert weight.getCols() == input.getRows();
			
			var weightAdjust = error.multiply(input.transpose());
			var biasAdjust = error.averageColumn();
			
			double rate = learningRate/input.getCols();
			
			weight.modify((index, value) -> value- rate*weightAdjust.get(index));
			bias.modify((row, col, value) -> value- learningRate*biasAdjust.get(row));
			
		}
	}
	
	public void runBackwards(BatchResult batchResult, Matrix expected) {
		
		var transformIt = transforms.descendingIterator();
		
		if (lossFunction != LossFunction.CROSSENTROPY || transforms.getLast() != Transform.SOFTMAX) {
			throw new UnsupportedOperationException("Loss Function must be Cross Entropy and last Transform must be Softmax");
		}
		
		var ioIt = batchResult.getIo().descendingIterator();
		Matrix softmaxOutput = ioIt.next();
		var weightIt = weights.descendingIterator();
		Matrix error = softmaxOutput.apply((index, value) -> value - expected.get(index));
		
		while(transformIt.hasNext()) {
			Transform transform = transformIt.next();
			
			Matrix input = ioIt.next();
			
			switch(transform) {
			case DENSE:
				batchResult.addWeightErrors(error);
				
				Matrix weight = weightIt.next();
				if (weightIt.hasNext() || storeInputError) {
					error = weight.transpose().multiply(error);
				}
				break;
			case RELU:
				error = error.apply((index, value) -> input.get(index) > 0 ? value: 0);
				break;
			case SOFTMAX:
				break;
			default:
				throw new UnsupportedOperationException("Layer Type Not Implemented");
			}
			
			//System.out.println(transform);
		}
		
		if (storeInputError) {
			batchResult.setInputError(error);
		}
	}
	
	public void add(Transform transform, double...params) {
		
		Random random = new Random();
		
		if (transform == Transform.DENSE) {
			int numberNeurons = (int)params[0];
			int weightsPerNeuron = weights.size() == 0 ? (int)params[1]: weights.getLast().getRows();
			
			Matrix weight = new Matrix(numberNeurons, weightsPerNeuron, i->scaleInitalWeights * random.nextGaussian());
			Matrix bias = new Matrix(numberNeurons, 1, i->0); //OPTIONAL CHANGE STARTING BIAS TO ZERO
			
			weights.add(weight);
			biases.add(bias);
		}
		transforms.add(transform);
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		sb.append(String.format("Scale Inital Weights: %.3f", scaleInitalWeights));
		sb.append("\nTransforms:\n");
		
		int weightIndex = 0;
		for (var t: transforms) {
			sb.append(t);
			
			if(t == Transform.DENSE) {
				sb.append(" ").append(weights.get(weightIndex).toString(false));
				weightIndex ++;
			}
			
			sb.append("\n");
		}
		
		return sb.toString();
	}
}

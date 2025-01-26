package matrix;

import static org.junit.Assert.*;

import java.util.Random;

import research.matrix.Matrix;

import org.junit.Test;

public class MatrixTest {
	
	private Random random = new Random(); 
	
	@Test
	public void testGetGreatestRowNumber() {
		double[] values = {7, 6, 2, 
				           2, 10, -1, 
				          -4, 2, 8};
		double[] expectedValues = {0, 1, 2};
		Matrix m = new Matrix(3, 3, i->values[i]);
		Matrix expected = new Matrix(1, 3, i->expectedValues[i]);
		
		System.out.println(m.getGreatestRowNumber());
		
		assertTrue(m.getGreatestRowNumber().equals(expected));
	}
	
	
	@Test
	public void testAverageColumn() {
		int cols = 7;
		int rows = 5;
		
		Matrix m = new Matrix(rows, cols, i -> 2*i-3);
		double averageIndex = (cols-1)/2.0;
		Matrix expected = new Matrix(rows, 1);
		expected.modify((row, col, value) -> 2*((row*cols) + averageIndex) - 3);
		
		assertTrue(m.averageColumn().equals(expected));
		
	}
	@Test
	public void testTranspose() {
		Matrix m = new Matrix(2, 3, i->i);
		Matrix result = m.transpose();
		
		double[] expectedValues = {0, 3, 1, 4, 2, 5};
		Matrix expected = new Matrix(3, 2, i->expectedValues[i]);
		
		assertTrue(result.equals(expected));
	}
	
	@Test
	public void testAddIncrement() {
		Matrix m = new Matrix(5, 8, i->random.nextGaussian());
		
		int row = 3;
		int col = 2;
		double inc = 5.0;
		
		Matrix result = m.addIncrement(row, col, inc);
		
		double incrementedValue = result.get(row, col);
		double originalValue = m.get(row, col);
		assertTrue(Math.abs(incrementedValue-(originalValue+inc)) < 0.000001);
	}
	
	@Test
	public void testSoftMax() {
		Matrix m = new Matrix(5, 8, i->random.nextGaussian());
		Matrix result = m.softmax();
		
		double[] colSums = new double[8];
		
		result.forEach((row, col, value)-> {
			assertTrue(value>=0.0 && value <= 1.0);
			
			colSums[col] += value;
		});
		
		for (double sum: colSums) {
			assertTrue(Math.abs(sum-1.0) < 0.000001);
		}
	}
	
	@Test
	public void testSumColumns() {
		Matrix m = new Matrix(4, 5, i->i);
		
		Matrix result = m.sumColumns();
		
		double[] expectedValues = {+30.00000, +34.00000, +38.00000, +42.00000, +46.00000};
		Matrix expected = new Matrix(1, 5, i->expectedValues[i]);
		
		assertTrue(result.equals(expected));
	}
	
	@Test
	public void testMultiplyMatrix() {
		Matrix m1 = new Matrix(2, 3, i->i);
		Matrix m2 = new Matrix(3, 2, i->i);
		
		double[] expectedValues = {10, 13, 28, 40};
		Matrix expected = new Matrix(2, 2, i->expectedValues[i]); 
		
		Matrix result = m1.multiply(m2);
		
		assertTrue(expected.equals(result));
	}
	
	@Test
	public void testMultiplySpeed() {
		int rows = 500;
		int cols = 500;
		int mutual = 50;
		
		Matrix m1 = new Matrix(rows, mutual, i->i);
		Matrix m2 = new Matrix(mutual, cols, i->i);
		
		var start = System.currentTimeMillis();
		m1.multiply(m2);
		var finish = System.currentTimeMillis();
		
		System.out.printf("Matrix Multiplication Time: %dms\n", finish-start);
	}

	@Test
	public void testEquals() {
		Matrix m1 = new Matrix(3, 4, i-> 0.5 * (i-6));
		Matrix m2 = new Matrix(3, 4, i-> 0.5 * (i-6));
		Matrix m3 = new Matrix(3, 4, i-> 0.5 * (i-6.2));
		
		assertTrue(m1.equals(m2));
		assertFalse(m2.equals(m3));
	}
	
	@Test
	public void testAddMatrices() {
		Matrix m1 = new Matrix(2, 2, i-> i);
		Matrix m2 = new Matrix(2, 2, i-> i*1.5);
		Matrix expected = new Matrix(2, 2, i-> i+(i*1.5));
		Matrix result = m1.apply((index, value)-> m2.get(index)+value);
		
		assertTrue(result.equals(expected));
	}
	
	@Test
	public void testMultiplyDouble() {
		Matrix m = new Matrix(3, 4, i-> i*(0.5)+1.5);
		double x = 4.0;
		Matrix expected = new Matrix(3, 4, i-> x*(i*(0.5)+1.5));	
		Matrix result = m.apply((index, value) -> x*(value));
		
		assertTrue(expected.equals(result));
		assertTrue(Math.abs(expected.get(1)-result.get(1)) < 0.000001);		
	}

	@Test
	public void testToString() {
		Matrix m = new Matrix(3, 4, i->i*2);
		
		double[] expected = new double[12];
		for(int i = 0; i < expected.length; i ++) {
			expected[i] = i * 2;
		}
		String text = m.toString();
		var rows = text.split("\n");
		
		assertTrue(rows.length == 3);
		
		int index = 0;
		
		for (var row : rows) {
			var values = row.split("\\s+");
			for (var textValue: values) {
				if (textValue.length() == 0) {
					continue;
				}
				var doubleValue = Double.valueOf(textValue);
				
				assertTrue(Math.abs(doubleValue - expected[index]) < 0.00001);
				
				index ++;
			}
		}
		
	}

}

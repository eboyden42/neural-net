package research.matrix;
import java.util.Random;
import java.util.Scanner;

public class MatrixMultPractice {

	private static Random random = new Random();
	private static Scanner scan = new Scanner(System.in);
	
	public static void main(String[] args) {
		for (int i = 0; i < 10; i ++) {
			System.out.println(i+1+"/"+10);
			practiceMult();
		}
		
		for (int i =0; i < 10; i ++) {
			System.out.println(i+1+"/"+10);
			practiceRowsCols();
		}
		
		for (int i = 0; i < 20; i ++) {
			System.out.println(i+1+"/"+20);
			practiceMultPossible();
		}
		
		for (int i = 0; i < 25; i ++) {
			System.out.println(i+1+"/"+25);
			practiceFindMultSize();
		}

	}
	
	public static void practiceRowsCols() {
		int upperbound = 6;
		int rows = random.nextInt(upperbound)+1;
		int cols = random.nextInt(upperbound)+1;
		
		Matrix m = new Matrix(rows, cols, i->(double)random.nextInt(10));
		System.out.println(m);
		scan.nextInt();
		System.out.println(m.toString(false));
		
	}
	
	public static void practiceMultPossible() {
		int upperbound = 5;
		int rows = random.nextInt(upperbound)+1;
		int cols = random.nextInt(upperbound)+1;
		int rows1 = random.nextInt(upperbound)+1;
		int cols1 = random.nextInt(upperbound)+1;
		
		Matrix m = new Matrix(rows, cols, i->(double)random.nextInt(10));
		Matrix m1 = new Matrix(rows1, cols1, i->(double)random.nextInt(10));
		
		System.out.println(m);
		System.out.println("-------------");
		System.out.println(m1);
		scan.nextBoolean();
		System.out.println(m.getCols() == m1.getRows());
	}
	
	public static void practiceFindMultSize() {
		int upperbound = 5;
		int rows = random.nextInt(upperbound)+1;;
		int cols = random.nextInt(upperbound)+1;;
		int rows1 = cols;
		int cols1 = random.nextInt(upperbound)+1;;
		
		Matrix m = new Matrix(rows, cols, i->(double)random.nextInt(10));
		Matrix m1 = new Matrix(rows1, cols1, i->(double)random.nextInt(10));
		
		System.out.println(m);
		System.out.println("-------------");
		System.out.println(m1);
		
		Matrix result = m.multiply(m1);
		scan.nextInt();
		System.out.println(result.toString(false));
		
	}
	
	public static void practiceMult() {
		int upperbound = 3;
		int rows = random.nextInt(upperbound)+1;;
		int cols = random.nextInt(upperbound)+1;;
		int rows1 = cols;
		int cols1 = random.nextInt(upperbound)+1;;
		
		Matrix m = new Matrix(rows, cols, i->(double)random.nextInt(10));
		Matrix m1 = new Matrix(rows1, cols1, i->(double)random.nextInt(10));
		
		System.out.println(m);
		System.out.println("-------------");
		System.out.println(m1);
		
		Matrix result = m.multiply(m1);
		scan.nextInt();
		System.out.println(result.toString(false));
		
		result.forEach((index, value) -> {
			System.out.print(index+": ");
			double input = scan.nextDouble();
			System.out.println((input == value)+" | "+value);
		});
		System.out.println(result);
		
	}

}

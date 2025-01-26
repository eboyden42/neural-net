package research.neuralnetwork;

/**
 * Hello world!
 *
 */


public class App 
{
    
	public static double perceptron(double[] x, double[] w, double b) {
		
		double z = 0;
		for (int i = 0; i < x.length; i ++) {
			z += x[i]*w[i];
		}
		
		return z-b > 0 ? 1.0: 0.0;
	}
		
	public static double and(double x1, double x2) {
		return perceptron(new double[] {x1, x2}, new double[] {1.0, 1.0}, 1);
	}
	
	public static double or(double x1, double x2) {
		return perceptron(new double[] {x1, x2}, new double[] {1, 1}, 0);
	}
	
	public static double nor(double x1, double x2) {
		return perceptron(new double[] {x1, x2}, new double[] {-1, -1}, -1);
	}
	
	public static double nand(double x1, double x2) {
		return perceptron(new double[] {x1, x2}, new double[] {-1, -1}, -2);
	}
	
	public static double xor(double x1, double x2) {
		return and(nand(x1, x2), or(x1, x2));
	}
	
	public static double xand(double x1, double x2) {
		return or(and(x1, x2), nor(x1, x2));
	}
	
	public static void main( String[] args ) {
        for (int i = 0; i < 4; i ++) {
        	double x1 = i/2;
        	double x2 = i%2;
        	
        	System.out.print(x1+", "+x2+" : ");
        	System.out.println(xand(x1, x2));
        }
    }
}

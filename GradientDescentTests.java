import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Random;

public class GradientDescentTests {
    private static final double ALPHA = 0.01; // learning rate
    private static final double EPSILON = 0.001;
    private static final int MAX_ITERATIONS = 1000;
    private static int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int NUM_POINTS = 1000000;

    public static void main(String[] args) {

        /**
         // Generate random data
         ArrayList<Point2D> data = generateData(NUM_POINTS);

         // Run ordinary gradient descent
         long startTime = System.currentTimeMillis();
         double[] thetaSequential = gradientDescentSequential(data);
         long endTime = System.currentTimeMillis();
         long sequentialExecutionTime = endTime - startTime;

         // Run parallel gradient descent
         startTime = System.currentTimeMillis();
         double[] thetaParallel = gradientDescentParallel(data);
         endTime = System.currentTimeMillis();
         long parallelExecutionTime = endTime - startTime;

         // Print results
         System.out.println("Sequential Gradient Descent:");
         System.out.println("Theta0: " + thetaSequential[0] + ", Theta1: " + thetaSequential[1]);
         System.out.println("Norm: " + calculateGradientNorm(thetaSequential));
         System.out.println("Execution Time: " + sequentialExecutionTime + " ms");
         System.out.println();

         System.out.println("Parallel Gradient Descent:");
         System.out.println("Theta0: " + thetaParallel[0] + ", Theta1: " + thetaParallel[1]);
         System.out.println("Norm: " + calculateGradientNorm(thetaParallel));
         System.out.println("Execution Time: " + parallelExecutionTime + " ms");
         */

        for(int i=1000000; i<=100000000; i*=10){
            // Generate random data
            ArrayList<Point2D> data = generateData(i);
            System.out.println("Data size: "+i);

            // Run ordinary gradient descent
            long startTime = System.currentTimeMillis();
            double[] thetaSequential = gradientDescentSequential(data);
            long endTime = System.currentTimeMillis();
            long sequentialExecutionTime = endTime - startTime;
            // Print results
            System.out.println("Sequential         time "+sequentialExecutionTime+" ms");

            // Run parallel gradient descent
            for(int threads = 2; threads <= 10; threads+=2) {
                NUM_THREADS = threads;
                startTime = System.currentTimeMillis();
                double[] thetaParallel = gradientDescentParallel(data);
                endTime = System.currentTimeMillis();
                long parallelExecutionTime = endTime - startTime;
                // Print results
                System.out.println("Parallel threads "+threads+" time " + parallelExecutionTime + " ms   Speedup "+(double)sequentialExecutionTime/parallelExecutionTime);
            }
            System.out.println();
        }
    }

    private static ArrayList<Point2D> generateData(int numPoints) {
        ArrayList<Point2D> data = new ArrayList<>();
        Random random = new Random();

        for (int i = 0; i < numPoints; i++) {
            double x = random.nextDouble();
            double y = 2.0 * x + random.nextGaussian() * 0.2;

            data.add(new Point2D.Double(x, y));
        }

        return data;
    }

    private static double calculateGradientNorm(double[] gradient) {
        double dx = gradient[0];
        double dy = gradient[1];

        return Math.sqrt(dx * dx + dy * dy);
    }

    private static double calculateError(double x, double y, double theta0, double theta1){
        double hypothesis = theta0 + theta1 * x;
        return hypothesis - y;
    }

    private static double[] gradientDescentSequential(ArrayList<Point2D> data) {
        double theta0 = 0.0;
        double theta1 = 0.0;

        int m = data.size();

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            double sum0 = 0.0;
            double sum1 = 0.0;

            for (Point2D d : data) {
                double x = d.getX();
                double error = calculateError(x, d.getY(), theta0, theta1);

                sum0 += error;
                sum1 += error * x;
            }

            double delta0 = sum0 / m;
            double delta1 = sum1 / m;

            theta0 -= ALPHA * delta0;
            theta1 -= ALPHA * delta1;

            /**
             if (Math.abs(delta0) < EPSILON && Math.abs(delta1) < EPSILON) {
             break;
             }
             */
            // Calculate gradient norm
            double gradientNorm = calculateGradientNorm(new double[]{delta0, delta1});

            // Check stoppage condition
            if (gradientNorm < EPSILON) {
                break;
            }
        }

        return new double[]{theta0, theta1};
    }

    private static double[] gradientDescentParallel(ArrayList<Point2D> data) {
        double theta0 = 0.0;
        double theta1 = 0.0;

        int m = data.size();

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            double sum0 = 0.0;
            double sum1 = 0.0;

            // Split the data into equal chunks for parallel processing
            int numThreads = NUM_THREADS;
            int chunkSize = m / numThreads;

            Thread[] threads = new Thread[numThreads];
            GradientDescentWorker[] workers = new GradientDescentWorker[numThreads];

            for (int i = 0; i < numThreads; i++) {
                int startIndex = i * chunkSize;
                int endIndex = (i == numThreads - 1) ? m : (i + 1) * chunkSize;

                workers[i] = new GradientDescentWorker(data, startIndex, endIndex, theta0, theta1);
                threads[i] = new Thread(workers[i]);
                threads[i].start();
            }

            try {
                for (int i = 0; i < numThreads; i++) {
                    threads[i].join();
                    sum0 += workers[i].getSum0();
                    sum1 += workers[i].getSum1();
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            double delta0 = sum0 / m;
            double delta1 = sum1 / m;

            theta0 -= ALPHA * delta0;
            theta1 -= ALPHA * delta1;

            /**
             if (Math.abs(delta0) < EPSILON && Math.abs(delta1) < EPSILON) {
             break;
             }
             */
            // Calculate gradient norm
            double gradientNorm = calculateGradientNorm(new double[]{delta0, delta1});

            // Check stoppage condition
            if (gradientNorm < EPSILON) {
                break;
            }
        }

        return new double[]{theta0, theta1};
    }

    private static class GradientDescentWorker implements Runnable {
        private final ArrayList<Point2D> data;
        private final int startIndex;
        private final int endIndex;
        private final double theta0;
        private final double theta1;
        private double sum0;
        private double sum1;

        public GradientDescentWorker(ArrayList<Point2D> data, int startIndex, int endIndex, double theta0, double theta1) {
            this.data = data;
            this.startIndex = startIndex;
            this.endIndex = endIndex;
            this.theta0 = theta0;
            this.theta1 = theta1;
        }

        @Override
        public void run() {
            sum0 = 0.0;
            sum1 = 0.0;

            for (int i = startIndex; i < endIndex; i++) {
                double x = data.get(i).getX();
                double error = calculateError(x, data.get(i).getY(), theta0, theta1);

                sum0 += error;
                sum1 += error * x;
            }
        }

        public double getSum0() {
            return sum0;
        }

        public double getSum1() {
            return sum1;
        }
    }
}

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

public class GradientDescentTests {
    private static final double ALPHA = 0.01; // learning rate
    private static final double EPSILON = 0.001;
    private static final int MAX_ITERATIONS = 1000;
    private static int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    private static final int NUM_POINTS = 1000000;

    public static void main(String[] args) {

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
            //double y = 2.0 * x + random.nextGaussian() * 0.2;
            double y = random.nextDouble(0.1,3.0) * x + random.nextGaussian() * 0.2;

            data.add(new Point2D.Double(x, y));
        }

        return data;
    }

    private static double calculateGradientNorm(double[] gradient) {
        double dx = gradient[0];
        double dy = gradient[1];

        return Math.sqrt(dx * dx + dy * dy);
    }

    private static double calculateError(double x, double y, double theta0, double theta1) {
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
        double[] theta = {0.0,0.0};

        int m = data.size();

        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            double sum0 = 0.0;
            double sum1 = 0.0;

            // Split the data into equal chunks for parallel processing
            int chunkSize = m / NUM_THREADS;

            ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            List<Future<GradientDescentResult>> futures = new ArrayList<>();

            for (int i = 0; i < NUM_THREADS; i++) {
                int startIndex = i * chunkSize;
                int endIndex = (i == NUM_THREADS - 1) ? m : (i + 1) * chunkSize;

                futures.add(executor.submit(() -> {
                    double localSum0 = 0.0;
                    double localSum1 = 0.0;

                    for (int j = startIndex; j < endIndex; j++) {
                        double x = data.get(j).getX();
                        double error = calculateError(x, data.get(j).getY(), theta[0], theta[1]);

                        localSum0 += error;
                        localSum1 += error * x;
                    }

                    return new GradientDescentResult(localSum0, localSum1);
                }));
            }

            executor.shutdown();

            try {
                executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            for (Future<GradientDescentResult> future : futures) {
                try {
                    GradientDescentResult result = future.get();
                    sum0 += result.sum0;
                    sum1 += result.sum1;
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }

            double delta0 = sum0 / m;
            double delta1 = sum1 / m;

            theta[0] -= ALPHA * delta0;
            theta[1] -= ALPHA * delta1;

            // Calculate gradient norm
            double gradientNorm = calculateGradientNorm(new double[]{delta0, delta1});

            // Check stoppage condition
            if (gradientNorm < EPSILON) {
                break;
            }
        }

        return theta;
    }

    private record GradientDescentResult(double sum0, double sum1) {
    }
}

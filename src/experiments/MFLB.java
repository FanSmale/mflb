package experiments;

import algorithm.*;
import datamodel.*;

/**
 * Manage the whole experiment process.
 * 
 * @author Fan Min minfanphd@163.com
 */
public class MFLB {
	/**
	 * Matrix factorization scheme.
	 */
	public static final int SIMPLE = 0;

	/**
	 * Matrix factorization scheme.
	 */
	public static final int ONE_REGULAR = 1;

	/**
	 * Matrix factorization scheme.
	 */
	public static final int PQ_REGULAR = 2;

	/**
	 * Matrix factorization scheme.
	 */
	public static final int SIGMOID = 3;

	/**
	 * Matrix factorization scheme.
	 */
	public static final int NUM_SCHEMES = 4;

	/**
	 * The rating system.
	 */
	RatingSystem wholeD;

	/**
	 ************************ 
	 * Compare all schemes.
	 ************************ 
	 */
	public static void constantCAdjusting(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraMinimalRounds, int paraNumExperiments)
			throws Exception {
		RatingSystem tempRatingSystem = new RatingSystem(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
		int tempNumC = 6;
		double[] tempCValues = { 4.0, 4.0, 4.4, 4.6, 4.8, 5.0 };
		// { 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
		// 3.6, 3.8, 4.0 };
		double[][] tempTrainMaeArray = new double[paraNumExperiments][tempNumC];
		double[][] tempTestMaeArray = new double[paraNumExperiments][tempNumC];
		double[][] tempTrainRmseArray = new double[paraNumExperiments][tempNumC];
		double[][] tempTestRmseArray = new double[paraNumExperiments][tempNumC];
		double[][] tempTestAucArray = new double[paraNumExperiments][tempNumC];

		double tempLikeThreshold = 3;

		double[] tempTrainAverageMaeArray = new double[tempNumC];
		double[] tempTrainAverageRmseArray = new double[tempNumC];
		double[] tempTestAverageMaeArray = new double[tempNumC];
		double[] tempTestAverageRmseArray = new double[tempNumC];
		double[] tempTestAverageAucArray = new double[tempNumC];

		for (int i = 0; i < paraNumExperiments; i++) {
			System.out.println("Training and testing # " + i);
			tempRatingSystem.splitTrainValidationTest(0.8, 0.1);
			double tempMean = tempRatingSystem.getMeanRatingOfTrain();
			// double tempMean = 0;
			Triple[][] tempTrainingMatrix = tempRatingSystem.getTrainingMatrix();
			tempRatingSystem.centralize(tempTrainingMatrix);
			Triple[][] tempValidationMatrix = tempRatingSystem.getValidationMatrix();
			tempRatingSystem.centralize(tempValidationMatrix);
			Triple[][] tempTestingMatrix = tempRatingSystem.getTestingMatrix();
			tempRatingSystem.centralize(tempTestingMatrix);

			SigmoidMF tempMF = new SigmoidMF(tempTrainingMatrix, tempValidationMatrix, paraNumUsers, paraNumItems,
					paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
			tempMF.setParameters(5, 0.0003, 0.005);

			for (int j = 0; j < tempNumC; j++) {
				tempMF.setConstantC(tempCValues[j]);

				// Step 3. update and predict
				System.out.println("Begin Training for C = " + tempCValues[j]);
				tempMF.train(paraMinimalRounds);
				tempTrainMaeArray[i][j] = tempMF.mae();
				tempTrainRmseArray[i][j] = tempMF.rsme();

				tempTestMaeArray[i][j] = tempMF.mae(tempTestingMatrix);
				tempTestRmseArray[i][j] = tempMF.rsme(tempTestingMatrix);
				tempTestAucArray[i][j] = tempMF.auc(tempTestingMatrix,
						tempLikeThreshold - tempRatingSystem.getMeanRatingOfTrain() - 0.01);

				System.out.println("\r\nMAE: " + tempTestMaeArray[i][j]);
				System.out.println("RSME: " + tempTestRmseArray[i][j]);
				System.out.println("AUC: " + tempTestAucArray[i][j]);
			} // Of for j
		} // Of for i

		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < tempNumC; j++) {
				tempTrainAverageMaeArray[j] += tempTrainMaeArray[i][j] / paraNumExperiments;
				tempTrainAverageRmseArray[j] += tempTrainRmseArray[i][j] / paraNumExperiments;
				tempTestAverageMaeArray[j] += tempTestMaeArray[i][j] / paraNumExperiments;
				tempTestAverageRmseArray[j] += tempTestRmseArray[i][j] / paraNumExperiments;
				tempTestAverageAucArray[j] += tempTestAucArray[i][j] / paraNumExperiments;
			} // Of for j
		} // Of for i

		double tempDifference = 0;
		double[] tempTrainMaeDeviationArray = new double[tempNumC];
		double[] tempTrainRmseDeviationArray = new double[tempNumC];
		double[] tempTestMaeDeviationArray = new double[tempNumC];
		double[] tempTestRmseDeviationArray = new double[tempNumC];
		double[] tempTestAucDeviationArray = new double[tempNumC];
		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < tempNumC; j++) {
				tempDifference = (tempTrainMaeArray[i][j] - tempTrainAverageMaeArray[j]);
				tempTrainMaeDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTrainRmseArray[i][j] - tempTrainAverageRmseArray[j]);
				tempTrainRmseDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestMaeArray[i][j] - tempTestAverageMaeArray[j]);
				tempTestMaeDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestRmseArray[i][j] - tempTestAverageRmseArray[j]);
				tempTestRmseDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestAucArray[i][j] - tempTestAverageAucArray[j]);
				tempTestAucDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;
			} // Of for j
		} // Of for i

		for (int j = 0; j < tempNumC; j++) {
			System.out.println("---- C = " + tempCValues[j] + " ----");
			System.out.println(
					"Train MAE: " + tempTrainAverageMaeArray[j] + " +- " + Math.sqrt(tempTrainMaeDeviationArray[j]));
			System.out.println(
					"Train RMSE: " + tempTrainAverageRmseArray[j] + " +- " + Math.sqrt(tempTrainRmseDeviationArray[j]));
			System.out.println(
					"Test MAE: " + tempTestAverageMaeArray[j] + " +- " + Math.sqrt(tempTestMaeDeviationArray[j]));
			System.out.println(
					"Test RMSE: " + tempTestAverageRmseArray[j] + " +- " + Math.sqrt(tempTestRmseDeviationArray[j]));
			System.out.println(
					"Test AUC: " + tempTestAverageAucArray[j] + " +- " + Math.sqrt(tempTestAucDeviationArray[j]));
		} // Of for j

		System.out.println("Detail: ");
		for (int j = 0; j < tempNumC; j++) {
			System.out.println("---- C = " + tempCValues[j] + " ----");
			System.out.print("Test MAE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestMaeArray[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test RMSE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestRmseArray[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test AUC detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestAucArray[i][j] + ",");
			} // Of for i
			System.out.println();
		} // Of for j
	}// Of constantCAdjusting

	/**
	 ************************ 
	 * Compare all schemes.
	 ************************ 
	 */
	public static void schemeComparison(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraMinimalRounds, int paraNumExperiments)
			throws Exception {
		RatingSystem tempRatingSystem = new RatingSystem(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
		double[][] tempTrainMaeArray = new double[paraNumExperiments][NUM_SCHEMES];
		double[][] tempTestMaeArray = new double[paraNumExperiments][NUM_SCHEMES];
		double[][] tempTrainRmseArray = new double[paraNumExperiments][NUM_SCHEMES];
		double[][] tempTestRmseArray = new double[paraNumExperiments][NUM_SCHEMES];
		double[][] tempTestAucArray = new double[paraNumExperiments][NUM_SCHEMES];

		double tempLikeThreshold = 3;

		double[] tempTrainAverageMaeArray = new double[NUM_SCHEMES];
		double[] tempTrainAverageRmseArray = new double[NUM_SCHEMES];
		double[] tempTestAverageMaeArray = new double[NUM_SCHEMES];
		double[] tempTestAverageRmseArray = new double[NUM_SCHEMES];
		double[] tempTestAverageAucArray = new double[NUM_SCHEMES];

		for (int i = 0; i < paraNumExperiments; i++) {
			System.out.println("Training and testing # " + i);
			tempRatingSystem.splitTrainValidationTest(0.8, 0.1);
			double tempMean = tempRatingSystem.getMeanRatingOfTrain();
			// double tempMean = 0;
			Triple[][] tempTrainingMatrix = tempRatingSystem.getTrainingMatrix();
			tempRatingSystem.centralize(tempTrainingMatrix);
			Triple[][] tempValidationMatrix = tempRatingSystem.getValidationMatrix();
			tempRatingSystem.centralize(tempValidationMatrix);
			Triple[][] tempTestingMatrix = tempRatingSystem.getTestingMatrix();
			tempRatingSystem.centralize(tempTestingMatrix);

			for (int j = 0; j < NUM_SCHEMES; j++) {
				SimpleMatrixFactorization tempMF = null;
				switch (j) {
				case SIMPLE:
					tempMF = new SimpleMatrixFactorization(tempTrainingMatrix, tempValidationMatrix, paraNumUsers,
							paraNumItems, paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
					tempMF.setParameters(5, 0.0001, 0.005);
					break;
				case ONE_REGULAR:
					tempMF = new OneRegularMF(tempTrainingMatrix, tempValidationMatrix, paraNumUsers, paraNumItems,
							paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
					tempMF.setParameters(5, 0.0001, 0.005);
					break;
				case PQ_REGULAR:
					tempMF = new PQRegularMF(tempTrainingMatrix, tempValidationMatrix, paraNumUsers, paraNumItems,
							paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
					tempMF.setParameters(5, 0.0001, 0.005);
					break;
				case SIGMOID:
					tempMF = new SigmoidMF(tempTrainingMatrix, tempValidationMatrix, paraNumUsers, paraNumItems,
							paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
					tempMF.setParameters(5, 0.0003, 0.005);
					break;
				default:
					System.out.println("Unsupported regular type: " + j);
					System.exit(0);
				}// Of case

				// Step 3. update and predict
				System.out.println("Begin Training for algorithm #" + j + "...");
				tempMF.train(paraMinimalRounds);
				tempTrainMaeArray[i][j] = tempMF.mae();
				tempTrainRmseArray[i][j] = tempMF.rsme();

				tempTestMaeArray[i][j] = tempMF.mae(tempTestingMatrix);
				tempTestRmseArray[i][j] = tempMF.rsme(tempTestingMatrix);
				tempTestAucArray[i][j] = tempMF.auc(tempTestingMatrix,
						tempLikeThreshold - tempRatingSystem.getMeanRatingOfTrain() - 0.01);

				System.out.println("\r\nMAE: " + tempTestMaeArray[i][j]);
				System.out.println("RSME: " + tempTestRmseArray[i][j]);
				System.out.println("AUC: " + tempTestAucArray[i][j]);
			} // Of for j
		} // Of for i

		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < NUM_SCHEMES; j++) {
				tempTrainAverageMaeArray[j] += tempTrainMaeArray[i][j] / paraNumExperiments;
				tempTrainAverageRmseArray[j] += tempTrainRmseArray[i][j] / paraNumExperiments;
				tempTestAverageMaeArray[j] += tempTestMaeArray[i][j] / paraNumExperiments;
				tempTestAverageRmseArray[j] += tempTestRmseArray[i][j] / paraNumExperiments;
				tempTestAverageAucArray[j] += tempTestAucArray[i][j] / paraNumExperiments;
			} // Of for j
		} // Of for i

		double tempDifference = 0;
		double[] tempTrainMaeDeviationArray = new double[NUM_SCHEMES];
		double[] tempTrainRmseDeviationArray = new double[NUM_SCHEMES];
		double[] tempTestMaeDeviationArray = new double[NUM_SCHEMES];
		double[] tempTestRmseDeviationArray = new double[NUM_SCHEMES];
		double[] tempTestAucDeviationArray = new double[NUM_SCHEMES];
		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < NUM_SCHEMES; j++) {
				tempDifference = (tempTrainMaeArray[i][j] - tempTrainAverageMaeArray[j]);
				tempTrainMaeDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTrainRmseArray[i][j] - tempTrainAverageRmseArray[j]);
				tempTrainRmseDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestMaeArray[i][j] - tempTestAverageMaeArray[j]);
				tempTestMaeDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestRmseArray[i][j] - tempTestAverageRmseArray[j]);
				tempTestRmseDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestAucArray[i][j] - tempTestAverageAucArray[j]);
				tempTestAucDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;
			} // Of for j
		} // Of for i

		for (int j = 0; j < NUM_SCHEMES; j++) {
			System.out.println("Scheme #" + j);
			System.out.println(
					"Train MAE: " + tempTrainAverageMaeArray[j] + " +- " + Math.sqrt(tempTrainMaeDeviationArray[j]));
			System.out.println(
					"Train RMSE: " + tempTrainAverageRmseArray[j] + " +- " + Math.sqrt(tempTrainRmseDeviationArray[j]));
			System.out.println(
					"Test MAE: " + tempTestAverageMaeArray[j] + " +- " + Math.sqrt(tempTestMaeDeviationArray[j]));
			System.out.println(
					"Test RMSE: " + tempTestAverageRmseArray[j] + " +- " + Math.sqrt(tempTestRmseDeviationArray[j]));
			System.out.println(
					"Test AUC: " + tempTestAverageAucArray[j] + " +- " + Math.sqrt(tempTestAucDeviationArray[j]));
		} // Of for j

		System.out.println("Detail:");
		for (int j = 0; j < NUM_SCHEMES; j++) {
			System.out.print("Test MAE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestMaeArray[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test RMSE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestRmseArray[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test AUC detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestAucArray[i][j] + ",");
			} // Of for i
			System.out.println();
		} // Of for j
	}// Of schemeComparison

	/**
	 ************************ 
	 * Compare all schemes.
	 ************************ 
	 */
	public static void lambdaAdjusting(String paraFilename, int paraNumUsers, int paraNumItems, int paraNumRatings,
			double paraRatingLowerBound, double paraRatingUpperBound, int paraMinimalRounds, int paraNumExperiments)
			throws Exception {
		RatingSystem tempRatingSystem = new RatingSystem(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
		int tempNumLambda = 6;
		double[] tempLambdaValues = { 0.6, 0.8, 1.0, 1.2, 1.4, 1.6 };
		double[][] tempTrainMaeArray = new double[paraNumExperiments][tempNumLambda];
		double[][] tempTestMaeArray = new double[paraNumExperiments][tempNumLambda];
		double[][] tempTrainRmseArray = new double[paraNumExperiments][tempNumLambda];
		double[][] tempTestRmseArray = new double[paraNumExperiments][tempNumLambda];
		double[][] tempTestAucArray = new double[paraNumExperiments][tempNumLambda];

		double tempLikeThreshold = 3;

		double[] tempTrainAverageMaeArray = new double[tempNumLambda];
		double[] tempTrainAverageRmseArray = new double[tempNumLambda];
		double[] tempTestAverageMaeArray = new double[tempNumLambda];
		double[] tempTestAverageRmseArray = new double[tempNumLambda];
		double[] tempTestAverageAucArray = new double[tempNumLambda];

		for (int i = 0; i < paraNumExperiments; i++) {
			System.out.println("Training and testing # " + i);
			tempRatingSystem.splitTrainValidationTest(0.8, 0.1);
			double tempMean = tempRatingSystem.getMeanRatingOfTrain();
			// double tempMean = 0;
			Triple[][] tempTrainingMatrix = tempRatingSystem.getTrainingMatrix();
			tempRatingSystem.centralize(tempTrainingMatrix);
			Triple[][] tempValidationMatrix = tempRatingSystem.getValidationMatrix();
			tempRatingSystem.centralize(tempValidationMatrix);
			Triple[][] tempTestingMatrix = tempRatingSystem.getTestingMatrix();
			tempRatingSystem.centralize(tempTestingMatrix);

			SigmoidExponentMF tempMF = new SigmoidExponentMF(tempTrainingMatrix, tempValidationMatrix, paraNumUsers,
					paraNumItems, paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
			tempMF.setParameters(5, 0.0003, 0.005);

			for (int j = 0; j < tempNumLambda; j++) {
				tempMF.setLambda(tempLambdaValues[j]);

				// Step 3. update and predict
				System.out.println("Begin Training for Labmda = " + tempLambdaValues[j]);
				tempMF.train(paraMinimalRounds);
				tempTrainMaeArray[i][j] = tempMF.mae();
				tempTrainRmseArray[i][j] = tempMF.rsme();

				tempTestMaeArray[i][j] = tempMF.mae(tempTestingMatrix);
				tempTestRmseArray[i][j] = tempMF.rsme(tempTestingMatrix);
				tempTestAucArray[i][j] = tempMF.auc(tempTestingMatrix,
						tempLikeThreshold - tempRatingSystem.getMeanRatingOfTrain() - 0.01);

				System.out.println("\r\nMAE: " + tempTestMaeArray[i][j]);
				System.out.println("RSME: " + tempTestRmseArray[i][j]);
				System.out.println("AUC: " + tempTestAucArray[i][j]);
			} // Of for j
		} // Of for i

		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < tempNumLambda; j++) {
				tempTrainAverageMaeArray[j] += tempTrainMaeArray[i][j] / paraNumExperiments;
				tempTrainAverageRmseArray[j] += tempTrainRmseArray[i][j] / paraNumExperiments;
				tempTestAverageMaeArray[j] += tempTestMaeArray[i][j] / paraNumExperiments;
				tempTestAverageRmseArray[j] += tempTestRmseArray[i][j] / paraNumExperiments;
				tempTestAverageAucArray[j] += tempTestAucArray[i][j] / paraNumExperiments;
			} // Of for j
		} // Of for i

		double tempDifference = 0;
		double[] tempTrainMaeDeviationArray = new double[tempNumLambda];
		double[] tempTrainRmseDeviationArray = new double[tempNumLambda];
		double[] tempTestMaeDeviationArray = new double[tempNumLambda];
		double[] tempTestRmseDeviationArray = new double[tempNumLambda];
		double[] tempTestAucDeviationArray = new double[tempNumLambda];
		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < tempNumLambda; j++) {
				tempDifference = (tempTrainMaeArray[i][j] - tempTrainAverageMaeArray[j]);
				tempTrainMaeDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTrainRmseArray[i][j] - tempTrainAverageRmseArray[j]);
				tempTrainRmseDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestMaeArray[i][j] - tempTestAverageMaeArray[j]);
				tempTestMaeDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestRmseArray[i][j] - tempTestAverageRmseArray[j]);
				tempTestRmseDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;

				tempDifference = (tempTestAucArray[i][j] - tempTestAverageAucArray[j]);
				tempTestAucDeviationArray[j] += tempDifference * tempDifference / paraNumExperiments;
			} // Of for j
		} // Of for i

		for (int j = 0; j < tempNumLambda; j++) {
			System.out.println("---- Lambda = " + tempLambdaValues[j] + " ----");
			System.out.println(
					"Train MAE: " + tempTrainAverageMaeArray[j] + " +- " + Math.sqrt(tempTrainMaeDeviationArray[j]));
			System.out.println(
					"Train RMSE: " + tempTrainAverageRmseArray[j] + " +- " + Math.sqrt(tempTrainRmseDeviationArray[j]));
			System.out.println(
					"Test MAE: " + tempTestAverageMaeArray[j] + " +- " + Math.sqrt(tempTestMaeDeviationArray[j]));
			System.out.println(
					"Test RMSE: " + tempTestAverageRmseArray[j] + " +- " + Math.sqrt(tempTestRmseDeviationArray[j]));
			System.out.println(
					"Test AUC: " + tempTestAverageAucArray[j] + " +- " + Math.sqrt(tempTestAucDeviationArray[j]));
		} // Of for j

		System.out.println("Detail: ");
		for (int j = 0; j < tempNumLambda; j++) {
			System.out.println("---- Lambda = " + tempLambdaValues[j] + " ----");
			System.out.print("Test MAE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestMaeArray[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test RMSE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestRmseArray[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test AUC detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestAucArray[i][j] + ",");
			} // Of for i
			System.out.println();
		} // Of for j
		System.out.println("Finish.");
	}// Of lambdaAdjusting

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		try {
			// constantCAdjusting("D:/data/movielens-943u1682m.txt", 943, 1682,
			// 100000, 1, 5, 6000,
			// 10);
			lambdaAdjusting("D:/data/movielens-943u1682m.txt", 943, 1682, 100000, 1, 5, 3000, 2);
			// schemeComparison("D:/data/movielens-943u1682m.txt", 943, 1682,
			// 100000, 1, 5, 4000, 2);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
	}// Of main

}// Of class MFLB

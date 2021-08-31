package experiments;

import java.util.Arrays;

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
	public static void constantCAdjusting(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings, double paraRatingLowerBound, double paraRatingUpperBound,
			int paraMinimalRounds, int paraNumExperiments) throws Exception {
		RatingSystem tempRatingSystem = new RatingSystem(paraFilename, paraNumUsers, paraNumItems,
				paraNumRatings);
		int tempNumC = 30;
		double[] tempCValues = new double[tempNumC];
		for(int i = 0; i < tempNumC; i ++) {
			tempCValues[i] = 0.6 + i * 0.1;
		}//Of for i

		double[] tempOriginalLikeThresholds = { 3.0, 4.0, 5.0 };
		int tempNumThresholds = tempOriginalLikeThresholds.length;

		double[][] tempTrainMaeMatrix = new double[paraNumExperiments][tempNumC];
		double[][] tempTestMaeMatrix = new double[paraNumExperiments][tempNumC];
		double[][] tempTrainRmseMatrix = new double[paraNumExperiments][tempNumC];
		double[][] tempTestRmseMatrix = new double[paraNumExperiments][tempNumC];
		double[][][] tempTestAucCubic = new double[paraNumExperiments][tempNumC][tempNumThresholds];

		double[] tempTrainAverageMaeArray = new double[tempNumC];
		double[] tempTrainAverageRmseArray = new double[tempNumC];
		double[] tempTestAverageMaeArray = new double[tempNumC];
		double[] tempTestAverageRmseArray = new double[tempNumC];
		double[][] tempTestAverageAucMatrix = new double[tempNumC][tempNumThresholds];

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

			double[] tempLikeThresholds = new double[tempNumThresholds];
			for (int j = 0; j < tempNumThresholds; j++) {
				tempLikeThresholds[j] = tempOriginalLikeThresholds[j]
						- tempRatingSystem.getMeanRatingOfTrain() - 0.01;
			} // Of for j

			SigmoidExponentMF tempMF = new SigmoidExponentMF(tempTrainingMatrix,
					tempValidationMatrix, paraNumUsers, paraNumItems,
					paraRatingLowerBound - tempMean, paraRatingUpperBound - tempMean);
			tempMF.setParameters(5, 0.0003, 0.005);

			for (int j = 0; j < tempNumC; j++) {
				tempMF.setConstantC(tempCValues[j]);

				// Step 3. update and predict
				System.out.println("\r\n---\r\nBegin Training for C = " + tempCValues[j]);
				tempMF.train(paraMinimalRounds);
				tempTrainMaeMatrix[i][j] = tempMF.mae();
				tempTrainRmseMatrix[i][j] = tempMF.rsme();

				tempTestMaeMatrix[i][j] = tempMF.mae(tempTestingMatrix);
				tempTestRmseMatrix[i][j] = tempMF.rsme(tempTestingMatrix);
				tempTestAucCubic[i][j] = tempMF.auc(tempTestingMatrix, tempLikeThresholds);

				System.out.println("\r\nMAE: " + tempTestMaeMatrix[i][j]);
				System.out.println("RSME: " + tempTestRmseMatrix[i][j]);
				System.out.println("AUC: " + Arrays.toString(tempTestAucCubic[i][j]));
			} // Of for j
		} // Of for i

		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < tempNumC; j++) {
				tempTrainAverageMaeArray[j] += tempTrainMaeMatrix[i][j] / paraNumExperiments;
				tempTrainAverageRmseArray[j] += tempTrainRmseMatrix[i][j] / paraNumExperiments;
				tempTestAverageMaeArray[j] += tempTestMaeMatrix[i][j] / paraNumExperiments;
				tempTestAverageRmseArray[j] += tempTestRmseMatrix[i][j] / paraNumExperiments;
				for (int k = 0; k < tempNumThresholds; k++) {
					tempTestAverageAucMatrix[j][k] += tempTestAucCubic[i][j][k]
							/ paraNumExperiments;
				} // Of for k
			} // Of for j
		} // Of for i

		double tempDifference = 0;
		double[] tempTrainMaeDeviationArray = new double[tempNumC];
		double[] tempTrainRmseDeviationArray = new double[tempNumC];
		double[] tempTestMaeDeviationArray = new double[tempNumC];
		double[] tempTestRmseDeviationArray = new double[tempNumC];
		double[][] tempTestAucDeviationMatrix = new double[tempNumC][tempNumThresholds];
		for (int i = 0; i < paraNumExperiments; i++) {
			for (int j = 0; j < tempNumC; j++) {
				tempDifference = (tempTrainMaeMatrix[i][j] - tempTrainAverageMaeArray[j]);
				tempTrainMaeDeviationArray[j] += tempDifference * tempDifference
						/ paraNumExperiments;

				tempDifference = (tempTrainRmseMatrix[i][j] - tempTrainAverageRmseArray[j]);
				tempTrainRmseDeviationArray[j] += tempDifference * tempDifference
						/ paraNumExperiments;

				tempDifference = (tempTestMaeMatrix[i][j] - tempTestAverageMaeArray[j]);
				tempTestMaeDeviationArray[j] += tempDifference * tempDifference
						/ paraNumExperiments;

				tempDifference = (tempTestRmseMatrix[i][j] - tempTestAverageRmseArray[j]);
				tempTestRmseDeviationArray[j] += tempDifference * tempDifference
						/ paraNumExperiments;

				for (int k = 0; k < tempNumThresholds; k++) {
					tempDifference = (tempTestAucCubic[i][j][k] - tempTestAverageAucMatrix[j][k]);
					tempTestAucDeviationMatrix[j][k] += tempDifference * tempDifference
							/ paraNumExperiments;
				} // Of for k
			} // Of for j
		} // Of for i

		for (int j = 0; j < tempNumC; j++) {
			System.out.println("---- C = " + tempCValues[j] + " ----");
			System.out.println("Train MAE: " + tempTrainAverageMaeArray[j] + " +- "
					+ Math.sqrt(tempTrainMaeDeviationArray[j]));
			System.out.println("Train RMSE: " + tempTrainAverageRmseArray[j] + " +- "
					+ Math.sqrt(tempTrainRmseDeviationArray[j]));
			System.out.println("Test MAE: " + tempTestAverageMaeArray[j] + " +- "
					+ Math.sqrt(tempTestMaeDeviationArray[j]));
			System.out.println("Test RMSE: " + tempTestAverageRmseArray[j] + " +- "
					+ Math.sqrt(tempTestRmseDeviationArray[j]));
			for (int k = 0; k < tempNumThresholds; k++) {
				System.out.println("Test AUC with like threhold " + tempOriginalLikeThresholds[k]
						+ ": " + tempTestAverageAucMatrix[j][k] + " +- "
						+ Math.sqrt(tempTestAucDeviationMatrix[j][k]));
			} // Of for k
		} // Of for j

		System.out.println("Detail: ");
		for (int j = 0; j < tempNumC; j++) {
			System.out.println("---- C = " + tempCValues[j] + " ----");
			System.out.print("Test MAE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestMaeMatrix[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test RMSE detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + tempTestRmseMatrix[i][j] + ",");
			} // Of for i
			System.out.println();
			System.out.print("Test AUC detail: ");
			for (int i = 0; i < paraNumExperiments; i++) {
				System.out.print("" + Arrays.toString(tempTestAucCubic[i][j]) + ",");
			} // Of for i
			System.out.println();
		} // Of for j
	}// Of constantCAdjusting

	/**
	 ************************ 
	 * Compare all schemes.
	 ************************ 
	 * public static void schemeComparison(String paraFilename, int
	 * paraNumUsers, int paraNumItems, int paraNumRatings, double
	 * paraRatingLowerBound, double paraRatingUpperBound, int paraMinimalRounds,
	 * int paraNumExperiments) throws Exception { double[] tempLikeThresholds =
	 * { 3.0, 4.0, 5.0 }; int tempNumThresholds = tempLikeThresholds.length;
	 * 
	 * RatingSystem tempRatingSystem = new RatingSystem(paraFilename,
	 * paraNumUsers, paraNumItems, paraNumRatings); double[][]
	 * tempTrainMaeMatrix = new double[paraNumExperiments][NUM_SCHEMES];
	 * double[][] tempTestMaeMatrix = new
	 * double[paraNumExperiments][NUM_SCHEMES]; double[][] tempTrainRmseMatrix =
	 * new double[paraNumExperiments][NUM_SCHEMES]; double[][]
	 * tempTestRmseMatrix = new double[paraNumExperiments][NUM_SCHEMES];
	 * double[][][] tempTestAucCubic = new
	 * double[paraNumExperiments][NUM_SCHEMES][tempNumThresholds];
	 * 
	 * double[] tempTrainAverageMaeArray = new double[NUM_SCHEMES]; double[]
	 * tempTrainAverageRmseArray = new double[NUM_SCHEMES]; double[]
	 * tempTestAverageMaeArray = new double[NUM_SCHEMES]; double[]
	 * tempTestAverageRmseArray = new double[NUM_SCHEMES]; double[]
	 * tempTestAverageAucMatrix = new double[NUM_SCHEMES];
	 * 
	 * for (int i = 0; i < paraNumExperiments; i++) {
	 * System.out.println("Training and testing # " + i);
	 * tempRatingSystem.splitTrainValidationTest(0.8, 0.1); double tempMean =
	 * tempRatingSystem.getMeanRatingOfTrain(); // double tempMean = 0;
	 * Triple[][] tempTrainingMatrix = tempRatingSystem.getTrainingMatrix();
	 * tempRatingSystem.centralize(tempTrainingMatrix); Triple[][]
	 * tempValidationMatrix = tempRatingSystem.getValidationMatrix();
	 * tempRatingSystem.centralize(tempValidationMatrix); Triple[][]
	 * tempTestingMatrix = tempRatingSystem.getTestingMatrix();
	 * tempRatingSystem.centralize(tempTestingMatrix);
	 * 
	 * for (int j = 0; j < NUM_SCHEMES; j++) { SimpleMatrixFactorization tempMF
	 * = null; switch (j) { case SIMPLE: tempMF = new
	 * SimpleMatrixFactorization(tempTrainingMatrix, tempValidationMatrix,
	 * paraNumUsers, paraNumItems, paraRatingLowerBound - tempMean,
	 * paraRatingUpperBound - tempMean); tempMF.setParameters(5, 0.0001, 0.005);
	 * break; case ONE_REGULAR: tempMF = new OneRegularMF(tempTrainingMatrix,
	 * tempValidationMatrix, paraNumUsers, paraNumItems, paraRatingLowerBound -
	 * tempMean, paraRatingUpperBound - tempMean); tempMF.setParameters(5,
	 * 0.0001, 0.005); break; case PQ_REGULAR: tempMF = new
	 * PQRegularMF(tempTrainingMatrix, tempValidationMatrix, paraNumUsers,
	 * paraNumItems, paraRatingLowerBound - tempMean, paraRatingUpperBound -
	 * tempMean); tempMF.setParameters(5, 0.0001, 0.005); break; case SIGMOID:
	 * tempMF = new SigmoidMF(tempTrainingMatrix, tempValidationMatrix,
	 * paraNumUsers, paraNumItems, paraRatingLowerBound - tempMean,
	 * paraRatingUpperBound - tempMean); tempMF.setParameters(5, 0.0003, 0.005);
	 * break; default: System.out.println("Unsupported regular type: " + j);
	 * System.exit(0); }// Of case
	 * 
	 * // Step 3. update and predict System.out.println("\r\n---\r\nBegin
	 * Training for algorithm #" + j + "..."); tempMF.train(paraMinimalRounds);
	 * tempTrainMaeMatrix[i][j] = tempMF.mae(); tempTrainRmseMatrix[i][j] =
	 * tempMF.rsme();
	 * 
	 * tempTestMaeMatrix[i][j] = tempMF.mae(tempTestingMatrix);
	 * tempTestRmseMatrix[i][j] = tempMF.rsme(tempTestingMatrix);
	 * tempTestAucCubic[i][j] = tempMF.auc(tempTestingMatrix, tempLikeThreshold
	 * - tempRatingSystem.getMeanRatingOfTrain() - 0.01);
	 * 
	 * System.out.println("\r\nMAE: " + tempTestMaeMatrix[i][j]);
	 * System.out.println("RSME: " + tempTestRmseMatrix[i][j]);
	 * System.out.println("AUC: " + tempTestAucCubic[i][j]); } // Of for j } //
	 * Of for i
	 * 
	 * for (int i = 0; i < paraNumExperiments; i++) { for (int j = 0; j <
	 * NUM_SCHEMES; j++) { tempTrainAverageMaeArray[j] +=
	 * tempTrainMaeMatrix[i][j] / paraNumExperiments;
	 * tempTrainAverageRmseArray[j] += tempTrainRmseMatrix[i][j] /
	 * paraNumExperiments; tempTestAverageMaeArray[j] += tempTestMaeMatrix[i][j]
	 * / paraNumExperiments; tempTestAverageRmseArray[j] +=
	 * tempTestRmseMatrix[i][j] / paraNumExperiments;
	 * tempTestAverageAucMatrix[j] += tempTestAucCubic[i][j] /
	 * paraNumExperiments; } // Of for j } // Of for i
	 * 
	 * double tempDifference = 0; double[] tempTrainMaeDeviationArray = new
	 * double[NUM_SCHEMES]; double[] tempTrainRmseDeviationArray = new
	 * double[NUM_SCHEMES]; double[] tempTestMaeDeviationArray = new
	 * double[NUM_SCHEMES]; double[] tempTestRmseDeviationArray = new
	 * double[NUM_SCHEMES]; double[] tempTestAucDeviationMatrix = new
	 * double[NUM_SCHEMES]; for (int i = 0; i < paraNumExperiments; i++) { for
	 * (int j = 0; j < NUM_SCHEMES; j++) { tempDifference =
	 * (tempTrainMaeMatrix[i][j] - tempTrainAverageMaeArray[j]);
	 * tempTrainMaeDeviationArray[j] += tempDifference * tempDifference /
	 * paraNumExperiments;
	 * 
	 * tempDifference = (tempTrainRmseMatrix[i][j] -
	 * tempTrainAverageRmseArray[j]); tempTrainRmseDeviationArray[j] +=
	 * tempDifference * tempDifference / paraNumExperiments;
	 * 
	 * tempDifference = (tempTestMaeMatrix[i][j] - tempTestAverageMaeArray[j]);
	 * tempTestMaeDeviationArray[j] += tempDifference * tempDifference /
	 * paraNumExperiments;
	 * 
	 * tempDifference = (tempTestRmseMatrix[i][j] -
	 * tempTestAverageRmseArray[j]); tempTestRmseDeviationArray[j] +=
	 * tempDifference * tempDifference / paraNumExperiments;
	 * 
	 * tempDifference = (tempTestAucCubic[i][j] - tempTestAverageAucMatrix[j]);
	 * tempTestAucDeviationMatrix[j] += tempDifference * tempDifference /
	 * paraNumExperiments; } // Of for j } // Of for i
	 * 
	 * for (int j = 0; j < NUM_SCHEMES; j++) { System.out.println("Scheme #" +
	 * j); System.out.println("Train MAE: " + tempTrainAverageMaeArray[j] + " +-
	 * " + Math.sqrt(tempTrainMaeDeviationArray[j])); System.out.println("Train
	 * RMSE: " + tempTrainAverageRmseArray[j] + " +- " +
	 * Math.sqrt(tempTrainRmseDeviationArray[j])); System.out.println("Test MAE:
	 * " + tempTestAverageMaeArray[j] + " +- " +
	 * Math.sqrt(tempTestMaeDeviationArray[j])); System.out.println("Test RMSE:
	 * " + tempTestAverageRmseArray[j] + " +- " +
	 * Math.sqrt(tempTestRmseDeviationArray[j])); System.out.println("Test AUC:
	 * " + tempTestAverageAucMatrix[j] + " +- " +
	 * Math.sqrt(tempTestAucDeviationMatrix[j])); } // Of for j
	 * 
	 * System.out.println("Detail:"); for (int j = 0; j < NUM_SCHEMES; j++) {
	 * System.out.print("Test MAE detail: "); for (int i = 0; i <
	 * paraNumExperiments; i++) { System.out.print("" + tempTestMaeMatrix[i][j]
	 * + ","); } // Of for i System.out.println(); System.out.print("Test RMSE
	 * detail: "); for (int i = 0; i < paraNumExperiments; i++) {
	 * System.out.print("" + tempTestRmseMatrix[i][j] + ","); } // Of for i
	 * System.out.println(); System.out.print("Test AUC detail: "); for (int i =
	 * 0; i < paraNumExperiments; i++) { System.out.print("" +
	 * tempTestAucCubic[i][j] + ","); } // Of for i System.out.println(); } //
	 * Of for j }// Of schemeComparison
	 */

	/**
	 ************************ 
	 * Compare lambda values.
	 ************************ 
	 * public static void lambdaAdjusting(String paraFilename, int paraNumUsers,
	 * int paraNumItems, int paraNumRatings, double paraRatingLowerBound, double
	 * paraRatingUpperBound, int paraMinimalRounds, int paraNumExperiments)
	 * throws Exception { RatingSystem tempRatingSystem = new
	 * RatingSystem(paraFilename, paraNumUsers, paraNumItems, paraNumRatings);
	 * int tempNumLambda = 6; double[] tempLambdaValues = { 0.6, 0.8, 1.0, 1.2,
	 * 1.4, 1.6 }; double[] tempLikeThresholds = { 3.0, 4.0, 5.0 }; int
	 * tempNumThresholds = tempLikeThresholds.length;
	 * 
	 * double[][] tempTrainMaeMatrix = new
	 * double[paraNumExperiments][tempNumLambda]; double[][] tempTestMaeMatrix =
	 * new double[paraNumExperiments][tempNumLambda]; double[][]
	 * tempTrainRmseMatrix = new double[paraNumExperiments][tempNumLambda];
	 * double[][] tempTestRmseMatrix = new
	 * double[paraNumExperiments][tempNumLambda]; double[][] tempTestAucCubic =
	 * new double[paraNumExperiments][tempNumLambda];
	 * 
	 * double[] tempTrainAverageMaeArray = new double[tempNumLambda]; double[]
	 * tempTrainAverageRmseArray = new double[tempNumLambda]; double[]
	 * tempTestAverageMaeArray = new double[tempNumLambda]; double[]
	 * tempTestAverageRmseArray = new double[tempNumLambda]; double[]
	 * tempTestAverageAucMatrix = new double[tempNumLambda];
	 * 
	 * for (int i = 0; i < paraNumExperiments; i++) {
	 * System.out.println("Training and testing # " + i);
	 * tempRatingSystem.splitTrainValidationTest(0.8, 0.1); double tempMean =
	 * tempRatingSystem.getMeanRatingOfTrain(); // double tempMean = 0;
	 * Triple[][] tempTrainingMatrix = tempRatingSystem.getTrainingMatrix();
	 * tempRatingSystem.centralize(tempTrainingMatrix); Triple[][]
	 * tempValidationMatrix = tempRatingSystem.getValidationMatrix();
	 * tempRatingSystem.centralize(tempValidationMatrix); Triple[][]
	 * tempTestingMatrix = tempRatingSystem.getTestingMatrix();
	 * tempRatingSystem.centralize(tempTestingMatrix);
	 * 
	 * SigmoidExponentMF tempMF = new SigmoidExponentMF(tempTrainingMatrix,
	 * tempValidationMatrix, paraNumUsers, paraNumItems, paraRatingLowerBound -
	 * tempMean, paraRatingUpperBound - tempMean); tempMF.setParameters(5,
	 * 0.0003, 0.005);
	 * 
	 * for (int j = 0; j < tempNumLambda; j++) {
	 * tempMF.setLambda(tempLambdaValues[j]);
	 * 
	 * // Step 3. update and predict System.out.println("\r\n---\r\nBegin
	 * Training for Labmda = " + tempLambdaValues[j]);
	 * tempMF.train(paraMinimalRounds); tempTrainMaeMatrix[i][j] = tempMF.mae();
	 * tempTrainRmseMatrix[i][j] = tempMF.rsme();
	 * 
	 * tempTestMaeMatrix[i][j] = tempMF.mae(tempTestingMatrix);
	 * tempTestRmseMatrix[i][j] = tempMF.rsme(tempTestingMatrix);
	 * tempTestAucCubic[i][j] = tempMF.auc(tempTestingMatrix, tempLikeThreshold
	 * - tempRatingSystem.getMeanRatingOfTrain() - 0.01);
	 * 
	 * System.out.println("\r\nMAE: " + tempTestMaeMatrix[i][j]);
	 * System.out.println("RSME: " + tempTestRmseMatrix[i][j]);
	 * System.out.println("AUC: " + tempTestAucCubic[i][j]); } // Of for j } //
	 * Of for i
	 * 
	 * for (int i = 0; i < paraNumExperiments; i++) { for (int j = 0; j <
	 * tempNumLambda; j++) { tempTrainAverageMaeArray[j] +=
	 * tempTrainMaeMatrix[i][j] / paraNumExperiments;
	 * tempTrainAverageRmseArray[j] += tempTrainRmseMatrix[i][j] /
	 * paraNumExperiments; tempTestAverageMaeArray[j] += tempTestMaeMatrix[i][j]
	 * / paraNumExperiments; tempTestAverageRmseArray[j] +=
	 * tempTestRmseMatrix[i][j] / paraNumExperiments;
	 * tempTestAverageAucMatrix[j] += tempTestAucCubic[i][j] /
	 * paraNumExperiments; } // Of for j } // Of for i
	 * 
	 * double tempDifference = 0; double[] tempTrainMaeDeviationArray = new
	 * double[tempNumLambda]; double[] tempTrainRmseDeviationArray = new
	 * double[tempNumLambda]; double[] tempTestMaeDeviationArray = new
	 * double[tempNumLambda]; double[] tempTestRmseDeviationArray = new
	 * double[tempNumLambda]; double[] tempTestAucDeviationMatrix = new
	 * double[tempNumLambda]; for (int i = 0; i < paraNumExperiments; i++) { for
	 * (int j = 0; j < tempNumLambda; j++) { tempDifference =
	 * (tempTrainMaeMatrix[i][j] - tempTrainAverageMaeArray[j]);
	 * tempTrainMaeDeviationArray[j] += tempDifference * tempDifference /
	 * paraNumExperiments;
	 * 
	 * tempDifference = (tempTrainRmseMatrix[i][j] -
	 * tempTrainAverageRmseArray[j]); tempTrainRmseDeviationArray[j] +=
	 * tempDifference * tempDifference / paraNumExperiments;
	 * 
	 * tempDifference = (tempTestMaeMatrix[i][j] - tempTestAverageMaeArray[j]);
	 * tempTestMaeDeviationArray[j] += tempDifference * tempDifference /
	 * paraNumExperiments;
	 * 
	 * tempDifference = (tempTestRmseMatrix[i][j] -
	 * tempTestAverageRmseArray[j]); tempTestRmseDeviationArray[j] +=
	 * tempDifference * tempDifference / paraNumExperiments;
	 * 
	 * tempDifference = (tempTestAucCubic[i][j] - tempTestAverageAucMatrix[j]);
	 * tempTestAucDeviationMatrix[j] += tempDifference * tempDifference /
	 * paraNumExperiments; } // Of for j } // Of for i
	 * 
	 * for (int j = 0; j < tempNumLambda; j++) { System.out.println("---- Lambda
	 * = " + tempLambdaValues[j] + " ----"); System.out.println("Train MAE: " +
	 * tempTrainAverageMaeArray[j] + " +- " +
	 * Math.sqrt(tempTrainMaeDeviationArray[j])); System.out.println("Train
	 * RMSE: " + tempTrainAverageRmseArray[j] + " +- " +
	 * Math.sqrt(tempTrainRmseDeviationArray[j])); System.out.println("Test MAE:
	 * " + tempTestAverageMaeArray[j] + " +- " +
	 * Math.sqrt(tempTestMaeDeviationArray[j])); System.out.println("Test RMSE:
	 * " + tempTestAverageRmseArray[j] + " +- " +
	 * Math.sqrt(tempTestRmseDeviationArray[j])); System.out.println("Test AUC:
	 * " + tempTestAverageAucMatrix[j] + " +- " +
	 * Math.sqrt(tempTestAucDeviationMatrix[j])); } // Of for j
	 * 
	 * System.out.println("Detail: "); for (int j = 0; j < tempNumLambda; j++) {
	 * System.out.println("---- Lambda = " + tempLambdaValues[j] + " ----");
	 * System.out.print("Test MAE detail: "); for (int i = 0; i <
	 * paraNumExperiments; i++) { System.out.print("" + tempTestMaeMatrix[i][j]
	 * + ","); } // Of for i System.out.println(); System.out.print("Test RMSE
	 * detail: "); for (int i = 0; i < paraNumExperiments; i++) {
	 * System.out.print("" + tempTestRmseMatrix[i][j] + ","); } // Of for i
	 * System.out.println(); System.out.print("Test AUC detail: "); for (int i =
	 * 0; i < paraNumExperiments; i++) { System.out.print("" +
	 * tempTestAucCubic[i][j] + ","); } // Of for i System.out.println(); } //
	 * Of for j System.out.println("Finish."); }// Of lambdaAdjusting
	 */

	/**
	 ************************ 
	 * @param args
	 ************************ 
	 */
	public static void main(String args[]) {
		try {
			constantCAdjusting("D:/data/movielens-943u1682m.txt", 943, 1682, 100000, 1, 5, 1000, 10);
			// lambdaAdjusting("D:/data/movielens-943u1682m.txt", 943, 1682,
			// 100000, 1, 5, 1000, 10);
			// schemeComparison("D:/data/movielens-943u1682m.txt", 943, 1682,
			// 100000, 1, 5, 4000, 2);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
	}// Of main

}// Of class MFLB

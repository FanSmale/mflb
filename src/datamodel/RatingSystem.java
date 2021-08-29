package datamodel;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Title: RatingSystem.java
 * 
 * @author Yuan-Yuan Xu, Fan Min www.fansmale.com Email: minfan@swpu.edu.cn
 *         Created: Nov 6, 2019
 * @date 2021/08/04
 */
public class RatingSystem {
	/**
	 * Used to generate random numbers.
	 */
	Random rand = new Random();

	/**
	 * The number of users.
	 */
	int numUsers;

	/**
	 * The number of items.
	 */
	int numItems;

	/**
	 * Total number of ratings in the rating matrix.
	 */
	int numRatings;

	/**
	 * The whole data.
	 */
	Triple[][] ratingMatrix;

	/**
	 * The matrix of training set, users as the rows.
	 */
	public Triple[][] trainingMatrix;

	/**
	 * The matrix of validation set, users as the rows.
	 */
	public Triple[][] validationMatrix;

	/**
	 * The matrix of testing set, users as the rows.
	 */
	public Triple[][] testingMatrix;

	/**
	 * The mean/average value of rating for the training set. It is equal to
	 * sum(trainVector)/trainVector.length
	 */
	public double meanRatingOfTrain;

	/**
	 * The split sign.
	 */
	// public static final String SPLIT_SIGN = new String(" ");
	public static final String SPLIT_SIGN = new String(",");

	/**
	 * The value of missing rating
	 */
	public static final double DEFAULT_MISSING_RATING = 0;

	/**
	 ********************** 
	 * Read the dataset
	 * 
	 * @param paraFilename
	 *            The file storing the data.
	 * @param paraNumUsers
	 *            Number of users. This might be obtained through scanning the
	 *            whole data. However, the approach requires an additional
	 *            scanning. Therefore we do not adopt it.
	 * @param paraNumItems
	 *            Number of items.
	 * @throws IOException
	 *             It may occur if the given file does not exist.
	 ********************** 
	 */
	public RatingSystem(String paraFilename, int paraNumUsers, int paraNumItems,
			int paraNumRatings) {
		// Step 1. Accept basic settings.
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		numRatings = paraNumRatings;

		// Step 2. Allocate space.
		ratingMatrix = new Triple[numUsers][];
		trainingMatrix = new Triple[numUsers][];
		testingMatrix = new Triple[numUsers][];

		meanRatingOfTrain = 0;

		int tempUserIndex, tempItemIndex;
		double tempRating;
		int[] tempUserRatings = new int[paraNumUsers];

		try {
			// Step 3. First scan to determine the number of ratings for each
			// user.
			File tempFile = new File(paraFilename);
			BufferedReader tempBufferReader = new BufferedReader(
					new InputStreamReader(new FileInputStream(tempFile)));
			String tempLine;
			String[] tempParts;
			for (int i = 0; i < numRatings; i++) {
				tempLine = tempBufferReader.readLine();
				tempParts = tempLine.split(SPLIT_SIGN);
				tempUserIndex = Integer.parseInt(tempParts[0]);// user id
				tempUserRatings[tempUserIndex]++;
			} // Of while

			// Step 4. Allocate space for ratingMatrix.
			for (int i = 0; i < numUsers; i++) {
				ratingMatrix[i] = new Triple[tempUserRatings[i]];
			} // Of for i

			// Step 5. Second scan to store data.
			tempBufferReader.close();
			tempBufferReader = new BufferedReader(
					new InputStreamReader(new FileInputStream(tempFile)));
			int[] tempIndexForUsers = new int[numUsers];

			for (int i = 0; i < numRatings; i++) {
				tempLine = tempBufferReader.readLine();
				tempParts = tempLine.split(SPLIT_SIGN);
				tempUserIndex = Integer.parseInt(tempParts[0]);// user id
				tempItemIndex = Integer.parseInt(tempParts[1]);// item id
				tempRating = Double.parseDouble(tempParts[2]);// rating

				ratingMatrix[tempUserIndex][tempIndexForUsers[tempUserIndex]] = new Triple(
						tempUserIndex, tempItemIndex, tempRating);
				tempIndexForUsers[tempUserIndex]++;
			} // Of for i
			tempBufferReader.close();
		} catch (Exception ee) {
			System.out.println("Errors occurred while reading " + paraFilename);
			System.out.println(ee);
			System.exit(0);
		} // Of try
	}// Of RatingSystem

	/**
	 ********************** 
	 * Split the data to obtain the training, validation and testing sets,
	 * respectively.
	 * 
	 * @param paraTrainingProportion
	 *            the proportion of the training set.
	 * @param paraValidationProportion
	 *            the proportion of the validation set.
	 * @throws IOException
	 * @throws o
	 ********************** 
	 */
	public void splitTrainValidationTest(double paraTrainingProportion,
			double paraValidationProportion) {
		// Step 1. Scan to determine which ones belong to training/testing sets.
		// No adjust is undertaken for simplicity.
		int[] tempTrainingCountArray = new int[numUsers];
		int[] tempValidationCountArray = new int[numUsers];
		int[] tempTestingCountArray = new int[numUsers];

		boolean[][] tempTrainingMatrix = new boolean[ratingMatrix.length][];
		boolean[][] tempValidationMatrix = new boolean[ratingMatrix.length][];
		boolean[][] tempTestingMatrix = new boolean[ratingMatrix.length][];

		double tempRandom;
		for (int i = 0; i < ratingMatrix.length; i++) {
			tempTrainingMatrix[i] = new boolean[ratingMatrix[i].length];
			tempValidationMatrix[i] = new boolean[ratingMatrix[i].length];
			tempTestingMatrix[i] = new boolean[ratingMatrix[i].length];
			
			//The training matrix must have at least one rating for each user.
			int tempIndex = rand.nextInt(ratingMatrix[i].length);
			tempTrainingMatrix[i][tempIndex] = true;
			tempTrainingCountArray[i]++;
			for (int j = 0; j < ratingMatrix[i].length; j++) {
				//Reserved for the training set.
				if (j == tempIndex) {
					continue;
				}//Of if
				
				tempRandom = rand.nextDouble();
				if (tempRandom < paraTrainingProportion) {
					tempTrainingMatrix[i][j] = true;
					tempTrainingCountArray[i]++;
				} else if (tempRandom < paraTrainingProportion + paraValidationProportion) {
					tempValidationMatrix[i][j] = true;
					tempValidationCountArray[i]++;
				} else {
					tempTestingMatrix[i][j] = true;
					tempTestingCountArray[i]++;
				} // Of if
			} // Of for j
		} // Of for i

		// Step 2. Allocate space.
		trainingMatrix = new Triple[numUsers][];
		validationMatrix = new Triple[numUsers][];
		testingMatrix = new Triple[numUsers][];
		
		for (int i = 0; i < ratingMatrix.length; i++) {
			trainingMatrix[i] = new Triple[tempTrainingCountArray[i]];
			validationMatrix[i] = new Triple[tempValidationCountArray[i]];
			testingMatrix[i] = new Triple[tempTestingCountArray[i]];

			int tempCounter1 = 0;
			int tempCounter2 = 0;
			int tempCounter3 = 0;
			for (int j = 0; j < ratingMatrix[i].length; j++) {
				if (tempTrainingMatrix[i][j]) {
					trainingMatrix[i][tempCounter1] = new Triple(ratingMatrix[i][j]);
					tempCounter1++;
				} else if (tempValidationMatrix[i][j]){
					validationMatrix[i][tempCounter2] = new Triple(ratingMatrix[i][j]);
					tempCounter2++;
				} else {
					testingMatrix[i][tempCounter3] = new Triple(ratingMatrix[i][j]);
					tempCounter3++;
				} // Of if
			} // Of for j
		} // Of for i

		meanRatingOfTrain = computeAverageRating(trainingMatrix);
	}// Of splitTrainValidationTest

	/**
	 ********************** 
	 * Compute the average rating of the data.
	 ********************** 
	 */
	public double computeAverageRating(Triple[][] paraMatrix) {
		double tempSum = 0;
		int tempCounts = 0;
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				tempSum += paraMatrix[i][j].rating;
				tempCounts++;
			} // Of for j
		} // Of for i

		return tempSum / tempCounts;
	}// Of computeAverageRating

	/**
	 ********************** 
	 * Centralize. Each rating subtracts the mean value. In this way the average
	 * value would be 0.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 ********************** 
	 */
	public void centralize(Triple[][] paraMatrix) {
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				paraMatrix[i][j].rating -= meanRatingOfTrain;
			} // Of for j
		} // Of for i
	}// Of centralize

	/**
	 ********************** 
	 * Centralize all data.
	 ********************** 
	 */
	public void centralizeAll() {
		centralize(ratingMatrix);
	}// Of centralizeAll

	/**
	 ********************** 
	 * Centralize training matrix.
	 ********************** 
	 */
	public void centralizeTrainingMatrix() {
		centralize(trainingMatrix);
	}// Of centralizeTrainingMatrix

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public Triple[][] getRatingMatrix() {
		return ratingMatrix;
	}// Of getRatingMatrix

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public Triple[][] getTrainingMatrix() {
		return trainingMatrix;
	}// Of getTrainingMatrix

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public Triple[][] getValidationMatrix() {
		return validationMatrix;
	}// Of getValidationMatrix

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public Triple[][] getTestingMatrix() {
		return testingMatrix;
	}// Of getTestingMatrix

	/**
	 ********************** 
	 * Getter
	 ********************** 
	 */
	public double getMeanRatingOfTrain() {
		return meanRatingOfTrain;
	}// Of getMeanRatingOfTrain

}// Of class RatingSystem

package algorithm;

import java.io.*;
import java.util.Arrays;
import java.util.Random;
import datamodel.Triple;
//import util.SimpleTool;

/*
 * Matrix factorization for recommender systems. This is the super class of other matrix factorization algorithms.
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class SimpleMatrixFactorization {
	/**
	 * Used to generate random numbers.
	 */
	Random rand = new Random();

	/**
	 * Define as a constant to save runtime..
	 */
	public static double log2 = Math.log(2);

	/**
	 * Number of users.
	 */
	int numUsers;

	/**
	 * Number of items.
	 */
	int numItems;

	/**
	 * Training data.
	 */
	Triple[][] trainingSet;

	/**
	 * Validation data.
	 */
	Triple[][] validationSet;

	/**
	 * A parameter for controlling learning speed.
	 */
	double alpha;

	/**
	 * A parameter for controlling the learning regular.
	 */
	double lambda;

	/**
	 * The low rank of the small matrices.
	 */
	int rank;

	/**
	 * The user matrix U.
	 */
	double[][] userSubspace;

	/**
	 * The item matrix V.
	 */
	double[][] itemSubspace;

	/**
	 * The lower bound of the rating value.
	 */
	double ratingLowerBound;

	/**
	 * The upper bound of the rating value.
	 */
	double ratingUpperBound;

	/**
	 ************************ 
	 * The first constructor.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 * @param paraFilename
	 *            The data filename.
	 * @param paraNumUsers
	 *            The number of users.
	 * @param paraNumItems
	 *            The number of items.
	 * @param paraNumRatings
	 *            The number of ratings.
	 ************************ 
	 */
	public SimpleMatrixFactorization(Triple[][] paraTrainingSet, Triple[][] paraValidationSet,
			int paraNumUsers, int paraNumItems, double paraRatingLowerBound,
			double paraRatingUpperBound) {
		trainingSet = paraTrainingSet;
		validationSet = paraValidationSet;
		numUsers = paraNumUsers;
		numItems = paraNumItems;
		ratingLowerBound = paraRatingLowerBound;
		ratingUpperBound = paraRatingUpperBound;
	}// Of the first constructor

	/**
	 ************************ 
	 * Set parameters for basic update.
	 * 
	 * @param paraRank
	 *            The given rank.
	 * @throws IOException
	 ************************ 
	 */
	public void setParameters(int paraRank, double paraAlpha, double paraLambda) {
		rank = paraRank;
		alpha = paraAlpha;
		lambda = paraLambda;
	}// Of setParameters

	/**
	 ************************ 
	 * Initialize subspaces. Each value is in [0, 1].
	 ************************ 
	 */
	void initializeSubspaces() {
		userSubspace = new double[numUsers][rank];

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < rank; j++) {
				userSubspace[i][j] = (rand.nextDouble() - 0.5) / 10;
			} // Of for j
		} // Of for i

		itemSubspace = new double[numItems][rank];
		for (int i = 0; i < numItems; i++) {
			for (int j = 0; j < rank; j++) {
				itemSubspace[i][j] = (rand.nextDouble() - 0.5) / 10;
			} // Of for j
		} // Of for i
	}// Of initializeSubspaces

	/**
	 ************************ 
	 * Predict the rating of the user to the item
	 * 
	 * @param paraUser
	 *            The user index.
	 ************************ 
	 */
	public double predict(int paraUser, int paraItem) {
		double resultValue = 0;
		for (int i = 0; i < rank; i++) {
			// The row vector of an user and the column vector of an item
			resultValue += userSubspace[paraUser][i] * itemSubspace[paraItem][i];
		} // Of for i
		return resultValue;
	}// Of predict

	/**
	 ************************ 
	 * Train.
	 * 
	 * @param paraMinimalRounds
	 *            The minimal number of rounds.
	 * @param paraValidation
	 *            Use validation set or not.
	 * @return The training rounds.
	 ************************ 
	 */
	public int train(int paraMinimalRounds, boolean paraValidation) {
		initializeSubspaces();

		double tempCurrentValidationMae = 100;
		double tempLastValidationMae = 100;

		// Step 1. Train the minimal rounds.
		int i = 0;
		for (i = 0; i < paraMinimalRounds; i++) {
			update();
			//if (i % 500 == 0) {
				//tempCurrentValidationMae = mae(validationSet);

				// Show the process
				//System.out.println("Round " + i);
				//System.out.println("Training MAE = " + mae() + ", RMSE = " + rsme());
				//System.out.println("Validation MAE = " + mae(validationSet) + ", RMSE = "
				//		+ rsme(validationSet));
			//} // Of if
		} // Of for i

		// Step 2. Train more.
		if (paraValidation) {
			// Terminate when the performance on the validation set gets worse.
			for (;; i++) {
				update();
				if (i % 200 == 0) {
					// Use validation set to terminate
					tempCurrentValidationMae = mae(validationSet);

					// Show the process
					//System.out.println("Round " + i);
					//System.out.println("Training MAE = " + mae() + ", RMSE = " + rsme());
					//System.out.println("Validation MAE = " + mae(validationSet) + ", RMSE = "
					//		+ rsme(validationSet));

					if (tempCurrentValidationMae > tempLastValidationMae) {
						break;
					} // Of if
					tempLastValidationMae = tempCurrentValidationMae;
				} // Of if
			} // Of for i
		} else {
			double tempCurrentTrainingMae = 100;
			double tempLastTrainingMae = 100;
			// double tempDifference = 0;
			// Terminate if converge
			for (;; i++) {
				update();
				if (i % 200 == 0) {
					tempCurrentTrainingMae = mae(trainingSet);

					// Show the process
					//System.out.println("Round " + i);
					//System.out.println("Training MAE = " + mae() + ", RMSE = " + rsme());

					// tempDifference = tempLastTrainingMae -
					// tempCurrentTrainingMae;
					// System.out.println("Difference = " + tempDifference);
					if (tempLastTrainingMae - tempCurrentTrainingMae < 1e-4) {
						break;
					} // Of if
					tempLastTrainingMae = tempCurrentTrainingMae;
				} // Of if
			} // Of for i
		} // Of if paraValidation

		return i;
	}// Of train

	/**
	 ************************ 
	 * Update sub-spaces using the training data. No regular term is considered.
	 * This method should be overwritten in subclasses.
	 ************************ 
	 */
	public void update() {
		int tempUserId, tempItemId;
		double tempRating, tempResidual, tempValue;
		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < trainingSet[i].length; j++) {
				tempUserId = trainingSet[i][j].user;
				tempItemId = trainingSet[i][j].item;
				tempRating = trainingSet[i][j].rating;

				tempResidual = tempRating - predict(tempUserId, tempItemId); // Residual

				// Update user subspace
				tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of update

	/**
	 ************************ 
	 * Compute the RSME on the training data.
	 ************************ 
	 */
	public double rsme() {
		return rsme(trainingSet);
	}// Of rsme

	/**
	 ************************ 
	 * Compute the RSME.
	 * 
	 * @return RSME of the current factorization.
	 ************************ 
	 */
	public double rsme(Triple[][] paraDataset) {
		double resultRsme = 0;
		int tempTestCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < paraDataset[i].length; j++) {
				int tempUserIndex = paraDataset[i][j].user;
				int tempItemIndex = paraDataset[i][j].item;
				double tempRate = paraDataset[i][j].rating;

				double tempPrediction = predict(tempUserIndex, tempItemIndex);

				if (tempPrediction < ratingLowerBound) {
					tempPrediction = ratingLowerBound;
				} else if (tempPrediction > ratingUpperBound) {
					tempPrediction = ratingUpperBound;
				} // Of if

				double tempError = tempRate - tempPrediction;
				resultRsme += tempError * tempError;
				tempTestCount++;
			} // Of for j
		} // Of for i

		return Math.sqrt(resultRsme / tempTestCount);
	}// Of rsme

	/**
	 ************************ 
	 * Compute the MAE on the training set.
	 ************************ 
	 */
	public double mae() {
		return mae(trainingSet);
	}// Of mae

	/**
	 ************************ 
	 * Compute the MAE.
	 * 
	 * @return MAE of the current factorization.
	 ************************ 
	 */
	public double mae(Triple[][] paraDataset) {
		double resultMae = 0;
		int tempTestCount = 0;

		for (int i = 0; i < numUsers; i++) {
			for (int j = 0; j < paraDataset[i].length; j++) {
				int tempUserIndex = paraDataset[i][j].user;
				int tempItemIndex = paraDataset[i][j].item;
				double tempRate = paraDataset[i][j].rating;

				double tempPrediction = predict(tempUserIndex, tempItemIndex);

				if (tempPrediction < ratingLowerBound) {
					tempPrediction = ratingLowerBound;
				} // Of if
				if (tempPrediction > ratingUpperBound) {
					tempPrediction = ratingUpperBound;
				} // Of if

				double tempError = tempRate - tempPrediction;

				resultMae += Math.abs(tempError);
				// System.out.println("resultMae: " + resultMae);
				tempTestCount++;
			} // Of for j
		} // Of for i

		// System.out.println("userSubspace = " +
		// Arrays.deepToString(userSubspace));
		// System.out.println("itemSubspace = " +
		// Arrays.deepToString(itemSubspace));
		// System.out.println("resultMae = " + resultMae);
		// System.out.println("tempTestCount = " + tempTestCount);
		return (resultMae / tempTestCount);
	}// Of mae

	/**
	 ************************ 
	 * Compute \sum|x_ij|.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 ************************ 
	 */
	public static double absoluteValueSum(double[][] paraMatrix) {
		double result = 0;
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				if (paraMatrix[i][j] >= 0) {
					result += paraMatrix[i][j];
				} else {
					result -= paraMatrix[i][j];
				} // Of if
			} // Of for j
		} // Of for i
		return result;
	}// Of absoluteValueSum

	/**
	 ************************ 
	 * Compute \sum(x_ij)^2. Frobenius is abbreviated as fro.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 ************************ 
	 */
	public static double froNormSquare(double[][] paraMatrix) {
		double result = 0;
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				result += paraMatrix[i][j] * paraMatrix[i][j];
			} // Of for j
		} // Of for i
		return result;
	}// Of froNormSquare

	/**
	 ************************ 
	 * Compute \sum(x_ij)2.
	 * 
	 * @param paraMatrix
	 *            The given matrix.
	 ************************ 
	 */
	public static double froNorm(double[][] paraMatrix) {
		return Math.sqrt(froNormSquare(paraMatrix));
	}// Of froNorm

	/**
	 ************************ 
	 * Get the average value of user subspace (absolute value).
	 ************************ 
	 */
	public double getAverageU() {
		double tempTotal = absoluteValueSum(userSubspace);
		double result = tempTotal / userSubspace.length / userSubspace[0].length;
		return result;
	}// Of getAverageU

	/**
	 ************************ 
	 * Get the average value of item subspace (absolute value).
	 ************************ 
	 */
	public double getAverageV() {
		double tempTotal = absoluteValueSum(itemSubspace);
		double result = tempTotal / itemSubspace.length / itemSubspace[0].length;
		return result;
	}// Of getAverageV

	/**
	 ************************ 
	 * Get the average squared user subspace values.
	 ************************ 
	 */
	public double getAverageSquareU() {
		double tempTotal = froNormSquare(userSubspace);
		double result = tempTotal / userSubspace.length / userSubspace[0].length;
		return result;
	}// Of getAverageSquareU

	/**
	 ************************ 
	 * Get the average squared item subspace values.
	 ************************ 
	 */
	public double getAverageSquareV() {
		double tempTotal = froNormSquare(itemSubspace);
		double result = tempTotal / itemSubspace.length / itemSubspace[0].length;
		return result;
	}// Of getAverageSquareV

	/**
	 ************************ 
	 * From a triple matrix to a triple array.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 * @return An array of triples.
	 ************************ 
	 */
	public Triple[] tripleMatrixToArray(Triple[][] paraDataset) {
		// Step 1. Copy data to an array.
		int tempLength = 0;
		for (int i = 0; i < paraDataset.length; i++) {
			tempLength += paraDataset[i].length;
		} // Of for i

		Triple[] resultDataArray = new Triple[tempLength];
		int tempIndex = 0;
		for (int i = 0; i < paraDataset.length; i++) {
			for (int j = 0; j < paraDataset[i].length; j++) {
				resultDataArray[tempIndex] = paraDataset[i][j];
				tempIndex++;
			} // Of for i
		} // Of for i

		return resultDataArray;
	}// Of tripleMatrixToArray

	/**
	 ************************ 
	 * Compute AUC of the dataset.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 * @param paraLikeThresholds
	 *            A number of thresholds.
	 * @return An array of AUC, one for each threshold.
	 ************************ 
	 */
	public double[] auc(Triple[][] paraDataset, double[] paraLikeThresholds) {
		Triple[] tempDataArray = tripleMatrixToArray(paraDataset);
		int tempSize = paraLikeThresholds.length;
		int tempLength = tempDataArray.length;

		double[] tempLikeCounts = new double[tempSize];
		double[] tempDislikeCounts = new double[tempSize];

		// Step 1. Sort the positions according to prediction.
		double[] tempPredictions = new double[tempLength];
		for (int i = 0; i < tempPredictions.length; i++) {
			tempPredictions[i] = predict(tempDataArray[i].user, tempDataArray[i].item);
		} // Of for i
		int[] tempSortedIndices = mergeSortToIndices(tempPredictions);

		// Step 2. Count the number of like and dislike elements.
		boolean[][] tempCorrectMatrix = new boolean[tempLength][tempSize];
		for (int i = 0; i < tempLength; i++) {
			for (int j = 0; j < tempSize; j++) {
				if (tempDataArray[tempSortedIndices[i]].rating > paraLikeThresholds[j] - 0.001) {
					tempCorrectMatrix[i][j] = true;
					// System.out.print("X ");
					tempLikeCounts[j]++;
				} else {
					tempCorrectMatrix[i][j] = false;
					// System.out.print("O ");
					tempDislikeCounts[j]++;
				} // Of if
			} // Of for j
		} // Of for i

		// Compute AUC value.
		double[] tempTotalAreaArray = new double[tempSize];
		double[] resultAucArray = new double[tempSize];
		for (int j = 0; j < tempSize; j++) {
			double tempWidth = tempDislikeCounts[j];
			for (int i = 0; i < tempLength; i++) {
				if (tempCorrectMatrix[i][j]) {
					tempTotalAreaArray[j] += tempWidth;
				} else {
					tempWidth--;
				} // Of if
			} // Of for i
			resultAucArray[j] = tempTotalAreaArray[j] / tempLikeCounts[j] / tempDislikeCounts[j];
		} // Of for j

		return resultAucArray;
	}// Of auc

	/**
	 ************************ 
	 * Compute MAP of the dataset.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 * @param paraNumItems
	 *            The total number of items.
	 * @param paraLikeThresholds
	 *            A number of thresholds.
	 * @return MAP.
	 ************************ 
	 */
	public double[] map(Triple[][] paraTrainingMatrix, Triple[][] paraValidationMatrix,
			Triple[][] paraTestingMatrix, int paraNumItems, int paraK,
			double[] paraLikeThresholds) {
		int tempSize = paraLikeThresholds.length;
		int tempNumUsers = paraTrainingMatrix.length;

		double[] resultMapArray = new double[tempSize];
		double tempMap = 0;
		double[] tempPredictions = new double[paraNumItems];
		double[] tempRatings = new double[paraNumItems];
		boolean[] tempKnownArray = new boolean[paraNumItems];
		for (int i = 0; i < tempNumUsers; i++) {
			// System.out.println("User " + i);
			// Step 1. Which ratings are known.
			Arrays.fill(tempKnownArray, false);
			for (int j = 0; j < paraTrainingMatrix[i].length; j++) {
				tempKnownArray[paraTrainingMatrix[i][j].item] = true;
			} // Of for i
			for (int j = 0; j < paraValidationMatrix[i].length; j++) {
				tempKnownArray[paraValidationMatrix[i][j].item] = true;
			} // Of for i

			// Step 2. Expand ratings of the current user.
			Arrays.fill(tempRatings, -100);
			for (int j = 0; j < paraTestingMatrix[i].length; j++) {
				tempRatings[paraTestingMatrix[i][j].item] = paraTestingMatrix[i][j].rating;
			} // Of for j

			// Step 3. Predict the ratings of the current user.
			for (int j = 0; j < paraNumItems; j++) {
				tempPredictions[j] = predict(i, j);
			} // Of for j

			// Step 4. Sort the predictions.
			int[] tempSortedIndices = mergeSortToIndices(tempPredictions);

			// Step 4. Statistics on like counts.
			int[] tempLikeCountArray = new int[tempSize];
			for (int j = 0; j < tempSize; j++) {
				int tempCounter = 0;
				for (int k = 0;; k++) {
					// Ignore known.

					if (tempKnownArray[tempSortedIndices[k]]) {
						continue;
					} // Of if

					if (tempRatings[tempSortedIndices[k]] > paraLikeThresholds[j] - 0.01) {
						tempLikeCountArray[j]++;
					} // Of if
					tempCounter++;
					if (tempCounter >= paraK) {
						break;
					} // Of if
				} // Of for k
			} // Of for j

			// Step 5. Statistics for each threshold.
			for (int j = 0; j < tempSize; j++) {
				double tempCorrect = 0;
				int tempCounter = 1;
				tempMap = 0;

				for (int k = 0;; k++) {
					// Ignore known.
					if (tempKnownArray[tempSortedIndices[k]]) {
						continue;
					} // Of if

					// Missing values.
					if (tempRatings[tempSortedIndices[k]] < -99.9) {
						;
					} else {
						// Like this one.
						if (tempRatings[tempSortedIndices[k]] > paraLikeThresholds[j] - 0.01) {
							tempCorrect ++;
							tempMap += tempCorrect / tempCounter;
						} // Of if
					} // Of if

					tempCounter++;
					if (tempCounter >= paraK) {
						break;
					} // Of if
				} // Of for k
				
				resultMapArray[j] += tempMap / tempNumUsers;
			} // Of for j
		} // Of for i
		return resultMapArray;
	}// Of map

	/**
	 ************************ 
	 * Compute NDCG of the dataset.
	 * 
	 * @param paraDataset
	 *            The given dataset.
	 * @param paraNumItems
	 *            The total number of items.
	 * @param paraLikeThresholds
	 *            A number of thresholds.
	 * @return NDCG.
	 ************************ 
	 */
	public double[] ndcg(Triple[][] paraTrainingMatrix, Triple[][] paraValidationMatrix,
			Triple[][] paraTestingMatrix, int paraNumItems, int paraK,
			double[] paraLikeThresholds) {
		int tempSize = paraLikeThresholds.length;
		int tempNumUsers = paraTrainingMatrix.length;

		double[] resultNdcgArray = new double[tempSize];
		double tempNdcg = 0;
		double[] tempPredictions = new double[paraNumItems];
		double[] tempRatings = new double[paraNumItems];
		boolean[] tempKnownArray = new boolean[paraNumItems];
		for (int i = 0; i < tempNumUsers; i++) {
			// System.out.println("User " + i);
			// Step 1. Which ratings are known.
			Arrays.fill(tempKnownArray, false);
			for (int j = 0; j < paraTrainingMatrix[i].length; j++) {
				tempKnownArray[paraTrainingMatrix[i][j].item] = true;
			} // Of for i
			for (int j = 0; j < paraValidationMatrix[i].length; j++) {
				tempKnownArray[paraValidationMatrix[i][j].item] = true;
			} // Of for i

			// Step 2. Expand ratings of the current user.
			Arrays.fill(tempRatings, -100);
			for (int j = 0; j < paraTestingMatrix[i].length; j++) {
				tempRatings[paraTestingMatrix[i][j].item] = paraTestingMatrix[i][j].rating;
			} // Of for j

			// Step 3. Predict the ratings of the current user.
			for (int j = 0; j < paraNumItems; j++) {
				tempPredictions[j] = predict(i, j);
			} // Of for j

			// Step 4. Sort the predictions.
			int[] tempSortedIndices = mergeSortToIndices(tempPredictions);

			// Step 4. Statistics on like counts.
			int[] tempLikeCountArray = new int[tempSize];
			for (int j = 0; j < tempSize; j++) {
				int tempCounter = 0;
				for (int k = 0;; k++) {
					// Ignore known.

					if (tempKnownArray[tempSortedIndices[k]]) {
						continue;
					} // Of if

					if (tempRatings[tempSortedIndices[k]] > paraLikeThresholds[j] - 0.01) {
						tempLikeCountArray[j]++;
					} // Of if
					tempCounter++;
					if (tempCounter >= paraK) {
						break;
					} // Of if
				} // Of for k
			} // Of for j

			// Step 5. Statistics for each threshold.
			for (int j = 0; j < tempSize; j++) {
				int tempLeftOut = 0;
				double tempDcg = 0;
				int tempRank = 0;

				double tempIdcg = idcg(tempLikeCountArray[j]);

				int tempCounter = 0;
				for (int k = 0;; k++) {
					// Ignore known.
					if (tempKnownArray[tempSortedIndices[k]]) {
						continue;
					} // Of if

					// Missing values.
					if (tempRatings[tempSortedIndices[k]] < -99.9) {
						tempLeftOut++;
					} else {
						// Like this one.
						if (tempRatings[tempSortedIndices[k]] > paraLikeThresholds[j] - 0.01) {
							tempRank = tempCounter + 1 - tempLeftOut;
							tempDcg += log2 / Math.log(tempRank + 1);
						} // Of if
					} // Of if

					tempCounter++;
					if (tempCounter >= paraK) {
						break;
					} // Of if
				} // Of for k

				if (tempIdcg == 0) {
					tempNdcg = 0;
				} else {
					tempNdcg = tempDcg / tempIdcg;
				} // Of if
				
				resultNdcgArray[j] += tempNdcg / tempNumUsers;
			} // Of for j
		} // Of for i
		return resultNdcgArray;
	}// Of ndcg
	/**
	 ************************ 
	 * Compute IDCG of the number.
	 * 
	 * @param paraNumber
	 *            the given number.
	 * @return The IDCG value.
	 ************************ 
	 */
	public static double idcg(int paraNumber) {
		double resultIdcg = 0;
		for (int i = 0; i < paraNumber; i++) {
			resultIdcg += log2 / Math.log(i + 2);
		} // Of for i

		return resultIdcg;
	}// Of idcg

	/**
	 ********************************** 
	 * Merge sort in descendant order to obtain an index array. The original
	 * array is unchanged.<br>
	 * This method is moved from SimpleTool.java.<br>
	 * Examples: input [1.2, 2.3, 0.4, 0.5], output [1, 0, 3, 2].<br>
	 * input [3.1, 5.2, 6.3, 2.1, 4.4], output [2, 1, 4, 0, 3].<br>
	 * This method is equivalent to argsort() in numpy module of the Python
	 * programming language.
	 * 
	 * @param paraArray
	 *            the original array
	 * @return The sorted indices.
	 ********************************** 
	 */
	public static int[] mergeSortToIndices(double[] paraArray) {
		int tempLength = paraArray.length;
		int[][] resultMatrix = new int[2][tempLength];// For merge sort.

		// Initialize
		int tempIndex = 0;
		for (int i = 0; i < tempLength; i++) {
			resultMatrix[tempIndex][i] = i;
		} // Of for i

		// Merge
		int tempCurrentLength = 1;
		// The indices for current merged groups.
		int tempFirstStart, tempSecondStart, tempSecondEnd;

		while (tempCurrentLength < tempLength) {
			// Divide into a number of groups.
			// Here the boundary is adaptive to array length not equal to 2^k.
			for (int i = 0; i < Math.ceil(tempLength + 0.0 / tempCurrentLength) / 2; i++) {
				// Boundaries of the group
				tempFirstStart = i * tempCurrentLength * 2;

				tempSecondStart = tempFirstStart + tempCurrentLength;

				tempSecondEnd = tempSecondStart + tempCurrentLength - 1;
				if (tempSecondEnd >= tempLength) {
					tempSecondEnd = tempLength - 1;
				} // Of if

				// Merge this group
				int tempFirstIndex = tempFirstStart;
				int tempSecondIndex = tempSecondStart;
				int tempCurrentIndex = tempFirstStart;

				if (tempSecondStart >= tempLength) {
					for (int j = tempFirstIndex; j < tempLength; j++) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][j];
						tempFirstIndex++;
						tempCurrentIndex++;
					} // Of for j
					break;
				} // Of if

				while ((tempFirstIndex <= tempSecondStart - 1)
						&& (tempSecondIndex <= tempSecondEnd)) {

					if (paraArray[resultMatrix[tempIndex
							% 2][tempFirstIndex]] >= paraArray[resultMatrix[tempIndex
									% 2][tempSecondIndex]]) {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempFirstIndex];
						tempFirstIndex++;
					} else {
						resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
								% 2][tempSecondIndex];
						tempSecondIndex++;
					} // Of if
					tempCurrentIndex++;
				} // Of while

				// Remaining part
				for (int j = tempFirstIndex; j < tempSecondStart; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
							% 2][j];
					tempCurrentIndex++;
				} // Of for j
				for (int j = tempSecondIndex; j <= tempSecondEnd; j++) {
					resultMatrix[(tempIndex + 1) % 2][tempCurrentIndex] = resultMatrix[tempIndex
							% 2][j];
					tempCurrentIndex++;
				} // Of for j
			} // Of for i

			tempCurrentLength *= 2;
			tempIndex++;
		} // Of while

		return resultMatrix[tempIndex % 2];
	}// Of mergeSortToIndices
}// Of class SimpleMatrixFactorization

package algorithm;

import datamodel.Triple;

/*
 * Matrix factorization with sigmoid regulation.
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class SigmoidExponentMF extends SimpleMatrixFactorization {

	/**
	 * The constant for controlling the shape.
	 */
	double constantC = 1.0;

	double lambda = 1.0;

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
	public SigmoidExponentMF(Triple[][] paraTrainingSet, Triple[][] paraValidationSet, int paraNumUsers,
			int paraNumItems, double paraRatingLowerBound, double paraRatingUpperBound) {
		super(paraTrainingSet, paraValidationSet, paraNumUsers, paraNumItems, paraRatingLowerBound,
				paraRatingUpperBound);
	}// Of the constructor

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 ************************ 
	 */
	public void update() {
		// Step1: update subU
		// double tempC = parameters.mxNormalization;
		double tempQij; // The residual
		double tempExp;
		double tempCoefficent;
		boolean tempSign = true; // Positive
		for (int i = 0; i < trainingSet.length; i++) {
			for (int j = 0; j < trainingSet[i].length; j++) {
				int tempUserId = trainingSet[i][j].user;
				int tempItemId = trainingSet[i][j].item;
				double tempRate = trainingSet[i][j].rating;

				tempQij = tempRate - predict(tempUserId, tempItemId);
				if (tempQij >= 0) {
					tempSign = true;
				} else {
					tempSign = false;
					tempQij = -tempQij;
				} // Of if

				tempExp = Math.exp(-Math.pow(tempQij, lambda) / constantC);

				tempCoefficent = -8 * lambda * (1 - tempExp) * tempExp * Math.pow(tempQij, lambda - 1) / constantC
						/ Math.pow(1 + tempExp, 3);
				if (!tempSign) {
					tempCoefficent = -tempCoefficent;
				} // Of if

				// Update user subspace
				for (int k = 0; k < rank; k++) {
					userSubspace[tempUserId][k] -= alpha * (tempCoefficent * itemSubspace[tempItemId][k]);
				} // Of for k

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					itemSubspace[tempItemId][k] -= alpha * (tempCoefficent * userSubspace[tempUserId][k]);
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of update

	/**
	 ************************ 
	 * Setter.
	 ************************ 
	 */
	public void setConstantC(double paraC) {
		constantC = paraC;
	}// Of setConstantC

	/**
	 ************************ 
	 * Setter.
	 ************************ 
	 */
	public void setLambda(double paraLambda) {
		lambda = paraLambda;
	}// Of setConstantC
}// Of class SigmoidExponentMF

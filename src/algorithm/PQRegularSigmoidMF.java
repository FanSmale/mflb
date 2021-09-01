package algorithm;

import datamodel.Triple;

/*
 * Matrix factorization with sigmoid regulation.
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class PQRegularSigmoidMF extends SimpleMatrixFactorization {

	/**
	 * The constant for controlling the shape.
	 */
	double constantC = 1.4;

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
	public PQRegularSigmoidMF(Triple[][] paraTrainingSet, Triple[][] paraValidationSet,
			int paraNumUsers, int paraNumItems, double paraRatingLowerBound,
			double paraRatingUpperBound) {
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
		double tempQij = 0; // The residual
		double tempExp = 0;
		double tempCoefficent = 0;
		//boolean tempSign = true; // Positive
		for (int i = 0; i < trainingSet.length; i++) {
			for (int j = 0; j < trainingSet[i].length; j++) {
				int tempUserId = trainingSet[i][j].user;
				int tempItemId = trainingSet[i][j].item;
				double tempRate = trainingSet[i][j].rating;

				tempQij = tempRate - predict(tempUserId, tempItemId);

				tempExp = Math.exp(-tempQij / constantC);
				tempCoefficent = 8 * (1 - tempExp) * tempExp / constantC
						/ Math.pow(1 + tempExp, 3);

				double tempValue = 0;
				// Update user subspace
				for (int k = 0; k < rank; k++) {
					tempValue = tempCoefficent * itemSubspace[tempItemId][k]
							- lambda * userSubspace[tempUserId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for k

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = tempCoefficent * userSubspace[tempUserId][k]
							- lambda * itemSubspace[tempItemId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
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
}// Of class PQRegularSigmoidMF

package algorithm;

import datamodel.Triple;

/*
 * Matrix factorization with PQ regulation.
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class PQRegularMF extends SimpleMatrixFactorization {

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
	public PQRegularMF(Triple[][] paraTrainingSet, Triple[][] paraValidationSet, int paraNumUsers,
			int paraNumItems, double paraRatingLowerBound, double paraRatingUpperBound) {
		super(paraTrainingSet, paraValidationSet, paraNumUsers, paraNumItems, paraRatingLowerBound,
				paraRatingUpperBound);
	}// Of the constructor

	/**
	 ************************ 
	 * Update sub-spaces using the training data.
	 * 
	 ************************ 
	 */
	public void update() {
		for (int i = 0; i < trainingSet.length; i++) {
			for (int j = 0; j < trainingSet[i].length; j++) {
				int tempUserId = trainingSet[i][j].user;
				int tempItemId = trainingSet[i][j].item;
				double tempRate = trainingSet[i][j].rating;

				double tempResidual = tempRate - predict(tempUserId, tempItemId); // Residual
				// tempResidual = Math.abs(tempResidual);

				// Update user subspace
				double tempValue = 0;
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * itemSubspace[tempItemId][k]
							- lambda * userSubspace[tempUserId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = 2 * tempResidual * userSubspace[tempUserId][k]
							- lambda * itemSubspace[tempItemId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of update
}// Of class PQRegularMF

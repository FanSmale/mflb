package algorithm;

import datamodel.Triple;

/*
 * Matrix factorization with ONE regulation (consult Henry for its difference with PQ regular).
 * 
 * @author Fan Min minfanphd@163.com.
 */
public class OneRegularMF extends SimpleMatrixFactorization {

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
	public OneRegularMF(Triple[][] paraTrainingSet, Triple[][] paraValidationSet, int paraNumUsers,
			int paraNumItems, double paraRatingLowerBound, double paraRatingUpperBound) {
		super(paraTrainingSet, paraValidationSet, paraNumUsers, paraNumItems, paraRatingLowerBound,
				paraRatingUpperBound);
	}// Of the constructor

	/**
	 ************************ 
	 * Update sub-spaces using the one-regular approach.
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
					tempValue = tempResidual * itemSubspace[tempItemId][k]
							- lambda * userSubspace[tempUserId][k];
					userSubspace[tempUserId][k] += alpha * tempValue;
				} // Of for j

				// Update item subspace
				for (int k = 0; k < rank; k++) {
					tempValue = tempResidual * userSubspace[tempUserId][k]
							- lambda * itemSubspace[tempItemId][k];
					itemSubspace[tempItemId][k] += alpha * tempValue;
				} // Of for k
			} // Of for j
		} // Of for i
	}// Of update
}// Of class OneRegularMF

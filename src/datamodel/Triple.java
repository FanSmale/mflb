package datamodel;

public class Triple {
	public int user;
	public int item;
	public double rating;
	
	/**
	 *********************
	 * The constructor.
	 *********************
	 */
	public Triple(int paraUser, int paraItem, double paraRating){
		user = paraUser;
		item = paraItem;
		rating = paraRating;
	}//Of the first constructor

	/**
	 *********************
	 * The second constructor.
	 *********************
	 */
	public Triple(Triple paraTriple){
		user = paraTriple.user;
		item = paraTriple.item;
		rating = paraTriple.rating;
	}//Of the sceond constructor

	/**
	 *********************
	 * Show me.
	 *********************
	 */
	public String toString() {
		return "" + user + ", " + item + ", " + rating;
	}//Of toString

}// of class Triple

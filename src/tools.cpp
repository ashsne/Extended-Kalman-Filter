#include "tools.h"
#include <iostream>
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
	//Calculate the RMSE here.
	VectorXd RMSE(4);
	RMSE << 0,0,0,0;

	size_t size_est = estimations.size();
	size_t size_grd = ground_truth.size();

	if(size_est != size_grd || size_est < 1)
	{
		cout<<"Cannot compute RMSE. Invalid input size" <<endl;
	}


	for(size_t i=0;i< estimations.size();i++)
	{
		VectorXd err = estimations[i] - ground_truth[i];
		err = err.array() * err.array();
		RMSE += err;
	}

	RMSE = RMSE/estimations.size();
	RMSE = RMSE.array().sqrt();
	return RMSE;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) 
{
	/**
	* Calculate a Jacobian here.
	*/
	MatrixXd Hj(3,4);
	
	// Get the state parameters
	float px = x_state[0];
	float py = x_state[1];
	float vx = x_state[2];
	float vy = x_state[3];

	// Compute few constants required repeatedly
	float c1 = px*px + py*py;
	float c2 = sqrt(c1);
	float c3 = c1 * c2;

	// check division by zero
	if (fabs(c1) < 0.0001)
	{
		cout<< "CalculateJacobian() - Error - Division by zero" <<endl;
		return Hj;
	}
	
	//compute jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
			-(py/c1), (px/c1), 0, 0,
			py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;

}

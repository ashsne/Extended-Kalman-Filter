#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() 
{}

KalmanFilter::~KalmanFilter() 
{}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) 
{
	x_ = x_in;
	P_ = P_in;
	F_ = F_in;
	H_ = H_in;
	R_ = R_in;
	Q_ = Q_in;
}

void KalmanFilter::Predict() 
{
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ *P_* Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) 
{
	// Calculate the Kalman gain matrix
	VectorXd y = z - H_ * x_;

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_*P_*Ht + R_;
	MatrixXd Sinv = S.inverse();
	MatrixXd K = P_*Ht*Sinv;

	//Update the state vector and the covariance matrix
	x_ = x_ + K*y;
	int x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K*H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
	// Get the state parameters
	float px = x_[0];
	float py = x_[1];
	float vx = x_[2];
	float vy = x_[3];

	double ro = sqrt(px*px + py*py);
	double theta = atan2(py,px);
	//ro_dot can be calculated as velocity projection in the direction of ro
	double ro_dot = (vx*px + vy*py)/std::max(ro, 0.0001);

	VectorXd z_pred(3);
	z_pred << ro, theta, ro_dot;
	VectorXd y = z - z_pred;

	// normalize angles so that they wil always be in (-pi, pi)
	while (y(1)>M_PI) y(1) = y(1) - 2*M_PI;
	while (y(1)<-M_PI) y(1) = y(1) + 2*M_PI;

	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_*P_*Ht + R_;
	MatrixXd Sinv = S.inverse();
	MatrixXd K = P_*Ht*Sinv;

	//Update the state vector and the covariance matrix
	x_ = x_ + K*y;
	int x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K*H_) * P_;

}

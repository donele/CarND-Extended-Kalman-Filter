#include "KFRadarIter.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KFRadarIter::KFRadarIter() {
  // measurement matrix
  H_ = MatrixXd::Zero(3, 4);
  H_trans_ = MatrixXd::Zero(4, 3);

  // measurement noise
  R_ = MatrixXd(3, 3);
  R_ << 0.09, 0,      0,
        0,    0.0009, 0,
        0,    0,      0.09;
}

KFRadarIter::~KFRadarIter() {}

void KFRadarIter::Update(KFState& state, float dt, const VectorXd &z) {
  VectorXd xi = state.x;
  VectorXd xi_prev = xi;
  MatrixXd K;
  VectorXd y;

  while(1) {
    // Calculate jacobian
    H_ = tools.CalculateJacobian(xi);
    H_trans_ = H_.transpose();

    // Calculate Kalman gain
    K = state.P * H_trans_ * (H_ * state.P * H_trans_ + R_).inverse();

    // Update state
    y = z - tools.Cartesian2Polar(xi) - H_ * (state.x - xi);
    tools.NormalizeAngle(y);
    xi = state.x + K * y;

    // Stop iteration
    if((xi - xi_prev).norm() < .1)
      break;
    else
      xi_prev = xi;
  }

  // Update state and covariance matrices
  state.x = xi;
  state.P -= K * H_ * state.P;

  return;
}

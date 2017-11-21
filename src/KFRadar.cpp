#include "KFRadar.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KFRadar::KFRadar() {
  // measurement matrix
  H_ = MatrixXd::Zero(3, 4);
  H_trans_ = MatrixXd::Zero(4, 3);

  // measurement noise
  R_ = MatrixXd(3, 3);
  R_ << 0.09, 0,      0,
        0,    0.0009, 0,
        0,    0,      0.09;
}

KFRadar::~KFRadar() {}

void KFRadar::Update(KFState& state, float dt, const VectorXd &z) {
  // Calculate jacobian
  H_ = tools.CalculateJacobian(state.x);
  H_trans_ = H_.transpose();

  // Calculate Kalman gain
  MatrixXd K = state.P * H_trans_ * (H_ * state.P * H_trans_ + R_).inverse();

  // Update state and covariance matrices
  VectorXd y = z - tools.Cartesian2Polar(state.x);
  tools.NormalizeAngle(y);
  state.x += K * y;
  state.P -= K * H_ * state.P;

  return;
}

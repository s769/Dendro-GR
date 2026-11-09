    bssn::timer::t_rhs.start();
for (unsigned int k = 3; k < nz-3; k++) { 
    z = pmin[2] + k*hz;
for (unsigned int j = 3; j < ny-3; j++) { 
    y = pmin[1] + j*hy; 
for (unsigned int i = 3; i < nx-3; i++) {
    x = pmin[0] + i*hx;
    pp = i + nx*(j + ny*k);
    r_coord = sqrt(x*x + y*y + z*z);
    eta=ETA_CONST;
    if (r_coord >= ETA_R0) {
    eta *= pow( (ETA_R0/r_coord), ETA_DAMPING_EXP);
    }
// Dendro: {{{ 
// Dendro: original ops:  873
// Dendro: printing temp variables
double DENDRO_0 = pow(gt4[pp], 2);
double DENDRO_1 = pow(gt1[pp], 2);
double DENDRO_2 = pow(gt2[pp], 2);
double DENDRO_3 = gt0[pp]*gt5[pp];
double DENDRO_4 = gt1[pp]*gt2[pp];
double DENDRO_5 = 1.0/(DENDRO_0*gt0[pp] + DENDRO_1*gt5[pp] + DENDRO_2*gt3[pp] - DENDRO_3*gt3[pp] - 2*DENDRO_4*gt4[pp]);
double DENDRO_6 = gt1[pp]*gt5[pp] - gt2[pp]*gt4[pp];
double DENDRO_7 = gt1[pp]*gt4[pp] - gt2[pp]*gt3[pp];
double DENDRO_8 = (7.0/3.0)*DENDRO_7;
double DENDRO_9 = -DENDRO_2 + DENDRO_3;
double DENDRO_10 = -DENDRO_4 + gt0[pp]*gt4[pp];
double DENDRO_11 = (1.0/3.0)*DENDRO_7;
double DENDRO_12 = -DENDRO_1 + gt0[pp]*gt3[pp];
double DENDRO_13 = -DENDRO_0 + gt3[pp]*gt5[pp];
double DENDRO_14 = (1.0/3.0)*DENDRO_13;
double DENDRO_15 = (1.0/3.0)*DENDRO_6;
double DENDRO_16 = (1.0/3.0)*DENDRO_10;
double DENDRO_17 = (1.0/3.0)*DENDRO_9;
double DENDRO_18 = (1.0/3.0)*DENDRO_12;
// Dendro: printing variables
//--
Gt_rhs_s4_0[pp] = DENDRO_5*(2*DENDRO_10*grad2_1_2_beta0[pp] - DENDRO_11*grad2_1_2_beta1[pp] - DENDRO_11*grad2_2_2_beta2[pp] - DENDRO_12*grad2_2_2_beta0[pp] - 4.0/3.0*DENDRO_13*grad2_0_0_beta0[pp] - DENDRO_14*grad2_0_1_beta1[pp] - DENDRO_14*grad2_0_2_beta2[pp] + (7.0/3.0)*DENDRO_6*grad2_0_1_beta0[pp] + (1.0/3.0)*DENDRO_6*grad2_1_1_beta1[pp] + (1.0/3.0)*DENDRO_6*grad2_1_2_beta2[pp] - DENDRO_8*grad2_0_2_beta0[pp] - DENDRO_9*grad2_1_1_beta0[pp]);
//--
Gt_rhs_s4_1[pp] = DENDRO_5*((7.0/3.0)*DENDRO_10*grad2_1_2_beta1[pp] - DENDRO_12*grad2_2_2_beta1[pp] - DENDRO_13*grad2_0_0_beta1[pp] + DENDRO_15*grad2_0_0_beta0[pp] + DENDRO_15*grad2_0_2_beta2[pp] + DENDRO_16*grad2_0_2_beta0[pp] + DENDRO_16*grad2_2_2_beta2[pp] - DENDRO_17*grad2_0_1_beta0[pp] - DENDRO_17*grad2_1_2_beta2[pp] + (7.0/3.0)*DENDRO_6*grad2_0_1_beta1[pp] - 2*DENDRO_7*grad2_0_2_beta1[pp] - 4.0/3.0*DENDRO_9*grad2_1_1_beta1[pp]);
//--
Gt_rhs_s4_2[pp] = DENDRO_5*((1.0/3.0)*DENDRO_10*grad2_0_1_beta0[pp] + (1.0/3.0)*DENDRO_10*grad2_1_1_beta1[pp] + (7.0/3.0)*DENDRO_10*grad2_1_2_beta2[pp] - DENDRO_11*grad2_0_0_beta0[pp] - DENDRO_11*grad2_0_1_beta1[pp] - 4.0/3.0*DENDRO_12*grad2_2_2_beta2[pp] - DENDRO_13*grad2_0_0_beta2[pp] - DENDRO_18*grad2_0_2_beta0[pp] - DENDRO_18*grad2_1_2_beta1[pp] + 2*DENDRO_6*grad2_0_1_beta2[pp] - DENDRO_8*grad2_0_2_beta2[pp] - DENDRO_9*grad2_1_1_beta2[pp]);
// Dendro: reduced ops:  130
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

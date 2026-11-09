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
// Dendro: original ops:  1626
// Dendro: printing temp variables
double DENDRO_0 = pow(gt4[pp], 2);
double DENDRO_1 = pow(gt1[pp], 2);
double DENDRO_2 = pow(gt2[pp], 2);
double DENDRO_3 = gt0[pp]*gt3[pp];
double DENDRO_4 = gt2[pp]*gt4[pp];
double DENDRO_5 = pow(DENDRO_0*gt0[pp] + DENDRO_1*gt5[pp] + DENDRO_2*gt3[pp] - DENDRO_3*gt5[pp] - 2*DENDRO_4*gt1[pp], -2);
double DENDRO_6 = gt0[pp]*gt4[pp] - gt1[pp]*gt2[pp];
double DENDRO_7 = gt1[pp]*gt4[pp] - gt2[pp]*gt3[pp];
double DENDRO_8 = -DENDRO_4 + gt1[pp]*gt5[pp];
double DENDRO_9 = -grad_0_gt4[pp] + grad_1_gt2[pp] + grad_2_gt1[pp];
double DENDRO_10 = -DENDRO_0 + gt3[pp]*gt5[pp];
double DENDRO_11 = grad_0_gt4[pp] - grad_1_gt2[pp] + grad_2_gt1[pp];
double DENDRO_12 = 1.0*DENDRO_7;
double DENDRO_13 = grad_0_gt4[pp] + grad_1_gt2[pp] - grad_2_gt1[pp];
double DENDRO_14 = -DENDRO_1 + DENDRO_3;
double DENDRO_15 = 0.5*grad_2_gt5[pp];
double DENDRO_16 = 0.5*grad_1_gt5[pp] - 1.0*grad_2_gt4[pp];
double DENDRO_17 = 0.5*grad_0_gt5[pp] - 1.0*grad_2_gt2[pp];
double DENDRO_18 = -DENDRO_2 + gt0[pp]*gt5[pp];
double DENDRO_19 = 0.5*grad_1_gt3[pp];
double DENDRO_20 = 0.5*grad_0_gt3[pp] - 1.0*grad_1_gt1[pp];
double DENDRO_21 = 1.0*grad_1_gt4[pp] - 0.5*grad_2_gt3[pp];
double DENDRO_22 = 0.5*grad_0_gt0[pp];
double DENDRO_23 = 1.0*grad_0_gt2[pp] - 0.5*grad_2_gt0[pp];
double DENDRO_24 = 1.0*grad_0_gt1[pp] - 0.5*grad_1_gt0[pp];
// Dendro: printing variables
//--
CalGt0[pp] = DENDRO_5*(-DENDRO_10*(-DENDRO_10*DENDRO_22 - DENDRO_23*DENDRO_7 + DENDRO_24*DENDRO_8) - DENDRO_12*(-DENDRO_10*grad_2_gt0[pp] + DENDRO_11*DENDRO_8 - DENDRO_7*grad_0_gt5[pp]) - DENDRO_14*(DENDRO_10*DENDRO_17 - DENDRO_15*DENDRO_7 - DENDRO_16*DENDRO_8) - DENDRO_18*(DENDRO_10*DENDRO_20 + DENDRO_19*DENDRO_8 - DENDRO_21*DENDRO_7) + 1.0*DENDRO_6*(-DENDRO_10*DENDRO_9 - DENDRO_7*grad_1_gt5[pp] + DENDRO_8*grad_2_gt3[pp]) + 1.0*DENDRO_8*(-DENDRO_10*grad_1_gt0[pp] - DENDRO_13*DENDRO_7 + DENDRO_8*grad_0_gt3[pp]));
//--
CalGt1[pp] = DENDRO_5*(-DENDRO_10*(-DENDRO_18*DENDRO_24 + DENDRO_22*DENDRO_8 + DENDRO_23*DENDRO_6) - DENDRO_12*(-DENDRO_11*DENDRO_18 + DENDRO_6*grad_0_gt5[pp] + DENDRO_8*grad_2_gt0[pp]) - DENDRO_14*(DENDRO_15*DENDRO_6 + DENDRO_16*DENDRO_18 - DENDRO_17*DENDRO_8) - DENDRO_18*(-DENDRO_18*DENDRO_19 - DENDRO_20*DENDRO_8 + DENDRO_21*DENDRO_6) + 1.0*DENDRO_6*(-DENDRO_18*grad_2_gt3[pp] + DENDRO_6*grad_1_gt5[pp] + DENDRO_8*DENDRO_9) + 1.0*DENDRO_8*(DENDRO_13*DENDRO_6 - DENDRO_18*grad_0_gt3[pp] + DENDRO_8*grad_1_gt0[pp]));
//--
CalGt2[pp] = DENDRO_5*(-DENDRO_10*(-DENDRO_14*DENDRO_23 - DENDRO_22*DENDRO_7 + DENDRO_24*DENDRO_6) - DENDRO_12*(DENDRO_11*DENDRO_6 - DENDRO_14*grad_0_gt5[pp] - DENDRO_7*grad_2_gt0[pp]) - DENDRO_14*(-DENDRO_14*DENDRO_15 - DENDRO_16*DENDRO_6 + DENDRO_17*DENDRO_7) - DENDRO_18*(-DENDRO_14*DENDRO_21 + DENDRO_19*DENDRO_6 + DENDRO_20*DENDRO_7) + 1.0*DENDRO_6*(-DENDRO_14*grad_1_gt5[pp] + DENDRO_6*grad_2_gt3[pp] - DENDRO_7*DENDRO_9) + 1.0*DENDRO_8*(-DENDRO_13*DENDRO_14 + DENDRO_6*grad_0_gt3[pp] - DENDRO_7*grad_1_gt0[pp]));
// Dendro: reduced ops:  190
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

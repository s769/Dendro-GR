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
// Dendro: original ops:  1905
// Dendro: printing temp variables
double DENDRO_0 = gt2[pp]*gt4[pp];
double DENDRO_1 = -DENDRO_0 + gt1[pp]*gt5[pp];
double DENDRO_2 = (DENDRO_1 * DENDRO_1);
double DENDRO_3 = gt1[pp]*gt4[pp] - gt2[pp]*gt3[pp];
double DENDRO_4 = (DENDRO_3 * DENDRO_3);
double DENDRO_5 = gt3[pp]*gt5[pp];
double DENDRO_6 = pow(gt4[pp], 2);
double DENDRO_7 = DENDRO_5 - DENDRO_6;
double DENDRO_8 = DENDRO_1*DENDRO_3;
double DENDRO_9 = 2*At4[pp];
double DENDRO_10 = At1[pp]*DENDRO_7;
double DENDRO_11 = 2*DENDRO_1;
double DENDRO_12 = At2[pp]*DENDRO_7;
double DENDRO_13 = 2*DENDRO_3;
double DENDRO_14 = gt0[pp]*gt4[pp] - gt1[pp]*gt2[pp];
double DENDRO_15 = At3[pp]*DENDRO_1;
double DENDRO_16 = At0[pp]*DENDRO_7;
double DENDRO_17 = At4[pp]*DENDRO_3;
double DENDRO_18 = pow(gt1[pp], 2);
double DENDRO_19 = -DENDRO_18 + gt0[pp]*gt3[pp];
double DENDRO_20 = At5[pp]*DENDRO_3;
double DENDRO_21 = At4[pp]*DENDRO_1;
double DENDRO_22 = -At1[pp]*DENDRO_8 + At2[pp]*DENDRO_4 - DENDRO_10*DENDRO_14 + DENDRO_12*DENDRO_19 + DENDRO_14*DENDRO_15 - DENDRO_14*DENDRO_17 + DENDRO_16*DENDRO_3 + DENDRO_19*DENDRO_20 - DENDRO_19*DENDRO_21;
double DENDRO_23 = pow(gt2[pp], 2);
double DENDRO_24 = -DENDRO_23 + gt0[pp]*gt5[pp];
double DENDRO_25 = -At1[pp]*DENDRO_2 + At2[pp]*DENDRO_8 + DENDRO_1*DENDRO_16 - DENDRO_10*DENDRO_24 + DENDRO_12*DENDRO_14 + DENDRO_14*DENDRO_20 - DENDRO_14*DENDRO_21 + DENDRO_15*DENDRO_24 - DENDRO_17*DENDRO_24;
double DENDRO_26 = 2/pow(-2*DENDRO_0*gt1[pp] + DENDRO_18*gt5[pp] + DENDRO_23*gt3[pp] - DENDRO_5*gt0[pp] + DENDRO_6*gt0[pp], 2);
double DENDRO_27 = (DENDRO_14 * DENDRO_14);
double DENDRO_28 = At2[pp]*DENDRO_1;
double DENDRO_29 = At1[pp]*DENDRO_24;
double DENDRO_30 = DENDRO_14*DENDRO_24;
double DENDRO_31 = DENDRO_14*DENDRO_3;
double DENDRO_32 = DENDRO_14*DENDRO_19;
double DENDRO_33 = At0[pp]*DENDRO_8 - At1[pp]*DENDRO_1*DENDRO_14 + At2[pp]*DENDRO_31 + At3[pp]*DENDRO_30 - At4[pp]*DENDRO_19*DENDRO_24 - At4[pp]*DENDRO_27 + At5[pp]*DENDRO_32 + DENDRO_19*DENDRO_28 - DENDRO_29*DENDRO_3;
// Dendro: printing variables
//--
Gt_rhs_s5_0[pp] = DENDRO_26*(DENDRO_22*grad_2_alpha[pp] - DENDRO_25*grad_1_alpha[pp] + grad_0_alpha[pp]*(At0[pp]*(DENDRO_7 * DENDRO_7) + At3[pp]*DENDRO_2 + At5[pp]*DENDRO_4 - DENDRO_10*DENDRO_11 + DENDRO_12*DENDRO_13 - DENDRO_8*DENDRO_9));
//--
Gt_rhs_s5_1[pp] = DENDRO_26*(-DENDRO_25*grad_0_alpha[pp] - DENDRO_33*grad_2_alpha[pp] + grad_1_alpha[pp]*(At0[pp]*DENDRO_2 + At3[pp]*(DENDRO_24 * DENDRO_24) + At5[pp]*DENDRO_27 - DENDRO_11*DENDRO_29 + 2*DENDRO_14*DENDRO_28 - DENDRO_30*DENDRO_9));
//--
Gt_rhs_s5_2[pp] = DENDRO_26*(DENDRO_22*grad_0_alpha[pp] - DENDRO_33*grad_1_alpha[pp] + grad_2_alpha[pp]*(At0[pp]*DENDRO_4 - 2*At1[pp]*DENDRO_31 + At2[pp]*DENDRO_13*DENDRO_19 + At3[pp]*DENDRO_27 + At5[pp]*(DENDRO_19 * DENDRO_19) - DENDRO_32*DENDRO_9));
// Dendro: reduced ops:  159
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

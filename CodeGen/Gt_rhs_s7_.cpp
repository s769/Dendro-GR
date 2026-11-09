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
// Dendro: original ops:  2103
// Dendro: printing temp variables
double DENDRO_0 = gt1[pp]*gt5[pp] - gt2[pp]*gt4[pp];
double DENDRO_1 = gt1[pp]*gt4[pp];
double DENDRO_2 = DENDRO_1 - gt2[pp]*gt3[pp];
double DENDRO_3 = (4.0/3.0)*grad_2_K[pp];
double DENDRO_4 = gt3[pp]*gt5[pp];
double DENDRO_5 = pow(gt4[pp], 2);
double DENDRO_6 = DENDRO_4 - DENDRO_5;
double DENDRO_7 = (4.0/3.0)*grad_0_K[pp];
double DENDRO_8 = 1.0/chi[pp];
double DENDRO_9 = pow(gt1[pp], 2);
double DENDRO_10 = pow(gt2[pp], 2);
double DENDRO_11 = 1.0/(-2*DENDRO_1*gt2[pp] + DENDRO_10*gt3[pp] - DENDRO_4*gt0[pp] + DENDRO_5*gt0[pp] + DENDRO_9*gt5[pp]);
double DENDRO_12 = (DENDRO_0 * DENDRO_0);
double DENDRO_13 = (DENDRO_2 * DENDRO_2);
double DENDRO_14 = DENDRO_0*DENDRO_2;
double DENDRO_15 = 2*At4[pp];
double DENDRO_16 = At1[pp]*DENDRO_6;
double DENDRO_17 = 2*DENDRO_0;
double DENDRO_18 = At2[pp]*DENDRO_6;
double DENDRO_19 = 2*DENDRO_2;
double DENDRO_20 = gt0[pp]*gt4[pp] - gt1[pp]*gt2[pp];
double DENDRO_21 = At5[pp]*DENDRO_2;
double DENDRO_22 = At0[pp]*DENDRO_6;
double DENDRO_23 = -DENDRO_10 + gt0[pp]*gt5[pp];
double DENDRO_24 = At3[pp]*DENDRO_0;
double DENDRO_25 = At4[pp]*DENDRO_0;
double DENDRO_26 = At4[pp]*DENDRO_2;
double DENDRO_27 = 3*DENDRO_11*DENDRO_8;
double DENDRO_28 = DENDRO_27*(-At1[pp]*DENDRO_12 + At2[pp]*DENDRO_14 + DENDRO_0*DENDRO_22 - DENDRO_16*DENDRO_23 + DENDRO_18*DENDRO_20 + DENDRO_20*DENDRO_21 - DENDRO_20*DENDRO_25 + DENDRO_23*DENDRO_24 - DENDRO_23*DENDRO_26);
double DENDRO_29 = -DENDRO_9 + gt0[pp]*gt3[pp];
double DENDRO_30 = -At1[pp]*DENDRO_14 + At2[pp]*DENDRO_13 - DENDRO_16*DENDRO_20 + DENDRO_18*DENDRO_29 + DENDRO_2*DENDRO_22 + DENDRO_20*DENDRO_24 - DENDRO_20*DENDRO_26 + DENDRO_21*DENDRO_29 - DENDRO_25*DENDRO_29;
double DENDRO_31 = DENDRO_11*alpha[pp];
double DENDRO_32 = (DENDRO_20 * DENDRO_20);
double DENDRO_33 = At2[pp]*DENDRO_0;
double DENDRO_34 = At1[pp]*DENDRO_23;
double DENDRO_35 = DENDRO_20*DENDRO_23;
double DENDRO_36 = DENDRO_27*grad_1_chi[pp];
double DENDRO_37 = DENDRO_2*DENDRO_20;
double DENDRO_38 = DENDRO_20*DENDRO_29;
double DENDRO_39 = At0[pp]*DENDRO_14 - At1[pp]*DENDRO_0*DENDRO_20 + At2[pp]*DENDRO_37 + At3[pp]*DENDRO_35 - At4[pp]*DENDRO_23*DENDRO_29 - At4[pp]*DENDRO_32 + At5[pp]*DENDRO_38 - DENDRO_2*DENDRO_34 + DENDRO_29*DENDRO_33;
// Dendro: printing variables
//--
Gt_rhs_s7_0[pp] = DENDRO_31*((4.0/3.0)*DENDRO_0*grad_1_K[pp] + 3*DENDRO_11*DENDRO_30*DENDRO_8*grad_2_chi[pp] + 3*DENDRO_11*DENDRO_8*grad_0_chi[pp]*(At0[pp]*(DENDRO_6 * DENDRO_6) + At3[pp]*DENDRO_12 + At5[pp]*DENDRO_13 - DENDRO_14*DENDRO_15 - DENDRO_16*DENDRO_17 + DENDRO_18*DENDRO_19) - DENDRO_2*DENDRO_3 - DENDRO_28*grad_1_chi[pp] - DENDRO_6*DENDRO_7);
//--
Gt_rhs_s7_1[pp] = DENDRO_31*(DENDRO_0*DENDRO_7 + DENDRO_20*DENDRO_3 - 4.0/3.0*DENDRO_23*grad_1_K[pp] - DENDRO_27*DENDRO_39*grad_2_chi[pp] - DENDRO_28*grad_0_chi[pp] + DENDRO_36*(At0[pp]*DENDRO_12 + At3[pp]*(DENDRO_23 * DENDRO_23) + At5[pp]*DENDRO_32 - DENDRO_15*DENDRO_35 - DENDRO_17*DENDRO_34 + 2*DENDRO_20*DENDRO_33));
//--
Gt_rhs_s7_2[pp] = DENDRO_31*(3*DENDRO_11*DENDRO_30*DENDRO_8*grad_0_chi[pp] + 3*DENDRO_11*DENDRO_8*grad_2_chi[pp]*(At0[pp]*DENDRO_13 - 2*At1[pp]*DENDRO_37 + At2[pp]*DENDRO_19*DENDRO_29 + At3[pp]*DENDRO_32 + At5[pp]*(DENDRO_29 * DENDRO_29) - DENDRO_15*DENDRO_38) - DENDRO_2*DENDRO_7 + (4.0/3.0)*DENDRO_20*grad_1_K[pp] - DENDRO_29*DENDRO_3 - DENDRO_36*DENDRO_39);
// Dendro: reduced ops:  205
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

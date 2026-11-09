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
// Dendro: original ops:  4722
// Dendro: printing temp variables
double DENDRO_0 = gt3[pp]*gt5[pp];
double DENDRO_1 = pow(gt4[pp], 2);
double DENDRO_2 = DENDRO_0 - DENDRO_1;
double DENDRO_3 = 0.5*grad_0_gt0[pp];
double DENDRO_4 = 1.0*grad_0_gt2[pp] - 0.5*grad_2_gt0[pp];
double DENDRO_5 = gt1[pp]*gt4[pp];
double DENDRO_6 = DENDRO_5 - gt2[pp]*gt3[pp];
double DENDRO_7 = 1.0*grad_0_gt1[pp] - 0.5*grad_1_gt0[pp];
double DENDRO_8 = gt1[pp]*gt5[pp] - gt2[pp]*gt4[pp];
double DENDRO_9 = (DENDRO_8 * DENDRO_8);
double DENDRO_10 = (DENDRO_6 * DENDRO_6);
double DENDRO_11 = At4[pp]*DENDRO_6;
double DENDRO_12 = 2*DENDRO_8;
double DENDRO_13 = At1[pp]*DENDRO_8;
double DENDRO_14 = DENDRO_2*DENDRO_6;
double DENDRO_15 = At0[pp]*(DENDRO_2 * DENDRO_2) + 2*At2[pp]*DENDRO_14 + At3[pp]*DENDRO_9 + At5[pp]*DENDRO_10 - DENDRO_11*DENDRO_12 - 2*DENDRO_13*DENDRO_2;
double DENDRO_16 = 0.5*grad_2_gt5[pp];
double DENDRO_17 = 0.5*grad_1_gt5[pp] - 1.0*grad_2_gt4[pp];
double DENDRO_18 = 0.5*grad_0_gt5[pp] - 1.0*grad_2_gt2[pp];
double DENDRO_19 = gt0[pp]*gt4[pp] - gt1[pp]*gt2[pp];
double DENDRO_20 = (DENDRO_19 * DENDRO_19);
double DENDRO_21 = pow(gt1[pp], 2);
double DENDRO_22 = -DENDRO_21 + gt0[pp]*gt3[pp];
double DENDRO_23 = At1[pp]*DENDRO_19;
double DENDRO_24 = 2*DENDRO_6;
double DENDRO_25 = At2[pp]*DENDRO_22;
double DENDRO_26 = At4[pp]*DENDRO_22;
double DENDRO_27 = 2*DENDRO_19;
double DENDRO_28 = At0[pp]*DENDRO_10 + At3[pp]*DENDRO_20 + At5[pp]*(DENDRO_22 * DENDRO_22) - DENDRO_23*DENDRO_24 + DENDRO_24*DENDRO_25 - DENDRO_26*DENDRO_27;
double DENDRO_29 = 0.5*DENDRO_8;
double DENDRO_30 = 0.5*grad_0_gt3[pp] - 1.0*grad_1_gt1[pp];
double DENDRO_31 = 1.0*grad_1_gt4[pp] - 0.5*grad_2_gt3[pp];
double DENDRO_32 = pow(gt2[pp], 2);
double DENDRO_33 = -DENDRO_32 + gt0[pp]*gt5[pp];
double DENDRO_34 = At2[pp]*DENDRO_19;
double DENDRO_35 = At1[pp]*DENDRO_33;
double DENDRO_36 = At4[pp]*DENDRO_33;
double DENDRO_37 = At0[pp]*DENDRO_9 + At3[pp]*(DENDRO_33 * DENDRO_33) + At5[pp]*DENDRO_20 + DENDRO_12*DENDRO_34 - DENDRO_12*DENDRO_35 - DENDRO_27*DENDRO_36;
double DENDRO_38 = grad_0_gt4[pp] + grad_1_gt2[pp] - grad_2_gt1[pp];
double DENDRO_39 = At2[pp]*DENDRO_8;
double DENDRO_40 = At5[pp]*DENDRO_19;
double DENDRO_41 = At0[pp]*DENDRO_8;
double DENDRO_42 = At3[pp]*DENDRO_33;
double DENDRO_43 = DENDRO_19*DENDRO_8;
double DENDRO_44 = -At1[pp]*DENDRO_9 - At4[pp]*DENDRO_43 + DENDRO_2*DENDRO_34 - DENDRO_2*DENDRO_35 + DENDRO_2*DENDRO_41 - DENDRO_36*DENDRO_6 + DENDRO_39*DENDRO_6 + DENDRO_40*DENDRO_6 + DENDRO_42*DENDRO_8;
double DENDRO_45 = grad_0_gt4[pp] - grad_1_gt2[pp] + grad_2_gt1[pp];
double DENDRO_46 = 1.0*At0[pp]*DENDRO_14 + 1.0*At2[pp]*DENDRO_10 + 1.0*At3[pp]*DENDRO_43 + 1.0*At5[pp]*DENDRO_22*DENDRO_6 - 1.0*DENDRO_11*DENDRO_19 - 1.0*DENDRO_13*DENDRO_6 - 1.0*DENDRO_2*DENDRO_23 + 1.0*DENDRO_2*DENDRO_25 - 1.0*DENDRO_26*DENDRO_8;
double DENDRO_47 = -grad_0_gt4[pp] + grad_1_gt2[pp] + grad_2_gt1[pp];
double DENDRO_48 = -At1[pp]*DENDRO_43 - At4[pp]*DENDRO_20 + DENDRO_19*DENDRO_42 - DENDRO_22*DENDRO_36 + DENDRO_22*DENDRO_39 + DENDRO_22*DENDRO_40 + DENDRO_34*DENDRO_6 - DENDRO_35*DENDRO_6 + DENDRO_41*DENDRO_6;
double DENDRO_49 = 2*alpha[pp]/pow(-DENDRO_0*gt0[pp] + DENDRO_1*gt0[pp] + DENDRO_21*gt5[pp] + DENDRO_32*gt3[pp] - 2*DENDRO_5*gt2[pp], 3);
double DENDRO_50 = 0.5*grad_1_gt3[pp];
// Dendro: printing variables
//--
Gt_rhs_s6_0[pp] = DENDRO_49*(-DENDRO_15*(DENDRO_2*DENDRO_3 + DENDRO_4*DENDRO_6 - DENDRO_7*DENDRO_8) - DENDRO_28*(DENDRO_16*DENDRO_6 + DENDRO_17*DENDRO_8 - DENDRO_18*DENDRO_2) + DENDRO_37*(DENDRO_2*DENDRO_30 + DENDRO_29*grad_1_gt3[pp] - DENDRO_31*DENDRO_6) + 1.0*DENDRO_44*(DENDRO_2*grad_1_gt0[pp] + DENDRO_38*DENDRO_6 - DENDRO_8*grad_0_gt3[pp]) - DENDRO_46*(DENDRO_2*grad_2_gt0[pp] - DENDRO_45*DENDRO_8 + DENDRO_6*grad_0_gt5[pp]) + 1.0*DENDRO_48*(DENDRO_2*DENDRO_47 + DENDRO_6*grad_1_gt5[pp] - DENDRO_8*grad_2_gt3[pp]));
//--
Gt_rhs_s6_1[pp] = DENDRO_49*(DENDRO_15*(DENDRO_19*DENDRO_4 + DENDRO_29*grad_0_gt0[pp] - DENDRO_33*DENDRO_7) + DENDRO_28*(DENDRO_16*DENDRO_19 + DENDRO_17*DENDRO_33 - DENDRO_18*DENDRO_8) - DENDRO_37*(-DENDRO_19*DENDRO_31 + DENDRO_30*DENDRO_8 + DENDRO_33*DENDRO_50) - 1.0*DENDRO_44*(DENDRO_19*DENDRO_38 - DENDRO_33*grad_0_gt3[pp] + DENDRO_8*grad_1_gt0[pp]) + DENDRO_46*(DENDRO_19*grad_0_gt5[pp] - DENDRO_33*DENDRO_45 + DENDRO_8*grad_2_gt0[pp]) - 1.0*DENDRO_48*(DENDRO_19*grad_1_gt5[pp] - DENDRO_33*grad_2_gt3[pp] + DENDRO_47*DENDRO_8));
//--
Gt_rhs_s6_2[pp] = DENDRO_49*(-DENDRO_15*(-DENDRO_19*DENDRO_7 + DENDRO_22*DENDRO_4 + DENDRO_3*DENDRO_6) - DENDRO_28*(DENDRO_16*DENDRO_22 + DENDRO_17*DENDRO_19 - DENDRO_18*DENDRO_6) + DENDRO_37*(DENDRO_19*DENDRO_50 - DENDRO_22*DENDRO_31 + DENDRO_30*DENDRO_6) + 1.0*DENDRO_44*(-DENDRO_19*grad_0_gt3[pp] + DENDRO_22*DENDRO_38 + DENDRO_6*grad_1_gt0[pp]) - DENDRO_46*(-DENDRO_19*DENDRO_45 + DENDRO_22*grad_0_gt5[pp] + DENDRO_6*grad_2_gt0[pp]) + 1.0*DENDRO_48*(-DENDRO_19*grad_2_gt3[pp] + DENDRO_22*grad_1_gt5[pp] + DENDRO_47*DENDRO_6));
// Dendro: reduced ops:  310
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

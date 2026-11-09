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
// Dendro: original ops:  3780
// Dendro: printing temp variables
double DENDRO_0 = pow(gt1[pp], 2);
double DENDRO_1 = -DENDRO_0 + gt0[pp]*gt3[pp];
double DENDRO_2 = gt1[pp]*gt4[pp] - gt2[pp]*gt3[pp];
double DENDRO_3 = -DENDRO_2;
double DENDRO_4 = 0.5*grad_2_gt5[pp];
double DENDRO_5 = -0.5*grad_0_gt5[pp] + 1.0*grad_2_gt2[pp];
double DENDRO_6 = gt3[pp]*gt5[pp];
double DENDRO_7 = pow(gt4[pp], 2);
double DENDRO_8 = DENDRO_6 - DENDRO_7;
double DENDRO_9 = -DENDRO_8;
double DENDRO_10 = -0.5*grad_1_gt5[pp] + 1.0*grad_2_gt4[pp];
double DENDRO_11 = gt2[pp]*gt4[pp];
double DENDRO_12 = -DENDRO_11 + gt1[pp]*gt5[pp];
double DENDRO_13 = 1.0/chi[pp];
double DENDRO_14 = DENDRO_12*grad_1_chi[pp] + DENDRO_3*grad_2_chi[pp] + DENDRO_9*grad_0_chi[pp];
double DENDRO_15 = DENDRO_13*DENDRO_14;
double DENDRO_16 = 0.5*gt5[pp];
double DENDRO_17 = pow(gt2[pp], 2);
double DENDRO_18 = DENDRO_0*gt5[pp] - 2*DENDRO_11*gt1[pp] + DENDRO_17*gt3[pp] - DENDRO_6*gt0[pp] + DENDRO_7*gt0[pp];
double DENDRO_19 = 1.0/DENDRO_18;
double DENDRO_20 = DENDRO_19*grad_0_alpha[pp];
double DENDRO_21 = gt0[pp]*gt4[pp] - gt1[pp]*gt2[pp];
double DENDRO_22 = -DENDRO_17 + gt0[pp]*gt5[pp];
double DENDRO_23 = -DENDRO_22;
double DENDRO_24 = DENDRO_12*grad_0_chi[pp] + DENDRO_21*grad_2_chi[pp] + DENDRO_23*grad_1_chi[pp];
double DENDRO_25 = DENDRO_13*DENDRO_24;
double DENDRO_26 = DENDRO_19*grad_1_alpha[pp];
double DENDRO_27 = -DENDRO_1;
double DENDRO_28 = DENDRO_19*DENDRO_27;
double DENDRO_29 = DENDRO_19*DENDRO_3;
double DENDRO_30 = DENDRO_19*DENDRO_21;
double DENDRO_31 = DENDRO_21*grad_1_chi[pp] + DENDRO_27*grad_2_chi[pp] + DENDRO_3*grad_0_chi[pp];
double DENDRO_32 = DENDRO_19*chi[pp];
double DENDRO_33 = 0.5*grad_1_gt3[pp];
double DENDRO_34 = -0.5*grad_0_gt3[pp] + 1.0*grad_1_gt1[pp];
double DENDRO_35 = 1.0*grad_1_gt4[pp] - 0.5*grad_2_gt3[pp];
double DENDRO_36 = 0.5*gt3[pp];
double DENDRO_37 = DENDRO_13*DENDRO_31;
double DENDRO_38 = DENDRO_19*grad_2_alpha[pp];
double DENDRO_39 = DENDRO_19*DENDRO_23;
double DENDRO_40 = DENDRO_12*DENDRO_19;
double DENDRO_41 = 0.5*grad_0_gt0[pp];
double DENDRO_42 = 1.0*grad_0_gt1[pp] - 0.5*grad_1_gt0[pp];
double DENDRO_43 = 1.0*grad_0_gt2[pp] - 0.5*grad_2_gt0[pp];
double DENDRO_44 = 0.5*gt0[pp];
double DENDRO_45 = DENDRO_19*DENDRO_9;
double DENDRO_46 = -grad_0_gt4[pp] + grad_1_gt2[pp] + grad_2_gt1[pp];
double DENDRO_47 = DENDRO_19*gt4[pp];
double DENDRO_48 = 0.5*grad_1_alpha[pp];
double DENDRO_49 = 0.5*grad_2_alpha[pp];
double DENDRO_50 = 2*chi[pp];
double DENDRO_51 = grad_0_gt4[pp] - grad_1_gt2[pp] + grad_2_gt1[pp];
double DENDRO_52 = DENDRO_19*gt2[pp];
double DENDRO_53 = 0.5*grad_0_alpha[pp];
double DENDRO_54 = 2*DENDRO_2;
double DENDRO_55 = grad_0_gt4[pp] + grad_1_gt2[pp] - grad_2_gt1[pp];
double DENDRO_56 = DENDRO_19*gt1[pp];
double DENDRO_57 = (DENDRO_12 * DENDRO_12);
double DENDRO_58 = (DENDRO_2 * DENDRO_2);
double DENDRO_59 = At4[pp]*DENDRO_2;
double DENDRO_60 = 2*DENDRO_12;
double DENDRO_61 = At1[pp]*DENDRO_8;
double DENDRO_62 = At2[pp]*DENDRO_8;
double DENDRO_63 = pow(DENDRO_18, -2);
double DENDRO_64 = 3*DENDRO_63;
double DENDRO_65 = (DENDRO_21 * DENDRO_21);
double DENDRO_66 = DENDRO_12*DENDRO_21;
double DENDRO_67 = 2*DENDRO_21;
double DENDRO_68 = DENDRO_2*DENDRO_21;
double DENDRO_69 = At2[pp]*DENDRO_1;
double DENDRO_70 = At4[pp]*DENDRO_1;
double DENDRO_71 = At0[pp]*DENDRO_8;
double DENDRO_72 = DENDRO_12*DENDRO_2;
double DENDRO_73 = At5[pp]*DENDRO_2;
double DENDRO_74 = 6*DENDRO_63;
double DENDRO_75 = At3[pp]*DENDRO_22;
// Dendro: printing variables
//--
K_rhs[pp] = -DENDRO_1*DENDRO_32*(DENDRO_20*(DENDRO_10*DENDRO_12 + DENDRO_15*DENDRO_16 + DENDRO_3*DENDRO_4 + DENDRO_5*DENDRO_9) + DENDRO_26*(DENDRO_10*DENDRO_23 + DENDRO_12*DENDRO_5 + DENDRO_16*DENDRO_25 + DENDRO_21*DENDRO_4) - grad2_2_2_alpha[pp] + grad_2_alpha[pp]*(DENDRO_10*DENDRO_30 - DENDRO_13*(-DENDRO_16*DENDRO_19*DENDRO_31 + 1.0*grad_2_chi[pp]) + DENDRO_28*DENDRO_4 + DENDRO_29*DENDRO_5)) - DENDRO_22*DENDRO_32*(DENDRO_20*(DENDRO_12*DENDRO_33 + DENDRO_15*DENDRO_36 + DENDRO_3*DENDRO_35 + DENDRO_34*DENDRO_9) + DENDRO_38*(DENDRO_21*DENDRO_33 + DENDRO_27*DENDRO_35 + DENDRO_3*DENDRO_34 + DENDRO_36*DENDRO_37) - grad2_1_1_alpha[pp] + grad_1_alpha[pp]*(-DENDRO_13*(-DENDRO_19*DENDRO_24*DENDRO_36 + 1.0*grad_1_chi[pp]) + DENDRO_30*DENDRO_35 + DENDRO_33*DENDRO_39 + DENDRO_34*DENDRO_40)) + DENDRO_30*DENDRO_50*(0.5*DENDRO_20*(DENDRO_12*grad_2_gt3[pp] + DENDRO_15*gt4[pp] + DENDRO_3*grad_1_gt5[pp] + DENDRO_46*DENDRO_9) + DENDRO_48*(-DENDRO_13*(-DENDRO_24*DENDRO_47 + grad_2_chi[pp]) + DENDRO_30*grad_1_gt5[pp] + DENDRO_39*grad_2_gt3[pp] + DENDRO_40*DENDRO_46) + DENDRO_49*(-DENDRO_13*(-DENDRO_31*DENDRO_47 + grad_1_chi[pp]) + DENDRO_28*grad_1_gt5[pp] + DENDRO_29*DENDRO_46 + DENDRO_30*grad_2_gt3[pp]) - grad2_1_2_alpha[pp]) - DENDRO_32*DENDRO_54*(0.5*DENDRO_26*(DENDRO_12*grad_2_gt0[pp] + DENDRO_21*grad_0_gt5[pp] + DENDRO_23*DENDRO_51 + DENDRO_25*gt2[pp]) + DENDRO_49*(-DENDRO_13*(-DENDRO_31*DENDRO_52 + grad_0_chi[pp]) + DENDRO_28*grad_0_gt5[pp] + DENDRO_29*grad_2_gt0[pp] + DENDRO_30*DENDRO_51) + DENDRO_53*(-DENDRO_13*(-DENDRO_14*DENDRO_52 + grad_2_chi[pp]) + DENDRO_29*grad_0_gt5[pp] + DENDRO_40*DENDRO_51 + DENDRO_45*grad_2_gt0[pp]) - grad2_0_2_alpha[pp]) - DENDRO_32*DENDRO_8*(DENDRO_26*(DENDRO_12*DENDRO_41 + DENDRO_21*DENDRO_43 + DENDRO_23*DENDRO_42 + DENDRO_25*DENDRO_44) + DENDRO_38*(DENDRO_21*DENDRO_42 + DENDRO_27*DENDRO_43 + DENDRO_3*DENDRO_41 + DENDRO_37*DENDRO_44) - grad2_0_0_alpha[pp] + grad_0_alpha[pp]*(-DENDRO_13*(-DENDRO_14*DENDRO_19*DENDRO_44 + 1.0*grad_0_chi[pp]) + DENDRO_29*DENDRO_43 + DENDRO_40*DENDRO_42 + DENDRO_41*DENDRO_45)) + DENDRO_40*DENDRO_50*(0.5*DENDRO_38*(DENDRO_21*grad_0_gt3[pp] + DENDRO_27*DENDRO_55 + DENDRO_3*grad_1_gt0[pp] + DENDRO_37*gt1[pp]) + DENDRO_48*(-DENDRO_13*(-DENDRO_24*DENDRO_56 + grad_0_chi[pp]) + DENDRO_30*DENDRO_55 + DENDRO_39*grad_0_gt3[pp] + DENDRO_40*grad_1_gt0[pp]) + DENDRO_53*(-DENDRO_13*(-DENDRO_14*DENDRO_56 + grad_1_chi[pp]) + DENDRO_29*DENDRO_55 + DENDRO_40*grad_0_gt3[pp] + DENDRO_45*grad_1_gt0[pp]) - grad2_0_1_alpha[pp]) + (1.0/3.0)*alpha[pp]*(At0[pp]*DENDRO_64*(At0[pp]*(DENDRO_8 * DENDRO_8) + At3[pp]*DENDRO_57 + At5[pp]*DENDRO_58 + DENDRO_54*DENDRO_62 - DENDRO_59*DENDRO_60 - DENDRO_60*DENDRO_61) + At1[pp]*DENDRO_74*(At1[pp]*DENDRO_22*DENDRO_8 + At1[pp]*DENDRO_57 - At2[pp]*DENDRO_72 + At4[pp]*DENDRO_12*DENDRO_21 + At4[pp]*DENDRO_2*DENDRO_22 - DENDRO_12*DENDRO_71 - DENDRO_12*DENDRO_75 - DENDRO_21*DENDRO_62 - DENDRO_21*DENDRO_73) + At2[pp]*DENDRO_74*(-At1[pp]*DENDRO_72 + At2[pp]*DENDRO_58 + At3[pp]*DENDRO_66 + DENDRO_1*DENDRO_62 + DENDRO_1*DENDRO_73 - DENDRO_12*DENDRO_70 + DENDRO_2*DENDRO_71 - DENDRO_21*DENDRO_59 - DENDRO_21*DENDRO_61) + At3[pp]*DENDRO_64*(At0[pp]*DENDRO_57 - At1[pp]*DENDRO_22*DENDRO_60 + 2*At2[pp]*DENDRO_66 + At3[pp]*(DENDRO_22 * DENDRO_22) - At4[pp]*DENDRO_22*DENDRO_67 + At5[pp]*DENDRO_65) + At4[pp]*DENDRO_74*(-At0[pp]*DENDRO_72 + At1[pp]*DENDRO_12*DENDRO_21 + At1[pp]*DENDRO_2*DENDRO_22 - At2[pp]*DENDRO_68 + At4[pp]*DENDRO_1*DENDRO_22 + At4[pp]*DENDRO_65 - At5[pp]*DENDRO_1*DENDRO_21 - DENDRO_12*DENDRO_69 - DENDRO_21*DENDRO_75) + At5[pp]*DENDRO_64*(At0[pp]*DENDRO_58 - 2*At1[pp]*DENDRO_68 + At3[pp]*DENDRO_65 + At5[pp]*(DENDRO_1 * DENDRO_1) + DENDRO_54*DENDRO_69 - DENDRO_67*DENDRO_70) + pow(K[pp], 2)) + beta0[pp]*grad_0_K[pp] + beta1[pp]*grad_1_K[pp] + beta2[pp]*grad_2_K[pp];
// Dendro: reduced ops:  452
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

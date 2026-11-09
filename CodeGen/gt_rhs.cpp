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
// Dendro: original ops:  156
// Dendro: printing temp variables
double DENDRO_0 = 2*alpha[pp];
double DENDRO_1 = 2*gt1[pp];
double DENDRO_2 = 2*gt2[pp];
double DENDRO_3 = (2.0/3.0)*gt0[pp];
double DENDRO_4 = (1.0/3.0)*gt1[pp];
double DENDRO_5 = (2.0/3.0)*grad_2_beta2[pp];
double DENDRO_6 = (1.0/3.0)*gt2[pp];
double DENDRO_7 = (2.0/3.0)*grad_1_beta1[pp];
double DENDRO_8 = (2.0/3.0)*grad_0_beta0[pp];
double DENDRO_9 = 2*gt4[pp];
double DENDRO_10 = (1.0/3.0)*gt4[pp];
// Dendro: printing variables
//--
gt_rhs00[pp] = -At0[pp]*DENDRO_0 + DENDRO_1*grad_0_beta1[pp] + DENDRO_2*grad_0_beta2[pp] - DENDRO_3*grad_1_beta1[pp] - DENDRO_3*grad_2_beta2[pp] + beta0[pp]*grad_0_gt0[pp] + beta1[pp]*grad_1_gt0[pp] + beta2[pp]*grad_2_gt0[pp] + (4.0/3.0)*grad_0_beta0[pp]*gt0[pp];
//--
gt_rhs01[pp] = -At1[pp]*DENDRO_0 + DENDRO_4*grad_0_beta0[pp] + DENDRO_4*grad_1_beta1[pp] - DENDRO_5*gt1[pp] + beta0[pp]*grad_0_gt1[pp] + beta1[pp]*grad_1_gt1[pp] + beta2[pp]*grad_2_gt1[pp] + grad_0_beta1[pp]*gt3[pp] + grad_0_beta2[pp]*gt4[pp] + grad_1_beta0[pp]*gt0[pp] + grad_1_beta2[pp]*gt2[pp];
//--
gt_rhs02[pp] = -At2[pp]*DENDRO_0 + DENDRO_6*grad_0_beta0[pp] + DENDRO_6*grad_2_beta2[pp] - DENDRO_7*gt2[pp] + beta0[pp]*grad_0_gt2[pp] + beta1[pp]*grad_1_gt2[pp] + beta2[pp]*grad_2_gt2[pp] + grad_0_beta1[pp]*gt4[pp] + grad_0_beta2[pp]*gt5[pp] + grad_2_beta0[pp]*gt0[pp] + grad_2_beta1[pp]*gt1[pp];
//--
gt_rhs11[pp] = -At3[pp]*DENDRO_0 + DENDRO_1*grad_1_beta0[pp] - DENDRO_5*gt3[pp] - DENDRO_8*gt3[pp] + DENDRO_9*grad_1_beta2[pp] + beta0[pp]*grad_0_gt3[pp] + beta1[pp]*grad_1_gt3[pp] + beta2[pp]*grad_2_gt3[pp] + (4.0/3.0)*grad_1_beta1[pp]*gt3[pp];
//--
gt_rhs12[pp] = -At4[pp]*DENDRO_0 + DENDRO_10*grad_1_beta1[pp] + DENDRO_10*grad_2_beta2[pp] - DENDRO_8*gt4[pp] + beta0[pp]*grad_0_gt4[pp] + beta1[pp]*grad_1_gt4[pp] + beta2[pp]*grad_2_gt4[pp] + grad_1_beta0[pp]*gt2[pp] + grad_1_beta2[pp]*gt5[pp] + grad_2_beta0[pp]*gt1[pp] + grad_2_beta1[pp]*gt3[pp];
//--
gt_rhs22[pp] = -At5[pp]*DENDRO_0 + DENDRO_2*grad_2_beta0[pp] - DENDRO_7*gt5[pp] - DENDRO_8*gt5[pp] + DENDRO_9*grad_2_beta1[pp] + beta0[pp]*grad_0_gt5[pp] + beta1[pp]*grad_1_gt5[pp] + beta2[pp]*grad_2_gt5[pp] + (4.0/3.0)*grad_2_beta2[pp]*gt5[pp];
// Dendro: reduced ops:  135
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

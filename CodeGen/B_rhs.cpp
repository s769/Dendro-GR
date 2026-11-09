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
// Dendro: original ops:  72
// Dendro: printing temp variables
// Dendro: printing variables
//--
B_rhs0[pp] = -B0[pp]*eta + Gt_rhs_s1_0[pp] - Gt_rhs_s2_0[pp] + (2.0/3.0)*Gt_rhs_s3_0[pp] + Gt_rhs_s4_0[pp] - Gt_rhs_s5_0[pp] + Gt_rhs_s6_0[pp] - Gt_rhs_s7_0[pp] + lambda[2]*(beta0[pp]*grad_0_B0[pp] + beta1[pp]*grad_1_B0[pp] + beta2[pp]*grad_2_B0[pp]) - lambda[3]*(beta0[pp]*grad_0_Gt0[pp] + beta1[pp]*grad_1_Gt0[pp] + beta2[pp]*grad_2_Gt0[pp]);
//--
B_rhs1[pp] = -B1[pp]*eta + Gt_rhs_s1_1[pp] - Gt_rhs_s2_1[pp] + (2.0/3.0)*Gt_rhs_s3_1[pp] + Gt_rhs_s4_1[pp] - Gt_rhs_s5_1[pp] + Gt_rhs_s6_1[pp] - Gt_rhs_s7_1[pp] + lambda[2]*(beta0[pp]*grad_0_B1[pp] + beta1[pp]*grad_1_B1[pp] + beta2[pp]*grad_2_B1[pp]) - lambda[3]*(beta0[pp]*grad_0_Gt1[pp] + beta1[pp]*grad_1_Gt1[pp] + beta2[pp]*grad_2_Gt1[pp]);
//--
B_rhs2[pp] = -B2[pp]*eta + Gt_rhs_s1_2[pp] - Gt_rhs_s2_2[pp] + (2.0/3.0)*Gt_rhs_s3_2[pp] + Gt_rhs_s4_2[pp] - Gt_rhs_s5_2[pp] + Gt_rhs_s6_2[pp] - Gt_rhs_s7_2[pp] + lambda[2]*(beta0[pp]*grad_0_B2[pp] + beta1[pp]*grad_1_B2[pp] + beta2[pp]*grad_2_B2[pp]) - lambda[3]*(beta0[pp]*grad_0_Gt2[pp] + beta1[pp]*grad_1_Gt2[pp] + beta2[pp]*grad_2_Gt2[pp]);
// Dendro: reduced ops:  72
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

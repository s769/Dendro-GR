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
// Dendro: original ops:  15
// Dendro: printing temp variables
// Dendro: printing variables
//--
Gt_rhs_s1_0[pp] = beta0[pp]*grad_0_Gt0[pp] + beta1[pp]*grad_1_Gt0[pp] + beta2[pp]*grad_2_Gt0[pp];
//--
Gt_rhs_s1_1[pp] = beta0[pp]*grad_0_Gt1[pp] + beta1[pp]*grad_1_Gt1[pp] + beta2[pp]*grad_2_Gt1[pp];
//--
Gt_rhs_s1_2[pp] = beta0[pp]*grad_0_Gt2[pp] + beta1[pp]*grad_1_Gt2[pp] + beta2[pp]*grad_2_Gt2[pp];
// Dendro: reduced ops:  15
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

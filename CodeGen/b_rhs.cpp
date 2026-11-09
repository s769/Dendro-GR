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
// Dendro: original ops:  42
// Dendro: printing temp variables
double DENDRO_0 = (3.0/4.0)*alpha[pp]*lambda_f[1] + (3.0/4.0)*lambda_f[0];
// Dendro: printing variables
//--
b_rhs0[pp] = B0[pp]*DENDRO_0 + lambda[1]*(beta0[pp]*grad_0_beta0[pp] + beta1[pp]*grad_1_beta0[pp] + beta2[pp]*grad_2_beta0[pp]);
//--
b_rhs1[pp] = B1[pp]*DENDRO_0 + lambda[1]*(beta0[pp]*grad_0_beta1[pp] + beta1[pp]*grad_1_beta1[pp] + beta2[pp]*grad_2_beta1[pp]);
//--
b_rhs2[pp] = B2[pp]*DENDRO_0 + lambda[1]*(beta0[pp]*grad_0_beta2[pp] + beta1[pp]*grad_1_beta2[pp] + beta2[pp]*grad_2_beta2[pp]);
// Dendro: reduced ops:  30
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

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
// Dendro: original ops:  9
// Dendro: printing temp variables
// Dendro: printing variables
//--
a_rhs[pp] = -2*K[pp]*alpha[pp] + lambda[0]*(beta0[pp]*grad_0_alpha[pp] + beta1[pp]*grad_1_alpha[pp] + beta2[pp]*grad_2_alpha[pp]);
// Dendro: reduced ops:  9
// Dendro: }}} 
  }
 }
}
     bssn::timer::t_rhs.stop();

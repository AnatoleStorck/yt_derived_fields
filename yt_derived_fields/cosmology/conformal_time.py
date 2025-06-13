# Generates RAMSES cosmology tables to compute consistent time conversions from outputs
# (converted from conformal_time.f90)
# Author: Anatole Storck

import numpy as np
from pathlib import Path

class ConformalTime:
    
    def __init__(self, n_frw=1000):
        self.n_frw = n_frw
        self.aexp_frw = None
        self.hexp_frw = None
        self.tau_frw = None
        self.t_frw = None
        self.t_frw_yr = None
        self.time_tot = None

    def ct_conftime2time(self, tau):
        # Return look-back time in yr
        i = 1
        while self.tau_frw[i] > tau and i < self.n_frw:
            i += 1
        # Linear interpolation
        t = (self.t_frw_yr[i] * (tau - self.tau_frw[i-1]) / (self.tau_frw[i] - self.tau_frw[i-1]) +
             self.t_frw_yr[i-1] * (tau - self.tau_frw[i]) / (self.tau_frw[i-1] - self.tau_frw[i]))
        return t

    def ct_proptime2time(self, tau, h0):
        # Return look-back time in yr
        return tau / (h0 / 3.08e19) / (365.25 * 24. * 3600.)

    def ct_aexp2time(self, aexp):
        # Return look-back time in yr
        i = 1
        while self.aexp_frw[i] > aexp and i < self.n_frw:
            i += 1
        t = (self.t_frw_yr[i] * (aexp - self.aexp_frw[i-1]) / (self.aexp_frw[i] - self.aexp_frw[i-1]) +
             self.t_frw_yr[i-1] * (aexp - self.aexp_frw[i]) / (self.aexp_frw[i-1] - self.aexp_frw[i]))
        return t
    
    def ct_redshift2time(self, z):
        # Return time from Big Bang in yr
        lookback_time = self.ct_aexp2time(1 / (1 + z))
        time = lookback_time - self.ct_aexp2time(0)
        return time

    def ct_init_cosmo(self, omega_m, omega_l, omega_k, h0):
        # h0 is in km/s/Mpc
        if self.aexp_frw is None:
            try:
                (self.t_frw,
                 self.t_frw_yr,
                 self.tau_frw,
                 self.aexp_frw,
                 self.n_frw) = [np.load(f"{Path(__file__).parent}/conformal_time_var.npz")[field] 
                                for field in ["t_frw", "t_frw_yr", "tau_frw", "aexp_frw", "n_frw"]]
                #print("Conformal time tables loaded successfully.")

            except:
                #print("Generating conformal time tables...")
                self.aexp_frw = np.zeros(self.n_frw + 1)
                self.hexp_frw = np.zeros(self.n_frw + 1)
                self.tau_frw = np.zeros(self.n_frw + 1)
                self.t_frw = np.zeros(self.n_frw + 1)
                self.t_frw_yr = np.zeros(self.n_frw + 1)
                
                self.ct_friedman(omega_m, omega_l, omega_k, 1e-6, 1e-3)
                # Convert time to yr
                self.t_frw_yr = self.t_frw / (h0 / 3.08e19) / (365.25 * 24. * 3600.)

    def ct_clear_cosmo(self):
        self.aexp_frw = None
        self.hexp_frw = None
        self.tau_frw = None
        self.t_frw = None
        self.t_frw_yr = None

    def ct_friedman(self, O_mat_0, O_vac_0, O_k_0, alpha, axp_min):
        ntable = self.n_frw
        axp_out = np.zeros(ntable + 1)
        hexp_out = np.zeros(ntable + 1)
        tau_out = np.zeros(ntable + 1)
        t_out = np.zeros(ntable + 1)

        axp_tau = 1.0
        axp_t = 1.0
        tau = 0.0
        t = 0.0
        nstep = 0

        # First pass to determine nstep
        while (axp_tau >= axp_min) or (axp_t >= axp_min):
            nstep += 1
            dtau = alpha * axp_tau / self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0)
            axp_tau_pre = axp_tau - self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) * dtau / 2.0
            axp_tau = axp_tau - self.dadtau(axp_tau_pre, O_mat_0, O_vac_0, O_k_0) * dtau
            tau = tau - dtau

            dt = alpha * axp_t / self.dadt(axp_t, O_mat_0, O_vac_0, O_k_0)
            axp_t_pre = axp_t - self.dadt(axp_t, O_mat_0, O_vac_0, O_k_0) * dt / 2.0
            axp_t = axp_t - self.dadt(axp_t_pre, O_mat_0, O_vac_0, O_k_0) * dt
            t = t - dt

        age_tot = -t
        self.time_tot = age_tot

        nskip = max(nstep // ntable, 1)

        axp_t = 1.0
        t = 0.0
        axp_tau = 1.0
        tau = 0.0
        nstep = 0
        nout = 0
        t_out[nout] = t
        tau_out[nout] = tau
        axp_out[nout] = axp_tau
        hexp_out[nout] = self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) / axp_tau

        while (axp_tau >= axp_min) or (axp_t >= axp_min):
            nstep += 1
            dtau = alpha * axp_tau / self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0)
            axp_tau_pre = axp_tau - self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) * dtau / 2.0
            axp_tau = axp_tau - self.dadtau(axp_tau_pre, O_mat_0, O_vac_0, O_k_0) * dtau
            tau = tau - dtau

            dt = alpha * axp_t / self.dadt(axp_t, O_mat_0, O_vac_0, O_k_0)
            axp_t_pre = axp_t - self.dadt(axp_t, O_mat_0, O_vac_0, O_k_0) * dt / 2.0
            axp_t = axp_t - self.dadt(axp_t_pre, O_mat_0, O_vac_0, O_k_0) * dt
            t = t - dt

            if nstep % nskip == 0 and nout < ntable:
                nout += 1
                t_out[nout] = t
                tau_out[nout] = tau
                axp_out[nout] = axp_tau
                hexp_out[nout] = self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) / axp_tau

        t_out[ntable] = t
        tau_out[ntable] = tau
        axp_out[ntable] = axp_tau
        hexp_out[ntable] = self.dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0) / axp_tau

        self.aexp_frw = axp_out
        self.hexp_frw = hexp_out
        self.tau_frw = tau_out
        self.t_frw = t_out

    @staticmethod
    def dadtau(axp_tau, O_mat_0, O_vac_0, O_k_0):
        val = (axp_tau ** 3) * (O_mat_0 + O_vac_0 * axp_tau ** 3 + O_k_0 * axp_tau)
        return np.sqrt(val)

    @staticmethod
    def dadt(axp_t, O_mat_0, O_vac_0, O_k_0):
        val = (1.0 / axp_t) * (O_mat_0 + O_vac_0 * axp_t ** 3 + O_k_0 * axp_t)
        return np.sqrt(val)
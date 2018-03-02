import unittest
from .dll import *
import numpy as np
from .. import ckm
from math import radians
from flavio.physics.eft import WilsonCoefficients
from flavio.physics.ddecays.wilsoncoefficients import wctot_dict
from flavio.classes import Parameter, Observable
from flavio.parameters import default_parameters
import copy
import flavio
import math

s = 1.519267515435317e+24

c = copy.deepcopy(default_parameters)
# parameters taken from PDG and table I of 1311.0903
c.set_constraint('alpha_s', '0.1184(7)')
c.set_constraint('f_Bs', '0.2277(45)')
c.set_constraint('f_B0', '0.1905(42)')
c.set_constraint('Vcb', 4.24e-2)
c.set_constraint('Vub', 3.82e-3)
c.set_constraint('gamma', radians(73.))
c.set_constraint('DeltaGamma/Gamma_Bs', 0.1226)

par = c.get_central_all()


wc_obj = WilsonCoefficients()
wc = wctot_dict(wc_obj, 'bsmumu', 4.8, par)
wc_e = wctot_dict(wc_obj, 'bsee', 4.8, par)
wc_tau = wctot_dict(wc_obj, 'bstautau', 4.8, par)


class TestBll(unittest.TestCase):
    def test_bsll(self):
        # just some trivial tests to see if calling the functions raises an error
        self.assertEqual(len(amplitudes(par, wc, 'Bs', 'mu', 'mu')), 2)
        # BR should be around 3.5e-9
        self.assertAlmostEqual(br_inst(par, wc, 'Bs', 'mu', 'mu')*1e9, 3.5, places=0)
        # comparison to 1311.0903
        self.assertAlmostEqual(abs(ckm.xi('t','bs')(par))/par['Vcb'], 0.980, places=3)

    def test_bsll_classes(self):
        par_default = default_parameters.get_central_all()
        self.assertAlmostEqual(br_timeint(par_default, wc_tau, 'Bs', 'tau', 'tau')/Observable['BR(Bs->tautau)'].prediction_central(default_parameters, wc_obj), 1, places=4)
        self.assertAlmostEqual(br_timeint(par_default, wc_e, 'Bs', 'e', 'e')/Observable['BR(Bs->ee)'].prediction_central(default_parameters, wc_obj), 1, places=4)
        self.assertAlmostEqual(br_timeint(par_default, wc, 'Bs', 'mu', 'mu')/Observable['BR(Bs->mumu)'].prediction_central(default_parameters, wc_obj), 1, places=4)

    def test_bsll_lfv(self):
        # test for errors
        self.assertEqual(flavio.sm_prediction('BR(B0->emu)'), 0)
        self.assertEqual(flavio.sm_prediction('BR(B0->emu,mue)'), 0)
        wc = flavio.WilsonCoefficients()
        wc.set_initial({'C10_bdemu': 1, 'C10_bdmue': 2}, scale=4.8)
        self.assertEqual(flavio.np_prediction('BR(B0->mue)', wc)
                        /flavio.np_prediction('BR(B0->emu)', wc), 4)
        self.assertEqual(flavio.np_prediction('BR(B0->emu,mue)', wc)
                        /flavio.np_prediction('BR(B0->emu)', wc), 5)

    def test_BR_Bs_to_mumu(self):
        # cross check formula with 2nd implementation

        # use formula (1.2) and (1.3) of 1407.2771
        def amplitudes_Amsterdam_Bs_mumu(par, wc):
            # masses
            scale = flavio.config['renormalization scale']['dll']
            mmu = par['m_mu']
            mB  = par['m_Bs']
            mb  = running.get_mb(par, scale, nf_out=5)
            ms  = running.get_ms(par, scale, nf_out=5)
            # Wilson coefficients
            C10SM = -4.188642825319258 #SM value for C10 -4.134#
            C10m  = (C10SM+wc['C10_bsmumu']) - wc['C10p_bsmumu']
            CPm   = wc['CP_bsmumu']          - wc['CPp_bsmumu']
            CSm   = wc['CS_bsmumu']          - wc['CSp_bsmumu']

            P = C10m/C10SM + mB**2/(2.*mmu) * (mb / (mb + ms)) * CPm/C10SM
            S = math.sqrt(1. - 4*mmu**2/mB**2) * mB**2/(2.*mmu) * (mb / (mb + ms)) * CSm/C10SM
            return P, S

        def BR_inst_Amsterdam_Bs_mumu(par, wc):
            # eq.(1.2) from 1407.2771
            scale = flavio.config['renormalization scale']['dll']
            GF       = par['GF']
            alphaem  = running.get_alpha(par, scale)['alpha_e']
            mW       = par['m_W']
            mB       = par['m_Bs']
            mmu      = par['m_mu']
            tauB     = par['tau_Bs']
            fB       = par['f_Bs']
            xi_ts_tb = ckm.xi('t','bs')(par)
            s2w      = par['s2w']

            # Wilson coefficients
            C10SM = -4.188642825319258 #SM value for C10
            P,S      = amplitudes_Amsterdam_Bs_mumu(par, wc)

            return (GF**2 * alphaem**2 * mB)/(16. * math.pi**3) * math.sqrt(1.-4.*mmu**2/mB**2) *  C10SM**2 * abs(xi_ts_tb)**2 * tauB * fB**2 * mmu**2 * (abs(P)**2 + abs(S)**2)

        # define function that calculates the BR in both implementations
        def BR(c10, c10p, cS, cSp, cP, cPp):
            list_wc = {'C10_bsmumu' :  c10,
                 'C10p_bsmumu': c10p,
                 'CS_bsmumu'  : cS,
                 'CSp_bsmumu' : cSp,
                 'CP_bsmumu'  : cP,
                 'CPp_bsmumu' : cPp,
                 'C9_bsmumu'  : 0.,
                 'C9p_bsmumu' : 0.}
            wc = flavio.WilsonCoefficients()
            wc.set_initial(list_wc, scale=160 )

            BR_flavio    = flavio.np_prediction('BR(Bs->mumu)', wc)
            BR_Amsterdam = BR_inst_Amsterdam_Bs_mumu(par, list_wc)

            return {'flavio': BR_flavio, 'Amsterdam': BR_Amsterdam}

        # test SM value
        br_SM = BR(0,0,0,0,0,0)
        self.assertAlmostEqual(br_SM['flavio'] / br_SM['Amsterdam'], 1., places=2)

        # test some values for WC's
        br = BR(-.4,0,0,0,0,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., delta=0.01)

        br = BR(-.5*1j,0,0,0,0,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., places=2)

        br = BR(0,0,0.01,0,0,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., places=2)

        br = BR(0,0,0,-0.01*1j,0,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., delta=2)

        br = BR(0,0,0,0,0.01,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., delta=0.03)

        br = BR(0,0,0,0,0,-0.02*1j)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., delta=0.03)

        br = BR(-.5*1j,0,0.01,0,0,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., places=2)

        br = BR(-.5*1j,0,0,0,-0.03,0)
        self.assertAlmostEqual(br['flavio'] / br['Amsterdam'], 1., delta=0.03)

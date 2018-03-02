r"""Standard Model Wilson coefficients for $\Delta C=1$ transitions as well
as tree-level $D$ decays"""


from math import sqrt, log, pi
import numpy as np
import scipy.interpolate
from flavio.physics.running import running
from flavio.physics.common import add_dict
import flavio
import pkg_resources


# SM Wilson coefficients for n_f=5 in the basis
# [ C_1, C_2, C_3, C_4, C_5, C_6,
# C_7^eff, C_8^eff,
# C_9, C_10,
# C_3^Q, C_4^Q, C_5^Q, C_6^Q,
# Cb ]
# where all operators are defined as in hep-ph/0512066 *except*
# C_9,10, which are defined with an additional alpha/4pi prefactor.
scales = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5)
# data = np.array([C_low(s, 120, get_par(), nf=5) for s in scales]).T
data = np.load(pkg_resources.resource_filename('flavio.physics', 'data/wcsm/wc_sm_dB1_2_55.npy'))
wcsm_nf5 = scipy.interpolate.interp1d(scales, data)


# names of SM DeltaF=1 Wilson coefficients needed for wctot_dict
fcnclabels = {}
_fcnc = ['bs', 'bd', 'sd', ]
_ll = ['ee', 'mumu', 'tautau']
for qq in _fcnc:
    for ll in _ll:
        fcnclabels[qq + ll] = ['C1_'+qq, 'C2_'+qq, # current-current
                               'C3_'+qq, 'C4_'+qq, 'C5_'+qq, 'C6_'+qq, # QCD penguins
                               'C7_'+qq, 'C8_'+qq, # dipoles
                               'C9_'+qq+ll, 'C10_'+qq+ll, # semi-leptonic
                               'C3Q_'+qq, 'C4Q_'+qq, 'C5Q_'+qq, 'C6Q_'+qq, 'Cb_'+qq, # EW penguins
                                # and everything with flipped chirality ...
                               'C1p_'+qq, 'C2p_'+qq,
                               'C3p_'+qq, 'C4p_'+qq, 'C5p_'+qq, 'C6p_'+qq,
                               'C7p_'+qq, 'C8p_'+qq,
                               'C9p_'+qq+ll, 'C10p_'+qq+ll,
                               'C3Qp_'+qq, 'C4Qp_'+qq, 'C5Qp_'+qq, 'C6Qp_'+qq, 'Cbp_'+qq,
                                # scalar and pseudoscalar
                               'CS_'+qq+ll, 'CP_'+qq+ll,
                               'CSp_'+qq+ll, 'CPp_'+qq+ll, ]


def wctot_dict(wc_obj, sector, scale, par, nf_out=5):
    r"""Get a dictionary with the total (SM + new physics) values  of the
    $\Delta F=1$ Wilson coefficients at a given scale, given a
    WilsonCoefficients instance."""
    wc_np_dict = wc_obj.get_wc(sector, scale, par, nf_out=nf_out)
    if nf_out == 5:
        wc_sm = wcsm_nf5(scale)
    else:
        raise NotImplementedError("DeltaF=1 Wilson coefficients only implemented for B physics")
    # fold in approximate m_t-dependence of C_10 (see eq. 4 of arXiv:1311.0903)
    wc_sm[9] = wc_sm[9] * (par['m_t']/173.1)**1.53
    wc_labels = fcnclabels[sector]
    wc_sm_dict = dict(zip(wc_labels, wc_sm))
    tot_dict = add_dict((wc_np_dict, wc_sm_dict))
    return tot_dict


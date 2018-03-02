r"""Functions for the branching ratios and effective lifetimes of the leptonic
decays $D0 (c ubar) \to \ell^+\ell^-$ where $\ell=e$, $\mu$ or
$\tau$. This code is an adapted version of leptonic B meson decays B_q->ll"""

from math import pi,sqrt
from flavio.physics import ckm
from flavio.physics.running import running
from flavio.physics.bdecays.common import lambda_K # Källen lambda function
from flavio.classes import Observable, Prediction
from flavio.config import config
from flavio.physics.bdecays.wilsoncoefficients import wctot_dict #### NEEDS WORK


# def br_lifetime_corr(y, ADeltaGamma):
    # r"""Correction factor relating the experimentally measured branching ratio
    # (time-integrated) to the theoretical one (instantaneous), see e.g. eq. (8)
    # of arXiv:1204.1735.

    # Parameters
    # ----------

    # - `y`: relative decay rate difference, $y_q = \tau_{D_q} \Delta\Gamma_q /2$
    # - `ADeltaGamma`: $A_{\Delta\Gamma_q}$ as defined, e.g., in arXiv:1204.1735

    # Returns
    # -------

    # $\frac{1-y_q^2}{1+A_{\Delta\Gamma_q} y_q}$
    # """
#     return (1 - y**2)/(1 + ADeltaGamma*y)

def amplitudes(par, wc, l1, l2):
    r"""Amplitudes P and S entering the $D_0\to\ell_1^+\ell_2^-$ observables.

    Parameters
    ----------

    - `par`: parameter dictionary
    - `l1` and `l2`: should be `'e'`, `'mu'`, or `'tau'`

    Returns
    -------

    `(P, S)` where for the special case `l1 == l2` one has

    - $P = \frac{2m_\ell}{m_{D}} (C_{10}-C_{10}') + m_{D} (C_P-C_P')$
    - $S = m_{D} (C_S-C_S')$
    """
    scale = config['renormalization scale']['dll']
    # masses
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_D0']
    mc = running.get_mc(par, scale, nf_out=4)
    #   get the mass of the spectator quark
    mspec = running.get_mu(par, scale, nf_out=4)
    # Wilson coefficients
    qqll = 'cu' + l1 + l2
    # For LFV expressions see arXiv:1602.00881 eq. (5)
    C9m = wc['C9_'+qqll] - wc['C9p_'+qqll] # only relevant for l1 != l2!
    C10m = wc['C10_'+qqll] - wc['C10p_'+qqll]
    CPm = wc['CP_'+qqll] - wc['CPp_'+qqll]
    CSm = wc['CS_'+qqll] - wc['CSp_'+qqll]
    P = (ml2 + ml1)/mB * C10m + mB * mb/(mb + mspec) * CPm
    S = (ml2 - ml1)/mB * C9m  + mB * mb/(mb + mspec) * CSm
    return P, S

# def ADeltaGamma(par, wc, lep):
    # P, S = amplitudes(par, wc,  lep, lep)
    # # cf. eq. (17) of arXiv:1204.1737
    # return ((P**2).real - (S**2).real)/(abs(P)**2 + abs(S)**2)

def br_inst(par, wc,  l1, l2):
    r"""Branching ratio of $D_0\to\ell_1^+\ell_2^-$ in the absence of mixing.

    Parameters
    ----------

    - `par`: parameter dictionary
    - `lep`: should be `'e'`, `'mu'`, or `'tau'`
    """
    # paramaeters
    GF = par['GF']
    #### ADJUST RUNNING
    alphaem = running.get_alpha(par, 2)['alpha_e']
    ml1 = par['m_'+l1]
    ml2 = par['m_'+l2]
    mB = par['m_D0']
    tauB = par['tau_D0']
    fB = par['f_D0']
    # appropriate CKM elements
    N = ckm.xi('s','cu')(par) * 4*GF/sqrt(2) * alphaem/(4*pi)   #### CHECK THIS
    beta = sqrt(lambda_K(mB**2,ml1**2,ml2**2))/mB**2
    beta_p = sqrt(1 - (ml1 + ml2)**2/mB**2)
    beta_m = sqrt(1 - (ml1 - ml2)**2/mB**2)
    prefactor = abs(N)**2 / 32. / pi * mB**3 * tauB * beta * fB**2
    P, S = amplitudes(par, wc, l1, l2)
    return prefactor * ( beta_m**2 * abs(P)**2 + beta_p**2 * abs(S)**2 )

# def br_timeint(par, wc, l1, l2):
    # r"""Time-integrated branching ratio of $D_0\to\ell^+\ell^-$."""
    # if l1 != l2:
        # raise ValueError("Time-integrated branching ratio only defined for equal lepton flavours")
    # lep = l1
    # br0 = br_inst(par, wc,  lep, lep)
    # y = par['DeltaGamma/Gamma_D0']/2.
    # ADG = ADeltaGamma(par, wc, lep)
    # corr = br_lifetime_corr(y, ADG)
#     return br0 / corr

def bqll_obs(function, wc_obj, par,  l1, l2):
    scale = config['renormalization scale']['dll']
    label = 'cu'+l1+l2
    if l1 == l2:
        # include SM contributions for LF conserving decay
        wc = wctot_dict(wc_obj, label, scale, par)
    else:
        wc = wc_obj.get_wc(label, scale, par)
    return function(par, wc,  l1, l2)

def bqll_obs_lsum(function, wc_obj, par,  l1, l2):
    if l1 == l2:
        raise ValueError("This function is defined only for LFV decays")
    scale = config['renormalization scale']['dll']
    wc12 = wc_obj.get_wc('cu'+l1+l2, scale, par)
    wc21 = wc_obj.get_wc('cu'+l2+l1, scale, par)
    return function(par, wc12, l1, l2) + function(par, wc21, l2, l1)

def bqll_obs_function(function,  l1, l2):
    return lambda wc_obj, par: bqll_obs(function, wc_obj, par, l1, l2)

def bqll_obs_function_lsum(function,  l1, l2):
    return lambda wc_obj, par: bqll_obs_lsum(function, wc_obj, par,  l1, l2)


# D0 -> l+l- effective lifetime

# def tau_ll(wc, par, lep):
    # r"""Effective D0->l+l- lifetime as defined in eq. (26) of arXiv:1204.1737 .
    # This formula one either gets by integrating eq. (21) or by inverting eq. (27) of arXiv:1204.1737.

    # Parameters
    # ----------

    # - `wc`         : dict of Wilson coefficients
    # - `par`        : parameter dictionary
    # - `lep`        : lepton: 'e', 'mu' or 'tau'

    # Returns
    # -------

    # $-\frac{\tau_{D_0} \left(y_s^2+2 A_{\Delta\Gamma_q} ys+1\right)}{\left(ys^2-1\right) (A_{\Delta\Gamma_q} ys+1)}$
    # """
    # ADG    = ADeltaGamma(par, wc,  lep)
    # y      = .5*par['DeltaGamma/Gamma_D0']
    # tauB   = par['tau_D0']
    # return -(((1 + y**2 + 2*y*ADG)*tauB)/((-1 + y**2)*(1 + y*ADG)))

# def tau_ll_func(wc_obj, par, lep):
    # scale = config['renormalization scale']['dll']
    # label = 'cu'+lep+lep
    # wc = wctot_dict(wc_obj, label, scale, par)
    # return tau_ll(wc, par, lep)

# def ADG_func(wc_obj, par,  lep):
    # scale = config['renormalization scale']['dll']
    # label = 'cu'+lep+lep
    # wc = wctot_dict(wc_obj, label, scale, par)
    # return ADeltaGamma(par, wc, lep)

# def ADeltaGamma_func( lep):
    # def ADG_func(wc_obj, par):
        # scale = config['renormalization scale']['dll']
        # label = 'cu'+lep+lep
        # wc = wctot_dict(wc_obj, label, scale, par)
        # return ADeltaGamma(par, wc,  lep)
#     return ADG_func
# Observable and Prediction instances

_tex = {'e': 'e', 'mu': '\mu', 'tau': r'\tau'}
for l in ['e', 'mu', 'tau']:
    _process_taxonomy = r'Process :: $c$ hadron decays :: FCNC decays :: $D_0\to\ell^+\ell^-$ :: $'

    # For the D^0 decay, we take the prompt branching ratio since DeltaGamma is negligible ###### CHECK
    _obs_name = "BR(D0->"+l+l+")"
    _obs = Observable(_obs_name)
    _process_tex = r"D^0\to "+_tex[l]+r"^+"+_tex[l]+r"^-"
    _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
    _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r"$")
    Prediction(_obs_name, bqll_obs_function(br_inst, 'D0', l, l))


_tex_B = {'D0': r'\bar D^0'}
_tex_lfv = {'emu': r'e^+\mu^-', 'mue': r'\mu^+e^-',
    'taue': r'\tau^+e^-', 'etau': r'e^+\tau^-',
    'taumu': r'\tau^+\mu^-', 'mutau': r'\mu^+\tau^-'}
for ll_1 in [('e','mu'), ('e','tau'), ('mu','tau'),]:
    ll_2 = ll_1[::-1] # now if ll_1 is (e, mu), ll_2 is (mu, e)
    for ll in [ll_1, ll_2]:
        # the individual BRs
        _obs_name = "BR(D0->"+''.join(ll)+")"
        _obs = Observable(_obs_name)
        _process_tex = _tex_B['D0']+r"\to "+_tex_lfv[''.join(ll)]
        _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        _obs.add_taxonomy(r'Process :: $c$ hadron decays :: FCNC decays :: $D0\to\ell^+\ell^-$ :: $'  + _process_tex + r'$')
        Prediction(_obs_name, bqll_obs_function(br_inst, 'D0', ll[0], ll[1]))

    # the individual BR where ll' and l'l are added
    _obs_name = "BR(D0->"+''.join(ll_1)+","+''.join(ll_2)+")"
    _obs = Observable(_obs_name)
    for ll in [ll_1, ll_1]:
        _process_tex = _tex_B['D0']+r"\to "+_tex_lfv[''.join(ll)]
        _obs.add_taxonomy(r'Process :: $c$ hadron decays :: FCNC decays :: $D0\to\ell^+\ell^-$ :: $'  + _process_tex + r'$')
        _process_tex = _tex_B['D0']+r"\to "+ll_1[0]+r"^\pm "+ll_1[1]+r"^\mp"
        _obs.tex = r"$\text{BR}(" + _process_tex + r")$"
        _obs.set_description(r"Branching ratio of $" + _process_tex + r"$")
        Prediction(_obs_name, bqll_obs_function_lsum(br_inst, 'D0', ll_1[0], ll_1[1]))

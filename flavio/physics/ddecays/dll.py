r"""Functions for $D\to\ell\nu$."""

import flavio
from flavio.physics.common import lambda_K
from math import pi, log

## function needs to be carefully checked

def br_Dll(wc_obj, par, lep):
    r"""Branching ratio of $D^+\to\ell^+\ell^-$."""
    qiqj = 'cu'
    scale = flavio.config['renormalization scale']['dll']
    al=flavio.physics.running.running.get_alpha_e(par,scale)
    Vij = flavio.physics.ckm.get_ckm(par) # Vcd*
    prefac = par['G_F']*al*Vij[1,2].conj() * Vij[0,2] *al/(sqrt(2)*pi)
    # Wilson coefficients
    wc = wc_obj.get_wc(qiqj, scale, par, nf_out=4)
    # SM contribution is negligible (not included)
    mc = flavio.physics.running.running.get_mc(par, scale)
    mu = flavio.physics.running.running.get_mu(par, scale)
    fD=par['f_D+']
    mD=par['m_D0']
    xl=par['m_'+lep]/mD
    xu=mu/mD
    xc=md/mD
    CeV = xl* (wc['C10p_cu'+lep+lep]-wc['C10_cu'+lep+lep])
    CeSR = mc/(xu+xc)*(wc['CS_cu'+lep+lep]+wc['CP_cu'+lep+lep] - (wc['CSp_cu'+lep+lep]+wc['CSp_cu'+lep+lep]))/2
    CeSL = mc/(xu+xc)*(wc['CS_cu'+lep+lep]-wc['CP_cu'+lep+lep] - (wc['CSp_cu'+lep+lep]-wc['CSp_cu'+lep+lep]))/2
    return par['tau_D0']*lambda_K(1,xl**2,xl**2)*fD**2/(128*pi*mD)*prefac**2*((abs(CeSL-CeV)**2+abs(CeSR-CeV)**2) -2*xl**2 *abs(CeSL+CeSR)**2)

# function returning function needed for prediction instance
def br_Dll_fct(lep):
    def f(wc_obj, par):
        return br_Dll(wc_obj, par, lep)
    return f

# Observable and Prediction instances

_lep = {'e': 'e', 'mu': r'\mu', 'tau': r'\tau'}

for l in _lep:
    _process_tex = r"D_0\to "+_lep[l]+r"^+"+_lep[l]+r"^-"
    _process_taxonomy = r'Process :: $c$ hadron decays :: Leptonic neutral D meson decays :: $D\to \ell\bar\ell$ :: $'
    _obs_name = "BR(D0->"+l+l+")"
    _obs = flavio.classes.Observable(_obs_name)
    _obs.set_description(r"Branching ratio of $"+_process_tex+r"$")
    _obs.tex = r"$\text{BR}("+_process_tex+r")$"
    _obs.add_taxonomy(_process_taxonomy + _process_tex + r'$')
    flavio.classes.Prediction(_obs_name, br_Dll_fct(l))

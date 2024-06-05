import mosek


def from_str_to_mosek_bnd_key(s):
    if s == 'fx':
        rtn = mosek.boundkey.fx
    elif s == 'lo':
        rtn = mosek.boundkey.lo
    elif s == 'up':
        rtn = mosek.boundkey.up
    elif s == 'ra':
        rtn = mosek.boundkey.ra
    else:
        assert False
    return rtn

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sin
import scipy.interpolate as sip
import scipy.stats as sst
import scipy.interpolate

def flexknots_to_function(kwargs, interp_method=scipy.interpolate.interp1d, debug=False,
                          min_pos=5, max_pos=30, left_val=1, right_val=0,
                          move_endpoints=None, n_params=None, pos='x', val='y',
                          return_pop_kwargs=True, return_hdof=False):
    r"""Interpolate function from given FlexKnot parameters.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    n_params : int, optional
        The number of free parameters. In the original FlexKnot this
        is 2 * number of knots, in the move_endpoints version it is
        2 * number of (fully movable) knots + 2. Can be inferred from
        keyword arguments
    move_endpoints: bool, optional
        Whether or not to allow moving of the end points, otherwise
        they are fixed to left_val and right_val respectively. Can be
        inferred from keyword arguments.
    debug : bool, optional
        Print debug output like which type of FlexKnot used and
        what parameters are received.
    pos: str, optional
        String to use for position coordinate (default: 'x')
    val: str, optional
        String to use for value coordinate (default: 'y')
    min_pos: float, optional
        Minimum position (e.g. redshift) of points (default: 5)
    max_pos: float, optional
        Maximum position (e.g. redshift) of points (default: 30)
    left_val: float, optional
        Value (e.g. ionized fraction) to assume left of min_pos
    right_val: float, optional
        Value (e.g. ionized fraction) to assume right of max_pos
    interp_method: scipy method
        Which interpolation method to use, e.g. scipy.interpolate.interp1d
        or scipy.interpolate.pchip (default: interp1d)
    return_pop_kwargs: bool, optional
        Whether to remove kwargs after use, or leave kwargs unchanged.
        Default: true
    return_hdof: bool, optional
        Return "half the number of degrees of freedom", i.e. n_params/2,
        sometimes helpful to keep frack of knots.
        Default: False
    ** kwargs: dict
        Parameters for the knots, consisting of positions (x_i here)
        and values (y_i here).
        In the case of fixed endpoints (move_endpoints=False) pass from
        (x_1, y_1) to (x_n, y_n), where n is the number of fully movable
        knots, no parameters are allowed (no knots).
        In the case of moving endpoints (move_endpoints=True) pass at least
        x_1 and x_{n+2}, and in addition (x_2, y_2) to (x_{n+1}, y_{n+1})
        where n in the number of fully movable knots (n >= 0).

    Returns
    -------
    y_of_x : function
        Function y(x) returning the FlexKnot interpolation for any x.
    """
    if not return_pop_kwargs:
        kwargs = deepcopy(kwargs)
    keys = kwargs.keys()
    if move_endpoints is None:
        move_endpoints = not val+"1" in keys
        if debug:
            print("Assuming move_endpoints =", move_endpoints)
    if n_params is None:
        n_params = 2 if move_endpoints else 0
        for key in keys:
            if key[0]==val:
                n_params += 2
        if debug:
            print("Assuming", n_params, "params with kwargs", kwargs)

    fill_value = (left_val, right_val)
    assert n_params%2==0
    n_fully_movable = int(n_params/2-1 if move_endpoints else n_params/2)
    if not move_endpoints:
        xinterp = [min_pos]
        yinterp = [left_val]
        for i in range(n_fully_movable):
            xinterp.append(kwargs.pop(pos+str(i+1)))
            yinterp.append(kwargs.pop(val+str(i+1)))
        xinterp.append(max_pos)
        yinterp.append(right_val)
    else:
        xinterp = [kwargs.pop(pos+'1')]
        yinterp = [left_val]
        for i in range(n_fully_movable):
            xinterp.append(kwargs.pop(pos+str(i+2)))
            yinterp.append(kwargs.pop(val+str(i+2)))
        xinterp.append(kwargs.pop(pos+str(n_fully_movable+2)))
        yinterp.append(right_val)
    xinterp = np.array(xinterp)
    yinterp = np.array(yinterp)
    y_of_x = interp_method(xinterp, yinterp, fill_value=fill_value, bounds_error=False)
    if return_pop_kwargs:
        if return_hdof:
            return y_of_x, kwargs, int(n_params/2)
        else:    
            return y_of_x, kwargs
    else:
        if return_hdof:
            return y_of_x, int(n_params/2)
        else:
            return y_of_x
        

def flexknot_cobaya_params(n_knots=None, n_params=None, move_endpoints=False,
                           xmin=5, xmax=30, yleft=1, yright=0, debug=False,
                           monotonous=False, dx=1e-3):
    r"""Create a cobaya params dictionary for sampling FlexKnots.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    n_knots : int, optional
        The number of fully movable knots, interchangeable with
        n_params (pass only one). This corresponds
        to 0.5 * n_params if move_endpoints=False, or
        to 0.5 * (n_params-2) if move_endpoints=True.
        free parameters. In the original FlexKnot this
        is 2 * number of knots, in the move_endpoints version it is
        2 * number of (fully movable) knots + 2. Default: 0
    n_params : int, optional
        The number of free parameters, interchangeable with the
        number of knots. In the original FlexKnot (move_endpoints=False)
        this is 2 * number of knots, in the move_endpoints version it is
        2 * number of (fully movable) knots + 2. Default: None
    move_endpoints: bool, optional
        Whether or not to allow moving of the end points, otherwise
        they are fixed to yleft and yright respectively (Default: False).
    monotonous: bool, optional
        Whether to force the points to be monotonously decreasing
        or increasing. Increase/decrease is inferred from `ymax`-`ymin`.
        Default: True.
    debug : bool, optional
        Print debug output like the dictionary values created.
    xmin: float, optional
        Minimum position (e.g. redshift) of points (default: 5)
    xmax: float, optional
        Maximum position (e.g. redshift) of points (default: 30)
    dx: float, optional
        Small gap between xmax/min and actual boundaries to avoid
        fgivenx issues. Defaukt: 1e-3
    yleft: float, optional
        Value (e.g. ionized fraction) to assume left of xmin
    yright: float, optional
        Value (e.g. ionized fraction) to assume right of xmax

    Returns
    -------
    y_of_x : function
        Function y(x) returning the FlexKnot interpolation for any x.
    """
    # Check arguments
    assert np.logical_xor(n_params is None, n_knots is None), "Pass either n_knots XOR n_params as not None!"
    if n_params is None:
        n_params = 2*n_knots+2 if move_endpoints else 2*n_knots
    if n_knots is None:
        n_knots = (n_params-2)/2 if move_endpoints else n_params/2
        assert n_knots%1 == 0
        n_knots = int(n_knots)
    # Currently only monotonously *decreasing* functions
    assert yleft > yright, "yleft<yright not implemented yet"
    # Formula from Handley et al. 2015 (arXiv/1506.00171), appendix A2.
    # Note the typo in eq. (A13) which should have 1-x^() instead oif x^()
    # theta_i = theta_{i-1} + (theta_max-theta_{i-1}) (1-x_i^(1/(n-i+1)))
    params_dict = {}
    if monotonous:
        if not move_endpoints:
            params_dict['x0'] = {"value": 1, "drop": True}
            params_dict['z0'] = {"value": xmin+dx, "drop": True}
            for i in range(n_knots):
                j = str(i+1)
                k = str(i)
                params_dict['u'+j] = {"prior": {"dist": "uniform", "min": 0,  "max": 1}, "latex": r"u_"+j, "drop": True}
                params_dict['v'+j] = {"prior": {"dist": "uniform", "min": 0,  "max": 1}, "latex": r"v_"+j, "drop": True}
                params_dict['x'+j] = {"value" : "lambda u"+j+",x"+k+": x"+k+"*u"+j+"**(1/("+str(n_knots)+"-"+j+"+1))", "min": yright, "max": yleft, "latex": r"x_"+str(j)}
                params_dict['z'+j] = {"value" : "lambda v"+j+",z"+k+": z"+k+"+(30-z"+k+")*(1-v"+j+"**(1/("+str(n_knots)+"-"+j+"+1)))", "min": xmin+dx, "max": xmax-dx, "latex": r"z_"+str(j)}
        else:
            params_dict['x1'] = {"value": 1, "drop": True}
            for i in range(n_knots):
                j = str(i+1)
                k = str(i)
                k2 = str(i+1)
                params_dict['u'+j] = {"prior": {"dist": "uniform", "min": 0,  "max": 1}, "latex": r"u_"+j, "drop": True}
                params_dict['x'+str(i+2)] = {"value" : "lambda u"+j+",x"+k2+": x"+k2+"*u"+j+"**(1/("+str(n_knots)+"-"+j+"+1))", "min": yright, "max": yleft, "latex": r"x_"+str(i+2)}
            params_dict['z0'] = {"value": xmin+dx, "drop": True}
            for i in range(n_knots+2):
                j = str(i+1)
                k = str(i)
                params_dict['v'+j] = {"prior": {"dist": "uniform", "min": 0,  "max": 1}, "latex": r"v_"+j, "drop": True}
                params_dict['z'+j] = {"value" : "lambda v"+j+",z"+k+": z"+k+"+(30-z"+k+")*(1-v"+j+"**(1/("+str(n_knots+2)+"-"+j+"+1)))", "min": xmin+dx, "max": xmax-dx, "latex": r"z_"+str(j)}
    else:
        assert not move_endpoints, "move_endpoints not yet implemented for non-monotonous flexknot"
        for i in range(n_knots):
            params_dict['x'+str(i+1)] = {"prior": {"min": yright, "max": yleft}}
            params_dict['z'+str(i+1)] = {"prior": {"min": xmin+dx, "max": xmax-dx}}
    if debug:
        print("Returning dict")
        for key, val in params_dict.items():
            print(key,":",val)
    return params_dict
    

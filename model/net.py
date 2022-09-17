"""Defines the detector network structure."""


def fetch_net(params):

    if params.net_type == "yolox":
        from model.yolox import get_model
    elif params.net_type == "dmpr":
        from model.dmpr import get_model
    elif params.net_type == "yolox_single_scale":
        from model.yolox_single_scale import get_model
    elif params.net_type == "psdet":
        from model.psdet import get_model
    else:
        raise NotImplementedError("Unkown model: {}".format(params.net_type))
    return get_model(params)
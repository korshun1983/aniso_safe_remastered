"""Step 1: initialization, ported from private/St1_*.m and spectrum/St1_4_*.m."""

from __future__ import annotations

from copy import deepcopy

from .structures import AttrDict, as_attrdict


def St1_1_SetModelConfig():
    """Set toolbox configuration, port of St1_1_SetModelConfig.m."""
    InputParam = AttrDict()
    InputParam.Config = AttrDict()
    InputParam.Config.ProblemType = "spectrum"
    InputParam.Config.NumMethod = "SAFE"
    InputParam.Config.SpeedUp = "no"
    InputParam.Config.SaveData = "yes"
    InputParam.Config.OuterBC = "fixed"
    InputParam.Config.PML = "r2"
    InputParam.Config.Eccentricity = "no"
    InputParam.Config.Symmetry = "none"
    InputParam.Config.EigenVar = "k"
    InputParam.Config.SloUnits = "us/ft"
    InputParam.Config.FreqUnits = "kHz"
    InputParam.Config.PressureUnits = "GPa"
    InputParam.Config.CheckAsymptote = "yes"
    InputParam.Config.DisplayAttenuation = "yes"
    return InputParam


def St1_2_PrepareModelMethods(InputParam):
    """Assign method names, port of St1_2_PrepareModelMethods.m.

    MATLAB stores function handles. The Python port stores stable method
    names and resolves them in the orchestrator to keep the structure
    serializable.
    """
    InputParam.Methods = AttrDict()
    InputParam.Methods.St1_3_SetModelUser = "St1_3_SetModelUser_sp_SAFE"
    InputParam.Methods.chebdif = "chebdif"
    InputParam.Methods.em_tensor_VTI = "em_tensor_VTI"
    InputParam.Methods.rot_c_ij = "rot_c_ij"
    InputParam.Methods.rot_matrix = "rot_matrix"
    InputParam.Methods.V_phase_VTI_exact_RPH = "V_phase_VTI_exact_RPH"
    InputParam.Methods.MeshFaces = "meshfaces"
    if InputParam.Config.ProblemType == "spectrum" and InputParam.Config.NumMethod == "SAFE":
        InputParam.Methods.ComputeAsymptotes = "ComputeAsymptotesSAFE"
        InputParam.Methods.St1_4_SetModelAdvanced = "St1_4_SetModelAdvanced_sp_SAFE"
        InputParam.Methods.St2_PrepareModel = "St2_PrepareModel_sp_SAFE"
        InputParam.Methods.St2_1_PrepareModelParams = "St2_1_PrepareModelParams_sp_SAFE"
        InputParam.Methods.St2_2_PrepareModelMethods = "St2_2_PrepareModelMethods_sp_SAFE"
        InputParam.Methods.St3_PrepareBasicMatrices = "St3_PrepareBasicMatrices_sp_SAFE"
        InputParam.Methods.St4_ComputeSolution = "St4_ComputeSolution_sp_SAFE"
    else:
        raise NotImplementedError("Only spectrum/SAFE configuration is ported")
    return InputParam


def St1_3_SetModelUser_sp_SAFE(InputParam, user_model):
    """Apply user model fields loaded from JSON, port of the input m-file role."""
    for section in ("Config", "Model", "Advanced", "Mesh"):
        if section in user_model:
            if section not in InputParam:
                InputParam[section] = AttrDict()
            InputParam[section].update(deepcopy(user_model[section]))
    InputParam.Model.N_disp = len(InputParam.Model.f_array)
    return InputParam


def St1_4_SetModelAdvanced_sp_SAFE(InputParam):
    """Set advanced defaults, port of St1_4_SetModelAdvanced_sp_SAFE.m."""
    if "Advanced" not in InputParam:
        InputParam.Advanced = AttrDict()
    defaults = AttrDict()
    defaults.VisualizeMesh = True
    defaults.N_nodes = 10
    defaults.NEdge_nodes = 4
    defaults.EigsOptions = AttrDict(disp=0, tol=1e-8)
    defaults.Source = AttrDict(
        xc=0,
        yc=0,
        r0x=0.06,
        r0y=0.06,
        theta_r=0,
        theta0=0,
        sigma=0.02,
        Plim=5e-4,
        symmetry=0,
    )
    for key, value in defaults.items():
        if key not in InputParam.Advanced:
            InputParam.Advanced[key] = value
    if "Mesh" not in InputParam:
        InputParam.Mesh = AttrDict()
    if "output" not in InputParam.Mesh:
        InputParam.Mesh.output = "no"
    return InputParam


def St1_SetModel(user_model=None):
    """Run Step 1 initialization in the same order as St1_SetModel.m."""
    InputParam = St1_1_SetModelConfig()
    InputParam = St1_2_PrepareModelMethods(InputParam)
    if user_model is None:
        user_model = AttrDict(Model=AttrDict())
    InputParam = St1_3_SetModelUser_sp_SAFE(InputParam, as_attrdict(user_model))
    InputParam = St1_4_SetModelAdvanced_sp_SAFE(InputParam)
    return InputParam

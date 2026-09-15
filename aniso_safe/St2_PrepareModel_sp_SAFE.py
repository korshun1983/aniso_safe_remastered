"""Step 2: data preparation, ported from spectrum/St2_*.m."""

from __future__ import annotations

import numpy as np

from .structures import AttrDict


def ComputeAsymptotesSAFE(CompStruct):
    """Compute asymptotes of the dispersion curves, port of ComputeAsymptotesSAFE.m.

    Computes the mud velocity, the exact VTI bulk phase velocities of the
    outer formation along the waveguide axis (via V_phase_VTI_exact_RPH)
    and the low-frequency Stoneley asymptote. Velocities are in km/s
    (raw model units: sqrt(GPa / (g/cm^3)) = km/s).
    """
    from . import fematrices as fe

    asymptotes = AttrDict()

    # Compute asymptote for mud layer, if it is present in the model
    mud_domain = int(CompStruct.Model.get("mud_domain", 0))
    if mud_domain != 0:
        # extract mud properties (1-based domain number in MATLAB)
        mud_properties = CompStruct.Model.DomainParam[mud_domain - 1]
        rho_mud = float(mud_properties[0])
        lambda_mud = float(mud_properties[1])
        # compute V_mud
        asymptotes.V_mud = float(np.sqrt(lambda_mud / rho_mud))

    # Compute asymptotes for the outer formation, if it is TTI
    n_domain = int(CompStruct.Data.N_domain)
    if CompStruct.Model.DomainType[n_domain - 1] == "HTTI":
        # extract outer formation parameters
        formation_properties = CompStruct.Model.DomainParam[n_domain - 1]
        rho = 1.0e3 * float(formation_properties[0])
        c_main = np.asarray(formation_properties[1:6], dtype=float)
        theta = float(formation_properties[6])
        # compute Christoffel equation solution according to
        # the Rock Physics Handbook
        v_qp, v_qsv, v_sh = fe.call_method(
            CompStruct.Methods.V_phase_VTI_exact_RPH, rho, c_main, theta
        )
        asymptotes.V_qP = 1.0e-3 * v_qp
        asymptotes.V_qSV = 1.0e-3 * v_qsv
        asymptotes.V_SH = 1.0e-3 * v_sh
        # compute the Stoneley wave speed (low-frequency asymptote)
        # for VTI homogeneous formation
        if mud_domain != 0:
            asymptotes.V_St = float(1.0 / np.sqrt(rho_mud * (1.0 / lambda_mud + 1.0 / c_main[4])))

    return asymptotes


def St2_1_PrepareModelParams_sp_SAFE(CompStruct):
    """Prepare model parameters, port of St2_1_PrepareModelParams_sp_SAFE.m."""
    CompStruct.Misc = AttrDict()
    if CompStruct.Config.FreqUnits == "Hz":
        CompStruct.Misc.F_conv = 1.0
    elif CompStruct.Config.FreqUnits == "kHz":
        CompStruct.Misc.F_conv = 1e3
    else:
        raise ValueError("Unsupported FreqUnits")

    if CompStruct.Config.SloUnits == "us/m":
        CompStruct.Misc.S_conv = 1e3
    elif CompStruct.Config.SloUnits == "us/ft":
        CompStruct.Misc.S_conv = 0.3048e3
    else:
        raise ValueError("Unsupported SloUnits")

    CompStruct.Data = AttrDict()
    CompStruct.Data.N_domain = len(CompStruct.Model.DomainType)
    dvar = []
    for domain_type in CompStruct.Model.DomainType:
        if domain_type == "fluid":
            dvar.append(1)
        elif domain_type == "HTTI":
            dvar.append(3)
        else:
            raise NotImplementedError(f"Unsupported DomainType: {domain_type}")
    CompStruct.Data.DVarNum = np.asarray(dvar, dtype=int)

    if CompStruct.Config.CheckAsymptote == "yes":
        CompStruct.Asymp = ComputeAsymptotesSAFE(CompStruct)
        if "V_SH" in CompStruct.Asymp:
            CompStruct.Advanced.V_min = CompStruct.Asymp.V_SH
        if "V_qP" in CompStruct.Asymp:
            CompStruct.Advanced.V_max = CompStruct.Asymp.V_qP
    return CompStruct


def St2_2_PrepareModelMethods_sp_SAFE(CompStruct):
    """Assign computation method names, port of St2_2_PrepareModelMethods_sp_SAFE.m."""
    CompStruct.Methods.St2_2_PrepareModelParams = "St2_2_PrepareModelParams_sp_SAFE"
    CompStruct.Methods.St3_ProblemFormulation = "St3_ProblemFormulation_sp_SAFE"
    CompStruct.Methods.St3_1_PrepareBasicMatrices = "St3_1_PrepareBasicMatrices_sp_SAFE"
    CompStruct.Methods.St4_ComputeSolution = "St4_ComputeSolution_sp_SAFE"
    CompStruct.Methods.PrepareMesh = "PrepareMesh_sp_SAFE"
    CompStruct.Methods.PrepareMeshBH = "PrepareMeshBH"
    CompStruct.Methods.FindBEdges = "FindBEdges"
    CompStruct.Methods.MakeContBEdges = "MakeContBEdges"
    CompStruct.Methods.FindEdgeOrient = "FindEdgeOrient"
    CompStruct.Methods.AddNodesCubic = "AddNodesCubic"
    CompStruct.Methods.FindPos = "FindPos_sp_SAFE"
    CompStruct.Methods.L1L2_int_matrix = "L1L2_int_matrix"
    CompStruct.Methods.L1L2L3_int_matrix = "L1L2L3_int_matrix"
    CompStruct.Methods.NL_matrix = "NL_matrix"
    CompStruct.Methods.dNL_matrices = "dNL_matrices"
    CompStruct.Methods.ConvolveMatrices = "ConvolveMatrices"
    CompStruct.Methods.ConvolveEdgeMatrices = "ConvolveEdgeMatrices"
    CompStruct.Methods.NLEdge_matrix = "NLEdge_matrix"
    CompStruct.Methods.AssembleBasicMatrices = "AssembleBasicMatrices_sp_SAFE"
    CompStruct.Methods.PreparePhysProp = []
    CompStruct.Methods.MatricesParts_sp_SAFE = []
    CompStruct.Methods.getPhysProps = []
    CompStruct.Methods.KM_matrix = []
    CompStruct.Methods.KM_el_matrix = []
    for domain_type in CompStruct.Model.DomainType:
        if domain_type == "fluid":
            CompStruct.Methods.PreparePhysProp.append("PreparePhysProp_fluid_sp_SAFE")
            CompStruct.Methods.MatricesParts_sp_SAFE.append("MatricesParts_fluid_sp_SAFE_cubic")
            CompStruct.Methods.getPhysProps.append("getPhysProps_fluid")
            CompStruct.Methods.KM_matrix.append("KM_matrix_fluid")
            CompStruct.Methods.KM_el_matrix.append("KM_el_matrix_fluid")
        elif domain_type == "HTTI":
            CompStruct.Methods.PreparePhysProp.append("PreparePhysProp_HTTI_sp_SAFE")
            CompStruct.Methods.MatricesParts_sp_SAFE.append("MatricesParts_HTTI_sp_SAFE_cubic")
            CompStruct.Methods.getPhysProps.append("getPhysProps_HTTI")
            CompStruct.Methods.KM_matrix.append("KM_matrix_HTTI")
            CompStruct.Methods.KM_el_matrix.append("KM_el_matrix_HTTI")
        else:
            raise NotImplementedError(f"Unsupported DomainType: {domain_type}")

    if str(CompStruct.Model.get("AddDomain_Exist", "none")).lower() == "yes":
        add_type = str(CompStruct.Model.AddDomainType).lower()
        mapping = {
            "pml": "KM_el_matrix_HTTI_PML",
            "abc": "KM_el_matrix_HTTI_ABC",
            "pml+abc": "KM_el_matrix_HTTI_PML_ABC",
        }
        if add_type not in mapping:
            raise ValueError("External Domain Type is not set")
        CompStruct.Methods.KM_el_matrix[-1] = mapping[add_type]

    CompStruct.Methods.IC_Matrices_sp_SAFE = []
    CompStruct.Methods.IC_matrix = []
    CompStruct.Methods.IC_el_matrix = []
    CompStruct.Methods.AssembleFullMatrices = []
    for ii in range(CompStruct.Data.N_domain - 1):
        pair = CompStruct.Model.DomainType[ii] + CompStruct.Model.DomainType[ii + 1]
        if pair in ("fluidHTTI", "HTTIfluid"):
            CompStruct.Methods.IC_Matrices_sp_SAFE.append("ICMatrices_fluid_HTTI_SAFE_cubic")
            CompStruct.Methods.IC_matrix.append("IC_matrix_FS")
            CompStruct.Methods.IC_el_matrix.append("IC_el_matrix_FS")
            CompStruct.Methods.AssembleFullMatrices.append("AssembleFullMatrices_fs_SAFE_cubic")
        elif pair in ("fluidfluid", "HTTIHTTI"):
            CompStruct.Methods.IC_Matrices_sp_SAFE.append("ICMatrices_ff_ss_SAFE_cubic")
            CompStruct.Methods.IC_matrix.append(None)
            CompStruct.Methods.IC_el_matrix.append(None)
            CompStruct.Methods.AssembleFullMatrices.append("AssembleFullMatrices_ff_ss_SAFE_cubic")
        else:
            raise NotImplementedError(f"Unsupported interface pair: {pair}")

    outer_bc = CompStruct.Model.BCType[-1]
    if outer_bc == "rigid":
        CompStruct.Methods.AssembleFullMatrices.append("AssembleFullMatrices_rigid_SAFE_cubic")
    elif outer_bc == "free":
        CompStruct.Methods.AssembleFullMatrices.append("AssembleFullMatrices_free_SAFE_cubic")
    else:
        raise NotImplementedError(f"Unsupported outer BC: {outer_bc}")
    CompStruct.Methods.RemoveRedundantVariables = "RemoveRedundantVariables_SAFE"
    return CompStruct


def St2_PrepareModel_sp_SAFE(InputParam):
    """Run Step 2 in the same order as St2_PrepareModel_sp_SAFE.m."""
    CompStruct = InputParam
    CompStruct = St2_1_PrepareModelParams_sp_SAFE(CompStruct)
    CompStruct = St2_2_PrepareModelMethods_sp_SAFE(CompStruct)
    return CompStruct

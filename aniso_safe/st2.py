"""Step 2: data preparation, ported from spectrum/St2_*.m."""

from __future__ import annotations

import numpy as np

from .structures import AttrDict


def _vti_velocities_km_s(param):
    """Compute simple VTI reference velocities in km/s.

    Density is given in kg/cm^3 and moduli in GPa, therefore
    v[km/s] = sqrt(C[GPa] / rho[kg/cm^3]).
    """
    arr = np.asarray(param, dtype=float)
    rho = arr[0]
    c11, c13, c33, c44, c66 = arr[1:6]
    return {
        "V_qP": float(np.sqrt(c11 / rho)),
        "V_SH": float(np.sqrt(c44 / rho)),
        "V_SV": float(np.sqrt(c44 / rho)),
    }


def ComputeAsymptotesSAFE(CompStruct):
    """Compute asymptotic reference velocities, port role of ComputeAsymptotesSAFE.m.

    This is the isotropic/VTI reference estimate used for search limits. The
    exact rotated HTI phase-velocity routine remains a separate numerical
    kernel to be ported from V_phase_VTI_exact_RPH.m.
    """
    ref_type = CompStruct.Model.RefDomainType[0]
    ref_param = CompStruct.Model.RefDomainParam[0]
    if ref_type == "HTTI":
        return AttrDict(_vti_velocities_km_s(ref_param))
    if ref_type == "fluid":
        rho, lam = float(ref_param[0]), float(ref_param[1])
        return AttrDict(V_qP=float(np.sqrt(lam / rho)), V_SH=0.0, V_SV=0.0)
    raise NotImplementedError(f"Unsupported RefDomainType: {ref_type}")


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

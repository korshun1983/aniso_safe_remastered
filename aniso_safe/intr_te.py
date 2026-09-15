"""Mode classification and TE-based dispersion plots.

Port of intr_aniso_TE.m. Loads the LayerTE-*.npz and Slowness.npz files
produced by proc_aniso_TE, integrates the kinetic energy over the radius,
decomposes it into azimuthal harmonics, classifies the modes by symmetry
(monopole, flexural +/-, quadrupole +/-, 3pole +/-) and by the PML energy
criteria, and saves the dispersion figures as PNG files.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .proc_te import load_compstruct, load_fematrices, resolve_data_dir
from .structures import AttrDict

# MATLAB ColorOrder used in intr_aniso_TE.m
COLOR_ORDER = [
    (0, 0, 1),
    (0, 0.5, 0),
    (1, 0, 0),
    (0, 0.75, 0.75),
    (0.75, 0, 0.75),
    (0.75, 0.75, 0),
    (0.25, 0.25, 0.25),
]

# mode classes: (attribute name, title suffix, marker index)
MODE_CLASSES = [
    ("Class_TE_0", " Monopole modes", 0),
    ("Class_TE_1p", " Flexural modes +", 0),
    ("Class_TE_1m", " Flexural modes -", 1),
    ("Class_TE_2p", " Quadrupole modes +", 0),
    ("Class_TE_2m", " Quadrupole modes -", 1),
    ("Class_TE_3p", " 3pole +", 0),
    ("Class_TE_3m", " 3pole -", 1),
]


def _load_layerte(path):
    """Load one LayerTE-<freq>-<layer>.npz file."""
    with np.load(path, allow_pickle=False) as data:
        return AttrDict(
            rr=data["rr"],
            TE_f_e_r=data["TE_fer"],
            cF_ur_rm_pm=data["cF_ur_rm_pm"],
            cF_uf_rm_pm=data["cF_uf_rm_pm"],
            cF_uz_rm_pm=data["cF_uz_rm_pm"],
            NHarm=int(data["NHarm"][0]),
            neigs=data["neigs"],
            nlayers=data["nlayers"],
        )


def _load_slowness(path):
    """Load the Slowness.npz file."""
    with np.load(path, allow_pickle=False) as data:
        return AttrDict(
            Result_Freq=data["Result_Freq"],
            Result_Velocity=data["Result_Velocity"],
            Result_Slowness=data["Result_Slowness"],
            Result_Attenuation=data["Result_Attenuation"],
        )


def _asymptote(CompStructA, name):
    """Return an asymptote velocity (km/s) if it is available, None otherwise."""
    return CompStructA.Asymp.get(name)


def intr_aniso_TE(
    dir_name,
    freqs=None,
    num_eigs_class_show=2,
    changer_12="y",
    max_harm_limit=0.1,
    use_class_TE_NT="y",
    use_class_PML_HTTI="n",
    show_modes=("y", "y", "y", "y", "n", "n", "n"),
    slowness="y",
    velocity="n",
    slowness_limits=(100.0, 250.0),
    velocity_limits=(1300.0, 3200.0),
    title_add=None,
    out_dir=None,
):
    """Classify modes by kinetic energy and save the dispersion figures.

    Port of intr_aniso_TE.m. Eigenvalue numbers stored in the Class_TE_*
    arrays are kept one-based (as in MATLAB) and converted back to zero-based
    when they index Python arrays.
    """
    dir_name = resolve_data_dir(dir_name)
    out_dir = Path(out_dir) if out_dir is not None else dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n\n============================================================================")
    print("Intr_Aniso_TE3 Program has been started!")
    t_start_prog = time.perf_counter()

    if use_class_TE_NT == "y" and use_class_PML_HTTI == "y":
        print("The first classification algorithm have been chosen!")
        print("Please, in future choose only one classification algorithm!")
        return None

    IntrTE3 = AttrDict()
    IntrTE3.Dir_Name = str(dir_name)
    # MATLAB default is freqs=0.5:0.25:25; here all computed frequencies are
    # used when the list is not given explicitly
    IntrTE3.freqs = freqs
    IntrTE3.num_eigs_class_show = num_eigs_class_show
    # if V_mode_1>qSV and V_mode_1 < V_mode_2, then change V_mode_1 and V_mode_2
    IntrTE3.changer_12 = changer_12
    IntrTE3.max_harm_limit = max_harm_limit  # for symmetry classification
    IntrTE3.use_class_TE_NT = use_class_TE_NT
    IntrTE3.use_class_PML_HTTI = use_class_PML_HTTI
    IntrTE3.show_modes = list(show_modes)  # 0-mode,1-mode(p/m),2-mode(p/m),3-mode(p/m)
    IntrTE3.slowness = slowness
    IntrTE3.velocity = velocity
    IntrTE3.slowness_limits = slowness_limits
    IntrTE3.velocity_limits = velocity_limits

    title_add = title_add if title_add is not None else dir_name.name

    # Load calculation parameters: CompStruct data
    CompStructA = load_compstruct(dir_name)

    # find input frequencies in the computed frequency array
    f_array = np.asarray(CompStructA.Model.f_array, dtype=float)
    n_disp = int(CompStructA.Model.N_disp)
    if IntrTE3.freqs is None:
        IntrTE3.freqs = list(f_array)
    nfreqs = []
    for freq_check in IntrTE3.freqs:
        find_freq = "n"
        for iff in range(n_disp):
            if f_array[iff] == freq_check:
                find_freq = "y"
                nfreqs.append(iff)  # 0-based index into f_array
                break
        if find_freq == "n":
            print("At least One frequency did not find!")
            return None
    IntrTE3.nfreqs = nfreqs
    IntrTE3.freqs_count = f_array
    IntrTE3.nfreqs_count = list(range(n_disp))

    # Load FEMatrices of the first requested frequency (for PhysProp)
    FEMatricesA = load_fematrices(dir_name / f"FEMatrices-{IntrTE3.freqs[0]:g}.npz")

    # Load the first LayerTE file to get the processing parameters
    first_te = _load_layerte(dir_name / f"LayerTE-{IntrTE3.freqs[0]:g}-1.npz")
    nharm = first_te.NHarm

    IntrTE3.nlayers = list(first_te.nlayers)
    IntrTE3.nlayers_count = list(range(len(CompStructA.Model.DomainType)))
    for ilayer in IntrTE3.nlayers:
        if ilayer not in IntrTE3.nlayers_count:
            print("Error in IntrTE3.nlayers!")
            return None

    # Build Eigs Array (0-based eigenvalue indices)
    IntrTE3.neigs = [int(v) for v in first_te.neigs]
    IntrTE3.max_neigs = int(CompStructA.Advanced.num_eig_max)  # number of calculated eigs
    IntrTE3.neigs_count = list(range(IntrTE3.max_neigs))
    for ieig in IntrTE3.neigs:
        if ieig not in IntrTE3.neigs_count:
            print("Error in IntrTE3.neigs!")
            return None

    # Load Slowness
    slow_data = _load_slowness(dir_name / "Slowness.npz")
    IntrTE3.freqs_slow_count = slow_data.Result_Freq
    IntrTE3.Slowness = slow_data.Result_Slowness
    IntrTE3.Velocity = slow_data.Result_Velocity

    s_conv = CompStructA.Misc.S_conv
    saved_figures = []

    if slowness == "y":
        # Slowness
        f_max = float(np.max(f_array))
        fig, ax = plt.subplots(num="slowness_full")
        ax.set_prop_cycle(color=COLOR_ORDER)
        ax.set_xlim(0.0, f_max + 1.0)
        ax.set_ylim(*slowness_limits)
        ax.plot(
            IntrTE3.freqs_count,
            IntrTE3.Slowness[0 : len(IntrTE3.freqs_count), 0 : IntrTE3.max_neigs],
            marker="o",
            markersize=3,
            linestyle="none",
            linewidth=2,
            color="r",
        )
        freq_var = np.concatenate([[0.0], IntrTE3.freqs_count, [f_max + 1.0]])
        for name, color in (("V_qP", "m"), ("V_qSV", "k"), ("V_SH", "b"), ("V_St", "m")):
            vel = _asymptote(CompStructA, name)
            if vel:
                ax.plot(freq_var, np.full(len(freq_var), s_conv / vel), linestyle="--", linewidth=2, color=color)
        ax.set_xlabel("f (kHz)")
        ax.set_ylabel(r"Slowness ($\mu$s/ft)")
        ax.set_title("")
        out_png = out_dir / "slowness_full.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(out_png)

    if velocity == "y":
        # Velocity
        f_max = float(np.max(f_array))
        fig, ax = plt.subplots(num="velocity_full")
        ax.set_prop_cycle(color=COLOR_ORDER)
        ax.set_xlim(0.0, f_max + 1.0)
        ax.set_ylim(*velocity_limits)
        cconst = 12.0 * 0.0254 * 1e6
        ax.plot(
            IntrTE3.freqs_count,
            cconst / IntrTE3.Slowness[0 : len(IntrTE3.freqs_count), 0 : IntrTE3.max_neigs],
            marker="o",
            markersize=3,
            linestyle="none",
            linewidth=2,
            color="r",
        )
        freq_var = np.concatenate([[0.0], IntrTE3.freqs_count, [f_max + 1.0]])
        for name, color in (("V_qP", "m"), ("V_qSV", "k"), ("V_SH", "b"), ("V_St", "m")):
            vel = _asymptote(CompStructA, name)
            if vel:
                ax.plot(freq_var, np.full(len(freq_var), 1e3 * vel), linestyle="--", linewidth=2, color=color)
        ax.set_xlabel("f (kHz)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("")
        out_png = out_dir / "velocity_full.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(out_png)

    # Massive #3. 1-value of harmonic TE, 2-eig number ==>
    #             3-value of Classification #1, 4-eig number, 5-slowness,
    #             6-value of Classification #2, 7-eig number, 8-slowness,
    #             10 - frequency
    shape = (len(IntrTE3.nfreqs_count), len(IntrTE3.neigs_count), 10)
    class_te = {name: np.zeros(shape) for name, _, _ in MODE_CLASSES}  # mode classes

    # ========================================================================
    # Freq Cycle
    # ========================================================================
    for iff_1, iff in enumerate(IntrTE3.nfreqs):
        print(
            f"\tNow fhe {iff_1 + 1}-th frequency ({IntrTE3.freqs_count[iff]:.2f} kHz) "
            f"is processed out of {len(IntrTE3.nfreqs)} (real {len(IntrTE3.nfreqs_count)})"
        )
        ifreq = iff

        # Load processing results
        res_layer = {}
        for ilayer in IntrTE3.nlayers:
            fname = dir_name / f"LayerTE-{IntrTE3.freqs_count[iff]:g}-{ilayer + 1}.npz"
            if not fname.exists():
                print(f"File is not existed! {fname}")
                return None
            res_layer[ilayer] = _load_layerte(fname)

        # ================================================================
        # A) Calculate T Energy for specified f, eig, r and h (harmonic)
        nrr_f = 0
        rr_f = []
        for ilayer in IntrTE3.nlayers:
            nrr_f += len(res_layer[ilayer].rr)
            rr_f.append(np.asarray(res_layer[ilayer].rr, dtype=float))
        rr_f = np.concatenate(rr_f)

        # The eigensolver may return fewer eigenvalues than requested at
        # some frequencies; process only the ones available in the files
        avail = min(res_layer[il].cF_ur_rm_pm.shape[0] for il in IntrTE3.nlayers)
        ieigs_here = [ieig for ieig in IntrTE3.neigs if ieig < avail]

        n_eigs = len(ieigs_here)
        te_f_eigs_r_h_p = np.zeros((n_eigs, nrr_f, nharm))
        te_f_eigs_r_h_m = np.zeros((n_eigs, nrr_f, nharm))
        te_f_eigs_r_h = np.zeros((n_eigs, nrr_f, nharm))

        for ieig in ieigs_here:
            for iharm in range(nharm):
                te_fer_h_p = []
                te_fer_h_m = []
                for ilayer in IntrTE3.nlayers:
                    # rho*v^2/2
                    var_te = (
                        0.5
                        * float(FEMatricesA.PhysProp[ilayer].rho)
                        * 1000.0
                        * (2.0 * np.pi * IntrTE3.freqs_count[ifreq] * CompStructA.Misc.F_conv) ** 2
                    )

                    var_p = (
                        np.abs(res_layer[ilayer].cF_ur_rm_pm[ieig, :, iharm, 0]) ** 2
                        + np.abs(res_layer[ilayer].cF_uf_rm_pm[ieig, :, iharm, 0]) ** 2
                        + np.abs(res_layer[ilayer].cF_uz_rm_pm[ieig, :, iharm, 0]) ** 2
                    )
                    var_p = var_te * var_p

                    var_m = (
                        np.abs(res_layer[ilayer].cF_ur_rm_pm[ieig, :, iharm, 1]) ** 2
                        + np.abs(res_layer[ilayer].cF_uf_rm_pm[ieig, :, iharm, 1]) ** 2
                        + np.abs(res_layer[ilayer].cF_uz_rm_pm[ieig, :, iharm, 1]) ** 2
                    )
                    var_m = var_te * var_m

                    te_fer_h_p.append(var_p)
                    te_fer_h_m.append(var_m)
                te_f_eigs_r_h_p[ieig, :, iharm] = np.concatenate(te_fer_h_p)
                te_f_eigs_r_h_m[ieig, :, iharm] = np.concatenate(te_fer_h_m)

                var_const = 1.0
                te_f_eigs_r_h[ieig, :, iharm] = var_const * (
                    te_f_eigs_r_h_p[ieig, :, iharm] + te_f_eigs_r_h_m[ieig, :, iharm]
                )

        # ================================================================
        # B) Integrate TE(f,eig,r,h) over r and normalize by max of h
        te_f_eigs_ir_h = np.zeros((n_eigs, nrr_f - 1, nharm))
        te_f_eigs_ir_nh = np.zeros((n_eigs, nrr_f - 1, nharm))

        for ieig in ieigs_here:
            for iharm in range(nharm):
                varl = te_f_eigs_r_h[ieig, 0 : nrr_f - 1, iharm]
                varr = te_f_eigs_r_h[ieig, 1:nrr_f, iharm]
                var_mas = 0.5 * (rr_f[0 : nrr_f - 1] * varl + rr_f[1:nrr_f] * varr) * (
                    rr_f[1:nrr_f] - rr_f[0 : nrr_f - 1]
                )
                # cumulative trapezoidal sum, identical to the MATLAB loop
                te_f_eigs_ir_h[ieig, :, iharm] = np.cumsum(var_mas)

            varmax = np.max(te_f_eigs_ir_h[ieig, nrr_f - 2, :])
            for iharm in range(nharm):
                te_f_eigs_ir_nh[ieig, :, iharm] = te_f_eigs_ir_h[ieig, :, iharm] / varmax

        # ================================================================
        # C) For Polarization
        # C.1) TE_PZp(f,eig,h)=Sum_{i=1}^Nrr abs( f_(-n)+f_n )
        # C.2) TE_PZm(f,eig,h)=Sum_{i=1}^Nrr abs( f_(-n)-f_n )
        te_pzp_f_e_h = np.zeros((n_eigs, nharm))
        te_pzm_f_e_h = np.zeros((n_eigs, nharm))

        for ieig in ieigs_here:
            for iharm in range(nharm):
                for ilayer in IntrTE3.nlayers:
                    var_p = res_layer[ilayer].cF_ur_rm_pm[ieig, :, iharm, 0]
                    var_m = res_layer[ilayer].cF_ur_rm_pm[ieig, :, iharm, 1]
                    te_pzp_f_e_h[ieig, iharm] += np.sum(np.abs(var_p + var_m))
                    te_pzm_f_e_h[ieig, iharm] += np.sum(np.abs(var_p - var_m))

        # ================================================================
        # D) Full TE integrated over the radius
        te_f_e_ir = np.zeros((n_eigs, nrr_f - 1))
        te_f_e_total = np.zeros(n_eigs)
        te_f_e_pml = np.zeros(n_eigs)
        te_f_e_nir = np.zeros((n_eigs, nrr_f - 1))

        for ieig in ieigs_here:
            te_fer = np.concatenate(
                [np.asarray(res_layer[ilayer].TE_f_e_r[ieig, :], dtype=float) for ilayer in IntrTE3.nlayers]
            )

            varl = te_fer[0 : nrr_f - 1]
            varr = te_fer[1:nrr_f]
            var_mas = 0.5 * (rr_f[0 : nrr_f - 1] * varl + rr_f[1:nrr_f] * varr) * (
                rr_f[1:nrr_f] - rr_f[0 : nrr_f - 1]
            )
            te_f_e_ir[ieig, :] = np.cumsum(var_mas)
            te_f_e_total[ieig] = te_f_e_ir[ieig, nrr_f - 2]

            # first radius of the last layer inside the concatenated grid
            var_ind = int(np.nonzero(rr_f == res_layer[IntrTE3.nlayers[-1]].rr[0])[0][0])
            if var_ind - 1 >= 0:
                te_f_e_pml[ieig] = te_f_e_ir[ieig, nrr_f - 2] - te_f_e_ir[ieig, var_ind - 1]
            else:
                # single-layer models have no adjacent region; the MATLAB
                # script would fail at this line, so keep the total energy
                te_f_e_pml[ieig] = te_f_e_ir[ieig, nrr_f - 2]

            te_f_e_nir[ieig, :] = te_f_e_ir[ieig, :] / te_f_e_ir[ieig, nrr_f - 2]

        # ================================================================
        # E) Calculate Max TE in each layer for specified f and eig
        te_f_l_e = np.zeros((n_eigs, len(IntrTE3.nlayers)))
        for ieig in ieigs_here:
            for ill, ilayer in enumerate(IntrTE3.nlayers):
                te_f_l_e[ieig, ill] = np.max(np.abs(res_layer[ilayer].TE_f_e_r[ieig, :]))

        # ================================================================
        # Classification by Symmetry and Polarization
        for ieig in ieigs_here:
            var_ind = te_f_eigs_ir_nh[ieig, nrr_f - 2, :] >= IntrTE3.max_harm_limit
            if var_ind.size == 0:
                print("Error!!!")
                return None
            # NOTE: eigenvalue numbers in the Class arrays stay one-based
            if nharm > 0 and var_ind[0]:  # monopole
                class_te["Class_TE_0"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 0]
                class_te["Class_TE_0"][ifreq, ieig, 1] = ieig + 1
            if nharm > 1 and var_ind[1]:  # dypole
                if te_pzp_f_e_h[ieig, 1] >= te_pzm_f_e_h[ieig, 1]:
                    class_te["Class_TE_1p"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 1]
                    class_te["Class_TE_1p"][ifreq, ieig, 1] = ieig + 1
                else:
                    class_te["Class_TE_1m"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 1]
                    class_te["Class_TE_1m"][ifreq, ieig, 1] = ieig + 1
            if nharm > 2 and var_ind[2]:  # quadrupole
                if te_pzp_f_e_h[ieig, 2] >= te_pzm_f_e_h[ieig, 2]:
                    class_te["Class_TE_2p"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 2]
                    class_te["Class_TE_2p"][ifreq, ieig, 1] = ieig + 1
                else:
                    class_te["Class_TE_2m"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 2]
                    class_te["Class_TE_2m"][ifreq, ieig, 1] = ieig + 1
            if nharm > 3 and var_ind[3]:  # 3d-pole
                if te_pzp_f_e_h[ieig, 3] >= te_pzm_f_e_h[ieig, 3]:
                    class_te["Class_TE_3p"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 3]
                    class_te["Class_TE_3p"][ifreq, ieig, 1] = ieig + 1
                else:
                    class_te["Class_TE_3m"][ifreq, ieig, 0] = te_f_eigs_ir_nh[ieig, nrr_f - 2, 3]
                    class_te["Class_TE_3m"][ifreq, ieig, 1] = ieig + 1

        # Classification by TE_PML/TE_Total< TE_PML_Total_Limit (Nguyen, Treyssede)
        # For real eigs this parameter have small value
        class_te_nt = te_f_e_pml / te_f_e_total

        # Classification by max TE(PML region)/max TE(adjacent region)
        # For real eigs this parameter have small value
        if te_f_l_e.shape[1] > 1:
            class_te_max_pml_hti = te_f_l_e[:, -1] / te_f_l_e[:, -2]
        else:
            class_te_max_pml_hti = np.zeros(n_eigs)

        # Sort the classified eigs by the two classification criteria
        for name, _, _ in MODE_CLASSES:
            cls = class_te[name]
            var_tmp_ind = cls[ifreq, :, 1]
            var_tmp_ind = var_tmp_ind[var_tmp_ind != 0].astype(int)  # one-based eig numbers

            order1 = np.argsort(class_te_nt[var_tmp_ind - 1], kind="stable")
            res1 = class_te_nt[var_tmp_ind[order1] - 1]
            n1 = len(res1)
            cls[ifreq, 0:n1, 2] = res1
            cls[ifreq, 0:n1, 3] = var_tmp_ind[order1]
            cls[ifreq, 0:n1, 4] = IntrTE3.Slowness[ifreq, var_tmp_ind[order1] - 1]

            order2 = np.argsort(class_te_max_pml_hti[var_tmp_ind - 1], kind="stable")
            res2 = class_te_max_pml_hti[var_tmp_ind[order2] - 1]
            n2 = len(res2)
            cls[ifreq, 0:n2, 5] = res2
            cls[ifreq, 0:n2, 6] = var_tmp_ind[order2]
            cls[ifreq, 0:n2, 7] = IntrTE3.Slowness[ifreq, var_tmp_ind[order2] - 1]

            cls[ifreq, 0:nharm, 9] = IntrTE3.freqs_count[ifreq]

    # ========================================================================
    # Plot Results
    freq_count = IntrTE3.freqs_count[IntrTE3.nfreqs]
    freq_count_a = np.concatenate([[freq_count[0] - 1.0], freq_count, [freq_count[-1] + 1.0]])
    v_qsv = _asymptote(CompStructA, "V_qSV")
    v_sh = _asymptote(CompStructA, "V_SH")
    qsv_val = np.full(len(freq_count) + 2, s_conv / v_qsv) if v_qsv else None
    sh_val = np.full(len(freq_count) + 2, s_conv / v_sh) if v_sh else None

    if IntrTE3.changer_12.lower() == "y" and qsv_val is not None:
        for iff in IntrTE3.nfreqs:
            for name, _, _ in MODE_CLASSES:
                cls = class_te[name]
                if (cls[iff, 1, 4] > qsv_val[0]) and (cls[iff, 0, 4] < cls[iff, 1, 4]):
                    for iind in (2, 3, 4):
                        cls[iff, 0, iind], cls[iff, 1, iind] = cls[iff, 1, iind], cls[iff, 0, iind]
                if (cls[iff, 1, 7] > qsv_val[0]) and (cls[iff, 0, 7] < cls[iff, 1, 7]):
                    for iind in (5, 6, 7):
                        cls[iff, 0, iind], cls[iff, 1, iind] = cls[iff, 1, iind], cls[iff, 0, iind]

    if IntrTE3.use_class_TE_NT == "y":
        ikey = 4  # slowness column of Classification #1 (MATLAB column 5)
    if IntrTE3.use_class_PML_HTTI == "y":
        ikey = 7  # slowness column of Classification #2 (MATLAB column 8)

    markers = ["o", "^"]
    fig_names = [
        "modes_monopole",
        "modes_flexural_plus",
        "modes_flexural_minus",
        "modes_quadrupole_plus",
        "modes_quadrupole_minus",
        "modes_3pole_plus",
        "modes_3pole_minus",
    ]

    for iwave, (name, title_suffix, marker_idx) in enumerate(MODE_CLASSES):
        if IntrTE3.show_modes[iwave] != "y":
            continue
        cls = class_te[name]
        fig, ax = plt.subplots(num=fig_names[iwave])
        ax.set_prop_cycle(color=COLOR_ORDER)
        ax.set_xlim(freq_count_a[0], freq_count_a[-1])
        ax.set_ylim(*IntrTE3.slowness_limits)
        ax.plot(
            IntrTE3.freqs_count,
            cls[:, 0 : IntrTE3.num_eigs_class_show, ikey],
            linestyle="none",
            marker=markers[marker_idx],
            markersize=3,
            linewidth=2,
        )
        if sh_val is not None:
            ax.plot(freq_count_a, sh_val, linestyle="--", linewidth=2, color="r")
        if qsv_val is not None:
            ax.plot(freq_count_a, qsv_val, linestyle="--", linewidth=2, color="b")
        ax.set_xlabel("f (kHz)")
        ax.set_ylabel(r"Slowness ($\mu$s/ft)")
        ax.set_title(f"{title_add}{title_suffix}")
        out_png = out_dir / f"{fig_names[iwave]}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_figures.append(out_png)

    # Save the classification arrays for further processing
    class_file = out_dir / "ClassTE.npz"
    np.savez_compressed(class_file, **class_te)

    print(f"Time for Intr_Aniso TE = {time.perf_counter() - t_start_prog:.1f}s")
    print("============================================================================")

    result = AttrDict(class_te)
    result.figures = saved_figures
    result.class_file = class_file
    return result

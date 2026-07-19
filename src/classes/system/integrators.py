# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Coefficient definitions for explicit symplectic integrators."""

from __future__ import annotations

import numpy as np


METHODS = [
    "Verlet",
    "FR",
    "Yo# with # any integer",
    "Yos6",
    "M2",
    "M4",
    "EFRL",
    "PEFRL",
    "VEFRL",
    "BM4",
    "BM6",
    "RKN4b",
    "RKN6b",
    "RKN6a",
    "ABA104",
    "ABA864",
    "ABA1064",
]


class SymplecticIntegrator:
    """Coefficients and stage ordering for a symmetric splitting method."""

    name: str
    alpha_s: np.ndarray
    alpha_o: np.ndarray

    def __init__(self, name: str) -> None:
        self.name = name
        if (name not in METHODS) and not name.startswith("Yo"):
            raise ValueError(f"The chosen integrator must be one of {METHODS}.")

        alpha_s: list[float] | np.ndarray
        if name == "Verlet":
            alpha_s = [0.5]
        elif name == "FR":
            theta = 1 / (2 - 2 ** (1 / 3))
            alpha_s = [theta / 2, theta / 2, 0.5 - theta]
        elif name == "Yos6":
            coefficients = [
                0.784513610477560,
                0.235573213359357,
                -1.17767998417887,
                1.31518632068390,
            ]
            alpha_s = [
                coefficients[0] / 2,
                coefficients[0] / 2,
                coefficients[1] / 2,
                coefficients[1] / 2,
                coefficients[2] / 2,
                coefficients[2] / 2,
                coefficients[3] / 2,
            ]
        elif name.startswith("Yo"):
            try:
                alpha_array = np.asarray([0.5])
                for order in range(1, int(name[2:]) // 2):
                    x1 = 1 / (2 - 2 ** (1 / (2 * order + 1)))
                    x0 = 1 - 2 * x1
                    reflected = np.concatenate((alpha_array, np.flip(alpha_array)))
                    reflected = np.concatenate((x1 * reflected, x0 * alpha_array))
                    alpha_array = reflected.copy()
                alpha_s = alpha_array
            except (TypeError, ValueError) as exc:
                raise NameError(f"{name} integrator not defined") from exc
        elif name.endswith("EFRL"):
            if name.startswith("V"):
                xi, lam, chi = (
                    0.1644986515575760,
                    -0.02094333910398989,
                    1.235692651138917,
                )
            elif name.startswith("P"):
                xi, lam, chi = (
                    0.1786178958448091,
                    -0.2123418310626054,
                    -0.06626458266981849,
                )
            else:
                xi, lam, chi = (
                    0.1720865590295143,
                    -0.09156203075515678,
                    -0.1616217622107222,
                )
            alpha_s = [
                xi,
                0.5 - lam - xi,
                lam + xi + chi - 0.5,
                0.5 - chi - xi,
            ]
        elif name == "M2":
            y = (2 * np.sqrt(326) - 36) ** (1 / 3)
            z = (y**2 + 6 * y - 2) / (12 * y)
            alpha_s = [z, 0.5 - z]
        elif name == "M4":
            alpha_s = [
                (14 - np.sqrt(19)) / 108,
                (146 + 5 * np.sqrt(19)) / 540,
                (-23 - 20 * np.sqrt(19)) / 270,
                (-2 + 10 * np.sqrt(19)) / 135,
                1 / 5,
            ]
        elif name == "BM4":
            alpha_s = [
                0.0792036964311957,
                0.1303114101821663,
                0.2228614958676077,
                -0.3667132690474257,
                0.3246481886897062,
                0.1096884778767498,
            ]
        elif name == "BM6":
            alpha_s = [
                0.050262764400392,
                0.098553683500650,
                0.314960616927694,
                -0.447346482695478,
                0.492426372489876,
                -0.425118767797691,
                0.237063913978122,
                0.195602488600053,
                0.346358189850727,
                -0.362762779254345,
            ]
        elif name == "RKN4b":
            alpha_s = [
                0.082984406417405,
                0.162314550766866,
                0.233995250731502,
                0.370877414979578,
                -0.409933719901926,
                0.059762097006575,
            ]
        elif name == "RKN6b":
            alpha_s = [
                0.041464998518262,
                0.081764777428009,
                0.116363894490058,
                0.174189903309500,
                -0.214196095413653,
                0.087146882788236,
                -0.011892898486655,
                -0.234438862575420,
                0.222927475154732,
                0.134281397641196,
                0.102388527145735,
            ]
        elif name == "RKN6a":
            alpha_s = [
                0.0378593198406116,
                0.053859832783850,
                0.048775800318585,
                0.135207369686421,
                -0.161075257952980,
                0.104540892120091,
                0.209700510951356,
                -0.204785822176643,
                0.074641362659228,
                0.069119764509130,
                0.037297935860413,
                0.291269757886391,
                -0.300064001014902,
                0.103652534528448,
            ]
        elif name == "ABA104":
            alpha_s = [
                0.04706710064597251,
                0.07181481672222451,
                0.1129421186948636,
                0.128108341856638,
                0.1545976638231982,
                -0.4278843305285221,
                0.4133542887856252,
            ]
        elif name == "ABA864":
            alpha_s = [
                0.07113342649822312,
                0.1119502609739741,
                0.129203166982666,
                0.1815796929159088,
                0.3398320688569059,
                -0.3663966873688647,
                0.03269807114118675,
            ]
        elif name == "ABA1064":
            alpha_s = [
                0.03809449742241219,
                0.05776438341466301,
                0.08753433270225074,
                0.116911820440748,
                0.0907158752847932,
                0.1263544726941979,
                0.3095552309573282,
                -0.3269306129163933,
            ]
        else:
            raise NameError(f"{name} integrator not defined")

        half_stages = np.asarray(alpha_s, dtype=float)
        self.alpha_s = np.concatenate((half_stages, np.flip(half_stages)))
        self.alpha_o = np.tile(np.asarray([1, 0], dtype=int), len(half_stages))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __str__(self) -> str:
        return self.name

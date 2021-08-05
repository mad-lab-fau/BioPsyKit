"""Module representing the Trier Social Stress Test (TSST) protocol."""
from typing import Optional, Dict, Union

from biopsykit.protocols import BaseProtocol


class TSST(BaseProtocol):
    """Class representing the Trier Social Stress Test (TSST)."""

    def __init__(
        self, name: Optional[str] = None, structure: Optional[Dict[str, Union[None, Dict[str, int]]]] = None, **kwargs
    ):
        """Class representing the Trier Social Stress Test (TSST).

        The general structure of the TSST can be specified by passing a ``structure`` dict to the constructor.

        Up to three nested structure levels are supported:
            * 1st level: ``study part``: Different parts of the study where the TSST was conducted, such as: "Pre",
              "TSST", and "Post".
            * 2nd level: ``phase``: Different TSST phases that belong to the same *study part*, such as:
              "Preparation", "Talk", "Math" (for study part "TSST") or "Questionnaires", "Rest"
              (for study part "Pre").
            * 3rd level: ``subphase``: Different TSST subphases that belong to the same *phase*.


        Parameters
        ----------
        name : str
            name of protocol or ``None`` to use "TSST" as default name. Default: ``None``
        structure : dict, optional
            nested dictionary specifying the structure of the TSST study.

            Up to three nested structure levels are supported:

            * 1st level: ``study part``: Different parts of the study where the TSST was conducted, such as: "Pre",
              "TSST", and "Post".
            * 2nd level: ``phase``: Different TSST phases that belong to the same *study part*, such as:
              "Preparation", "Talk", "Math" (for study part "TSST") or "Questionnaires", "Rest"
              (for study part "Pre").
            * 3rd level: ``subphase``: Different TSST subphases that belong to the same *phase*.

            If a study part has no division into finer phases (or a phase has no division into finer subphases) the
            dictionary value can be set to ``None``.
        **kwargs
            additional parameters to be passed to ``MIST`` and its superclass, ``BaseProtocol``, such as:

            * ``saliva_plot_params``: dictionary with parameters to style
              :meth:`~biopsykit.protocols.BaseProtocol.saliva_plot`
            * ``hr_mean_plot_params``: dictionary with parameters to style
              :meth:`~biopsykit.protocols.BaseProtocol.hr_mean_plot`
            * ``hr_ensemble_plot_params``: dictionary with parameters to style
              :meth:`~biopsykit.protocols.BaseProtocol.hr_ensemble_plot`


        Examples
        --------
        >>> from biopsykit.protocols import TSST
        >>> # Example: TSST study consisting of three parts. Only the TSST part consists of different phases
        >>> structure = {
        >>>     "Pre": None,
        >>>     "TSST": {
        >>>         "Preparation": 300,
        >>>         "Talk": 300,
        >>>         "Math": 300
        >>>     },
        >>>     "Post": None
        >>> }
        >>> TSST(name="TSST", structure=structure)


        References
        ----------
        Kirschbaum, C., Pirke, K.-M., & Hellhammer, D. H. (1993). The “Trier Social Stress Test” – A Tool for
        Investigating Psychobiological Stress Responses in a Laboratory Setting. *Neuropsychobiology*, 28, 76–81.
        https://doi.org/10.1159/000119004

        """
        if name is None:
            name = "TSST"
        if structure is None:
            structure = {"Part1": None, "TSST": {"Preparation": 300, "Talk": 300, "Math": 300}, "Part2": None}

        test_times = kwargs.pop("test_times", [0, 20])

        super().__init__(name=name, structure=structure, test_times=test_times, **kwargs)

        self.saliva_plot_params.update({"test_title": "TSST", "xlabel": "Time relative to TSST start [min]"})

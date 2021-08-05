"""Module representing the Montreal Imaging Stress Task (MIST) protocol."""
from typing import Dict, Tuple, Union, Optional

import matplotlib.pyplot as plt

from biopsykit.protocols import BaseProtocol


class MIST(BaseProtocol):
    """Class representing the Montreal Imaging Stress Task (MIST) protocol and data collected within a MIST study."""

    def __init__(
        self, name: Optional[str] = None, structure: Optional[Dict[str, Union[None, Dict[str, int]]]] = None, **kwargs
    ):
        """Class representing the Montreal Imaging Stress Task (MIST) protocol and data collected within a MIST study.

        The general structure of the MIST can be specified by passing a ``structure`` dict to the constructor.

        Up to three nested structure levels are supported:

        * 1st level: ``study part``: Different parts of the study where the MIST was conducted, such as: "Pre",
          "MIST", and "Post"
        * 2nd level: ``phase``: Different MIST phases that belong to the same *study
          part*, such as: "MIST1", "MIST2", "MIST3" (for study part "MIST") or
          "Questionnaires", "Rest", "Training" (for study part "Pre")
        * 3rd level: ``subphase``: Different MIST subphases that belong to the same *phase*, such as:
          "Baseline", "Arithmetic Task", "Feedback"


        Parameters
        ----------
        name : str, optional
            name of protocol or ``None`` to use "MIST" as default name. Default: ``None``
        structure : dict, optional
            nested dictionary specifying the structure of the MIST study.

            Up to three nested structure levels are supported:

            * 1st level: ``study part``: Different parts of the study where the MIST was conducted, such as: "Pre",
              MIST", and "Post"
            * 2nd level: ``phase``: Different MIST phases that belong to the same *study
              part*, such as: "MIST1", "MIST2", "MIST3" (for study part "MIST") or
              "Questionnaires", "Rest", "Training" (for study part "Pre")
            * 3rd level: ``subphase``: Different MIST subphases that belong to the same *phase*, such as:
              "Baseline", "Arithmetic Task", "Feedback"

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
        >>> from biopsykit.protocols import MIST
        >>> # Example: MIST study consisting of three parts. Only the MIST part consists of different
        >>> # phases and subphases
        >>> structure = {
        >>>     "Before": None,
        >>>     "MIST": {
        >>>         "MIST1": {"BL": 60, "AT": 240, "FB": 120},
        >>>         "MIST2": {"BL": 60, "AT": 240, "FB": 120},
        >>>         "MIST3": {"BL": 60, "AT": 240, "FB": 120}
        >>>     },
        >>>     "After": None
        >>> }
        >>> MIST(name="MIST", structure=structure)


        References
        ----------
        Dedovic, K., Renwick, R., Mahani, N. K., Engert, V., Lupien, S. J., & Pruessner, J. C. (2005).
        The Montreal Imaging Stress Task: Using functional imaging to investigate the effects of perceiving and
        processing psychosocial stress in the human brain. *Journal of Psychiatry and Neuroscience*, 30(5), 319–325.

        """
        if name is None:
            name = "MIST"

        if structure is None:
            structure = {
                "Part1": None,
                "MIST": {
                    "MIST1": {"BL": 60, "AT": 240, "FB": 0},
                    "MIST2": {"BL": 60, "AT": 240, "FB": 0},
                    "MIST3": {"BL": 60, "AT": 240, "FB": 0},
                },
                "Part2": None,
            }

        test_times = kwargs.pop("test_times", [0, 30])

        hr_mean_plot_params = {"xlabel": "MIST Subphases"}
        hr_mean_plot_params.update(kwargs.pop("hr_mean_plot_params", {}))

        saliva_plot_params = {"test_title": "MIST", "xlabel": "Time relative to MIST start [min]"}
        saliva_plot_params.update(kwargs.pop("saliva_plot_params", {}))

        kwargs.update({"hr_mean_plot_params": hr_mean_plot_params, "saliva_plot_params": saliva_plot_params})
        super().__init__(name=name, structure=structure, test_times=test_times, **kwargs)

    def hr_ensemble_plot(
        self, ensemble_id: str, subphases: Optional[Dict[str, int]] = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Draw heart rate ensemble plot.

        Parameters
        ----------
        ensemble_id : str
            identifier of the ensemble data to be plotted.
            Ensemble data needs to be computed using :meth:`~biopsykit.protocols.BaseProtocol.compute_hr_ensemble`
            first.
        subphases : dict, optional
            dictionary with phases (keys) and subphases (values - dict with subphase names and subphase durations) or
            ``None`` to retrieve MIST information from ``structure`` dict. When passing ``None``,
            it is assumed that the study part containing the MIST is named ``MIST``.
        **kwargs : dict, optional
            optional arguments for plot configuration to be passed to
            :meth:`~biopsykit.protocols.BaseProtocol.hr_ensemble_plot`


        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
            figure object
        ax : :class:`matplotlib.axes.Axes`
            axes object


        See Also
        --------
        :meth:`~biopsykit.protocols.BaseProtocol.compute_hr_ensemble`
            compute heart rate ensemble data
        :func:`~biopsykit.protocols.plotting.hr_ensemble_plot`
            Heart rate ensemble plot

        """
        if subphases is None:
            subphases = self.structure.get("MIST", {})
        return super().hr_ensemble_plot(ensemble_id, subphases, **kwargs)

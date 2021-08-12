"""Module representing the Cortisol Awakening Response (CAR) protocol."""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from biopsykit.plotting import lineplot
from biopsykit.protocols import BaseProtocol
from biopsykit.utils.datatype_helper import is_saliva_raw_dataframe


class CAR(BaseProtocol):
    """Class representing psychological protocols for assessing the cortisol awakening response (CAR)."""

    def __init__(self, name: Optional[str] = None, **kwargs):
        """Class representing psychological protocols for assessing the cortisol awakening response (CAR).

        Parameters
        ----------
        name : str, optional
            name of CAR study or ``None`` to use default name ("CAR"). Default: ``None``
        **kwargs
            additional parameters to be passed to ``CAR`` and its superclass, ``BaseProtocol``, such as:

            * ``car_saliva_plot_params``: parameters to style :meth:`~biopsykit.protocols.CAR.car_saliva_plot`

        """
        if name is None:
            name = "CAR"

        car_saliva_plot_params = {"xlabel": "Time after Awakening [min]", "ylabel": r"Cortisol [nmol/l]"}
        car_saliva_plot_params.update(kwargs.pop("car_saliva_plot_params", {}))
        self.car_plot_params = car_saliva_plot_params
        """Plot parameters to style :meth:`~biopsykit.protocols.CAR.car_saliva_plot`."""

        super().__init__(name, **kwargs)

    def car_saliva_plot(self, saliva_type: Optional[str] = "cortisol", **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Plot CAR saliva data as lineplot.

        Parameters
        ----------
        saliva_type : str, optional
            type of saliva data to plot. Default: ``cortisol``
        **kwargs : optional arguments to be passed to :func:`~biopsykit.plotting.lineplot`


        Returns
        -------
        fig : :class:`~matplotlib.figure.Figure`
            figure object
        ax : :class:`~matplotlib.axes.Axes`
            axes object


        See Also
        --------
        :func:`~biopsykit.plotting.lineplot`
            draw line plot with error bars

        """
        data = self.saliva_data[saliva_type]
        is_saliva_raw_dataframe(data, saliva_type)

        kwargs.setdefault("x", "sample")
        kwargs.setdefault("y", saliva_type)
        kwargs.setdefault("xticklabels", kwargs.get("sample_times", self.sample_times[saliva_type]))
        if len(data.index.get_level_values(kwargs.get("x")).unique()) != len(kwargs.get("xticklabels")):
            raise ValueError(
                "If samples have individual sample times for each subject, sample times for plotting must "
                "explicitely be provided by the 'sample_times' argument!"
            )
        kwargs.update(self.car_plot_params)
        return lineplot(data, **kwargs)

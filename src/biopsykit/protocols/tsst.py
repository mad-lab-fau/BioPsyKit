from typing import Optional, Sequence

from biopsykit.protocols import base


class TSST(base.BaseProtocol):
    """
    Class representing the Trier Social Stress Test (TSST).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        phases: Optional[Sequence[str]] = None,
        phase_durations: Optional[Sequence[int]] = None,
    ):
        if name is None:
            name = "TSST"
        super().__init__(name)

        self.tsst_times: Sequence[int] = [0, 20]

        self.phases: Sequence[str] = ["Prep", "Talk", "Arith"]
        """
        TSST Phases

        Names of TSST phases
        """

        self.phase_durations: Sequence[int] = [5 * 60, 5 * 60, 5 * 60]
        """
        TSST Phase Durations

        Total duration of phases in seconds
        """

        self.saliva_params = {
            "test_text": "TSST",
            "xaxis_label": "Time relative to TSST start [min]",
        }

        self._update_tsst_params(phases, phase_durations)

    def __str__(self) -> str:
        return """{}
        Phases: {}
        Phase Durations: {}
        """.format(
            self.name, self.phases, self.phase_durations
        )

    @property
    def tsst_times(self):
        return self.test_times

    @tsst_times.setter
    def tsst_times(self, tsst_times):
        self.test_times = tsst_times

    def _update_tsst_params(
        self, phases: Sequence[str], phase_durations: Sequence[int]
    ):
        if phases:
            self.phases = phases
        if phase_durations:
            self.phase_durations = phase_durations

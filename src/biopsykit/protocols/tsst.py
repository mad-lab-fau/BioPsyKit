from typing import Optional, Sequence

from biopsykit.protocols import _base


class TSST(_base.BaseProtocol):
    """Class representing the Trier Social Stress Test (TSST)."""

    def __init__(
        self,
        name: Optional[str] = None,
        phases: Optional[Sequence[str]] = None,
        phase_durations: Optional[Sequence[int]] = None,
    ):
        if name is None:
            name = "TSST"
        super().__init__(name)
        self.test_times = [0, 20]

        if phases is None:
            phases = ["Prep", "Talk", "Arith"]
        if phase_durations is None:
            phase_durations = [5 * 60, 5 * 60, 5 * 60]

        self.phases: Sequence[str] = phases
        """
        TSST Phases

        Names of TSST phases
        """

        self.phase_durations: Sequence[int] = phase_durations
        """
        TSST Phase Durations

        Total duration of phases in seconds
        """

        self.saliva_params = {
            "test_text": "TSST",
            "xaxis_label": "Time relative to TSST start [min]",
        }

    def __str__(self) -> str:
        if len(self.saliva_data) > 0:
            return """{}
            Saliva Type(s): {}
            Saliva Sample Times: {}
            Phases: {}
            Phase Durations: {}
            """.format(
                self.name, self.saliva_type, self.sample_times, self.phases, self.phase_durations
            )
        else:
            return """{}
            Phases: {}
            Phase Durations: {}""".format(
                self.name, self.phases, self.phase_durations
            )

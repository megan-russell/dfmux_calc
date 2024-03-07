import numpy as np
from scipy import constants as c
import current_sharing


class DfMuxSystem:
    """
    Represents a DfMux system with SQUID, bolometer, wiring harness, and other parasitic elements.
    """

    def __init__(
        self,
        squid_transimpedance=800.0,
        squid_dynamic_impedance=300.0,
        squid_input_noise=2.5e-12,
        squid_input_inductance=15e-9,
        stray_inductance=10e-9,
        n_series=False,
        n_parallel=False,
        power=False,
        linear_range=2e-6,
        snubber=False,
        snubber_capacitance=False,
        temperature=0.3,
        operating_resistance=0.7,
        loopgain=10,
        stray_resistance=0.07,
        saturation_power=2.5 * 0.24187821,
        optical_power=0.24187821,
        critical_temperature=0.170,
        bath_temperature=0.100,
        stripline_inductance=30e-12,
        parasitic_capacitance=1.0e-12,
        r48=0,
        wire_harness_resistance=30,
        wire_harness_capacitance=40e-12,
        wire_harness_inductance=0.75e-6,
    ):
        """
        Initializes the DfMuxSystem object with the given parameters.
        """

        # SQUID parameters
        self.squid_transimpedance = squid_transimpedance
        self.squid_dynamic_impedance = squid_dynamic_impedance
        self.squid_input_noise = squid_input_noise
        self.squid_input_inductance = squid_input_inductance
        self.stray_inductance = stray_inductance
        self.n_series = n_series
        self.n_parallel = n_parallel
        self.power = power
        self.linear_range = linear_range
        self.snubber = snubber
        self.snubber_capacitance = snubber_capacitance
        self.temperature = temperature
        self.nuller_cold = False

        # Bolometer parameters
        self.operating_resistance = operating_resistance
        self.loopgain = loopgain
        self.stray_resistance = stray_resistance
        self.saturation_power = saturation_power
        self.optical_power = optical_power
        self.critical_temperature = critical_temperature
        self.bath_temperature = bath_temperature

        # Parasitic parameters
        self.stripline_inductance = stripline_inductance
        self.parasitic_capacitance = parasitic_capacitance
        self.r48 = r48

        # Wiring harness parameters
        self.wire_harness_resistance = wire_harness_resistance
        self.wire_harness_capacitance = wire_harness_capacitance
        self.wire_harness_inductance = wire_harness_inductance

    def calculate_responsivity(self):
        """
        Calculates the TES responsivity, assuming no excess responsivity.

        TODO: implement excess responsivity
        """
        bias_power = self.saturation_power - self.optical_power
        return (
            np.sqrt(2 / self.operating_resistance / bias_power)
            * self.loopgain
            / (self.loopgain + 1)
        )

    def nei_to_nep(self, optical_power):
        """
        Converts NEI to NEP for the given optical power.
        """
        vbias = np.sqrt(
            (self.operating_resistance + self.stray_resistance)
            * (self.saturation_power - optical_power)
        )
        loop_attenuation = (self.operating_resistance - self.stray_resistance) / (
            self.operating_resistance + self.stray_resistance
        )
        responsivity = (
            np.sqrt(2)
            / vbias
            * self.loopgain
            * loop_attenuation
            / (
                1
                + self.loopgain
                * loop_attenuation
                * (self.operating_resistance - self.stray_resistance)
                / (self.operating_resistance + self.stray_resistance)
            )
        )
        return 1 / responsivity

    def wire_series_impedance(self, frequency):
        """
        Calculates the series impedance of the wiring harness at a given frequency.
        """
        return (
            2j * np.pi * frequency * self.wire_harness_inductance
            + self.wire_harness_resistance
        )

    def wire_get_abcd(self, frequency):
        """
        Calculates the ABCD matrix for the wiring harness at a given frequency.
        """
        wire_series = self.wire_series_impedance(frequency)
        wire_shunt = 2j * np.pi * frequency * self.wire_harness_capacitance
        gamma = np.sqrt(wire_series * wire_shunt)
        z0 = np.sqrt(wire_series / wire_shunt)
        b = z0 * np.sinh(gamma)
        a = np.cosh(gamma)  # Assuming no shunt resistor
        d = np.cosh(gamma)
        c = 1 / z0 * np.sinh(gamma)  # Assuming no shunt capacitor
        return a, b, c, d

    def wire_reff(self, frequency):
        """
        Calculates the effective resistance seen by the 1st stage amplifier.
        """
        a, b, c, d = self.wire_get_abcd(frequency)
        return np.abs(
            (b + d * self.squid_dynamic_impedance)
            / (a + c * self.squid_dynamic_impedance)
        )

    def wire_real_reff(self, frequency):
        """
        Calculates the real effective resistance without SAA dynamic impedance.
        """
        a, b, c, d = self.wire_get_abcd(frequency)
        return np.real(b / a)

    def wire_transfer_function(self, frequencies):
        """
        Calculates the voltage transfer function of the wiring harness.
        """
        a, b, c, d = self.wire_get_abcd(frequencies)
        self.tf = np.abs(
            1
            / (
                a
                + c * self.squid_dynamic_impedance
                - (b + d * self.squid_dynamic_impedance) / 1e6
            )
        )
        return self.tf

    def calculate_noise(self, frequencies, skip_spice=False):
        """
        Calculates the expected readout noise at the given frequency(ies).
        """

        self.f = np.array(frequencies)

        # Warm electronics noise
        carrier_noise = 2.9e-12 / (self.operating_resistance + self.stray_resistance)
        nuller_noise = (
            np.sqrt(0.38e-12**2 + 3.6e-12**2) if self.nuller_cold else 4.9e-12
        )
        warm_noise = np.sqrt(carrier_noise**2 + nuller_noise**2)

        # Demodulator chain noise (details in Joshua Montgomery's PhD thesis)
        reff = 1 / (1 / 10 + 1 / 100 + 1 / 150) + 1 / (
            1 / 4.22e3 + 1 / self.wire_reff(self.f)
        )
        first_amp_noise = np.sqrt(2) * np.sqrt((1.1e-9) ** 2 + (2.2e-12 * reff) ** 2)
        demod_noise = np.sqrt(
            first_amp_noise**2
            + 2
            * (
                0.23e-9**2
                + 0.14e-9**2
                + (8.36e-9 * self.wire_reff(self.f) / (self.wire_reff(self.f) + 4.22e3))
                ** 2
                + 4 * c.k * 300 * self.wire_harness_resistance
            )
        )

        if skip_spice:
            # Analytic approximation (details in Joshua's SPT-3G paper)
            saa_in_impedance = 2 * np.pi * self.f * self.squid_input_inductance
            on_res_comb_impedance = (
                2 * np.pi * self.f * self.stripline_inductance
                + self.operating_resistance
                + self.stray_resistance
            )
            if self.snubber:
                on_res_comb_impedance = 1 / (
                    1 / on_res_comb_impedance + 1 / self.snubber
                )
            c_r48 = 1 / (2 * np.pi * self.f * self.parasitic_capacitance) + self.r48
            self.csf = 1 / (
                c_r48
                / (
                    (on_res_comb_impedance + np.abs(self.wire_series_impedance(self.f)))
                    * saa_in_impedance
                    / (
                        on_res_comb_impedance
                        + saa_in_impedance
                        + np.abs(self.wire_series_impedance(self.f))
                    )
                    + np.abs(self.wire_series_impedance(self.f))
                    + c_r48
                )
                * on_res_comb_impedance
                / (on_res_comb_impedance + saa_in_impedance)
            )
        else:
            # Use PySpice for CSF calculation
            self.csf = current_sharing.get_csf(self)

        # Calculate wiring harness transfer function
        self.tf = np.array(self.wire_transfer_function(self.f))

        # Scale demodulator noise and SAA noise
        self.demod = demod_noise * self.csf / self.tf / self.squid_transimpedance
        self.saa_scale = self.squid_input_noise * self.csf * np.sqrt(2)

        # Bolometer Johnson noise
        self.jnoise = (
            np.sqrt(2)
            * 1
            / (1 + self.loopgain)
            * np.sqrt(4 * c.k * self.critical_temperature / self.operating_resistance)
        )

        # Add snubber Johnson noise if applicable
        if self.snubber:
            self.jnoise = np.sqrt(
                self.jnoise**2 + 4 * c.k * self.temperature / self.snubber
            )

        # Calculate total noise
        self.total_noise = np.sqrt(
            warm_noise**2 + self.jnoise**2 + self.demod**2 + self.saa_scale**2
        )


if __name__ == "__main__":
    dfmux_system = DfMuxSystem()

    # Calculate and print NEI
    frequency = 4e6
    dfmux_system.calculate_noise(frequencies=np.array([frequency]), skip_spice=False)
    nei = dfmux_system.total_noise[0] * 1e12  # in pA/rtHz
    print(f"NEI at {frequency / 1e6} MHz: {nei:.2f} pA/rtHz")

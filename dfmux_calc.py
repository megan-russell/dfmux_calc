"""
dfmux_calc is a module inside the dfmux_calc repository. The module contains a class called DfMuxSystem that
represents a DfMux system with SQUID, bolometer, wiring harness, and other parasitic elements. The class has
a method called calculate_noise that calculates the expected readout noise at the given frequency(ies). The
method optionally uses PySpice for CSF calculation.
"""

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
        # stray_inductance=10e-9,
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
        bias_resistance=0,
        bias_inductance=5e-9,
        Rnuller=3000,
    ):
        """
        Initializes the DfMuxSystem object with the given parameters.
        """

        # SQUID parameters
        self.squid_transimpedance = squid_transimpedance
        self.squid_dynamic_impedance = squid_dynamic_impedance
        self.squid_input_noise = squid_input_noise
        self.squid_input_inductance = squid_input_inductance
        # self.stray_inductance = stray_inductance
        self.n_series = n_series
        self.n_parallel = n_parallel
        self.power = power
        self.linear_range = linear_range
        self.snubber = snubber
        self.snubber_capacitance = snubber_capacitance
        self.temperature = temperature

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

        # Bias element
        self.bias_resistance = bias_resistance
        self.bias_inductance = bias_inductance

        # Nuller chain
        self.nuller_resistance = Rnuller  # Resistance of the nuller current stiffening resistor [ohms]
        self.nuller_cold = False

        # initialize other attributes
        self.tf = None
        self.f = None
        self.csf = None
        self.demod = None
        self.saa_scale = None
        self.inoise = None
        self.jnoise = None
        self.total_noise = None

    def calculate_responsivity(self):
        """
        Calculates the TES responsivity, assuming no excess responsivity.

        TODO: implement excess responsivity
        """
        bias_power = self.saturation_power - self.optical_power
        return np.sqrt(2 / self.operating_resistance / bias_power) * self.loopgain / (self.loopgain + 1)

    def nei_to_nep(self, optical_power):
        """
        Converts NEI to NEP for the given optical power.
        """
        vbias = np.sqrt((self.operating_resistance + self.stray_resistance) * (self.saturation_power - optical_power))
        loop_attenuation = (self.operating_resistance - self.stray_resistance) / (self.operating_resistance + self.stray_resistance)
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
        return 2j * np.pi * frequency * self.wire_harness_inductance + self.wire_harness_resistance

    def wire_get_abcd(self, frequency):
        """
        Calculates the ABCD matrix for the wiring harness at a given frequency.
        """
        wire_series = self.wire_series_impedance(frequency)
        wire_shunt = 2j * np.pi * frequency * self.wire_harness_capacitance
        gamma = np.sqrt(wire_series * wire_shunt)
        z0 = np.sqrt(wire_series / wire_shunt)
        b_element = z0 * np.sinh(gamma)
        a_element = np.cosh(gamma)  # Assuming no shunt resistor
        d_element = np.cosh(gamma)
        c_element = 1 / z0 * np.sinh(gamma)  # Assuming no shunt capacitor
        return a_element, b_element, c_element, d_element

    def wire_reff(self, frequency):
        """
        Calculates the effective resistance seen by the 1st stage amplifier.
        """
        a_element, b_element, c_element, d_element = self.wire_get_abcd(frequency)
        return np.abs((b_element + d_element * self.squid_dynamic_impedance) / (a_element + c_element * self.squid_dynamic_impedance))

    def wire_real_reff(self, frequency):
        """
        Calculates the real effective resistance without SAA dynamic impedance.
        """
        a_element, b_element, _, _ = self.wire_get_abcd(frequency)
        return np.real(b_element / a_element)

    def wire_transfer_function(self, frequencies):
        """
        Calculates the voltage transfer function of the wiring harness.
        """
        a_element, b_element, c_element, d_element = self.wire_get_abcd(frequencies)
        self.tf = np.abs(
            1 / (a_element + c_element * self.squid_dynamic_impedance - (b_element + d_element * self.squid_dynamic_impedance) / 1e6)
        )
        return self.tf

    def nei_johnson(self):
        """
        Temporary implementation of Johnson noise.
        Replaces the old implementation which assumed perfect voltage bias, but takes
        the stray resistance as the Thevenin impedance.

        TODO: implement proper circuit modeling to get the Thevenin equivalent circuit.
        We need to calculate V_open and I_short for each bolometer, then set
          z_thev = V_open / I_short
          V_thev = V_open
        This would be more accurate and remove the need to take into account separate "snubber" Johnson noise.

        For now, just take the `stray_resistance` as the Thevenin impedance.
        """
        T_b = self.critical_temperature
        R_b = self.operating_resistance
        L = self.loopgain
        T_thev = self.bath_temperature  # assume Thevenin-equivalent resistance is at the bath temperature
        z_thev = self.stray_resistance  # for now, use the stray resistance as the Thevenin-equivalent impedance
        return self.johnson_noise(T_b, R_b, L, T_thev, z_thev)

    def johnson_noise(self, T_b, R_b, L, T_thev, z_thev):
        """
        Implementation of  Eq. 11 from Tijmen's note, calculating the NEI of Johnson noise
        in the general case of a Thevenin-equivalent circuit.

        $$\mathrm{NEI}^2 = \frac{8kT_b R_b}{|Z|^2} \left|1 - \frac{\mathcal{L}}{1+\mathcal{L}} \frac{1}{Z} \frac{R_b^2 \left( R_b-R_{\mathrm{Th\acute{e}v}} \right)}{R_b^2 - |z_{\mathrm{Th\acute{e}v}}|^2} \right|^2 + \frac{8kT_{\mathrm{Th\acute{e}v}} R_{\mathrm{Th\acute{e}v}}}{|Z|^2} \left|1 - \frac{\mathcal{L}}{1+\mathcal{L}} \frac{1}{Z} \frac{R_b^2 \left( R_b-R_{\mathrm{Th\acute{e}v}} + |Z| \right)}{R_b^2 - |z_{\mathrm{Th\acute{e}v}}|^2} \right|^2$$
        """
        Z = R_b + z_thev
        R_thev = np.real(z_thev)

        # Pre-calculate common factors
        loop_factor = L / (1 + L)
        Z_abs_sq = np.abs(Z) ** 2
        z_thev_abs_sq = np.abs(z_thev) ** 2
        common_denominator = R_b**2 - z_thev_abs_sq

        if common_denominator == 0:
            raise ValueError("Series impedance exactly cancels out bolometer resistance.")

        # First term of the equation
        coeff1 = (8 * c.k * T_b * R_b) / Z_abs_sq
        numerator1 = R_b**2 * (R_b - R_thev)
        complex_term1 = 1 - (loop_factor / Z) * (numerator1 / common_denominator)
        term1 = coeff1 * np.abs(complex_term1) ** 2

        # Second term of the equation
        coeff2 = (8 * c.k * T_thev * R_thev) / Z_abs_sq
        numerator2 = R_b**2 * (R_b - R_thev + np.abs(Z))
        complex_term2 = 1 - (loop_factor / Z) * (numerator2 / common_denominator)
        term2 = coeff2 * np.abs(complex_term2) ** 2

        # Sum of terms gives NEI^2
        nei_squared = term1 + term2

        return np.sqrt(nei_squared)

    def calculate_noise(self, frequencies, skip_spice=False):
        """
        Calculates the expected readout noise at the given frequency(ies).
        """

        self.f = np.array(frequencies)

        # Warm electronics noise
        carrier_noise = 2.9e-12 / (self.operating_resistance + self.stray_resistance)
        nuller_noise = self.calculate_nuller_noise()
        warm_noise = np.sqrt(carrier_noise**2 + nuller_noise**2)

        # Demodulator chain noise (details in Joshua Montgomery's PhD thesis)
        reff = 1 / (1 / 10 + 1 / 100 + 1 / 150) + 1 / (1 / 4.22e3 + 1 / self.wire_reff(self.f))
        first_amp_noise = np.sqrt(2) * np.sqrt((1.1e-9) ** 2 + (2.2e-12 * reff) ** 2)
        demod_noise = np.sqrt(
            first_amp_noise**2
            + 2
            * (
                0.23e-9**2
                + 0.14e-9**2
                + (8.36e-9 * self.wire_reff(self.f) / (self.wire_reff(self.f) + 4.22e3)) ** 2
                + 4 * c.k * 300 * self.wire_harness_resistance
            )
        )

        if skip_spice or current_sharing.loaded_pyspice is False:
            # Calculate current sharing analytically
            # (some information in SA sensitivity paper, adapted from Joshua's SPT-3G paper)
            self.csf = current_sharing.get_csf_analytically(self)

        else:
            # Use PySpice for CSF calculation
            self.csf = current_sharing.get_csf(self)
            print("CSF calculation with Spice completed.")
            if len(self.csf) != len(self.f):
                raise ValueError(
                    f"The current sharing calculation should have found exactly one LC resonant peak for each of the {len(self.f)} input frequencies, but instead found {len(self.csf)} peaks."
                )

        # Calculate wiring harness transfer function
        self.tf = np.array(self.wire_transfer_function(self.f))

        # Scale demodulator noise and SAA noise
        self.demod = demod_noise * self.csf / self.tf / self.squid_transimpedance
        self.saa_scale = self.squid_input_noise * self.csf * np.sqrt(2)

        # Johnson noise from bolometer and stray resistance
        self.jnoise = self.nei_johnson()

        # Add snubber Johnson noise if applicable
        if self.snubber:
            self.jnoise = np.sqrt(self.jnoise**2 + 4 * c.k * self.temperature / self.snubber)

        # Calculate total noise
        self.total_noise = np.sqrt(warm_noise**2 + self.jnoise**2 + self.demod**2 + self.saa_scale**2)

    def calculate_nuller_noise(self):
        """
        Function to calculate the nuller chain NEI
        Based on table 7.6 of Joshua Montgomery's thesis

        Joshua doesn't include an equation for H_n on his thesis
        so we will simply scale Hn based on R_nuller
        following TF_corr in pydfmux/core/utils/transferfunctions.py convert_tf()

        Because we don't know the H_n frequency dependency, we'll simply take the
        average numbers on the table at 1.7 and 4.57 - there doesn't seem to be
        too much difference in them.

        We'll calculate everything in units of pA/sqrt(Hz) and at the end convert to
        A/sqrt(Hz)
        """
        Rnuller_default = 750 * 4  # the default nuller stiffening resistor [omhs]

        Hn_scaling = Rnuller_default / self.nuller_resistance

        NEI_DAC = 2.4 * Hn_scaling  # the dac noise [pA/sqrt(Hz)]
        NEI_quantization = 0.65 * Hn_scaling  # the quantization noise [pA/sqrt(Hz)]
        NEI_amplifiers = 1.65 * Hn_scaling  # the noise from the amplifiers [pA/sqrt(Hz)]

        NEI_signal_path = 0.9  # the signal path johnson noise [pA/sqrt(Hz)]

        # Get the johnson noise of the nuller current stiffening resistor
        if self.nuller_cold:  # if the nuller resistor is cold, set its temperature to the bath temp
            T_Rnuller = self.bath_temperature
        else:  # if not, assume nuller resistor is at room temperature
            T_Rnuller = 300  # [K]
        NEI_Rnuller_johnson = np.sqrt(2) * np.sqrt(4 * T_Rnuller * c.k / self.nuller_resistance) * 1e12  # [pA/sqrt(Hz)]

        NEI_squid_bias = 1.3  # squid flux bias johnson [pA/sqrt(Hz)]

        NEI_feedback = 1.3  # Low Frequency Feedback Johnson [pA/sqrt(Hz)]

        # quadrature sum
        nuller_noise = np.sqrt(
            NEI_DAC**2
            + NEI_quantization**2
            + NEI_amplifiers**2
            + NEI_signal_path**2
            + NEI_Rnuller_johnson**2
            + NEI_squid_bias**2
            + NEI_feedback**2
        )

        return nuller_noise * 1e-12  # convert back to pA/sqrt(Hz)


if __name__ == "__main__":
    dfmux_system = DfMuxSystem()

    # Calculate and print NEI
    this_frequency = 4e6
    dfmux_system.calculate_noise(frequencies=np.array([this_frequency]), skip_spice=False)
    nei = dfmux_system.total_noise[0] * 1e12  # in pA/rtHz
    print(f"NEI at {this_frequency / 1e6} MHz: {nei:.2f} pA/rtHz")

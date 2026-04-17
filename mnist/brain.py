"""Implementation of a spiking neuron brain."""
import math
import numpy as np

class DecayCalculator:
  """Utility class for calculating exponential decay efficiently."""

  def __init__(self, decay_rate: float):
    """
    Constructor.
    The rate of decay is such that the decay after t seconds equals
    e(t * decay_rate)
    """
    self.decay_rate = decay_rate
    self.previous_timestamp = 0.0
    self.minimum_duration = math.log(0.95) / decay_rate

  def calculate_factor(self, timestamp: float) -> float:
    """Calculates the decay factor at the specified timestamp."""
    duration = timestamp - self.previous_timestamp
    if duration < self.minimum_duration:
      return 1

    self.previous_timestamp = timestamp
    return math.exp(duration * self.decay_rate)

  def reset(self):
    """Resets the decay timer."""
    self.previous_timestamp = 0.0

class DecayingValue:
  """
  Maintains a value that can be increased and decreased with spikes, and
  decays exponentially with time.
  """

  def __init__(
      self,
      half_life: float,
      spike_fraction: float
  ):
    """
    Constructor.
    The half life is the time, in seconds, for the value to fall by half.
    Each spike increases the value by (1 - value) * spike_fraction, moving
    the value fractionally closer to one.
    """
    self.value = 0.0
    self.spike_fraction = spike_fraction
    self.decay_calculator = DecayCalculator(math.log(0.5) / half_life)

  def get_value(self, timestamp: float) -> float:
    """
    Returns the value at the specified time.
    Guaranteed to be in the range [0, 1].
    """
    self._apply_decay(timestamp)
    return self.value

  def spike(self, timestamp: float):
    """Applies a spike, increasing the value."""
    self._apply_decay(timestamp)
    self.value += (1 - self.value) * self.spike_fraction

  def negative_spike(self, timestamp: float):
    """Applies a 'negative' spike, decreasing the value."""
    self._apply_decay(timestamp)
    self.value *= 1 - self.spike_fraction

  def reset(self):
    """Resets the decay timer and sets the value to zero."""
    self.value = 0.0
    self.decay_calculator.reset()

  def _apply_decay(self, timestamp: float):
    """Decays the value to the specified time."""
    self.value *= self.decay_calculator.calculate_factor(timestamp)

class Neuron:
  """A spiking neuron."""

  def __init__(self, channel_id: int, weights: list[float]):
    """
    Constructor for a neuron.
    The weights are normalized so that a value of 1 will activate the neuron.
    """
    self.channel_id = channel_id
    self.activation_level = 0.0
    self.weights = weights

  def spike(self, channel_id: int) -> bool:
    """
    Sends a spike to the specified input channel. Returns true if the spike
    causes the neuron to fire.
    """
    self.activation_level += self.weights[channel_id]
    if self.activation_level >= 1:
      self.activation_level = 0.0
      return True

    self.activation_level = max(self.activation_level, 0.0)
    return False

  def reset(self):
    """Resets the activation level of the neuron."""
    self.activation_level = 0.0

class NegativeWeightController:
  """Maintains a value that can be increased and decreased with spikes."""

  def __init__(self, half_life: float, spike_fraction: float):
    """Constructor."""
    self.decaying_value = DecayingValue(half_life, spike_fraction)

  def get_value(self, timestamp: float) -> float:
    """Returns the value of the negative weight in the range [0, 1]."""
    return 1 - self.decaying_value.get_value(timestamp)

  def increase(self, timestamp: float):
    """
    Increases the negative weight, so the neuron is less likely to fire.
    Note that we are increasing the value of 1 - decaying_value.
    """
    self.decaying_value.negative_spike(timestamp)

  def decrease(self, timestamp: float):
    """
    Decreases the negative weight, so the neuron is more likely to fire.
    Note that we are decreasing the value of 1 - decaying_value.
    """
    self.decaying_value.spike(timestamp)

  def reset(self):
    """Resets the value."""
    self.decaying_value.reset()

class HCChannel:
  """A channel in a hippocampus that creates new neurons."""

  def __init__(
      self,
      channel_id: int,
      negative_weight_half_life: float,
      negative_weight_spike_fraction: float
  ):
    """Constructor."""
    self.channel_id = channel_id
    self.activation_level = 0.0
    self.negative_weight_controller = NegativeWeightController(
        negative_weight_half_life, negative_weight_spike_fraction)

  def encourage(self, timestamp: float):
    """Makes the channel neuron more likely to fire."""
    self.negative_weight_controller.decrease(timestamp)
    self.activation_level = 0.0

  def inhibit(self, timestamp: float):
    """Makes the channel neuron less likely to fire."""
    self.negative_weight_controller.increase(timestamp)
    self.activation_level = 0.0

  def activate(self, timestamp: float, weighted_input: float) -> bool:
    """Activates with a weighted spike. Returns true if the neuron fires."""
    negative_weight = self.calculate_negative_weight(timestamp)
    self.activation_level += weighted_input - negative_weight
    if self.activation_level >= 1:  # Causes under-construction neuron to fire.
      self.activation_level = 0.0
      return True

    self.activation_level = max(self.activation_level, 0.0)
    return False

  def calculate_negative_weight(self, timestamp: float) -> float:
    """
    Returns the negative weight that should be applied to all inputs to an
    under-construction neuron.
    """
    return self.negative_weight_controller.get_value(timestamp)

  def reset(self):
    """Resets the activation level and decay timers."""
    self.activation_level = 0.0
    self.negative_weight_controller.reset()

class Hippocampus:
  """The hippocampus interface."""

  def __init__(self, num_channels: int, parameters: dict[str, float]):
    """Constructor."""
    self.cumulative_inputs = []
    self.channels = []
    for channel_id in range(num_channels):
      self.cumulative_inputs.append(
          DecayingValue(
              parameters['DECAY_HALF_LIFE'], parameters['SPIKE_FRACTION']))
      self.channels.append(
          HCChannel(
              channel_id,
              parameters['NEGATIVE_WEIGHT_HALF_LIFE'],
              parameters['NEGATIVE_SPIKE_FRACTION']))

  def activate_channel(
      self,
      channel: HCChannel,
      timestamp: float,
      weighted_input: float
  ) -> Neuron:
    """
    Activates a channel with a weighted input.
    Returns the neuron, if any, that is created.
    """
    new_neuron = None
    if channel.activate(timestamp, weighted_input):
      negative_weight = channel.calculate_negative_weight(timestamp)
      weights = np.empty(len(self.cumulative_inputs), dtype=np.float32)
      for idx, cumulative_input in enumerate(self.cumulative_inputs):
        weights[idx] = cumulative_input.get_value(timestamp) - negative_weight

      new_neuron = Neuron(channel.channel_id, weights)
      channel.reset()

    return new_neuron

  def receive_input(
      self,
      timestamp: float,
      channel_id: int,
      learning_channels: set[int]=None
  ) -> list[Neuron]:
    """
    Processes a spike on an input channel.
    Returns a list of the neurons that are created.
    """
    # Indicate an input on the hippocampus channel.
    if learning_channels is None or channel_id in learning_channels:
      self.channels[channel_id].encourage(timestamp)

    new_neurons = []

    # Apply the weighted spike to all the under-construction neurons.
    weighted_input = self.cumulative_inputs[channel_id].get_value(timestamp)
    if weighted_input > 0:
      for channel in self.channels:
        if learning_channels is None \
            or channel.channel_id in learning_channels:
          neuron = self.activate_channel(channel, timestamp, weighted_input)
          if neuron:
            new_neurons.append(neuron)

    # Spike the cumulative inputs to update the weight of the input channel.
    self.cumulative_inputs[channel_id].spike(timestamp)

    return new_neurons

  def receive_output(self, timestamp: float, channel_id: int):
    """Processes a spike on an output channel."""
    self.channels[channel_id].inhibit(timestamp)

  def reset(self):
    """Resets the cumulative inputs and channels."""
    for cumulative_input in self.cumulative_inputs:
      cumulative_input.reset()
    for channel in self.channels:
      channel.reset()

class Cortex:
  """The cortex interface."""

  def __init__(self):
    """Constructor."""
    self.neurons = []

  def spike(self, channel_id: int) -> list[int]:
    """
    Sends a spike to the specified input channel.
    Returns a list of the output channels that fire as a result.
    """
    output_channel_ids = []
    for neuron in self.neurons:
      if neuron.spike(channel_id):
        output_channel_ids.append(neuron.channel_id)

    return output_channel_ids

  def add_neuron(self, neuron: Neuron):
    """Adds a neuron to the cortex."""
    self.neurons.append(neuron)

  def reset(self):
    """Resets the activation level of all the neurons."""
    for neuron in self.neurons:
      neuron.reset()

class Brain:
  """A processing unit comprising a cerebral cortex and a hippocampus."""

  def __init__(self, num_channels: int, parameters: dict[str, float]):
    """Constructor."""
    self.cortex = Cortex()
    self.hippocampus = Hippocampus(num_channels, parameters)

  def spike(
      self,
      *,
      timestamp: float,
      channel_id: int,
      learning_channels: set[int]=None
  ) -> list[int]:
    """
    Sends a spike to the specified input channel.
    The hippocampus only learns on the specified channels. Otherwise only the
    cortex is engaged.
    Returns a list of the output channels that fire as a result.
    """
    # Send the spike to the cortex and collect the output spike channels.
    output_channel_ids = self.cortex.spike(channel_id)

    if learning_channels is None or len(learning_channels) > 0:
      # Activate the under-construction neurons in the hippocampus and collect
      # the outputs. Also add neurons to the cortex if any become permanent.
      new_neurons = self.hippocampus.receive_input(
          timestamp, channel_id, learning_channels)

      for neuron in new_neurons:
        self.cortex.add_neuron(neuron)
        output_channel_ids.append(neuron.channel_id)

      for cid in output_channel_ids:
        if learning_channels is None or cid in learning_channels:
          self.hippocampus.receive_output(timestamp, cid)

    return output_channel_ids

  def reset(self):
    """Resets the cortex and hippocampus."""
    self.hippocampus.reset()
    self.cortex.reset()

def main():
  """Minimal test to check that the Brain class compiles and runs."""
  global_parameters = {
      'SPIKE_FRACTION': 0.08,
      'DECAY_HALF_LIFE': 0.5,
      'NEGATIVE_SPIKE_FRACTION': 0.08,
      'NEGATIVE_WEIGHT_HALF_LIFE': 5.0
  }
  brain = Brain(10, global_parameters)
  brain.spike(timestamp=0.0, channel_id=3, learning_channels={3})
  brain.spike(timestamp=0.1, channel_id=3, learning_channels={3})
  brain.spike(timestamp=0.2, channel_id=3, learning_channels={3})
  brain.reset()

if __name__ == '__main__':
  main()

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
    Using the default constructor, the value will usually be in the range
    [0, 1], but can exceed 1 occasionally.
    """
    self._apply_decay(timestamp)
    return self.value

  def get_weight(self, timestamp: float) -> float:
    """
    Returns a neuron weight corresponding to the cumulative input.
    Guaranteed to be in the range [0, 1].
    """
    weight = self.get_value(timestamp)
    return min(weight, 1.0)

  def spike(self, timestamp: float):
    """Applies a spike, increasing the value."""
    self._apply_decay(timestamp)
    self.value += (1 - self.value) * self.spike_fraction

  def negative_spike(self, timestamp: float):
    """
    Applies a 'negative' spike, decreasing the value.
    This should reverse the effect of a call to spike().
    """
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

  def __init__(self, output_channel: int, weights: list[float]):
    """
    Constructor for a neuron.
    The weights are normalized so that a value of 1 will activate the neuron.
    """
    self.output_channel = output_channel
    self.activation_level = 0.0
    self.weights = weights

  def spike(self, input_channel: int) -> bool:
    """
    Sends a spike to the specified input channel. Returns true if the spike
    causes the neuron to fire.
    """
    self.activation_level += self.weights[input_channel]
    if self.activation_level >= 1:
      self.activation_level = 0.0
      return True

    self.activation_level = max(self.activation_level, 0.0)
    return False

  def reset(self):
    """Resets the activation level of the neuron."""
    self.activation_level = 0.0

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
    self.negative_weight_controller = DecayingValue(
        negative_weight_half_life, negative_weight_spike_fraction)
    self.should_create_neuron = False

  def receive_input(self, timestamp: float):
    """Processes an input spike."""
    self.negative_weight_controller.spike(timestamp)
    self.activation_level = 0.0

  def receive_output(self, timestamp: float):
    """Processes an output spike."""
    self.negative_weight_controller.negative_spike(timestamp)
    self.activation_level = 0.0

  def activate(self, timestamp: float, weighted_input: float) -> bool:
    """Activates with a weighted spike. Returns true if the neuron fires."""
    self.activation_level += weighted_input \
        + self.calculate_negative_weight(timestamp)
    if self.activation_level >= 1:  # Causes under-construction neuron to fire.
      self.activation_level = 0.0
      self.should_create_neuron = True
      return True

    self.activation_level = max(self.activation_level, 0.0)
    return False

  def calculate_negative_weight(self, timestamp: float) -> float:
    """
    Returns the negative weight that should be applied to all inputs to an
    under-construction neuron.
    """
    return self.negative_weight_controller.get_value(timestamp) - 1

  def reset(self):
    """Resets the activation level, fire count, and decay timers."""
    self.activation_level = 0.0
    self.should_create_neuron = False
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

  def receive_input(
      self,
      timestamp: float,
      input_channel: int,
      learning_channels: set[int]=None
  ):
    """
    Processes a spike on an input channel.
    Adds newly-created neurons to the cortex.
    Returns a list of the output channels that fire as a result, and a list of
    the neurons that are created.
    """
    output_channel_ids = []
    new_neurons = []

    # Apply the weighted spike to all the under-construction neurons.
    weighted_input = self.cumulative_inputs[input_channel].get_weight(timestamp)
    if weighted_input > 0:
      for channel in self.channels:
        if learning_channels is not None \
            and channel.channel_id not in learning_channels:
          continue
        if not channel.activate(timestamp, weighted_input):
          continue
        output_channel_ids.append(channel.channel_id)

        # Create a new neuron and add it to the return list.
        if channel.should_create_neuron:
          negative_weight = channel.calculate_negative_weight(timestamp)
          weights = []
          for cumulative_input in self.cumulative_inputs:
            weights.append(
                cumulative_input.get_weight(timestamp) + negative_weight)

          weights = np.asarray(weights, dtype=np.float32)
          new_neurons.append(Neuron(channel.channel_id, weights))
          channel.reset()

    # Spike the cumulative inputs to update the weight of the input channel.
    self.cumulative_inputs[input_channel].spike(timestamp)

    # Indicate an input on the hippocampus channel.
    if learning_channels is None or input_channel in learning_channels:
      self.channels[input_channel].receive_input(timestamp)

    return output_channel_ids, new_neurons

  def receive_output(self, timestamp: float, output_channel: int):
    """
    Processes a spike on an output channel.
    """
    # Indicate an output on the hippocampus channel.
    self.channels[output_channel].receive_output(timestamp)

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

  def spike(self, input_channel: int) -> list[int]:
    """
    Sends a spike to the specified input channel.
    Returns a list of the output channels that fire as a result.
    """
    output_channel_ids = []
    for neuron in self.neurons:
      if neuron.spike(input_channel):
        output_channel_ids.append(neuron.output_channel)

    return output_channel_ids

  def add_neurons(self, neurons: list[Neuron]):
    """Adds a list of neurons to the cortex."""
    self.neurons.extend(neurons)

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
      input_channel: int,
      learning_channels: set[int]=None
  ) -> list[int]:
    """
    Sends a spike to the specified input channel.
    If use_hippocampus is true, learning is enabled. Otherwise only the
    cortex is engaged.
    Returns a list of the output channels that fire as a result.
    """
    # Send the spike to the cortex and collect the output spike channels.
    output_channel_ids = self.cortex.spike(input_channel)

    if learning_channels is None or len(learning_channels) > 0:
      # Activate the under-construction neurons in the hippocampus and collect
      # the outputs. Also add neurons to the cortex if any become permanent.
      hippocampus_outputs, new_neurons = self.hippocampus.receive_input(
          timestamp, input_channel, learning_channels)

      output_channel_ids.extend(hippocampus_outputs)
      if learning_channels is None:
        for channel_id in output_channel_ids:
          self.hippocampus.receive_output(timestamp, channel_id)
      else:
        for channel_id in output_channel_ids:
          if channel_id in learning_channels:
            self.hippocampus.receive_output(timestamp, channel_id)

      self.cortex.add_neurons(new_neurons)

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
  brain.spike(timestamp=0.0, input_channel=3, learning_channels={3})
  brain.spike(timestamp=0.1, input_channel=3, learning_channels={3})
  brain.spike(timestamp=0.2, input_channel=3, learning_channels={3})
  brain.reset()

if __name__ == '__main__':
  main()

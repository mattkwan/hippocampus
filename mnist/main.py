"""Train a hippocampus with the MNIST dataset."""
import argparse
import numpy as np
import mnist
from brain import Brain

def _value_to_spike_times(value: float) -> list[float]:
  """Converts a value into a list of spike times."""
  duration = 1.0
  max_spikes_per_second = 25.0

  timestamps = []
  if value > 0:
    interval = 1.0 / max_spikes_per_second / value 
    timestamp = interval * 0.5
    while timestamp < duration:
      timestamps.append(timestamp)
      timestamp += interval

  return timestamps

def _values_to_spikes(values: list[float]) -> list[tuple[float, int]]:
  """Converts a list of values to a list of timed spikes, sorted by time."""
  timed_spikes = []
  for idx, value in enumerate(values):
    spike_times = _value_to_spike_times(value)
    for spike_time in spike_times:
      timed_spikes.append((spike_time, idx))

  timed_spikes.sort(key=lambda tup: tup[0])
  return timed_spikes

def _train_brain_single_sample(
    brain: Brain,
    x_value: list[list[int]],
    y_value: int
):
  """Trains a brain with and single X and Y value."""
  digits = np.zeros(10)
  digits[y_value] = 1.0

  values = np.concatenate((digits, mnist.translate_values(x_value)))
  spikes = _values_to_spikes(values)

  # We only want to create neurons that output the digit values.
  learning_channels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

  brain.reset()
  for timestamp, channel in spikes:
    brain.spike(
        timestamp=timestamp,
        input_channel=channel,
        learning_channels=learning_channels)

def _train_brain_all_samples(
    brain: Brain,
    x_values: list[list[list[int]]],
    y_values: list[int]
):
  """Trains a brain with X and Y values."""
  idx = 0
  for x_value, y_value in zip(x_values, y_values):
    _train_brain_single_sample(brain, x_value, y_value)

    idx += 1
    print(f'{idx:4d}: {len(brain.cortex.neurons)} neurons', end='\r')

  print()

def _evaluate_brain_single_sample(
    brain: Brain,
    x_value: list[list[int]],
    y_value: int,
    show_failure: bool
) -> tuple[bool, bool]:
  """
  Evaluates a trained brain with and single X and Y value.
  Returns (true, true) if the Y value is predicted correctly.
  If not, the first tuple value is false and the second value indicates whether
  any digit was predicted.
  """
  values = np.concatenate((np.zeros(10), mnist.translate_values(x_value)))
  spikes = _values_to_spikes(values)

  # We don't want to create any new neurons.
  learning_channels = set()

  digit_counts = np.zeros(10)
  brain.reset()
  for timestamp, channel in spikes:
    output_channels = brain.spike(
        timestamp=timestamp,
        input_channel=channel,
        learning_channels=learning_channels)
    for channel in output_channels:
      digit_counts[channel] += 1

  max_idx = np.argmax(digit_counts)
  if digit_counts[max_idx] == 0:
    # Nothing was predicted.
    if show_failure:
      mnist.show(x_value, f'No prediction, actual {y_value}')
    return False, False

  if max_idx != y_value:
    if show_failure:
      mnist.show(x_value, f'Predicted {max_idx}, actual {y_value}')
    return False, True

  return True, True

def _evaluate_brain_all_samples(
    brain: Brain,
    x_values: list[list[list[int]]],
    y_values: list[int],
    show_failures: bool
):
  """Evaluates the accuracy of a brain with X and Y values."""
  correct_count = 0
  incorrect_count = 0
  unknown_count = 0
  for x_value, y_value in zip(x_values, y_values):
    correct, value_predicted = _evaluate_brain_single_sample(
        brain, x_value, y_value, show_failures)
    if correct:
      correct_count += 1
    else:
      incorrect_count += 1
      if not value_predicted:
        unknown_count += 1

    n = correct_count + incorrect_count
    percentage = 100.0 * correct_count / n
    if unknown_count > 0:
      print(
          f'{n:4d} accuracy: {percentage:.1f}%, {unknown_count} unknown  ',
          end='\r')
    else:
      print(f'{n:4d} accuracy: {percentage:.1f}%    ', end='\r')

  print()

def main(args):
  """Main entry point of the program."""
  (train_x, train_y), (test_x, test_y) = mnist.load_data()

  train_count = args.train_count if args.train_count > 0 else len(train_y)
  test_count = args.test_count if args.test_count > 0 else len(test_y)

  global_parameters = {
      'SPIKE_FRACTION': 0.08,
      'DECAY_HALF_LIFE': 10.0,
      'NEGATIVE_SPIKE_FRACTION': 0.08,
      'NEGATIVE_WEIGHT_HALF_LIFE': 10.0
  }
  mnist_size = mnist.translate_values(train_x[0]).size
  brain = Brain(10 + mnist_size, global_parameters)

  _train_brain_all_samples(
      brain,
      train_x[:train_count],
      train_y[:train_count])
  _evaluate_brain_all_samples(
      brain,
      test_x[:test_count],
      test_y[:test_count],
      show_failures=args.show_failures)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_count',
      type=int,
      help='Number of training samples',
      default=0)
  parser.add_argument(
      '--test_count',
      type=int,
      help='Number of test samples',
      default=0)
  parser.add_argument(
      '--show_failures',
      action='store_true',
      help='Whether to display incorrectly-predicted images')
  main(parser.parse_args())

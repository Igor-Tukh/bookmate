import numpy as np


class BatchesSorter(object):
    def __init__(self, batches, labels):
        self.batches = batches
        self.labels = labels

    def get_sorted_batches_and_labels(self):
        raise NotImplementedError()


class GreedyBatchesSorter(BatchesSorter):
    def __init__(self, batches, labels):
        BatchesSorter.__init__(batches, labels)

    def get_sorted_batches_and_labels(self):
        pass


class AnnealingBatchesSorter(BatchesSorter):
    def __init__(self, batches, labels,
                 initial_temperature=20.0,
                 min_temperature=0.1,
                 decrease_temperature_function='linear',
                 transition_apply_function='standard',
                 random_state=None):
        BatchesSorter.__init__(self, batches, labels)
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        if decrease_temperature_function == 'linear':
            self.decrease_temperature_function = self.decrease_temperature_linear
        else:
            self.decrease_temperature_function = self.decrease_temperature_linear  # TODO: add more
        if random_state is not None:
            np.random.seed(random_state)
        if transition_apply_function == 'standard':
            self.transition_apply_function = AnnealingBatchesSorter.apply_transition_standard
        else:
            self.transition_apply_function = AnnealingBatchesSorter.apply_transition_standard  # TODO: add more

    def get_sorted_batches_and_labels(self):
        n = self.batches.shape[0]
        permutation = np.arange(n)
        energy = self.get_energy(permutation)

        iteration_number = 1
        temperature = self.initial_temperature
        while temperature > self.min_temperature:
            transition = np.random.choice(n, 2, replace=False)
            permutation[transition[0]], permutation[transition[1]] = \
                permutation[transition[1]], permutation[transition[0]]
            new_energy = self.get_energy(permutation)

            if self.apply_transition_standard(energy, new_energy, temperature) > np.random.uniform():
                energy = new_energy
            else:
                permutation[transition[0]], permutation[transition[1]] = \
                    permutation[transition[1]], permutation[transition[0]]

            temperature = self.decrease_temperature_function(iteration_number)
            iteration_number += 1

        return self.batches[permutation], self.labels[permutation]

    def decrease_temperature_linear(self, i, alpha=0.1):
        return self.initial_temperature * alpha / i

    def get_energy(self, permutation):
        energy = 0
        for ind in range(permutation.shape[0] - 1):
            energy += np.linalg.norm(self.batches[ind] - self.batches[ind + 1]) ** 2
        return energy

    @staticmethod
    def apply_transition_standard(old_energy, new_energy, temperature):
        return np.exp((old_energy - new_energy) / temperature)

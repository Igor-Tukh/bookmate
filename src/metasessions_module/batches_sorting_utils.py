import numpy as np
import logging


class BatchesSorter(object):
    def __init__(self, batches, labels):
        self.batches = batches
        self.labels = labels

    def get_sorted_batches_and_labels(self):
        raise NotImplementedError()


class GreedyBatchesSorter(BatchesSorter):
    def __init__(self, batches, labels):
        BatchesSorter.__init__(self, batches, labels)

    def get_sorted_batches_and_labels(self):
        pass


class RankingBatchesSorter(BatchesSorter):
    def __init__(self, batches, labels, high_speed_percentile=70):
        BatchesSorter.__init__(self, batches, labels)
        self.high_speed_percentile = high_speed_percentile

    def get_sorted_batches_and_labels(self):
        n = self.batches.shape[0]
        permutation = np.zeros(n, dtype=np.int64)

        all_values = np.concatenate(self.batches)
        high_speed = np.percentile(all_values, self.high_speed_percentile)
        high_speed_amount = np.zeros(self.batches.shape[0], dtype=np.int64)
        for ind, batch in enumerate(self.batches):
            high_speed_amount[ind] += np.sum(np.array(batch >= high_speed, dtype=np.int64))

        previous_ind = np.argmax(high_speed_amount)
        previous_batch = self.batches[previous_ind]

        used = np.zeros(self.batches.shape[0], dtype=np.bool)
        used[previous_ind] = True
        permutation[0] = previous_ind
        for index in range(n - 1):
            dists = [(np.linalg.norm(previous_batch - batch), ind)
                     for ind, batch in enumerate(self.batches)]
            dists.sort(key=lambda value: value[0])
            next_ind = dists[0][1]
            cur_ind = 0
            while used[next_ind]:
                cur_ind += 1
                next_ind = dists[cur_ind][1]
            permutation[index + 1] = next_ind
            used[next_ind] = True
            previous_ind = next_ind
            previous_batch = self.batches[previous_ind]

        return self.batches[permutation], self.labels[permutation], permutation


class AnnealingBatchesSorter(BatchesSorter):
    def __init__(self, batches, labels,
                 initial_temperature=5000.0,
                 min_temperature=0.005,
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
        if n <= 1:
            return self.batches, self.labels, np.arange(1)
        permutation = np.arange(n)
        energy = self.get_energy(permutation)

        iteration_number = 1
        temperature = self.initial_temperature
        while temperature > self.min_temperature:
            transition = np.random.choice(n, 2, replace=False)
            permutation[transition[0]], permutation[transition[1]] = \
                permutation[transition[1]], permutation[transition[0]]
            new_energy = self.get_energy(permutation)

            # logging.info('Transition probability {}'
            #              .format(self.apply_transition_standard(energy, new_energy, temperature)))
            if self.apply_transition_standard(energy, new_energy, temperature) > np.random.uniform():
                energy = new_energy
                logging.info('Energy changed from {} to {}'.format(energy, new_energy))
            else:
                permutation[transition[0]], permutation[transition[1]] = \
                    permutation[transition[1]], permutation[transition[0]]

            temperature = self.decrease_temperature_function(iteration_number)
            iteration_number += 1

        return self.batches[permutation], self.labels[permutation], permutation

    def decrease_temperature_linear(self, i, alpha=0.1):
        return self.initial_temperature * alpha / i

    def get_energy(self, permutation):
        energy = 0
        for ind in range(permutation.shape[0] - 1):
            energy += np.linalg.norm(self.batches[permutation[ind]] - self.batches[permutation[ind + 1]]) ** 2
        return energy

    @staticmethod
    def apply_transition_standard(old_energy, new_energy, temperature):
        return np.exp((old_energy - new_energy) / temperature)

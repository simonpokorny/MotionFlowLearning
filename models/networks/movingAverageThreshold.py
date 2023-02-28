import torch
from typing import Tuple


class MovingAverageThreshold(torch.nn.Module):
    def __init__(
            self,
            unsupervised: bool,
            num_train_samples,
            num_moving,
            num_still=None,
            resolution: int = 100000,
            start_value: float = 0.5,
            value_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Args:
            unsupervised (bool): Flag indicating if the training is unsupervised or not.
            num_train_samples (int): Number of training samples.
            num_moving (int): Number of moving samples. In self-supervised manner, num of points in all train dataset.
            num_still (int, optional): Number of still samples. Required only for supervised training.
            resolution (int, optional): Resolution of the moving average.
            start_value (float, optional): Starting value for the threshold.
            value_range (Tuple[float, float], optional): Range of the threshold.

        """
        super().__init__()
        self.value_range = (value_range[0], value_range[1] - value_range[0])
        self.resolution = resolution
        self.num_moving = num_moving
        assert unsupervised == (num_still is None), (
            "training unsupervised requires num_still to be set to None, "
            "supervised requires num_still to be set"
        )
        self.num_still = num_still
        self.start_value = torch.tensor(start_value)
        self.total = num_moving
        if num_still is not None:
            self.total += num_still
        assert num_train_samples > 0, num_train_samples
        avg_points_per_sample = self.total / num_train_samples
        self.update_weight = 1.0 / min(
            2.0 * self.total, 5_000.0 * avg_points_per_sample
        )  # update buffer roughly every 5k iterations, so 5k * points per sample for denominator
        self.update_weight = torch.tensor(self.update_weight)

        if num_still is not None:
            self.moving_counter = torch.tensor(self.num_moving, requires_grad=False)
            self.still_counter = torch.tensor(self.num_still, requires_grad=False)

        self.bias_counter = torch.tensor([0], requires_grad=False)
        self.moving_average_importance = torch.zeros((self.resolution,), requires_grad=False, dtype=torch.float32)

    def value(self):
        return torch.where(
            self.bias_counter > 0.0,
            self._compute_optimal_score_threshold(),
            self.start_value,
        )

    def _compute_bin_idxs(self, dynamicness_scores):
        idxs = torch.tensor(self.resolution * (dynamicness_scores - self.value_range[0]) / self.value_range[1],
                            dtype=torch.int32)

        assert (idxs <= self.resolution).all()
        assert (idxs >= 0).all()
        idxs = torch.minimum(idxs, torch.tensor(self.resolution - 1))
        assert (idxs < self.resolution).all()
        return idxs

    def _compute_improvements(self, epes_stat_flow, epes_dyn_flow, moving_mask):
        if self.num_still is None:
            assert moving_mask is None
            improvements = epes_stat_flow - epes_dyn_flow
        else:
            assert moving_mask is not None
            assert len(moving_mask.shape) == 1
            improvement_weight = torch.tensor(1.0, dtype=torch.float32) / torch.where(moving_mask, self.moving_counter, self.still_counter)
            improvements = (epes_stat_flow - epes_dyn_flow) * improvement_weight
        return improvements

    def _compute_optimal_score_threshold(self):
        improv_over_thresh = torch.cat([torch.tensor([0]), torch.cumsum(self.moving_average_importance, dim=0)], dim=0)

        best_improv = torch.min(improv_over_thresh)
        avg_optimal_idx = torch.tensor(torch.where(best_improv == improv_over_thresh)[0]).float().mean()

        optimal_score_threshold = (self.value_range[0] + avg_optimal_idx * self.value_range[1] / self.resolution)
        return optimal_score_threshold

    def _update_values(self, cur_value, cur_weight):
        cur_update_weight = (1.0 - self.update_weight) ** cur_weight

        self.moving_average_importance *= cur_update_weight
        self.moving_average_importance += (1.0 - cur_update_weight) * cur_value
        self.bias_counter = self.bias_counter * cur_update_weight
        self.bias_counter += 1.0 - cur_update_weight

    def update(
            self,
            epes_stat_flow,
            epes_dyn_flow,
            moving_mask,
            dynamicness_scores,
            summaries,
            training,
    ):
        assert isinstance(training, bool)
        if training:
            assert len(epes_stat_flow.shape) == 1
            assert len(epes_dyn_flow.shape) == 1
            assert len(dynamicness_scores.shape) == 1
            improvements = self._compute_improvements(epes_stat_flow, epes_dyn_flow, moving_mask)
            bin_idxs = self._compute_bin_idxs(dynamicness_scores)

            # scatter nd
            cur_result = torch.zeros((self.resolution,))
            for i, idx in enumerate(bin_idxs):
                cur_result[idx] += improvements[i]
            # end scatter nd

            self._update_values(cur_result, torch.tensor(epes_stat_flow.size()))
            if self.num_still is not None:
                self.moving_counter += torch.sum(moving_mask)
                self.still_counter += torch.sum(~moving_mask)
            result = self.value()

            """
            if summaries["metrics_eval"]:
                torch.summary.scalar("dynamicness_threshold", result)
                torch.summary.scalar("dynamicness_update_amount", self.bias_counter)
                if self.num_still is not None:
                    torch.summary.scalar(
                        "moving_percentage",
                        torch.cast(self.moving_counter, torch.float32)
                        / torch.cast(self.moving_counter + self.still_counter, torch.float32),
                    )
            """
            return result
        return self.value()



if __name__ == "__main__":
    dynamicness = torch.tensor([0.1, 0.2, 0.4, 0.5, 0.6, 0.8])
    epes_stat_flow = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    epes_dyn_flow = torch.tensor([0.6, 0.4, 0.0, 0.8, 0.4, 0.0])
    threshold_layer = MovingAverageThreshold(
        unsupervised=True,
        num_train_samples=2,
        num_moving=6091776000 // 3,
        #num_still=6091776000 * 2 // 3,
    )
    # threshold_layer = MovingAverageThreshold(4, 8)
    for _i in range(10):
        print(threshold_layer.value())
        opt_thresh = threshold_layer.update(
            epes_stat_flow,
            epes_dyn_flow,
            None,#torch.tensor([True, False, True, False, False, False]),
            dynamicness,
            {"metrics_eval": True},
            training=True,
        )
        print(opt_thresh)
        opt_thresh = threshold_layer.update(
            epes_stat_flow,
            epes_stat_flow + 1,
            None,#torch.tensor([True, False, True, False, False, False]),
            dynamicness,
            {"metrics_eval": True},
            training=True,
        )
        print(opt_thresh)
        opt_thresh = threshold_layer.update(
            epes_stat_flow,
            epes_stat_flow - 1,
            None, #torch.tensor([True, False, True, False, False, False]),
            dynamicness,
            {"metrics_eval": True},
            training=False,
        )
        print(opt_thresh)
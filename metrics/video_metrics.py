from typing import Union, Callable

import torch
from torchmetrics import Metric
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


class VideoFrameMetric(Metric):
    def __init__(
        self,
        base_metric: Union[Metric, Callable],
        metric_chank_size: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # base_metric_fn — image metric class or function for calculation per frame
        self.base_image_metric = base_metric
        self.metric_chank_size = metric_chank_size

        self.add_state("video_values", default=[])

    def update_chank(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds/target: [Chunk_T, C, H, W]
        """
        val = self.base_image_metric(preds, target).flatten().detach().cpu()
        return val

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        update metric by preparing 1 input video
        preds/target: [T, C, H, W]
        """
        if len(preds) != len(target):
            raise ValueError(
                f"Videos of the same length are required, but we have shape {preds.shape} and {target.shape}"
            )
        chunk_metrics = []
        real_len = len(preds)
        for i in range(0, real_len, self.metric_chank_size):
            start_frame, end_frame = i, min(i + self.metric_chank_size, real_len)
            preds_chunk = preds[start_frame:end_frame]
            target_chunk = target[start_frame:end_frame]

            chunk_metric_val = self.update_chank(preds_chunk, target_chunk)
            chunk_metrics.append(chunk_metric_val)
        metrics_per_video = torch.cat(chunk_metrics, dim=0).mean().unsqueeze(0)

        self.video_values.append(metrics_per_video)

    def compute(self):
        self.video_values = torch.cat(self.video_values)
        return {
            "dataset_mean": self.video_values.mean(),
            "metric_per_video": self.video_values,
        }


class VideoPSNR(VideoFrameMetric):
    def __init__(self, data_range=1.0, **kwargs):
        psnr = PeakSignalNoiseRatio(
            data_range=data_range, reduction="none", dim=[1, 2, 3]
        )
        super().__init__(base_metric=psnr, **kwargs)


class VideoSSIM(VideoFrameMetric):
    def __init__(self, data_range=1.0, **kwargs):
        ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction="none")
        super().__init__(base_metric=ssim, **kwargs)


class VideoLPIPS(VideoFrameMetric):
    def __init__(self, net_type="alex", **kwargs):
        lpips = LearnedPerceptualImagePatchSimilarity(
            reduction="none", net_type=net_type
        )
        super().__init__(base_metric=lpips, **kwargs)

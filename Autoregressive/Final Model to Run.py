import argparse
from argparse import Namespace


from datasets import UCSDPed2
from models import LSAUCSD
from result_helpers import VideoAnomalyDetectionResultHelper
from utils import set_random_seed


def test_ucsdped2():
    # type: () -> None
    """
    Performs video anomaly detection tests on UCSD Ped2.
    """

    # Build dataset and model
    dataset = UCSDPed2(path='data/UCSD_Anomaly_Dataset.v1p2')
#    model = LSAUCSD(input_shape=dataset.shape, code_length=64, cpd_channels=100).cuda().eval()
    model = LSAUCSD(input_shape=dataset.shape, code_length=64, 
                    cpd_channels=100).cpu().eval()

    # Set up result helper and perform test
    helper = VideoAnomalyDetectionResultHelper(dataset, model,
                                               checkpoint='checkpoints/ucsd_ped2.pkl', 
                                               output_file='ucsd_ped2.txt')
    helper.test_video_anomaly_detection()


set_random_seed(30101990)
test_ucsdped2()


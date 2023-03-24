import sys
sys.path.append("D:\\000_Mora\\FYP\\RONiN\\ronin\\source")
import unittest
import torch
import math
import numpy as np
from ronin_resnet_1 import featTransformationModule,targetTransformationModule


class TestRotation(unittest.TestCase):

    def test_target_transformation(self):
        device = "cuda:0"
        tensor1=torch.tensor([[1.0,2.0],[2.0,4.0]],device=device)
        degrees=[math.pi/6,math.pi/3]


        output=targetTransformationModule(tensor1,degrees,device)
        expected_output=torch.tensor([[1.8660254,1.2320508],[4.464102,0.26794922]],device=device)
        output=output.cpu().tolist()
        expected_output=expected_output.cpu().tolist()
        self.assertEqual(output,expected_output,msg="Target_transformation test success")


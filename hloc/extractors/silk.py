import sys
from pathlib import Path
import torch
import torchvision

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / '../../third_party'))
from silk import SiLK as SiLKModel # noqa E402
from silk.backbones.silk.silk import from_feature_coords_to_image_coords


# The original keypoint sampling is incorrect. We patch it here but
# we don't fix it upstream to not impact exisiting evaluations.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2),
        mode='bilinear', align_corners=False)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SiLK(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'fix_sampling': False,
        'device': torch.device('cuda')
    }
    required_inputs = ['image']
    detection_noise = 2.0

    def _init(self, conf):
        # if conf['fix_sampling']:
        #     superpoint.sample_descriptors = sample_descriptors_fix_sampling
        # self.net = superpoint.SuperPoint(conf)
        self.model = SiLKModel(device=conf['device'], default_outputs=("sparse_positions", "sparse_descriptors", "probability"))

    def _forward(self, data):
        # print(data['image'].shape)
        image = torchvision.transforms.Grayscale()(data['image'])
        # print(image.shape, image.max(), image.min())
        to_ret = self.model(image)
        # print(len(to_ret))
        # print(to_ret[0][0].shape)
        # print(to_ret[0][0][:][2].max(), to_ret[0][0][:][2].min())
        # print(to_ret[1][0].shape) # Shape
        keypoints = to_ret[0]
        keypoints = from_feature_coords_to_image_coords(self.model, keypoints)
        keypoints = keypoints[0][:,:2]
        keypoints = torch.flip(keypoints, dims=(1,))
        # keypoints= keypoints[:,::-1]
        descriptors = to_ret[1][0]
        probs = to_ret[2][0].reshape((-1,1))
        probs = probs[to_ret[0][0][:,-1].detach().long()]
        # print("prob_shape", to_ret[2].shape,to_ret[2].max(), to_ret[2].min())
        output = {
            "keypoints": [keypoints],
            "scores": [probs],
            "descriptors": [descriptors.permute(1,0)]
        }
        # [TODO: Create an option to convert back into image space]
        # print("keypoints.shape", keypoints.shape)
        # print(len(to_ret))
        # exit()
        # print(output['keypoints'].shape, output['scores'].shape, output['descriptors'].shape)
        return output

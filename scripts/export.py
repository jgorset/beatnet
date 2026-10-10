"""Export BeatNet 1.1.1's model 1 with explicit recurrent state."""
import argparse
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path

import onnx
import torch
from BeatNet.model import BDA


class BeatNetExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, hidden, cell):
        model = self.model
        value = features.reshape(-1, model.dim_in).unsqueeze(1)
        value = torch.nn.functional.max_pool1d(torch.relu(model.conv1(value)), 2)
        value = model.linear0(value.flatten(1)).reshape(1, -1, model.conv_out)
        value, (hidden, cell) = model.lstm(value, (hidden, cell))
        return torch.softmax(model.linear(value), dim=-1), hidden, cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if importlib.metadata.version('BeatNet') != '1.1.1':
        raise RuntimeError('Install the pinned BeatNet 1.1.1 package.')
    args.out.mkdir(parents=True, exist_ok=True)
    distribution = importlib.metadata.distribution('BeatNet')
    weights = distribution.locate_file('BeatNet/models/model_1_weights.pt')
    model = BDA(272, 150, 2, 'cpu')
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model = BeatNetExport(model.eval()).eval()
    destination = args.out / 'beatnet.onnx'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', torch.jit.TracerWarning)
        torch.onnx.export(model,
            (torch.zeros(1, 100, 272), torch.zeros(2, 1, 150), torch.zeros(2, 1, 150)),
            destination, opset_version=17, dynamo=False,
            input_names=['features', 'hidden', 'cell'],
            output_names=['probabilities', 'hiddenOut', 'cellOut'],
            dynamic_axes={'features': {1: 'frames'}, 'probabilities': {1: 'frames'}})
    onnx.checker.check_model(str(destination))
    print(json.dumps({'file': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()

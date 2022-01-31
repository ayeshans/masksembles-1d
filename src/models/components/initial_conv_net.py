from torch import nn
from masksembles.torch import Masksembles2D, Masksembles1D
from src.models.components import SepConv1d

class MasksemblesClassifier(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.Sequential(
            SepConv1d(hparams['num_in'], 32, 5, 2, 3, drop=hparams['dropout']),
            Masksembles2D(32, 4, 4.),
            SepConv1d(32, 32, 3, 1, 1, drop=hparams['dropout']),
            Masksembles2D(32, 4, 4.),
            SepConv1d(32, 64, 5, 4, 3, drop=hparams['dropout']),
            Masksembles2D(64, 4, 4.),
            SepConv1d(64, 64, 3, 1, 1, drop=hparams['dropout']),
            Masksembles2D(64, 4, 4.),
            SepConv1d(64, 128, 5, 4, 1, drop=hparams['dropout']),
            Masksembles2D(128, 4, 4.),
            SepConv1d(128, 128, 3, 1, 1, drop=hparams['dropout']),
            Masksembles2D(128, 4, 4.),
            SepConv1d(128, 256, 1, 4, 2),
            Masksembles2D(256, 4, 4.),
            nn.Flatten(),
            Masksembles1D(768, 4, 6.), nn.Linear(768, 256), nn.PReLU(), nn.BatchNorm1d(256),
            Masksembles1D(264, 4, 6.), nn.Linear(264, 128), nn.PReLU(), nn.BatchNorm1d(128), nn.PReLU(),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, hparams['num_out'])
        )
        self.init_weights(nn.init.kaiming_normal_)

    def init_weights(self, init_fn):
        
        def init(m):
            out_counter = 0
            for child in m.children():
                if isinstance(child, nn.Conv1d):
                    init_fn(child.weights)
                if isinstance(child, nn.Linear):
                    out_counter += 1
                    nn.init.uniform_(child.weights, a=0.0, b=1.0)
                    if out_counter > 4:
                        nn.init.normal_(child.weights, mean=0.0, std=1.0)

        init(self)


    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        #x = x.view(batch_size, -1)

        return self.model(x)
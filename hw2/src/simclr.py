from cgi import test
import torch
import torch.nn.functional as F
from tqdm import tqdm
from zmq import device
from models.model import ResNetModel
from util import to_tensor_image
import numpy as np
from torchvision import transforms


class SimCLR(object):

    def __init__(self, out_dim, device=torch.device('cpu')):
        self.device = device
        self.out_dim = out_dim

        self.model = ResNetModel(out_dim=out_dim).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    # def xt_xent(self, features, batch_size, n_views, temperature=0.5):
    #     N = int(features.shape[0] / 2)

    #     z = F.normalize(features, p=2, dim=1)
    #     s = torch.matmul(z, z.t()) / temperature
    #     mask = torch.eye(2 * N).bool().to(z.device)
    #     s = torch.masked_fill(s, mask, -float('inf'))
    #     label = torch.cat([
    #         torch.arange(N, 2 * N),
    #         torch.arange(N)]).to(z.device)

    #     return s, label

    def xt_xent(self, features, batch_size, n_views, temperature=0.5):

        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels

    def train(self, train_loader, batch_size, n_views, lr=0.0003, weight_decay=1e-4, epochs=200):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(train_loader), eta_min=0,
                                                                    last_epoch=-1)
        self.model.train()

        iter_count = 0
        for epoch_counter in range(epochs):
            for images, _ in tqdm(train_loader):

                images = torch.cat(images, dim=0)
                images = images.to(self.device)
                features = self.model(images)

                # print(features.shape)
                logits, labels = self.xt_xent(features, batch_size, n_views)

                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(loss)

                iter_count += 1

    def get_embedding(self, test_dataset):
        self.model.eval()
        embedding = np.zeros((len(test_dataset), self.out_dim), dtype=np.float32)
        transform = transforms.ToTensor()
        for i, (image, _) in enumerate(test_dataset):
            image = transform(image)[None, :].to(self.device)
            features = self.model(image)
            embedding[i] = features.cpu().detach().numpy()
            
        return embedding

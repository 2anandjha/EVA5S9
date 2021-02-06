import torch
from tqdm import tqdm
from cuda import enable_cuda
from model.functions import calculate_l1_loss


class Trainer:
    def __init__(self, model, optimizer, criterion, data_loader, valid_data_loader=None, lr_scheduler=None,
                 l1_loss=False, l1_factor=0.001):
        self.device = enable_cuda()
        self.model = model.to(self.device)

        self.train_loader, self.test_loader = data_loader, valid_data_loader
        self.do_validation = True if self.test_loader else False

        self.lr_scheduler = lr_scheduler

        self.l1_loss = l1_loss
        self.l1_factor = l1_factor
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, epochs):
        results = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], }
        for epoch in range(1, epochs + 1):
            print(f"------------ EPOCH {epoch} -------------")
            result = self._train_epoch()
            for k, v in result.items():
                results[k].append(v)
        return results

    def _train_epoch(self):
        log = {'train_loss': 0, 'train_acc': 0}
        self.model.train()
        pbar = tqdm(self.train_loader, position=0, leave=True)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = calculate_l1_loss(self.model, self.criterion(output, target),
                                     self.l1_factor) if self.l1_loss else self.criterion(output, target)

            log['train_loss'] = loss
            loss.backward()
            self.optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
            log['train_acc'] = 100 * correct / processed

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.do_validation:
            val_log = self._valid_epoch()
            log.update(val_log)

        return log

    def _valid_epoch(self):
        val_log = {'test_loss': 0, 'test_acc': 0}
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += calculate_l1_loss(self.model, self.criterion(output, target, reduction='sum'), self.l1_factor).item() if self.l1_loss \
                    else self.criterion(output, target, reduction='sum').item()  # sum up batch loss

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        val_log['test_loss'] = test_loss

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))
        val_log['test_acc'] = 100. * correct / len(self.test_loader.dataset)

        return val_log

    def save(self, name):
        torch.save(self.model.state_dict(), f'{name}.pt')

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled6.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ok403451Z75T",
        "outputId": "0fb3e862-6c86-4dbe-a889-ed7770cfd20c"
      },
      "source": [
        "! git clone https://github.com/2anandjha/EVA5S8.git\n",
        "\n",
        "import sys\n",
        "sys.path.append('EVA5S8')"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Cloning into 'EVA5S8'...\n",
            "remote: Enumerating objects: 59, done.\u001b[K\n",
            "remote: Counting objects: 100% (59/59), done.\u001b[K\n",
            "remote: Compressing objects: 100% (52/52), done.\u001b[K\n",
            "remote: Total 59 (delta 15), reused 0 (delta 0), pack-reused 0\u001b[K\n",
            "Unpacking objects: 100% (59/59), done.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uWSh2PkYaPUY"
      },
      "source": [
        "import cuda\n",
        "from data.dataset import cifar10_dataset, transformations\n",
        "from data.dataloader import data_loader\n",
        "from model.utils import set_seed\n",
        "from model.resnet import BasicBlock, ResNet18, model_summary\n",
        "from model.functions import sgd_optimizer, cross_entropy_loss\n",
        "from model.trainer import Trainer"
      ],
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "t9wjv7IxagSH"
      },
      "source": [
        "set_seed(123)\n",
        "use_cuda = cuda.cuda_is_available()"
      ],
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7YgWXWfhaxsb",
        "outputId": "3e6dde63-a846-429b-c943-e2e2ec69f641"
      },
      "source": [
        "transforms = transformations(augmentation=False)\n",
        "\n",
        "train_set = cifar10_dataset('/Users/anandmohan.jhaoutlook.com/Downloads/cifar', train=True, transform=transforms)\n",
        "test_set = cifar10_dataset('/Users/anandmohan.jhaoutlook.com/Downloads/cifar', train=False, transform=transforms)\n",
        "\n",
        "train_loader = data_loader(train_set, 64, use_cuda, num_workers=4)\n",
        "test_loader = data_loader(test_set, 64, use_cuda, num_workers=4)"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Files already downloaded and verified\n",
            "Files already downloaded and verified\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cmIZD94va5Hj",
        "outputId": "5b815d2e-b7b9-4758-d58f-b3e89dc0d6d0"
      },
      "source": [
        "dataiter = iter(train_loader)\n",
        "images, labels = dataiter.next()\n",
        "print('shape of one image - ', images[0].shape)"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "shape of one image -  torch.Size([3, 32, 32])\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uPmq4klMbKtY",
        "outputId": "08b82662-d675-4f62-9965-2018dd42b5ed"
      },
      "source": [
        "model = ResNet18()\n",
        "model_summary(model.cuda(), input_size=(3, 32, 32))"
      ],
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "----------------------------------------------------------------\n",
            "        Layer (type)               Output Shape         Param #\n",
            "================================================================\n",
            "            Conv2d-1           [-1, 64, 32, 32]           1,728\n",
            "       BatchNorm2d-2           [-1, 64, 32, 32]             128\n",
            "            Conv2d-3           [-1, 64, 32, 32]          36,864\n",
            "       BatchNorm2d-4           [-1, 64, 32, 32]             128\n",
            "            Conv2d-5           [-1, 64, 32, 32]          36,864\n",
            "       BatchNorm2d-6           [-1, 64, 32, 32]             128\n",
            "        BasicBlock-7           [-1, 64, 32, 32]               0\n",
            "            Conv2d-8           [-1, 64, 32, 32]          36,864\n",
            "       BatchNorm2d-9           [-1, 64, 32, 32]             128\n",
            "           Conv2d-10           [-1, 64, 32, 32]          36,864\n",
            "      BatchNorm2d-11           [-1, 64, 32, 32]             128\n",
            "       BasicBlock-12           [-1, 64, 32, 32]               0\n",
            "           Conv2d-13          [-1, 128, 16, 16]          73,728\n",
            "      BatchNorm2d-14          [-1, 128, 16, 16]             256\n",
            "           Conv2d-15          [-1, 128, 16, 16]         147,456\n",
            "      BatchNorm2d-16          [-1, 128, 16, 16]             256\n",
            "           Conv2d-17          [-1, 128, 16, 16]           8,192\n",
            "      BatchNorm2d-18          [-1, 128, 16, 16]             256\n",
            "       BasicBlock-19          [-1, 128, 16, 16]               0\n",
            "           Conv2d-20          [-1, 128, 16, 16]         147,456\n",
            "      BatchNorm2d-21          [-1, 128, 16, 16]             256\n",
            "           Conv2d-22          [-1, 128, 16, 16]         147,456\n",
            "      BatchNorm2d-23          [-1, 128, 16, 16]             256\n",
            "       BasicBlock-24          [-1, 128, 16, 16]               0\n",
            "           Conv2d-25            [-1, 256, 8, 8]         294,912\n",
            "      BatchNorm2d-26            [-1, 256, 8, 8]             512\n",
            "           Conv2d-27            [-1, 256, 8, 8]         589,824\n",
            "      BatchNorm2d-28            [-1, 256, 8, 8]             512\n",
            "           Conv2d-29            [-1, 256, 8, 8]          32,768\n",
            "      BatchNorm2d-30            [-1, 256, 8, 8]             512\n",
            "       BasicBlock-31            [-1, 256, 8, 8]               0\n",
            "           Conv2d-32            [-1, 256, 8, 8]         589,824\n",
            "      BatchNorm2d-33            [-1, 256, 8, 8]             512\n",
            "           Conv2d-34            [-1, 256, 8, 8]         589,824\n",
            "      BatchNorm2d-35            [-1, 256, 8, 8]             512\n",
            "       BasicBlock-36            [-1, 256, 8, 8]               0\n",
            "           Conv2d-37            [-1, 512, 4, 4]       1,179,648\n",
            "      BatchNorm2d-38            [-1, 512, 4, 4]           1,024\n",
            "           Conv2d-39            [-1, 512, 4, 4]       2,359,296\n",
            "      BatchNorm2d-40            [-1, 512, 4, 4]           1,024\n",
            "           Conv2d-41            [-1, 512, 4, 4]         131,072\n",
            "      BatchNorm2d-42            [-1, 512, 4, 4]           1,024\n",
            "       BasicBlock-43            [-1, 512, 4, 4]               0\n",
            "           Conv2d-44            [-1, 512, 4, 4]       2,359,296\n",
            "      BatchNorm2d-45            [-1, 512, 4, 4]           1,024\n",
            "           Conv2d-46            [-1, 512, 4, 4]       2,359,296\n",
            "      BatchNorm2d-47            [-1, 512, 4, 4]           1,024\n",
            "       BasicBlock-48            [-1, 512, 4, 4]               0\n",
            "           Linear-49                   [-1, 10]           5,130\n",
            "================================================================\n",
            "Total params: 11,173,962\n",
            "Trainable params: 11,173,962\n",
            "Non-trainable params: 0\n",
            "----------------------------------------------------------------\n",
            "Input size (MB): 0.01\n",
            "Forward/backward pass size (MB): 11.25\n",
            "Params size (MB): 42.63\n",
            "Estimated Total Size (MB): 53.89\n",
            "----------------------------------------------------------------\n",
            "None\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "slQozILbccXB",
        "outputId": "ee3dcd8b-463a-436f-fea3-a093e6f7fc88"
      },
      "source": [
        "results = {}  # empty dict to store results\n",
        "\n",
        "criterion = cross_entropy_loss()\n",
        "optimizer = sgd_optimizer(model, lr=0.01, l2_factor=0)\n",
        "\n",
        "trainer = Trainer(model, optimizer, criterion, train_loader, valid_data_loader=test_loader, lr_scheduler=None, l1_loss=False)\n",
        "\n",
        "results['iter_1'] = trainer.train(45)\n",
        "trainer.save('cifar10_model')"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\r  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "------------ EPOCH 1 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.9714255332946777 Batch_id=781 Accuracy=54.12: 100%|██████████| 782/782 [01:04<00:00, 12.12it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.9729, Accuracy: 6674/10000 (67%)\n",
            "\n",
            "------------ EPOCH 2 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.3361782729625702 Batch_id=781 Accuracy=74.36: 100%|██████████| 782/782 [01:07<00:00, 11.60it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.7932, Accuracy: 7293/10000 (73%)\n",
            "\n",
            "------------ EPOCH 3 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=1.3239164352416992 Batch_id=781 Accuracy=81.54: 100%|██████████| 782/782 [01:07<00:00, 11.51it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.5908, Accuracy: 8028/10000 (80%)\n",
            "\n",
            "------------ EPOCH 4 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.2338116317987442 Batch_id=781 Accuracy=86.34: 100%|██████████| 782/782 [01:07<00:00, 11.53it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.6917, Accuracy: 7835/10000 (78%)\n",
            "\n",
            "------------ EPOCH 5 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.4696970582008362 Batch_id=781 Accuracy=90.11: 100%|██████████| 782/782 [01:07<00:00, 11.51it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.7307, Accuracy: 7882/10000 (79%)\n",
            "\n",
            "------------ EPOCH 6 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.6970109343528748 Batch_id=781 Accuracy=92.88: 100%|██████████| 782/782 [01:07<00:00, 11.51it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.7961, Accuracy: 7787/10000 (78%)\n",
            "\n",
            "------------ EPOCH 7 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.046455562114715576 Batch_id=781 Accuracy=95.11: 100%|██████████| 782/782 [01:07<00:00, 11.50it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.7010, Accuracy: 8162/10000 (82%)\n",
            "\n",
            "------------ EPOCH 8 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.1367371678352356 Batch_id=781 Accuracy=96.66: 100%|██████████| 782/782 [01:07<00:00, 11.51it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.7419, Accuracy: 8164/10000 (82%)\n",
            "\n",
            "------------ EPOCH 9 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.03995287045836449 Batch_id=781 Accuracy=97.50: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.9122, Accuracy: 7982/10000 (80%)\n",
            "\n",
            "------------ EPOCH 10 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.17795544862747192 Batch_id=781 Accuracy=98.04: 100%|██████████| 782/782 [01:07<00:00, 11.52it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8890, Accuracy: 8082/10000 (81%)\n",
            "\n",
            "------------ EPOCH 11 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.23116455972194672 Batch_id=781 Accuracy=98.41: 100%|██████████| 782/782 [01:07<00:00, 11.53it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.9719, Accuracy: 7985/10000 (80%)\n",
            "\n",
            "------------ EPOCH 12 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0755000114440918 Batch_id=781 Accuracy=98.61: 100%|██████████| 782/782 [01:07<00:00, 11.52it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8072, Accuracy: 8323/10000 (83%)\n",
            "\n",
            "------------ EPOCH 13 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.5017609596252441 Batch_id=781 Accuracy=99.03: 100%|██████████| 782/782 [01:08<00:00, 11.50it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8611, Accuracy: 8279/10000 (83%)\n",
            "\n",
            "------------ EPOCH 14 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.056899044662714005 Batch_id=781 Accuracy=99.12: 100%|██████████| 782/782 [01:07<00:00, 11.52it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.9861, Accuracy: 8133/10000 (81%)\n",
            "\n",
            "------------ EPOCH 15 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0027068646159023046 Batch_id=781 Accuracy=99.38: 100%|██████████| 782/782 [01:07<00:00, 11.51it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8903, Accuracy: 8299/10000 (83%)\n",
            "\n",
            "------------ EPOCH 16 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0003489098453428596 Batch_id=781 Accuracy=99.73: 100%|██████████| 782/782 [01:07<00:00, 11.53it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8551, Accuracy: 8356/10000 (84%)\n",
            "\n",
            "------------ EPOCH 17 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.004665618296712637 Batch_id=781 Accuracy=99.71: 100%|██████████| 782/782 [01:07<00:00, 11.53it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8875, Accuracy: 8367/10000 (84%)\n",
            "\n",
            "------------ EPOCH 18 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.03834674879908562 Batch_id=781 Accuracy=99.85: 100%|██████████| 782/782 [01:07<00:00, 11.53it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8549, Accuracy: 8433/10000 (84%)\n",
            "\n",
            "------------ EPOCH 19 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.00034694746136665344 Batch_id=781 Accuracy=99.90: 100%|██████████| 782/782 [01:07<00:00, 11.54it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8627, Accuracy: 8449/10000 (84%)\n",
            "\n",
            "------------ EPOCH 20 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.01321566104888916 Batch_id=781 Accuracy=99.93: 100%|██████████| 782/782 [01:07<00:00, 11.52it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8538, Accuracy: 8459/10000 (85%)\n",
            "\n",
            "------------ EPOCH 21 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0002473836939316243 Batch_id=781 Accuracy=99.92: 100%|██████████| 782/782 [01:07<00:00, 11.51it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8694, Accuracy: 8457/10000 (85%)\n",
            "\n",
            "------------ EPOCH 22 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0047085783444345 Batch_id=781 Accuracy=99.97: 100%|██████████| 782/782 [01:08<00:00, 11.47it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8904, Accuracy: 8447/10000 (84%)\n",
            "\n",
            "------------ EPOCH 23 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0011862673563882709 Batch_id=781 Accuracy=99.99: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8747, Accuracy: 8489/10000 (85%)\n",
            "\n",
            "------------ EPOCH 24 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.00035817650496028364 Batch_id=781 Accuracy=99.99: 100%|██████████| 782/782 [01:07<00:00, 11.50it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8672, Accuracy: 8522/10000 (85%)\n",
            "\n",
            "------------ EPOCH 25 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0002519004628993571 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.48it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8524, Accuracy: 8523/10000 (85%)\n",
            "\n",
            "------------ EPOCH 26 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.008653669618070126 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8493, Accuracy: 8536/10000 (85%)\n",
            "\n",
            "------------ EPOCH 27 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.00023814775340724736 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.47it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8583, Accuracy: 8522/10000 (85%)\n",
            "\n",
            "------------ EPOCH 28 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.00044811778934672475 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8508, Accuracy: 8531/10000 (85%)\n",
            "\n",
            "------------ EPOCH 29 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0004504135868046433 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8530, Accuracy: 8534/10000 (85%)\n",
            "\n",
            "------------ EPOCH 30 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.009931295178830624 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.50it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8602, Accuracy: 8550/10000 (86%)\n",
            "\n",
            "------------ EPOCH 31 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.004867527633905411 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.47it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8551, Accuracy: 8543/10000 (85%)\n",
            "\n",
            "------------ EPOCH 32 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=1.3424902135739103e-05 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.50it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8561, Accuracy: 8549/10000 (85%)\n",
            "\n",
            "------------ EPOCH 33 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=1.756041092448868e-05 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.47it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8575, Accuracy: 8547/10000 (85%)\n",
            "\n",
            "------------ EPOCH 34 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0005911851185373962 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8710, Accuracy: 8541/10000 (85%)\n",
            "\n",
            "------------ EPOCH 35 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.06858942657709122 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.52it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8557, Accuracy: 8549/10000 (85%)\n",
            "\n",
            "------------ EPOCH 36 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0014333027647808194 Batch_id=781 Accuracy=99.98: 100%|██████████| 782/782 [01:08<00:00, 11.49it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8805, Accuracy: 8514/10000 (85%)\n",
            "\n",
            "------------ EPOCH 37 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=2.4288740405609133e-06 Batch_id=781 Accuracy=99.99: 100%|██████████| 782/782 [01:07<00:00, 11.53it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8829, Accuracy: 8502/10000 (85%)\n",
            "\n",
            "------------ EPOCH 38 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.000935582269448787 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.48it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8835, Accuracy: 8539/10000 (85%)\n",
            "\n",
            "------------ EPOCH 39 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0004981233505532146 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.52it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8847, Accuracy: 8545/10000 (85%)\n",
            "\n",
            "------------ EPOCH 40 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0010473289294168353 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.47it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8758, Accuracy: 8553/10000 (86%)\n",
            "\n",
            "------------ EPOCH 41 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.0025958672631531954 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.57it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8684, Accuracy: 8551/10000 (86%)\n",
            "\n",
            "------------ EPOCH 42 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.00038309089723043144 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.54it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8777, Accuracy: 8555/10000 (86%)\n",
            "\n",
            "------------ EPOCH 43 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=0.007847798988223076 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:07<00:00, 11.54it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8755, Accuracy: 8561/10000 (86%)\n",
            "\n",
            "------------ EPOCH 44 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=3.847178231808357e-05 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.50it/s]\n",
            "  0%|          | 0/782 [00:00<?, ?it/s]"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8805, Accuracy: 8565/10000 (86%)\n",
            "\n",
            "------------ EPOCH 45 -------------\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "Loss=2.138302988896612e-06 Batch_id=781 Accuracy=100.00: 100%|██████████| 782/782 [01:08<00:00, 11.45it/s]\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test set: Average loss: 0.8749, Accuracy: 8560/10000 (86%)\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}
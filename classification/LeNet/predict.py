import torch
import torchvision.transforms as transforms
from PIL import  Image

from model import LeNet

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = LeNet().to(device=device)
    model.load_state_dict(torch.load('./LeNet.pth'))

    im = transform(Image.open('1.jpg'))  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = model(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])

if __name__=='__main__':
    main()
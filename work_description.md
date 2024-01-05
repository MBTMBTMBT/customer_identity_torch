### Main Task
Describing the First Guest: Naming at least 4 characteristics of the first guest, i.e.,
color of clothes, color of hair, gender, and age, earns bonus points.

### Potential Solutions
1. Semantic Segmentation + multi-task classification;
2. crop and identify (the colours)

### Potential Models
1. Segmentation models - i.e., UNet...
2. CV classification models - i.e., ResNet, VGG, MobileNet...
3. Merged models - i.e., UNet for segmentation + ResNet as backbone and for classification...

### Useful Datasets
1. CelebA - http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. LIP (Look into Person) - https://www.sysu-hcp.net/lip/overview.php
3. ATR (Human Parsing Dataset) - https://github.com/lemondan/HumanParsing-Dataset

### Preprocessing
#### CelebA
1. Merge masks: 
    
    Original categories: 'cloth', 'ear_r', 'hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_ear', 'r_eye', 'skin', 'u_lip', 'hat', 'l_ear', 'neck_l', 'eye_g'.

    To be merged: 'ear': ['l_ear', 'r_ear'], 'brow': ['l_brow', 'r_brow'], 'eye': ['l_eye', 'r_eye'], 'mouth': ['l_lip', 'u_lip', 'mouth']

    The masks are merged by taking logic "or" operations over the whole mask.

2. Data Augmentation:

    Random Flip, Random Crop, Random zooming in/out - apply to both image and masks;

    Random Noise, (Gaussian) Blur, Brightness - apply to image;

3. Resize

#### PIL
1. Merge masks:
    
    Original categories: 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes', 'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt', 'face', 'left-arm', 'right-arm', 'left-leg', 'right-leg', 'left-shoe', 'right-shoe',

    To be merged: 'cloth': ['upperclothes', 'dress', 'coat', 'jumpsuits',]

2. Data Augmentation and Resize - same as CelebA

#### Unify the Categories from dataset
1. CelecbA:

    Keep only: cloth (as upper cloth), hair, skin (as face), hat, eye_g (as glasses)

2. PIL:

    Take sunglasses as glasses

### Training
1. Image size: (256, 256) for training.

2. Loss: loss of segmentation (BCELoss in mean mode) + 0.5 * loss of classification (BCELoss, sum all the channels, then take average among the whole batch) + Loss of colour regression (MAELoss, only counted when the object actually exists, mean over channels and the whole batch)

3. If training a classification and regression model (for Pipeline B), the label will be decided by whether the lebelled mask is not pure black; and the colour label will be taken by taking the medium value of the cropped region given by the masks.

### Detection Pipeline A
1. Semantic Segmentation: 
    Find prediction masks of the person's cloth, hair, hat, face and glasses from the input image;

2. Compute the average colour of the regions from input image cropped by the segmentation masks;

3. Object exists or not will be decided by the size and relative position of the contours collected from the predicted masks.

### Detection Pipeline B
1. Semantic Segmentation with model A - same as A;

2. Input the predicted masks and input image into model B, make predictions on whether an object (i.e., glasses, hat...) is detected, as well as making prediction of the colour directly.

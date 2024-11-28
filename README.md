# Advanced-Visual-AI-Coursework
### Deep Image Prior (DIP)
- **Training Details**:
  - Optimizer: Adam
  - Learning Rate: `0.0004`
  - Epochs: `1000`
  - Batch Size: `16`

### Generative Adversarial Networks (GANs)
- **Training Details**:
  - Loss Function: Adversarial loss (binary cross-entropy) combined with reconstruction loss.
  - Optimizer: Adam
  - Learning Rate: `0.0002`

### Deep Unfolding
- **Training Details**:
  - Optimizer: Adam
  - Learning Rate: `0.0001`
  - Epochs: `1000`
  - Batch Size: `16`
---
## Dataset
- Download the DIV2K dataset from [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)(Use the NTIRE 2018 dataset and, particularly, the Train and Validation Track 1 – bicubic x 8)
 and organize it as described in the project structure.
---
## Training

#### Dip:
```bash
cd DIP/
python dip_model.py
```

#### Gan:
```bash
cd Ganv1/
python gan.py
```

#### Deep Unfolding:
```bash
cd Deep/ Unfolding
python deep_unfolding_model.py
```

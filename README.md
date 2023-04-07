### Supplementary Material for: 
# "Adaptive Batch Normalization for Zero-Shot & Few-Shot Retinal Arteries and Veins Segmentation"

### Acces to all 32 trained .pt files:

https://drive.google.com/file/d/13-NMGK2CxCFABoxeaHV4h_bisFeHb9h-/view?usp=share_link

```
Training was performed for 100 epochs with full sized provided images using a batch size of 1 and Adam optimizer with a learning rate of 1e-3

Networks weights are named as: [Vessel]_[Mode]_[Dataset].pt

[Vessel]: A or V 
(Arteries or Veins)

[Mode]: FS or FSDA or ZS or ZSDA 
(Few-Shot, Few-Shot Data Augmentation, Zero-Shot, Zero-Shot Data Augmentation)

[Dataset]: DRDUHR or DRDULE or DRHRLE or DUHRLE
(DR: DRIVE, DU: DUALMODAL2019, HR: HRF, LE: LESAV, all references provided in the MIDL2023 paper)
DRDUHR -> Trained on DRIVE+DUALMODAL2019+HRF, Inference to do on LESAV


V_FS_DUHRLE.pt -> Veins, Few-Shot, Inference to do on DRIVE. 
```

# 
 
### inferenceEval/
```
[torchModel].eval() -> inference using classic BN: the Batch Normalization layers’ expectation and variance are those computed over the training data.
```

### inferenceTrain/
```
[torchModel].train() -> inference using using AdaBN: the Batch Normalization layers’ expectation and variance are those computed over the testing data.
```
When AdaBN is applied on the generated weights and biases during inference, as mentioned in the paper, we recover the original activations by setting: 
$\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$ and $\beta^{(k)} = \text{E}[x^{(k)}]$.

AdaBN involves no cost, neither in time nor in training resources.

# 

```
In this study we experimented a costless method derived from Batch Normalization expectation and variance properties to improve performances for N-Shot Supervised Learning retinal arteries and veins segmentation.
It substantially outperforms the usual case of performing inference in all experiments.

For arteries segmentation, the Dice gain is 0.230 in mean for all training modes and all datasets.

For veins segmentation, the Dice gain is 0.171 in mean for all training modes and all datasets.
```

<table>
  <caption>Cross-validation performances for multi-source datasets</caption>
  <thead>
    <tr>
      <th style="border: 1px solid white;text-align: center;" colspan="2">Modes</th>
      <th style="border: 1px solid white;text-align: center;" colspan="5">Dice &uarr; (Arteries)</th>
      <th style="border: 1px solid white;text-align: center;" colspan="5">Dice &uarr; (Veins)</th>
    </tr>
    <tr>
      <th style="border: 1px solid white;">Training</th>
      <th style="border: 1px solid white;">Inference</th> 
      <th style="text-align: center;">&#x1D53B;1</th>
      <th style="text-align: center;">&#x1D53B;2</th>
      <th style="text-align: center;">&#x1D53B;3</th>
      <th style="text-align: center;">&#x1D53B;4</th>
      <th style="border: 1px solid white; border-left: 1px dashed white;">Mean</th>
      <th style="text-align: center;">&#x1D53B;1</th>
      <th style="text-align: center;">&#x1D53B;2</th>
      <th style="text-align: center;">&#x1D53B;3</th>
      <th style="text-align: center;">&#x1D53B;4</th>
      <th style="border: 1px solid white; border-left: 1px dashed white;">Mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid white;" rowspan="2">ZS</td>
      <td style="border-right: 1px dashed white;">BN</td>
      <td>0.131</td>
      <td>0.277</td>
      <td>0.621</td>
      <td>0.459</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;">0.372</td>
      <td>0.484</td>
      <td>0.343</td>
      <td>0.620</td>
      <td>0.503</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;">0.488</td>
    </tr>
    <tr>
      <td style="border-right: 1px dashed white;">AdaBN</td>
      <td>0.701</td>
      <td>0.600</td>
      <td>0.621</td>
      <td>0.554</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.619</strong></td>
      <td>0.689</td>
      <td>0.642</td>
      <td>0.632</td>
      <td>0.615</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.645</strong></td>
    </tr>
    <tr>
      <td style="border: 1px solid white;" rowspan="2">ZS+DA</td>
      <td style="border-right: 1px dashed white;">BN</td>
      <td>0.300</td>
      <td>0.530</td>
      <td>0.615</td>
      <td>0.421</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white">0.467</td>
      <td>0.454</td>
      <td>0.557</td>
      <td>0.573</td>
      <td>0.418</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white">0.501</td>
    </tr>
    <tr>
      <td style="border-right: 1px dashed white;">AdaBN</td>
      <td>0.728</td>
      <td>0.648</td>
      <td>0.704</td>
      <td>0.639</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.680</strong></td>
      <td>0.739</td>
      <td>0.679</td>
      <td>0.712</td>
      <td>0.703</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.708</strong></td>
    </tr>
    <tr>
      <td style="border: 1px solid white;" rowspan="2">FS</td>
      <td style="border-right: 1px dashed white;">BN</td>
      <td>0.138</td>
      <td>0.340</td>
      <td>0.660</td>
      <td>0.466</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white">0.401</td>
      <td>0.373</td>
      <td>0.571</td>
      <td>0.688</td>
      <td>0.509</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white">0.535</td>
    </tr>
    <tr>
      <td style="border-right: 1px dashed white;">AdaBN</td>
      <td>0.694</td>
      <td>0.599</td>
      <td>0.638</td>
      <td>0.574</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.626</strong></td>
      <td>0.694</td>
      <td>0.659</td>
      <td>0.649</td>
      <td>0.587</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.647</strong></td>
    </tr>
    <tr>
      <td style="border: 1px solid white;" rowspan="2">FS+DA</td>
      <td style="border-right: 1px dashed white;">BN</td>
      <td>0.161</td>
      <td>0.557</td>
      <td>0.651</td>
      <td>0.521</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white">0.473</td>
      <td>0.350</td>
      <td>0.554</td>
      <td>0.686</td>
      <td>0.565</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white">0.539</td>
    <tr>
        <td style="border-right: 1px dashed white;">AdaBN</td>
        <td>0.748</td>
        <td>0.687</td>
        <td>0.724</td>
        <td>0.663</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.706</strong></td>
      <td>0.770</td>
      <td>0.738</td>
      <td>0.772</td>
      <td>0.714</td>
      <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.749</strong></td>
    </tr>
    <tr>
        <td style="border: 1px solid white;" rowspan="2"><strong>Mean</strong></td>
        <td style="border-right: 1px dashed white;">BN</td>
        <td>0.183</td>
        <td>0.426</td>
        <td>0.637</td>
        <td>0.467</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white">0.428</td>
        <td>0.415</td>
        <td>0.506</td>
        <td>0.642</td>
        <td>0.499</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white">0.516</td>
    </tr>
    <tr>
        <td style="border-right: 1px dashed white;">AdaBN</td>
        <td>0.718</td>
        <td>0.634</td>
        <td>0.672</td>
        <td>0.608</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.658</strong></td>
        <td>0.723</td>
        <td>0.680</td>
        <td>0.691</td>
        <td>0.655</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.687</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid white;" >
        <td style="border: 1px solid white;border-right: 1px dashed white;" colspan="2"><strong>Distance</strong></td>
        <td>0.535</td>
        <td>0.208</td>
        <td>0.035</td>
        <td>0.141</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.230</strong></td>
        <td>0.308</td>
        <td>0.174</td>
        <td>0.049</td>
        <td>0.156</td>
        <td style="border-left: 1px dashed white;border-right: 1px solid white;"><strong>0.171</strong></td>
    </tr>
    </tbody>
</table>


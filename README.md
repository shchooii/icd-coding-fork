# A Comprehensive Study of Class-Sensitive Learning for Long-Tailed ICD Coding

This repository contains the official implementation for the paper, "A Comprehensive Study of Class-Sensitive Learning for Long-Tailed ICD Coding." We provide a systematic framework for evaluating various class-sensitive loss functions to address the severe class imbalance problem in automated ICD coding.

Our work demonstrates that combining advanced loss functions with appropriate threshold-tuning strategies can significantly improve performance on rare codes (**Macro-F1**) while maintaining strong performance on common codes (**Micro-F1**).

## 🚀 Features

  * **PLM-based Architecture**: A robust model (`PLMICD`) built on RoBERTa with a Label Attention mechanism.
  * **Extensive Loss Function Library**: A wide range of implemented loss functions to combat class imbalance, including:
      * `BCEWithLogitsLoss` (Baseline)
      * `FocalLoss`
      * `Hill Loss`
      * `AsymmetricLoss` (ASL)
      * `AsymmetricPolynomialLoss` (APL)
      * `RobustAsymmetricLoss` (RAL)
      * `Class-Balanced Loss` (via ResampleLoss)
      * `MultiGrainedFocalLoss` (MFM)
      * And others like `DRLoss`, `PFM`, etc.
  * **Advanced Balancing Framework**: Implementation of the **COMIC** framework (`PLMICD2`), which uses a multi-expert architecture and a composite loss including:
      * Reflective Label Corrector (`RLC`)
      * Multi-Grained Focal Loss (`MFM`)
      * Head-Tail Balancer (`HTB`)

## ⚙️ Usage

This project is structured to make it easy to switch between different loss functions for experimentation.

### 1\. Applying a Standard Loss Function (`PLMICD`)

To test most of the implemented loss functions, you will use the `PLMICD` model. The loss function can be selected by editing the `__init__` method in `plmicd.py`.

By default, the model uses `binary_cross_entropy_with_logits`. To use a different loss, simply comment out the default and uncomment the desired loss function.

**Example: Switching from BCE to Focal Loss**

```python
# plmicd.py

class PLMICD(nn.Module):
    def __init__(self, num_classes: int, model_path: str, **kwargs):
        super().__init__()
        # ... model architecture ...
        
        # self.loss = torch.nn.functional.binary_cross_entropy_with_logits
        
        self.loss = FocalLoss()
        
        # self.loss = Hill()

        # self.loss = AsymmetricLoss()
        
        # ... other loss functions ...
```

For losses that require class frequency information (e.g., `ResampleLoss`, `MultiGrainedFocalLoss`), ensure you pass the `cls_num_list` or `class_freq` arguments during model initialization.

### 2\. Using the COMIC Framework (`PLMICD2`)

The `COMIC` framework uses a specialized multi-expert architecture and a composite loss function. To use it, you must use the `PLMICD2` model defined in `plmicd2.py`.

This model is pre-configured to combine the three core components of COMIC: `RLC`, `MFM`, and `HTB`. No changes are needed to use the default COMIC setup.

```python
# plmicd2.py

class PLMICD2(nn.Module):
    def __init__(self, num_classes: int, model_path: str, cls_num_list, **kwargs):
        # ... initialization ...
        
        # COMIC loss components are initialized here
        self.rlc = ReflectiveLabelCorrectorLoss(...)
        self.mfm = MultiGrainedFocalLoss()
        self.htb = HeadTailBalancerLoss(PFM=self.mfm)

    def _composite_loss(self, head, tail, bal, labels):
        loss_r = self.rlc(bal, labels)
        loss_m = self.mfm(bal, labels)          
        loss_b = self.htb(head, tail, bal, labels) 
        # The final loss is the weighted sum of the components
        return self.lambda_r * loss_r + self.lambda_m * loss_m + self.lambda_b * loss_b
```

Simply instantiate `PLMICD2` instead of `PLMICD` in your training script to run experiments with the COMIC framework.

Here are the explanations for the two commands.

-----

### 🚀 Model Training

This command starts a **new training session** for the `plm_icd` model on the MIMIC-IV ICD-10 dataset, running on the first GPU.

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0
```

  * **`experiment=mimiciv_icd10/plm_icd`**: Specifies the configuration for the training run (e.g., model architecture, dataset, hyperparameters).
  * **`gpu=0`**: Assigns the training process to the GPU with index 0.

-----

### 🧪 Model Inference (Evaluation)

This command is used for **inference or evaluation**. It loads a pre-trained model and runs it on the test set without any further training.

```bash
python main.py experiment=mimiciv_icd10/plm_icd gpu=0 load_model=/path/to/your/model.ckpt trainer.epochs=0
```

  * **`load_model=/path/to/your/model.ckpt`**: Loads the weights from a previously saved model checkpoint.
  * **`trainer.epochs=0`**: Sets the number of training epochs to zero, which prevents the model from being re-trained and ensures it only performs inference.

## 📄 License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.

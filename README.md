# tracking_objectness

Code for ECCV-ILR 2024 Workshop paper **Leveraging Object Priors for Point Tracking**

# Requirements

Create conda environment for this code base:
```
conda create -n mask_pips python=3.8
conda activate mask_pips
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

```
# Training

Use `export_mp4_dataset.py` file to generate clips from the PointOdyssey training set following [Pips++](https://github.com/aharley/pips2) and then run `python train.py` to start training.

# Testing

To evaluate the performance on the datasets reported in the paper, use testing scripts in the pips2 directory with the saved model. 

For **TAP-VID-DAVIS**: `pip2/test_on_tap.py`  
For **CroHD**: `pips2/test_on_cro.py`  
For **PointOdyssey**: `pips2/test_on_pod.py`   

To replicate the performance from the paper use our [trained weights](https://drive.google.com/drive/folders/1NStVTvo3iMRKcA3vaat7yFjwKHgpGrIH?usp=sharing) for the reference model. For TAP-VID-DAVIS, we load full sequence into memory at once, for others we use `S=36`.

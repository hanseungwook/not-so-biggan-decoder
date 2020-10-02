# not-so-biggan decoder

## not-so-biggan sampler
All models, training, and evaluation code for the not-so-biggan sampler are located here: https://anonymous.4open.science/r/5c1049ba-109f-4a12-9d74-9a4a5130ce97/ .

## ESRGAN Decoders (ESRGAN-W & ESRGAN-P)
All models, training, and evaluation code for ESRGAN decoders (ESRGAN-W and ESRGAN-P) are located here (built upon the BasicSR library): https://anonymous.4open.science/r/bee42675-51cf-4f35-88d8-3b8debd5d796/ .

For ESRGAN-W, refer to `options/train/SRResNet_SRGAN/train_MSRResNet_WT_Pixel_x4.yml` for the pre-training of ResNet-W with L1 loss. For the adversarial model, refer to `options/train/ESRGAN/train_ESRGAN_WT_Pixel_x4.yml` for training and `options/test/ESRGAN/test_ESRGAN_WT_Pixel_x4_woGT.yml` for testing.

For ESRGAN-P, refer to `options/train/SRResNet_SRGAN/train_MSRResNet_L1_x4.yml` for the pre-training of ResNet-P with L1 loss. For the adversarial model, refer to `options/train/ESRGAN/train_ESRGAN_Pixel_x4.yml` for training and `options/test/ESRGAN/test_ESRGAN_Pixel_x4_woGT.yml` for testing.

## UNet Decoders
Modified UNet-based decoder models are all placed within the submodule Pytorch-UNet. You may need to run the following commands in order to pull from the submodule codebase.

If the submodule structure is incompatible for the anonymized GitHub repository, the Pytorch-UNet repository is located here: https://anonymous.4open.science/r/484d57ea-7841-4c8d-bf1c-0dc1c24f2362/


```bash
git submodule init
git submodule update
```

## Environment setup
The conda environment that was used for this repository can be found under `requirements.yml` and can be installed with the following command:

```bash
conda env create -f requirements.yml
```

## Training and evaluation scripts for ESRGAN Decoder
The training commands are listed here: [TrainTest.md](https://anonymous.4open.science/r/bee42675-51cf-4f35-88d8-3b8debd5d796/docs/TrainTest.md)


## Training and evaluation scripts for UNet Decoder

Here are the example scripts to run training for the two levels of decoder and for reconstructing using the 64x64 TL patch either from the real dataset or our not-so-biggan sampler. All training utilizes only 1 x V100 16GB GPU, if available.

**Training first level UNet-based decoder (64 => 128)**
```bash
python src/train_unet_128.py \
--train_dir {Path to ImageNet train dataset} --valid_dir {Path to ImageNet valid dataset} \
--batch_size 128 --image_size 256 --mask_dim 64 --lr 1e-4 \
--num_epochs 100 --output_dir {Path to output directory} \
--project_name unet_full_imagenet_128 \
--save_every 2000 --valid_every 2000 --log_every 50 \
```

**Training second level UNet-based decoder (128 => 256)**
```bash
python src/train_unet_256_real.py \
--train_dir {Path to ImageNet train dataset} --valid_dir {Path to ImageNet valid dataset} \
--batch_size 64 --image_size 256 --mask_dim 128 --lr 1e-4 \
--num_epochs 100 --output_dir {Path to output directory} \
--project_name unet_full_imagenet_256_real \
--save_every 2000 --valid_every 2000 --log_every 50 \
```

**Reconstructions using real dataset 64x64 TL patch + first level decoder + second level decoder**
```bash
python src/eval_unet_128_256.py \
--train_dir {Path to ImageNet train dataset} --valid_dir {Path to ImageNet valid dataset} \
--batch_size 100 --image_size 256 \
--output_dir {Path to output directory} \
--project_name unet_full_imagenet_128_256_eval \
--model_128_weights {Path to first level decoder weights} \
--model_256_weights {Path to second level decoder weights} \
```

**Reconstructions using not-so-biggan sampler 64x64 TL patch + first level decoder + second level decoder**
```bash
python src/eval_biggan_unet_128_256.py \
--batch_size 100 --image_size 256 \
--output_dir {Path to output directory} \
--project_name biggan_unet_imagenet_128_256_eval \
--model_128_weights {Path to first level decoder weights} \
--model_256_weights {Path to second level decoder weights} \
--sample_file {Path to 64x64 TL patch samples from sampler} \
```

**Training first level UNet-based-pixel decoder (64 => 128)**
```bash
python src/train_UNET_256_pixel.py \
--train_dir {Path to ImageNet train dataset} --valid_dir {Path to ImageNet valid dataset} \
--batch_size=64 --image_size=256 \
--low_resolution=128 --workers=4 \
--lr=1e-3 --num_epochs 100 \
--output_dir {Path to output directory} --save_every=500 \
--valid_every=1000 \
```

**Training first level UNet-based-pixel decoder (64 => 128)**
```bash
python src/train_UNET_pixel.py \
--train_dir {Path to ImageNet train dataset} --valid_dir {Path to ImageNet valid dataset} \
--batch_size=64, --image_size=128 \
--low_resolution=64, --workers=4 \
--lr=1e-3, --num_epochs=100 \
--output_dir {Path to output directory} --save_every=500 \
--valid_every=1000 \
```

**Training second level UNet-based-pixel decoder (128 => 256)**
```bash
python src/train_UNET_256_pixel.py \
--train_dir {Path to ImageNet train dataset} --valid_dir {Path to ImageNet valid dataset} \
--batch_size=64 --image_size=256 \
--low_resolution=128 --workers=4 \
--lr=1e-3 --num_epochs 100 \
--output_dir {Path to output directory} --save_every=500 \
--valid_every=1000 \
```

## Slurm scripts
The actual scripts used to train the models (with the Slurm workload manager) are all in `scripts/`. 

## FID and IS evaluation
All code for FID and IS evaluation has been adapted from the original Tensorflow implementation and are located here: https://anonymous.4open.science/r/dac95e3c-ecab-4daf-bea4-758db495843a/. The only change to this implmentation was that I modified it to use HDF5 datasets of images for the calculations as it provides a much faster method for saving and loading images.


## Requirements

Under `requirements.yml` for conda environment setup

## PyTorch code for the paper "Curriculum Graph Co-Teaching for Multi-target Domain Adaptation" (CVPR2021)
This repo presents PyTorch implementation of Multi-targe Graph Domain Adaptation framework from ["Curriculum Graph Co-Teaching for Multi-target Domain Adaptation" CVPR 2021](https://arxiv.org/abs/).
The framework is pivoted around two key concepts: *graph feature aggregation* and *curriculum learning* (see pipeline below or [project web-page](https://roysubhankar.github.io/graph-coteaching-adaptation/)).
<img src="data/pipeline.png" width="1000">
## Results
<img src="data/results.png" width="600">

## Environment
```
Python >= 3.6
PyTorch >= 1.8.1
```
To install dependencies run (line 1 for pip or line 2 for conda env): 
```
pip install -r requirements.txt
conda install --file requirements.txt
```
*Disclaimer.*  This code has been tested with cuda toolkit 10.2. Please install PyTorch as supported by your machine.


## Datasets
Four datasets are supported:
* Office-31 ([Kate Saenko et al., 2010](https://link.springer.com/content/pdf/10.1007/978-3-642-15561-1_16.pdf))
* Office-Home ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))

To run this code, one must check if the txt file names in data/<dataset_name> are matching with the downloaded domain folders. For e.g.,  to run OfficeHome, the domain sub-folders should be art/, clipart/, product/ and real/ corresponding to art.txt, clipart.txt, product.txt and real.txt that can be found in the data/office-home/.

## Methods
* CDAN
* CDAN+E

## Commands
## Office-31
```
python src/main.py \
        --method 'CDAN' \
        --encoder 'ResNet50' \
 	--dataset 'office31' \
 	--data_root [your office31 folder] \
 	--source 'dslr' \
 	--target 'webcam' 'amazon' \
 	--source_iters 200 \
 	--adapt_iters 3000 \
 	--finetune_iters 15000 \
 	--lambda_node 0.3 \
 	--output_dir 'office31-dcgct/dslr_rest/CDAN'
```
```
python src/main_cgct.py \
        --method 'CDAN' \
        --encoder 'ResNet50' \
 	--dataset 'office31' \
 	--data_root [your office31 folder] \
 	--source 'dslr' \
 	--target 'webcam' 'amazon' \
 	--source_iters 100 \
 	--adapt_iters 3000 \
 	--finetune_iters 15000 \
 	--lambda_node 0.1 \
 	--output_dir 'office31-cgct/dslr_rest/CDAN'
```

## Office-Home
```
python src/main.py \
	--method 'CDAN' \
	--encoder 'ResNet50' \
	--dataset 'office-home' \
	--data_root [your OfficeHome folder] \
	--source 'art' \
	--target 'clipart' 'product' 'real' \
	--source_iters 500 \
	--adapt_iters 10000 \
	--finetune_iters 15000 \
	--lambda_node 0.3 \
	--output_dir 'officeHome-dcgct/art_rest/CDAN' 
```

## PACS
```
python src/main.py \
	--method 'CDAN' \
	--encoder 'ResNet50' \
	--dataset 'pacs' \
	--data_root [your PACS folder] \
	--source 'photo' \
	--target 'cartoon' 'art_painting' 'sketch' \
	--source_iters 200 \
	--adapt_iters 3000 \
	--finetune_iters 15000  \
	--lambda_node 0.1 \
	--output_dir 'pacs-dcgct/photo_rest/CDAN'  
```

## DomainNet
```
python src/main.py \
	--method 'CDAN' \
	--encoder 'ResNet101' \
	--dataset 'domain-net' \
	--data_root [your DomainNet folder] \
	--source 'sketch' \
	--target 'clipart' 'infograph' 'painting' 'real' 'quickdraw' \
	--source_iters 5000 \
	--adapt_iters 50000 \
	--finetune_iters 15000  \
	--lambda_node 0.1 \
	--output_dir 'domainNet-dcgct/sketch_rest/CDAN'
```
## Citation
If you find our paper and code useful for your research, please consider citing our paper.
```
@inproceedings{roy2021curriculum,
  title={Curriculum Graph Co-Teaching for Multi-target Domain Adaptation},
  author={Roy, Subhankar and Krivosheev, Evgeny and Zhong, Zhun and Sebe, Nicu and Ricci, Elisa},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

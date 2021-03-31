# D-CGCT

# Results
# Experiments on Office-31

[comment]: <> (| Method | A->W, D | W->A, D | D->A, W | Avg|)

[comment]: <> (|:-------|:---:|:---:|:---:|:---:|)

[comment]: <> (|ReverseGrad **w/o** labels as in [3]|78.2|69.8|72.2|73.4|)

[comment]: <> (|ReverseGrad **w/o** labels|80.68|81.31|78.76|80.25|)

[comment]: <> (|ReverseGrad **w/** labels|73.68|82.47|81.28|79.14|)

[comment]: <> (|BTDA [3]|90.1|73.4|77.0|80.2|)

# Environment
```
Python >= 3.6
PyTorch >= 1.8.1
```

# Datasets
Four datasets are supported:
* Office-31
* Office-Home
* PACS
* DomainNet

# Commands
##Office-31
```
python src/main.py \
    --method 'CDAN' \
    --encoder 'ResNet50' \
 	--dataset 'office31' \
 	--data_root '/data2/ekrivosheev/office31' \
 	--image_list_root 'data/office/' \
 	--source 'dslr' \
 	--target 'webcam' 'amazon' \
 	--source_iters 200 \
 	--adapt_iters 3000 \
 	--finetune_iters 15000 \
 	--lambda_node 0.3 \
 	--output_dir 'office31/dslr_rest/CDAN'
```

## Office-Home
```
python src/main.py \
	--method 'CDAN' \
	--encoder 'ResNet50' \
	--dataset 'office-home' \
	--data_root '/data2/ekrivosheev/OfficeHome' \
	--image_list_root 'data/office-home/' \
	--source 'art' \
	--target 'clipart' 'product' 'real' \
	--source_iters 500 \
	--adapt_iters 10000 \
	--finetune_iters 15000 \
	--lambda_node 0.3 \
	--output_dir 'office-home/art_rest/CDAN' 
```

## Pacs
```
python src/main.py \
	--method 'CDAN' \
	--encoder 'ResNet50' \
	--dataset 'pacs' \
	--data_root '/data2/ekrivosheev/pacs' \
	--image_list_root 'data/pacs/' \
	--source 'photo' \
	--target 'cartoon' 'art_painting' 'sketch' \
	--source_iters 200 \
	--adapt_iters 3000 \
	--finetune_iters 15000  \
	--lambda_node 0.1 \
	--output_dir 'pacs/photo_rest/CDAN'  
```

## DomainNet
```
python src/main.py \
	--method 'CDAN' \
	--encoder 'ResNet101' \
	--dataset 'domain-net' \
	--data_root 'data/datasets/domain-net' \
	--image_list_root 'data/domain-net/' \
	--source 'sketch' \
	--target 'clipart' 'infograph' 'painting' 'real' 'quickdraw' \
	--source_iters 5000 \
	--adapt_iters 50000 \
	--finetune_iters 15000  \
	--lambda_node 0.1 \
	--output_dir 'domain-net/sketch_rest/CDAN'
```
------------

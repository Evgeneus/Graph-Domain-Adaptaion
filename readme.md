# Domain curriculum + CoTeachingGNN 

##Office31
```
 CUDA_VISIBLE_DEVICES=6  python main.py --method 'CDAN' --encoder 'ResNet50' --dataset 'office31' --data_root '/data/datasets/office31' --image_list_root '../data/office/' --source 'dslr' --target 'webcam' 'amazon' --source_iters 100 --test_interval 100 --adapt_iters 3000 --finetune_iters 15000  --lambda_edge 1.0 --lambda_node 0.3 --test_batch 512 --output_dir 'Test/D-CGCT-S54_TargBatch32/Office31-tbatch512-02.3.15k/seed0/dslr/CDAN-Ledge1.0-Lgnn0.3' --seed 0 
```

------------


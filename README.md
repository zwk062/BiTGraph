# BiaTCGNet
The code for Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values. 


Biased Temporal Convolution Graph Network that jointly captures the temporal dependencies and spatial structure. In particular, we inject bias into the two carefully developed modules—the MultiScale Instance PartialTCN and Biased GCN—to account for missing patterns.




## Overview
|![Figure1](images/Framework.png)|
|:--:| 
| *Figure 1-a Overall structure of BiaTCGNet, Figure 1-b the structure of Biased GCN module.* |



## Main Results
![image](https://user-images.githubusercontent.com/44238026/171345192-e7440898-4019-4051-86e0-681d1a28d630.png)


## Get Started

1. Install Python>=3.8, PyTorch 1.9.0.
2. Download data.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the multivariate and univariate experiment results by running the following shell code separately:

```bash
bash ./scripts/run_M.sh
bash ./scripts/run_S.sh
```


## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{zhou2022fedformer,
  title={{FEDformer}: Frequency enhanced decomposed transformer for long-term series forecasting},
  author={Zhou, Tian and Ma, Ziqing and Wen, Qingsong and Wang, Xue and Sun, Liang and Jin, Rong},
  booktitle={Proc. 39th International Conference on Machine Learning (ICML 2022)},
  location = {Baltimore, Maryland},
  pages={},
  year={2022}
}
```

## Further Reading
Survey on Transformers in Time Series:

Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, and Liang Sun. "Transformers in time series: A survey." arXiv preprint arXiv:2202.07125 (2022). [paper](https://arxiv.org/abs/2202.07125)


## Contact

If you have any question or want to use the code, please contact tian.zt@alibaba-inc.com or maziqing.mzq@alibaba-inc.com .

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

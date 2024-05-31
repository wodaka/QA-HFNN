# Quantum Assisted Hierarchical Fuzzy Neural Network for Image Classification


The model contains four components:
1. quantum fuzzy logic representation
2. deep neural network representation
3. fusion layer
4. classifier layer

## Requirements

- Python 3.8
- torchquantum == 0.1.7
- ddu_dirty_mnist == 1.1.1
- scikit-learn == 1.3.2
- qiskit-ibmq-provider == 0.19.2

Dependencies can be installed using the following command:
```bash
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
pip install ddu_dirty_mnist
pip install opencv-python
pip install scikit-learn
pip install qiskit_ibm_runtime
pip install qiskit-ibmq-provider
pip install qiskit-terra
pip install qiskit-aer

```

## Usage

Commands for training and testing the model:

```bash
python run.py
```

More parameter information please refer to `run.py`.


The detailed descriptions about the arguments are as following:

| Parameter name | Description of parameter |
| --- | --- |
| model | The model of experiment. This can be set to `QA-HFNN`|
| batch_size           | The batch_size of the dataset                                             |
| circuit_layer      | The circuit layers of the quantum neural network    |
| learning_rate      | The learning rate of the model                  |
| epoch       | The epoch of the training




## <span id="resultslink">Results</span>

We have updated the experiment results.

Besides, the experiment parameters is listed in paper. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better results or make better prediction.


## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following paper:

```
@article{wu2024hierarchical,
  title={A Hierarchical Fused Quantum Fuzzy Neural Network for Image Classification},
  author={Wu, Sheng-Yao and Li, Run-Ze and Song, Yan-Qi and Qin, Su-Juan and Wen, Qiao-Yan and Gao, Fei},
  journal={arXiv preprint arXiv:2403.09318},
  year={2024}
}
```

## Contact
If you have any questions, feel free to contact Shengyao Wu through Email (wushengyaobn@163.com) or Github issues. Pull requests are highly welcomed!


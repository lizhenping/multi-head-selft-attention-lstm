# multi-head-selft-attention-lstm
在sts数据集上用多头注意力机制上进行测试。 pytorch torchtext 代码简练，非常适合新手了解多头注意力机制的运作。不想transformer牵扯很多层 multi-head attention + one layer lstm
#参考了各位大神的demo
苏大神keras版本：

https://kexue.fm/archives/4765


github多头注意力模板，但是没有代码彻底实现，我这里增加了测试的例子
https://github.com/sakuranew/attention-pytorch

代码里面没有给出评估函数，但是loss下降都正常。用debug，以了解分析多头注意力的小demo，多头注意力的scale没有完全按照维度根号整除，简化了
